import os
import shutil
import random
import types
import torch
import math
import numpy as np
from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.text import CharErrorRate
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Import core modules
from comer.datamodule.dataset import CROHMEDataset
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from comer.datamodule.datamodule import collate_fn, build_dataset_unzipped

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
BASE_CKPT_PATH = "lightning_logs/version_0/checkpoints/epoch=183-step=69183-val_ExpRate=0.6182.ckpt"

# Custom Dataset Paths
CUSTOM_CAPTION_PATH = "data/custom/caption.txt"
CUSTOM_IMG_DIR = "data/custom/maml"
MAML_SPLITS_DIR = "data/custom_maml_splits"
MAML_CKPT_DIR = "maml_checkpoints"

TRAIN_EPOCHS = 20  
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Micro-adaptation learning rate to prevent catastrophic forgetting
ADAPTATION_LR = 1e-5

seed_everything(7)

# ==========================================
# LOCAL DATAMODULE FOR AUTHOR DYNAMICS
# ==========================================
class AuthorAdaptationDataModule(pl.LightningDataModule):
    """Dynamic datamodule supporting 30% Train, 10% Val, and 60% Test routing."""
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, batch_size: int = 4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 30% Train split
        self.train_dataset = CROHMEDataset(
            build_dataset_unzipped(self.train_dir, self.batch_size),
            is_train=True, scale_aug=False, freq_remove_num=3
        )
        # 10% Val split
        self.val_dataset = CROHMEDataset(
            build_dataset_unzipped(self.val_dir, self.batch_size),
            is_train=False, scale_aug=False, freq_remove_num=3
        )
        # 60% Test split
        self.test_dataset = CROHMEDataset(
            build_dataset_unzipped(self.test_dir, self.batch_size),
            is_train=False, scale_aug=False, freq_remove_num=3
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=2, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn)


# ==========================================
# PHASE 1: DATA SPLITTING (30/10/60 TRAIN/VAL/TEST)
# ==========================================
def prepare_custom_dataset():
    print("\n--- PHASE 1: Preparing Custom Dataset (30/10/60 Split) ---")
    
    if not os.path.exists(CUSTOM_CAPTION_PATH):
        raise FileNotFoundError(f"Missing caption file at: {CUSTOM_CAPTION_PATH}")
    if not os.path.exists(CUSTOM_IMG_DIR):
        raise FileNotFoundError(f"Missing image directory at: {CUSTOM_IMG_DIR}")
        
    with open(CUSTOM_CAPTION_PATH, "r", encoding="utf-8") as f:
        raw_captions = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Loaded {len(raw_captions)} total equation expressions from {CUSTOM_CAPTION_PATH}.")
    
    # 1. Check each token against vocab.word2idx and filter out invalid rows
    paired_data = []
    removed_count = 0
    
    for idx, caption in enumerate(raw_captions):
        tokens = caption.split()
        
        # Identify any tokens not present in CoMER's vocabulary dictionary
        oov_tokens = [t for t in tokens if t not in vocab.word2idx]
        
        if oov_tokens:
            removed_count += 1
            print(f"[Info] Removing item index {idx} due to unknown symbols {set(oov_tokens)}: '{caption}'")
            continue
            
        img_id = f"equations_1_{idx}"
        paired_data.append((img_id, caption))
        
    print(f"Filtered out {removed_count} samples containing unknown symbols. Remaining valid samples: {len(paired_data)}.")
    
    # 2. Shuffle the paired dataset while preserving caption-to-image matching
    random.shuffle(paired_data)
    
    # 3. Perform 30% Train / 10% Val / 60% Test split
    n_total = len(paired_data)
    train_idx = int(n_total * 0.30)
    val_idx = train_idx + int(n_total * 0.10)
    
    train_items = paired_data[:train_idx]
    val_items = paired_data[train_idx:val_idx]
    test_items = paired_data[val_idx:]
    
    print(f"Split distribution -> Train: {len(train_items)} ({len(train_items)/n_total*100:.1f}%) | "
          f"Val: {len(val_items)} ({len(val_items)/n_total*100:.1f}%) | "
          f"Test: {len(test_items)} ({len(test_items)/n_total*100:.1f}%)")
    
    author = "custom_author"
    
    if os.path.exists(MAML_SPLITS_DIR):
        shutil.rmtree(MAML_SPLITS_DIR)

    train_dir = os.path.join(MAML_SPLITS_DIR, author, "train")
    val_dir = os.path.join(MAML_SPLITS_DIR, author, "val")
    test_dir = os.path.join(MAML_SPLITS_DIR, author, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def save_split(split_items, target_dir):
        with open(os.path.join(target_dir, "caption.txt"), "w", encoding="utf-8") as f:
            for img_id, caption in split_items:
                # Write standard CROHME entry: "<img_id> <caption_tokens>"
                f.write(f"{img_id} {caption}\n")
                
                # Copy corresponding image and convert to 1-channel grayscale
                img_name = f"{img_id}.bmp"
                src_img = os.path.join(CUSTOM_IMG_DIR, img_name)
                dst_img = os.path.join(target_dir, img_name)
                
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    with Image.open(src_img) as img:
                        img.convert("L").save(dst_img)
                elif not os.path.exists(src_img):
                    print(f"[Warning] Image file not found: {src_img}")
                    
    save_split(train_items, train_dir)
    save_split(val_items, val_dir)
    save_split(test_items, test_dir)
        
    return [author]

# ==========================================
# PHASE 2: EVALUATION UTILITY
# ==========================================
def evaluate_model(model, data_dir):
    """Evaluates the model on targeted paths, calculating Accuracy (ExpRate) and CER."""
    dataset = CROHMEDataset(build_dataset_unzipped(data_dir, BATCH_SIZE), False, False, 3)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=2)
    
    model.eval()
    model.to(DEVICE)
    cer_metric = CharErrorRate().to(DEVICE)
    
    correct_sentences = 0
    total_sentences = 0
    total_cer = 0.0
    batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            hyps = model.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)
            
            pred_labels = [vocab.indices2label(h.seq) for h in hyps]
            target_labels = [vocab.indices2label(i) for i in batch.indices]
            
            for pred, target in zip(pred_labels, target_labels):
                if pred == target:
                    correct_sentences += 1
                total_sentences += 1
                
            batch_cer = cer_metric(pred_labels, target_labels).item()
            total_cer += batch_cer
            batches += 1
            
    accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0
    avg_cer = total_cer / batches if batches > 0 else 0
    
    return accuracy, avg_cer

def load_base_model_safely(ckpt_path):
    """Loads initial weights and manages checkpoint layer remappings safely."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("comer_model.PAM.", "comer_model.FAB.")
        new_state_dict[new_key] = v
        
    checkpoint["state_dict"] = new_state_dict
    tmp_ckp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, tmp_ckp_path)
    
    model = LitCoMER.load_from_checkpoint(tmp_ckp_path, strict=True)
    if os.path.exists(tmp_ckp_path):
        os.remove(tmp_ckp_path)
        
    return model

# ==========================================
# PHASE 3: EXECUTION ORCHESTRATOR
# ==========================================
def main():
    authors = prepare_custom_dataset()
    os.makedirs(MAML_CKPT_DIR, exist_ok=True)
    results = {}
    
    print("\n--- PHASE 2: Unfreezing Full Weights & Running MAML Adaptation Loop ---")
    for author in authors:
        train_dir = os.path.join(MAML_SPLITS_DIR, author, "train")
        val_dir = os.path.join(MAML_SPLITS_DIR, author, "val")
        test_dir = os.path.join(MAML_SPLITS_DIR, author, "test")
        
        if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
            continue
            
        print(f"\n[{author}] Adapting model weights on 30% train split...")
        
        # 1. Baseline Run (Evaluation prior to optimization steps on 60% test split)
        base_model = load_base_model_safely(BASE_CKPT_PATH)
        base_acc, base_cer = evaluate_model(base_model, test_dir)
        
        # 2. MAML Fast Adaptation (Unfrozen parameters optimized via task support set)
        maml_model = load_base_model_safely(BASE_CKPT_PATH)
        for param in maml_model.parameters():
            param.requires_grad = True
            
        # >> HOT-PATCH OPTIMIZER CONFIGURATION HERE <<
        def adaptation_optimizers(self):
            # Using an ultra-low learning rate to fine-tune without destroying base features
            return torch.optim.AdamW(self.parameters(), lr=ADAPTATION_LR, weight_decay=1e-4)
        
        # Bind the custom method safely to the maml_model instance
        maml_model.configure_optimizers = types.MethodType(adaptation_optimizers, maml_model)
            
        dm = AuthorAdaptationDataModule(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            batch_size=BATCH_SIZE
        )
        
        trainer = Trainer(
            logger=False,
            checkpoint_callback=False,
            max_epochs=TRAIN_EPOCHS,
            gpus=1 if torch.cuda.is_available() else 0,
            weights_summary=None,
        )
        trainer.fit(maml_model, datamodule=dm)
        
        # 3. Save optimized author state
        save_path = os.path.join(MAML_CKPT_DIR, f"maml_model_{author}.pth")
        torch.save(maml_model.state_dict(), save_path)
        
        # 4. Post-Adaptation Query Evaluation (evaluated on 60% test split)
        maml_acc, maml_cer = evaluate_model(maml_model, test_dir)
        
        results[author] = {
            "base_acc": base_acc, "base_cer": base_cer,
            "maml_acc": maml_acc, "maml_cer": maml_cer
        }
        print(f"[{author}] Complete -> Base Acc: {base_acc:.4f} | MAML Acc: {maml_acc:.4f}")

    # ==========================================
    # PHASE 4: RECAPITULATIVE PERFORMANCE REPORT
    # ==========================================
    print("\n\n" + "="*90)
    print("MAML ADAPTATION EXPERIMENT REPORT (CUSTOM AUTHOR)")
    print("="*90)
    print(f"{'Author ID':<15} | {'Base Accuracy':<15} | {'Base CER':<15} | {'MAML Accuracy':<15} | {'MAML CER':<15}")
    print("-" * 90)
    
    total_base_acc, total_base_cer = 0.0, 0.0
    total_maml_acc, total_maml_cer = 0.0, 0.0
    
    for author, metrics in results.items():
        print(f"{author:<15} | {metrics['base_acc']:<15.4f} | {metrics['base_cer']:<15.4f} | {metrics['maml_acc']:<15.4f} | {metrics['maml_cer']:<15.4f}")
        total_base_acc += metrics['base_acc']
        total_base_cer += metrics['base_cer']
        total_maml_acc += metrics['maml_acc']
        total_maml_cer += metrics['maml_cer']
        
    num_authors = len(results)
    if num_authors > 0:
        print("-" * 90)
        print(f"{'AVERAGE':<15} | {total_base_acc/num_authors:<15.4f} | {total_base_cer/num_authors:<15.4f} | {total_maml_acc/num_authors:<15.4f} | {total_maml_cer/num_authors:<15.4f}")
        print("="*90)

if __name__ == "__main__":
    main()