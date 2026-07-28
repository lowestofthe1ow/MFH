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

# Import core modules
from comer.datamodule.dataset import CROHMEDataset
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from comer.datamodule.datamodule import collate_fn, build_dataset_unzipped

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
BASE_CKPT_PATH = "lightning_logs/version_0/checkpoints/epoch=183-step=69183-val_ExpRate=0.6182.ckpt"
SOURCE_DATA_DIR = "data/2016"
MAML_SPLITS_DIR = "data/maml_author_splits"
MAML_CKPT_DIR = "maml_checkpoints"

# Set to an integer (e.g., 3) to test a few authors before committing to all filtered authors
MAX_AUTHORS_TO_PROCESS = None  

TRAIN_EPOCHS = 20  
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SAMPLES_REQUIRED = 20  # Filter threshold

# Micro-adaptation learning rate to prevent catastrophic forgetting
ADAPTATION_LR = 1e-5

seed_everything(7)

# ==========================================
# LOCAL DATAMODULE FOR AUTHOR DYNAMICS
# ==========================================
class AuthorAdaptationDataModule(pl.LightningDataModule):
    """Dynamic datamodule that routes specific author paths without hardcoding data/2014."""
    def __init__(self, train_dir: str, test_dir: str, batch_size: int = 4):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 80% split for adaptation training
        self.train_dataset = CROHMEDataset(
            build_dataset_unzipped(self.train_dir, self.batch_size),
            is_train=True, scale_aug=False, freq_remove_num=3
        )
        # 20% split for validation/evaluation checks
        self.val_dataset = CROHMEDataset(
            build_dataset_unzipped(self.test_dir, self.batch_size),
            is_train=False, scale_aug=False, freq_remove_num=3
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=2, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn)


# ==========================================
# PHASE 1: DATA SPLITTING WITH FILTERING (>= 20 SAMPLES)
# ==========================================
def prepare_maml_datasets():
    print("\n--- PHASE 1: Splitting Dataset by Author (80/20 MAML Setup) ---")
    caption_path = os.path.join(SOURCE_DATA_DIR, "caption.txt")
    img_dir = os.path.join(SOURCE_DATA_DIR, "img")
    
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Missing {caption_path}")
        
    with open(caption_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # Group expressions by author ID (first 6 characters)
    all_author_data = defaultdict(list)
    for line in lines:
        img_name = line.split()[0]
        author_id = img_name[:6]
        all_author_data[author_id].append(line)
        
    print(f"Found {len(all_author_data)} unique authors total in the 2016 dataset.")
    
    # Filter for authors with 20 or more samples
    author_data = {
        auth: lines for auth, lines in all_author_data.items() 
        if len(lines) >= MIN_SAMPLES_REQUIRED
    }
    print(f"Filtered down to {len(author_data)} authors with >= {MIN_SAMPLES_REQUIRED} total samples.")
    
    authors_to_process = list(author_data.keys())
    if MAX_AUTHORS_TO_PROCESS:
        authors_to_process = authors_to_process[:MAX_AUTHORS_TO_PROCESS]
        print(f"Limiting execution to the first {MAX_AUTHORS_TO_PROCESS} filtered authors for validation testing.")

    # Wipe the directory cleanly before reconstructing splits to clear out stale author folders
    if os.path.exists(MAML_SPLITS_DIR):
        shutil.rmtree(MAML_SPLITS_DIR)

    for author in tqdm(authors_to_process, desc="Generating Filtered Splits"):
        lines = author_data[author]
        random.shuffle(lines)
        
        split_idx = int(len(lines) * 0.8)
        train_lines = lines[:split_idx]
        test_lines = lines[split_idx:]
        
        train_dir = os.path.join(MAML_SPLITS_DIR, author, "train")
        test_dir = os.path.join(MAML_SPLITS_DIR, author, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        def save_split(split_lines, target_dir):
            with open(os.path.join(target_dir, "caption.txt"), "w") as f:
                for line in split_lines:
                    f.write(line + "\n")
                    img_name = line.split()[0] + ".bmp"
                    src_img = os.path.join(img_dir, img_name)
                    dst_img = os.path.join(target_dir, img_name)
                    if os.path.exists(src_img) and not os.path.exists(dst_img):
                        shutil.copy2(src_img, dst_img)
                        
        save_split(train_lines, train_dir)
        save_split(test_lines, test_dir)
        
    return authors_to_process

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
    authors = prepare_maml_datasets()
    os.makedirs(MAML_CKPT_DIR, exist_ok=True)
    results = {}
    
    print("\n--- PHASE 2: Unfreezing Full Weights & Running MAML Adaptation Loop ---")
    for author in authors:
        train_dir = os.path.join(MAML_SPLITS_DIR, author, "train")
        test_dir = os.path.join(MAML_SPLITS_DIR, author, "test")
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            continue
            
        print(f"\n[{author}] Adapting model weights...")
        
        # 1. Baseline Run (Evaluation prior to optimization steps)
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
            
        dm = AuthorAdaptationDataModule(train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE)
        
        trainer = Trainer(
            logger=False,
            checkpoint_callback=False,
            max_epochs=TRAIN_EPOCHS,
            gpus=1 if torch.cuda.is_available() else 0,
            weights_summary=None,  # Use this for older PyTorch Lightning versions
        )
        trainer.fit(maml_model, datamodule=dm)
        
        # 3. Save optimized author state
        save_path = os.path.join(MAML_CKPT_DIR, f"maml_model_{author}.pth")
        torch.save(maml_model.state_dict(), save_path)
        
        # 4. Post-Adaptation Query Evaluation
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
    print("MAML ADAPTATION EXPERIMENT REPORT (AUTHORS WITH >= 20 SAMPLES)")
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