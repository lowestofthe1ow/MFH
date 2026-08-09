import os
import types
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.text import CharErrorRate
from torch.utils.data import DataLoader

from comer.datamodule.dataset import CROHMEDataset
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from comer.datamodule.datamodule import collate_fn, build_dataset_unzipped

BASE_CKPT_PATH = (
    "lightning_logs/version_0/checkpoints/epoch=183-step=69183-val_ExpRate=0.6182.ckpt"
)
FINETUNE_CKPT_DIR = "finetune_checkpoints"
TRAIN_EPOCHS = 20
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Not sure if this is an optimal value or if we should do ablations
# We probably should but ablations in the data size direction might be a better
# use of our time? Look into it
ADAPTATION_LR = 1e-5

seed_everything(765)  # ナムコプロ最強！！！！


class AuthorAdaptationDataModule(pl.LightningDataModule):
    """Data module for the author-specific fine-tuning stage"""

    def __init__(self, train_dir, test_dir, val_dir=None, batch_size=BATCH_SIZE):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir or test_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def _dataset(self, data_dir, is_train):
        return CROHMEDataset(
            build_dataset_unzipped(data_dir, self.batch_size),
            is_train=is_train,
            scale_aug=False,
            freq_remove_num=3,
        )

    def setup(self, stage=None):
        self.train_dataset = self._dataset(self.train_dir, True)
        self.val_dataset = self._dataset(self.val_dir, False)
        self.test_dataset = self._dataset(self.test_dir, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, num_workers=2, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn
        )


def evaluate_model(model, data_dir, return_samples=False):
    """Provides metrics like ExpRate and average CER"""

    dataset = CROHMEDataset(
        build_dataset_unzipped(data_dir, BATCH_SIZE), False, False, 3
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=2)

    model.eval()
    model.to(DEVICE)
    cer_metric = CharErrorRate().to(DEVICE)

    correct_sentences = total_sentences = batches = 0
    total_cer = 0.0
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            hyps = model.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)

            pred_labels = [vocab.indices2label(h.seq) for h in hyps]
            target_labels = [vocab.indices2label(i) for i in batch.indices]

            for pred, target in zip(pred_labels, target_labels):
                correct_sentences += int(pred == target)
                total_sentences += 1
                if return_samples:
                    sample_cer = cer_metric([pred], [target]).item()
                    samples.append(
                        {"expected": target, "predicted": pred, "cer": sample_cer}
                    )

            total_cer += cer_metric(pred_labels, target_labels).item()
            batches += 1

    exprate = correct_sentences / total_sentences if total_sentences else 0
    avg_cer = total_cer / batches if batches else 0

    if return_samples:
        return exprate, avg_cer, samples
    return exprate, avg_cer


def load_base_model(ckpt_path):
    """Loads the .ckpt file provided by the original MFH repo"""

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint["state_dict"] = {
        k.replace("comer_model.PAM.", "comer_model.FAB."): v
        for k, v in checkpoint["state_dict"].items()
    }

    tmp_ckp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, tmp_ckp_path)
    try:
        return LitCoMER.load_from_checkpoint(tmp_ckp_path, strict=True)
    finally:
        if os.path.exists(tmp_ckp_path):
            os.remove(tmp_ckp_path)


def _bind_adaptation_optimizer(model):
    """Updates the optimizer in the model with the new one"""

    # TODO: Are we also ablating the optimizer? Or just stick with AdamW? If
    # sticking with AdamW, we might need some kind of justification for it
    def adaptation_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=ADAPTATION_LR, weight_decay=1e-4)

    model.configure_optimizers = types.MethodType(adaptation_optimizers, model)
    return model


def build_trainer():
    return Trainer(
        logger=False,
        checkpoint_callback=False,
        max_epochs=TRAIN_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        weights_summary=None,
    )


def adapt_author(
    author,
    train_dir,
    test_dir,
    val_dir=None,
    ckpt_path=BASE_CKPT_PATH,
    ckpt_dir=FINETUNE_CKPT_DIR,
):
    base_model = load_base_model(ckpt_path)
    base_exprate, base_cer = evaluate_model(base_model, test_dir)

    finetune_model = load_base_model(ckpt_path)

    # TODO: Should we be more selective regarding which parameters to freeze/train?
    for param in finetune_model.parameters():
        param.requires_grad = True

    _bind_adaptation_optimizer(finetune_model)

    dm = AuthorAdaptationDataModule(train_dir, test_dir, val_dir, batch_size=BATCH_SIZE)
    build_trainer().fit(finetune_model, datamodule=dm)

    torch.save(
        finetune_model.state_dict(),
        os.path.join(ckpt_dir, f"finetune_model_{author}.pth"),
    )
    finetune_exprate, finetune_cer = evaluate_model(finetune_model, test_dir)

    return {
        "base_exprate": base_exprate,
        "base_cer": base_cer,
        "finetune_exprate": finetune_exprate,
        "finetune_cer": finetune_cer,
    }


def run_experiment(
    authors,
    splits_dir,
    has_val=False,
    ckpt_path=BASE_CKPT_PATH,
    ckpt_dir=FINETUNE_CKPT_DIR,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    results = {}

    for author in authors:
        train_dir = os.path.join(splits_dir, author, "train")
        test_dir = os.path.join(splits_dir, author, "test")
        val_dir = os.path.join(splits_dir, author, "val") if has_val else None

        if not all(
            os.path.exists(d)
            for d in (train_dir, test_dir, *([val_dir] if has_val else []))
        ):
            continue

        results[author] = adapt_author(
            author, train_dir, test_dir, val_dir, ckpt_path=ckpt_path, ckpt_dir=ckpt_dir
        )

    print("-" * 40)
    print(results)
    print("-" * 40)

    return results

def print_worst_samples(samples, n=10):
    """Prints the n samples with the highest (worst) CER"""
    worst = sorted(samples, key=lambda s: s["cer"], reverse=True)[:n]
    for i, s in enumerate(worst, 1):
        print(f"#{i}  CER={s['cer']:.3f}")
        print(f"   Expected:  {s['expected']}")
        print(f"   Predicted: {s['predicted']}")
        print("-" * 40)