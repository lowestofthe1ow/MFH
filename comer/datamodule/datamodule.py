import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from comer.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 24e4  # change here according to your GPU memory


# load data
def data_iterator(
        data: Data,
        batch_size: int,
        batch_Imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )

        if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
            fname_total.append(fname_batch)
            feature_total.append(feature_batch)
            label_total.append(label_batch)
            i = 0
            biggest_image_size = size
            fname_batch = []
            feature_batch = []
            label_batch = []
            fname_batch.append(fname)
            feature_batch.append(fea)
            label_batch.append(lab)
            i += 1
        else:
            fname_batch.append(fname)
            feature_batch.append(fea)
            label_batch.append(lab)
            i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data_from_dir(base_dir: str) -> Data:
    """Extracts all data needed for a dataset directly from an unzipped local directory.
    
    Args:
        base_dir (str): Path to the folder containing caption.txt and image assets
        
    Returns:
        Data: list of tuple of image and formula
    """
    caption_path = os.path.join(base_dir, "caption.txt")
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Missing caption.txt inside directory: {base_dir}")

    with open(caption_path, "r", encoding="utf-8") as f:
        captions = f.readlines()
        
    data = []
    for line in captions:
        line_str = line.strip()
        if not line_str:
            continue
            
        tmp = line_str.split()
        img_name = tmp[0]
        formula = tmp[1:]
        
        # Look for loose images (your split folder) or nested structures (data/2014/img/)
        img_path = os.path.join(base_dir, f"{img_name}.bmp")
        if not os.path.exists(img_path):
            img_path = os.path.join(base_dir, "img", f"{img_name}.bmp")
            
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_name}.bmp missing from {base_dir}. Skipping.")
            continue
            
        # Move image data to memory immediately to avoid lazy loading errors
        with Image.open(img_path) as img_file:
            img = img_file.copy()
            
        data.append((img_name, img, formula))

    print(f"Extract data from folder: {base_dir}, with data size: {len(data)}")
    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]
    img_dct: FloatTensor  # [b, 1, H, W]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
            img_dct=self.img_dct.to(device)
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]
    img_dct = batch[3]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]
    heights_dct = [s.size(1) for s in img_dct]
    widths_dct = [s.size(2) for s in img_dct]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)
    max_height_dct = max(heights_dct)
    max_width_dct = max(widths_dct)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    x_dct = torch.zeros(n_samples, 1, max_height_dct, max_width_dct)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0
    for idx, s_x in enumerate(img_dct):
        x_dct[idx, :, :heights_dct[idx], : widths_dct[idx]] = s_x

    return Batch(fnames, x, x_mask, seqs_y, x_dct)


def build_dataset_unzipped(folder_path: str, batch_size: int):
    data = extract_data_from_dir(folder_path)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            train_dir: str,   # CHANGED: pass train folder
            val_dir: str,     # CHANGED: pass test/val folder
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            freq_remove_num: int = 3,
            scale_aug: bool = False,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.freq_remove_num = freq_remove_num
        self.scale_aug = scale_aug

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                build_dataset_unzipped(self.train_dir, self.train_batch_size),
                True, self.scale_aug, self.freq_remove_num
            )
            # Now uses the author's specific test set for validation
            self.val_dataset = CROHMEDataset(
                build_dataset_unzipped(self.val_dir, self.eval_batch_size),
                False, self.scale_aug, self.freq_remove_num
            )
        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                build_dataset_unzipped(self.val_dir, self.eval_batch_size),
                False, self.scale_aug, self.freq_remove_num
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )