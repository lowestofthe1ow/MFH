import os
import torch
import shutil
import random
import argparse
from PIL import Image

from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from scripts.test.finetune_common import evaluate_model, print_worst_samples
from scripts.test.finetune_common import run_experiment, BASE_CKPT_PATH, FINETUNE_CKPT_DIR

CUSTOM_CAPTION_PATH = "data/custom/caption_braced.txt"
CUSTOM_IMG_DIR = "data/custom/author_0/physical"
FINETUNE_SPLITS_DIR = "data/custom/author_0/physical/splits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="finetune adaptation over a custom author dataset"
    )
    parser.add_argument("--caption-path", default=CUSTOM_CAPTION_PATH)
    parser.add_argument("--img-dir", default=CUSTOM_IMG_DIR)
    parser.add_argument("--splits-dir", default=FINETUNE_SPLITS_DIR)
    parser.add_argument("--ckpt-path", default=BASE_CKPT_PATH)
    parser.add_argument("--ckpt-dir", default=FINETUNE_CKPT_DIR)
    return parser.parse_args()


def prepare_custom_dataset(caption_path, img_dir, splits_dir, train_size=5):
    """Handles the dataset for author-specific fine-tuning"""

    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Missing caption file at: {caption_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Missing image directory at: {img_dir}")

    with open(caption_path, "r", encoding="utf-8") as f:
        raw_captions = [line.strip() for line in f.readlines() if line.strip()]

    paired_data = []
    removed_count = 0
    for idx, caption in enumerate(raw_captions):
        oov_tokens = [t for t in caption.split() if t not in vocab.word2idx]
        if oov_tokens:
            removed_count += 1
            continue
        paired_data.append((f"equations_1_{idx:02d}", caption))

    print("=" * 40)

    # NOTE: I already removed the invalid ones, so this should remove 0 samples.
    print(f"Removed: {removed_count}\nRemaining: {len(paired_data)}.")

    random.shuffle(paired_data)

    n_total = len(paired_data)

    # TODO: Decide if we're going to stick with this initial train set size
    train_idx = train_size  # int(n_total * 0.30)
    val_idx = train_idx + 5  # int(n_total * 0.10)

    train_items = paired_data[:train_idx]
    val_items = paired_data[train_idx:val_idx]
    test_items = paired_data[-46:]

    print(
        f"Train: {len(train_items)} ({len(train_items)/n_total*100:.1f}%)\n"
        f"Val:   {len(val_items)} ({len(val_items)/n_total*100:.1f}%)\n"
        f"Test:  {len(test_items)} ({len(test_items)/n_total*100:.1f}%)"
    )

    print("=" * 40)

    author = "custom_author"

    if os.path.exists(splits_dir):
        shutil.rmtree(splits_dir)

    def save_split(split_items, target_dir):
        """Saves the train/test/validation splits to their own subdirectories"""

        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "caption.txt"), "w", encoding="utf-8") as f:
            for img_id, caption in split_items:
                f.write(f"{img_id} {caption}\n")

                img_name = f"{img_id}.bmp"
                src_img = os.path.join(img_dir, img_name)
                dst_img = os.path.join(target_dir, img_name)

                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    with Image.open(src_img) as img:
                        img.convert("L").save(dst_img)
                elif not os.path.exists(src_img):
                    print(f"[Warning] Image file not found: {src_img}")

    save_split(train_items, os.path.join(splits_dir, author, "train"))
    save_split(val_items, os.path.join(splits_dir, author, "val"))
    save_split(test_items, os.path.join(splits_dir, author, "test"))

    return [author]


if __name__ == "__main__":
    args = parse_args()

    results = []

    # TODO: Are we just going to be testing these?
    ablations = [5, 8, 11, 14, 17, 20]

    for train_size in ablations:
        authors = prepare_custom_dataset(
            args.caption_path, args.img_dir, args.splits_dir, train_size=train_size
        )

        result = run_experiment(
            authors,
            args.splits_dir,
            has_val=True,
            ckpt_path=args.ckpt_path,
            ckpt_dir=args.ckpt_dir,
        )
        results.append(result)

        if train_size == 20:
            author = authors[0]
            test_dir = os.path.join(args.splits_dir, author, "test")
            ckpt_file = os.path.join(args.ckpt_dir, f"finetune_model_{author}.pth")

            model = LitCoMER.load_from_checkpoint(args.ckpt_path, strict=False)
            model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))

            _, _, samples = evaluate_model(model, test_dir, return_samples=True)
            print_worst_samples(samples, n=10)

    print(results)
