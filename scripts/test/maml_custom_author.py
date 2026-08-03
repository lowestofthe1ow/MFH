import os
import shutil
import random
import argparse
from PIL import Image

from comer.datamodule import vocab
from maml_common import run_experiment, BASE_CKPT_PATH, MAML_CKPT_DIR

CUSTOM_CAPTION_PATH = "data/custom/caption.txt"
CUSTOM_IMG_DIR = "data/custom/author_0/digital"
MAML_SPLITS_DIR = "data/custom/author_0/digital/splits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAML adaptation over a custom author dataset"
    )
    parser.add_argument("--caption-path", default=CUSTOM_CAPTION_PATH)
    parser.add_argument("--img-dir", default=CUSTOM_IMG_DIR)
    parser.add_argument("--splits-dir", default=MAML_SPLITS_DIR)
    parser.add_argument("--ckpt-path", default=BASE_CKPT_PATH)
    parser.add_argument("--ckpt-dir", default=MAML_CKPT_DIR)
    return parser.parse_args()


def prepare_custom_dataset(caption_path, img_dir, splits_dir):
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Missing caption file at: {caption_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Missing image directory at: {img_dir}")

    with open(caption_path, "r", encoding="utf-8") as f:
        raw_captions = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(raw_captions)} total equation expressions from {caption_path}.")

    paired_data = []
    removed_count = 0
    for idx, caption in enumerate(raw_captions):
        oov_tokens = [t for t in caption.split() if t not in vocab.word2idx]
        if oov_tokens:
            removed_count += 1
            print(
                f"[Info] Removing item index {idx} due to unknown symbols {set(oov_tokens)}: '{caption}'"
            )
            continue
        paired_data.append((f"equations_1_{idx:02d}", caption))

    print(
        f"Filtered out {removed_count} samples containing unknown symbols. Remaining valid samples: {len(paired_data)}."
    )

    random.shuffle(paired_data)

    n_total = len(paired_data)
    train_idx = 5  # int(n_total * 0.30)
    val_idx = train_idx + 5  # int(n_total * 0.10)

    train_items = paired_data[:train_idx]
    val_items = paired_data[train_idx:val_idx]
    test_items = paired_data[-46:]

    print(
        f"Split distribution -> Train: {len(train_items)} ({len(train_items)/n_total*100:.1f}%) | "
        f"Val: {len(val_items)} ({len(val_items)/n_total*100:.1f}%) | "
        f"Test: {len(test_items)} ({len(test_items)/n_total*100:.1f}%)"
    )

    author = "custom_author"

    if os.path.exists(splits_dir):
        shutil.rmtree(splits_dir)

    def save_split(split_items, target_dir):
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


def main():
    args = parse_args()
    authors = prepare_custom_dataset(args.caption_path, args.img_dir, args.splits_dir)
    run_experiment(
        authors,
        args.splits_dir,
        "MAML ADAPTATION EXPERIMENT REPORT (CUSTOM AUTHOR)",
        has_val=True,
        ckpt_path=args.ckpt_path,
        ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
