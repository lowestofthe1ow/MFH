import os
import shutil
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

from finetune_common import run_experiment, BASE_CKPT_PATH, FINETUNE_CKPT_DIR

SOURCE_DATA_DIR = "data/crohme/2016"
FINETUNE_SPLITS_DIR = "data/crohme/author_splits"
MIN_SAMPLES_REQUIRED = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="finetune adaptation over CROHME 2016 authors"
    )
    parser.add_argument("--source-data-dir", default=SOURCE_DATA_DIR)
    parser.add_argument("--splits-dir", default=FINETUNE_SPLITS_DIR)
    parser.add_argument("--ckpt-path", default=BASE_CKPT_PATH)
    parser.add_argument("--ckpt-dir", default=FINETUNE_CKPT_DIR)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_REQUIRED)
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Limit to the first N filtered authors, for validation testing",
    )
    return parser.parse_args()


def prepare_FINETUNE_datasets(
    source_data_dir, splits_dir, min_samples, max_authors=None
):
    """Handles the datasets for author-specific fine-tuning"""

    caption_path = os.path.join(source_data_dir, "caption.txt")
    img_dir = os.path.join(source_data_dir, "img")

    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Missing {caption_path}")

    with open(caption_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    all_author_data = defaultdict(list)
    for line in lines:
        author_id = line.split()[0][:6]
        all_author_data[author_id].append(line)

    print(f"Found {len(all_author_data)} unique authors total in the 2016 dataset.")

    author_data = {
        auth: lines
        for auth, lines in all_author_data.items()
        if len(lines) >= min_samples
    }
    print(
        f"Filtered down to {len(author_data)} authors with >= {min_samples} total samples."
    )

    authors_to_process = list(author_data.keys())
    if max_authors:
        authors_to_process = authors_to_process[:max_authors]
        print(
            f"Limiting execution to the first {max_authors} filtered authors for validation testing."
        )

    if os.path.exists(splits_dir):
        shutil.rmtree(splits_dir)

    def save_split(split_lines, target_dir):
        with open(os.path.join(target_dir, "caption.txt"), "w") as f:
            for line in split_lines:
                f.write(line + "\n")
                img_name = line.split()[0] + ".bmp"
                src_img = os.path.join(img_dir, img_name)
                dst_img = os.path.join(target_dir, img_name)
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)

    for author in tqdm(authors_to_process, desc="Generating filtered splits"):
        lines = author_data[author]
        random.shuffle(lines)

        split_idx = int(len(lines) * 0.8)
        train_dir = os.path.join(splits_dir, author, "train")
        test_dir = os.path.join(splits_dir, author, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        save_split(lines[:split_idx], train_dir)
        save_split(lines[split_idx:], test_dir)

    return authors_to_process


if __name__ == "__main__":
    args = parse_args()
    authors = prepare_FINETUNE_datasets(
        args.source_data_dir, args.splits_dir, args.min_samples, args.max_authors
    )
    run_experiment(
        authors,
        args.splits_dir,
        ckpt_path=args.ckpt_path,
        ckpt_dir=args.ckpt_dir,
    )
