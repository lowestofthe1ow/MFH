import os
import re
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

from PIL import Image

from comer.datamodule import vocab

SPLIT_SEED = 2024  # same role as finetune_custom_author.py's SPLIT_SEED

YEARS = ["2014", "2016", "2019"]
DEFAULT_INKML_GT_DIRS = {
    "2014": "data/valid_testing/TestEM2014GT",
    "2016": "data/valid_testing/TEST2016_INKML_GT",
    "2019": "data/valid_testing/Test2019_GT",
}
EXCEPTION_PREFIXES = ("18_em", "RIT_2014")

TEST_FRAC = 0.5
FINETUNE_FRAC = 0.3
VAL_FRAC = 0.2
FINETUNE_SIZE_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]

STEM_RE = re.compile(r"^(?P<name>.+)_(?P<num>\d{1,4}[a-zA-Z]*)$")


class MissingInkmlAuthorError(Exception):
    pass

def naive_author_from_stem(stem):
    """Standard-case author extraction: strip the trailing '_<1-3 digit
    number>', then strip a trailing '_em' if present so IDs match the
    literal inkml writer values (e.g. '37_em_10' -> '37')."""
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"Filename stem '{stem}' doesn't match the expected "
            f"'[name]_[1-3 digit number]' pattern -- check for a naming "
            f"convention this script doesn't know about."
        )
    name = m.group("name")
    if name.endswith("_em"):
        name = name[: -len("_em")]
    return name


def needs_inkml_lookup(stem):
    return stem.startswith(EXCEPTION_PREFIXES)


def _local_tag(elem):
    return elem.tag.rsplit("}", 1)[-1] if "}" in elem.tag else elem.tag


def build_inkml_writer_lookup(inkml_dir):
    """Scans every .inkml file in inkml_dir, returns {stem: writer_id}.
    Same parsing approach as inspect_inkml_authors.py's step 2, reused here
    now that 'writer' is confirmed as the right annotation type."""
    inkml_dir = Path(inkml_dir)
    lookup = {}
    if not inkml_dir.exists():
        return lookup
    for path in sorted(inkml_dir.glob("*.inkml")):
        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            print(f"  [warn] failed to parse {path}: {e}")
            continue
        values = [
            (elem.text or "").strip()
            for elem in tree.getroot().iter()
            if _local_tag(elem) == "annotation" and elem.get("type") == "writer"
        ]
        values = [v for v in values if v]
        if not values:
            continue
        if len(set(values)) > 1:
            print(f"  [warn] {path.stem}: multiple distinct writer values {values}, using first")
        lookup[path.stem] = values[0]
    return lookup


def resolve_author(stem, inkml_lookup):
    if needs_inkml_lookup(stem):
        if stem not in inkml_lookup:
            raise MissingInkmlAuthorError(
                f"'{stem}' matches an exception prefix ({EXCEPTION_PREFIXES}) but has no "
                f"matching .inkml entry in the ground-truth folder. Refusing to silently "
                f"fall back to filename parsing for a stem we know is unreliable -- confirm "
                f"the inkml GT folder covers this file, or adjust EXCEPTION_PREFIXES / the "
                f"GT dir path if it genuinely doesn't need lookup."
            )
        return f"crohme_{inkml_lookup[stem]}"
    return f"crohme_{naive_author_from_stem(stem)}"

def parse_caption_line(line):
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        return None
    img_id, caption = parts
    return img_id, caption


def load_captions(caption_path):
    """Returns list of (img_id, caption), skipping OOV-token samples (same
    policy as finetune_custom_author.py's build_fixed_split)."""
    with open(caption_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]

    paired, removed = [], 0
    for line in lines:
        parsed = parse_caption_line(line)
        if parsed is None:
            print(f"  [warn] unparseable caption line, skipping: {line!r}")
            continue
        img_id, caption = parsed
        oov = [t for t in caption.split() if t not in vocab.word2idx]
        if oov:
            removed += 1
            continue
        paired.append((img_id, caption))
    print(f"  Captions: kept {len(paired)}, removed (OOV) {removed}")
    return paired

def compute_split_sizes(n, fracs=(TEST_FRAC, FINETUNE_FRAC, VAL_FRAC)):
    """
    Returns (n_test, n_finetune, n_val) summing exactly to n, each >=1
    whenever n >= len(fracs) (guaranteed here since we only call this for
    authors with n > min_samples >= 20). Largest-remainder rounding: floor
    each, then hand out leftover units to the largest fractional
    remainders; then top up any zero-sized split by borrowing one unit
    from the currently-largest split.
    """
    raw = [n * f for f in fracs]
    sizes = [int(x) for x in raw]
    remainder = n - sum(sizes)
    order = sorted(range(len(fracs)), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in range(remainder):
        sizes[order[i % len(sizes)]] += 1

    for i in range(len(sizes)):
        while sizes[i] < 1:
            j = max(range(len(sizes)), key=lambda k: sizes[k])
            if sizes[j] <= 1:
                break 
            sizes[j] -= 1
            sizes[i] += 1
    return tuple(sizes)


def nested_finetune_sizes(pool_size, fracs=FINETUNE_SIZE_FRACS):
    sizes = sorted({max(1, round(pool_size * f)) for f in fracs})
    sizes = [s for s in sizes if s <= pool_size]
    if pool_size not in sizes:
        sizes.append(pool_size)
        sizes.sort()
    return sizes

def save_split(items, target_dir, img_dir):
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "caption.txt"), "w", encoding="utf-8") as f:
        for img_id, caption in items:
            f.write(f"{img_id} {caption}\n")
            src_img = os.path.join(img_dir, f"{img_id}.bmp")
            dst_img = os.path.join(target_dir, f"{img_id}.bmp")
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                with Image.open(src_img) as img:
                    img.convert("L").save(dst_img)
            elif not os.path.exists(src_img):
                print(f"    [warn] image not found: {src_img}")


def build_year_splits(year, crohme_root, inkml_gt_dir, out_root, min_samples, split_rng):
    year_dir = Path(crohme_root) / year
    caption_path = year_dir / "caption.txt"
    img_dir = year_dir / "img"
    if not caption_path.exists() or not img_dir.exists():
        print(f"[skip] {year}: missing {caption_path} or {img_dir}")
        return {}

    print(f"\n=== {year} ===")
    print(f"Building inkml writer lookup from {inkml_gt_dir} (exception prefixes only) ...")
    inkml_lookup = build_inkml_writer_lookup(inkml_gt_dir)
    print(f"  {len(inkml_lookup)} inkml entries available for lookup")

    paired = load_captions(caption_path)

    by_author = defaultdict(list)
    n_missing_inkml = 0
    for img_id, caption in paired:
        try:
            author = resolve_author(img_id, inkml_lookup)
        except MissingInkmlAuthorError as e:
            n_missing_inkml += 1
            print(f"  [warn] {e}")
            continue
        by_author[author].append((img_id, caption))
    if n_missing_inkml:
        print(f"  [warn] {n_missing_inkml} samples dropped: exception-prefix stem with no inkml match")

    qualifying = {a: items for a, items in by_author.items() if len(items) > min_samples}
    skipped = len(by_author) - len(qualifying)
    print(f"  {len(by_author)} distinct authors found, {len(qualifying)} with >{min_samples} samples "
          f"({skipped} skipped)")

    year_summary = {}
    for author, items in sorted(qualifying.items()):
        items = list(items)
        split_rng.shuffle(items)
        n = len(items)
        n_test, n_finetune, n_val = compute_split_sizes(n)

        test_items = items[:n_test]
        finetune_pool = items[n_test:n_test + n_finetune]
        val_items = items[n_test + n_finetune:n_test + n_finetune + n_val]
        assert len(test_items) + len(finetune_pool) + len(val_items) == n

        sizes = nested_finetune_sizes(len(finetune_pool))

        author_dir = Path(out_root) / year / author
        save_split(test_items, author_dir / "test", img_dir)
        save_split(val_items, author_dir / "val", img_dir)
        for size in sizes:
            save_split(finetune_pool[:size], author_dir / f"train_{size}", img_dir)

        print(f"  {author:<20} n={n:>4}  test={n_test:>3} finetune_pool={n_finetune:>3} "
              f"val={n_val:>3}  nested_sizes={sizes}")

        year_summary[author] = {
            "n_total": n, "n_test": n_test, "n_finetune_pool": n_finetune, "n_val": n_val,
            "finetune_sizes": sizes,
        }

    return year_summary


def parse_args():
    p = argparse.ArgumentParser(description="Build per-author CROHME 2014/2016/2019 splits")
    p.add_argument("--crohme-root", default="data/crohme")
    p.add_argument("--out-root", default="data/crohme_author_splits")
    p.add_argument("--min-samples", type=int, default=20, help="Authors need MORE than this many samples")
    for year in YEARS:
        p.add_argument(f"--inkml-gt-dir-{year}", default=DEFAULT_INKML_GT_DIRS[year])
    return p.parse_args()


def main():
    args = parse_args()
    split_rng = random.Random(SPLIT_SEED)  # local generator, never touches global random state

    summary = {}
    for year in YEARS:
        inkml_gt_dir = getattr(args, f"inkml_gt_dir_{year}")
        summary[year] = build_year_splits(
            year, args.crohme_root, inkml_gt_dir, args.out_root, args.min_samples, split_rng
        )

    out_path = Path(args.out_root) / "split_summary.json"
    os.makedirs(args.out_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")

    total_authors = sum(len(v) for v in summary.values())
    print(f"Total qualifying authors across all years: {total_authors}")


if __name__ == "__main__":
    main()