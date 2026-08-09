"""
Author-adaptation ablation over training-set size, for a single custom
author.

Key changes from the original version:

1. FIXED, NESTED SPLIT (build_fixed_split): the train/val/test partition is
   generated exactly ONCE, with its own SPLIT_SEED, decoupled from model
   training seeds. Train sets are nested (train_5 subset of train_8 subset
   of ... subset of train_20) and val/test are identical across every
   ablation size. This replaces the original behavior, which re-shuffled
   the full dataset inside every ablation iteration -- meaning each
   train_size was evaluated against a *different* random test set, and
   (for small datasets) train/test could silently overlap.

2. MULTIPLE SEEDS PER SIZE: each train_size in TRAIN_SIZES is trained with
   every seed in TRAIN_SEEDS (imported from finetune_common), giving 5
   independent replicates per point on the ablation curve.

3. Baseline is evaluated ONCE (deterministic), not redundantly inside every
   adapt_author call.

4. Reporting: mean +/- std across seeds, a bootstrap 95% CI over seeds
   (training-stochasticity uncertainty), and a per-run Wilson 95% CI
   (test-set-size uncertainty, since n=46).

5. McNemar's exact test between baseline and each seed's adapted model at
   the max-data ablation point, plus a Fisher's-method p-value combining
   all 5 seeds.

Results (including per-sample predictions needed for later frequency-domain
analysis) are dumped to ablation_results.json in ckpt_dir.
"""

import os
import json
import random
import argparse
from PIL import Image

from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from scripts.test.finetune_common import (
    evaluate_model,
    print_worst_samples,
    load_base_model,
    adapt_author,
    mean_std,
    wilson_ci,
    bootstrap_ci_over_seeds,
    exact_mcnemar,
    combine_pvalues_fisher,
    paired_correctness,
    TRAIN_SEEDS,
    SPLIT_SEED,
    BASE_CKPT_PATH,
    FINETUNE_CKPT_DIR,
)

CUSTOM_CAPTION_PATH = "data/custom/caption_braced.txt"
CUSTOM_IMG_DIR = "data/custom/author_1/phys_rush"
FINETUNE_SPLITS_DIR = "data/custom/author_1/phys_rush/splits"

TRAIN_SIZES = [5, 8, 11, 14, 17, 20]
VAL_SIZE = 5
TEST_SIZE = 46


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


def build_fixed_split(caption_path, img_dir, splits_dir):
    """
    Builds ONE fixed, nested train/val/test partition, shared by every
    ablation size and every training seed.

    - test (TEST_SIZE items) and val (VAL_SIZE items) are fixed and never
      touched by the ablation loop.
    - train sets are NESTED: train_5 subset of train_8 subset of ...
      subset of train_20, so the ablation curve reflects "more data helps"
      rather than "different random data helps."
    - Shuffling happens exactly once, via a local RNG seeded by SPLIT_SEED
      that never touches global random state -- so it can't be perturbed by,
      or accidentally reused as, one of the TRAIN_SEEDS used later for
      model stochasticity.

    Returns the author name ("custom_author").
    """
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
    # NOTE: invalid samples were already removed upstream, so this should
    # remove 0 here.
    print(f"Removed: {removed_count}\nRemaining: {len(paired_data)}.")

    max_train = max(TRAIN_SIZES)
    n_needed = max_train + VAL_SIZE + TEST_SIZE
    if len(paired_data) < n_needed:
        raise ValueError(
            f"Need at least {n_needed} usable samples for a fixed, non-overlapping "
            f"nested split (train_max={max_train} + val={VAL_SIZE} + test={TEST_SIZE}), "
            f"but only {len(paired_data)} remain after OOV filtering. The original "
            f"code's `paired_data[-46:]` test slice could silently overlap with "
            f"training data in this situation -- this check exists to catch that."
        )

    split_rng = random.Random(SPLIT_SEED)  # local generator: doesn't touch global state
    split_rng.shuffle(paired_data)

    train_pool = paired_data[:max_train]
    val_items = paired_data[max_train:max_train + VAL_SIZE]
    test_items = paired_data[max_train + VAL_SIZE:max_train + VAL_SIZE + TEST_SIZE]

    print(
        f"Fixed split (SPLIT_SEED={SPLIT_SEED}): "
        f"train_pool={len(train_pool)}, val={len(val_items)}, test={len(test_items)}"
    )
    print("=" * 40)

    author = "custom_author"

    if os.path.exists(splits_dir):
        import shutil
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

    # val and test are identical across every ablation size
    save_split(val_items, os.path.join(splits_dir, author, "val"))
    save_split(test_items, os.path.join(splits_dir, author, "test"))

    # nested train subsets
    for size in TRAIN_SIZES:
        save_split(train_pool[:size], os.path.join(splits_dir, author, f"train_{size}"))

    return author


if __name__ == "__main__":
    args = parse_args()

    author = build_fixed_split(args.caption_path, args.img_dir, args.splits_dir)
    test_dir = os.path.join(args.splits_dir, author, "test")
    val_dir = os.path.join(args.splits_dir, author, "val")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- baseline: deterministic at eval time, so evaluate exactly once ----
    baseline_model = load_base_model(args.ckpt_path)
    base_exprate, base_le1, base_le2, base_cer, base_samples = evaluate_model(
        baseline_model, test_dir, return_samples=True
    )
    n_test = len(base_samples)
    base_correct_count = sum(s["expected"] == s["predicted"] for s in base_samples)
    print(
        f"\n[Baseline] ExpRate={base_exprate:.4f}  <=1 error={base_le1:.4f}  "
        f"<=2 error={base_le2:.4f}  CER={base_cer:.4f}  (n_test={n_test})"
    )

    # ---- ablation sweep: every train_size x every seed ----
    all_results = {}  # all_results[size][seed] -> adapt_author() result dict

    for size in TRAIN_SIZES:
        train_dir = os.path.join(args.splits_dir, author, f"train_{size}")
        all_results[size] = {}
        seed_exprates, seed_le1s, seed_le2s = [], [], []

        for seed in TRAIN_SEEDS:
            print(f"\n=== train_size={size}  seed={seed} ===")
            result = adapt_author(
                author, train_dir, test_dir, val_dir=val_dir,
                seed=seed, train_size=size,
                ckpt_path=args.ckpt_path, ckpt_dir=args.ckpt_dir,
            )
            all_results[size][seed] = result
            seed_exprates.append(result["finetune_exprate"])
            seed_le1s.append(result["finetune_le1"])
            seed_le2s.append(result["finetune_le2"])
            print(
                f"  ExpRate={result['finetune_exprate']:.4f}  "
                f"<=1 error={result['finetune_le1']:.4f}  <=2 error={result['finetune_le2']:.4f}  "
                f"CER={result['finetune_cer']:.4f}"
            )

        mean_er, std_er = mean_std(seed_exprates)
        boot_lo, boot_hi = bootstrap_ci_over_seeds(seed_exprates)
        mean_le1, std_le1 = mean_std(seed_le1s)
        mean_le2, std_le2 = mean_std(seed_le2s)
        print(
            f"[train_size={size}] ExpRate mean={mean_er:.4f} std={std_er:.4f}  "
            f"95% bootstrap CI over {len(TRAIN_SEEDS)} seeds=({boot_lo:.4f}, {boot_hi:.4f})  "
            f"| <=1 error mean={mean_le1:.4f} std={std_le1:.4f}  "
            f"<=2 error mean={mean_le2:.4f} std={std_le2:.4f}"
        )

    # ---- per-run Wilson 95% CIs (uncertainty from test-set size, n=46) ----
    # ExpRate, <=1 error, and <=2 error are all proportions over the same
    # n_test items, so the same Wilson interval applies to each directly.
    print("\n" + "=" * 60)
    print(f"Per-run Wilson 95% CIs (test-set uncertainty, n={n_test})")
    print("=" * 60)
    base_le1_count = sum(s["edit_distance"] <= 1 for s in base_samples)
    base_le2_count = sum(s["edit_distance"] <= 2 for s in base_samples)
    _, blo, bhi = wilson_ci(base_correct_count, n_test)
    _, ble1lo, ble1hi = wilson_ci(base_le1_count, n_test)
    _, ble2lo, ble2hi = wilson_ci(base_le2_count, n_test)
    print(
        f"Baseline: ExpRate={base_exprate:.4f} CI=({blo:.4f}, {bhi:.4f})  "
        f"<=1 err={base_le1:.4f} CI=({ble1lo:.4f}, {ble1hi:.4f})  "
        f"<=2 err={base_le2:.4f} CI=({ble2lo:.4f}, {ble2hi:.4f})"
    )
    for size in TRAIN_SIZES:
        for seed in TRAIN_SEEDS:
            r = all_results[size][seed]
            k = sum(s["expected"] == s["predicted"] for s in r["finetune_samples"])
            k_le1 = sum(s["edit_distance"] <= 1 for s in r["finetune_samples"])
            k_le2 = sum(s["edit_distance"] <= 2 for s in r["finetune_samples"])
            _, lo, hi = wilson_ci(k, n_test)
            _, le1lo, le1hi = wilson_ci(k_le1, n_test)
            _, le2lo, le2hi = wilson_ci(k_le2, n_test)
            print(
                f"  size={size:>2} seed={seed:>4}: ExpRate={r['finetune_exprate']:.4f} CI=({lo:.4f}, {hi:.4f})  "
                f"<=1={r['finetune_le1']:.4f} CI=({le1lo:.4f}, {le1hi:.4f})  "
                f"<=2={r['finetune_le2']:.4f} CI=({le2lo:.4f}, {le2hi:.4f})"
            )

    # ---- McNemar's exact test: baseline vs. each seed's adapted model, at
    # the max-data ablation point (the primary "adapted model" per the
    # earlier discussion on not selecting-by-test-accuracy) ----
    final_size = max(TRAIN_SIZES)
    print("\n" + "=" * 60)
    print(f"McNemar's exact test: baseline vs. adapted (train_size={final_size})")
    print("=" * 60)

    per_seed_pvalues = []
    for seed in TRAIN_SEEDS:
        r = all_results[final_size][seed]
        b_correct, a_correct = paired_correctness(base_samples, r["finetune_samples"])
        mc = exact_mcnemar(b_correct, a_correct)
        per_seed_pvalues.append(mc["p_value"])
        print(
            f"  seed={seed:>4}: n01(base wrong -> adapted correct)={mc['n01_base_wrong_adapted_correct']:>2}  "
            f"n10(base correct -> adapted wrong)={mc['n10_base_correct_adapted_wrong']:>2}  "
            f"p={mc['p_value']:.4g}  ({mc['direction']})"
        )

    fisher_stat, fisher_p = combine_pvalues_fisher(per_seed_pvalues)
    print(
        f"\nFisher's combined p-value across {len(TRAIN_SEEDS)} independent seeds: "
        f"{fisher_p:.4g}  (chi2 stat={fisher_stat:.3f}, df={2 * len(TRAIN_SEEDS)})"
    )

    # ---- qualitative inspection: pick the seed closest to the mean ExpRate
    # at the final size, rather than cherry-picking the best-performing one ----
    final_seed_results = all_results[final_size]
    final_mean_er, _ = mean_std([r["finetune_exprate"] for r in final_seed_results.values()])
    representative_seed = min(
        final_seed_results,
        key=lambda s: abs(final_seed_results[s]["finetune_exprate"] - final_mean_er),
    )
    print(
        f"\nRepresentative seed for qualitative inspection at train_size={final_size}: "
        f"seed={representative_seed} (closest to the {len(TRAIN_SEEDS)}-seed mean ExpRate)"
    )
    print_worst_samples(final_seed_results[representative_seed]["finetune_samples"], n=10)

    # ---- persist a compact summary for building the ablation-curve figure
    # and for the frequency-domain analysis that consumes these checkpoints ----
    export = {
        "split_seed": SPLIT_SEED,
        "train_seeds": TRAIN_SEEDS,
        "n_test": n_test,
        "baseline": {
            "exprate": base_exprate,
            "le1": base_le1,
            "le2": base_le2,
            "cer": base_cer,
        },
        "ablation": {
            str(size): {
                str(seed): {
                    "exprate": all_results[size][seed]["finetune_exprate"],
                    "le1": all_results[size][seed]["finetune_le1"],
                    "le2": all_results[size][seed]["finetune_le2"],
                    "cer": all_results[size][seed]["finetune_cer"],
                    "ckpt": all_results[size][seed]["finetune_ckpt"],
                }
                for seed in TRAIN_SEEDS
            }
            for size in TRAIN_SIZES
        },
        "mcnemar_at_final_size": {
            "per_seed_pvalues": dict(zip(TRAIN_SEEDS, per_seed_pvalues)),
            "fisher_combined_pvalue": fisher_p,
        },
        "representative_seed_at_final_size": representative_seed,
    }
    export_path = os.path.join(args.ckpt_dir, "ablation_results.json")
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nSaved summary to {export_path}")