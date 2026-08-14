import os
import json
import argparse
import traceback
from pathlib import Path

from scripts.test.finetune_common import (
    evaluate_model,
    load_base_model,
    adapt_author,
    mean_std,
    wilson_ci,
    bootstrap_ci_over_seeds,
    exact_mcnemar,
    combine_pvalues_fisher,
    paired_correctness,
    TRAIN_SEEDS,
    BASE_CKPT_PATH,
)

YEARS = ["2014", "2016", "2019"]


def author_ckpt_dir(ckpt_root, split_author_id, year):
    return Path(ckpt_root) / f"author_{split_author_id}" / year


def parse_args():
    p = argparse.ArgumentParser(description="CROHME per-author fine-tuning driver (single size, x5 seeds)")
    p.add_argument("--splits-root", default="data/crohme_author_splits")
    p.add_argument("--split-summary", default=None,
                    help="Defaults to <splits-root>/split_summary.json")
    p.add_argument("--ckpt-path", default=BASE_CKPT_PATH)
    p.add_argument("--ckpt-root", default="finetune_checkpoints_crohme")
    p.add_argument("--years", nargs="+", default=YEARS, choices=YEARS)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Override TRAIN_SEEDS from finetune_common.py if needed")
    p.add_argument("--force", action="store_true",
                    help="Re-run authors that already have an ablation_results.json")
    return p.parse_args()


def process_author(year, author, pool_size, splits_root, ckpt_root, ckpt_path, seeds, baseline_model, force):
    author_dir = Path(splits_root) / year / author
    test_dir = author_dir / "test"
    val_dir = author_dir / "val"
    train_dir = author_dir / f"train_{pool_size}"

    for d in (test_dir, val_dir, train_dir):
        if not d.exists():
            print(f"  [skip] {year}/{author}: missing split dir {d}")
            return False

    out_dir = author_ckpt_dir(ckpt_root, author, year)
    out_path = out_dir / "ablation_results.json"
    if out_path.exists() and not force:
        print(f"  [skip] {year}/{author}: {out_path} already exists (use --force to re-run)")
        return False
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {year}/{author}  (n_finetune_pool={pool_size}) ===")

    base_exprate, base_le1, base_le2, base_cer, base_samples = evaluate_model(
        baseline_model, str(test_dir), return_samples=True
    )
    n_test = len(base_samples)
    print(f"  [Baseline] ExpRate={base_exprate:.4f}  <=1={base_le1:.4f}  <=2={base_le2:.4f}  "
          f"CER={base_cer:.4f}  (n_test={n_test})")

    seed_results = {}
    seed_exprates = []
    for seed in seeds:
        result = adapt_author(
            author, str(train_dir), str(test_dir), val_dir=str(val_dir),
            seed=seed, train_size=pool_size,
            ckpt_path=ckpt_path, ckpt_dir=str(out_dir),
        )
        seed_results[seed] = result
        seed_exprates.append(result["finetune_exprate"])
        print(f"    seed={seed:>4}: ExpRate={result['finetune_exprate']:.4f}  "
              f"<=1={result['finetune_le1']:.4f}  <=2={result['finetune_le2']:.4f}")

    mean_er, std_er = mean_std(seed_exprates)
    boot_lo, boot_hi = bootstrap_ci_over_seeds(seed_exprates)
    print(f"  ExpRate mean={mean_er:.4f} std={std_er:.4f}  95% bootstrap CI over "
          f"{len(seeds)} seeds=({boot_lo:.4f}, {boot_hi:.4f})")

    per_seed_pvalues = []
    for seed in seeds:
        r = seed_results[seed]
        b_correct, a_correct = paired_correctness(base_samples, r["finetune_samples"])
        mc = exact_mcnemar(b_correct, a_correct)
        per_seed_pvalues.append(mc["p_value"])
    fisher_stat, fisher_p = combine_pvalues_fisher(per_seed_pvalues)
    print(f"  McNemar (vs baseline) Fisher-combined p={fisher_p:.4g} across {len(seeds)} seeds")

    representative_seed = min(
        seed_results, key=lambda s: abs(seed_results[s]["finetune_exprate"] - mean_er)
    )

    _, base_lo, base_hi = wilson_ci(sum(s["expected"] == s["predicted"] for s in base_samples), n_test)

    export = {
        "year": year,
        "author": author,
        "n_test": n_test,
        "n_finetune_pool": pool_size,
        "train_seeds": list(seeds),
        "baseline": {
            "exprate": base_exprate, "le1": base_le1, "le2": base_le2, "cer": base_cer,
            "wilson_ci_95": [base_lo, base_hi],
        },
        "ablation": {
            str(pool_size): {
                str(seed): {
                    "exprate": seed_results[seed]["finetune_exprate"],
                    "le1": seed_results[seed]["finetune_le1"],
                    "le2": seed_results[seed]["finetune_le2"],
                    "cer": seed_results[seed]["finetune_cer"],
                    "ckpt": seed_results[seed]["finetune_ckpt"],
                }
                for seed in seeds
            }
        },
        "mcnemar_at_final_size": {
            "per_seed_pvalues": {str(s): p for s, p in zip(seeds, per_seed_pvalues)},
            "fisher_combined_pvalue": fisher_p,
        },
        "representative_seed_at_final_size": representative_seed,
        "mean_exprate_over_seeds": mean_er,
        "std_exprate_over_seeds": std_er,
        "bootstrap_ci_95_over_seeds": [boot_lo, boot_hi],
    }
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Saved {out_path}")
    return True


def main():
    args = parse_args()
    summary_path = Path(args.split_summary) if args.split_summary else Path(args.splits_root) / "split_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found -- run build_crohme_author_splits.py first")
    with open(summary_path) as f:
        split_summary = json.load(f)

    seeds = args.seeds if args.seeds is not None else TRAIN_SEEDS
    print(f"Using seeds: {seeds}")

    print(f"Loading base checkpoint: {args.ckpt_path}")
    baseline_model = load_base_model(args.ckpt_path)

    n_done, n_skipped, n_failed = 0, 0, 0
    for year in args.years:
        year_authors = split_summary.get(year, {})
        if not year_authors:
            print(f"\n[skip] {year}: no qualifying authors in {summary_path}")
            continue
        for author, info in sorted(year_authors.items()):
            try:
                ran = process_author(
                    year, author, info["n_finetune_pool"], args.splits_root, args.ckpt_root,
                    args.ckpt_path, seeds, baseline_model, args.force,
                )
                n_done += int(ran)
                n_skipped += int(not ran)
            except Exception:
                n_failed += 1
                print(f"  [FAILED] {year}/{author}:")
                traceback.print_exc()
                continue

    print(f"\nDone. {n_done} authors fine-tuned, {n_skipped} skipped (already done or missing data), "
          f"{n_failed} failed.")


if __name__ == "__main__":
    main()
