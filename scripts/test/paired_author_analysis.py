import json
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.test.build_report import discover_results


def load_author_row(json_path, use_representative_seed):
    with open(json_path) as f:
        data = json.load(f)

    baseline_er = data["baseline"]["exprate"]
    rep_seed = data.get("representative_seed_at_final_size")
    final_size = max(int(s) for s in data["ablation"].keys())

    if use_representative_seed:
        if rep_seed is None:
            return None
        entry = data["ablation"].get(str(final_size), {}).get(str(rep_seed))
        if entry is None:
            return None
        adapted_er = entry["exprate"]
    else:
        adapted_er = data.get("mean_exprate_over_seeds")
        if adapted_er is None:
            return None

    mcnemar_p = data.get("mcnemar_at_final_size", {}).get("fisher_combined_pvalue")

    return {
        "year": data.get("year"),
        "author": data.get("author"),
        "n_test": data.get("n_test"),
        "n_finetune_pool": data.get("n_finetune_pool"),
        "baseline_exprate": baseline_er,
        "adapted_exprate": adapted_er,
        "delta": adapted_er - baseline_er,
        "mcnemar_fisher_p": mcnemar_p,
    }


def benjamini_hochberg(pvalues):
    """Returns FDR-corrected q-values, same order as input. Standard BH
    step-up procedure; NaN p-values (missing McNemar result) pass through
    as NaN."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    valid_mask = ~np.isnan(pvalues)
    q = np.full(n, np.nan)

    valid_idx = np.where(valid_mask)[0]
    valid_p = pvalues[valid_idx]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)

    q_ranked = ranked_p * m / (np.arange(m) + 1)
    # enforce monotonicity (step-up): q_i = min(q_i, q_{i+1}, ..., q_m)
    for i in range(m - 2, -1, -1):
        q_ranked[i] = min(q_ranked[i], q_ranked[i + 1])
    q_ranked = np.clip(q_ranked, 0, 1)

    q_valid = np.empty(m)
    q_valid[order] = q_ranked
    q[valid_idx] = q_valid
    return q.tolist()


def paired_delta_test(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    mean_d, std_d = float(np.mean(deltas)), float(np.std(deltas, ddof=1))

    t_stat, t_p = stats.ttest_1samp(deltas, 0.0)
    se = std_d / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_lo, ci_hi = mean_d - tcrit * se, mean_d + tcrit * se

    cohens_d = mean_d / std_d if std_d > 0 else float("nan")

    if np.allclose(deltas, 0):
        w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = stats.wilcoxon(deltas)

    return {
        "n": n, "mean_delta": mean_d, "std_delta": std_d,
        "ci_95": (ci_lo, ci_hi),
        "t_stat": float(t_stat), "t_pvalue": float(t_p),
        "cohens_d": cohens_d,
        "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Paired across-author ExpRate delta analysis")
    p.add_argument("--ckpt-root", default="finetune_checkpoints_crohme")
    p.add_argument("--use-representative-seed", action="store_true",
                    help="Use each author's single representative-seed adapted ExpRate instead "
                         "of the default mean-over-5-seeds")
    p.add_argument("--out", default="paired_author_analysis.json")
    return p.parse_args()


def main():
    args = parse_args()
    entries = discover_results(Path(args.ckpt_root))
    if not entries:
        raise FileNotFoundError(f"No ablation_results.json found under {args.ckpt_root}")

    rows = []
    for author, year, json_path in entries:
        row = load_author_row(json_path, args.use_representative_seed)
        if row is None:
            print(f"[skip] {author}/{year}: missing required fields in {json_path}")
            continue
        rows.append(row)

    if not rows:
        print("No usable rows -- nothing to analyze.")
        return

    mcnemar_ps = [r["mcnemar_fisher_p"] if r["mcnemar_fisher_p"] is not None else float("nan") for r in rows]
    q_values = benjamini_hochberg(mcnemar_ps)
    for r, q in zip(rows, q_values):
        r["mcnemar_fisher_q_bh"] = q

    adapted_col = "adapted_exprate (repseed)" if args.use_representative_seed else "adapted_exprate (mean-5-seed)"
    print(f"\n{'author':<20} {'year':<6} {'n_test':>7} {'baseline':>9} {adapted_col:>26} "
          f"{'delta':>8} {'mcnemar_p':>10} {'BH_q':>8}")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: (r["year"], r["author"])):
        mp = f"{r['mcnemar_fisher_p']:.4g}" if r["mcnemar_fisher_p"] is not None else "n/a"
        q = f"{r['mcnemar_fisher_q_bh']:.4g}" if not np.isnan(r["mcnemar_fisher_q_bh"]) else "n/a"
        print(f"{r['author']:<20} {r['year']:<6} {r['n_test']:>7} {r['baseline_exprate']:>9.4f} "
              f"{r['adapted_exprate']:>26.4f} {r['delta']:>+8.4f} {mp:>10} {q:>8}")

    deltas = [r["delta"] for r in rows]
    result = paired_delta_test(deltas)

    print("\n" + "=" * 60)
    print(f"Paired across-author analysis (n={result['n']} authors, "
          f"using {'representative seed' if args.use_representative_seed else 'mean over seeds'})")
    print("=" * 60)
    print(f"Mean delta (adapted - baseline) = {result['mean_delta']:+.4f}  (std={result['std_delta']:.4f})")
    print(f"95% CI on mean delta            = ({result['ci_95'][0]:+.4f}, {result['ci_95'][1]:+.4f})")
    print(f"Paired t-test:    t={result['t_stat']:.3f}, df={result['n']-1}, p={result['t_pvalue']:.4g}")
    print(f"Wilcoxon signed-rank: W={result['wilcoxon_stat']:.1f}, p={result['wilcoxon_pvalue']:.4g}")
    print(f"Cohen's d (paired) = {result['cohens_d']:.3f}")

    n_sig_uncorrected = sum(1 for r in rows if r["mcnemar_fisher_p"] is not None and r["mcnemar_fisher_p"] < 0.05)
    n_sig_bh = sum(1 for r in rows if not np.isnan(r["mcnemar_fisher_q_bh"]) and r["mcnemar_fisher_q_bh"] < 0.05)
    print(f"\nPer-author McNemar (Fisher-combined) at alpha=0.05: {n_sig_uncorrected}/{len(rows)} "
          f"significant uncorrected, {n_sig_bh}/{len(rows)} significant after Benjamini-Hochberg FDR correction")

    with open(args.out, "w") as f:
        json.dump({"rows": rows, "paired_test": result}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
