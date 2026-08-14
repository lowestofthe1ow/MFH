"""
Turns freq_analysis_results.json (produced by run_frequency_analysis.py)
into two sets of figures:

  1. Energy-map heatmaps, one per category (+ one combined comparison
     figure), from results["input_spectra"]["result"][category]["energy_map"].

  2. FAB attention-map deltas (adapted - baseline), one figure per
     author/category label, from
     results[label]["fab_attention"]["result"]["baseline"/"adapted"]
     ["mean_map_spatial"/"mean_map_frequency"].
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

def safe_slug(label):
    return label.replace("/", "_").replace(" ", "_")

# ---------------------------------------------------------------------------
# 1. Energy-map heatmaps per category
# ---------------------------------------------------------------------------

def _draw_energy_heatmap(ax, energy_map, title, retained_frac, n_images, patch_size, retain_m):
    energy_map = np.asarray(energy_map, dtype=np.float64)
    positive = energy_map[energy_map > 0]
    vmin = positive.min() if positive.size else 1e-8
    vmax = energy_map.max() if energy_map.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 10

    im = ax.imshow(energy_map, cmap="magma", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(f"{title}\n(n={n_images} imgs, retained energy={retained_frac:.1%})", fontsize=10)
    ax.set_xlabel("horizontal freq (u)")
    ax.set_ylabel("vertical freq (v)")
    ax.set_xticks(range(patch_size))
    ax.set_yticks(range(patch_size))

    lo = patch_size - retain_m
    ax.add_patch(patches.Rectangle(
        (lo - 0.5, lo - 0.5), retain_m, retain_m,
        linewidth=2, edgecolor="cyan", facecolor="none",
    ))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |DCT coef| (log)")

def plot_energy_map_heatmaps(spectra_block, out_dir, patch_size, retain_m):
    if spectra_block is None or not spectra_block.get("ok"):
        return

    data = spectra_block["result"]
    if not data:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(data.keys())

    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels), 4.4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        entry = data[label]
        _draw_energy_heatmap(ax, entry["energy_map"], label, entry["retained_energy_frac"],
                              entry["n_images"], patch_size, retain_m)
    fig.suptitle("Input Patch-DCT Energy Maps by Category", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    combined_path = out_dir / "energy_maps_all_categories.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    for label in labels:
        entry = data[label]
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        _draw_energy_heatmap(ax, entry["energy_map"], label, entry["retained_energy_frac"],
                              entry["n_images"], patch_size, retain_m)
        fig.tight_layout()
        out_path = out_dir / f"energy_map_{safe_slug(label)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

# ---------------------------------------------------------------------------
# 2. FAB attention-map deltas per author/category
# ---------------------------------------------------------------------------

def _draw_delta_heatmap(ax, delta_map, title):
    delta_map = np.asarray(delta_map, dtype=np.float64)
    vmax = float(np.abs(delta_map).max()) or 1e-6
    im = ax.imshow(delta_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="adapted \u2212 baseline")

def plot_attention_deltas(results, out_dir):
    labels = [k for k in results.keys() if k != "input_spectra"]
    if not labels:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        fab_block = results[label].get("fab_attention")
        if fab_block is None or not fab_block.get("ok"):
            continue

        result = fab_block["result"]
        baseline, adapted, n_samples = result["baseline"], result["adapted"], result["n_samples"]

        delta_spatial = np.array(adapted["mean_map_spatial"]) - np.array(baseline["mean_map_spatial"])
        delta_freq = np.array(adapted["mean_map_frequency"]) - np.array(baseline["mean_map_frequency"])

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
        _draw_delta_heatmap(axes[0], delta_spatial, f"Spatial Gate \u0394 (x)")
        _draw_delta_heatmap(axes[1], delta_freq, f"Frequency Gate \u0394 (y)")
        
        fig.suptitle(
            f"{label}  \u2014  FAB attention delta (adapted \u2212 baseline), n={n_samples}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = out_dir / f"attention_delta_{safe_slug(label)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot energy-map heatmaps and FAB attention deltas from json"
    )
    p.add_argument("--results", default="freq_analysis_results.json")
    p.add_argument("--out-dir", default="freq_analysis_plots")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--retain-m", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found.")
        
    with open(results_path) as f:
        results = json.load(f)

    out_dir = Path(args.out_dir)

    plot_energy_map_heatmaps(results.get("input_spectra"), out_dir / "energy_maps",
                              patch_size=args.patch_size, retain_m=args.retain_m)
    plot_attention_deltas(results, out_dir / "attention_deltas")

    print(f"\nDone. Figures written under {out_dir}/")

if __name__ == "__main__":
    main()