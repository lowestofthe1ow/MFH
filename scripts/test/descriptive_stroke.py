"""
scripts/compare_stroke_structure.py

Exploratory comparison of low-level stroke structure between digital and
physical versions of handwritten equations using pandas and matplotlib.
Outputs summary tables formatted directly in LaTeX.
"""

import os
import sys
import argparse

import numpy as np
import cv2
import pandas as pd

# Fix for headless environments / _tkinter errors
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comer.datamodule.transforms import ScaleToLimitRange  # noqa: E402

CAPTION_PATH = "data/custom/caption.txt"
DIGITAL_DATA_DIR = "data/custom/author_0/digital"
PHYSICAL_DATA_DIR = "data/custom/author_0/phys_norush"
H_LO, H_HI, W_LO, W_HI = 16, 256, 16, 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Compare stroke metrics.")
    parser.add_argument("--caption-path", default=CAPTION_PATH)
    parser.add_argument("--digital-dir", default=DIGITAL_DATA_DIR)
    parser.add_argument("--physical-dir", default=PHYSICAL_DATA_DIR)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--small-speck-area", type=int, default=2)
    parser.add_argument("--use-resized-images", action="store_true")
    return parser.parse_args()


def load_binary_array(img_dir, img_id):
    img_path = os.path.join(img_dir, f"{img_id}.bmp")
    if not os.path.exists(img_path):
        return None
    with Image.open(img_path) as img_file:
        return np.array(img_file.convert("L")) > 127


def resize_and_rebinarize(binary_mask, transform):
    return transform((binary_mask.astype(np.uint8)) * 255) > 127


def fragmentation_stats(binary_mask, connectivity, small_speck_area):
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=connectivity
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    n_comp = len(areas)
    n_small = int((areas <= small_speck_area).sum())

    total_ink = areas.sum() if n_comp > 0 else 0
    mean_area = areas.mean() if n_comp > 0 else np.nan

    denoised_areas = areas[areas > small_speck_area]
    denoised_ink = denoised_areas.sum() if len(denoised_areas) > 0 else 0
    mean_area_denoised = denoised_areas.mean() if len(denoised_areas) > 0 else np.nan

    # Component Density: connected components per 1,000 ink pixels
    density_ink = (n_comp / total_ink * 1000.0) if total_ink > 0 else np.nan
    density_ink_denoised = (
        (len(denoised_areas) / denoised_ink * 1000.0) if denoised_ink > 0 else np.nan
    )

    stats_dict = {
        "n_components": n_comp,
        "small_frac": n_small / n_comp if n_comp > 0 else np.nan,
        "mean_area": mean_area,
        "mean_area_denoised": mean_area_denoised,
        "density_ink": density_ink,
        "density_ink_denoised": density_ink_denoised,
    }
    return stats_dict, areas


def stroke_width_stats(binary_mask):
    if not binary_mask.any():
        return {"width_mean": np.nan, "width_cv": np.nan}

    dist = ndimage.distance_transform_edt(binary_mask)
    widths = 2.0 * dist[binary_mask]

    mean_w = widths.mean()
    std_w = widths.std(ddof=1) if widths.size > 1 else 0.0
    return {
        "width_mean": mean_w,
        "width_cv": (std_w / mean_w) if mean_w > 0 else np.nan,
    }


def extract_all_stats(binary_mask, args):
    frag, areas = fragmentation_stats(
        binary_mask, args.connectivity, args.small_speck_area
    )
    width = stroke_width_stats(binary_mask)
    return {**frag, **width}, areas


def save_combined_histogram(digital_areas, physical_areas, label, filepath):
    if len(digital_areas) == 0 and len(physical_areas) == 0:
        return

    combined = np.concatenate([digital_areas, physical_areas])
    bins = np.histogram_bin_edges(combined, bins=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.hist(digital_areas, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Digital")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(physical_areas, bins=bins, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_title("Physical")
    ax2.set_xlabel("Component Area (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Connected Component Size Distribution ({label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved combined histogram to {filepath}")


def print_latex_table(df, label):
    metrics = [c for c in df.columns if c not in ("img_id", "modality")]
    piv = df.pivot(index="img_id", columns="modality")

    metric_labels = {
        "n_components": r"Component Count $N_{\text{cc}}$",
        "small_frac": r"Speck Fraction ($\le \tau_{\text{speck}}$)",
        "mean_area": r"Mean Area $\mu_A$ (px)",
        "mean_area_denoised": r"Denoised Mean Area $\mu_{A,\text{denoised}}$ (px)",
        "density_ink": r"Ink Density $D_{\text{ink}}$ (/1k px)",
        "density_ink_denoised": r"Denoised Density $D_{\text{ink,denoised}}$ (/1k px)",
        "width_mean": r"Stroke Width Mean $\mu_W$ (px)",
        "width_cv": r"Stroke Width $\text{CV}_W$",
    }

    print(f"\n% --- LaTeX Table: {label} ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Comparison of Stroke Structure Metrics Between Digital and Physical Modalities ("
        + label
        + r")}"
    )
    print(r"\label{tab:stroke_metrics_" + label.lower().split()[0] + r"}")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(
        r"\textbf{Metric} & \textbf{Digital (Mean $\pm$ SD)} & \textbf{Physical (Mean $\pm$ SD)} & \textbf{Pearson } $r$ & \textbf{Wilcoxon } $p$ \\"
    )
    print(r"\midrule")

    for m in metrics:
        display_name = metric_labels.get(m, m.replace("_", r"\_"))
        sub = piv[m].dropna()

        d_mean, d_std = sub["digital"].mean(), sub["digital"].std()
        p_mean, p_std = sub["physical"].mean(), sub["physical"].std()

        if len(sub) >= 2:
            corr, _ = scipy_stats.pearsonr(sub["digital"], sub["physical"])
            _, p_wil = scipy_stats.wilcoxon(sub["digital"], sub["physical"])

            sig = "*" if p_wil < 0.05 else ""
            p_val_str = (
                f"$<0.001^{{{sig}}}$" if p_wil < 0.001 else f"${p_wil:.4f}^{{{sig}}}$"
            )
            r_val_str = f"${corr:.3f}$"
        else:
            r_val_str, p_val_str = "--", "--"

        # Auto-adjust decimal places for small fraction/density metrics
        fmt = ".3f" if "frac" in m or "cv" in m else ".2f"
        d_str = f"${d_mean:{fmt}} \pm {d_std:{fmt}}$"
        p_str = f"${p_mean:{fmt}} \pm {p_std:{fmt}}$"

        print(f"{display_name} & {d_str} & {p_str} & {r_val_str} & {p_val_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}\n")


if __name__ == "__main__":
    args = parse_args()
    equation_ids = load_equation_ids(args.caption_path)
    transform = (
        ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
        if args.use_resized_images
        else None
    )

    records_raw, records_rs = [], []
    raw_digital_areas, raw_physical_areas = [], []
    rs_digital_areas, rs_physical_areas = [], []

    for img_id in equation_ids:
        d_raw = load_binary_array(args.digital_dir, img_id)
        p_raw = load_binary_array(args.physical_dir, img_id)
        if d_raw is None or p_raw is None:
            continue

        d_stats, d_areas = extract_all_stats(d_raw, args)
        p_stats, p_areas = extract_all_stats(p_raw, args)

        records_raw.append({"img_id": img_id, "modality": "digital", **d_stats})
        records_raw.append({"img_id": img_id, "modality": "physical", **p_stats})
        raw_digital_areas.extend(d_areas)
        raw_physical_areas.extend(p_areas)

        if args.use_resized_images:
            d_rs = resize_and_rebinarize(d_raw, transform)
            p_rs = resize_and_rebinarize(p_raw, transform)
            d_rs_stats, d_rs_areas = extract_all_stats(d_rs, args)
            p_rs_stats, p_rs_areas = extract_all_stats(p_rs, args)

            records_rs.append({"img_id": img_id, "modality": "digital", **d_rs_stats})
            records_rs.append({"img_id": img_id, "modality": "physical", **p_rs_stats})
            rs_digital_areas.extend(d_rs_areas)
            rs_physical_areas.extend(p_rs_areas)

    if not records_raw:
        print("No matched image pairs found.")
        return

    print_latex_table(pd.DataFrame(records_raw), "bmp images")
    save_combined_histogram(
        raw_digital_areas,
        raw_physical_areas,
        "bmp images",
        f"cc_size_distribution_raw_{PHYSICAL_DATA_DIR.replace('/', '_')}.png",
    )

    if args.use_resized_images:
        print_latex_table(pd.DataFrame(records_rs), "Resized Images")
        save_combined_histogram(
            rs_digital_areas,
            rs_physical_areas,
            "Resized Images",
            "cc_size_distribution_rs.png",
        )
