import math
from pathlib import Path

import numpy as np
import scipy.stats as stats

FAB_PATH = "comer_model.FAB"

PATCH_SIZE = 8
RETAIN_M = 5

def get_submodule(model, dotted_path):
    obj = model
    parts = dotted_path.split(".")
    for i, p in enumerate(parts):
        if not hasattr(obj, p):
            raise AttributeError(
                f"'{'.'.join(parts[:i]) or '<model>'}' has no attribute '{p}' while "
                f"resolving '{dotted_path}'."
            )
        obj = getattr(obj, p)
    return obj

def safe_run(fn, *args, **kwargs):
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def load_sample_batch(test_dir, n=10):
    from comer.datamodule.dataset import CROHMEDataset
    from comer.datamodule.datamodule import collate_fn, build_dataset_unzipped
    from torch.utils.data import DataLoader

    dataset = CROHMEDataset(build_dataset_unzipped(test_dir, n), False, False, 3)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {test_dir}")
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)
    return next(iter(loader))

# ---------------------------------------------------------------------------
# Analysis 1: FAB channel-attention weights & Statistical Tests
# ---------------------------------------------------------------------------

def find_fab_attention_conv(fab_module):
    import torch.nn as nn
    candidates = [
        (name, m) for name, m in fab_module.named_modules()
        if isinstance(m, nn.Conv2d) and m.out_channels == 2
    ]
    if len(candidates) == 0:
        raise RuntimeError("No Conv2d with out_channels=2 found inside FAB.")
    if len(candidates) > 1:
        names = ", ".join(n for n, _ in candidates)
        raise RuntimeError(f"Ambiguous Conv2d candidates found: {names}")
    return candidates[0]

def compute_fab_attention_maps(model, batch):
    import torch
    fab = get_submodule(model, FAB_PATH)
    conv_name, conv_module = find_fab_attention_conv(fab)
    captured = {}

    def _hook(module, inp, out):
        captured["A"] = out.detach()

    handle = conv_module.register_forward_hook(_hook)
    model.eval()
    try:
        with torch.no_grad():
            model.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)
    finally:
        handle.remove()

    if "A" not in captured:
        raise RuntimeError("Forward hook never fired.")
    
    A = captured["A"]
    return conv_name, A

def extract_v_params(model):
    """Extracts the 256-dimensional v_x and v_y scaling vectors from FAB."""
    fab = get_submodule(model, FAB_PATH)
    v_x = fab.v_x.detach().cpu().flatten().numpy()
    v_y = fab.v_y.detach().cpu().flatten().numpy()
    return v_x, v_y

def run_paired_tests(baseline_vals, adapted_vals):
    """Runs a paired t-test and Wilcoxon signed-rank test on 1D arrays."""
    delta = np.array(adapted_vals) - np.array(baseline_vals)
    mean_delta = delta.mean()
    
    # Paired t-test
    if np.all(delta == 0):
        return {"mean_delta": mean_delta, "t_stat": 0.0, "p_t": 1.0, "w_stat": 0.0, "p_w": 1.0}

    t_stat, p_t = stats.ttest_rel(adapted_vals, baseline_vals)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_w = stats.wilcoxon(adapted_vals, baseline_vals)
    except ValueError:
        # Happens if all differences are zero (handled above) or N is too small
        w_stat, p_w = float("nan"), float("nan")
        
    return {
        "mean_delta": float(mean_delta),
        "t_stat": float(t_stat),
        "p_t": float(p_t),
        "w_stat": float(w_stat),
        "p_w": float(p_w)
    }

def summarize_attention_map(A):
    """
    Applies the confirmed Sigmoid activation. 
    Channel 0 = spatial gate (T), Channel 1 = frequency gate (K).
    """
    import torch
    
    A_sig = torch.sigmoid(A.detach().float())
    spatial_gate = A_sig[:, 0]
    freq_gate = A_sig[:, 1]
    
    return {
        "spatial_gate_mean": spatial_gate.mean().item(),
        "frequency_gate_mean": freq_gate.mean().item(),
        "mean_map_spatial": spatial_gate.mean(dim=0).tolist(),
        "mean_map_frequency": freq_gate.mean(dim=0).tolist(),
        # For paired statistical tests across images
        "per_image_spatial_mean": spatial_gate.mean(dim=(1, 2)).tolist(),
        "per_image_frequency_mean": freq_gate.mean(dim=(1, 2)).tolist(),
    }

def compare_fab_attention(baseline_model, adapted_model, batch):
    base_conv_name, base_A = compute_fab_attention_maps(baseline_model, batch)
    adapt_conv_name, adapt_A = compute_fab_attention_maps(adapted_model, batch)
    
    base_stats = summarize_attention_map(base_A)
    adapt_stats = summarize_attention_map(adapt_A)
    
    base_vx, base_vy = extract_v_params(baseline_model)
    adapt_vx, adapt_vy = extract_v_params(adapted_model)
    
    return {
        "baseline": {"conv_module": base_conv_name, **base_stats},
        "adapted": {"conv_module": adapt_conv_name, **adapt_stats},
        "n_samples": int(base_A.shape[0]),
        "stats": {
            "spatial_gate_per_image": run_paired_tests(base_stats["per_image_spatial_mean"], adapt_stats["per_image_spatial_mean"]),
            "frequency_gate_per_image": run_paired_tests(base_stats["per_image_frequency_mean"], adapt_stats["per_image_frequency_mean"]),
            "v_x_channels": run_paired_tests(base_vx, adapt_vx),
            "v_y_channels": run_paired_tests(base_vy, adapt_vy),
        }
    }

# ---------------------------------------------------------------------------
# Analysis 2: input spectrum characterization by category (model-free)
# ---------------------------------------------------------------------------

def _dct2_patch(patch):
    n = patch.shape[0]
    x = np.arange(n)
    k = np.arange(n).reshape(-1, 1)
    C = np.ones(n)
    C[0] = 1.0 / math.sqrt(2)
    basis = np.cos((2 * x + 1)[None, :] * k * math.pi / (2 * n))
    B = math.sqrt(2.0 / n) * C.reshape(-1, 1) * basis
    return B @ patch.astype(np.float64) @ B.T

def compute_patch_dct_energy_map(image, patch_size=PATCH_SIZE):
    h, w = image.shape
    pad_h, pad_w = (-h) % patch_size, (-w) % patch_size
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    H, W = padded.shape

    acc = np.zeros((patch_size, patch_size), dtype=np.float64)
    n_patches = 0
    for pr in range(0, H, patch_size):
        for pc in range(0, W, patch_size):
            patch = padded[pr:pr + patch_size, pc:pc + patch_size]
            acc += np.abs(_dct2_patch(patch))
            n_patches += 1
    return acc / max(n_patches, 1)

def characterize_input_spectra(image_paths_by_category, patch_size=PATCH_SIZE, retain_m=RETAIN_M):
    from PIL import Image
    results = {}
    for category, paths in image_paths_by_category.items():
        if not paths:
            continue
        total_map = np.zeros((patch_size, patch_size), dtype=np.float64)
        used = 0
        for p in paths:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("L"), dtype=np.float64)
            total_map += compute_patch_dct_energy_map(arr, patch_size)
            used += 1
        avg_map = total_map / used

        lo = patch_size - retain_m
        retained_energy = avg_map[lo:, lo:].sum()
        total_energy = avg_map.sum()
        results[category] = {
            "n_images": used,
            "energy_map": avg_map.tolist(),
            "retained_energy_frac": float(retained_energy / total_energy) if total_energy > 0 else float("nan"),
        }
    return results