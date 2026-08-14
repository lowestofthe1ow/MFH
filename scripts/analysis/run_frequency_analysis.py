import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

from scripts.analysis.freq_analysis import load_sample_batch, compare_fab_attention, characterize_input_spectra, safe_run
from scripts.test.build_report import discover_results, CATEGORY_LABELS


def default_test_dir(author, category, splits_template):
    return splits_template.format(author=author, category=category)


def collect_category_image_paths(entries, splits_template, exts=(".bmp",)):
    out = defaultdict(list)
    for author, category, _ in entries:
        d = Path(default_test_dir(author, category, splits_template))
        if not d.exists():
            print(f"[warn] test dir not found for {author}/{category}: {d}")
            continue
        label = CATEGORY_LABELS.get(category, category)
        for ext in exts:
            out[label].extend(sorted(str(p) for p in d.glob(f"*{ext}")))
    return dict(out)


def parse_args():
    p = argparse.ArgumentParser(description="FAB attention + input-spectra frequency-domain analysis")
    p.add_argument("--checkpoints-root", default="finetune_checkpoints")
    p.add_argument("--base-ckpt", help="Path to the original MFH-CoMER base checkpoint",
                   default="lightning_logs/version_0/checkpoints/epoch=183-step=69183-val_ExpRate=0.6182.ckpt")
    p.add_argument(
        "--splits-template",
        default="data/custom/{author}/{category}/splits/custom_author/test",
        help="Format string for each category's test-split directory ({author}/{category} "
             "are substituted with the folder names found under --checkpoints-root). Default "
             "matches the layout in the finetune_custom_author.py you shared for "
             "author_1/phys_rush -- override if your other categories use a different layout.",
    )
    p.add_argument("--n-samples", type=int, default=10,
                    help="Number of test images to load per category for the FAB attention hook")
    p.add_argument("--out", default="freq_analysis_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    from scripts.test.finetune_common import load_base_model
    import torch

    entries = discover_results(Path(args.checkpoints_root))
    if not entries:
        raise FileNotFoundError(f"No ablation_results.json files found under {args.checkpoints_root}")

    print(f"Loading base checkpoint: {args.base_ckpt}")
    baseline_model = load_base_model(args.base_ckpt)

    results = {}
    for author, category, json_path in entries:
        label = f"{author}/{category}"
        print(f"\n=== {label} ===")
        with open(json_path) as f:
            ablation = json.load(f)

        rep_seed = ablation.get("representative_seed_at_final_size")
        if rep_seed is None:
            print(f"  [skip] no representative_seed_at_final_size in {json_path} "
                  f"(regenerate ablation_results.json with the current finetune_custom_author.py)")
            continue
        final_size = max(int(s) for s in ablation["ablation"].keys())
        entry = ablation["ablation"].get(str(final_size), {}).get(str(rep_seed))
        if entry is None or "ckpt" not in entry:
            print(f"  [skip] no checkpoint recorded for size={final_size} seed={rep_seed}")
            continue

        ckpt_path = entry["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"  [skip] checkpoint file not found on disk: {ckpt_path}")
            continue

        print(f"  Adapted checkpoint: {ckpt_path}  (representative seed={rep_seed}, train_size={final_size})")
        adapted_model = load_base_model(args.base_ckpt)
        adapted_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        test_dir = default_test_dir(author, category, args.splits_template)
        if not os.path.exists(test_dir):
            print(f"  [skip] test dir not found: {test_dir} (check --splits-template)")
            continue

        batch_result = safe_run(load_sample_batch, test_dir, n=args.n_samples)
        fab_result = safe_run(compare_fab_attention, baseline_model, adapted_model, batch_result["result"])
        results[label] = {"fab_attention": fab_result}
        
        if fab_result["ok"]:
            s = fab_result["result"]["stats"]
            print("\n    [Stats] Spatial Gate (per image):  p_t={:.4f}, p_w={:.4f}".format(s['spatial_gate_per_image']['p_t'], s['spatial_gate_per_image']['p_w']))
            print("    [Stats] Frequency Gate (per image): p_t={:.4f}, p_w={:.4f}".format(s['frequency_gate_per_image']['p_t'], s['frequency_gate_per_image']['p_w']))
            print("    [Stats] v_x parameter (per channel): p_t={:.4g}, p_w={:.4g}".format(s['v_x_channels']['p_t'], s['v_x_channels']['p_w']))
            print("    [Stats] v_y parameter (per channel): p_t={:.4g}, p_w={:.4g}".format(s['v_y_channels']['p_t'], s['v_y_channels']['p_w']))

        if not batch_result["ok"]:
            print(f"  [skip] couldn't load sample batch: {batch_result['error']}")
            continue
        batch = batch_result["result"]

        fab_result = safe_run(compare_fab_attention, baseline_model, adapted_model, batch)
        if fab_result["ok"]:
            print("  FAB attention comparison: OK")
        else:
            print(f"  FAB attention comparison FAILED: {fab_result['error']}")

        results[label] = {"fab_attention": fab_result}

    print("\n=== Input spectrum characterization (all categories) ===")
    image_paths = collect_category_image_paths(entries, args.splits_template)
    for label, paths in image_paths.items():
        print(f"  {label}: {len(paths)} test images")
    spectra_result = safe_run(characterize_input_spectra, image_paths)
    results["input_spectra"] = spectra_result
    if spectra_result["ok"]:
        for label, r in spectra_result["result"].items():
            print(f"    {label}: retained_energy_frac={r['retained_energy_frac']:.4f} (n={r['n_images']})")
    else:
        print(f"  FAILED: {spectra_result['error']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
