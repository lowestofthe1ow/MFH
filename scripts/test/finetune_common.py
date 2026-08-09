"""
Common utilities for author-adaptation fine-tuning experiments.

Key changes from the original version, made to support rigorous ablations:

1. Removed the module-level `seed_everything(765)` call. Seeding is now
   explicit via `set_all_seeds(seed)`, called once per training run inside
   `adapt_author`. This keeps "which seed produced this model" traceable and
   avoids one global seed silently governing every run in a process.

2. `adapt_author` no longer re-loads and re-evaluates the base checkpoint on
   every call -- that was redundant (baseline is deterministic at eval time)
   and wasteful across a 6-size x 5-seed sweep. Evaluate baseline ONCE in the
   driver script and reuse it.

3. Real validation-based model selection: `build_trainer` now attaches a
   `ModelCheckpoint` (monitor="val_ExpRate", mode="max") and an
   `EarlyStopping` callback, and `adapt_author` reloads the *best* validation
   checkpoint before final test evaluation, instead of just taking
   whatever weights exist after a fixed epoch budget.

   ASSUMPTION: this assumes `LitCoMER.validation_step` logs a metric under
   the key "val_ExpRate" via `self.log(...)`. The provided BASE_CKPT_PATH
   filename embeds "val_ExpRate", which suggests this key exists, but you
   should confirm it against lit_comer.py. If the key is different, pass the
   correct name via `build_trainer(..., monitor="your_key")`; if it doesn't
   exist at all, this will raise a MisconfigurationException at fit-time
   rather than silently doing nothing.

4. Statistics helpers for seed aggregation and paired significance testing:
   mean_std, wilson_ci, bootstrap_ci_over_seeds, exact_mcnemar,
   combine_pvalues_fisher, paired_correctness.

5. `run_experiment` now runs multiple seeds per author and aggregates, so
   the same pattern can be reused for the other dataset categories (digital
   handwriting, non-rushed physical) beyond just the custom-author ablation.
"""

import os
import types
import math
import random
import statistics
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.text import CharErrorRate
from torch.utils.data import DataLoader

from comer.datamodule.dataset import CROHMEDataset
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from comer.datamodule.datamodule import collate_fn, build_dataset_unzipped

BASE_CKPT_PATH = (
    "lightning_logs/version_0/checkpoints/epoch=183-step=69183-val_ExpRate=0.6182.ckpt"
)
FINETUNE_CKPT_DIR = "finetune_checkpoints/author_1/phys_rush"
TRAIN_EPOCHS = 20
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADAPTATION_LR = 1e-5

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
# SPLIT_SEED fixes the train/val/test partition ONCE (see build_fixed_split
# in finetune_custom_author.py). TRAIN_SEEDS control model init / dropout /
# minibatch order for each training run and are entirely independent of the
# data split. There's no special statistical requirement on these values
# beyond being distinct from each other -- any fixed set of integers works.
# 765 is kept from the original codebase's `seed_everything(765)` call for
# continuity; the rest are arbitrary.
SPLIT_SEED = 2024
TRAIN_SEEDS = [961, 765, 100, 346, 283]


def set_all_seeds(seed):
    """
    Reseeds python `random`, numpy, and torch RNGs before a training run.
    Call this at the start of every `adapt_author` call (already done below).

    Caveat: `workers=True` helps propagate seeding to DataLoader workers, but
    with num_workers=2 and no explicit worker_init_fn, full determinism
    across dataloader worker processes isn't guaranteed. This is accepted as
    residual run-to-run noise, distinct from the seeded weight-init /
    dropout / optimizer-step variance we're actually trying to measure.
    """
    seed_everything(seed, workers=True)


class AuthorAdaptationDataModule(pl.LightningDataModule):
    """Data module for the author-specific fine-tuning stage"""

    def __init__(self, train_dir, test_dir, val_dir=None, batch_size=BATCH_SIZE):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir or test_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def _dataset(self, data_dir, is_train):
        return CROHMEDataset(
            build_dataset_unzipped(data_dir, self.batch_size),
            is_train=is_train,
            scale_aug=False,
            freq_remove_num=3,
        )

    def setup(self, stage=None):
        self.train_dataset = self._dataset(self.train_dir, True)
        self.val_dataset = self._dataset(self.val_dir, False)
        self.test_dataset = self._dataset(self.test_dir, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, num_workers=2, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, shuffle=False, num_workers=2, collate_fn=collate_fn
        )


def _edit_distance(a_tokens, b_tokens):
    """
    Token-level Levenshtein (edit) distance via standard O(n*m) DP: the
    minimum number of token insertions/deletions/substitutions to turn
    a_tokens into b_tokens. This is what the MFH paper's "<=1 error" /
    "<=2 error" tolerance metrics are built on -- ExpRate is the special
    case of "how many samples have edit distance exactly 0."
    """
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def evaluate_model(model, data_dir, return_samples=False):
    """
    Provides ExpRate, average CER, and the MFH-paper-style tolerance
    metrics "<=1 error" and "<=2 error" -- the fraction of test samples
    whose predicted token sequence is within a token-level edit distance
    of 1 (resp. 2) of the target. ExpRate is computed the same way (edit
    distance == 0) rather than via a separate string-equality check, so
    all three metrics are guaranteed internally consistent.

    NOTE on determinism for pairing (needed for McNemar's test below):
    test_dataloader uses shuffle=False, so as long as the same `data_dir` is
    used and the underlying directory listing order in build_dataset_unzipped
    is stable within a run (typical, but not guaranteed across filesystems),
    repeated calls against the same test set will yield samples in the same
    order. `paired_correctness` below asserts this rather than assuming it
    silently.

    Returns (exprate, le1_rate, le2_rate, avg_cer[, samples]).
    """

    dataset = CROHMEDataset(
        build_dataset_unzipped(data_dir, BATCH_SIZE), False, False, 3
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=2)

    model.eval()
    model.to(DEVICE)
    cer_metric = CharErrorRate().to(DEVICE)

    correct_sentences = total_sentences = batches = 0
    le1_count = le2_count = 0
    total_cer = 0.0
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            hyps = model.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)

            pred_labels = [vocab.indices2label(h.seq) for h in hyps]
            target_labels = [vocab.indices2label(i) for i in batch.indices]

            for pred, target in zip(pred_labels, target_labels):
                dist = _edit_distance(pred.split(), target.split())
                correct_sentences += int(dist == 0)
                le1_count += int(dist <= 1)
                le2_count += int(dist <= 2)
                total_sentences += 1
                if return_samples:
                    sample_cer = cer_metric([pred], [target]).item()
                    samples.append(
                        {
                            "expected": target,
                            "predicted": pred,
                            "cer": sample_cer,
                            "edit_distance": dist,
                        }
                    )

            total_cer += cer_metric(pred_labels, target_labels).item()
            batches += 1

    exprate = correct_sentences / total_sentences if total_sentences else 0
    le1_rate = le1_count / total_sentences if total_sentences else 0
    le2_rate = le2_count / total_sentences if total_sentences else 0
    avg_cer = total_cer / batches if batches else 0

    if return_samples:
        return exprate, le1_rate, le2_rate, avg_cer, samples
    return exprate, le1_rate, le2_rate, avg_cer


def load_base_model(ckpt_path):
    """Loads the .ckpt file provided by the original MFH repo"""

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint["state_dict"] = {
        k.replace("comer_model.PAM.", "comer_model.FAB."): v
        for k, v in checkpoint["state_dict"].items()
    }

    tmp_ckp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, tmp_ckp_path)
    try:
        return LitCoMER.load_from_checkpoint(tmp_ckp_path, strict=True)
    finally:
        if os.path.exists(tmp_ckp_path):
            os.remove(tmp_ckp_path)


def _bind_adaptation_optimizer(model):
    """Updates the optimizer in the model with the new one"""

    def adaptation_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=ADAPTATION_LR, weight_decay=1e-4)

    model.configure_optimizers = types.MethodType(adaptation_optimizers, model)
    return model


def build_trainer(ckpt_dir, run_tag, monitor="val_ExpRate", mode="max", patience=5):
    """
    Builds a Trainer with real validation-based model selection:
      - ModelCheckpoint saves only the best epoch (by `monitor`), not every
        epoch, and save_weights_only=True keeps this lightweight.
      - EarlyStopping halts training once `monitor` hasn't improved for
        `patience` epochs (out of the TRAIN_EPOCHS budget).

    `run_tag` must be unique per (author, train_size, seed) combination so
    that concurrent/sequential runs in an ablation sweep don't clobber each
    other's checkpoint files.
    """
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "tmp_seed_ckpts"),
        filename=run_tag + "-{epoch}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_weights_only=True,
    )
    early_stop = EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=False)

    trainer = Trainer(
        logger=False,
        callbacks=[ckpt_callback, early_stop],
        max_epochs=TRAIN_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        weights_summary=None,
        # Optional: add `deterministic=True` for stricter reproducibility.
        # Left off by default because some CoMER ops may lack a deterministic
        # CUDA kernel, which would raise at runtime -- test before enabling.
    )
    return trainer, ckpt_callback


def adapt_author(
    author,
    train_dir,
    test_dir,
    val_dir=None,
    seed=765,
    train_size=None,
    ckpt_path=BASE_CKPT_PATH,
    ckpt_dir=FINETUNE_CKPT_DIR,
):
    """
    Fine-tunes one author-adapted model from the fixed base checkpoint.

    `seed` controls model init / dropout / minibatch order ONLY -- the data
    split itself must already be fixed upstream (see build_fixed_split in
    finetune_custom_author.py). `train_size` is optional metadata folded
    into the checkpoint filename so ablation runs don't collide; pass None
    outside an ablation context (e.g. from `run_experiment`).

    Returns the fine-tuned model's test-set exprate/cer plus per-sample
    predictions (needed for McNemar's test against the baseline).
    """
    set_all_seeds(seed)

    finetune_model = load_base_model(ckpt_path)
    for param in finetune_model.parameters():
        param.requires_grad = True
    _bind_adaptation_optimizer(finetune_model)

    dm = AuthorAdaptationDataModule(train_dir, test_dir, val_dir, batch_size=BATCH_SIZE)

    run_tag = f"{author}_size{train_size}_seed{seed}" if train_size is not None else f"{author}_seed{seed}"
    trainer, ckpt_callback = build_trainer(ckpt_dir, run_tag)
    trainer.fit(finetune_model, datamodule=dm)

    best_path = ckpt_callback.best_model_path
    if best_path:
        best_state = torch.load(best_path, map_location="cpu")["state_dict"]
        finetune_model.load_state_dict(best_state)
    else:
        print(
            f"[Warning] No best checkpoint recorded for {run_tag}; falling back to "
            f"final-epoch weights. Verify that build_trainer's `monitor` matches a "
            f"key actually logged in LitCoMER.validation_step."
        )

    final_ckpt_path = os.path.join(ckpt_dir, f"finetune_model_{run_tag}.pth")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(finetune_model.state_dict(), final_ckpt_path)

    finetune_exprate, finetune_le1, finetune_le2, finetune_cer, finetune_samples = evaluate_model(
        finetune_model, test_dir, return_samples=True
    )

    # Clean up the intermediate lightning checkpoint now that weights are
    # captured in final_ckpt_path (a plain state_dict, consistent with the
    # rest of the pipeline's checkpoint format).
    if best_path and os.path.exists(best_path):
        try:
            os.remove(best_path)
        except OSError:
            pass

    return {
        "author": author,
        "train_size": train_size,
        "seed": seed,
        "finetune_exprate": finetune_exprate,
        "finetune_le1": finetune_le1,
        "finetune_le2": finetune_le2,
        "finetune_cer": finetune_cer,
        "finetune_samples": finetune_samples,
        "finetune_ckpt": final_ckpt_path,
    }


def run_experiment(
    authors,
    splits_dir,
    has_val=False,
    ckpt_path=BASE_CKPT_PATH,
    ckpt_dir=FINETUNE_CKPT_DIR,
    seeds=None,
):
    """
    Runs author adaptation for each author, across every seed in `seeds`
    (defaults to TRAIN_SEEDS), aggregating ExpRate with mean/std and a
    bootstrap CI. Reusable across dataset categories (digital handwriting,
    non-rushed physical, rushed physical) -- just point `splits_dir`/
    `authors` at the relevant category; each author's directory is expected
    to already contain a fixed train/val/test split (see
    build_fixed_split-style preparation for that category).
    """
    seeds = seeds if seeds is not None else TRAIN_SEEDS
    os.makedirs(ckpt_dir, exist_ok=True)
    results = {}

    for author in authors:
        train_dir = os.path.join(splits_dir, author, "train")
        test_dir = os.path.join(splits_dir, author, "test")
        val_dir = os.path.join(splits_dir, author, "val") if has_val else None

        if not all(
            os.path.exists(d)
            for d in (train_dir, test_dir, *([val_dir] if has_val else []))
        ):
            continue

        seed_results = {}
        exprates, le1s, le2s = [], [], []
        for seed in seeds:
            r = adapt_author(
                author, train_dir, test_dir, val_dir,
                seed=seed, ckpt_path=ckpt_path, ckpt_dir=ckpt_dir,
            )
            seed_results[seed] = r
            exprates.append(r["finetune_exprate"])
            le1s.append(r["finetune_le1"])
            le2s.append(r["finetune_le2"])

        mean_er, std_er = mean_std(exprates)
        boot_lo, boot_hi = bootstrap_ci_over_seeds(exprates)
        mean_le1, std_le1 = mean_std(le1s)
        mean_le2, std_le2 = mean_std(le2s)

        results[author] = {
            "per_seed": seed_results,
            "mean_exprate": mean_er,
            "std_exprate": std_er,
            "bootstrap_95ci": (boot_lo, boot_hi),
            "mean_le1": mean_le1,
            "std_le1": std_le1,
            "mean_le2": mean_le2,
            "std_le2": std_le2,
        }

    print("-" * 40)
    for author, r in results.items():
        print(
            f"{author}: mean ExpRate={r['mean_exprate']:.4f} std={r['std_exprate']:.4f} "
            f"95% bootstrap CI={r['bootstrap_95ci']}  "
            f"| <=1 error={r['mean_le1']:.4f} (std={r['std_le1']:.4f})  "
            f"<=2 error={r['mean_le2']:.4f} (std={r['std_le2']:.4f})"
        )
    print("-" * 40)

    return results


def print_worst_samples(samples, n=10):
    """Prints the n samples with the highest (worst) CER"""
    worst = sorted(samples, key=lambda s: s["cer"], reverse=True)[:n]
    for i, s in enumerate(worst, 1):
        edit_dist = s.get("edit_distance")
        suffix = f"  edit_distance={edit_dist}" if edit_dist is not None else ""
        print(f"#{i}  CER={s['cer']:.3f}{suffix}")
        print(f"   Expected:  {s['expected']}")
        print(f"   Predicted: {s['predicted']}")
        print("-" * 40)


# ---------------------------------------------------------------------------
# Statistics: seed aggregation, confidence intervals, McNemar's test
# ---------------------------------------------------------------------------

def mean_std(values):
    """Sample mean and sample standard deviation (ddof=1) across seed runs."""
    values = list(values)
    if len(values) < 2:
        return (values[0] if values else float("nan")), 0.0
    return statistics.mean(values), statistics.stdev(values)


def wilson_ci(k, n, z=1.96):
    """
    Wilson score interval for a binomial proportion k/n -- e.g. ExpRate from
    a single run on a test set of n items. More reliable than a normal
    approximation at small n (here n=46), which is exactly our test-set
    size. z=1.96 corresponds to a ~95% interval.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half_width = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return phat, max(0.0, center - half_width), min(1.0, center + half_width)


def bootstrap_ci_over_seeds(values, n_boot=10000, alpha=0.05, rng=None):
    """
    Percentile bootstrap CI over per-seed ExpRate values for one ablation
    condition (e.g. the 5 numbers from TRAIN_SEEDS at a given train_size).
    This captures training-stochasticity variance across seeds, distinct
    from wilson_ci which captures test-set sampling variance within a
    single run.
    """
    values = list(values)
    rng = rng or random.Random(12345)  # fixed resample seed -> reproducible reporting
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return values[0], values[0]
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return boot_means[lo_idx], boot_means[hi_idx]


def paired_correctness(base_samples, adapted_samples):
    """
    Aligns baseline and adapted-model per-item outcomes by position, with a
    hard check that the ground-truth targets match at every position --
    this is what guarantees the pairing is valid for McNemar's test. If your
    build_dataset_unzipped does not enumerate files in a guaranteed-stable
    order, add explicit sorting there; this check will catch (not silently
    ignore) any resulting mismatch.
    """
    base_targets = [s["expected"] for s in base_samples]
    adapted_targets = [s["expected"] for s in adapted_samples]
    if base_targets != adapted_targets:
        raise RuntimeError(
            "Sample order mismatch between baseline and adapted evaluation runs. "
            "McNemar's test requires per-item pairing; evaluate_model must yield "
            "test items in the same order every call. Check that both runs used "
            "the exact same test_dir and that build_dataset_unzipped enumerates "
            "files deterministically (e.g. sorted by filename)."
        )
    base_correct = [s["expected"] == s["predicted"] for s in base_samples]
    adapted_correct = [s["expected"] == s["predicted"] for s in adapted_samples]
    return base_correct, adapted_correct


def exact_mcnemar(base_correct, adapted_correct):
    """
    Exact (binomial / sign-test) McNemar's test on paired correct/incorrect
    outcomes. Appropriate here because the number of discordant pairs is
    small (test set n=46), where the chi-square approximation is unreliable.

    n01 = baseline wrong, adapted correct  (evidence FOR adaptation)
    n10 = baseline correct, adapted wrong  (evidence AGAINST adaptation)

    Returns contingency counts, the two-sided exact p-value under
    H0: n01, n10 ~ Binomial(n01+n10, 0.5), and a directional summary.
    """
    assert len(base_correct) == len(adapted_correct)
    n01 = sum(1 for b, a in zip(base_correct, adapted_correct) if (not b) and a)
    n10 = sum(1 for b, a in zip(base_correct, adapted_correct) if b and (not a))
    n = n01 + n10

    if n == 0:
        p_value = 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p_value = min(1.0, 2 * tail)

    return {
        "n01_base_wrong_adapted_correct": n01,
        "n10_base_correct_adapted_wrong": n10,
        "n_discordant": n,
        "p_value": p_value,
        "direction": "adapted better" if n01 > n10 else ("baseline better" if n10 > n01 else "tied"),
    }


def combine_pvalues_fisher(p_values):
    """
    Fisher's method to combine p-values from the per-seed McNemar tests.
    Valid here because each seed is an independently initialized/trained
    model evaluated on the same fixed test set -- these are independent
    experiments testing the same directional hypothesis (does adaptation
    change per-item correctness relative to baseline).
    """
    p_values = [p if p > 0 else 1e-300 for p in p_values]  # guard against log(0)
    stat = -2 * sum(math.log(p) for p in p_values)
    k = len(p_values)
    p_combined = 1 - _chi2_cdf(stat, df=2 * k)
    return stat, p_combined


def _chi2_cdf(x, df):
    """Chi-square CDF via the regularized lower incomplete gamma function."""
    return _lower_incomplete_gamma_reg(df / 2, x / 2)


def _lower_incomplete_gamma_reg(s, x, iterations=200):
    """Regularized lower incomplete gamma function P(s, x), series expansion."""
    if x < 0 or s <= 0:
        return float("nan")
    if x == 0:
        return 0.0
    term = 1.0 / s
    total = term
    for n in range(1, iterations):
        term *= x / (s + n)
        total += term
        if abs(term) < 1e-14 * abs(total):
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))