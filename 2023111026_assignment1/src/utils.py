"""Metrics, greedy decoding, and plotting utilities.

Evaluation metrics required by the assignment (all generated with greedy decoding):
  * Bit-Level Accuracy  — % of exact bit matches between the re-encoded
    prediction and the ground-truth cipher line.
  * Sequence Accuracy   — % of perfectly reconstructed lines.
  * Levenshtein         — mean character-level edit distance (own DP implementation).
  * BLEU / ROUGE        — n-gram overlap, tokenized configurations (C1-C4) only.

Bit re-encoding uses the cipher scheme recovered from the dataset (and verified
on all 5000 lines in dataset.verify_dataset): each plaintext character c at
position i maps to the 8-bit big-endian encoding of ord(c) XOR KEY[i % 8]
with the repeating key b"ANLP2026".
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

CIPHER_KEY = b"ANLP2026"


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_env(path: str | Path) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser (no external dependency)."""
    env: dict[str, str] = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


# ---------------------------------------------------------------------------
# cipher scheme
# ---------------------------------------------------------------------------


def text_to_bits(text: str) -> str:
    """Re-encode plaintext into the cipher bit representation."""
    return "".join(f"{ord(c) ^ CIPHER_KEY[i % 8]:08b}" for i, c in enumerate(text))


def bits_to_text(bits: str) -> str:
    """Decrypt a cipher bit string (used only for dataset verification)."""
    return "".join(
        chr(int(bits[i : i + 8], 2) ^ CIPHER_KEY[(i // 8) % 8])
        for i in range(0, len(bits), 8)
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def bit_accuracy(pred_text: str, ref_bits: str) -> float:
    """Fraction of matching bits; positions beyond the shorter string count as wrong."""
    pred_bits = text_to_bits(pred_text)
    n = min(len(pred_bits), len(ref_bits))
    denom = max(len(pred_bits), len(ref_bits))
    if denom == 0:
        return 1.0
    matches = sum(1 for a, b in zip(pred_bits[:n], ref_bits[:n]) if a == b)
    return matches / denom


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance, two-row dynamic programming."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def compute_metrics(
    pred_texts: list[str],
    ref_texts: list[str],
    ref_bits: list[str],
    tokenized: bool,
) -> dict[str, float]:
    """All assignment metrics over a decoded corpus. BLEU/ROUGE only when
    `tokenized` is True (the PDF restricts them to tokenized models)."""
    assert len(pred_texts) == len(ref_texts) == len(ref_bits)
    n = len(pred_texts)
    results = {
        "bit_accuracy": sum(bit_accuracy(p, rb) for p, rb in zip(pred_texts, ref_bits))
        / n,
        "sequence_accuracy": sum(p == r for p, r in zip(pred_texts, ref_texts)) / n,
        "levenshtein_mean": sum(
            levenshtein(p, r) for p, r in zip(pred_texts, ref_texts)
        )
        / n,
    }
    if tokenized:
        import sacrebleu
        from rouge_score import rouge_scorer

        results["bleu"] = sacrebleu.corpus_bleu(pred_texts, [ref_texts]).score
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        r1 = r2 = rl = 0.0
        for p, r in zip(pred_texts, ref_texts):
            scores = scorer.score(r, p)
            r1 += scores["rouge1"].fmeasure
            r2 += scores["rouge2"].fmeasure
            rl += scores["rougeL"].fmeasure
        results["rouge1"] = r1 / n
        results["rouge2"] = r2 / n
        results["rougeL"] = rl / n
    return results


# ---------------------------------------------------------------------------
# greedy decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode_subword(
    model,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
) -> list[list[int]]:
    """Batched greedy decoding for Seq2SeqTransformer. Returns token ids
    (without BOS/EOS) per example."""
    model.eval()
    B = src.size(0)
    device = src.device
    memory, src_mask = model.encode(src)
    tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model.decode(tgt, memory, src_mask)  # (B, Lt, V)
        next_ids = logits[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
        tgt = torch.cat([tgt, next_ids[:, None]], dim=1)
        finished = finished | (next_ids == eos_id)
        if bool(finished.all()):
            break
    outputs = []
    for row in tgt[:, 1:].tolist():
        ids = []
        for t in row:
            if t in (eos_id, pad_id):
                break
            ids.append(t)
        outputs.append(ids)
    return outputs


@torch.no_grad()
def greedy_decode_blt(
    model, src_bytes: torch.Tensor, patch_member: torch.Tensor, max_len: int
) -> list[bytes]:
    """Batched greedy byte decoding for BLTModel (no KV cache; the target side
    is recomputed each step, the encoder memory is computed once)."""
    from .models.blt import BYTE_BOS, BYTE_EOS, BYTE_PAD

    model.eval()
    B = src_bytes.size(0)
    device = src_bytes.device
    memory, memory_mask = model.encode(src_bytes, patch_member)
    tgt = torch.full((B, 1), BYTE_BOS, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = model.decode(tgt, memory, memory_mask)
        next_ids = logits[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, BYTE_PAD), next_ids)
        tgt = torch.cat([tgt, next_ids[:, None]], dim=1)
        finished = finished | (next_ids == BYTE_EOS)
        if bool(finished.all()):
            break
    outputs = []
    for row in tgt[:, 1:].tolist():
        out = bytearray()
        for t in row:
            if t in (BYTE_EOS, BYTE_PAD, BYTE_BOS):
                break
            out.append(t)
        outputs.append(bytes(out))
    return outputs


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

# One colour per configuration, shared by every figure so a config keeps the
# same colour whether it appears as a line in the loss curves or as a bar in
# the metric charts. These are the default matplotlib cycle colours the loss
# curves already used.
CONFIG_COLORS = {
    "C1-base": "#1f77b4",  # blue
    "C2-rope": "#ff7f0e",  # orange
    "C3-gqa": "#2ca02c",  # green
    "C4-rmsnorm": "#d62728",  # red
    "C5-blt": "#9467bd",  # purple
}
FALLBACK_COLOR = "#888888"


def config_color(name: str) -> str:
    return CONFIG_COLORS.get(name, FALLBACK_COLOR)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_path) -> None:
    """Write the figure as vector PDF.

    Vector output keeps text and lines sharp at any zoom; the previous raster
    PNGs were visibly soft once scaled down into the report.
    """
    fig.savefig(out_path, format="pdf", bbox_inches="tight")


def plot_loss_curves(histories: dict[str, list[dict]], out_path: str | Path) -> None:
    """histories: config name -> list of {step, train_loss?, val_loss?} dicts."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, hist in histories.items():
        tr = [(h["step"], h["train_loss"]) for h in hist if "train_loss" in h]
        va = [(h["step"], h["val_loss"]) for h in hist if "val_loss" in h]
        color = config_color(name)
        if tr:
            axes[0].plot(*zip(*tr), label=name, linewidth=1.2, color=color)
        if va:
            axes[1].plot(
                *zip(*va),
                label=name,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
            )
    for ax, title in zip(axes, ["Training loss", "Validation loss"]):
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_metric_bars(
    results: dict[str, dict[str, float]], metrics: list[str], out_path: str | Path
) -> None:
    """Grouped bar chart of final test metrics per configuration."""
    plt = _mpl()
    configs = list(results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals = [results[c].get(metric, float("nan")) for c in configs]
        ax.bar(configs, vals, color=[config_color(c) for c in configs])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for i, v in enumerate(vals):
            if v == v:  # not NaN
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def plot_c5_vs_c1(bench: dict[str, dict[str, float]], out_path: str | Path) -> None:
    """C1 vs C5 benchmark: throughput, epoch time, peak GPU memory."""
    plt = _mpl()
    keys = [
        ("samples_per_sec", "Train samples/sec"),
        ("epoch_seconds", "Wall-clock / epoch (s)"),
        ("peak_mem_gb", "Peak GPU memory (GB)"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 4))
    for ax, (key, title) in zip(axes, keys):
        names = [n for n in ("C1-base", "C5-blt") if n in bench]
        vals = [bench[n].get(key, float("nan")) for n in names]
        ax.bar(names, vals, color=[config_color(n) for n in names])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)
    plt.close(fig)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
