"""Main training script: 5-configuration ablation study with WandB logging.

Usage (from the assignment folder root):
    python -m src.train --config c1            # train + evaluate one config
    python -m src.train --plots                # aggregate plots after all runs
    python -m src.train --upload --hf-repo user/repo

The five configurations (Table 1 of the assignment): each of C2-C5 changes
exactly one component of the base model C1. All shared hyperparameters
(depth, width, LR, batching) are identical across configurations.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import (
    ASSIGNMENT_DIR,
    OUTPUTS_DIR,
    SRC_PAD,
    TGT_BOS,
    TGT_EOS,
    TGT_PAD,
    BucketBatchSampler,
    ByteCipherDataset,
    TokenizedCipherDataset,
    collate_bytes,
    collate_tokenized,
    ByteEntropyModel,
    cipher_to_bytes,
    get_splits,
    load_pairs,
    make_chunks,
    train_bpe_tokenizer,
    verify_dataset,
)
from .models import Seq2SeqTransformer
from .models.blt import BYTE_PAD, BLTModel
from .utils import (
    compute_metrics,
    greedy_decode_blt,
    greedy_decode_subword,
    load_env,
    plot_c5_vs_c1,
    plot_loss_curves,
    plot_metric_bars,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# configurations (Table 1) + shared hyperparameters
# ---------------------------------------------------------------------------

# Identical across all five configurations, as the assignment requires, so that
# each measured difference is attributable to the single ablated component.
SHARED = dict(
    d_model=256,
    n_heads=8,
    enc_layers=4,
    dec_layers=4,
    ffn_dim=1024,
    dropout=0.1,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    label_smoothing=0.1,
    grad_clip=1.0,
    # The cross-attention copy mechanism emerges late in training on this task
    # (long language-model plateau first), so the budget is generous and
    # patience high enough not to stop during the plateau.
    max_epochs=200,
    patience=10,
    token_budget=8192,  # source tokens per batch
    max_batch_size=64,
    bpe_vocab_size=8000,  # plaintext (target) BPE
    # Ciphertext (source) BPE. Kept small deliberately: the cipher alphabet has
    # only two symbols, so 1024 merges already yield ~10-bit tokens, and a
    # larger table would both inflate the source embedding well beyond the
    # other configurations and leave most of its rows undertrained on a corpus
    # of ~1.9M source tokens.
    src_bpe_vocab_size=1024,
    # BLT entropy patching (C5): an order-2 byte n-gram model supplies
    # H(next byte | context), and max_patch bounds how long a predictable run
    # may grow before a boundary is forced.
    entropy_order=2,
    entropy_min_count=4,
    # Cutoff expressed as a percentile of the training entropy distribution:
    # a new patch begins at bytes in the hardest-to-predict quartile.
    entropy_percentile=75,
    max_patch=12,
    seed=42,
)

CONFIGS = {
    "c1": dict(
        name="C1-base",
        change="none (base)",
        pos="sinusoidal",
        attn="mha",
        norm="layernorm",
        tok="subword",
    ),
    "c2": dict(
        name="C2-rope",
        change="positional encoding",
        pos="rope",
        attn="mha",
        norm="layernorm",
        tok="subword",
    ),
    "c3": dict(
        name="C3-gqa",
        change="attention mechanism",
        pos="sinusoidal",
        attn="gqa",
        n_kv_heads=2,
        norm="layernorm",
        tok="subword",
    ),
    "c4": dict(
        name="C4-rmsnorm",
        change="normalization",
        pos="sinusoidal",
        attn="mha",
        norm="rmsnorm",
        tok="subword",
    ),
    "c5": dict(
        name="C5-blt",
        change="tokenization",
        pos="sinusoidal",
        attn="mha",
        norm="layernorm",
        tok="blt",
        local_layers=2,
        local_window=128,
        patch_tgt=4,
    ),
}

CKPT_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def build_model(
    cfg: dict, hp: dict, src_vocab_size: int, tgt_vocab_size: int
) -> torch.nn.Module:
    if cfg["tok"] == "blt":
        return BLTModel(
            d_model=hp["d_model"],
            n_heads=hp["n_heads"],
            enc_layers=hp["enc_layers"],
            dec_layers=hp["dec_layers"],
            ffn_dim=hp["ffn_dim"],
            dropout=hp["dropout"],
            local_layers=cfg["local_layers"],
            local_window=cfg["local_window"],
            patch_tgt=cfg["patch_tgt"],
        )
    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_id=SRC_PAD,
        tgt_pad_id=TGT_PAD,
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        enc_layers=hp["enc_layers"],
        dec_layers=hp["dec_layers"],
        ffn_dim=hp["ffn_dim"],
        dropout=hp["dropout"],
        pos_encoding=cfg["pos"],
        attn_type=cfg["attn"],
        n_kv_heads=cfg.get("n_kv_heads"),
        norm=cfg["norm"],
    )


def build_data(cfg: dict, hp: dict):
    plain, cipher = load_pairs()
    stats = verify_dataset(plain, cipher)
    print(
        f"dataset verified: {stats['lines']} lines, alphabet {stats['alphabet_size']}, "
        f"XOR key '{stats['key']}' reproduces every line"
    )
    splits = get_splits(len(plain), seed=hp["seed"])
    chunked = {k: make_chunks(plain, cipher, idx) for k, idx in splits.items()}

    if cfg["tok"] == "blt":
        # Entropy model for BLT patch boundaries, fit on the training split only.
        ent_path = OUTPUTS_DIR / "entropy_model.json"
        if ent_path.exists():
            entropy_model = ByteEntropyModel.load_state_dict(
                json.loads(ent_path.read_text())
            )
        else:
            train_bytes = [cipher_to_bytes(c) for c in chunked["train"][1]]
            entropy_model = ByteEntropyModel(
                order=hp["entropy_order"], min_count=hp["entropy_min_count"]
            ).fit(train_bytes)
            ent_path.parent.mkdir(parents=True, exist_ok=True)
            ent_path.write_text(json.dumps(entropy_model.state_dict()))
        threshold = entropy_model.threshold_at_percentile(hp["entropy_percentile"])
        print(
            f"BLT entropy patching: threshold {threshold:.3f} bits "
            f"(p{hp['entropy_percentile']} of training distribution)"
        )
        datasets = {
            k: ByteCipherDataset(pc, cc, entropy_model, threshold, hp["max_patch"])
            for k, (pc, cc, _) in chunked.items()
        }
        collate = collate_bytes
        tokenizer = None
        src_vocab = tgt_vocab = None
        # One byte per 8 cipher bits, so the byte sequence is the same length
        # as the plaintext and needs no extra batch budget.
        budget = hp["token_budget"]
        pad_id = BYTE_PAD
    else:
        # Both tokenizers are trained on the training split only, and cached so
        # that C1-C4 share exactly the same segmentation. The source tokenizer
        # is keyed by vocab size so changing it cannot silently reuse a stale
        # cache from a previous setting.
        tokenizer = train_bpe_tokenizer(
            [plain[i] for i in splits["train"]],
            hp["bpe_vocab_size"],
            OUTPUTS_DIR / "tokenizer.json",
        )
        src_tokenizer = train_bpe_tokenizer(
            make_chunks(plain, cipher, splits["train"])[1],
            hp["src_bpe_vocab_size"],
            OUTPUTS_DIR / f"src_tokenizer_{hp['src_bpe_vocab_size']}.json",
        )
        datasets = {
            k: TokenizedCipherDataset(pc, cc, src_tokenizer, tokenizer)
            for k, (pc, cc, _) in chunked.items()
        }
        collate = collate_tokenized
        src_vocab = src_tokenizer.vocab_size
        tgt_vocab = tokenizer.vocab_size
        budget = hp["token_budget"]
        pad_id = TGT_PAD

    loaders = {}
    for split, ds in datasets.items():
        sampler = BucketBatchSampler(
            ds.src_lengths(),
            budget,
            hp["max_batch_size"],
            shuffle=(split == "train"),
            seed=hp["seed"],
        )
        # pin_memory intentionally off: it breaks CPU-only runs when CUDA is
        # merely present, and buys nothing on unified-memory devices (GB10).
        loaders[split] = DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate, num_workers=0
        )
    # per split: full reference lines + cipher (keyed by original line index)
    # and the chunk metadata needed to reassemble decoded chunks into lines
    refs = {
        k: {
            "plain": {i: plain[i] for i in idx},
            "cipher": {i: cipher[i] for i in idx},
            "meta": chunked[k][2],
        }
        for k, idx in splits.items()
    }
    return loaders, tokenizer, src_vocab, tgt_vocab, pad_id, refs


def loss_fn(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int, smoothing: float
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=smoothing,
    )


def make_scheduler(optimizer, warmup: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(model, loader, pad_id: int, device, autocast_ctx) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        extra = (batch["patch_member"].to(device),) if "patch_member" in batch else ()
        with autocast_ctx():
            logits = model(src, tgt[:, :-1], *extra)
            # smoothing 0 so validation reports true cross-entropy, comparable
            # across configs and unaffected by the training-time regularizer
            loss = loss_fn(logits, tgt[:, 1:], pad_id, 0.0)
        n = (tgt[:, 1:] != pad_id).sum().item()
        total += loss.item() * n
        count += n
    return total / max(count, 1)


@torch.no_grad()
def evaluate_test(
    model, loader, cfg, tokenizer, refs_test, device, max_examples_dump: int = 10
):
    """Greedy decode all test chunks, reassemble them into full lines, and
    compute the assignment metrics at line level."""
    meta = refs_test["meta"]
    chunk_preds: dict[int, str] = {}
    t0 = time.time()
    for batch in loader:
        src = batch["src"].to(device)
        if cfg["tok"] == "blt":
            outs = greedy_decode_blt(
                model,
                src,
                batch["patch_member"].to(device),
                max_len=src.size(1) + 8,
            )
            texts = [o.decode("ascii", errors="replace") for o in outs]
        else:
            outs = greedy_decode_subword(
                model, src, TGT_BOS, TGT_EOS, TGT_PAD, max_len=src.size(1) + 8
            )
            texts = [tokenizer.decode(ids) for ids in outs]
        for local_i, chunk_id in enumerate(batch["indices"]):
            chunk_preds[chunk_id] = texts[local_i]
    decode_seconds = time.time() - t0

    lines: dict[int, dict[int, str]] = {}
    for chunk_id, text in chunk_preds.items():
        line_idx, pos = meta[chunk_id]
        lines.setdefault(line_idx, {})[pos] = text
    order = sorted(lines.keys())
    pred_texts = ["".join(lines[i][p] for p in sorted(lines[i])) for i in order]
    ref_plain = [refs_test["plain"][i] for i in order]
    ref_cipher = [refs_test["cipher"][i] for i in order]

    metrics = compute_metrics(
        pred_texts, ref_plain, ref_cipher, tokenized=cfg["tok"] != "blt"
    )
    metrics["greedy_decode_seconds"] = decode_seconds
    examples = [
        {"reference": r, "prediction": p}
        for r, p in list(zip(ref_plain, pred_texts))[:max_examples_dump]
    ]
    return metrics, examples


def train_config(key: str, args) -> None:
    cfg = CONFIGS[key]
    hp = dict(SHARED)
    if args.epochs:
        hp["max_epochs"] = args.epochs
    if args.src_vocab:
        hp["src_bpe_vocab_size"] = args.src_vocab

    set_seed(hp["seed"])
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = device.type == "cuda"
    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_amp
        else (lambda: torch.autocast("cpu", enabled=False))
    )

    loaders, tokenizer, src_vocab, tgt_vocab, pad_id, refs = build_data(cfg, hp)
    model = build_model(cfg, hp, src_vocab, tgt_vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']}] {n_params / 1e6:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        betas=(0.9, 0.98),
    )
    total_steps = len(loaders["train"]) * hp["max_epochs"]
    scheduler = make_scheduler(optimizer, hp["warmup_steps"], total_steps)

    run = None
    if not args.no_wandb:
        import wandb

        env = load_env(ASSIGNMENT_DIR.parent / ".env")
        if env.get("WANDB_KEY"):
            wandb.login(key=env["WANDB_KEY"])
        run = wandb.init(
            project="anlp-a1",
            name=cfg["name"],
            config={**hp, **cfg, "n_params": n_params},
        )

    ckpt_dir = CKPT_DIR / key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOGS_DIR / f"{key}_history.jsonl"
    history_path.write_text("")

    def log(record: dict) -> None:
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if run is not None:
            run.log(
                {k: v for k, v in record.items() if k != "step"}, step=record["step"]
            )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val, bad_epochs, step = float("inf"), 0, 0
    epoch_times, epoch_samples, epoch_tokens = [], [], []
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        loaders["train"].batch_sampler.set_epoch(epoch)
        t_epoch = time.time()
        samples = tokens = 0
        running, running_n = 0.0, 0
        for batch in loaders["train"]:
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            extra = (
                (batch["patch_member"].to(device),) if "patch_member" in batch else ()
            )
            with autocast_ctx():
                logits = model(src, tgt[:, :-1], *extra)
                loss = loss_fn(logits, tgt[:, 1:], pad_id, hp["label_smoothing"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            samples += src.size(0)
            tokens += int((tgt[:, 1:] != pad_id).sum())
            running += loss.item()
            running_n += 1
            if step % 50 == 0:
                log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": running / running_n,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                running, running_n = 0.0, 0

        elapsed = time.time() - t_epoch
        epoch_times.append(elapsed)
        epoch_samples.append(samples)
        epoch_tokens.append(tokens)
        val_loss = validate(model, loaders["val"], pad_id, device, autocast_ctx)
        peak_gb = (
            (torch.cuda.max_memory_allocated(device) / 1e9)
            if device.type == "cuda"
            else 0.0
        )
        log(
            {
                "step": step,
                "epoch": epoch,
                "val_loss": val_loss,
                "epoch_seconds": elapsed,
                "samples_per_sec": samples / elapsed,
                "tokens_per_sec": tokens / elapsed,
                "peak_mem_gb": peak_gb,
            }
        )
        print(
            f"[{cfg['name']}] epoch {epoch}: val_loss {val_loss:.4f}, "
            f"{samples / elapsed:.1f} samples/s, peak {peak_gb:.2f} GB"
        )

        state = {
            "model": model.state_dict(),
            "config_key": key,
            "cfg": cfg,
            "hp": hp,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(state, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                print(f"[{cfg['name']}] early stopping at epoch {epoch}")
                break

    # benchmark summary (skip first epoch: warmup/compilation noise)
    bench_epochs = slice(1, None) if len(epoch_times) > 1 else slice(None)
    bench = {
        "epoch_seconds": sum(epoch_times[bench_epochs])
        / len(epoch_times[bench_epochs]),
        "samples_per_sec": sum(epoch_samples[bench_epochs])
        / sum(epoch_times[bench_epochs]),
        "tokens_per_sec": sum(epoch_tokens[bench_epochs])
        / sum(epoch_times[bench_epochs]),
        "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9)
        if device.type == "cuda"
        else 0.0,
        "n_params": n_params,
        "best_val_loss": best_val,
    }

    # test evaluation with the best checkpoint
    best = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    metrics, examples = evaluate_test(
        model, loaders["test"], cfg, tokenizer, refs["test"], device
    )
    print(
        f"[{cfg['name']}] test: "
        + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
    )

    results = {
        "config": cfg,
        "hyperparameters": hp,
        "benchmark": bench,
        "test_metrics": metrics,
    }
    save_json(results, OUTPUTS_DIR / f"results_{key}.json")
    save_json(examples, OUTPUTS_DIR / f"examples_{key}.json")

    bench_path = OUTPUTS_DIR / "benchmarks.json"
    all_bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    all_bench[cfg["name"]] = bench
    save_json(all_bench, bench_path)

    if run is not None:
        run.summary.update({f"test/{k}": v for k, v in metrics.items()})
        run.summary.update({f"bench/{k}": v for k, v in bench.items()})
        run.finish()


# ---------------------------------------------------------------------------
# aggregation & upload
# ---------------------------------------------------------------------------


def make_plots() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    histories, results = {}, {}
    for key, cfg in CONFIGS.items():
        hpath = LOGS_DIR / f"{key}_history.jsonl"
        rpath = OUTPUTS_DIR / f"results_{key}.json"
        if hpath.exists():
            histories[cfg["name"]] = [
                json.loads(l) for l in hpath.read_text().splitlines() if l
            ]
        if rpath.exists():
            results[cfg["name"]] = json.loads(rpath.read_text())["test_metrics"]
    if histories:
        plot_loss_curves(histories, PLOTS_DIR / "loss_curves.pdf")
    if results:
        plot_metric_bars(
            results,
            ["bit_accuracy", "sequence_accuracy", "levenshtein_mean"],
            PLOTS_DIR / "metrics_all.pdf",
        )
        tokenized = {k: v for k, v in results.items() if "bleu" in v}
        if tokenized:
            plot_metric_bars(
                tokenized,
                ["bleu", "rouge1", "rouge2", "rougeL"],
                PLOTS_DIR / "metrics_bleu_rouge.pdf",
            )
    bench_path = OUTPUTS_DIR / "benchmarks.json"
    if bench_path.exists():
        plot_c5_vs_c1(
            json.loads(bench_path.read_text()), PLOTS_DIR / "c5_vs_c1_benchmark.pdf"
        )
    # combined results table (markdown + csv) for the report
    if results:
        metric_names = [
            "bit_accuracy",
            "sequence_accuracy",
            "levenshtein_mean",
            "bleu",
            "rouge1",
            "rouge2",
            "rougeL",
        ]
        lines_md = [
            "| Config | " + " | ".join(metric_names) + " |",
            "|" + "---|" * (len(metric_names) + 1),
        ]
        lines_csv = ["config," + ",".join(metric_names)]
        for name, m in results.items():
            vals = [f"{m[k]:.4f}" if k in m else "-" for k in metric_names]
            lines_md.append(f"| {name} | " + " | ".join(vals) + " |")
            lines_csv.append(name + "," + ",".join(v if v != "-" else "" for v in vals))
        (OUTPUTS_DIR / "results_table.md").write_text("\n".join(lines_md) + "\n")
        (OUTPUTS_DIR / "results_table.csv").write_text("\n".join(lines_csv) + "\n")
    print(f"plots and tables written to {PLOTS_DIR} and {OUTPUTS_DIR}")


def upload_to_hf(repo_id: str) -> None:
    """Push checkpoints, tokenizer, and results to a Hugging Face model repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    for key, cfg in CONFIGS.items():
        ckpt = CKPT_DIR / key / "best.pt"
        if ckpt.exists():
            api.upload_file(
                path_or_fileobj=str(ckpt),
                repo_id=repo_id,
                path_in_repo=f"{cfg['name']}/best.pt",
            )
            print(f"uploaded {cfg['name']}/best.pt")
    for extra in [
        "tokenizer.json",
        "benchmarks.json",
        "results_table.md",
        "splits.json",
    ]:
        p = OUTPUTS_DIR / extra
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), repo_id=repo_id, path_in_repo=extra)
    print(f"https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANLP A1 ablation training")
    parser.add_argument(
        "--config", choices=list(CONFIGS), help="configuration to train"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--src-vocab",
        type=int,
        default=None,
        help="override the ciphertext BPE vocabulary size",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--plots", action="store_true", help="aggregate plots/tables")
    parser.add_argument(
        "--upload", action="store_true", help="upload checkpoints to HF"
    )
    parser.add_argument("--hf-repo", default=None)
    args = parser.parse_args()

    if args.plots:
        make_plots()
        return
    if args.upload:
        if not args.hf_repo:
            raise SystemExit("--upload requires --hf-repo <user/name>")
        upload_to_hf(args.hf_repo)
        return
    if not args.config:
        raise SystemExit("specify --config c1..c5 (or --plots / --upload)")
    train_config(args.config, args)


if __name__ == "__main__":
    main()
