# Assignment 1 — Transformers from Scratch, Architectural Variants & BLT

Advanced NLP, Monsoon 2026 — Roll number 2023111026

A full encoder-decoder Transformer built from basic PyTorch operations (no
`nn.Transformer` / `nn.MultiheadAttention`), trained to map encrypted binary
sequences to plaintext, with a five-configuration ablation study over
positional encodings (Sinusoidal vs RoPE), attention (MHA vs GQA),
normalization (LayerNorm vs RMSNorm), and tokenization (BPE subword vs
token-free BLT).

The BPE tokenizer is also implemented from scratch (`BPETokenizer` in
`src/dataset.py`), with no tokenizer library: merges are learned by frequency
over a linked-list corpus with incremental pair counts, and confined within
whitespace-delimited units. C1-C4 use a learned BPE on both sides, vocabulary
1024 over the raw `0`/`1` ciphertext and 8000 over the plaintext, each trained
on the training split only.

## Setup

Python ≥ 3.12 with:

```
pip install torch wandb sacrebleu rouge-score matplotlib numpy huggingface_hub
```

The dataset files `brown_plain.txt` / `brown_cipher.txt` are expected in
`../datasets/` relative to this folder (adjust `DATA_DIR` in `src/dataset.py`
if yours live elsewhere). Set the environment variable `WANDB_KEY` (or put it
in `../.env`) to enable Weights & Biases logging.

## Running

From this folder:

```
python -m src.train --config c1        # C1-base      (sinusoidal, MHA, LayerNorm, subword)
python -m src.train --config c2        # C2-rope      (RoPE positional encoding)
python -m src.train --config c3        # C3-gqa       (Grouped-Query Attention)
python -m src.train --config c4        # C4-rmsnorm   (RMSNorm)
python -m src.train --config c5        # C5-blt       (token-free Byte Latent Transformer)

python -m src.train --plots            # aggregate plots + results tables into outputs/
python -m src.train --upload --hf-repo <user>/<repo>   # push checkpoints to Hugging Face
```

Useful flags: `--epochs N`, `--device cpu|cuda`, `--no-wandb`.

Each run verifies the dataset structure, trains with early stopping on
validation loss, evaluates the best checkpoint on the test split with greedy
decoding (bit-level accuracy, sequence accuracy, Levenshtein, and BLEU/ROUGE
for the tokenized configs), and writes everything under `outputs/`.

Lines are split into phase-aligned 256-character windows for training and
decoding (the window is a multiple of the 8-character key period); decoded
windows are concatenated back into full lines before any metric is computed,
so all reported numbers are line-level.

## Code layout

```
src/
├── models/
│   ├── attention.py    # ScaledDotProductAttention, MHA, GQA
│   ├── positional.py   # Sinusoidal absolute PE, RoPE
│   ├── norm.py         # LayerNorm, RMSNorm (used Pre-LN)
│   ├── blt.py          # BLT local encoder/decoder, patching, full C5 model
│   └── __init__.py     # FFN, Pre-LN encoder/decoder layers, Seq2SeqTransformer
├── dataset.py          # verification, splits, BPE, tokenized & token-free loaders
├── train.py            # C1–C5 configs, training loop with WandB, eval, HF upload
└── utils.py            # metrics, greedy decoding, plots
```

## Results (test split, greedy decoding, line-level)

| Config | Bit acc. | Seq. acc. | Levenshtein | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| C1-base | 0.712 | 0.004 | 75.91 | 61.74 | 0.786 | 0.652 | 0.785 |
| C2-rope | 0.780 | 0.046 | 19.30 | **84.31** | **0.920** | **0.859** | **0.920** |
| C3-gqa | 0.704 | 0.006 | 86.24 | 59.03 | 0.770 | 0.633 | 0.768 |
| C4-rmsnorm | 0.707 | 0.000 | 80.45 | 60.30 | 0.779 | 0.642 | 0.777 |
| C5-blt | **0.990** | **0.296** | **4.02** | – | – | – | – |

BLEU/ROUGE are reported for tokenized models only, per the assignment. Full
numbers are in `outputs/results_table.csv` and `outputs/results_*.json`; training
efficiency for the C5-vs-C1 comparison is in `outputs/benchmarks.json`.

Model checkpoints are hosted on Hugging Face (link below) rather than shipped
in this archive.

## Links

- **WandB project**: https://wandb.ai/arihanttr-iiit-hyderabad/anlp-a1
- **Hugging Face checkpoints**: https://huggingface.co/Arihant25/anlp-a1-transformer-ablations
