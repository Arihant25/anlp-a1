"""Data pipeline: verification, splits, chunking, BPE tokenizer, and the two
loader families — tokenized (C1-C4) and token-free byte-level (C5 / BLT).

Source representation (tokenized configs): a learned byte-level BPE trained on
the training-split ciphertext. Its merges are built over an alphabet of two
symbols ('0' and '1') and are free to cross the 8-bit character boundaries, so
a source token covers a variable number of plaintext characters. Target: a
second learned BPE over the plaintext.

Token-free (C5): the cipher bit string is grouped 8 bits at a time into byte
values 0-255 (the underlying binary data, not its text encoding), and the
target is the raw plaintext bytes - no vocabulary beyond 256 byte values +
PAD/BOS/EOS specials (see models/blt.py). Patch boundaries over those bytes
come from ByteEntropyModel rather than a fixed stride.

Chunking: lines (up to 2670 chars) are split into CHUNK_CHARS-character
windows for training and decoding. Empirically the cross-attention copy
mechanism takes far longer to emerge on full-length lines; shorter windows
multiply the number of training pairs and shrink the alignment range the
model must learn. CHUNK_CHARS is a multiple of 8, so every chunk starts at
XOR-key phase 0 and the per-position cipher mapping is identical across
chunks. Evaluation decodes chunks independently and concatenates them back
into full lines, so all reported metrics remain line-level.
"""

import json
import math
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .models.blt import BYTE_BOS, BYTE_EOS, BYTE_PAD
from .utils import CIPHER_KEY, bits_to_text

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ASSIGNMENT_DIR.parent / "datasets"
OUTPUTS_DIR = ASSIGNMENT_DIR / "outputs"

# Both sides use a learned byte-level BPE trained with the same special
# tokens, so the pad id is shared. The source vocabulary size is a property of
# the trained ciphertext tokenizer and is read off it at build time.
SRC_PAD = 0

CHUNK_CHARS = 256  # multiple of 8: every chunk starts at key phase 0


def make_chunks(
    plain: list[str],
    cipher: list[str],
    indices: list[int],
    chunk_chars: int = CHUNK_CHARS,
) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    """Split the selected lines into phase-aligned windows.

    Returns (plain_chunks, cipher_chunks, meta) with meta[k] = (line_index,
    position) so evaluation can stitch decoded chunks back into full lines.
    """
    plain_chunks, cipher_chunks, meta = [], [], []
    for i in indices:
        p, c = plain[i], cipher[i]
        for pos, start in enumerate(range(0, len(p), chunk_chars)):
            plain_chunks.append(p[start : start + chunk_chars])
            cipher_chunks.append(c[8 * start : 8 * (start + chunk_chars)])
            meta.append((i, pos))
    return plain_chunks, cipher_chunks, meta


# BPE special-token ids (fixed by training order in train_bpe_tokenizer)
TGT_PAD, TGT_UNK, TGT_BOS, TGT_EOS = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# loading & verification
# ---------------------------------------------------------------------------


def load_pairs(data_dir: Path = DATA_DIR) -> tuple[list[str], list[str]]:
    plain = (data_dir / "brown_plain.txt").read_text(encoding="utf-8").splitlines()
    cipher = (data_dir / "brown_cipher.txt").read_text(encoding="ascii").splitlines()
    return plain, cipher


def verify_dataset(plain: list[str], cipher: list[str]) -> dict:
    """Fail loudly if any structural assumption about the dataset is violated.

    Checks 1:1 alignment, the binary alphabet, the 8-bits-per-character ratio,
    and that the recovered repeating-key XOR scheme (KEY = b"ANLP2026") decrypts
    every line exactly — this last check justifies using the scheme for the
    bit-level accuracy metric.
    """
    assert len(plain) == len(cipher), f"line mismatch: {len(plain)} vs {len(cipher)}"
    alphabet: set[str] = set()
    for i, (p, c) in enumerate(zip(plain, cipher)):
        assert len(c) == 8 * len(p), f"line {i}: len(cipher) != 8*len(plain)"
        assert set(c) <= {"0", "1"}, f"line {i}: non-binary cipher symbol"
        assert bits_to_text(c) == p, (
            f"line {i}: XOR scheme does not reproduce plaintext"
        )
        alphabet.update(p)
    return {
        "lines": len(plain),
        "alphabet_size": len(alphabet),
        "alphabet": "".join(sorted(alphabet)),
        "key": CIPHER_KEY.decode(),
        "max_plain_len": max(len(p) for p in plain),
    }


def get_splits(
    n: int,
    seed: int = 42,
    fractions: tuple[float, float] = (0.8, 0.1),
    path: Path = OUTPUTS_DIR / "splits.json",
) -> dict[str, list[int]]:
    """Deterministic 80/10/10 split, persisted so every configuration reuses it."""
    if path.exists():
        return json.loads(path.read_text())
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    splits = {
        "train": sorted(indices[:n_train]),
        "val": sorted(indices[n_train : n_train + n_val]),
        "test": sorted(indices[n_train + n_val :]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))
    return splits


# ---------------------------------------------------------------------------
# BPE tokenizers (C1-C4: one for the ciphertext, one for the plaintext)
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Byte-pair encoding implemented from scratch, with no tokenizer library.

    Training follows Sennrich et al. (2016), 'Neural Machine Translation of Rare
    Words with Subword Units' (ACL): start from the character alphabet
    of the corpus and repeatedly merge the most frequent adjacent symbol pair
    until the vocabulary reaches the requested size.

    Merges are confined within whitespace-delimited words, as in the original
    algorithm. Without that restriction BPE happily merges across spaces and
    learns multi-word phrases ("Church of Scientology ") as single tokens: they
    memorise particular training strings, occur a handful of times, and are
    useless at test time. The ciphertext contains no whitespace, so the same
    pre-tokenizer leaves it as one unit and merges span the whole string.

    The corpus is held as a doubly linked list over one flat symbol array, and
    pair counts are updated only around each merge rather than recounted from
    scratch. That matters here because the ciphertext contains no whitespace,
    so the usual trick of collapsing a corpus into unique space-separated words
    buys nothing and a naive implementation would cost O(corpus x merges).
    A lazy heap keeps the "most frequent pair" query off the critical path:
    stale entries are discarded on pop instead of being deleted eagerly.
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, vocab: list[str], merges: list[tuple[int, int, int]]):
        self.vocab = vocab
        self.merges = merges
        self._rank = {(a, b): r for r, (a, b, _) in enumerate(merges)}
        self._new = {(a, b): n for a, b, n in merges}
        self._sym = {t: i for i, t in enumerate(vocab)}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        """Split into merge units: each leading run of spaces stays attached to
        the word that follows, so joining the pieces reproduces the input
        exactly. Text without whitespace comes back as a single unit."""
        return re.findall(r"\s*\S+|\s+", text)

    # -- training ----------------------------------------------------------

    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "BPETokenizer":
        import heapq
        from collections import Counter, defaultdict

        vocab = list(cls.SPECIALS)
        vocab.extend(sorted({ch for t in texts for ch in t}))
        sym = {t: i for i, t in enumerate(vocab)}

        units = [w for t in texts for w in cls.pre_tokenize(t)]
        syms: list[int] = []
        prv: list[int] = []
        nxt: list[int] = []
        for t in units:
            if not t:
                continue
            start = len(syms)
            for ch in t:
                syms.append(sym[ch])
                prv.append(len(syms) - 2)
                nxt.append(len(syms))
            prv[start] = -1
            nxt[-1] = -1

        counts: Counter = Counter()
        where: defaultdict = defaultdict(set)
        for i, j in enumerate(nxt):
            if j != -1:
                pair = (syms[i], syms[j])
                counts[pair] += 1
                where[pair].add(i)

        heap = [(-c, p) for p, c in counts.items()]
        heapq.heapify(heap)
        alive = [True] * len(syms)
        merges: list[tuple[int, int, int]] = []

        def bump(pair, delta, pos, add):
            counts[pair] += delta
            if add:
                where[pair].add(pos)
            else:
                where[pair].discard(pos)
            if counts[pair] > 0:
                heapq.heappush(heap, (-counts[pair], pair))

        while len(vocab) < vocab_size and heap:
            neg, pair = heapq.heappop(heap)
            if counts.get(pair, 0) != -neg or counts.get(pair, 0) < 2:
                continue  # stale heap entry
            new_id = len(vocab)
            vocab.append(vocab[pair[0]] + vocab[pair[1]])
            merges.append((pair[0], pair[1], new_id))

            for i in sorted(where[pair]):
                if not alive[i]:
                    continue
                j = nxt[i]
                if j == -1 or not alive[j] or (syms[i], syms[j]) != pair:
                    continue
                p, k = prv[i], nxt[j]
                if p != -1:
                    bump((syms[p], syms[i]), -1, p, add=False)
                if k != -1:
                    bump((syms[j], syms[k]), -1, j, add=False)
                syms[i] = new_id
                alive[j] = False
                nxt[i] = k
                if k != -1:
                    prv[k] = i
                if p != -1:
                    bump((syms[p], new_id), 1, p, add=True)
                if k != -1:
                    bump((new_id, syms[k]), 1, i, add=True)

            counts.pop(pair, None)
            where.pop(pair, None)

        return cls(vocab, merges)

    # -- inference ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string: pre-tokenize, merge within each unit, concatenate.

        Units repeat constantly in natural text, so results are memoised; this
        is what keeps encoding a large corpus affordable in pure Python.
        """
        out: list[int] = []
        for unit in self.pre_tokenize(text):
            cached = self._cache.get(unit)
            if cached is None:
                cached = self._encode_unit(unit)
                self._cache[unit] = cached
            out.extend(cached)
        return out

    def _encode_unit(self, text: str) -> list[int]:
        """Greedily apply the learned merges in rank order to one unit."""
        import heapq

        ids = [self._sym.get(ch, self.UNK) for ch in text]
        n = len(ids)
        if n < 2:
            return ids
        prv = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        alive = [True] * n

        heap = []
        for i in range(n - 1):
            r = self._rank.get((ids[i], ids[i + 1]))
            if r is not None:
                heap.append((r, i))
        heapq.heapify(heap)

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j] or self._rank.get((ids[i], ids[j])) != r:
                continue
            ids[i] = self._new[(ids[i], ids[j])]
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prv[k] = i
            p = prv[i]
            if p != -1:
                rr = self._rank.get((ids[p], ids[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p))
            if k != -1:
                rr = self._rank.get((ids[i], ids[k]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i))

        return [ids[i] for i in range(n) if alive[i]]

    def decode(self, ids: list[int]) -> str:
        """Ids back to text; special tokens carry no surface form and are dropped."""
        return "".join(self.vocab[i] for i in ids if i >= len(self.SPECIALS))

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "merges": [list(m) for m in self.merges]})
        )

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])


def train_bpe_tokenizer(
    train_texts: list[str], vocab_size: int, path: Path
) -> BPETokenizer:
    """Train (or reload) a from-scratch BPE tokenizer, cached at `path`.

    Used for both sides of the tokenized configurations: once on the plaintext
    and once on the raw '0'/'1' ciphertext. The ciphertext case starts from an
    alphabet of only two symbols, so its merges build multi-bit patterns that
    straddle the 8-bit character boundaries of the cipher.
    """
    if Path(path).exists():
        return BPETokenizer.load(path)
    tokenizer = BPETokenizer.train(train_texts, vocab_size)
    tokenizer.save(path)
    return tokenizer


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


class TokenizedCipherDataset(Dataset):
    """C1-C4: source = learned BPE over the ciphertext, target = learned BPE
    over the plaintext (BOS ... EOS).

    Operates on parallel chunk lists (see make_chunks); "index" identifies the
    chunk so evaluation can reassemble lines.
    """

    def __init__(
        self,
        plain_chunks: list[str],
        cipher_chunks: list[str],
        src_tokenizer,
        tgt_tokenizer,
    ):
        self.items = []
        for i, (p, c) in enumerate(zip(plain_chunks, cipher_chunks)):
            src = src_tokenizer.encode(c)
            tgt = [TGT_BOS] + tgt_tokenizer.encode(p) + [TGT_EOS]
            self.items.append((src, tgt, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        src, tgt, orig = self.items[idx]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "index": orig,
        }

    def src_lengths(self) -> list[int]:
        return [len(s) for s, _, _ in self.items]


class ByteEntropyModel:
    """Order-k byte n-gram model used to place BLT patch boundaries.

    The BLT paper (Pagnoni et al., 2024) segments a byte stream into patches by
    estimating the entropy of the next byte under a small language model and
    starting a new patch wherever the stream becomes hard to predict. A full
    byte-level LM is unnecessary at this scale: a smoothed count-based Markov
    model over the training-split bytes gives the same signal in seconds and
    with no GPU.

    H(next | context) is computed from the conditional distribution of the
    following byte given the previous `order` bytes, backing off to shorter
    contexts (and finally to the unigram distribution) whenever a context was
    not seen often enough during training.
    """

    def __init__(self, order: int = 2, min_count: int = 4):
        self.order = order
        self.min_count = min_count
        self.tables: list[dict] = []
        self.unigram_entropy = 0.0

    def fit(self, sequences: list[list[int]]) -> "ByteEntropyModel":
        from collections import defaultdict

        self.tables = [
            defaultdict(lambda: defaultdict(int)) for _ in range(self.order + 1)
        ]
        for seq in sequences:
            for i, nxt in enumerate(seq):
                for k in range(self.order + 1):
                    if i - k < 0:
                        break
                    ctx = tuple(seq[i - k : i])
                    self.tables[k][ctx][nxt] += 1
        self.entropies = []
        for k, table in enumerate(self.tables):
            ent = {}
            for ctx, counts in table.items():
                total = sum(counts.values())
                if total >= self.min_count:
                    ent[ctx] = -sum(
                        (c / total) * math.log2(c / total) for c in counts.values()
                    )
            self.entropies.append(ent)
        self.unigram_entropy = self.entropies[0].get((), 8.0)
        self.tables = []  # counts no longer needed; keep only entropies
        # Record the distribution of H over the training bytes so the patch
        # threshold can be set as a percentile of it. An absolute figure in bits
        # is meaningless without knowing the spread: on this ciphertext H tops
        # out around 6 bits, so a "reasonable-looking" cutoff of 7 would never
        # fire and every patch would silently fall back to the length cap.
        sample = [h for seq in sequences[:400] for h in self.entropy_curve(seq)]
        sample.sort()
        self.curve_quantiles = (
            [
                sample[min(len(sample) - 1, int(len(sample) * q / 100))]
                for q in range(101)
            ]
            if sample
            else [0.0] * 101
        )
        return self

    def threshold_at_percentile(self, pct: float) -> float:
        """Absolute entropy cutoff sitting at `pct` of the training distribution."""
        return self.curve_quantiles[max(0, min(100, int(round(pct))))]

    def entropy_curve(self, seq: list[int]) -> list[float]:
        """H(byte i | preceding bytes) for every position of `seq`."""
        out = []
        for i in range(len(seq)):
            h = None
            for k in range(min(self.order, i), -1, -1):
                h = self.entropies[k].get(tuple(seq[i - k : i]))
                if h is not None:
                    break
            out.append(self.unigram_entropy if h is None else h)
        return out

    def patch_boundaries(
        self, seq: list[int], threshold: float, max_patch: int = 16
    ) -> list[int]:
        """Start offsets of each patch.

        A new patch begins wherever the next byte is hard to predict, which is
        the paper's global-threshold rule. `max_patch` caps how long a run of
        predictable bytes may grow, which keeps a single patch from swallowing
        an entire sequence and bounds the pooling cost.
        """
        curve = self.entropy_curve(seq)
        starts, run = [0], 1
        for i in range(1, len(seq)):
            if curve[i] > threshold or run >= max_patch:
                starts.append(i)
                run = 1
            else:
                run += 1
        return starts

    def state_dict(self) -> dict:
        return {
            "order": self.order,
            "min_count": self.min_count,
            "unigram_entropy": self.unigram_entropy,
            "curve_quantiles": self.curve_quantiles,
            "entropies": [
                {" ".join(map(str, ctx)): h for ctx, h in tab.items()}
                for tab in self.entropies
            ],
        }

    @classmethod
    def load_state_dict(cls, d: dict) -> "ByteEntropyModel":
        m = cls(d["order"], d["min_count"])
        m.unigram_entropy = d["unigram_entropy"]
        m.curve_quantiles = d["curve_quantiles"]
        m.entropies = [
            {tuple(int(x) for x in k.split()) if k else (): v for k, v in tab.items()}
            for tab in d["entropies"]
        ]
        return m


def cipher_to_bytes(bits: str) -> list[int]:
    """Group the cipher bit string into byte values 0-255.

    The ciphertext is a binary sequence written out as text, so the underlying
    data is one byte per 8 characters. Feeding the ASCII codes of the '0'/'1'
    characters instead would model the text encoding rather than the data.
    """
    return [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)]


class ByteCipherDataset(Dataset):
    """C5 / BLT: source = cipher byte values (8 bits -> one byte), target =
    BOS + plaintext bytes + EOS. Patch boundaries come from the entropy model."""

    def __init__(
        self,
        plain_chunks: list[str],
        cipher_chunks: list[str],
        entropy_model=None,
        threshold: float = 0.0,
        max_patch: int = 16,
    ):
        self.items = []
        for i, (p, c) in enumerate(zip(plain_chunks, cipher_chunks)):
            src = cipher_to_bytes(c)
            tgt = [BYTE_BOS] + list(p.encode("ascii")) + [BYTE_EOS]
            starts = (
                entropy_model.patch_boundaries(src, threshold, max_patch)
                if entropy_model is not None
                else list(range(len(src)))
            )
            self.items.append((src, tgt, starts, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        src, tgt, starts, orig = self.items[idx]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "patch_starts": starts,
            "index": orig,
        }

    def src_lengths(self) -> list[int]:
        return [len(s) for s, _, _, _ in self.items]


def make_collate(src_pad: int, tgt_pad: int):
    def collate(batch: list[dict]):
        src_len = max(len(b["src"]) for b in batch)
        tgt_len = max(len(b["tgt"]) for b in batch)
        src = torch.full((len(batch), src_len), src_pad, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_len), tgt_pad, dtype=torch.long)
        for i, b in enumerate(batch):
            src[i, : len(b["src"])] = b["src"]
            tgt[i, : len(b["tgt"])] = b["tgt"]
        out = {"src": src, "tgt": tgt, "indices": [b["index"] for b in batch]}

        if "patch_starts" in batch[0]:
            # Patches are entropy-segmented, so each sequence has a different
            # number of them and each patch a different length. They are passed
            # to the model as a (B, K, L) boolean membership mask: entry
            # [b, k, i] is True when source byte i belongs to patch k. This keeps
            # the model free of any fixed patch-size assumption.
            n_patches = max(len(b["patch_starts"]) for b in batch)
            member = torch.zeros(len(batch), n_patches, src_len, dtype=torch.bool)
            for i, b in enumerate(batch):
                starts = b["patch_starts"]
                ends = starts[1:] + [len(b["src"])]
                for k, (s, e) in enumerate(zip(starts, ends)):
                    member[i, k, s:e] = True
            out["patch_member"] = member
        return out

    return collate


collate_tokenized = make_collate(SRC_PAD, TGT_PAD)
collate_bytes = make_collate(BYTE_PAD, BYTE_PAD)


class BucketBatchSampler(Sampler[list[int]]):
    """Length-bucketed batching under a source-token budget.

    Sorting by length keeps padding waste low; batches are shuffled every epoch
    (deterministically from seed + epoch). A hard cap on batch size keeps very
    short-line batches from becoming enormous.
    """

    def __init__(
        self,
        lengths: list[int],
        token_budget: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        self.batches: list[list[int]] = []
        batch: list[int] = []
        batch_max = 0
        for i in order:
            candidate_max = max(batch_max, lengths[i])
            if batch and (
                (len(batch) + 1) * candidate_max > token_budget
                or len(batch) >= max_batch_size
            ):
                self.batches.append(batch)
                batch, batch_max = [], 0
                candidate_max = lengths[i]
            batch.append(i)
            batch_max = candidate_max
        if batch:
            self.batches.append(batch)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)
