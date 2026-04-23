import regex as re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Any
import json
import random
import numpy as np

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents = re.split("|".join(map(re.escape, special_tokens)), text) if special_tokens else [text]

    # Pretokenization
    pretoken_counts = defaultdict(int)
    for document in documents:
        for match in re.finditer(PATTERN, document):
            pretoken = match.group(0).encode("utf-8")
            pretoken_seq = tuple(pretoken)
            if pretoken_seq:
                pretoken_counts[pretoken_seq] += 1
    
    # Vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    curr_size = 256

    for token in special_tokens:
        vocab[curr_size] = token.encode("utf-8")
        curr_size += 1

    # Merges
    merges = []
    while curr_size < vocab_size:
        pair_counts = defaultdict(int)

        for seq, freq in pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        x, y = best_pair
        new_id = curr_size
        vocab[new_id] = vocab[x] + vocab[y]
        merges.append((vocab[x], vocab[y]))
        curr_size += 1

        new_pretoken_counts = defaultdict(int)

        for pretoken_seq, freq in pretoken_counts.items():
            new_pretoken_seq = []
            i = 0

            while i < len(pretoken_seq):
                if i < len(pretoken_seq) - 1 and pretoken_seq[i] == x and pretoken_seq[i + 1] == y:
                    new_pretoken_seq.append(new_id)
                    i += 2
                else:
                    new_pretoken_seq.append(pretoken_seq[i])
                    i += 1

            new_pretoken_counts[tuple(new_pretoken_seq)] += freq

        pretoken_counts = new_pretoken_counts

    return vocab, merges


def train_bpe_tinystories():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    out_dir = Path("data")

    vocab_json = {
        str(token_id): list(token_bytes)
        for token_id, token_bytes in vocab.items()
    }
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)

    merges_json = [
        [list(x), list(y)]
        for x, y in merges
    ]
    with open(out_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f)

    longest_id, longest_bytes = max(vocab.items(), key=lambda x: len(x[1]))
    print("Longest token id:", longest_id)
    print("Length:", len(longest_bytes))


class Tokenizer:
    def __init__(
      self,
      vocab: dict[int, bytes],
      merges: list[tuple[bytes, bytes]],
      special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {}
        for token_id, token_bytes in self.vocab.items():
            self.bytes_to_id[token_bytes] = token_id

        next_id = max(self.vocab.keys()) + 1
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.bytes_to_id:
                self.vocab[next_id] = st_bytes
                self.bytes_to_id[st_bytes] = next_id
                next_id += 1

        self.merge_order = {}
        for i, merge in enumerate(self.merges):
            self.merge_order[merge] = i

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(map(re.escape, sorted_special_tokens)) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {}
        for token_id, byte_list in vocab_json.items():
            vocab[int(token_id)] = bytes(byte_list)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
        merges = []
        for x, y in merges_json:
            merges.append((bytes(x), bytes(y)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]

        while True:
            best_pair = None
            best_rank = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_order:
                    rank = self.merge_order[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = (i, pair)

            if best_pair is None:
                break

            i, (left, right) = best_pair
            tokens = tokens[:i] + [left + right] + tokens[i + 2 :]

        return [self.bytes_to_id[token] for token in tokens]

    def encode(self, text: str):
        ids = []

        if self.special_pattern is None:
            documents = [text]
        else:
            documents = self.special_pattern.split(text)

        for document in documents:
            if not document:
                continue

            if document in self.special_tokens:
                ids.append(self.bytes_to_id[document.encode("utf-8")])
                continue

            for match in re.finditer(PATTERN, document):
                pretoken = match.group(0)
                ids.extend(self._encode_pretoken(pretoken))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]):
        bytes_list = b"".join(self.vocab[token_id] for token_id in ids)
        return bytes_list.decode("utf-8", errors="replace")

def tokenizer_experiments():
    out_dir = Path("data")
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(out_dir / "vocab.json"),
        merges_filepath=str(out_dir / "merges.json"),
        special_tokens=["<|endoftext|>"],
    )

    train_path = out_dir / "TinyStoriesV2-GPT4-train.txt"
    dev_path = out_dir / "TinyStoriesV2-GPT4-valid.txt"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_text = f.read()

    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
    dev_ids = np.array(tokenizer.encode(dev_text), dtype=np.uint16)

    np.save(out_dir / "tinystories_train_ids.npy", train_ids)
    np.save(out_dir / "tinystories_dev_ids.npy", dev_ids)
