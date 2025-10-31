# src/vocab.py
from typing import List

# digits + lowercase (no uppercase)
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

BLANK = 0
ITOCH = ["<blank>"] + list(CHARS)
CTOIX = {c: i+1 for i, c in enumerate(CHARS)}

def encode_label(text: str) -> List[int]:
    # keep chars only if in vocabulary
    return [CTOIX[c] for c in text if c in CTOIX]

def decode_greedy(indices: List[int]) -> str:
    res, prev = [], None
    for x in indices:
        if x != BLANK and x != prev:
            res.append(ITOCH[x])
        prev = x
    return "".join(res)
