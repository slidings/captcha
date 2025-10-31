# src/decode.py
import torch
from .vocab import decode_greedy

def greedy_decode(logits: torch.Tensor) -> str:
    """
    logits: (T,B,C) raw scores
    returns list[str] length B
    """
    probs = logits.softmax(dim=-1)           # (T,B,C)
    best = probs.argmax(dim=-1)              # (T,B)
    best = best.permute(1,0).cpu().tolist()  # (B,T) as python lists
    return [decode_greedy(seq) for seq in best]
