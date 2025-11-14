# src/decode.py
import torch
from .vocab import decode_greedy, ITOCH, BLANK

# Converts a 3d matrix into 1d where each image in batch 
# returns the prediction string
def greedy_decode(logits: torch.Tensor) -> str:
    """
    logits: (Time,Batch,Classes) raw scores
    returns list[str] length B
    """
    probs = logits.softmax(dim=-1)           # (T,B,C)
    best = probs.argmax(dim=-1)              # (T,B)
    best = best.permute(1,0).cpu().tolist()  # (B,T) as python lists
    return [decode_greedy(seq) for seq in best]


# --- beam search decode using torchaudio's CTC decoder ---
# Tried beamsize 2,5,10 but results were worse than greedy.

# from torchaudio.models.decoder import ctc_decoder
'''

# --- fast beam search using torchaudio's built-in C++ backend ---
_beam_decoder = ctc_decoder(
    lexicon=None,           # no word-level lexicon
    tokens=ITOCH,           # your character set (includes "<blank>" as index 0)
    beam_size=5,            # same as beam_width
    nbest=1,                # only return top sequence
    beam_threshold=50.0,    # optional prune threshold
    blank_token=ITOCH[BLANK],
    sil_token=None,
    unk_word=None,
)

def beam_search_decode(logits: torch.Tensor, beam_width: int = 5) -> list[str]:
    """
    logits: (T, B, C) raw scores from model
    returns: list[str] length B
    """
    # torchaudio expects (B, T, C) on CPU
    log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2).cpu()

    # beam size can be adjusted dynamically
    _beam_decoder.beam_size = beam_width

    results = _beam_decoder(log_probs)

    # extract text from the nbest hypothesis
    preds = []
    for hyp in results:
        if len(hyp) > 0 and len(hyp[0].words) > 0:
            preds.append("".join(hyp[0].words))
        else:
            preds.append("")
    return preds
'''