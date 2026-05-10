from collections import OrderedDict
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu

_BLEU_SCORER = None

def init_scorer():
    global _BLEU_SCORER
    _BLEU_SCORER = _BLEU_SCORER or Bleu(4)


def normalize_text(s: str) -> str:
    return ' '.join(str(s).strip().split())


def strip_prompt_from_caption(caption: str, num_prompt_tokens: int = 18) -> str:
    parts = str(caption).strip().split()
    if len(parts) <= num_prompt_tokens:
        return ''
    return ' '.join(parts[num_prompt_tokens:]).strip()


def get_self_critical_reward_text(greedy_reports, gt_reports, sample_reports):
    init_scorer()
    batch_size = len(gt_reports)
    res = OrderedDict()
    for i, txt in enumerate(sample_reports):
        res[i] = [normalize_text(txt)]
    for i, txt in enumerate(greedy_reports):
        res[batch_size + i] = [normalize_text(txt)]
    gts = OrderedDict()
    for i, txt in enumerate(gt_reports):
        gts[i] = [normalize_text(txt)]
    res_dict = {i: res[i] for i in range(2 * batch_size)}
    gts_dict = {i: gts[i] for i in range(batch_size)}
    gts_dict.update({batch_size + i: gts[i] for i in range(batch_size)})
    _, bleu_scores = _BLEU_SCORER.compute_score(gts_dict, res_dict, verbose=0)
    bleu4 = np.array(bleu_scores[3], dtype=np.float32)
    return bleu4[:batch_size] - bleu4[batch_size:]
