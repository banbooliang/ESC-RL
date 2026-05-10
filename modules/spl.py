
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig

from modules.llm_refiner import BaseLLMRefiner, RefinementPayload

CONDITION_NAMES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'lung opacity', 'lung lesion', 'edema',
    'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion',
    'pleural other', 'fracture', 'support devices', 'no finding',
]
STATUS_WORDS = {0: 'blank', 1: 'positive', 2: 'negative', 3: 'uncertain'}

def status_to_description(disease: str, status: int) -> str:
    if status == 1: return f'{disease} is present.'
    if status == 2: return f'{disease} is absent.'
    if status == 3: return f'{disease} is uncertain.'
    return f'{disease} is not mentioned.'

def build_soft_preference_label(obs_status: int, gt_status: int) -> torch.Tensor:
    if obs_status == gt_status: return torch.tensor([1.0, 0.0])
    if (obs_status == 1 and gt_status != 1) or (gt_status == 1 and obs_status == 0):
        return torch.tensor([0.0, 1.0])
    return torch.tensor([0.5, 0.5])

class PreferencePredictor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dropout_prob=0.1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        self.encoder = BertModel.from_pretrained(model_name, config=config)
        self.head = nn.Linear(config.hidden_size, 2)

    def forward(self, left_texts: List[str], right_texts: List[str], device: torch.device):
        tokens = self.tokenizer(left_texts, right_texts, padding=True, truncation=True, max_length=96, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.encoder(**tokens, return_dict=True)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(cls)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

@dataclass
class SPLResult:
    loss_pref: torch.Tensor
    trusted_mask: torch.Tensor
    pair_confidence: torch.Tensor
    refined_reports: List[str]
    trusted_descriptions: List[List[str]]
    disease_votes: List[Dict[str, str]]

class SelfCorrectingPreferenceLearning(nn.Module):
    def __init__(self, predictor: PreferencePredictor, tau_lower: float = 0.0, tau_upper: float = 1.5,
                 infer_consensus_threshold: float = 0.55):
        super().__init__()
        self.predictor = predictor
        self.tau_lower = tau_lower
        self.tau_upper = tau_upper
        self.infer_consensus_threshold = infer_consensus_threshold

    def forward(self, candidate_reports: List[List[str]], candidate_status: torch.Tensor,
                gt_reports: List[str], gt_status: torch.Tensor, device: torch.device,
                llm_refiner: Optional[BaseLLMRefiner] = None) -> SPLResult:
        bsz, num_candidates, num_disease = candidate_status.shape
        left_texts, right_texts, target_soft = [], [], []
        for b in range(bsz):
            for n in range(num_candidates):
                for k in range(num_disease):
                    left_texts.append(status_to_description(CONDITION_NAMES[k], int(candidate_status[b, n, k].item())))
                    right_texts.append(status_to_description(CONDITION_NAMES[k], int(gt_status[b, k].item())))
                    target_soft.append(build_soft_preference_label(int(candidate_status[b, n, k].item()), int(gt_status[b, k].item())))
        target_soft = torch.stack(target_soft, dim=0).to(device)
        logits, probs = self.predictor(left_texts, right_texts, device)
        log_probs = F.log_softmax(logits, dim=-1)
        loss_pref = -(target_soft * log_probs).sum(dim=-1).mean()
        kl = (target_soft * (target_soft.clamp_min(1e-8).log() - probs.clamp_min(1e-8).log())).sum(dim=-1)
        kl = kl.view(bsz, num_candidates, num_disease)
        pair_conf = probs[..., 0].view(bsz, num_candidates, num_disease)
        trusted_mask = ((kl > self.tau_lower) & (kl < self.tau_upper)).float()
        trusted_descriptions, disease_votes = self._build_train_trusted_descriptions(candidate_status, gt_status, trusted_mask, pair_conf)
        refined_reports = self._refine(candidate_reports, trusted_descriptions, disease_votes, llm_refiner)
        return SPLResult(loss_pref=loss_pref, trusted_mask=trusted_mask, pair_confidence=pair_conf,
                         refined_reports=refined_reports, trusted_descriptions=trusted_descriptions, disease_votes=disease_votes)

    @torch.no_grad()
    def infer(self, candidate_reports: List[List[str]], candidate_status: torch.Tensor,
              device: torch.device, llm_refiner: Optional[BaseLLMRefiner] = None) -> SPLResult:
        bsz, num_candidates, num_disease = candidate_status.shape
        majority = []
        for b in range(bsz):
            votes = []
            for k in range(num_disease):
                vals = candidate_status[b, :, k].tolist()
                counts = {v: vals.count(v) for v in set(vals)}
                best = sorted(counts.items(), key=lambda x: (-x[1], {1:0,2:1,3:2,0:3}.get(x[0], 4)))[0][0]
                votes.append(best)
            majority.append(votes)
        majority = torch.tensor(majority, device=device, dtype=candidate_status.dtype)
        left_texts, right_texts = [], []
        for b in range(bsz):
            for n in range(num_candidates):
                for k in range(num_disease):
                    left_texts.append(status_to_description(CONDITION_NAMES[k], int(candidate_status[b, n, k].item())))
                    right_texts.append(status_to_description(CONDITION_NAMES[k], int(majority[b, k].item())))
        logits, probs = self.predictor(left_texts, right_texts, device)
        pair_conf = probs[..., 0].view(bsz, num_candidates, num_disease)
        trusted_mask = (pair_conf >= self.infer_consensus_threshold).float()
        trusted_descriptions, disease_votes = self._build_infer_trusted_descriptions(candidate_status, majority, trusted_mask, pair_conf)
        refined_reports = self._refine(candidate_reports, trusted_descriptions, disease_votes, llm_refiner)
        zero = torch.zeros((), device=device)
        return SPLResult(loss_pref=zero, trusted_mask=trusted_mask, pair_confidence=pair_conf,
                         refined_reports=refined_reports, trusted_descriptions=trusted_descriptions, disease_votes=disease_votes)

    def _build_train_trusted_descriptions(self, candidate_status, gt_status, trusted_mask, pair_conf):
        bsz, num_candidates, num_disease = candidate_status.shape
        trusted_descriptions, disease_votes = [], []
        for b in range(bsz):
            descs, vote_map = [], {}
            for k in range(num_disease):
                gt_k = int(gt_status[b, k].item())
                vote_map[CONDITION_NAMES[k]] = STATUS_WORDS[gt_k]
                scores = []
                for n in range(num_candidates):
                    if trusted_mask[b, n, k] > 0 and int(candidate_status[b, n, k].item()) == gt_k:
                        scores.append((float(pair_conf[b, n, k].item()), status_to_description(CONDITION_NAMES[k], gt_k)))
                if scores:
                    descs.append(sorted(scores, key=lambda x: -x[0])[0][1])
            trusted_descriptions.append(descs)
            disease_votes.append(vote_map)
        return trusted_descriptions, disease_votes

    def _build_infer_trusted_descriptions(self, candidate_status, majority_status, trusted_mask, pair_conf):
        bsz, num_candidates, num_disease = candidate_status.shape
        trusted_descriptions, disease_votes = [], []
        for b in range(bsz):
            descs, vote_map = [], {}
            for k in range(num_disease):
                maj = int(majority_status[b, k].item())
                vote_map[CONDITION_NAMES[k]] = STATUS_WORDS[maj]
                scores = []
                for n in range(num_candidates):
                    if trusted_mask[b, n, k] > 0 and int(candidate_status[b, n, k].item()) == maj:
                        scores.append((float(pair_conf[b, n, k].item()), status_to_description(CONDITION_NAMES[k], maj)))
                if scores:
                    descs.append(sorted(scores, key=lambda x: -x[0])[0][1])
            trusted_descriptions.append(descs)
            disease_votes.append(vote_map)
        return trusted_descriptions, disease_votes

    def _refine(self, candidate_reports, trusted_descriptions, disease_votes, llm_refiner):
        if llm_refiner is None:
            return [reports[0] if reports else '' for reports in candidate_reports]
        payloads = [RefinementPayload(candidate_reports=r, trusted_descriptions=d, disease_votes=v)
                    for r, d, v in zip(candidate_reports, trusted_descriptions, disease_votes)]
        return llm_refiner.refine_batch(payloads)
