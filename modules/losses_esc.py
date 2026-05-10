import torch
import torch.nn as nn


class RewardCriterionWithPrefix(nn.Module):
    def __init__(self, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, token_logprobs, seq, reward, ignore_prefix_len=0):
        target = seq[:, 1:]
        mask = (target != self.pad_token_id).float()
        if ignore_prefix_len > 0:
            mask[:, :ignore_prefix_len] = 0.0
        if reward.dim() == 1:
            reward = reward.unsqueeze(1).expand_as(token_logprobs)
        elif reward.shape != token_logprobs.shape:
            reward = reward.expand_as(token_logprobs)
        loss = -(token_logprobs * reward * mask).sum() / mask.sum().clamp_min(1.0)
        return loss
