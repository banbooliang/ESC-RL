import torch
import torch.nn as nn


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output