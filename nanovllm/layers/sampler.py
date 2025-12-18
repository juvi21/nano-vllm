import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor | None):
        # Match mini-sglang's sampling path (in-place softmax + multinomial), and avoid fp32 upcasts.
        if temperatures is None:
            return torch.argmax(logits, dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        torch.softmax(logits, dim=-1, out=logits)
        return torch.multinomial(logits, num_samples=1).view(-1)
