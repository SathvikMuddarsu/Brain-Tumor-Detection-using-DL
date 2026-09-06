import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import CSCANConfig


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy that accepts soft (e.g. MixUp/CutMix-blended) targets
    instead of integer class indices. nn.CrossEntropyLoss cannot consume
    soft targets directly, so this is used only for mixed batches; regular
    batches keep using get_loss_function() below.
    """

    def __init__(self, label_smoothing: float = 0.1, num_classes: int = 4):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits, soft_targets):
        if self.label_smoothing > 0:
            soft_targets = soft_targets * (1 - self.label_smoothing) + \
                self.label_smoothing / self.num_classes
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def get_loss_function(weights=None):
    """
    Returns the loss function.
    By default, uses Cross-Entropy Loss, which is the standard loss function for
    multi-class classification. It combines log_softmax and NLLLoss in a single class.
    
    Why it is used:
    Cross-Entropy directly calculates the Kullback-Leibler divergence between
    the predicted logits and true class probabilities. If class weights are provided,
    it applies them during loss scaling, preventing class-imbalance bias.
    """
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32).to(CSCANConfig.DEVICE)
        print(f"[Info] Applying class weights to Cross-Entropy Loss: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    return nn.CrossEntropyLoss(label_smoothing=0.1)
