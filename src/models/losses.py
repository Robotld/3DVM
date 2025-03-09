import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        """
        :param smoothing: Smoothing factor for label smoothing.
        :param class_weights: Tensor of shape (num_classes,) containing weights for each class.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred, target):
        """
        :param pred: Predictions of shape (batch_size, num_classes)
        :param target: Ground truth labels of shape (batch_size,)
        :return: Loss (scalar)
        """
        n_classes = pred.size(1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(pred, dim=1)

        if self.class_weights is not None:
            # Ensure class_weights is on the same device as pred
            self.class_weights = self.class_weights.to(pred.device)
            loss = -(true_dist * log_probs * self.class_weights.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(true_dist * log_probs).sum(dim=1)

        return loss.mean()