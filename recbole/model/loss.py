# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters

    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class EmbMarginLoss(nn.Module):
    """ EmbMarginLoss, regularization on embeddings
    """

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class SampledSoftmaxLoss(nn.Module):
    """ SampledSoftmaxLoss with Label Smoothing

    Sampled Softmax Loss for recommendation systems.
    Computes softmax over positive item + sampled negative items,
    with optional label smoothing for regularization.

    Args:
        - label_smoothing (float): Label smoothing factor (0.0 ~ 1.0)
          0.0 means no smoothing, 0.1 means 10% smoothing

    Shape:
        - pos_score: (N,) scores for positive items
        - neg_score: (N, num_neg) scores for negative items
        - Output: scalar loss
    """

    def __init__(self, label_smoothing=0.1):
        super(SampledSoftmaxLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pos_score, neg_score):
        """
        Args:
            pos_score: (batch_size,) - scores for positive items
            neg_score: (batch_size, num_neg) - scores for sampled negative items
        """
        # Concatenate pos and neg scores: (batch_size, 1 + num_neg)
        # pos_score at index 0
        pos_score = pos_score.unsqueeze(1)  # (batch_size, 1)
        all_scores = torch.cat([pos_score, neg_score], dim=1)  # (batch_size, 1 + num_neg)

        # Target is always 0 (positive item is at index 0)
        batch_size = all_scores.size(0)
        target = torch.zeros(batch_size, dtype=torch.long, device=all_scores.device)

        if self.label_smoothing > 0:
            # Apply label smoothing manually
            n_classes = all_scores.size(1)
            log_probs = torch.nn.functional.log_softmax(all_scores, dim=-1)

            # Create smoothed target distribution
            # confidence for positive, smooth_value for negatives
            confidence = 1.0 - self.label_smoothing
            smooth_value = self.label_smoothing / (n_classes - 1)

            # One-hot with smoothing
            smooth_target = torch.full_like(all_scores, smooth_value)
            smooth_target.scatter_(1, target.unsqueeze(1), confidence)

            # Cross entropy with soft labels
            loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross entropy without smoothing
            loss = torch.nn.functional.cross_entropy(all_scores, target)

        return loss


class InBatchSoftmaxLoss(nn.Module):
    """ In-Batch Softmax Loss (Contrastive Loss)

    Uses all items in the batch as negatives for each user.
    This is more efficient and often more effective than random sampling.

    Args:
        - temperature (float): Temperature for scaling scores (default: 0.1)
        - label_smoothing (float): Label smoothing factor (default: 0.0)

    Shape:
        - user_emb: (batch_size, embedding_dim) user embeddings
        - item_emb: (batch_size, embedding_dim) positive item embeddings
        - Output: scalar loss
    """

    def __init__(self, temperature=0.1, label_smoothing=0.0):
        super(InBatchSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, user_emb, item_emb):
        """
        Args:
            user_emb: (batch_size, dim) - user embeddings
            item_emb: (batch_size, dim) - positive item embeddings
        """
        # L2 normalize embeddings
        user_emb = nn.functional.normalize(user_emb, dim=1)
        item_emb = nn.functional.normalize(item_emb, dim=1)

        # Positive scores: diagonal elements (user_i with item_i)
        pos_score = (user_emb * item_emb).sum(dim=-1)  # (batch_size,)
        pos_score = torch.exp(pos_score / self.temperature)

        # Total scores: all user-item pairs in batch
        # (batch_size, batch_size) - each user against all items
        ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)  # (batch_size,)

        # Softmax loss: -log(pos / total)
        loss = -torch.log(pos_score / (ttl_score + 1e-6))

        return torch.mean(loss)
