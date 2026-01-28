# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

        # Label smoothing parameter (0 = no smoothing, use standard BPR)
        self.label_smoothing = config['label_smoothing'] if 'label_smoothing' in config.final_config_dict else 0.0

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        # MPS doesn't support sparse tensors - keep sparse on CPU, compute there
        self.use_mps = (self.device.type == 'mps')
        if self.use_mps:
            self.norm_adj_matrix = self.get_norm_adj_mat()  # Keep on CPU
        else:
            self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        # For wandb logging - tracking log(sigmoid(scores))
        self.log_pos_score_sum = 0.0
        self.log_neg_score_sum = 0.0
        self.score_count = 0

    def reset_score_tracking(self):
        """Reset accumulated scores at the start of each epoch."""
        self.log_pos_score_sum = 0.0
        self.log_neg_score_sum = 0.0
        self.score_count = 0

    def get_avg_log_scores(self):
        """Get average log(sigmoid(scores)) for the epoch."""
        if self.score_count == 0:
            return 0.0, 0.0
        avg_pos = self.log_pos_score_sum / self.score_count
        avg_neg = self.log_neg_score_sum / self.score_count
        return avg_pos, avg_neg

    def bpr_loss_with_label_smoothing(self, pos_scores, neg_scores):
        """
        BPR Loss with Label Smoothing.

        Standard BPR: -log(sigmoid(pos - neg))
        With label smoothing (ε): -(1-ε)*log(p) - ε*log(1-p)
        """
        diff = pos_scores - neg_scores
        prob = torch.sigmoid(diff)

        soft_target = 1.0 - self.label_smoothing
        loss = -soft_target * torch.log(prob + 1e-10) - self.label_smoothing * torch.log(1 - prob + 1e-10)

        return loss.mean()

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix using COO format directly
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        # Create row, col, data arrays for COO matrix
        rows = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
        cols = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        data = np.ones(len(rows), dtype=np.float32)

        A = sp.coo_matrix((data, (rows, cols)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        A = A.tocsr()

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        indices = torch.LongTensor(np.array([row, col]))
        values = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(indices, values, L.shape)
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            if self.use_mps:
                # MPS: move embeddings to CPU for sparse.mm, then back to MPS
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings.cpu()).to(self.device)
            else:
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss (with optional label smoothing)
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        if self.label_smoothing > 0:
            mf_loss = self.bpr_loss_with_label_smoothing(pos_scores, neg_scores)
        else:
            mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Track log(softmax(scores)) for wandb logging - measures overconfidence
        with torch.no_grad():
            # Compute scores for all items: [batch_size, n_items]
            all_scores = torch.matmul(u_embeddings, item_all_embeddings.T)
            # Log softmax over all items
            log_probs = F.log_softmax(all_scores, dim=1)
            # Get log probabilities for positive and negative items
            batch_size = pos_item.shape[0]
            log_pos = log_probs[torch.arange(batch_size, device=pos_item.device), pos_item].mean().item()
            log_neg = log_probs[torch.arange(batch_size, device=neg_item.device), neg_item].mean().item()
            self.log_pos_score_sum += log_pos
            self.log_neg_score_sum += log_neg
            self.score_count += 1

        # calculate reg Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
