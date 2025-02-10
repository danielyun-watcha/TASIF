# -*- coding: utf-8 -*-
# @Time    : 2022/02/22 19:32
# @Author  : Peilin Zhou, Yueqi Xie
# @Email   : zhoupl@pku.edu.cn
r"""
SASRecD
################################################

Reference:
    Yueqi Xie and Peilin Zhou et al. "Decouple Side Information Fusion for Sequential Recommendation"
    Submited to SIGIR 2022.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, VanillaAttention, FeedForward
from recbole.model.loss import BPRLoss
import copy
import math
import numpy as np
import torch.nn.functional as F

class ContrastiveHIE(nn.Module):
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = temperature
        
    def forward(self, hx, ha):
        # 投影到同一空间
        z_x = torch.nn.functional.normalize(self.proj(hx), dim=-1)
        z_a = torch.nn.functional.normalize(self.proj(ha), dim=-1)
        
        # 计算相似度矩阵
        sim = torch.matmul(z_x, z_a.transpose(-2, -1)) / self.temperature
        
        # 软掩码：保留相似度高的部分
        mask = torch.nn.functional.softmax(sim, dim=-1)
        
        # 提取同质信息
        ha_homo = torch.matmul(mask, ha)
        
        return ha_homo
    
class SubspaceHIE(nn.Module):
    def __init__(self, hidden_dim, num_subspaces=4):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.hidden_dim = hidden_dim
        self.subspace_dim = hidden_dim // num_subspaces
        
        # 编码器: d -> d//4
        self.subspace_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.subspace_dim) 
            for _ in range(num_subspaces)
        ])
        
        # 解码器: d//4 -> d
        self.subspace_reconstructs = nn.ModuleList([
            nn.Linear(self.subspace_dim, hidden_dim)
            for _ in range(num_subspaces)
        ])
        
    def forward(self, hx, ha):  # [b,n,d]
        # 多子空间投影
        x_subspaces = [proj(hx) for proj in self.subspace_projs]  # list of [b,n,d//4]
        a_subspaces = [proj(ha) for proj in self.subspace_projs]  # list of [b,n,d//4]
        
        # 计算每个子空间的相似度
        similarities = []
        for x_sub, a_sub in zip(x_subspaces, a_subspaces):
            sim = F.cosine_similarity(x_sub.unsqueeze(2), 
                                    a_sub.unsqueeze(1), dim=-1)  # [b,n,n]
            similarities.append(sim)
        
        # [num_subspaces,b,n,n]
        subspace_weights = F.softmax(torch.stack(similarities), dim=0)
        
        # 重建同质特征
        ha_homo = torch.zeros(ha.size(0), ha.size(1), self.hidden_dim).to(ha.device)
        
        for i, (weight, reconstruct) in enumerate(zip(subspace_weights, self.subspace_reconstructs)):
            # weighted_features: [b,n,d//4]
            weighted_features = torch.matmul(weight, a_subspaces[i])
            # 重建到原始维度: [b,n,d]
            ha_homo += reconstruct(weighted_features)
            
        return ha_homo  # [b,n,d]
    
class ManifoldAlignmentHIE(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.alignment_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, hx, ha):
        """
        Args:
            hx: Hidden states of IDs, shape (b, n, d)
            ha: Hidden states of attributes, shape (b, n, d)

        Returns:
            ha_homo: Homogeneous part of ha, shape (b, n, d)
        """
        b, n, d = hx.shape

        # Feature transformation
        tx = self.transform(hx.view(-1, d))  # Shape: (b * n, d)
        ta = self.transform(ha.view(-1, d))  # Shape: (b * n, d)

        tx = tx.view(b, n, d)  # Reshape back to (b, n, d)
        ta = ta.view(b, n, d)  # Reshape back to (b, n, d)

        # Compute local manifold structure
        dx = torch.cdist(tx, tx, p=2)  # Shape: (b, n, n)
        da = torch.cdist(ta, ta, p=2)  # Shape: (b, n, n)

        # Structure similarity
        structure_sim = torch.nn.functional.cosine_similarity(
            dx.view(b, -1),
            da.view(b, -1),
            dim=-1
        ).unsqueeze(-1)  # Shape: (b, 1)

        # Adaptive alignment weights
        align_weight = self.alignment_net(
            torch.cat([tx, ta], dim=-1)  # Shape: (b, n, 2d)
        ).view(b, n, 1)  # Shape: (b, n, 1)

        # Homogeneous information extraction
        ha_homo = structure_sim.unsqueeze(1) * align_weight * ha  # Shape: (b, n, d)

        return ha_homo

class DistillationHIE(nn.Module):
    def __init__(self, hidden_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
        self.student = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.teacher = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, hx, ha):
        # 教师网络处理item特征
        teacher_out = self.teacher(hx)
        
        # 学生网络处理属性特征
        student_out = self.student(ha)
        
        # 知识蒸馏
        soft_targets = torch.nn.functional.softmax(teacher_out / self.temperature, dim=-1)
        soft_prob = torch.nn.functional.softmax(student_out / self.temperature, dim=-1)
        
        # 提取同质信息
        ha_homo = soft_prob * ha
        
        return ha_homo

class HomogeneousInformationExtraction(nn.Module):
    def __init__(self, n, r):
        """
        Initialize the HIE layer.

        Args:
            n: Original sequence length.
            r: Reduced sequence length (output).
        """
        super(HomogeneousInformationExtraction, self).__init__()
        self.W_r = nn.Parameter(torch.randn(n, r))  # Shape: (n, r)

    def phi(self, x):
        """Indicator function φ(x): Return 1 if x > 0, else 0."""
        return (x > 0).float()

    def forward(self, h_X, h_A):
        """
        Perform Homogeneous Information Extraction (HIE).

        Args:
            h_X: Hidden states of IDs, shape (b, n, d)
            h_A: Hidden states of attributes, shape (b, n, d)

        Returns:
            h_A_star: Homogeneous part of h_A, shape (b, n, d)
        """
        b, n, d = h_X.shape  # b: batch size, n: sequence length, d: hidden dimension

        # Reduce dimensionality of h_X along sequence length (n)
        h_X_reduced = torch.einsum('bnd,nr->brd', h_X, self.W_r)  # Shape: (b, r, d)

        # QR decomposition of h_X_reduced (batch-wise on last two dimensions)
        Q_list = []
        for i in range(b):
            Q, _ = torch.linalg.qr(h_X_reduced[i].T, mode='reduced')  # Shape: (d, r)
            Q_list.append(Q)
        Q = torch.stack(Q_list, dim=0)  # Shape: (b, d, r)

        # Project h_A onto Q
        proj_h_X = torch.matmul(h_X, Q)  # Shape: (b, n, r)
        proj_h_A = torch.matmul(h_A, Q)  # Shape: (b, n, r)

        # Compute the homogeneous part
        proj_h_A_tilde = self.phi(proj_h_X * proj_h_A) * proj_h_A  # Shape: (b, n, r)
        h_A_star = torch.matmul(proj_h_A_tilde, Q.transpose(-1, -2))  # Shape: (b, n, d)

        return h_A_star

class ASIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(ASIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_f = nn.Linear(hidden_size, self.all_head_size)
        self.key_f = nn.Linear(hidden_size, self.all_head_size)
        self.value_f = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)
        self.value_p = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.fusion_dense = nn.Linear(hidden_size, hidden_size)
        self.fusion_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)        
        
        self.pos_dense = nn.Linear(hidden_size, hidden_size)
        self.pos_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.hie = HomogeneousInformationExtraction(max_len, 16)
        # self.hie = ContrastiveHIE(hidden_size, 0.2)
        # self.hie = SubspaceHIE(hidden_size, 1)
        # self.hie = ManifoldAlignmentHIE(hidden_size)
        # self.hie = DistillationHIE(hidden_size, 0.2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute(self, input_tensor, value_layer, attention_scores, attention_mask, dense, LayerNorm):
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores = attribute_attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def forward(self, input_tensor, fusion_tensor, pos_tensor, attention_mask):
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        fusion_query_layer = self.transpose_for_scores(self.query_f(fusion_tensor))
        fusion_key_layer = self.transpose_for_scores(self.key_f(fusion_tensor))
        fusion_value_layer = self.transpose_for_scores(self.value_f(fusion_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(pos_tensor))
        pos_key_layer = self.transpose_for_scores(self.key_p(pos_tensor))
        pos_value_layer = self.transpose_for_scores(self.value_p(pos_tensor))

        item_attention_scores = torch.matmul(fusion_query_layer, fusion_key_layer.transpose(-1, -2))
        pos_attention_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attention_scores = item_attention_scores + pos_attention_scores

        hidden_states = self.compute(input_tensor, item_value_layer, attention_scores, attention_mask, self.dense, self.LayerNorm)
        fusion_hidden_states = self.compute(fusion_tensor, fusion_value_layer, item_attention_scores, attention_mask, self.fusion_dense, self.fusion_LayerNorm)
        pos_hidden_states = self.compute(pos_tensor, pos_value_layer, pos_attention_scores, attention_mask, self.pos_dense, self.pos_LayerNorm)

        hidden_states = hidden_states + self.hie(hidden_states, fusion_hidden_states)
        return hidden_states, fusion_hidden_states, pos_hidden_states

class ASIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(ASIFTransformerLayer, self).__init__()
        self.multi_head_attention = ASIFMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.fusion_feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.pos_feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, fusion_hidden_states, pos_hidden_states, attention_mask):
        attention_output, fusion_attention_output, pos_attention_output = self.multi_head_attention(hidden_states, fusion_hidden_states, pos_hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        fusion_feedforward_output = self.feed_forward(fusion_attention_output)
        pos_feedforward_output = self.feed_forward(pos_attention_output)
        return feedforward_output, fusion_feedforward_output, pos_feedforward_output

class ASIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(ASIFTransformerEncoder, self).__init__()
        layer = ASIFTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, fusion_hidden_states, pos_hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, fusion_hidden_states, pos_hidden_states = layer_module(hidden_states, fusion_hidden_states, pos_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ASIF(SequentialRecommender):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, dataset):
        super(ASIF, self).__init__(config, dataset)
        self.config = config

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.ts_embedding = nn.Embedding(self.n_time_span_ids, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset,self.attribute_hidden_size[_],[self.selected_features[_]],self.pooling_mode,self.device)) for _
             in range(len(self.selected_features))])

        self.trm_encoder = ASIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.hidden_size),
                                             nn.BatchNorm1d(num_features=self.hidden_size),
                                             nn.ReLU(),
                                             # final logits
                                             nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.n_attributes)
                                             )
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList(
                [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_]))
                 for _ in self.selected_features])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_attr_emb(self, item_seq=None):
        attr_feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            if item_seq is not None:
                sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            else:
                all_items = torch.arange(self.n_items, dtype=torch.long, device=self.device)
                sparse_embedding, dense_embedding = feature_embed_layer(None, all_items)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']
            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                attr_feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                attr_feature_table.append(dense_embedding)
        
        return attr_feature_table

    def forward(self, item_seq, ts_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)# + self.ts_embedding(ts_seq)
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        attr_feature_table = self.get_attr_emb(item_seq)
        attr_emb = torch.sum(torch.cat(attr_feature_table, dim=-2), dim=-2)

        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        fusion_emb = input_emb + attr_emb

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, fusion_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)

        cl_loss = self.config['cl_weight'] * self.contrastive_Loss(item_emb, attr_emb, 0.2)
        return seq_output, cl_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = torch.nn.functional.normalize(view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def contrastive_Loss(self, X, A, tau=1.0):
        """
        计算对比损失，支持批量处理
        X: shape [b, n, d] - ID embeddings
        A: shape [b, n, d] - Attribute embeddings
        """
        # 获取 batch size 和序列长度
        b, n, d = X.size()
        
        # 步骤1: 归一化 X 和 A
        X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)  # [b, n, d]
        A_norm = A / (A.norm(dim=-1, keepdim=True) + 1e-8)  # [b, n, d]

        # 步骤2: 计算相似度矩阵
        sim_XA = torch.matmul(X_norm, A_norm.transpose(1, 2)) / tau  # [b, n, n]
        sim_AX = torch.matmul(A_norm, X_norm.transpose(1, 2)) / tau  # [b, n, n]

        # 步骤3: 应用 softmax
        Y_X_hat = F.softmax(sim_XA, dim=-1)  # [b, n, n]
        Y_A_hat = F.softmax(sim_AX, dim=-1)  # [b, n, n]

        # 步骤4: 构造 ground truth 矩阵 (对角矩阵)
        Y = torch.eye(n, device=X.device).unsqueeze(0).expand(b, -1, -1)  # [b, n, n]

        # 步骤5: 计算损失
        loss_X = (Y * torch.log(Y_X_hat + 1e-8)).sum(dim=-1).mean(dim=-1)  # [b]
        loss_A = (Y * torch.log(Y_A_hat + 1e-8)).sum(dim=-1).mean(dim=-1)  # [b]

        # 平均 batch 的损失
        loss = -(loss_X + loss_A).mean() / 2
        return loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, cl_loss = self.forward(item_seq, ts_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

            return loss + cl_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, ts_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, ts_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores