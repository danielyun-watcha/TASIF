"""
TiSASRec
################################################

Reference:
    Jiacheng Li et al. "Time Interval Aware Self-Attention for Sequential Recommendation." in WSDM 2020.

Reference:
    https://github.com/JiachengLi1995/TiSASRec

"""

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import FeedForward
import math
import copy

## TiSASRec
class TimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(TimeAwareMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
        absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
        time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]
        time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2,
                                                                                  4)  # [B, n_heads, L, L, D/n_heads]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
        attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(
            -1)  # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]

        attention_scores = attention_scores + attention_scores_pos + attention_scores_time

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
        context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
        context_layer = context_layer + context_layer_pos + context_layer_time

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps
    ):
        super(TimeAwareTransformerLayer, self).__init__()
        self.multi_head_attention = TimeAwareMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K,
                time_matrix_emb_V):
        attention_output = self.multi_head_attention(hidden_states, attention_mask, absolute_pos_K, absolute_pos_V,
                                                     time_matrix_emb_K, time_matrix_emb_V)
        feedforward_output = self.feed_forward(attention_output)

        return feedforward_output


class TimeAwareTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

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
            layer_norm_eps=1e-12
    ):

        super(TimeAwareTransformerEncoder, self).__init__()
        layer = TimeAwareTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,
                attention_mask,
                absolute_pos_K,
                absolute_pos_V,
                time_matrix_emb_K,
                time_matrix_emb_V,
                output_all_encoded_layers=True):
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
            hidden_states = layer_module(hidden_states, attention_mask, absolute_pos_K,
                                         absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TiSASRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(TiSASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.time_span = config['time_span']
        self.timestamp = config['TIME_FIELD'] + '_list'

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.absolute_pos_K_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.absolute_pos_V_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_K_embedding = nn.Embedding(self.time_span + 1, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_V_embedding = nn.Embedding(self.time_span + 1, self.hidden_size, padding_idx=0)

        self.ti_trm_encoder = TimeAwareTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

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

    def forward(self, item_seq, item_seq_len, time_matrix):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        absolute_pos_K = self.absolute_pos_K_embedding(position_ids)  # [B, L, D]
        absolute_pos_V = self.absolute_pos_V_embedding(position_ids)

        time_matrix_emb_K = self.time_matrix_emb_K_embedding(time_matrix)  # [B, L, L, D]
        time_matrix_emb_V = self.time_matrix_emb_V_embedding(time_matrix)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        absolute_pos_K = self.dropout(absolute_pos_K)
        absolute_pos_V = self.dropout(absolute_pos_V)
        time_matrix_emb_K = self.dropout(time_matrix_emb_K)
        time_matrix_emb_V = self.dropout(time_matrix_emb_V)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.ti_trm_encoder(
            input_emb,
            extended_attention_mask,
            absolute_pos_K,
            absolute_pos_V,
            time_matrix_emb_K,
            time_matrix_emb_V,
            output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output  # [B H]

    def get_time_matrix(self, time_seq):  # time_seq -> time_matrix: [B, L] -> [B, L, L]
        time_seq = time_seq

        time_matrix_i = time_seq.unsqueeze(-1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix_j = time_seq.unsqueeze(1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix = torch.abs(time_matrix_i - time_matrix_j)
        max_time_matrix = (torch.ones_like(time_matrix) * self.time_span).to(self.device)
        time_matrix = torch.where(time_matrix > self.time_span, max_time_matrix, time_matrix).int()

        return time_matrix

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        seq_output = self.forward(item_seq, item_seq_len, time_matrix)

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
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        seq_output = self.forward(item_seq, item_seq_len, time_matrix)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = interaction[self.timestamp]
        time_matrix = self.get_time_matrix(time_seq)

        seq_output = self.forward(item_seq, item_seq_len, time_matrix)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
