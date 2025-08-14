import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, FeedForward, VanillaAttention
from recbole.model.loss import BPRLoss
import copy
import math

class GatingFusor(nn.Module):
    def __init__(self, feature_dim):
        """
        initial Gating
        :param feature_dim: D
        """
        super().__init__()
        # shape (D, 1)
        self.weight = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, features):
        """
        forward
        :param features: features, shape (B, L, N, D)
        :return: fused_feature, shape (B, L, D)
        """
        # energy scores: (B, L, N, D) @ (D, 1) -> (B, L, N, 1)
        energy = features @ self.weight  

        # weights (softmax across the `N` dimension): (B, L, N)
        weights = torch.nn.functional.softmax(energy.squeeze(-1), dim=-2)

        # weight sum: (B, L, N, D) * (B, L, N, 1) -> (B, L, D)
        fused_feature = torch.sum(features * weights.unsqueeze(-1), dim=-2)

        return fused_feature

class FilterLayer(nn.Module):
    def __init__(self, hidden_size, filter_dropout_prob, max_seq_length, filter_weight=0.5):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, max_seq_length//2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(filter_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.weight = nn.Parameter(torch.tensor(filter_weight))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        _, seq_len, _ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        sequence_emb_fft = self.out_dropout(sequence_emb_fft)
        # hidden_states = self.LayerNorm(sequence_emb_fft + input_tensor)

        weight = torch.sigmoid(self.weight)
        hidden_states = weight * sequence_emb_fft + (1 - weight) * input_tensor
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
    
class TASIFMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, fusion_type, max_len):
        super(TASIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.max_len = max_len
        self.fusion_type = fusion_type

        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.query_f = nn.Linear(hidden_size, self.all_head_size)
        self.key_f = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.value_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        
        if self.fusion_type == 'concat':
            self.tensor_fusion_layer = nn.Linear(hidden_size * (2 + self.feat_num), hidden_size)
        elif self.fusion_type == 'gate':
            self.tensor_fusion_layer = GatingFusor(hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attr_dense = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], attribute_hidden_size[_])) for _ in range(self.feat_num)])
        self.attr_LayerNorm = nn.ModuleList([copy.deepcopy(nn.LayerNorm(attribute_hidden_size[_], eps=layer_norm_eps)) for _ in range(self.feat_num)])

        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def compute(self, input_tensor, value_layer, attention_scores, attention_mask, dense, LayerNorm):
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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

    def forward(self, input_tensor, attribute_table, position_embedding, attention_mask):
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        if self.fusion_type == 'sum':
            fusion_tensor = input_tensor + torch.sum(torch.cat(attribute_table, dim=-2), dim=-2) + position_embedding
        elif self.fusion_type == 'concat':
            attr_tensor = torch.cat(attribute_table, dim=-1).squeeze(-2)
            fusion_tensor = torch.cat((input_tensor, attr_tensor, position_embedding), dim=-1)
            fusion_tensor = self.tensor_fusion_layer(fusion_tensor)
        elif self.fusion_type == 'gate':
            attr_tensor = torch.cat(attribute_table, dim=-2)
            fusion_tensor = torch.cat([input_tensor.unsqueeze(-2), attr_tensor, position_embedding.unsqueeze(-2)], dim=-2)
            fusion_tensor = self.tensor_fusion_layer(fusion_tensor)

        fusion_query_layer = self.transpose_for_scores(self.query_f(fusion_tensor))
        fusion_key_layer = self.transpose_for_scores(self.key_f(fusion_tensor))

        fusion_attention_scores = torch.matmul(fusion_query_layer, fusion_key_layer.transpose(-1, -2))

        attribute_hidden_states_table = []
        for i, (attribute_query, attribute_key, attribute_value, attr_dense, attr_layernorm) in enumerate(
                zip(self.query_layers, self.key_layers, self.value_layers,self.attr_dense, self.attr_LayerNorm)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            attribute_query_layer = self.transpose_for_scores(attribute_query(attribute_tensor))
            attribute_key_layer = self.transpose_for_scores(attribute_key(attribute_tensor))
            attribute_value_layer = self.transpose_for_scores(attribute_value(attribute_tensor))
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_hidden_states = self.compute(attribute_tensor, attribute_value_layer, attribute_attention_scores, attention_mask, attr_dense, attr_layernorm)
            attribute_hidden_states_table.append(attribute_hidden_states.unsqueeze(-2))

        item_hidden_states = self.compute(input_tensor, item_value_layer, fusion_attention_scores, attention_mask, self.dense, self.LayerNorm)
        return item_hidden_states, attribute_hidden_states_table

class TASIFTransformerLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps, fusion_type, max_len, filter_weight, filter_dropout_prob 
    ):
        super(TASIFTransformerLayer, self).__init__()
        self.multi_head_attention = TASIFMultiHeadAttention(
            n_heads, hidden_size, attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, fusion_type, max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.attr_feed_forward = nn.ModuleList([copy.deepcopy(FeedForward(attribute_hidden_size[_], intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)) for _ in range(feat_num)])
        self.filter = FilterLayer(hidden_size, filter_dropout_prob, max_len, filter_weight)
        self.filter_attr = nn.ModuleList([copy.deepcopy(FilterLayer(attribute_hidden_size[_], filter_dropout_prob, max_len, filter_weight)) for _ in range(feat_num)])

    def forward(self, hidden_states, attribute_hidden_states, position_embedding, attention_mask):
        hidden_states = self.filter(hidden_states)
        for i in range(len(attribute_hidden_states)):
            attribute_hidden_states[i] = self.filter_attr[i](attribute_hidden_states[i].squeeze(-2)).unsqueeze(-2)

        attention_output, attribute_attention_output = self.multi_head_attention(hidden_states, attribute_hidden_states, position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        feedforward_attribute_output = []
        for i, output in enumerate(attribute_attention_output):
            output = self.attr_feed_forward[i](output)
            feedforward_attribute_output.append(output)        
        return feedforward_output, feedforward_attribute_output

class TASIFTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None, 
        filter_weight = None,
        filter_dropout_prob = None
    ):

        super(TASIFTransformerEncoder, self).__init__()
        layer = TASIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len, filter_weight, filter_dropout_prob 
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attribute_hidden_states, position_embedding, attention_mask, output_all_encoded_layers=True):
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
            hidden_states, attribute_hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((hidden_states, attribute_hidden_states))
        if not output_all_encoded_layers:
            all_encoder_layers.append((hidden_states, attribute_hidden_states))
        return all_encoder_layers

class TASIF(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TASIF, self).__init__(config, dataset)
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

        self.trm_encoder = TASIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length,
            filter_weight=self.config['filter_weight'],
            filter_dropout_prob=self.config['filter_dropout_prob']
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

        self.attribute_weight = nn.Parameter(torch.ones(len(self.selected_features)), requires_grad=True)

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
        item_emb = self.item_embedding(item_seq) + self.ts_embedding(ts_seq)
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        attr_feature_table = self.get_attr_emb(item_seq)

        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, attr_feature_table, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1][0]
        attribute_output_table = trm_output[-1][1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        attribute_seq_output_table = []
        for attribute_output in attribute_output_table:
            attribute_seq_output = self.gather_indexes(attribute_output.squeeze(-2), item_seq_len - 1)
            attribute_seq_output_table.append(attribute_seq_output)
        return seq_output, attribute_seq_output_table

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = torch.nn.functional.normalize(view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, attribute_seq_output_table = self.forward(item_seq, ts_seq, item_seq_len)
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

            attr_emb_table = self.get_attr_emb()
            al_loss = 0.0
            cl_loss = 0.0
            weights = torch.softmax(self.attribute_weight, dim=0)
            for i, (attribute_seq_output, atrribute_emb) in enumerate(zip(attribute_seq_output_table, attr_emb_table)):
                atrribute_emb = atrribute_emb.squeeze(-2)                
                attr_logits = torch.matmul(attribute_seq_output, atrribute_emb.transpose(0, 1))
                al_loss += weights[i] * self.loss_fct(attr_logits, pos_items)
                cl_loss += weights[i] * self.InfoNCE(test_item_emb[torch.unique(pos_items)], atrribute_emb[torch.unique(pos_items)], 0.2)

            # attr_emb_all = torch.sum(torch.cat(attr_emb_table, dim=-2), dim=-2) 
            # cl_loss = self.InfoNCE(test_item_emb[torch.unique(pos_items)], attr_emb_all[torch.unique(pos_items)], 0.2) + self.InfoNCE(attr_emb_all[torch.unique(pos_items)], test_item_emb[torch.unique(pos_items)], 0.2)

            total_loss = loss + self.config['al_weight'] * al_loss + self.config['cl_weight'] * cl_loss
            if self.attribute_predictor !='' and self.attribute_predictor != 'not':
                loss_dic = {}
                attribute_loss_sum = 0
                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output)
                    attribute_labels = interaction.interaction[self.selected_features[i]]
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[
                        self.selected_features[i]])

                    if len(attribute_labels.shape) > 2:
                        attribute_labels = attribute_labels.sum(dim=1)
                    attribute_labels = attribute_labels.float()
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                    attribute_loss = torch.mean(attribute_loss[:, 1:])
                    loss_dic[self.selected_features[i]] = attribute_loss

                for i,attribute in enumerate(self.selected_features):
                    attribute_loss_sum += weights[i] * loss_dic[attribute]
                total_loss += self.lamdas * attribute_loss_sum
            return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, ts_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, attribute_seq_output_table = self.forward(item_seq, ts_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        atrribute_emb_table = self.get_attr_emb()

        atrribute_scores_table = []
        weights = torch.softmax(self.attribute_weight, dim=0)
        for i, (attribute_seq_output, atrribute_emb) in enumerate(zip(attribute_seq_output_table, atrribute_emb_table)):
            atrribute_emb = atrribute_emb.squeeze(-2)                
            atrribute_scores = torch.matmul(attribute_seq_output, atrribute_emb.transpose(0, 1))
            atrribute_scores_table.append(weights[i] * atrribute_scores)
        atrribute_scores = torch.stack(atrribute_scores_table).sum(dim=0)
        scores = (1 - self.config['attr_score_weight']) * scores + self.config['attr_score_weight'] * atrribute_scores
        return scores