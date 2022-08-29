import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
import logging
import copy
import math
import torch.nn.init as init
from transformers.activations import ACT2FN


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        #self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        #self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bertconfig = config
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        past_key_values_length = 0,
        params = None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_ids = position_ids.to(input_ids.device)

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, params['meta_learner.bert.embeddings.word_embeddings.weight'])
        token_type_embeddings = F.embedding(token_type_ids, params['meta_learner.bert.embeddings.token_type_embeddings.weight'])

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            #print(input_ids.device, position_ids.device)
            position_embeddings = F.embedding(position_ids,
                                              params['meta_learner.bert.embeddings.position_embeddings.weight'])
            embeddings += position_embeddings
        embeddings = F.layer_norm(embeddings, (self.bertconfig.hidden_size,),
                                  weight=params['meta_learner.bert.embeddings.LayerNorm.weight'],
                                  bias=params['meta_learner.bert.embeddings.LayerNorm.bias'],
                                  eps=self.bertconfig.layer_norm_eps)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_values = None,
        use_cache = None,
        output_attentions = False,
        output_hidden_states = False,
        return_dict = True,
        params = None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                """
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
                """
                pass
            else:
                layer_module = BertLayer(self.config)
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    params,
                    i
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, params):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = F.linear(first_token_tensor,
                                 params['meta_learner.bert.pooler.dense.weight'],
                                 params['meta_learner.bert.pooler.dense.bias'])
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #self.query = nn.Linear(config.hidden_size, self.all_head_size)
        #self.key = nn.Linear(config.hidden_size, self.all_head_size)
        #self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
        params = None,
        layer_i = None
    ):
        #mixed_query_layer = self.query(hidden_states)
        mixed_query_layer = F.linear(hidden_states,
                                     params['meta_learner.bert.encoder.layer.'+str(layer_i)+".attention.self.query.weight"],
                                     params['meta_learner.bert.encoder.layer.' + str(layer_i) + ".attention.self.query.bias"])


        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(F.linear(encoder_hidden_states,
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.weight"],
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.bias"]
                                                           ))
            value_layer = self.transpose_for_scores(F.linear(encoder_hidden_states,
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.weight"],
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.bias"]
                                                             ))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(F.linear(hidden_states,
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.weight"],
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.bias"]
                                                           ))
            value_layer = self.transpose_for_scores(F.linear(hidden_states,
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.weight"],
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.bias"]
                                                             ))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(F.linear(hidden_states,
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.weight"],
                                                           params['meta_learner.bert.encoder.layer.' + str(
                                                               layer_i) + ".attention.self.key.bias"]
                                                           ))
            value_layer = self.transpose_for_scores(F.linear(hidden_states,
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.weight"],
                                                             params['meta_learner.bert.encoder.layer.' + str(
                                                                 layer_i) + ".attention.self.value.bias"]
                                                             ))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, params, layer_i):
        hidden_states = F.linear(hidden_states, params['meta_learner.bert.encoder.layer.'+str(layer_i)+".attention.output.dense.weight"],
                                     params['meta_learner.bert.encoder.layer.' + str(layer_i) + ".attention.output.dense.bias"])
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.layer_norm(hidden_states + input_tensor, (self.config.hidden_size,),
                                     weight=params['meta_learner.bert.encoder.layer.'+str(layer_i)+".attention.output.LayerNorm.weight"],
                                     bias=params['meta_learner.bert.encoder.layer.' + str(layer_i) + ".attention.output.LayerNorm.bias"],
                                     eps=self.config.layer_norm_eps)

        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        #self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        #self.output = BertSelfOutput(config)
        self.config = config
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
        params = None,
        layer_i = None
    ):
        selfatt = BertSelfAttention(self.config)
        self_outputs = selfatt(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            params,
            layer_i
        )
        bertselfoutput = BertSelfOutput(self.config)
        attention_output = bertselfoutput(self_outputs[0], hidden_states, params, layer_i)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, params, layer_i):
        hidden_states = F.linear(hidden_states,
                                 params['meta_learner.bert.encoder.layer.' + str(
                                     layer_i) + ".intermediate.dense.weight"],
                                 params['meta_learner.bert.encoder.layer.' + str(
                                     layer_i) + ".intermediate.dense.bias"],
                                 )
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        #self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, params, layer_i):
        hidden_states = F.linear(hidden_states,
                                 params['meta_learner.bert.encoder.layer.' + str(
                                     layer_i) + ".output.dense.weight"],
                                 params['meta_learner.bert.encoder.layer.' + str(
                                     layer_i) + ".output.dense.bias"]
                                 )
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.layer_norm(hidden_states + input_tensor, (self.config.hidden_size,),
                                     params['meta_learner.bert.encoder.layer.' + str(
                                         layer_i) + ".output.LayerNorm.weight"],
                                     params['meta_learner.bert.encoder.layer.' + str(
                                         layer_i) + ".output.LayerNorm.bias"],
                                     eps=self.config.layer_norm_eps
                                     )
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        #self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.config = config
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        #self.intermediate = BertIntermediate(config)
        #self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_value = None,
        output_attentions = False,
        params = None,
        layer_i = None
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        attention = BertAttention(self.config)
        self_attention_outputs = attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            params=params,
            layer_i=layer_i
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output, params, layer_i)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output, params, layer_i):
        intermediate = BertIntermediate(self.config)
        intermediate_output = intermediate(attention_output, params, layer_i)

        bertoutput = BertOutput(self.config)
        layer_output = bertoutput(intermediate_output, attention_output, params, layer_i)
        return layer_output


class BERTSentencePair(nn.Module):
    def __init__(self, config):
        super(BERTSentencePair, self).__init__()

        self.max_length = config.bert_max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.celoss = nn.CrossEntropyLoss(reduction="mean")
        self.pred_fc1 = nn.Linear(768, 2)
        self.config = config
        self.bertconfig = BertConfig()

        self.device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")

    def process_data(self, X):
        total = []
        total_y = []
        lens = []

        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])

        o, s = X
        s_x, s_y = s
        o = o[0]

        for i, each_s in enumerate(s_x):
            cur_id = cls + o[0] + sep + each_s
            # print(len(o[0]), len(each_s), len(cur_id))
            bert_id = np.zeros([self.max_length], dtype=np.int64)

            if len(cur_id) > self.max_length:
                cur_id = cur_id[0: self.max_length]

            bert_id[0: len(cur_id)] = cur_id

            total.append(bert_id)
            lens.append(len(cur_id))
            total_y.append(s_y[i])

        total = torch.LongTensor(total).to(self.device)
        total_y = torch.LongTensor(total_y).to(self.device)
        lens = torch.LongTensor(lens).to(self.device)

        # get mask
        n = total.size()[0]
        idxes = torch.arange(self.max_length).expand(n, -1).to(self.device)
        mask = (idxes < lens.unsqueeze(1)).float()

        return total, mask, total_y

    def forward(self, X, params=None):
        if params == None:
            return self.first_update(X)
        else:
            return self.second_update(X, params)

    def first_update(self, X):
        x, mask, y = self.process_data(X)

        out = self.bert(x, attention_mask=mask)
        out = out['pooler_output']

        out = self.pred_fc1(out)

        loss = self.celoss(out, y)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y = y.cpu()
        # print(logits)
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out, acc_val, f1_val

    def get_extended_attention_mask(
            self, attention_mask, input_shape, device = None
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if not (attention_mask.dim() == 2 and self.bertconfig.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.bertconfig.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask#.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers, is_attention_chunked= False
    ):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def second_bert_emb_model(self,
            input_ids = None,
            attention_mask = None,
            params = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            past_key_values = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):

        embs = BertEmbeddings(self.bertconfig)

        output_attentions = output_attentions if output_attentions is not None else self.bertconfig.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bertconfig.output_hidden_states

        )
        return_dict = return_dict if return_dict is not None else self.bertconfig.use_return_dict

        if self.bertconfig.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bertconfig.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(embs, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bertconfig.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.bertconfig.num_hidden_layers)

        embedding_output = embs(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            params=params
        )

        encoder = BertEncoder(self.bertconfig)
        encoder_outputs = encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            params=params
        )
        sequence_output = encoder_outputs[0]

        pooler = BertPooler(self.bertconfig)
        pooled_output = pooler(sequence_output, params)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def second_update(self, X, params):
        x, mask, y = self.process_data(X)

        out = self.second_bert_emb_model(x, mask, params)
        out = out['pooler_output']

        out = F.linear(out, params['meta_learner.pred_fc1.weight'],
                       params['meta_learner.pred_fc1.bias'])

        loss = self.celoss(out, y)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y = y.cpu()
        # print(logits)
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out, acc_val, f1_val

    def predict(self, X):
        x, mask, y = self.process_data(X)

        out = self.bert(x, attention_mask=mask)
        out = out['pooler_output']

        out = self.pred_fc1(out)

        loss = self.celoss(out, y)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y = y.cpu()
        # print(logits)
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out, acc_val, f1_val

    def predict_feature(self, X):
        x, mask, y = self.process_data(X)

        out = self.bert(x, attention_mask=mask)
        out_feature = out['pooler_output']

        y = y.cpu()
        return out_feature, y


class BERT_Meta(nn.Module):
    def __init__(self, config):
        super(BERT_Meta, self).__init__()
        """
        The class defines meta-learner for MAML algorithm.
        """
        self.config = config
        self.meta_learner = BERTSentencePair(config)

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def predict(self, X):
        features, labels = self.meta_learner.predict_feature(X)
        return features, labels

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict


import torch
import torch.nn as nn
from torch import optim
import os
import pickle
import numpy as np
import random
from torch.backends import cudnn
import argparse
from tqdm import tqdm
from collections import OrderedDict
import copy
from sklearn import metrics
import torch.nn.functional as F
from torch.autograd import Variable


class Config():
    def __init__(self):
        pass

    model = "bert_meta"
    optimizer = "adam"
    lr = 2e-5
    task_lr = 2e-3
    batch_size = 1
    nepochs = 120
    dataset = "huffpost"
    data_path = "../datasets/oneshot_huffpost/"
    nepoch_no_imprv = 3
    device = 0
    train_epochs = 400
    dev_epochs = 300
    test_epochs = 300
    classes_per_set = 5
    samples_per_class = 10
    one_shot = 1
    dim_word = 50
    lstm_hidden = 50
    seed = 25
    num_train_updates = 1
    num_eval_updates = 3
    initializer_range = 0.1

    bert_max_length = 100
    target_type = "single"  # used for ACD
    test_feature_epochs = 12


config = Config()
print(config)
print("current seed-----", config.seed)
device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")


def data_loader(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes = np.random.choice(len(data_pack), config.classes_per_set * 2, False)
        classes_positive = classes[0: config.classes_per_set]
        classes_negative = classes[config.classes_per_set:]
        for j, cur_class in enumerate(classes_positive):  # each class
            x, y = [], []

            if config.target_type == "single":
                example_inds = np.random.choice(len(data_pack[cur_class][2]),
                                                config.one_shot, False)
                data_temp = np.array(data_pack[cur_class][2])[example_inds]
            else:
                example_inds = np.random.choice(len(data_pack[cur_class][3]),
                                                config.one_shot, False)
                data_temp = np.array(data_pack[cur_class][3])[example_inds]
            data_temp_x = data_temp[:, 0]

            positive_ids = np.random.choice(len(data_pack[cur_class][3]),
                                            config.samples_per_class, False)
            target_positive = np.array(data_pack[cur_class][3])[positive_ids]
            positive_x = target_positive[:, 0]

            negative_class_num = classes_negative[j]
            positive_class = data_pack[cur_class][1]
            negative_x = []
            negative_ids = np.random.choice(len(data_pack[negative_class_num][3]),
                                            len(data_pack[negative_class_num][3]) // 2, False)
            n = 0
            while len(negative_x) < config.samples_per_class:
                n_id = negative_ids[n]
                n += 1
                cur = data_pack[negative_class_num][3][n_id]
                if positive_class not in cur[1]:
                    negative_x.append(cur[0])

            x.extend(positive_x)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, y))
            random.shuffle(t_tmp)
            x[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x],
                [x[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def data_loader_huffpost(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes_positive = np.random.choice(len(data_pack), config.classes_per_set, False)
        for j, cur_class in enumerate(classes_positive):  # each class
            x, y = [], []

            example_inds = np.random.choice(len(data_pack[cur_class][1]),
                                            config.one_shot + config.samples_per_class, False)

            one_shot_id = np.array([example_inds[0]])
            data_temp = np.array(data_pack[cur_class][1])[one_shot_id]
            data_temp_x = data_temp[:, 0]

            positive_ids = example_inds[1:]
            target_positive = np.array(data_pack[cur_class][1])[positive_ids]
            positive_x = target_positive[:, 0]

            # print(cur_class)
            my_range = list(range(0, cur_class)) + list(range(cur_class + 1, len(data_pack)))
            # print(my_range)
            negative_class_num = np.random.choice(my_range, 1, False)
            negative_class_num = negative_class_num[0]
            # print(negative_class_num)
            negative_ids = np.random.choice(len(data_pack[negative_class_num][1]),
                                            config.samples_per_class, False)

            neg_temp = np.array(data_pack[negative_class_num][1])[negative_ids]
            negative_x = neg_temp[:, 0]

            x.extend(positive_x)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, y))
            random.shuffle(t_tmp)
            x[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x],
                [x[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def train_single_task(model, data, config):
    num_train_updates = config.num_train_updates

    model.train()

    one_shot, support, _ = data
    x = [one_shot, support]

    loss, _, _, _ = model(x)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

    # performs updates using calculated gradients
    # we manually compute adapted parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        if grad != None:
            adapted_params[key] = val - config.task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict


def evaluate(logits, y):
    logits = logits.detach().cpu()
    y = y.detach().cpu()

    pred = torch.max(logits, dim=-1)[1].float()
    # print(logits)
    f1_val = metrics.f1_score(y, pred, average="macro")
    acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))

    return acc_val, f1_val


def train_epoch(model, data_loader, meta_optimizer, config):
    model.train()

    total_acc, total_loss, total_f1 = [], [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        adapted_state_dicts = []

        for i_batch in range(len(batch)):
            data = batch[i_batch]
            a_dict = train_single_task(model, data, config)
            adapted_state_dicts.append(a_dict)

        # Update the parameters of meta-learner
        # Compute losses with adapted parameters along with corresponding tasks
        # Updated the parameters of meta-learner using sum of the losses
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            a_dict = adapted_state_dicts[i_batch]

            data = batch[i_batch]
            one_shot, _, target = data
            target_x, target_y = target

            y_meta = target_y
            x_meta = [one_shot, target]

            loss_t, logits_t, _, _ = model(x_meta, a_dict)

            meta_loss += loss_t
            meta_logits.append(logits_t)
            meta_label.extend(y_meta)

        meta_loss /= float(len(batch))
        # print(meta_loss)
        meta_acc, meta_f1 = evaluate(torch.cat(meta_logits, dim=0), torch.LongTensor(meta_label))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        total_loss.append(meta_loss.item())
        total_acc.append(meta_acc.item())
        total_f1.append(meta_f1.item())

    total_loss = np.array(total_loss)
    total_acc = np.array(total_acc)
    total_f1 = np.array(total_f1)

    return np.mean(total_loss), np.mean(total_acc), np.mean(total_f1)


def eval_epoch(model, data_loader, config):
    model.eval()
    total_loss, total_auc = [], []
    total_results_acc, total_results_f1 = [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, support, target = data

            net_clone = copy.deepcopy(model)
            optim = torch.optim.Adam(net_clone.parameters(), lr=config.lr)
            for _ in range(config.num_eval_updates):
                loss_first, _, _, _ = net_clone([one_shot, support])

                optim.zero_grad()
                loss_first.backward()
                optim.step()

            loss_second, logits_t, _, _ = net_clone([one_shot, target])

            meta_loss += loss_second.item()
            meta_logits.append(logits_t)
            meta_label.extend(target[1])

        meta_loss /= float(len(batch))
        acc, f1 = evaluate(torch.cat(meta_logits, dim=0), torch.LongTensor(meta_label))

        total_loss.append(meta_loss)

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)
    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))
    return np.mean(total_loss), total_results_acc, total_results_f1


def eval_epoch_tune(model, data_loader, config):
    total_all_features, total_all_labels = [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, support, target = data

            net_clone = copy.deepcopy(model)
            optim = torch.optim.Adam(net_clone.parameters(), lr=config.lr)
            for _ in range(config.num_eval_updates):
                loss_first, _, _, _ = net_clone([one_shot, support])

                optim.zero_grad()
                loss_first.backward()
                optim.step()

            net_clone.eval()
            features, labels = net_clone.predict([one_shot, target])

            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def eval_epoch_feature(model, data_loader, config):
    total_all_features, total_all_labels = [], []
    model.eval()
    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, _, target = data
            features, labels = model.predict([one_shot, target])

            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def train(model, optimizer, train_data, valid_data, test_data, config):
    max_score = 0
    max_test_score = 0
    no_imprv = 0
    max_results = []

    if config.dataset == "acd":
        dev_data_loader = data_loader(valid_data, config.dev_epochs, config)
        test_data_loader = data_loader(test_data, config.test_epochs, config)
    else:
        dev_data_loader = data_loader_huffpost(valid_data, config.dev_epochs, config)
        test_data_loader = data_loader_huffpost(test_data, config.test_epochs, config)
    for epoch_i in range(config.nepochs):
        if config.dataset == "acd":
            train_data_loader = data_loader(train_data, config.train_epochs, config)
        else:
            train_data_loader = data_loader_huffpost(train_data, config.train_epochs, config)

        print("----------Epoch {:} out of {:}---------".format(epoch_i + 1, config.nepochs))
        print("training evaluation..........")
        train_loss, train_acc, train_f1 = train_epoch(model, train_data_loader, optimizer, config)
        train_msg = "train_loss: {:04.5f} train_acc: {:04.5f} train_f1: {:04.5f} ".format(train_loss, train_acc,
                                                                                          train_f1)
        print(train_msg)

        print("deving evaluation...........")
        dev_loss, dev_acc, dev_f1 = eval_epoch(model, dev_data_loader, config)
        dev_msg = "dev_loss: {:04.6f} dev_acc: {:04.6f} dev_f1: {:04.6f} ".format(dev_loss, dev_acc, dev_f1)
        print(dev_msg)

        print("testing evaluation...........")
        test_loss, test_acc, test_f1 = eval_epoch(model, test_data_loader, config)
        test_msg = "test_loss: {:04.6f} test_acc: {:04.6f} test_f1: {:04.6f} ".format(test_loss, test_acc, test_f1)
        print(test_msg)

        model_name = "results_" + str(config.seed) + "/" + "model.pkl"
        if dev_f1 >= max_score:
            no_imprv = 0
            torch.save(model.state_dict(), model_name)
            saved_model = copy.deepcopy(model)
            print("new best score!! The checkpoint file has been updated")
            max_score = dev_f1
            max_test_score = test_acc
            max_results = test_f1
        else:
            no_imprv += 1
            if no_imprv >= config.nepoch_no_imprv:
                break
        print("max score: {:04.4f}----max test acc: {:04.4f}".format(max_score, max_test_score))
        print("acc " + str(max_test_score) + " f1 " + str(max_results))

    """
    model_name = "results_"+str(config.seed)+"/" + "model.pkl"
    torch.save(model.state_dict(), model_name)
    saved_model = copy.deepcopy(model)
    """
    print("-------------- getting features -----------")
    if config.dataset == "acd":
        loader = data_loader(test_data, config.test_feature_epochs, config)
    else:
        loader = data_loader_huffpost(test_data, config.test_feature_epochs, config)
    features, labels = eval_epoch_feature(saved_model, test_data_loader, config)
    pickle.dump([features, labels], open('BERT_features_before_maml.p', 'wb'))
    features_tune, labels_tune = eval_epoch_tune(saved_model, loader, config)
    pickle.dump([features_tune, labels_tune], open('BERT_features_maml.p', "wb"))
    print(len(features), len(features_tune))
    print("done!")


def main():
    dir_output = "results_" + str(config.seed) + "/"
    path_log = dir_output

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    path = config.data_path + "data_info_bert.p"

    train_data, valid_data, test_data = pickle.load(open(path, "rb"))

    if config.model == "bert_meta":
        model = BERT_Meta(config)

    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    train(model, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    #seed_list = [25, 5, 10, 15, 20]
    #for seed in seed_list:
    seed = 25
    config.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()