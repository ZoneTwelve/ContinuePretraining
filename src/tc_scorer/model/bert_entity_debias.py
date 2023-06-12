import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput)


class MLP(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            #layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class CnnExtractor(nn.Module):

    def __init__(self, feature_kernel, input_size):
        super(CnnExtractor, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(input_size, feature_num, kernel)
            for kernel, feature_num in feature_kernel.items()
        ])
        input_shape = sum(
            [feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class EndefHead(nn.Module):

    def __init__(self, dropout, num_labels, embed_size):
        super().__init__()

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        self.ent_cnn = CnnExtractor(feature_kernel, embed_size)
        mlp_input_shape = sum(
            [feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_cls = nn.Sequential(
            self.ent_cnn,
            MLP(mlp_input_shape, [embed_size // 2], dropout, 'none'),
            nn.LazyLinear(num_labels))

    def forward(self, encoder, entity, entity_mask, **kwargs):
        ent_feats = encoder(
            input_ids=entity,
            attention_mask=entity_mask,
        ).last_hidden_state
        entity_logits = self.entity_cls(ent_feats)
        return entity_logits


class BertStyleScorer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config, False)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"): 
                    #or name.startswith('encoder.layer.10') \

                #or name.startswith('encoder.layer.9'): \
                # or name.startswith('encoder.layer.8') \
                # or name.startswith('encoder.layer.7') \
                # or name.startswith('encoder.layer.6')\
                # or name.startswith('encoder.layer.5') \
                # or name.startswith('encoder.layer.4')\
                # or name.startswith('encoder.layer.3'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.cls_head = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LeakyReLU(), nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_labels))
        self.endef_head = EndefHead(classifier_dropout, num_labels=config.num_labels, embed_size=config.hidden_size)
        
        self.post_init()

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        entity_ids: Tensor = None,
        entity_mask: Tensor = None,
        position_ids: Tensor = None,
        head_mask: Tensor = None,
        inputs_embeds: Tensor = None,
        labels: Optional[Tensor] = None,
        encoder_hidden_states: Tensor = None,
        encoder_attention_mask: Tensor = None,
        past_key_values: List[FloatTensor] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if attention_mask is None:
            x = last_hidden_state.mean(1)
        else:
            x = (last_hidden_state * attention_mask.unsqueeze(-1)
                 ).sum(1) / attention_mask.sum(1).unsqueeze(-1)

        logits = self.cls_head(x)

        if labels is not None:
            ent_logits = self.endef_head(self.bert, entity_ids, entity_mask)
            logits = 0.9 * logits + 0.1 * ent_logits
            if self.num_labels == 1:
                labels = labels.float()
                ent_loss = F.binary_cross_entropy_with_logits(ent_logits.squeeze(-1), labels)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels) + .2 * ent_loss
            else:
                ent_loss = F.cross_entropy(ent_logits, labels)
                loss = F.cross_entropy(logits, labels) + .2 * ent_loss
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )