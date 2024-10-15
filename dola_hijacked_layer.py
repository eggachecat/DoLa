import argparse
import os
import math
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)


class DoLAHijackedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.weighting_method = 'linear' # self_attention
        self.layers_to_be_focused = list(range(33))

        # 如果选择 self_attention，则需要定义 query 和 key 参数
        if self.weighting_method == 'linear':
            self.focuse_attention_layer = torch.nn.Linear(len(self.layers_to_be_focused), 1, bias=False)
        else:
            self.query = torch.nn.Parameter(torch.randn(config.hidden_size))
            self.key_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def compute_weighted_hidden_states(self, stacked_hidden_states):
        """
        根据配置选择的加权方式生成对不同层的权重。
        可以扩展为使用线性、注意力等不同方法。
        """
        if self.weighting_method == 'linear':
            # weights = torch.softmax(self.focuse_attention_layer.weight, dim=1)  # (num_layers, 1)
            weights = self.focuse_attention_layer.weight
            weights = weights.view(1, len(self.layers_to_be_focused), 1, 1)  # 调整形状以便广播
            weighted_hidden_states = (stacked_hidden_states * weights).sum(dim=1)  # 沿着 num_layers 维度进行加权求和
            return weighted_hidden_states

        elif self.weighting_method == 'self_attention':
            # 使用自注意力机制来生成权重
            # 对 stacked_hidden_states 应用线性变换生成 Q, K, V
            # stacked_hidden_states: (batch_size, num_layers, seq_len, hidden_size)
            query = self.query_layer(stacked_hidden_states)  # (batch_size, num_layers, seq_len, hidden_size)
            key = self.key_layer(stacked_hidden_states)  # (batch_size, num_layers, seq_len, hidden_size)
            value = self.value_layer(stacked_hidden_states)  # (batch_size, num_layers, seq_len, hidden_size)

            # 计算注意力分数，形状为 (batch_size, num_layers, seq_len, seq_len)
            attention_scores = torch.einsum('bnqd,bnkd->bnqk', query, key) * self.attention_scale

            # 对注意力分数进行 softmax，得到归一化的注意力权重
            attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_layers, seq_len, seq_len)

            # 使用注意力权重对 value 进行加权求和，得到形状 (batch_size, num_layers, seq_len, hidden_size)
            weighted_hidden_states = torch.einsum('bnqk,bnvd->bnqd', attention_weights, value)

            return weighted_hidden_states
        
        raise ValueError(f"Unknown weighting method: {self.weighting_method}")


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # with torch.no_grad():
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )


        # 提取指定层的隐藏状态，形状为 (batch_size, seq_len, hidden_size)
        hidden_states_list = [outputs["hidden_states"][idx] for idx in self.layers_to_be_focused]

        # 将这些隐藏状态堆叠在一起，形状为 (batch_size, num_layers, seq_len, hidden_size)
        stacked_hidden_states = torch.stack(hidden_states_list, dim=1)
        weighted_hidden_states = self.compute_weighted_hidden_states(stacked_hidden_states)

        hidden_states = weighted_hidden_states
        # hidden_states = torch.rand_like(outputs[0], dtype=torch.bfloat16,  requires_grad=False)
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # if labels is None and not is_torchdynamo_compiling():
            #     print(
            #         "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            #     )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
AutoModelForCausalLM._model_mapping.register(key=LlamaConfig, value=DoLAHijackedLlamaForCausalLM, exist_ok=True)
print("update!!!!!!")