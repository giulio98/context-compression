import copy
import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from dataclasses import dataclass

from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Block


# BEGIN Change
@dataclass
class Seq2SeqLMOutputWithCompressedIds(Seq2SeqLMOutput):
    """
    Base class for sequence-to-sequence language models outputs with additional
    field for important indices.

    Args:
        compressed_input_ids (`torch.LongTensor` of shape `(batch_size, num_important_indices)`, *optional*):
            Indices of the important tokens selected after compression.
    """
    compressed_input_ids: Optional[torch.LongTensor] = None


# END Change

class T5ForCompressedConditionalGeneration(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, compression_factor, split_size):  # changed by GC
        super().__init__(config)
        self.compression_factor = compression_factor  # added by GC
        self.split_size = split_size  # added by GC
        self.segment_length = self.split_size

    # BEGIN Change: GC
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        question_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,
        **generate_kwargs
    ):
        encoder_outputs_questions = self.encoder(
            input_ids=question_ids,
            attention_mask=question_attention_mask,
            return_dict=True
        )
        hidden_states_questions = encoder_outputs_questions[0]
        generate_kwargs['attention_mask'] = attention_mask
        encoder_outputs = None
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        bsz = inputs_embeds.size(0)
        emb_size = inputs_embeds.size(2)
        device = inputs_embeds.device
        attn_dtype = attention_mask.dtype
        inputs_embeds_list = torch.split(inputs_embeds, self.segment_length, dim=1)
        attention_mask_list = torch.split(attention_mask, self.segment_length, dim=1)
        softprompt = inputs_embeds[:, :0, :]
        compressed_hidden_states_list = []
        for step, (segment_embeds, segment_attention_mask) in enumerate(
            zip(inputs_embeds_list, attention_mask_list)):
            segment_embeds = torch.cat([softprompt, segment_embeds], dim=1)
            segment_attention_mask = torch.cat([
                torch.ones(bsz, softprompt.size(1), device=device, dtype=attn_dtype),
                segment_attention_mask
            ], dim=1)
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=segment_attention_mask,
                inputs_embeds=segment_embeds,
                return_dict=True,
            )
            hidden_states = encoder_outputs[0]

            seq_length = hidden_states[:, softprompt.size(1):, :].size(1)
            k = int(seq_length // self.compression_factor)
            compressed_hidden_states = torch.empty(bsz, k, emb_size, device=device)
            compressed_input_ids = torch.empty(bsz, k, dtype=torch.long, device=device)
            for j in range(bsz):
                compressed_batch = hidden_states[:, softprompt.size(1):, :][j]
                u, s, v = torch.pca_lowrank(compressed_batch, center=True, q=k + 2)
                important_indices = u[:, 0].abs().argsort(descending=True)[:k].to(dtype=torch.long)
                compressed_hidden_states[j] = compressed_batch[important_indices, :]
            if softprompt.size(1) == 0:
                softprompt = compressed_hidden_states
            else:
                softprompt = compressed_hidden_states + hidden_states[:, :softprompt.size(1), :]
            compressed_hidden_states_list.append(compressed_hidden_states)
        hidden_states = torch.cat(compressed_hidden_states_list, dim=1)
        attention_mask = torch.ones(bsz, hidden_states.size(1), device=device, dtype=attn_dtype)
        hidden_states = torch.cat((hidden_states_questions, hidden_states), dim=1)

        attention_mask = torch.cat((question_attention_mask, attention_mask), dim=1)
        encoder_outputs.last_hidden_state = hidden_states
        generate_kwargs['attention_mask'] = attention_mask

        return super().generate(encoder_outputs=encoder_outputs, **generate_kwargs)

    # END Change

    # noinspection PyTypeChecker
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        question_ids: Optional[torch.LongTensor] = None,  # added by GC
        attention_mask: Optional[torch.FloatTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,  # added by GC
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_outputs_questions: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # added by GC
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        __HEAD_MASK_WARNING_MSG = """
        The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
        `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
        If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
        num_heads)`.
        """
        compressed_input_ids = None
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            if encoder_outputs_questions is None:
                encoder_outputs_questions = self.encoder(
                    input_ids=question_ids,
                    attention_mask=question_attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            hidden_states_questions = encoder_outputs_questions[0]
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            # inputs_embeds = self.get_input_embeddings()(input_ids)
            bsz = inputs_embeds.size(0)
            emb_size = inputs_embeds.size(2)
            device = inputs_embeds.device
            attn_dtype = attention_mask.dtype
            inputs_embeds_list = torch.split(inputs_embeds, self.segment_length, dim=1)
            attention_mask_list = torch.split(attention_mask, self.segment_length, dim=1)
            softprompt = inputs_embeds[:, :0, :]
            compressed_hidden_states_list = []
            for step, (segment_embeds, segment_attention_mask) in enumerate(
                zip(inputs_embeds_list, attention_mask_list)):
                segment_embeds = torch.cat([softprompt, segment_embeds], dim=1)
                segment_attention_mask = torch.cat([
                    torch.ones(bsz, softprompt.size(1), device=device, dtype=attn_dtype),
                    segment_attention_mask
                ], dim=1)
                encoder_outputs = self.encoder(
                    input_ids=None,
                    attention_mask=segment_attention_mask,
                    inputs_embeds=segment_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = encoder_outputs[0]

                seq_length = hidden_states[:,softprompt.size(1):,:].size(1)
                k = int(seq_length // self.compression_factor)
                compressed_hidden_states = torch.empty(bsz, k, emb_size, device=device)
                compressed_input_ids = torch.empty(bsz, k, dtype=torch.long, device=device)
                for j in range(bsz):
                    compressed_batch = hidden_states[:,softprompt.size(1):,:][j]
                    u, s, v = torch.pca_lowrank(compressed_batch, center=True, q=k + 2)
                    important_indices = u[:, 0].abs().argsort(descending=True)[:k].to(dtype=torch.long)
                    compressed_hidden_states[j] = compressed_batch[important_indices, :]
                softprompt = compressed_hidden_states
                compressed_hidden_states_list.append(compressed_hidden_states)
            hidden_states = torch.cat(compressed_hidden_states_list, dim=1)
            attention_mask = torch.ones(bsz, hidden_states.size(1), device=device, dtype=attn_dtype)
            hidden_states = torch.cat((hidden_states_questions, hidden_states), dim=1)

            attention_mask = torch.cat((question_attention_mask, attention_mask), dim=1)
            encoder_outputs.last_hidden_state = hidden_states
            # END Change
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputWithCompressedIds(  # added by GC
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            compressed_input_ids=compressed_input_ids  # added by GC
        )
