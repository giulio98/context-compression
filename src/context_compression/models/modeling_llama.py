import math
import time
from typing import Optional, Tuple, Dict, Any
import torch
from transformers.cache_utils import DynamicCache


class ModifiedDynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.cos_sin_cache = []

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    @staticmethod
    def _rerotate_cos_sin(cos, sin, important_pos_batch):
        original_dtype = cos.dtype
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        batch_size, seq_length = important_pos_batch.shape
        idx = torch.arange(seq_length, device=important_pos_batch.device)
        idx = idx[None, :]
        original_cos = cos[important_pos_batch, :]
        shifted_cos = cos[idx, :]
        original_sin = sin[important_pos_batch, :]
        shifted_sin = sin[idx, :]
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        # SIGN BEGIN
        less_than_mask = (important_pos_batch < idx)
        rerotation_sin = torch.where(less_than_mask[:, :, None],
                                     original_sin * shifted_cos - original_cos * shifted_sin,
                                     - (original_sin * shifted_cos - original_cos * shifted_sin)
                                     )
        # END SIGN
        # rerotation_sin = - (original_sin * shifted_cos - original_cos * shifted_sin)
        same_pos_mask = (important_pos_batch == idx)
        rerotation_cos[same_pos_mask] = 1
        rerotation_sin[same_pos_mask] = 0
        new_cos = rerotation_cos.to(original_dtype)
        new_sin = rerotation_sin.to(original_dtype)

        return new_cos, new_sin

    @staticmethod
    def gather_important_tokens(states, indices):
        return torch.gather(states, 2,
                            indices.unsqueeze(1).unsqueeze(-1).expand(-1, states.size(1), -1, states.size(3)))

    def update_rope(self, layer_index, key_states, important_pos):
        seq_length = key_states.shape[-2]
        new_cos, new_sin = self._rerotate_cos_sin(self.cos_sin_cache[layer_index]["cos"][:seq_length],
                                                  self.cos_sin_cache[layer_index]["sin"][:seq_length], important_pos)
        self.key_cache[layer_index] = self._apply_key_rotary_pos_emb(
            self.gather_important_tokens(self.key_cache[layer_index], important_pos),
            new_cos,
            new_sin
        )
        self.value_cache[layer_index] = self.gather_important_tokens(self.value_cache[layer_index], important_pos)
        # self.cos_sin_cache[layer_index]["cos"] = new_cos
        # self.cos_sin_cache[layer_index]["sin"] = new_sin
        self.seen_tokens = important_pos.size(1)
        return self.key_cache[layer_index], self.value_cache[layer_index]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        if cache_kwargs is not None:
            sin = cache_kwargs.get("sin")
            cos = cache_kwargs.get("cos")
        else:
            sin = None
            cos = None
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            if sin is not None and cos is not None:
                self.cos_sin_cache.append({"sin": sin, "cos": cos})
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if sin is not None and cos is not None:
                self.cos_sin_cache[layer_idx] = {"sin": sin, "cos": cos}
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


import transformers.cache_utils

transformers.cache_utils.DynamicCache = ModifiedDynamicCache

from transformers import LlamaForCausalLM
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaConfig
logger = logging.get_logger(__name__)

class LlamaForCompressedCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, mode, compression_factor, split_size, target_token, distance_metric=None):  # changed by GC
        config._attn_implementation = "eager"
        super().__init__(config)
        self.compression_factor = compression_factor  # added by GC
        self.split_size = split_size  # added by GC
        self.segment_length = self.split_size
        self.mode = mode
        self.target_token = target_token
        if distance_metric == "euclidean":
            self.p = 2
        elif distance_metric == "manhattan":
            self.p = 1
        elif distance_metric == "minkowski":
            self.p = 3
        else:
            self.p = 0




    def generate(
        self,
        accelerator,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        question_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,
        context_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):

        past_key_values = ModifiedDynamicCache()
        accelerator.log({
            "context_size_mean": context_ids.size(1),
            "context_size_min": context_ids.size(1),
            "context_size_max": context_ids.size(1)
        })
        if context_ids.size(1) > self.target_token:
            start_processing_time = time.time()
            self.compression_factor = int(math.ceil(context_ids.size(1) / self.target_token))
            context_ids_list = torch.split(context_ids, self.segment_length, dim=1)
            context_attention_mask_list = torch.split(context_attention_mask, self.segment_length, dim=1)
            past_attention_mask = torch.zeros(context_attention_mask.size(0), 0, dtype=context_attention_mask.dtype, device=context_attention_mask.device)
            for step, (segment_context_ids, segment_attention_mask) in enumerate(zip(context_ids_list, context_attention_mask_list)):
                segment_attention_mask = torch.cat([past_attention_mask, segment_attention_mask], dim=1)
                past_cache_len = past_key_values.seen_tokens
                current_ids = torch.cat([segment_context_ids, question_ids], dim=1)
                current_attention_mask = torch.cat([segment_attention_mask, question_attention_mask], dim=1)
                position_ids = (current_attention_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(current_attention_mask == 0, 1)  # can be filled with anything >= 0
                position_ids = position_ids[:, -current_ids.shape[1]:]
                with torch.no_grad():
                    output_question_aware = self.model(
                        input_ids=current_ids,
                        attention_mask=current_attention_mask,
                        position_ids=position_ids,
                        output_attentions=True,
                        # output_hidden_states=True,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                current_seq_length = segment_context_ids.size(1)
                k = int(current_seq_length // self.compression_factor) + past_cache_len
                # BEGIN DYNAMIC K
                # max_k = int(current_seq_length // self.compression_factor) + past_cache_len
                # min_k = past_cache_len

                # context_layer_embeddings = output_question_aware.hidden_states[-1][:,
                #                                 :current_seq_length + past_cache_len]
                # question_layer_embedding = output_question_aware.hidden_states[-1][:, current_seq_length:].mean(dim=1)
                # cosine_sim = torch.nn.functional.cosine_similarity(context_layer_embeddings,
                #                                                      question_layer_embedding.unsqueeze(1), dim=-1)
                # mean_cosine_sim = cosine_sim.mean()
                # norm_mean_cosine_sim = (mean_cosine_sim + 1) / 2
                # k = int(min_k + (max_k - min_k) * norm_mean_cosine_sim)
                # END DYNAMIC K
                for layer_idx, layer_attention in enumerate(output_question_aware.attentions):
                    if self.mode == "attention_score":
                        summed_attention = layer_attention.sum(dim=1)
                        context_attention = summed_attention[:, current_seq_length:, :current_seq_length + past_cache_len]
                        # BEGIN
                        tot_seq_len = summed_attention.size(2)
                        non_zero_counts = torch.arange(1, tot_seq_len + 1, device=context_attention.device)
                        non_zero_counts = non_zero_counts[current_seq_length + past_cache_len:]
                        normalization_factors = non_zero_counts.float() / tot_seq_len
                        context_attention = context_attention * normalization_factors[None, :, None]
                        # END
                        # non_zero_mask = context_attention != 0
                        # non_zero_counts = non_zero_mask.sum(dim=2)
                        # normalization_factors = non_zero_counts / context_attention.size(2)
                        # context_attention = context_attention * normalization_factors[:, :, None]
                        aggregated_attention = context_attention.sum(dim=1)
                        _, important_tokens = torch.topk(aggregated_attention, k=k, dim=-1, largest=True)
                    elif self.mode == "cosine_similarity":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        cosine_sim = torch.nn.functional.cosine_similarity(context_layer_embeddings,
                                                                            question_layer_embedding.unsqueeze(1), dim=-1)
                        _, important_tokens = torch.topk(cosine_sim, k=k, dim=-1, largest=True)
                    elif self.mode == "knn":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        distances = torch.cdist(context_layer_embeddings.to(dtype=torch.double),
                                                question_layer_embedding.unsqueeze(1).to(dtype=torch.double), p=self.p).squeeze(
                            -1)
                        _, important_tokens = torch.topk(distances, k=k, dim=-1, largest=False)


                    elif self.mode == "svd":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        important_tokens = torch.empty((context_layer_embeddings.shape[0], k),
                                                        device=context_layer_embeddings.device, dtype=torch.long)
                        for batch_idx, batch_context_layer_embeddings in enumerate(context_layer_embeddings):
                            # Add question embedding to each context embedding and perform PCA
                            combined_embeddings = batch_context_layer_embeddings + question_layer_embedding[batch_idx]
                            u, s, v = torch.pca_lowrank(combined_embeddings.to(dtype=torch.double), center=True, q=k + 2)

                            # Take the absolute values of the first column of u, sort and select top k
                            _, indices = torch.abs(u[:, 0]).sort(descending=True)
                            important_tokens[batch_idx] = indices[:k]
                    important_tokens, _ = torch.sort(important_tokens, dim=-1, descending=False)
                    past_key_values.update_rope(layer_idx, past_key_values[layer_idx][0][:, :, :current_seq_length + past_cache_len],
                                                            important_tokens)

                past_attention_mask = torch.ones(segment_attention_mask.size(0), k, device=segment_attention_mask.device, dtype=segment_attention_mask.dtype)
            end_processing_time = time.time()
            accelerator.log({
                    "target_token_mean": past_key_values.seen_tokens,
                    "target_token_min": past_key_values.seen_tokens,
                    "target_token_max": past_key_values.seen_tokens
            })
            start_generation_time = time.time()
            generate_kwargs['attention_mask'] = torch.cat([past_attention_mask, attention_mask], dim=-1)
            model_output = super().generate(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, **generate_kwargs)
            end_generation_time = time.time()
            accelerator.log({"processing_time": end_processing_time - start_processing_time,
                        "generation_time": end_generation_time - start_generation_time}
                        )
            return model_output
        else:
            # context_ids_len = context_ids.size(1)
            start_processing_time = time.time()
            with torch.no_grad():
                self.model(
                    input_ids=context_ids,
                    attention_mask=context_attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            end_processing_time = time.time()
            accelerator.log({
                "target_token_mean": past_key_values.seen_tokens,
                "target_token_min": past_key_values.seen_tokens,
                "target_token_max": past_key_values.seen_tokens
            })
            start_generation_time = time.time()
            generate_kwargs['attention_mask'] = torch.cat([context_attention_mask, attention_mask], dim=-1)
            model_output = super().generate(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, **generate_kwargs) # [:, context_ids_len:, ...]
            end_generation_time = time.time()
            accelerator.log({"processing_time": end_processing_time - start_processing_time,
                        "generation_time": end_generation_time - start_generation_time}
                        )
            return model_output


