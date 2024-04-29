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
import inspect
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.cache_utils import Cache
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
from einops import rearrange, repeat
from flash_attn.flash_attn_interface import _get_block_size_n
import torch.nn.functional as F

class ModifiedMistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0001

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output, attn_weights = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
            output_attentions=output_attentions
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def construct_local_mask(
        self,
        seqlen_q,
        seqlen_k,
        window_size=(-1, -1),  # -1 means infinite window size
        query_padding_mask=None,
        key_padding_mask=None,
        device=None,
    ):
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )
    
    
    def convert_flash_attn_S_to_softmax(
        self,
        S,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        head_dim,
        is_dropout,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
    ):
        """FlashAttention stores the S matrix in a different way.
        Arguments:
            S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
            query_padding_mask: (batch_size, seqlen_q_rounded)
            key_padding_mask: (batch_size, seqlen_k_rounded)
        """
        if causal:
            window_size = (window_size[0], 0)
        seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
        S_converted = S
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                S.device,
            )
            local_mask = F.pad(
                local_mask,
                (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
                value=True,
            )
            S_converted = S_converted.masked_fill(local_mask, 0.0)

        # Need to zero out things not in attention_mask in case S was initialized with random values
        # and some of those values aren't overwritten.
        seqlen_q_og = (
            query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
        )
        if query_padding_mask is not None:
            query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
            S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
        seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
            S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
        return S_converted[:, :, :seqlen_q, :seqlen_k]
    
    def normalize_flash_attn_S(
        self,
        attn_unnorm,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        is_dropout=False,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k, v: (batch_size, seqlen_k, nheads, head_dim)
            key_padding_mask: (batch_size, seqlen_q)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        Output:
            softmax_lse: (batch_size, nheads, seqlen_q)
            softmax_max: (batch_size, nheads, seqlen_q)
        """
        if causal:
            window_size = (window_size[0], 0)
        q, k, v = q.float(), k.float(), v.float()
        _, seqlen_q, _, head_dim = q.shape
        seqlen_k = k.shape[1]
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
        if key_padding_mask is not None:
            scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias.to(dtype=scores.dtype)
        block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
        scores_block = scores.split(block_size_n, dim=-1)
        lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
        lse = torch.logsumexp(lse_block, dim=-1)
        # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
        # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
        lse[lse == float("-inf")] = float("inf")
        scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
        cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
        attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
        attn_norm = torch.cat(
            [
                a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )
        if query_padding_mask is not None:
            attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
        return attn_norm.to(dtype=attn_unnorm.dtype)

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
        output_attentions=False
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad, _, attn_weights  = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_attn_prob=True
                )
            else:
                attn_output_unpad, _, attn_weights = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                    return_attn_prob=True
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            if key_states.size(1) != attention_mask.shape[-1]:
                attention_mask_num_tokens = attention_mask.shape[-1]
                key_padding_mask  = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
            else:
                key_padding_mask = attention_mask
            query_padding_mask = key_padding_mask[:, -query_length:]
            S_dmask_converted = self.convert_flash_attn_S_to_softmax(
                attn_weights,
                query_states.size(1),
                key_states.size(1),
                query_padding_mask,
                key_padding_mask,
                query_states.size(3),
                dropout > 0.0,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, 1),
            )
            attn_unnorm = S_dmask_converted.abs()
            k_rep = repeat(key_states, "b s h d -> b s (h g) d", g=query_states.size(2) // key_states.size(2))
            v_rep = repeat(value_states, "b s h d -> b s (h g) d", g=query_states.size(2) // key_states.size(2))
            attn_weights = self.normalize_flash_attn_S(
                attn_unnorm,
                query_states,
                k_rep,
                v_rep,
                query_padding_mask,
                key_padding_mask,
                None,
                dropout > 0.0,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, 1),
            )
            
        else:
            if not use_sliding_windows:
                
                attn_output, _, attn_weights = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_attn_probs=True
                )
            else:
                attn_output, _, attn_weights = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                    return_attn_probs=True
                )
            S_dmask_converted = self.convert_flash_attn_S_to_softmax(
                S=attn_weights,
                seqlen_q=query_states.size(1),
                seqlen_k=key_states.size(1),
                query_padding_mask=None,
                key_padding_mask=None,
                head_dim=query_states.size(3),
                is_dropout=dropout > 0.0,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, 1),
            )
            attn_unnorm = S_dmask_converted.abs()
            k_rep = repeat(key_states, "b s h d -> b s (h g) d", g=query_states.size(2) // key_states.size(2))
            v_rep = repeat(value_states, "b s h d -> b s (h g) d", g=query_states.size(2) // key_states.size(2))
            attn_weights = self.normalize_flash_attn_S(
                attn_unnorm,
                query_states,
                k_rep,
                v_rep,
                None,
                None,
                None,
                dropout > 0.0,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else (-1, 1),
            )
        return attn_output, attn_weights

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
transformers.models.mistral.modeling_mistral.MistralFlashAttention2 = ModifiedMistralFlashAttention2

from transformers.models.mistral.modeling_mistral import MistralAttention, MistralSdpaAttention

MODIFIED_MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention_2": ModifiedMistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}
transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MODIFIED_MISTRAL_ATTENTION_CLASSES

from transformers import MistralForCausalLM
from transformers.utils import logging
from transformers.models.mistral.modeling_mistral import MistralConfig
logger = logging.get_logger(__name__)

class MistralForCompressedCausalLM(MistralForCausalLM):
    def __init__(self, config: MistralConfig, mode, compression_factor, split_size, target_token, distance_metric=None):  # changed by GC
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


