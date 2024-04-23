import time
from typing import Optional
from llmlingua import PromptCompressor
from transformers import LlamaForCausalLM
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaConfig
import torch
from transformers import AutoTokenizer



logger = logging.get_logger(__name__)

class LlamaForCompressedCausalLMLingua(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, tokenizer_name, split_size, target_token):  # changed by GC
        super().__init__(config)
        self.split_size = split_size  # added by GC
        self.segment_length = self.split_size
        self.target_token = target_token  # added by GC
        # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.compressor = PromptCompressor("microsoft/phi-2")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name, use_fast=True, trust_remote_code=True, padding_side="left")

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
        prompt = self.tokenizer.decode(context_ids[0])
        question = self.tokenizer.decode(question_ids[0])

        accelerator.log({
            "context_size_mean": context_ids.size(1),
            "context_size_min": context_ids.size(1),
            "context_size_max": context_ids.size(1)
        })
        start_processing_time = time.time()
        compressed_prompt = self.compressor.compress_prompt(context=prompt,
                                                            instruction="",
                                                            question=question,
                                                            target_token=self.target_token,
                                                            iterative_size=self.split_size,
                                                            rank_method="longllmlingua",
                                                            condition_compare=True,
                                                            condition_in_question="after_condition",
                                                            reorder_context="sort",
                                                            dynamic_context_compression_ratio=0.25,
                                                            context_budget="+300")
        end_processing_time = time.time()

        compressed_context = self.tokenizer(compressed_prompt["compressed_prompt"], return_tensors="pt").to(context_ids.device)

        context_ids_len = compressed_context["input_ids"].size(1)

        accelerator.log({
               "target_token_mean": context_ids_len,
               "target_token_min": context_ids_len,
               "target_token_max": context_ids_len
        })
        start_generation_time = time.time()
        generate_kwargs['attention_mask'] = torch.cat([compressed_context["attention_mask"], attention_mask], dim=-1)
        model_output =  super().generate(input_ids=torch.cat([compressed_context["input_ids"], input_ids], dim=1),
                                use_cache=True, **generate_kwargs)[:, context_ids_len:, ...]
        end_generation_time = time.time()
        accelerator.log({"processing_time": end_processing_time - start_processing_time,
           "generation_time": end_generation_time - start_generation_time}
        )
        return model_output

