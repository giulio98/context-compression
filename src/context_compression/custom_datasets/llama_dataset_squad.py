from abc import ABC

from .causal_dataset_squad import CausalModelingDataset
from transformers import PreTrainedTokenizer


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
DEFAULT_SYSTEM_PROMPT = f"""Answer the question based solely on the given context. If the provided context allows for an answer, provide a concise response; otherwise, return a single blank space ' '. No inferences, guesses, or explanations."""

class LlamaDataset(CausalModelingDataset, ABC):

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(split, data_config, tokenizer, model)
        self.column_names = None
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length
        self.columns_to_remove_for_model = ["example_id", "offset_mapping"]
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    @staticmethod
    def generate_input(_question, _context):
        prompt = " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])
        return f"{BOS}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{(prompt.strip())} {E_INST}"
