from abc import ABC

from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling, default_data_collator

from .base_dataset import BaseDataset

from accelerate.logging import get_logger
logger = get_logger(__name__)

class CausalModelingDataset(BaseDataset, ABC):
    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(tokenizer, model, split, data_config)
        # Initialize column_names here if not initialized in BaseDataset
        self.column_names = data_config.column_names if hasattr(data_config, 'column_names') else ['question', 'context', 'answers']
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length

    @staticmethod
    def generate_input(_question, _context):
        return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip(), "answer:"])

    def filter(self, example):
        # Assuming column_names are correctly initialized
        question = example['question']
        context = example['context']
        answers = example['answers']
        answer = answers["text"][0] if len(answers["text"]) > 0 else ""
        input_example = self.generate_input(question, context) + " " + answer
        return len(self.tokenizer.encode(input_example)) <= self.max_seq_length + self.max_answer_length

    def tokenize(self, examples):
        # Tokenize the

        questions = examples['question']
        contexts = examples['context']
        answers = examples['answers']
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]

        inputs = [self.generate_input(question, context) for question, context in zip(questions, contexts)]
        max_length = self.max_seq_length + self.max_answer_length - 1
        if self.split == "train":
            tokenized_examples = self.tokenizer(
                inputs,
                targets,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation="only_first"
            )
        else:
            tokenized_examples = self.tokenizer(
                inputs,
                targets,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation="only_first",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
        labels = self.tokenizer(targets, add_special_tokens=False)
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):
            tokenized_examples["input_ids"][idx] = input_ids + [self.tokenizer.eos_token_id]
            tokenized_examples["attention_mask"][idx] = tokenized_examples["attention_mask"][idx] + [1]
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):
            label_input_ids = labels["input_ids"][idx] + [self.tokenizer.eos_token_id]
            labels["input_ids"][idx] = [-100] * (len(input_ids) - len(label_input_ids)) + label_input_ids
        if self.split == "train":
            tokenized_examples["labels"] = labels["input_ids"]
        else:
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id, and we will store the offset mappings.
            tokenized_examples["example_id"] = []
            # Augment the overflowing tokens to the labels
            labels_out = []

            for i in range(len(tokenized_examples["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])
                labels_out.append(labels["input_ids"][sample_index])

            tokenized_examples["labels"] = labels_out
        return tokenized_examples

    def get_data_collator(self):
        if self.data_config.pad_to_max_length:
            return default_data_collator
        else:
            return default_data_collator
