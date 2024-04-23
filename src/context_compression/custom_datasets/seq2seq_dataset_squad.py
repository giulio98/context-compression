from abc import ABC

from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq, default_data_collator

from .base_dataset import BaseDataset


class Seq2SeqModelingDataset(BaseDataset, ABC):

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(tokenizer, model, split, data_config)
        self.column_names = None
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length
        self.columns_to_remove_for_model = ["example_id", "offset_mapping"]

    @staticmethod
    def generate_input(_question, _context):
        return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

    def filter(self, example):
        question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        question = example[question_column_name]
        context = example[context_column_name]
        input_example = self.generate_input(question, context)
        return len(self.tokenizer.encode(input_example)) <= 512

    def tokenize(self, examples):
        # Tokenize the examples
        # For language modeling, we might only be interested in the text for both training and evaluation
        question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
        questions = examples[question_column_name]
        contexts = examples[context_column_name]
        answers = examples[answer_column_name]

        inputs = [self.generate_input(question, context) for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        if self.split == "train":
            tokenized_examples = self.tokenizer(
                inputs,
                max_length=self.max_seq_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation=True
            )
        else:
            tokenized_examples = self.tokenizer(
                inputs,
                max_length=self.max_seq_length,
                padding="max_length" if self.pad_to_max_length else False,
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
        tokenized_targets = self.tokenizer(
            text_target=targets,
            max_length=self.max_answer_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )
        if self.pad_to_max_length:
            tokenized_targets["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in tokenized_targets["input_ids"]
            ]
        if self.split == "train":
            tokenized_examples["labels"] = tokenized_targets["input_ids"]
        else:
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
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
                labels_out.append(tokenized_targets["input_ids"][sample_index])

            tokenized_examples["labels"] = labels_out
        return tokenized_examples

    def get_data_collator(self):
        if self.data_config.pad_to_max_length:
            return default_data_collator
        else:
            label_pad_token_id = -100
            return DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.data_config.use_fp16 else None,
            )
