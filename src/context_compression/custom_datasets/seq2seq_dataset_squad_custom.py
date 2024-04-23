from abc import ABC

from transformers import PreTrainedTokenizer

from .seq2seq_dataset_squad import Seq2SeqModelingDataset


class Seq2SeqModelingCustomDataset(Seq2SeqModelingDataset, ABC):

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(split, data_config, tokenizer, model)
        self.question_max_length = data_config.question_max_length

    @staticmethod
    def generate_input(_question, _context):
        return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

    @staticmethod
    def generate_question(_question):
        return " ".join(["question:", _question.lstrip()])

    def tokenize(self, examples):
        question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
        questions = examples[question_column_name]
        contexts = examples[context_column_name]
        answers = examples[answer_column_name]

        inputs = [self.generate_input(question, context) for question, context in zip(questions, contexts)]
        questions = [self.generate_question(question) for question in questions]
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
        tokenized_questions = self.tokenizer(
            questions,
            max_length=self.question_max_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
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
        tokenized_examples["question_ids"] = []
        tokenized_examples["question_attention_mask"] = []
        tokenized_examples["question_ids"] = tokenized_questions["input_ids"]
        tokenized_examples["question_attention_mask"] = tokenized_questions["attention_mask"]
        return tokenized_examples
