from abc import ABC

from .causal_dataset_squad import CausalModelingDataset
from transformers import PreTrainedTokenizer


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

class LlamaDatasetCustom(CausalModelingDataset, ABC):

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(split, data_config, tokenizer, model)
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length
        self.max_question_length = data_config.question_max_length
        self.max_context_length = data_config.context_max_length
        self.columns_to_remove_for_model = ["example_id", "offset_mapping"]
        self.question_column = data_config.question_column  # e.g. 'question'
        self.context_column = data_config.context_column  # e.g. 'context'
        self.answer_column = data_config.answer_column  # e.g. 'answers'
        self.id_column = data_config.id_column  # e.g. 'id'
        self.system_prompt = data_config.prompt
        self.question_prompt = data_config.question_prompt
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate_input(self, _question, _context):
        prompt = " ".join(["question:", _question.lstrip()])
        return f"{BOS}{B_INST} {B_SYS}{self.system_prompt}{E_SYS} {prompt.strip()} {E_INST}"

    @staticmethod
    def generate_question(_question):
        question = " ".join(["question:", _question.lstrip()])
        return question

    @staticmethod
    def generate_context(_context):
        context = " ".join(["context:", _context.lstrip()])
        return context

    def filter(self, example):
        context = example['context']
        input_example = self.generate_context(context)
        return len(self.tokenizer.encode(input_example)) <= self.max_context_length

    @staticmethod
    def extract_targets(answers):
        if all(isinstance(answer, dict) and 'text' in answer for answer in answers):
            # Handling the structure where the answer is in the form {'text': ..., ...}
            return [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        elif all(isinstance(answer[0], str) for answer in answers):
            # Handling the structure where the answer is a string
            return [answer[0] if len(answer[0]) > 0 else "" for answer in answers]
        else:
            raise ValueError("The structure of the answers field is not recognized.")


    def tokenize(self, examples):
        if self.question_prompt is not None:
            questions = [self.question_prompt] * len(examples[self.context_column])
        else:
            questions = examples[self.question_column]

        contexts = examples[self.context_column]
        answers = examples[self.answer_column]
        targets = self.extract_targets(answers)

        inputs = [self.generate_input(question, context) for question, context in zip(questions, contexts)]
        if self.question_prompt is not None:
            questions = [self.question_prompt] * len(contexts)
        else:
            questions = [self.generate_question(question) for question in questions]

        max_length = self.max_seq_length + self.max_answer_length
        contexts = [self.generate_context(context) for context in contexts]
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
        tokenized_questions = self.tokenizer(
            questions,
            add_special_tokens=False,
            max_length=self.max_question_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True

        )
        tokenized_contexts = self.tokenizer(
            contexts,
            add_special_tokens=False,
            max_length=self.max_context_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
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
                tokenized_examples["example_id"].append(examples[self.id_column][sample_index])
                labels_out.append(labels["input_ids"][sample_index])

            tokenized_examples["labels"] = labels_out
        tokenized_examples["question_ids"] = []
        tokenized_examples["question_attention_mask"] = []
        tokenized_examples["context_ids"] = []
        tokenized_examples["context_attention_mask"] = []
        tokenized_examples["question_ids"] = tokenized_questions["input_ids"]
        tokenized_examples["question_attention_mask"] = tokenized_questions["attention_mask"]
        tokenized_examples["context_ids"] = tokenized_contexts["input_ids"]
        tokenized_examples["context_attention_mask"] = tokenized_contexts["attention_mask"]
        return tokenized_examples

