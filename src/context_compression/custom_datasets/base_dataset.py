from abc import ABC, abstractmethod

from datasets import load_dataset
from transformers import PreTrainedTokenizer, default_data_collator, DataCollatorWithPadding


class BaseDataset(ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer, model, split: str, data_config):
        self.data_config = data_config
        self.column_names = None
        self.split = split
        self.tokenizer = tokenizer
        self.model = model
        self.columns_to_remove_for_model = []

    def filter(self, example):
        return True

    def load(self):
        # Load the dataset
        if self.data_config.dataset_name is not None:
            print(f"Loading dataset {self.data_config.dataset_name} with config {self.data_config.dataset_config_name}")
            raw_datasets = load_dataset(self.data_config.dataset_name, self.data_config.dataset_config_name)
            print("Loaded!")
        else:
            data_files = {}
            if self.data_config.train_file is not None:
                data_files["train"] = self.data_config.train_file
            if self.data_config.validation_file is not None:
                data_files["validation"] = self.data_config.validation_file
            if self.data_config.test_file is not None:
                data_files["test"] = self.data_config.test_file
            extension = self.data_config.train_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, field="data")
        self.column_names = raw_datasets[self.split].column_names

        return raw_datasets[self.split]

    def get_data_collator(self):
        if self.data_config.pad_to_max_length:
            return default_data_collator
        else:
            return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.data_config.use_fp16 else None))


    @abstractmethod
    def tokenize(self, examples):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the tokenize method.")
