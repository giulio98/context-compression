import math
from abc import ABC
from typing import Tuple

import torch
from accelerate.logging import get_logger
from tqdm import tqdm
import numpy as np

from ..logging_utils import log_wandb_table
from .base_qa_predictor import ModelQAPredictor

logger = get_logger(__name__)


class LanguageModelingQAPredictor(ModelQAPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset)

    def post_processing_fn(
        self,
        predictions: Tuple[np.ndarray, np.ndarray]
    ):
        # predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        # decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        example_id_to_index = {k: i for i, k in enumerate(self.eval_examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(self.eval_dataset)}
        pred = {}
        for example_index, example in enumerate(tqdm(self.eval_examples)):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            pred[example["id"]] = predictions[feature_index]
        return pred

    def modify_for_clm(self, input_ids, labels, attention_mask):
        batch_size = input_ids.size(0)
        max_length = input_ids.size(1)
        padded_input_ids_list = []
        padded_attention_mask_list = []

        for i in range(batch_size):
            input_ids_example = input_ids[i]
            labels_example = labels[i]
            attention_mask_example = attention_mask[i]
            keep_mask = labels_example == -100
            truncated_input_ids = input_ids_example[keep_mask]
            truncated_attention_mask = attention_mask_example[keep_mask]
            padding_length = max_length - truncated_input_ids.size(0)
            padding_tensor = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=input_ids_example.dtype,
                                        device=input_ids_example.device)
            padding_tensor_mask = torch.zeros((padding_length,), dtype=attention_mask_example.dtype,
                                              device=attention_mask_example.device)
            padded_input_ids = torch.cat([padding_tensor, truncated_input_ids], dim=0)
            padded_attention_mask = torch.cat([padding_tensor_mask, truncated_attention_mask], dim=0)
            padded_input_ids_list.append(padded_input_ids)
            padded_attention_mask_list.append(padded_attention_mask)
        padded_input_ids_batch = torch.stack(padded_input_ids_list)
        padded_attention_mask_batch = torch.stack(padded_attention_mask_list)

        return padded_input_ids_batch, padded_attention_mask_batch

    def predict(self, accelerator, model, dataloader):
        gen_kwargs = {
            "max_new_tokens": self.predictor_config.max_answer_length,
            "num_beams": self.predictor_config.num_beams,
            "do_sample": self.predictor_config.do_sample,
            "temperature": self.predictor_config.temperature,
            "top_k": self.predictor_config.top_k,
            "top_p": self.predictor_config.top_p,
            "repetition_penalty": self.predictor_config.repetition_penalty
        }
        model.eval()
        # losses = []
        predictions = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            with torch.no_grad():
                # Assuming the batch contains input_ids for generative models
                # outputs = model(batch["input_ids"], batch["attention_mask"])
                # loss = outputs.loss
                # losses.append(
                #     accelerator.gather_for_metrics(loss.repeat(self.predictor_config.batch_size)))
                # Generate the output sequence
                if self.predictor_config.task_type == 'CLM':
                    # print("----------------------------before")
                    # print(self.tokenizer.decode(batch["input_ids"][0]))
                    # input_ids_for_generation, attention_mask_for_generation = self.modify_for_clm(
                    #     batch["input_ids"],
                    #     batch["labels"],
                    #     batch["attention_mask"])
                    # print(batch["labels"])
                    # print(self.tokenizer.decode(batch["labels"][0]))
                    input_ids_for_generation = batch["input_ids"][:, batch["labels"][0] == -100]
                    input_ids_len = input_ids_for_generation.size(1)
                    attention_mask_for_generation = torch.ones((1, input_ids_for_generation.size(1)), dtype=torch.long, device=batch["attention_mask"].device)
                    batch["input_ids"] = input_ids_for_generation
                    batch["attention_mask"] = attention_mask_for_generation
                # print("------------------ input ids")
                # print(self.tokenizer.decode(batch["input_ids"][0]))
                # print("-------------------------------")
                # print(self.tokenizer.decode(batch["context_ids"][0]))
                # print("-------------------------------------")
                # print(self.tokenizer.decode(batch["question_ids"][0]))
                generated_ids = accelerator.unwrap_model(model).generate(
                    **batch,
                    **gen_kwargs
                )
                # print("-------------------------------------")
                # print(self.tokenizer.decode(generated_ids[0]))
                # print("-------------------------------------")
                if self.predictor_config.task_type == 'CLM' and generated_ids.size(1) > input_ids_len:
                    generated_ids = generated_ids[:, input_ids_len:, ...]
                generated_ids = accelerator.pad_across_processes(
                    generated_ids, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                generated_ids = accelerator.gather(generated_ids).cpu().numpy()
                predictions.append(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
                # print(predictions)

        # losses = torch.cat(losses)
        # try:
        #     eval_loss = torch.mean(losses)
        #     perplexity = math.exp(eval_loss)
        # except OverflowError:
        #     eval_loss = torch.mean(losses)
        #     perplexity = float("inf")

        predictions_concat = np.concatenate(predictions, axis=0)
        predictions, references = self.post_processing(predictions_concat)
        log_wandb_table(accelerator, predictions[:10], references[:10])
        references = [
            {k: v for k, v in ex.items() if k not in ["question", "context"]} for ex in references
        ]
        predict_metric = self.metric.compute(predictions=predictions, references=references)
        results = {
            self.metric_name: predict_metric,
            # "perplexity": perplexity,
            # "eval_loss": eval_loss,
        }

        return results
