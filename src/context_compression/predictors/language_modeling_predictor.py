import math
from abc import ABC
from typing import Tuple

import torch
import wandb
from accelerate.logging import get_logger
from tqdm import tqdm
import numpy as np

from ..logging_utils import log_wandb_table, log_predictions_as_csv
from .base_qa_predictor import ModelQAPredictor
from ..metrics.longbench_metrics import compute_longbench_metric

logger = get_logger(__name__)


class LanguageModelingQAPredictor(ModelQAPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset, data_config)


    def post_processing_fn(
        self,
        predictions: Tuple[np.ndarray, np.ndarray]
    ):
        example_id_to_index = {k: i for i, k in enumerate(self.eval_examples[self.data_config.id_column])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(self.eval_dataset)}
        pred = {}
        for example_index, example in enumerate(tqdm(self.eval_examples)):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            pred[example[self.data_config.id_column]] = predictions[feature_index]
        return pred

    def predict(self, accelerator, model, dataloader):

        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
            wandb_tracker.define_metric("processing_time", summary="mean")
            wandb_tracker.define_metric("generation_time", summary="mean")
            wandb_tracker.define_metric("context_size_mean", summary="mean")
            wandb_tracker.define_metric("context_size_min", summary="min")
            wandb_tracker.define_metric("context_size_max", summary="max")
            wandb_tracker.define_metric("target_token_mean", summary="mean")
            wandb_tracker.define_metric("target_token_min", summary="min")
            wandb_tracker.define_metric("target_token_max", summary="max")
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
        losses = []
        predictions = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            with torch.no_grad():
                if self.compute_perplexity:
                    outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss
                    losses.append(
                        accelerator.gather_for_metrics(loss.repeat(self.predictor_config.batch_size)))
                if self.predictor_config.task_type == 'CLM':
                    input_ids_for_generation = batch["input_ids"][:, batch["labels"][0] == -100]
                    input_ids_len = input_ids_for_generation.size(1)
                    attention_mask_for_generation = torch.ones((1, input_ids_for_generation.size(1)), dtype=torch.long, device=batch["attention_mask"].device)
                    batch["input_ids"] = input_ids_for_generation
                    batch["attention_mask"] = attention_mask_for_generation
                generated_ids = accelerator.unwrap_model(model).generate(
                    accelerator=accelerator,
                    **batch,
                    **gen_kwargs
                )
                if self.predictor_config.task_type == 'CLM' and generated_ids.size(1) > input_ids_len:
                    generated_ids = generated_ids[:, input_ids_len:, ...]
                generated_ids = accelerator.pad_across_processes(
                    generated_ids, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                generated_ids = accelerator.gather(generated_ids).cpu().numpy()
                predictions.append(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        predictions_concat = np.concatenate(predictions, axis=0)
        predictions, references = self.post_processing(predictions_concat)
        # log_wandb_table(accelerator, predictions, references, self.data_config)
        # log_predictions_as_csv(predictions, references, self.predictor_config.output_file_path, self.data_config)
        #references = [
        #    {k: v for k, v in ex.items() if k not in [self.data_config.question_column, self.data_config.context_column]} for ex in references
        #]
        if "squad" in self.metric_name or "rouge" in self.metric_name:
            predict_metric = self.metric.compute(predictions=predictions, references=references)
        else:
            predict_metric = compute_longbench_metric(self.predictor_config.metric_name, predictions, references)
        results = {
            self.metric_name: predict_metric,
        }

        if self.compute_perplexity:
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                eval_loss = torch.mean(losses)
                perplexity = float("inf")
            results["perplexity"] = perplexity
            results["eval_loss"] = eval_loss

        return results
