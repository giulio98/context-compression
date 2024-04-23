from abc import ABC

import numpy as np
import torch
from tqdm import tqdm

from ..logging_utils import log_wandb_table
from .base_qa_predictor import ModelQAPredictor
import logging
from accelerate.logging import get_logger
import collections
import json
import os

logger = get_logger(__name__)


class QAModelPredictor(ModelQAPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset, data_config)

    @staticmethod
    def _create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have to create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # noinspection PyTypeChecker
    def post_processing_fn(self, predictions):
        """
        Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
        original contexts. This is the base postprocessing functions for models that only return start and end logits.

        Args:
            predictions: predictions
        """
        prefix = "eval"
        log_level = logging.WARNING
        if len(predictions) != 2:
            raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(self.eval_dataset):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(self.eval_dataset)} features.")

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(self.eval_examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(self.eval_dataset):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if self.predictor_config.version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(self.eval_examples)} example predictions split into {len(self.eval_dataset)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(self.eval_examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = self.eval_dataset[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = self.eval_dataset[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -self.predictor_config.n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -self.predictor_config.n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self.predictor_config.max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if self.predictor_config.version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:self.predictor_config.n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                self.predictor_config.version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

            # Compute the softmax of all scores (we do it with numpy to stay independent of torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not self.predictor_config.version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > self.predictor_config.null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if self.predictor_config.output_dir is not None:
            if not os.path.isdir(self.predictor_config.output_dir):
                raise EnvironmentError(f"{self.predictor_config.output_dir} is not a directory.")

            prediction_file = os.path.join(
                self.predictor_config.output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                self.predictor_config.output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
            )
            if self.predictor_config.version_2_with_negative:
                null_odds_file = os.path.join(
                    self.predictor_config.output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if self.predictor_config.version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions

    def predict(self, accelerator, model, dataloader):
        all_start_logits = []
        all_end_logits = []

        model.eval()

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not self.predictor_config.pad_to_max_length:
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        # Concatenate the logits from all batches
        start_logits_concat = np.concatenate(all_start_logits, axis=0)
        end_logits_concat = np.concatenate(all_end_logits, axis=0)
        predictions = (start_logits_concat, end_logits_concat)

        # Post-processing step
        predictions, references = self.post_processing(predictions)
        log_wandb_table(accelerator, predictions[:10], references[:10])
        predict_metric = self.metric.compute(predictions=predictions, references=[
            {k: v for k, v in ex.items() if k not in ["question", "context"]} for ex in references])
        results = {
            self.metric_name: predict_metric
        }
        return results

