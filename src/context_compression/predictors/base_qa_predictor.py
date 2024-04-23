from abc import ABC, abstractmethod
from .base_predictor import ModelPredictor

class ModelQAPredictor(ModelPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset)

    @abstractmethod
    def post_processing_fn(self, predictions):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the post_processing_fn method.")

    def post_processing(self, predictions):
        predictions = self.post_processing_fn(
            predictions=predictions
        )
        if self.predictor_config.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v.lstrip(), "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v.lstrip()} for k, v in predictions.items()]

        references = [{"id": ex["id"], "context": ex["context"], "question": ex["question"], "answers": ex["answers"]} for ex in self.eval_examples]

        return formatted_predictions, references
