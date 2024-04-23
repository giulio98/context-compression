from abc import ABC, abstractmethod
from .base_predictor import ModelPredictor

class ModelQAPredictor(ModelPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset, data_config)

    @abstractmethod
    def post_processing_fn(self, predictions):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the post_processing_fn method.")


    def _format_references(self, example):
        # Check if 'answers' is already in the expected format
        if isinstance(example['answers'], dict) and 'text' in example['answers']:
            if "squad" in self.predictor_config.metric_name:
                return {
                    'id': example[self.data_config.id_column],
                    'answers': example['answers'],
                    #self.data_config.question_column: example[self.data_config.question_column],
                    #self.data_config.context_column: example[self.data_config.context_column],
                }
            elif "classification_score" in self.predictor_config.metric_name:
                return {
                    'answers': example['answers'],
                    'all_classes': example['all_classes']
                    }
            elif "rouge" in self.predictor_config.metric_name:
                return example['answers']
            else:
                return  {
                    'answers': example['answers'],
                }

        # Convert 'answers' to the expected format if it's just a list of answers
        elif isinstance(example['answers'], list):
            if "squad" in self.predictor_config.metric_name:
                return {
                    'id': example[self.data_config.id_column],
                    'answers': {'text': example['answers'], 'answer_start': [0] * len(example['answers'])},
                    #self.data_config.question_column: example[self.data_config.question_column],
                    #self.data_config.context_column: example[self.data_config.context_column],
                }
            elif "classification_score" in self.predictor_config.metric_name:
                return {
                    'answers': example['answers'],
                    'all_classes': example['all_classes']
                    }
            elif "rouge" in self.predictor_config.metric_name:
                return example['answers']
            else:
                return  {
                    'answers': example['answers'],
                }
        else:
            raise ValueError("Unexpected format of 'answers' in the dataset.")

    def post_processing(self, predictions):
        predictions = self.post_processing_fn(
            predictions=predictions
        )
        if "squad" in self.predictor_config.metric_name:
            if self.predictor_config.version_2_with_negative:
                formatted_predictions = [
                    {"id": k, "prediction_text": v.lstrip(), "no_answer_probability": 0.0} for k, v in predictions.items()
                ]
            else:
                formatted_predictions = [{"id": k, "prediction_text": v.lstrip()} for k, v in predictions.items()]

            references = [self._format_references(ex) for ex in self.eval_examples]

        else:
            formatted_predictions = [v.strip() for k, v in predictions.items()]
            references = [self._format_references(ex) for ex in self.eval_examples]

        return formatted_predictions, references
