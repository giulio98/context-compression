from abc import ABC, abstractmethod
import evaluate

class ModelPredictor(ABC):
    """
    Abstract base class for model predictions.

    This class provides a generic interface for model predictions. It requires
    the implementation of the predict and post_processing methods.
    """

    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        """
        Initialize the ModelPredictor.

        Parameters:
        - model (torch.nn.Module): The model to be used for predictions.
        - accelerator: An accelerator object for distributed computing.
        - args: A namespace or an object containing the relevant parameters and settings.
        """
        self.predictor_config = predictor_config
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.metric_name = predictor_config.metric_name
        self.compute_perplexity = predictor_config.compute_perplexity
        self.data_config = data_config
        if "squad" in self.metric_name or "rouge" in self.metric_name:
            self.metric = evaluate.load(self.metric_name)

    @abstractmethod
    def predict(self, accelerator, model, dataloader):
        """
        Generate predictions for the given dataloader.

        This method should be overridden by subclasses to implement
        task-specific prediction logic.

        Parameters:
        - dataloader (torch.utils.data.DataLoader): The dataloader containing the data to predict on.

        Returns:
        - A list or a tuple containing the raw prediction results.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the predict method.")

    @abstractmethod
    def post_processing(self, predictions):
        """
        Post-process the raw predictions to convert them into a more usable format.

        This method should be overridden by subclasses to implement
        task-specific post-processing logic.

        Parameters:
        - predictions: The raw predictions output by the predict method.

        Returns:
        - The post-processed prediction results.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the post_processing method.")
