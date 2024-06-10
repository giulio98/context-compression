import logging
import os
import datasets
import hydra
import omegaconf
import transformers
from accelerate import Accelerator
from omegaconf import DictConfig
import torch

from nn_core.common import PROJECT_ROOT

# Force the execution of __init__.py if this file is executed directly.
import context_compression # noqa

from accelerate.logging import get_logger

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param}")


def run(cfg: DictConfig):
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    accelerator_log_kwargs = {}
    if not cfg.trainers.mode == "eval":
        if cfg.trainers.training_config.with_tracking:
            accelerator_log_kwargs["log_with"] = cfg.trainers.training_config.report_to
            accelerator_log_kwargs["project_dir"] = cfg.trainers.training_config.output_dir
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.trainers.training_config.gradient_accumulation_steps,
            **accelerator_log_kwargs)
    else:
        if cfg.trainers.evaluation_config.with_tracking:
            accelerator_log_kwargs["log_with"] = cfg.trainers.evaluation_config.report_to
            accelerator_log_kwargs["project_dir"] = cfg.trainers.evaluation_config.output_dir
        accelerator = Accelerator(**accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()
    # Instantiate tokenizer and datasets
    logger.info(f"Instantiating <{cfg.tokenizers['_target_']}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizers, _recursive_=False)

    # Instantiate model
    logger.info(f"Instantiating <{cfg.models['_target_']}>")
    model = hydra.utils.instantiate(cfg.models)
    if model.dtype == torch.float32:
        model = model.half()
    if not cfg.trainers.mode == "eval":
        print_trainable_parameters(model=model)

    # Instantiate trainer
    logger.info(f"Instantiating <{cfg.trainers['_target_']}>")
    trainer = hydra.utils.instantiate(cfg.trainers, _recursive_=False)

    if not cfg.trainers.mode == "eval":
        logger.info(f"Instantiating <{cfg.custom_datasets.train['_target_']}>")
        ds_train_obj = hydra.utils.instantiate(cfg.custom_datasets.train, tokenizer=tokenizer, model=model, _recursive_=False)
        logger.info(f"Instantiating <{cfg.custom_datasets.validation['_target_']}>")
        ds_valid_obj = hydra.utils.instantiate(cfg.custom_datasets.validation, tokenizer=tokenizer, model=model, _recursive_=False)
        logger.info(f"Instantiating <{cfg.predictors['_target_']}>")
        predictor_config = cfg.predictors
        logger.info("Starting training!")
        trainer.train(accelerator=accelerator, model=model, tokenizer=tokenizer, ds_train_obj=ds_train_obj, ds_val_obj=ds_valid_obj, predictor_config=predictor_config)
        logger.info("Training finished!")

    if cfg.trainers.mode in ["train_eval", "eval"]:
        # Instantiate the metric
        logger.info(f"Instantiating <{cfg.custom_datasets.test['_target_']}>")
        ds_eval_obj = hydra.utils.instantiate(cfg.custom_datasets.test, tokenizer=tokenizer, model=model, _recursive_=False)
        logger.info(f"Instantiating <{cfg.predictors['_target_']}>")
        predictor_config = cfg.predictors

        logger.info("Starting testing!")
        trainer.evaluate(accelerator=accelerator, model=model, tokenizer=tokenizer, ds_eval_obj=ds_eval_obj, predictor_config=predictor_config)
        logger.info("Testing finished!")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    """Run the main function."""
    run(cfg)


if __name__ == "__main__":
    main()
