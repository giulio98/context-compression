import abc
import math
import os
from functools import partial
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from huggingface_hub import create_repo, Repository
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger

logger = get_logger(__name__)



class Trainer(abc.ABC):
    """Class for training a model."""

    def __init__(
        self,
        mode: str,
        training_config: DictConfig,
        optimization_config: DictConfig,
        optimizer,
        evaluation_config: DictConfig,
        logging_config: DictConfig
    ) -> None:
        self.mode = mode
        self.logging_config = logging_config
        if self.mode in ["train", "train_eval"]:
            self.training_config = training_config
            self.optimization_config = optimization_config
            self.optimizer = optimizer
        if self.mode in ["eval", "train_eval"]:
            self.evaluation_config = evaluation_config

    # noinspection PyTypeChecker
    def train(self, accelerator, model, tokenizer, ds_train_obj, ds_val_obj, predictor_config) -> None:
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here, and it will by default pick up all supported trackers
        # in the environment
        repo = None
        # Handle the repository creation
        if accelerator.is_main_process:
            if self.training_config.push_to_hub:
                # Retrieve of infer repo_name
                repo_name = self.training_config.hub_model_id
                if repo_name is None:
                    repo_name = Path(self.training_config.output_dir).absolute().name
                # Create repo and retrieve repo_id
                repo_id = create_repo(repo_name, exist_ok=True, token=self.training_config.hub_token).repo_id
                # Clone repo locally
                repo = Repository(self.training_config.output_dir, clone_from=repo_id, token=self.training_config.hub_token)


                with open(os.path.join(self.training_config.output_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            elif self.training_config.output_dir is not None:
                os.makedirs(self.training_config.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        ds_train = ds_train_obj.load()
        ds_val = ds_val_obj.load()
        ds_train = ds_train.filter(ds_train_obj.filter)
        ds_val = ds_val.filter(ds_val_obj.filter)

        with accelerator.main_process_first():
            ds_train_data = ds_train.map(
                ds_train_obj.tokenize,
                batched=True,
                num_proc=1,
                remove_columns=ds_train_obj.column_names,
                desc="Running tokenizer on train dataset"
            )
            ds_val_data = ds_val.map(
                ds_val_obj.tokenize,
                batched=True,
                num_proc=1,
                remove_columns=ds_val_obj.column_names,
                desc="Running tokenizer on validation dataset"
            )
        predictor = hydra.utils.instantiate(predictor_config, tokenizer=tokenizer, eval_examples=ds_val, eval_dataset=ds_val_data, _recursive_=False)

        # train_batch_size = per_gpu_train_batch_size * accelerator.state.num_processes
        train_dataloader = DataLoader(ds_train_data, shuffle=True, batch_size=self.training_config.per_device_train_batch_size, collate_fn=ds_train_obj.get_data_collator())
        val_dataloader = DataLoader(ds_val_data.remove_columns(ds_val_obj.columns_to_remove_for_model), batch_size=self.training_config.per_device_val_batch_size, collate_fn=ds_val_obj.get_data_collator())
        # for param in model.encoder.parameters():
        #     param.requires_grad = False

        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimization_config.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer_class = hydra.utils.get_class(self.optimizer._target_)
        optimizer_cfg = {k: v for k, v in self.optimizer.items() if k != "_target_"}
        partial_optimizer = partial(optimizer_class, **optimizer_cfg)
        optimizer = partial_optimizer(params=optimizer_grouped_parameters)

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.training_config.gradient_accumulation_steps)
        if self.training_config.total_steps is None:
            total_steps = self.training_config.num_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        else:
            total_steps = self.training_config.total_steps

        scheduler = get_scheduler(
            name=self.optimization_config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.optimization_config.num_warmup_steps * self.training_config.gradient_accumulation_steps,
            num_training_steps=total_steps * self.training_config.gradient_accumulation_steps
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.training_config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            total_steps = self.training_config.num_epochs * num_update_steps_per_epoch
        # We recalculate our number of training epochs
        self.training_config.num_epochs = math.ceil(total_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        if self.training_config.checkpointing_steps is not None and self.training_config.checkpointing_steps.isdigit():
            checkpointing_steps = int(self.training_config.checkpointing_steps)
        else:
            checkpointing_steps = self.training_config.checkpointing_steps

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if self.training_config.with_tracking:
            experiment_config = {**self.training_config, **self.optimization_config}
            accelerator.init_trackers(project_name=self.logging_config.project_name, config=experiment_config, init_kwargs={"wandb": {"entity": self.logging_config.entity_name, "name": os.path.basename(self.logging_config.name)}})

        # Train!
        total_batch_size = self.training_config.per_device_train_batch_size * accelerator.num_processes * self.training_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(ds_train_data)}")
        logger.info(f"  Num Epochs = {self.training_config.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.training_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.training_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")

        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.training_config.resume_from_checkpoint:
            if self.training_config.resume_from_checkpoint is not None or self.training_config.resume_from_checkpoint != "":
                checkpoint_path = self.training_config.resume_from_checkpoint
                path = os.path.basename(self.training_config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]
            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.training_config.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // self.training_config.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)
        else:
            resume_step = None

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)
        for epoch in range(starting_epoch, self.training_config.num_epochs):
            model.train()
            if self.training_config.with_tracking:
                total_loss = 0
            else:
                total_loss = None
            if self.training_config.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if self.training_config.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        logger.info("***** Running validation *****")
                        results = predictor.predict(accelerator, model, val_dataloader)
                        if self.training_config.with_tracking:
                            results["train_loss"] = total_loss / len(train_dataloader)
                            results["epoch"] = epoch
                            results["step"] = completed_steps
                            accelerator.log(
                                results,
                                step=completed_steps,
                            )
                        model.train()
                        output_dir = f"step_{completed_steps}"
                        if self.training_config.output_dir is not None:
                            output_dir = os.path.join(self.training_config.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if self.training_config.with_tracking:
                    lr = optimizer.param_groups[0]["lr"]
                    accelerator.log(
                        {
                         "lr": lr,
                         "loss_per_step/train": loss.item()
                         },
                        step=completed_steps,
                    )
                if completed_steps >= total_steps:
                    break
            logger.info("***** Running validation at epoch finished *****")
            results = predictor.predict(accelerator, model, val_dataloader)
            if self.training_config.with_tracking:
                results["train_loss"] = total_loss / len(train_dataloader)
                results["epoch"] = epoch
                results["step"] = completed_steps
                accelerator.log(
                    results,
                    step=completed_steps,
                )

            if self.training_config.push_to_hub and epoch < self.training_config.num_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    self.training_config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(self.training_config.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

            if checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if output_dir is not None:
                    output_dir = os.path.join(output_dir, output_dir)
                accelerator.save_state(output_dir)

        if self.training_config.with_tracking:
            accelerator.end_training()

        if self.training_config.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                self.training_config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(self.training_config.output_dir)
                if self.training_config.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    # noinspection PyTypeChecker
    def evaluate(self, accelerator, model, tokenizer, ds_eval_obj, predictor_config):
        """
        Evaluate the model on the validation set using the provided predictor.

        Args:
            model: The model to evaluate.
            tokenizer: tokenizer
            ds_eval_obj: DataLoader for the evaluation set.
            predictor_config: Object responsible for prediction logic.

        Returns:
            A dictionary containing evaluation metrics.
            :param accelerator:
        """
        # if model.get_input_embeddings().num_embeddings != len(tokenizer):
        #     model.resize_token_embeddings(len(tokenizer))
        if self.evaluation_config.seed is not None:
            set_seed(self.evaluation_config.seed)
        # if model.get_input_embeddings().num_embeddings <= len(tokenizer):
        #     logger.info("Resizing model token embeddings")
        #     model.resize_token_embeddings(len(tokenizer))

        ds_eval = ds_eval_obj.load()
        ds_eval = ds_eval.filter(ds_eval_obj.filter)
        with accelerator.main_process_first():
            ds_eval_data = ds_eval.map(
                ds_eval_obj.tokenize,
                batched=True,
                num_proc=1,
                remove_columns=ds_eval_obj.column_names,
                desc="Running tokenizer on evaluation dataset"
            )
        predictor = hydra.utils.instantiate(predictor_config, tokenizer=tokenizer, eval_examples=ds_eval, eval_dataset=ds_eval_data)
        eval_dataloader = DataLoader(ds_eval_data.remove_columns(ds_eval_obj.columns_to_remove_for_model), batch_size=self.evaluation_config.per_device_eval_batch_size, collate_fn = ds_eval_obj.get_data_collator())

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        if not self.evaluation_config.zero_shot:
            if self.evaluation_config.resume_from_checkpoint is not None or self.evaluation_config.resume_from_checkpoint != "":
                checkpoint_path = self.training_config.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
        if self.evaluation_config.with_tracking:
            experiment_config = {**self.evaluation_config}
            accelerator.init_trackers(project_name=self.logging_config.project_name, config=experiment_config, init_kwargs={"wandb": {"entity": self.logging_config.entity_name, "name": os.path.basename(self.logging_config.name)}})
        # Use the predictor to get the evaluation results
        results = predictor.predict(accelerator, model, eval_dataloader)
        if self.evaluation_config.with_tracking:
            accelerator.log(results)
            accelerator.end_training()


