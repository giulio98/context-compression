PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_few_shot=evaluate_llama_compress_zeroshot_samsum  \
  trainers.logging_config.name="full_llama" \
  models.target_token=4096 \
  custom_datasets.test.data_config.context_max_length=3336
