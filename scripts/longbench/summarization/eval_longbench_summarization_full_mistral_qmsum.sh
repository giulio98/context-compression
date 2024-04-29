PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_summarization=evaluate_mistral_compress_zeroshot_qmsum  \
  trainers.logging_config.name="full_mistral" \
  models.target_token=32768 \
  custom_datasets.test.data_config.context_max_length=32128
