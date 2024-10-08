PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_syntetic=evaluate_mistral_compress_zeroshot_passage_count \
  trainers.logging_config.name="rag_mistral" \
  models.target_token=32768 \
  custom_datasets.test.data_config.context_max_length=512,1000,2000 \
  custom_datasets.test.data_config.use_rag=True

