PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_code=evaluate_llama_compress_zeroshot_repobench \
  trainers.logging_config.name="truncate_llama" \
  models.target_token=4096 \
  custom_datasets.test.data_config.context_max_length=512,1000,2000
