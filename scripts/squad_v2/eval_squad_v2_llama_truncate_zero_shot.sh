PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_squad_2=evaluate_llama_compress_zeroshot_qa_squad \
  trainers.logging_config.name="truncate_llama" \
  models.target_token=4096 \
  custom_datasets.test.data_config.context_max_length=256,128,64,32 