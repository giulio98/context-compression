PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_multifieldqa=evaluate_llama_compress_zeroshot_qa_multifield  \
  trainers.logging_config.name="full_llama" \
  models.target_token=4096 \
  models.is_full=True \
  custom_datasets.test.data_config.context_max_length=3776
