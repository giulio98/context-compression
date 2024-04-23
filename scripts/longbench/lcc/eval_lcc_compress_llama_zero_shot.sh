PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/generative_modeling/run.py --multirun +experiments_longbench_code=evaluate_llama_compress_zeroshot_lcc \
  models.target_token=2000 \
  models.split_size=256,512
