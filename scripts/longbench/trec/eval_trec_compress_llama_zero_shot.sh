PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/generative_modeling/run.py --multirun +experiments_longbench_few_shot=evaluate_llama_compress_zeroshot_trec \
  models.target_token=512,1000,2000 \
  models.split_size=128,256,512
