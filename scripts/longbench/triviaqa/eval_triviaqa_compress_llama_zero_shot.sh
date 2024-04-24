PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_triviaqa=evaluate_llama_compress_zeroshot_qa_triviaqa \
  models.target_token=512 \
  models.split_size=128
