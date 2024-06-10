PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_code=evaluate_llama_compress_zeroshot_repobench \
  models.target_token=512 \
  models.split_size=64,256,1024,2048 \
  models.condition="question" \
  models.normalize=True
