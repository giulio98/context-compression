PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_summarization=evaluate_llama_compress_zeroshot_qmsum \
  models.target_token=512 \
  models.split_size=64,256,1024,2048 \
  models.condition="question" \
  models.normalize=True
