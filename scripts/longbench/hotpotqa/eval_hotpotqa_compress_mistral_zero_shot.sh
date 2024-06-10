PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_hotpotqa=evaluate_mistral_compress_zeroshot_qa_hotpot \
  models.target_token=512,1000,2000 \
  models.split_size=2048 \
  models.condition="question","context","all" \
  models.normalize=True,False