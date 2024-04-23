PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_squad_2=evaluate_llmlingua_compress_zeroshot_qa_squad \
  models.target_token=256
