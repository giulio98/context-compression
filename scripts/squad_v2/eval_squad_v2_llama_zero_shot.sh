PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu_fp16.yaml \
  src/context_compression/run.py +experiments_squad_2=evaluate_llama_zeroshot_qa_squad
