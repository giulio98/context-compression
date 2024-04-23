PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/generative_modeling/run.py --multirun +experiments_longbench_wikimqa=evaluate_llama_compress_zeroshot_qa_wikimqa \
  models.target_token=512,1000,2000 \
  models.split_size=128,256,512