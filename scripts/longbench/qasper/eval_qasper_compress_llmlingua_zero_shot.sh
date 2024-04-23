for target_token in 512 1000 2000; do
PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/generative_modeling/run.py --multirun +experiments_longbench_qasper=evaluate_llmlingua_compress_zeroshot_qa_qasper \
  models.target_token=${target_token}
done
