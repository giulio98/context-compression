for target_token in 512 1000 2000; do
PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/multi_gpu.yaml \
  src/context_compression/run.py --multirun +experiments_longbench_musique=evaluate_llmlingua_mistral_compress_zeroshot_qa_musique \
  models.target_token=${target_token}
done
