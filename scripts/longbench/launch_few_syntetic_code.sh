#!/bin/sh
echo "Testing script execution"
chmod +x ./scripts/longbench/trec/eval_trec_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/samsum/eval_samsum_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/passage_count/eval_passage_count_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/passage_retrieval/eval_passage_retrieval_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/lcc/eval_lcc_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/repobench/eval_repobench_compress_llama_zero_shot.sh


sh ./scripts/longbench/trec/eval_trec_compress_llama_zero_shot.sh
sh ./scripts/longbench/samsum/eval_samsum_compress_llama_zero_shot.sh
sh ./scripts/longbench/passage_count/eval_passage_count_compress_llama_zero_shot.sh
sh ./scripts/longbench/passage_retrieval/eval_passage_retrieval_compress_llama_zero_shot.sh
sh ./scripts/longbench/lcc/eval_lcc_compress_llama_zero_shot.sh
sh ./scripts/longbench/repobench/eval_repobench_compress_llama_zero_shot.sh
