#!/bin/sh
echo "Testing script execution"
chmod +x ./scripts/longbench/narrativeqa/eval_narrativeqa_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/qasper/eval_qasper_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/multifieldqa/eval_multifieldqa_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/wikimqa/eval_wikimqa_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/hotpotqa/eval_hotpotqa_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/musique/eval_musiqueqa_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_llama_govreport.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_llama_qmsum.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_llama_multinews.sh
chmod +x ./scripts/longbench/trec/eval_trec_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/samsum/eval_samsum_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/passage_count/eval_passage_count_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/passage_retrieval/eval_passage_retrieval_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/lcc/eval_lcc_compress_llama_zero_shot.sh
chmod +x ./scripts/longbench/repobench/eval_repobench_compress_llama_zero_shot.sh

# sh ./scripts/longbench/narrativeqa/eval_narrativeqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/qasper/eval_qasper_compress_llama_zero_shot.sh
# sh ./scripts/longbench/multifieldqa/eval_multifieldqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/wikimqa/eval_wikimqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/hotpotqa/eval_hotpotqa_compress_llama_zero_shot.sh
sh ./scripts/longbench/musique/eval_musiqueqa_compress_llama_zero_shot.sh
sh ./scripts/longbench/summarization/eval_longbench_summarization_llama_govreport.sh
sh ./scripts/longbench/summarization/eval_longbench_summarization_llama_qmsum.sh
sh ./scripts/longbench/summarization/eval_longbench_summarization_llama_multinews.sh
sh ./scripts/longbench/trec/eval_trec_compress_llama_zero_shot.sh
# sh ./scripts/longbench/samsum/eval_samsum_compress_llama_zero_shot.sh
sh ./scripts/longbench/passage_count/eval_passage_count_compress_llama_zero_shot.sh
# sh ./scripts/longbench/passage_retrieval/eval_passage_retrieval_compress_llama_zero_shot.sh
sh ./scripts/longbench/lcc/eval_lcc_compress_llama_zero_shot.sh
sh ./scripts/longbench/repobench/eval_repobench_compress_llama_zero_shot.sh
