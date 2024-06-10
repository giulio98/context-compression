#!/bin/sh
echo "Testing script execution"
chmod +x ./scripts/longbench/narrativeqa/eval_narrativeqa_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/qasper/eval_qasper_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/multifieldqa/eval_multifieldqa_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/wikimqa/eval_wikimqa_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/hotpotqa/eval_hotpotqa_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/musique/eval_musiqueqa_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_govreport.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_qmsum.sh
chmod +x ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_multinews.sh
chmod +x ./scripts/longbench/trec/eval_trec_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/samsum/eval_samsum_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/passage_count/eval_passage_count_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/passage_retrieval/eval_passage_retrieval_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/lcc/eval_lcc_full_mistral_zero_shot.sh
chmod +x ./scripts/longbench/repobench/eval_repobench_full_mistral_zero_shot.sh

# sh ./scripts/longbench/narrativeqa/eval_narrativeqa_full_mistral_zero_shot.sh
# sh ./scripts/longbench/qasper/eval_qasper_full_mistral_zero_shot.sh
# sh ./scripts/longbench/multifieldqa/eval_multifieldqa_full_mistral_zero_shot.sh
# sh ./scripts/longbench/wikimqa/eval_wikimqa_full_mistral_zero_shot.sh
# sh ./scripts/longbench/hotpotqa/eval_hotpotqa_full_mistral_zero_shot.sh
# sh ./scripts/longbench/musique/eval_musiqueqa_full_mistral_zero_shot.sh
# sh ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_govreport.sh
sh ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_qmsum.sh
sh ./scripts/longbench/summarization/eval_longbench_summarization_full_mistral_multinews.sh
# sh ./scripts/longbench/trec/eval_trec_full_mistral_zero_shot.sh
# sh ./scripts/longbench/samsum/eval_samsum_full_mistral_zero_shot.sh
# sh ./scripts/longbench/passage_count/eval_passage_count_full_mistral_zero_shot.sh
# sh ./scripts/longbench/passage_retrieval/eval_passage_retrieval_full_mistral_zero_shot.sh
# sh ./scripts/longbench/lcc/eval_lcc_full_mistral_zero_shot.sh
# sh ./scripts/longbench/repobench/eval_repobench_full_mistral_zero_shot.sh
