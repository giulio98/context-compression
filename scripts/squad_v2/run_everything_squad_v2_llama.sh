#!/bin/sh
echo "Testing script execution"
chmod +x ./scripts/squad_v2/eval_squad_v2_llama_truncate_zero_shot.sh
chmod +x ./scripts/squad_v2/eval_squad_v2_compress_llama_zero_shot.sh
chmod +x ./scripts/squad_v2/eval_squad_v2_llama_full_zero_shot.sh


sh ./scripts/squad_v2/eval_squad_v2_compress_llama_zero_shot.sh
sh ./scripts/squad_v2/eval_squad_v2_llama_full_zero_shot.sh
sh ./scripts/squad_v2/eval_squad_v2_llama_truncate_zero_shot.sh
