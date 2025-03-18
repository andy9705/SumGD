#!/bin/bash
#SBATCH --job-name test-public-code
#SBATCH --time 96:00:00
#SBATCH --cpus-per-gpu 8
#SBATCH --nodelist ac01
#SBATCH --gpus 1
#SBATCH --mem-per-gpu 150G


conda activate kl_div
#llava-1.5  instructblip  minigpt4
#python shr_eval_contrastive_self_summary_13b.py --model llava-1.5 --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/13b_model/llava_13b_self_summary_shr.jsonl


# python shr_eval_contrastive_self_summary_13b.py --model instructblip --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_self_summary_shr.jsonl
# python chair_eval_contrastive_self_summary_13b.py --model instructblip --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_self_summary_chair.jsonl

#python shr_eval_contrastive_self_summary.py --model instructblip --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_7b_self_summary_shr.jsonl
#python chair_eval_contrastive_self_summary_llava_next.py --model llava-1.5 --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/llava_next_self_summary_inference_cost.jsonl
python chair_eval_contrastive_self_summary.py --model llava-1.5 --sumgd_mode sumgd-s --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/llava_pos_optimize_test.jsonl
python chair_eval_contrastive_self_summary.py --model llava-1.5 --sumgd_mode sumgd-d --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/llava_distill_original_pos.jsonl




#python chair_eval_contrastive_self_summary.py --model llava-1.5  --max_new_token 512 --min_new_token 1 --result_path /home/kyungmin/my_project/OPERA/summary_quality/tests.jsonl

# python chair_eval_contrastive_self_summary.py --model instructblip --max_new_token 128 --min_new_token 128 --result_path /home/kyungmin/my_project/OPERA/last/fixed_instructblip_self_summary_128.jsonl
# python chair_eval_contrastive_self_summary.py --model instructblip --max_new_token 256 --min_new_token 256 --result_path /home/kyungmin/my_project/OPERA/last/fixed_instructblip_self_summary_256.jsonl
#python chair_eval_contrastive_self_summary_distilled_13b.py --model instructblip
#python chair_eval_contrastive_self_summary_dataset.py --model instructblip