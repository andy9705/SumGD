#!/bin/bash

conda activate kl_div

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/minigpt4_greedy_5000.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/minigpt4_greedy_5000.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/minigpt4_greedy_5000_kl.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/minigpt4_greedy_5000_kl.json

#/home/kyungmin/OPERA/llava-1.5_only_summary_ours-s_50.0-t_15-num_can_5-p_1.0.jsonl
# #/home/kyungmin/OPERA/llava-1.5_opera_ours-s_50.0-t_15-num_can_5-p_1.0.jsonl
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_greedy_100_top10.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_greedy_100.json


# /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100.jsonl
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_01_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_01_jsd.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_03_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_03_jsd.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_05_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_05_jsd.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_01_jsd_there.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_01_jsd_there.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_03_jsd_there.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_03_jsd_there.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava-1.5_noun_masking_100_05_jsd_there.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_noun_masking_100_05_jsd_there.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_adj_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_adj_05.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_adj_1.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_adj_1.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_m3id.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_noun_1_inference_1.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_self_summary_noun_1_inference_1.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_noun_masking_noun_adj_1.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_noun_masking_noun_adj_1.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_1_alpha06.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_1_alpha06.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_1_alpha08.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_1_alpha08.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_12_alpha06.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_12_alpha06.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_12_alpha08.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_12_alpha08.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_15_alpha01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_15_alpha01.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_15_alpha03.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_image_adj_noun_num_repetition_15_alpha03.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_25.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_25.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_50.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_50.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_75.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_75.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_25.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_25.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_50.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_50.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_75.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_75.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_25.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_25.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_50.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_50.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_75.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_75.json



#python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_50_random.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_token_50_random.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_50_random.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_logs/llava-1.5_jsd_masking_phrase_50_random.json

# echo "greedy"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_greedy.json
# echo "greedy"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_greedy.json
# echo "greedy"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_greedy_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_greedy_64.json


# echo "m3id"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_m3id.json
# echo "m3id"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_m3id_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_m3id_64.json

# echo "VCD"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD.json
# echo "VCD"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64.json

# echo "VCD 0.5"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_512_alpha05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_512_alpha05.json
# echo "VCD 0.5"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64_alpha05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64_alpha05.json

# echo "VCD paper"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64_paper.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_64_paper.json
# echo "VCD paper"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_512_paper.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_VCD_512_paper.json



# echo "opera"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_opera.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_opera.json
# echo "opera"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_opera_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/original_opera_llava-1.5_opera_64.json




# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num_verb.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num_verb.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_01.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_05.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_1.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_1.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_1_confidence_01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_1_confidence_01.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_05_confidence_01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_05_confidence_01.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_01_confidence_01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_every_token_alpha_01_confidence_01.json

# echo "finetuned flan base"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_base_noun_adj_num.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_base_noun_adj_num.json
# echo "finetuned flan large"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_large_noun_adj_num.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_large_noun_adj_num.json

# echo "finetuned flan base 64"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_base_noun_adj_num_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_t5_base_noun_adj_num_64.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_self_summary_noun_adj_num_64.json
# echo "original flan base 64"
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_base_noun_adj_num_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_base_noun_adj_num_64.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_base_noun_adj_num.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_base_noun_adj_num.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_large_noun_adj_num.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/llava-1.5_original_t5_large_noun_adj_num.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/instructblip_self_summary_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/instructblip_self_summary_noun_adj_num_512.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_64.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_128.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_256.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_512.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_greedy_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_greedy_512.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_nucleus_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_nucleus_512.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_beam_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_beam_512.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_VCD_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_VCD_512.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_m3id_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_m3id_512.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_opera_512.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_self_summary_noun_adj_num_512_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_self_summary_noun_adj_num_512_500.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_distilled_t5_base_noun_adj_num_512_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_distilled_t5_base_noun_adj_num_512_500.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_distilled_t5_base_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_distilled_t5_base_noun_adj_num_512.json



#python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/instructblip_self_summary_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/self_summary/instructblip_self_summary_noun_adj_num_512.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/minigpt4_self_summary_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/instructblip_self_summary_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/self_summary/instructblip_distilled_t5_base_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# # python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_self_summary_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# # python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_m3id_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/instructblip_m3id_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# #python chair.py --cap_file /home/kyungmin/my_project/OPERA/opera_filelist_result/instructblip_self_summary_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_self_summary_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_distilled_t5_base_noun_adj_num_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_m3id_512_0_500.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# #python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/llava-1.5_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/minigpt4_self_summary_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/minigpt4_m3id_512_use_cache_true.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/minigpt4_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/instructblip_self_summary_noun_adj_num_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/instructblip_m3id_512_use_cache_true.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/experiment_log_paper/instructblip_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_opera_512_second.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_m3id_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_self_summary_noun_adj_num_512_third.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/llava-1.5_nucleus_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_greedy_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_nucleus_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/instructblip_13b_beam_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_opera_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_opera_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_opera_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_opera_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_self_summary_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_distilled_summary_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_self_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_distilled_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_self_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_distilled_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_self_summary_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_greedy_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_m3id_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_greedy_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_m3id_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_greedy_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_instructblip_m3id_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_opera.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_distilled_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_nucleus.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_beam.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/llava-1.5_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_opera.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_distilled_summary_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_distilled_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_distilled_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_distilled_summary_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_self_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_sampling.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_sampling.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_m3id_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_m3id_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava-1.5_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_greedy_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_nucleus_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_beam_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_vcd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_icd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_m3id_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_self_summary_chair_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_distilled_summary_chair_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_greedy_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_nucleus_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_beam_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_vcd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_icd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# #python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_vcd_chair_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_icd_chair_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_self_summary_chair_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_self_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_distilled_summary_chair_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_vcd_chair_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/OPERA/instructblip_opera_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/OPERA/instructblip_opera_ac01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/OPERA/instructblip_opera.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_64_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_128_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_256_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_vcd_512_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_64_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_128_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_256_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_icd_512_05.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava_m3id_pos_control.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_icd_05_pos_control.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_icd_1_pos_control.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_vcd_05_pos_control.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_vcd_1_pos_control.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_icd_05_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava-1.5_vcd_05_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/SID/llava_m3id_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_64.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_128.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_256.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_self_summary_512.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_greedy_64.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_greedy_128.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_greedy_256.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_greedy_512.json

# #python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_1_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_1_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_1_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_1_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_boost_1_512.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_2_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_2_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_2_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_2_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_boost_2_512.json


#/home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_512.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_greedy_512.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_distilled_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_distilled_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path temp.json


#/home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_greedy_512_5000_full_stop.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_greedy_512_5000_full_stop.json --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_greedy_512_5000_full_stop_chair.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_3_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_3_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_3_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_3_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_boost_3_512.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_boost_4_512.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/real_last/llava-1.5_self_summary_constraints.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava_greedy_64_0911.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava_greedy_128_0911.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava_greedy_256_0911.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/motivation_experiment/llava_greedy_512_0911.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/attention_layer_llava/llava_distilled_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_1_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_1_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_1_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_1_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_2_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_2_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_2_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_2_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_3_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_3_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_3_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_3_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_4_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_4_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_4_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava-1.5_boosting_512_modify_alpha_4_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/modify_attention_weight/boost_alpha_4_pos_control_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_m3id_log_prob_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#/home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_greedy_chair.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distilled_summary_chair_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#/home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_512_real.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distilled_summary_chair_10455.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distilled_summary_chair_14637.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_10455.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_14637.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distill_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distill_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distill_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distill_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_vcd_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_vcd_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_vcd_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_nucleus_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_nucleus_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_nucleus_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava_nucleus_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/m3id_log_prob/llava-1.5_m3id_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp_version2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_modified_0921.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_modified_0921.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_greedy_original.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_nucleus_original.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_beam_original.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_icd_chair_rp_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_vcd_chair_rp_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_greedy_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_opera_chair_original_version.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result_no_repetition/instructblip_m3id_log_prob_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_opera_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_vcd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_vcd_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_icd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_icd_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_rp_version2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp_version2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_result/instructblip_self_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#/home/kyungmin/my_project/OPERA/instructblip_result/instructblip_icd_chair.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/llava_beam_chair_64_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json




# python chair.py --cap_file /home/kyungmin/my_project/OPERA/summary_quality/llava-1.5_gpt4o_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/summary_quality/llava-1.5_there_is.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/summary_quality/llava_self_summary_all_pos.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_modified_0921_version2_ac01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/instructblip_distilled_summary_chair_modified_0921_version2_ac01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#/home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_vcd_128.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_vcd_64.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_vcd_128.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_vcd_256.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/last/fixed_llava-1.5_vcd_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava-1.5_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/instructblip_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/instructblip_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_64_real_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_128_real_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_256_real_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_512_real_jsd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_64_real_jsd_a5k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_128_real_jsd_a5k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_256_real_jsd_a5k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_512_real_jsd_a5k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_64_real_jsd_a6k01_2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_128_real_jsd_a6k01_2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_256_real_jsd_a6k01_2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_512_real_jsd_a6k01_2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_64_real_jsd_a5k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_128_real_jsd_a5k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_256_real_jsd_a5k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_512_real_jsd_a5k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_64_real_jsd_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_128_real_jsd_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_256_real_jsd_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_512_real_jsd_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp_version2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_opera_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_opera_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_opera_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_icd_256_real_jsd_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_icd_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_icd_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_icd_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_icd_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava_vcd_analysis_1009_a6k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_64_a6k_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_256_a6k_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_vcd_64_a6k_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/new_icd/instructblip_icd_7b_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/new_icd/llava_icd_7b_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/greedy_new/llava_greedy_a6k01_max.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json



#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/instructblip_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_icd_chair_rp.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_0920/instructblip_icd_chair_rp_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json










# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_64_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_128_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_256_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240922/llava_vcd_512_real.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_chair_paper_results/llava-1.5_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json




# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_nucleus.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_beam.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_opera_chair_max_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_chair_max_512_m3id.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_self_summary_chair_revised.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_distill_chair_max_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_nucleus.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_beam.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_vcd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_icd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_m3id_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_chair_self_summary_max_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/instructblip_13b_distill_chair_max_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_distilled_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_greedy_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_nucleus.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_beam.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_icd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_7b_opera_last.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_greedy_ac01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/llava_13b_greedy_chair_a6k.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_opera_last.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_self_summary_constraints.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_vcd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_icd_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_m3id_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json





#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_self_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_opera_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#/home/kyungmin/my_project/OPERA/instructblip_last/instructblip_7b_self_summary_chair.jsonl
#/home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_opera_chair.jsonl
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/instructblip_self_summary_chair_rp_version2.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/distilled_result/llava_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_self_summary_chair_revised.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/13b_model/llava_13b_distill_chair_max_512.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_7b_self_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_7b_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_self_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/instructblip_13b_distilled_summary_chair.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/llava_7b_self_summary_all_token_a5k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/llava_7b_self_summary_all_token_a6k01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

# python chair.py --cap_file /home/kyungmin/my_project/OPERA/instructblip_last/llava_7b_self_summary_all_token_ac01.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/20240908_propn/llava_self_summary.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/llava_next/llava_next_greedy.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/llava_next/llava_next_sgd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_5.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_6.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_7.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_8.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_9.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_original.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_pos_optimize.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_distill_inference_cost.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_distill_optimize.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_distill_original_pos.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_3.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_4.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/llava_next/llava_next_sgd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json

#python chair.py --cap_file /home/kyungmin/my_project/OPERA/llava_self_summary_pos_cost_original.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json




#python chair.py --cap_file /home/kyungmin/llava_next_sgd.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json
#python chair.py --cap_file /home/kyungmin/my_project/llava_next/llava_next_nucleus.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#python chair.py --cap_file /home/kyungmin/my_project/OPERA/7b_model/llava-1.5_self_summary_constraints.jsonl --image_id_key image_id --caption_key caption --coco_path /home/kyungmin/emnlp2024/OPERA/annotations/ --save_path /home/kyungmin/my_project/OPERA/experiment_log_paper/temp.json


#/home/kyungmin/my_project/llava_next/llava_next_sgd.jsonl








