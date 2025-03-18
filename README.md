## Overview

This repository provides an implementation of [**Summary-Guided Decoding**](https://arxiv.org/abs/2410.13321), a novel decoding strategy designed to mitigate hallucinations in Large Vision-Language Models (LVLMs). Our method leverages summary-based guidance to enhance response generation, particularly in tasks requiring a stronger focus on visual information, while effectively reducing reliance on language priors. Our paper is accepted at *NAACL 2025 Findings.*

## Setup

We implement our Summary-Guided Decoding in `transformers-4.29.2/src/transformers/generation/utils.py` followed by OPERA, summary_guided_decoding.py, and summary_guided_decoding_distill.py

Follow below steps to use our code.
```
# Create and activate a new conda environment
conda create -n sumgd 
conda activate sumgd

# Install the modified transformers library
python -m pip install -e transformers-4.29.2

# Install dependencies
pip install -r requirements.txt
```

## Implementation of Summary-Guided Decoding

We present two versions of Summary-Guided Decoding: 

- **SumGD with Self-Summarization (SumGD-S)**, where the MLLM generates its own summary before performing guided decoding.
- **SumGD with the Distilled-FlanT5 Model (SumGD-D)**, which utilizes a distilled summary model for more efficient summary generation.

The respective codes can be found in `summary_guided_decoding.py` and `summary_guided_decoding_distill.py`.

## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

### CHAIR
- Generate the MLLM's responses and save them in a jsonl file:
```bash
conda activate sumgd

### SumGD-S (Self-Summary Guidance)
python chair_summary_guided_decoding.py --model llava-1.5 --sumgd_mode sumgd-s --max_new_token 512 --min_new_token 1 --result_path /path/to/save/jsonl

### SumGD-D (Distilled Summary Guidance)
python chair_summary_guided_decoding.py --model llava-1.5 --sumgd_mode sumgd-d --max_new_token 512 --min_new_token 1 --result_path /path/to/save/jsonl

### Base Decoding
If you want to use standard decoding methods instead of Summary-Guided Decoding, modify the following code in chair_summary_guided_decoding.py:
with torch.inference_mode():
        with torch.no_grad():
            out = model.generate( 
                {"image": norm(image).half(), "prompt": qu}, 
            use_nucleus_sampling=False, num_beams=1,
            max_new_tokens=args.max_new_token,
            min_new_tokens=args.min_new_token, summary_guided_decoding=True # Set to False for standard decoding
            )

```

### Compute CHAIR Scores

- After generating responses, compute CHAIR metrics using the following command:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl

## Acknowledgement
This repo is based on the MLLM codebase of [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and the CHAIR code of [Maxlinn](https://github.com/Maxlinn/CHAIR-metric-standalone) and [OPERA](https://github.com/shikiw/OPERA). Thanks for their impressive works!
