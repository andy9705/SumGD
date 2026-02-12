# Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding (NAACL 2025 Findings)

[![arXiv](https://img.shields.io/badge/arXiv-2410.13321-b31b1b.svg)](https://arxiv.org/abs/2410.13321)
[![Conference](https://img.shields.io/badge/NAACL-2025%20-blue)](https://arxiv.org/abs/2410.13321)

## Abstract

This repository provides the official implementation of our paper [**Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding**](https://arxiv.org/abs/2410.13321). We introduce a novel decoding strategy designed to reduce hallucinations in Large Vision-Language Models (LVLMs) by leveraging summary-based guidance to enhance response generation. Our approach significantly improves model performance on tasks requiring strong visual grounding while effectively reducing reliance on language priors. Our paper is accepted at *NAACL 2025 Findings.*

## Method Overview

We present two variants of our Summary-Guided Decoding approach:

1. **SumGD with Self-Summarization (SumGD-S)**: The LVLM generates its own summary before performing guided decoding
2. **SumGD with Distilled Flan-T5 (SumGD-D)**: Utilizes a distilled summary model for more efficient summary generation

## Installation

```bash
# Create and activate a conda environment
conda create -n sumgd
conda activate sumgd

# Install our modified transformers library
python -m pip install -e transformers-4.29.2

# Install dependencies
pip install -r requirements.txt
```

## Required Model Checkpoints

Please download the following model checkpoints:

- [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b)
  - Update the path in `eval_configs/llava-1.5_eval.yaml` at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14)

- [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat)
  - Update the path in `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml` at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25)

## Implementation Details

The core implementation of our Summary-Guided Decoding approach can be found in:

- `transformers-4.29.2/src/transformers/generation/utils.py` (base implementation)
- `summary_guided_decoding.py` (SumGD-S implementation)
- `summary_guided_decoding_distill.py` (SumGD-D implementation)

## Evaluation

### Dataset Preparation

Our evaluation requires the MSCOCO 2014 dataset. Download it from [the official website](https://cocodataset.org/#home) and extract it to your preferred data path.

### Running Inference

Generate responses using our Summary-Guided Decoding approach:

```bash
# SumGD-S (Self-Summary Guidance)
python chair_summary_guided_decoding.py \
  --model llava-1.5 \ #or instructblip
  --sumgd_mode sumgd-s \
  --max_new_token 512 \
  --min_new_token 1 \
  --result_path /path/to/save/results.jsonl

# SumGD-D (Distilled Summary Guidance)
python chair_summary_guided_decoding.py \
  --model llava-1.5 \ #or instructblip
  --sumgd_mode sumgd-d \
  --max_new_token 512 \
  --min_new_token 1 \
  --result_path /path/to/save/results.jsonl
```

### Computing CHAIR Metrics
After generating responses, compute CHAIR metrics with:
```bash
python chair.py \
  --cap_file /path/to/results.jsonl \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path /path/to/COCO/annotations_trainval2014/annotations/ \
  --save_path /path/to/save/chair_metrics.json
```

### Baseline Comparison
To run baseline decoding methods instead of Summary-Guided Decoding, modify the following code in `chair_summary_guided_decoding.py`:
```
# Base Decoding
with torch.inference_mode():
    with torch.no_grad():
        out = model.generate( 
            {"image": norm(image).half(), "prompt": qu}, 
            use_nucleus_sampling=False, 
            num_beams=1,
            max_new_tokens=args.max_new_token,
            min_new_tokens=args.min_new_token, 
            summary_guided_decoding=False  # Set to False for standard decoding
        )
```

## Acknowledgements

This repository builds upon the following excellent codebases:
- [LAVIS](https://github.com/salesforce/LAVIS)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [OPERA](https://github.com/shikiw/OPERA)
- [CHAIR Evaluation](https://github.com/Maxlinn/CHAIR-metric-standalone)

We sincerely appreciate their contributions to the research community.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{min-etal-2025-mitigating,
    title = "Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding",
    author = "Min, Kyungmin  and
      Kim, Minbeom  and
      Lee, Kang-il  and
      Lee, Dongryeol  and
      Jung, Kyomin",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.235/",
    doi = "10.18653/v1/2025.findings-naacl.235",
    pages = "4183--4198",
    ISBN = "979-8-89176-195-7",
    abstract = "Large Vision-Language Models (LVLMs) demonstrate impressive capabilities in generating detailed and coherent responses from visual inputs.However, they are prone to generate hallucinations due to an over-reliance on language priors. To address this issue, we investigate the language priors in LVLMs and make two key observations: (1) Even when predicting the tokens associated with image-related part-of-speech (POS), models increasingly rely on linguistic priors as the token sequences grow, thereby amplifying hallucinations. (2) Methods that directly calibrate LVLM{'}s output distribution to mitigate language priors can lead to a degradation in text quality or even exacerbate hallucinations.Based on these findings, we propose a novel method, \textbf{Sum}mary-\textbf{G}uided \textbf{D}ecoding \textbf{(SumGD)}. This method naturally encourages the model to focus more on image information by reducing the text context through summaries, while controlling only the image-related POS tokens to maintain text quality.Through experiments, we demonstrate that SumGD achieves state-of-the-art performance on object hallucination benchmarks. Furthermore, in terms of the trade-off between precision and recall, SumGD achieves Pareto optimality among the existing methods.Lastly, we observe that although existing methods struggle to balance the reduction of object hallucinations with maintaining text quality, SumGD demonstrates robustness in handling this challenge."
}
```
