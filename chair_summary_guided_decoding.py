
import spacy
nlp = spacy.load('en_core_web_sm')



import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess





from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}
NON_IMAGE_INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <question> ###Assistant:",
    "instructblip": "<question>",
    "lrv_instruct": "###Human: <question> ###Assistant:",
    "shikra": "USER: <question> ASSISTANT:",
    "llava-1.5": "USER: <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True





parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--sumgd_mode", type=str, default="sumgd-s", help="summary-guided decoding mode")
parser.add_argument("--data_path", type=str, default="/home/kyungmin/emnlp2024/OPERA/val2014/", help="data path")
parser.add_argument("--decoding_strategy", type=str, default="greedy", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--max_new_token", type=int)
parser.add_argument("--min_new_token", type=int)
parser.add_argument("--beam", type=int)
parser.add_argument("--opera_decoding", action='store_true')
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--result_path", type=str, default="", help="data path")
args = parser.parse_known_args()[0]




if args.sumgd_mode == "sumgd-s": ### sumgd-s
    if args.model == "llava-1.5":
        from summary_guided_decoding import summary_guided_decoding_function
        summary_guided_decoding_function()
    if args.model == "instructblip":
        from summary_guided_decoding_instructblip import summary_guided_decoding_function
        summary_guided_decoding_function()
elif args.sumgd_mode == "sumgd-d":
    if args.model == "llava-1.5":
        from summary_guided_decoding_distill import summary_guided_decoding_function
        summary_guided_decoding_function()
    if args.model == "instructblip":
        from summary_guided_decoding_distill_instructblip import summary_guided_decoding_function
        summary_guided_decoding_function()



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)




import json

with open("/~your image location",'r') as f: ## your image location.
    json_data=json.load(f)


for i in tqdm(range(0,200)): ## Image Indexes

    img_file = json_data[i]
    img_id = int(img_file.split(".jpg")[0][-6:])
    
    img_save = {}
    img_save["image_id"] = img_id

    img = json_data[i]
    image_path = args.data_path + img
    
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    

    qu = "Please describe this image in detail."
    
    template = INSTRUCTION_TEMPLATE[args.model]
    
    qu = template.replace("<question>", qu)
    
    
    
    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate( 
                {"image": norm(image).half(), "prompt": qu}, 
            use_nucleus_sampling=False, num_beams=1,
            max_new_tokens=args.max_new_token,
            min_new_tokens=args.min_new_token, summary_guided_decoding=True
            )

    img_save["caption"] = out[0]
    
    
    with open(args.result_path, "a") as f:
        json.dump(img_save, f)
        f.write('\n')

        
          
