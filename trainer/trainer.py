import argparse
import os
import torch.backends.cudnn as cudnn
import yaml

import sys
sys.path.insert(0, "TextRecognitionDataGenerator/")
sys.path.insert(0, "TextRecognitionDataGenerator/server")

from train import train
from utils import AttrDict
import pandas as pd
from pathlib import Path

cudnn.benchmark = True
cudnn.deterministic = False

def find_config(configfile: str):
    p = Path(configfile)
    if not p.exists():
        pp = "config_files" / p
        if not pp.exists():
            raise RuntimeError(f"Config file not found: {configfile}")
        return pp
    return str(p.absolute())

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    opt.tg_settings['symbols'] = opt.symbol
    opt.tg_settings['max_len'] = opt.batch_max_length
    opt.tg_settings['height'] = opt.imgH
    opt = AttrDict(dict(opt))
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()    
#opt = get_config("config_files/en_filtered_config.yaml")
#opt = get_config("config_files/en_filtered_config_ft.yaml")
#opt = get_config("config_files/orig_config_ft.yaml")
#opt = get_config("config_files/orig_config_ft2.yaml")
#opt = get_config("config_files/vgg_lstm_ctc.yaml")
#opt = get_config("config_files/resnet_none_attn.yaml")
#opt = get_config("config_files/resnet_lstm_attn.yaml")

opt = get_config(find_config(args.config))
    
train(opt)
