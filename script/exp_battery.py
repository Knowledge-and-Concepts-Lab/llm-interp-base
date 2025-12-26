from torch import Tensor, nn

from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import gc
import requests
import os
import sys
import json

import torch
##paths/config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STIM_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "stim_battery.json")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "model_battery.json")
EXP_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "exp_battery.json")

os.environ["TORCH_USE_CUDA_DSA"] = "1"

PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)



def general_eval():
    """
    Run general eval for full model suite
    Args:
        api_models (List): optional- runs additional families of api_model families as specified, ie. openai, gemini
    """
    pass

def wm_eval():
    """
    Run world model eval for full model suite
    Args:
        api_models (List): optional- runs additional families of api_model families as specified, ie. openai, gemini
    """
    pass

def general_explanations():
    """
    Get explanations for large model suite
    Args:
        api_models (List): optional- runs additional families of api_model families as specified, ie. openai, gemini
    """
    pass

def explanation_code():
    """
    classify model + human explanations based on pre-specified code
    """
    pass



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script/exp_battery.py <function_name> [args...]")
        sys.exit(1)

    fn_name = sys.argv[1]
    fn_args = sys.argv[2:]

    if fn_name not in globals() or not callable(globals()[fn_name]):
        raise ValueError(f"Unknown function: {fn_name}")

    globals()[fn_name](*fn_args)