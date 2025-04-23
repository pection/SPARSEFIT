
import gpt3
import logging
import math
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, NewType

InputDataClass = NewType("InputDataClass", Any)

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, AdaLoraConfig, IA3Config
from transformers.trainer_utils import EvaluationStrategy
from transformers.integrations import TensorBoardCallback
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForLanguageModeling

from feature_conversion_methods import format_instance

from custom_args import (
    DataTrainingArguments,
    ModelArguments
)
from metrics import evaluate
import torch
import datasets
import git
import time
from datetime import datetime
import sys
from tqdm import trange
import random 
import pandas as pd 
import jsonlines
from copy import deepcopy
from typing import List, Dict
from accelerate import Accelerator

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()
import re
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-base")
print(model)
# Freeze encoder and shared embedding
for name, param in model.named_parameters():

    if 'layer_norm' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
# for param in model.parameters():
#     param.requires_grad = False
# Deactivate language model head

# model.lm_head.weight.requires_grad = True
# Double check what's still trainable
print(f"model_parameter")
for name , param in model.named_parameters():
    print(name)
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable parameters:")
for name in trainable:
    print(name)

# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_percent = 100 * trainable_params / total_params
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable %: {trainable_percent:.2f}%")


