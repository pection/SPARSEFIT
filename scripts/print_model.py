
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

model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

print(model)
# Freeze encoder and shared embedding
for name, param in model.named_parameters():
    if name.startswith("encoder") or name.startswith("shared"):
        param.requires_grad = False
    else:
        param.requires_grad = False

# Double check what's still trainable
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable parameters:")
for name in trainable:
    print(name)
