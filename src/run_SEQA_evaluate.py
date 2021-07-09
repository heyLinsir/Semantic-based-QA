# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import json, csv

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    BertPreTrainedModel,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from sentence_transformers import SentenceTransformer, util


logger = logging.getLogger(__name__)

def upper_first_word(sentence):
    sentence = [word for word in sentence.split(' ') if word != '']
    sentence = ' '.join(sentence)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence

class COPADataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            question = None
            options = []
            label = None
            for text in f:
                text = text.strip()
                if text[:3] == '<p>':
                    question = text[3:-4]
                    continue
                elif text[:4] in ['<a1>', '<a2>']:
                    sentence = upper_first_word(text[4:-5].strip())
                elif 'most-plausible-alternative' in text:
                    label = int(text[-3]) - 1
                    continue
                else:
                    continue

                options.append(sentence)
                assert len(options) <= 2
                if len(options) < 2:
                    continue

                options = [options[label], options[1 - label]]
                self.examples.append(options)

                options = []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class ROCStoryDataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.examples = []
        with open(file_path) as f:
            f_csv = csv.reader(f)

            for row_id, row in enumerate(f_csv):
                if row_id == 0:
                    continue

                options = row[-3:-1]
                label = int(row[-1]) - 1
                options = [options[label], options[1 - label]]

                self.examples.append(options)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class CosmosQADataset(Dataset):
    def __init__(self,file_path):
        assert os.path.isfile(file_path)

        self.examples = []
        with open(file_path, encoding="utf-8") as f:

            for text in f:

                contents = text.split('\t')

                label = int(contents[-1])
                options = contents[2:-1]

                options = [options[label]] + options[:label] + options[label + 1:]

                self.examples.append(options)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class SocialIQADataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            assert file_path[-8:] == 'convertQ'
            label_path = file_path[:-15]
            with open(label_path + '-labels.lst', encoding="utf-8") as f_label:

                for text, label in zip(f, f_label):

                    label = int(label.strip()) - 1

                    item = json.loads(text)

                    options = [item['answerA'], item['answerB'], item['answerC']]
                    options = [options[label]] + options[:label] + options[label + 1:]

                    self.examples.append(options)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def load_and_cache_examples(args):
    file_path = args.eval_data_file
    if 'COPA' in file_path:
        return COPADataset(file_path)
    elif 'CosmosQA' in file_path:
        return CosmosQADataset(file_path)
    elif 'ROCStory' in file_path:
        return ROCStoryDataset(file_path)
    elif 'SocialIQA' in file_path:
        return SocialIQADataset(file_path)
    raise ValueError("COPA/CosmosQA/ROCStory/SocialIQA should exist in eval_data_file. (e.g. ./data/COPA/xxx)")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def scale_scores(scores, temperature=0.1):
    return torch.exp(scores / temperature)

def evaluate(args):

    supporters_list = pickle.load(open('%s.gpt2xlarge.qa.1.00penalty.topP0.90.minlen2.sample500.pkl' % (args.eval_data_file), 'rb'))
    for i, supporters in enumerate(supporters_list):
        supporters_list[i] = [upper_first_word(supporter) for supporter in supporters[:500]]

    dataset = load_and_cache_examples(args)

    # embedder = SentenceTransformer('/home/niuyilin/pre-trained-models/sentence-robert-large-nli-mean-tokens', device=args.device)
    # embedder = SentenceTransformer('/home/niuyilin/pre-trained-models/test', device=args.device)
    embedder = SentenceTransformer(args.model_name_or_path, device=args.device)

    acc = []
    for options, supporters in tqdm(list(zip(dataset, supporters_list))):
        embeddings = embedder.encode(options + supporters, convert_to_tensor=True)

        option_embeddings = embeddings[:len(options)]
        supporter_embeddings = embeddings[len(options):]

        scores = []
        for option_embedding in option_embeddings:
            score = scale_scores(util.pytorch_cos_sim(option_embedding, supporter_embeddings)[0], temperature=0.1).mean().item()
            scores.append(score)
        acc.append(float(scores[0] > max(scores[1:])))
    print('Accuracy:', np.mean(acc))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    evaluate(args)

if __name__ == "__main__":
    main()
