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
import csv
import json
from typing import Dict, List, Tuple

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

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class COPADataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.corpus = []
        self.data_type = 'COPA'
        with open(file_path, encoding="utf-8") as f:
            sentence = None
            query_type = None
            for text in f:
                text = text.strip()
                if text[:3] == '<p>':
                    sentence = text[3:-4].strip()[:-1] + ' ' + query_type
                elif 'most-plausible-alternative' in text:
                    if 'effect' in text:
                        query_type = 'so'
                    elif 'cause' in text:
                        query_type = 'because'
                    continue
                else:
                    continue

                self.corpus.append(sentence)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]


class ROCStoryDataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.corpus = []
        self.data_type = 'ROCStory'
        with open(file_path, encoding="utf-8") as f:
            f_csv = csv.reader(f)

            for row_id, row in enumerate(f_csv):
                if row_id == 0:
                    continue

                P = row[1:5]

                for i, sentence in enumerate(P):
                    P[i] = ' '.join([word for word in sentence.split(' ') if word != ''])
                P = ' '.join(P)

                self.corpus.append(P)
                            
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]


class SocialIQADataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.corpus = []
        self.data_type = 'SocialIQA'
        with open(file_path, encoding="utf-8") as f:

            for text in f:

                item = json.loads(text)

                context = item['context']
                question = item['question']

                prefix = '%s %s' % (context, question)

                self.corpus.append(prefix)


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]


class CosmosQADataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)

        self.corpus = []
        self.data_type = 'CosmosQA'
        with open(file_path, encoding="utf-8") as f:

            for text in f:

                contents = text.split('\t')

                context = contents[0]
                question = contents[1]

                self.corpus.append('%s %s' % (context, question))


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]


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


def generate(model, tokenizer, encoded_prompt, repetition_penalty=1., top_k=0, top_p=1, min_length=2, data_type='COPA'):
    '''
    encoded_prompt: [1, docuemnt_length]
    '''

    filtered_output_sequences = []
    prefix = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)

    max_length = 15
    if data_type == 'CosmosQA':
        max_length = 20
    
    for generation_round in range(500):
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=max_length + len(encoded_prompt[0]),
            temperature=1.,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=100, # If out-of-memory, use a smaller num_return_sequences.
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        texts = [tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)[len(prefix) : ] for generated_sequence in output_sequences]
        for text, generated_sequence in zip(texts, output_sequences):
            if len(text.strip().split(' ')) < min_length:
                continue
            if '\n' in text:
                continue
            if data_type == 'SocialIQA':
                if '<|endoftext|>' in text:
                    filtered_output_sequences.append(generated_sequence)
                elif '.' in text:
                    filtered_output_sequences.append(generated_sequence)
                else:
                    continue
            else:
                if '<|endoftext|>' in text and '.' in text[: text.find('<|endoftext|>')]:
                    filtered_output_sequences.append(generated_sequence)
                elif '<|endoftext|>' not in text and '.' in text:
                    filtered_output_sequences.append(generated_sequence)
                else:
                    continue

        print(len(filtered_output_sequences))
        if len(filtered_output_sequences) > 500:
            break

    generated_sequences = []
    entire_texts = []

    for generated_sequence in filtered_output_sequences:
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[len(prefix) : ]

        if '<|endoftext|>' in text:
            text = text[: text.find('<|endoftext|>')].strip()
        else:
            text = text.strip()
        if data_type == 'SocialIQA':
            if '.' in text:
                text = text[: text.find('.') + 1].strip()
        else:
            text = text[: text.find('.') + 1].strip()
        
        entire_texts.append('%s %s' % (prefix, text))

        generated_sequences.append(text)

    print('Non-repetitive voter number:', len(set(generated_sequences)))

    return entire_texts, generated_sequences

        

def generate_file(args, model, tokenizer):
    top_p = 0.9
    min_length = 2

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args)

    # Eval!
    model.eval()

    save_path = args.eval_data_file + '.gpt2xlarge.qa.%.2fpenalty.topP%.2f.minlen%d.sample500.pkl' % (args.repetition_penalty, top_p, min_length)

    generated_sequences_list = []

    for sentence in tqdm(eval_dataset):

        tokenized_prompt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
        with torch.no_grad():
            input_ids = torch.tensor([tokenized_prompt]).long().to(args.device)
            entire_texts, generated_sequences = generate(model, tokenizer, input_ids, repetition_penalty=args.repetition_penalty, top_k=0, top_p=top_p, min_length=min_length, data_type=eval_dataset.data_type)

        print('Input:', entire_texts[0][:-len(generated_sequences[0])])
        print('Generation example:', generated_sequences[0])
        print('Generated number: ', len(generated_sequences))
        generated_sequences_list.append(generated_sequences)

    pickle.dump(generated_sequences_list, open(save_path, 'wb'))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
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

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )

    parser.add_argument("--repetition_penalty", default=1.2, type=float, help="Repetition penalty for generation.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()


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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Generation
    generate_file(args, model, tokenizer)


if __name__ == "__main__":
    main()
