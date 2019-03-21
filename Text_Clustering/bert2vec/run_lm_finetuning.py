# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sklearn.preprocessing as preprocessing
from .modeling2 import BertForPreTraining, BertConfig
from .optimization import BertAdam, warmup_linear

from torch.utils.data import Dataset
import random
from collections import Counter, OrderedDict
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, corpus_path_or_list, seq_len, encoding="utf-8", corpus_lines=None):
        self.tokenizer = None
        self.vocab = None
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path_or_list = corpus_path_or_list
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0

        self.all_docs = []
        doc = ""
        self.corpus_lines = 0
        if isinstance(corpus_path_or_list, str):
            with open(corpus_path_or_list, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = ""
                    else:
                        doc += line
                        self.corpus_lines += 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc and doc:
                self.all_docs.append(doc)
        else:
            self.all_docs = corpus_path_or_list

        self.num_docs = len(self.all_docs)

    def build_vocab(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.num_docs

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        # tokenize
        doc_tok = self.tokenizer.tokenize(self.all_docs[item])
        # only keep the words in vocab
        doc_tok = list(filter(lambda x: self.tokenizer.vocab.get(x), doc_tok))
        
        # combine to one sample
        cur_example = InputExample(guid=cur_id, doc_tok=doc_tok, doc_id=item)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.doc_id),
                       torch.tensor(cur_features.lm_label_ids))

        return cur_tensors

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, doc_tok, doc_id, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.doc_tok = doc_tok
        self.lm_labels = lm_labels  # masked words for language model
        self.doc_id = doc_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, doc_id, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.lm_label_ids = lm_label_ids
        self.doc_id = doc_id


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            tokens[i] = "[MASK]"
            # # 80% randomly change token to mask token
            # if prob < 0.8:
            #     tokens[i] = "[MASK]"
            #
            # # 10% randomly change token to random token
            # elif prob < 0.9:
            #     tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            #
            # # -> rest 10% randomly keep current token
            #
            # # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    doc_tok = example.doc_tok

    doc_tokens, lm_label_ids = random_word(doc_tok, tokenizer)
    input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
    input_mask = [1] * len(input_ids)
    doc_id = [example.doc_id] * len(input_ids)

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        doc_id = doc_id[:max_seq_length]
        lm_label_ids = lm_label_ids[:max_seq_length]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        doc_id.append(0)
        lm_label_ids.append(-1)

    if example.guid < 1:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in doc_tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "doc_id: %s" % " ".join([str(x) for x in doc_id]))
        logger.info("LM label: %s " % (lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             doc_id=doc_id,
                             lm_label_ids=lm_label_ids)
    return features


class Tokenizer():
    def __init__(self, docs, vocab_size=30000, lower_case=True):
        self.doc_len = len(docs)
        self.lower_case = lower_case
        from sklearn.feature_extraction.text import CountVectorizer
        counter = CountVectorizer(max_df=0.8, min_df=2, max_features=vocab_size - 2, tokenizer=self.tokenize)
        counter.fit_transform(docs)
        self.vocab = counter.vocabulary_
        # self.vocab = self.build_dict(self.tokenize(' '.join(docs)), vocab_size - 2, 2)
        self.vocab['[UNK]'] = len(self.vocab)
        self.vocab['[MASK]'] = len(self.vocab)

        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def tokenize(self, doc):
        if self.lower_case:
            doc = doc.lower()
        token_pattern = re.compile(r'(?u)\b\w\w+\b')
        return token_pattern.findall(doc)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if self.vocab.get(token) is None:
                ids.append(self.vocab["[UNK]"])
            else:
                ids.append(self.vocab[token])
        return ids

    # def build_dict(self, words, max_words=10000, offset=0, max_df=0.8):
    #     cnt = Counter(words)
    #     words = dict(filter(lambda x: x[1] < max_df * self.doc_len, cnt.items()))
    #     words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    #     words = words[:max_words]  # [(word, count)]
    #     return {word: offset + i for i, (word, _) in enumerate(words)}


def main(dataset, args, hook):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #train_examples = None
    num_train_optimization_steps = None
    train_dataset = BERTDataset(dataset.data, seq_len=args.max_seq_length,
                                corpus_lines=None)
    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    tokenizer = Tokenizer(train_dataset.all_docs, vocab_size =args.vocab_size,  lower_case=args.do_lower_case)
    train_dataset.build_vocab(tokenizer)

    # Prepare model
    bert_config = BertConfig(args.bert_config)
    bert_config.type_vocab_size = len(train_dataset)
    bert_config.vocab_size = len(train_dataset.vocab)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if os.path.exists(output_model_file):
        model = BertForPreTraining.from_pretrained(args.output_dir)
        args.do_train = False

    else:
        model = BertForPreTraining(bert_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, doc_id, lm_label_ids = batch
                loss = model(input_ids, doc_id, input_mask, lm_label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            X = model.get_doc_embed().weight.tolist()
            X = preprocessing.normalize(X)
            hook(dataset, X)
        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    return model.get_doc_embed().weight.tolist()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default="../data/20news.txt",
                        type=str,
                        # required=True,
                        help="The input train corpus.")
    parser.add_argument("--vocab", default=r"./vocab.txt", type=str,
                        # required=True
                        )
    parser.add_argument("--bert_config",
                        default=r"./bert_config.json",
                        type=str,
                        # required=True
                        )
    parser.add_argument("--output_dir",
                        default=r"./output_dir",
                        type=str,
                        # required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--vocab_size",
                        default=28000,
                        type=int)
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    main(args.train_file, args)
