# -*- coding:utf8 -*-
from __future__ import absolute_import, division, print_function

import codecs
import logging
import math
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from tools import *
from decoders import *
from embedders import *
from encoders import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class NERModel(nn.Module):
    def __init__(self, config_path_or_type,
                 num_labels, layer_num, embedder, encoder, decoder,
                 cal_X_loss=False, model_state=None):
        super(NERModel, self).__init__()
        config = get_config(config_path_or_type, logger)
        self.dropout_rate = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.hidden_size

        if embedder == "BertEmbed":
            self.embedder = BertEmbed.from_pretrained(config_path_or_type, model_state)
        else:
            self.embedder = eval(embedder)(config)
        if encoder != "None":
            self.encoder = eval(encoder)(self.hidden_size, self.dropout_rate, layer_num)
        else:
            self.encoder = None
        self.decoder = eval(decoder).create(num_labels, self.hidden_size, self.dropout_rate, cal_X_loss)


    def forward(self, input_ids, segment_ids, input_mask, predict_mask, label_ids=None):
        ''' return mean loss of words or preds'''
        if not config['task']['doc_level']:
            output = self.embedder(input_ids, input_mask, segment_ids)  # (batch_size, max_seq_len, hidden_size)
        else:
            output = self.embedder(input_ids, input_mask)  # (batch_size, max_seq_len, hidden_size)
        if self.encoder:
            output = self.encoder(output, input_mask)  # (batch_size, max_seq_len, hidden_size)
        return self.decoder(output, predict_mask, label_ids)

class InputExample(object):
    '''
     An 'InputExample' represents a document.
    '''
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    '''
    An 'InputFeature' represents an input to the model. Can be multiple sentences.
    '''

    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids, ex_id, start_ix=0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids
        self.ex_id = ex_id  # the id of the source InputExample
        self.start_ix = start_ix  # the start index in the source InputExample.

def memory_usage_psutil():
    # return the memory usage( xx MB)
    import psutil, os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


class NERProcessor():

    def get_examples(self, data_dir, data_type):
        return self.create_examples_from_conll_format_file(
            os.path.join(data_dir, 'tiny.txt' if config['task']['debug'] else data_type + '.txt'), data_type)

    @staticmethod
    def get_labels():
        if config['task']['data_type'] == "ai":
            return ['O',
                    'B-FIELD', 'I-FIELD', 'E-FIELD', 'S-FIELD',
                    'B-TEC', 'I-TEC', 'E-TEC', 'S-TEC',
                    'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC']
        else:
            return ['O',
                    'B-PER', 'I-PER', 'E-PER', 'S-PER',
                    'B-ORG', 'I-ORG', 'E-ORG', 'S-ORG',
                    'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                    'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC']

    @staticmethod
    def create_examples_from_conll_format_file(data_file, set_type):
        if not os.path.exists(data_file):
            raise ValueError(data_file + " is not exists!")

        examples = []
        words = []
        labels = []
        start = 0
        max_len = 0
        for index, line in enumerate(codecs.open(data_file, encoding='utf-8')):
            segs = line.split()
            if not line.strip():
                if words and words[-1]!='[SEP]':
                    words.append("[SEP]")
                    labels.append("")
                continue
            if segs[0]=='-DOCSTART-' and words != []:
                guid = "%s-%d" % (set_type, start)
                max_len = max(max_len, len(words)-1)
                examples.append(InputExample(guid=guid, words=words[:-1], labels=labels[:-1]))
                words = []
                labels = []
                start = -1
            elif segs[0]=='-DOCSTART-':
                continue
            else:
                words.append(segs[0])
                labels.append(segs[-1] if len(segs)>1 else "")
                if start == -1:
                    start = index
        guid = "%s-%d" % (set_type, start)
        examples.append(InputExample(guid=guid, words=words[:-1], labels=labels[:-1]))
        max_len = max(max_len, len(words) - 1)
        logger.info("%s max_doc_len = %d"%(set_type, max_len))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list=[]):
    """Convert `InputExamples`s into a list of `InputFeatures`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    tokenize_info = []
    add_label = 'X'
    for (ex_index, example) in enumerate(examples):
        # 先将词全部变成token的形式
        tokenize_count = [[]]
        tokens = [[]]
        segment_ids = [[]]  # 每个token对应的句子id（相对于doc而言）
        sent_index = [0] # doc里面每个句子起始的index
        predict_mask = [[]]
        label_ids = [[]]
        sent_id = 0 # 当前的句子id
        for i, w in enumerate(example.words):
            if w == '[SEP]':  # 遇到SEP表示一个句子结束，这里不将[SEP]加到feature中
                sent_id += 1
                sent_index.append(i+1)
                segment_ids.append([])
                tokenize_count.append([])
                tokens.append([])
                predict_mask.append([])
                label_ids.append([])
                continue
            sub_words = tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            tokenize_count[sent_id].append(len(sub_words))
            tokens[sent_id].extend(sub_words)
            segment_ids[sent_id].extend([sent_id] * len(sub_words))
            for j in range(len(sub_words)):
                if j == 0:
                    predict_mask[sent_id].append(1)
                    label_ids[sent_id].append(label_map[example.labels[i]])
                else:
                    predict_mask[sent_id].append(0)
                    label_ids[sent_id].append(0)  # X -> 0
        tokenize_info.append(tokenize_count)

        words_num = 0
        part_segment_ids = []
        part_predict_mask = []
        part_label_ids = []
        part_tokens = []
        start_ix = 0
        for i,sent in enumerate(segment_ids):
            words_num += len(sent)
            if words_num > max_seq_length - 2 or not config['task']['doc_level']:
                if part_segment_ids:
                    part_segment_ids = [i - part_segment_ids[0] for i in part_segment_ids]
                    part_segment_ids = [0] + part_segment_ids + [part_segment_ids[-1]]
                    part_predict_mask = [0] + part_predict_mask + [0]
                    part_label_ids = [0] + part_label_ids + [0]
                    part_tokens = ['[CLS]'] + part_tokens + ['[SEP]']

                    doc_input_ids = tokenizer.convert_tokens_to_ids(part_tokens)
                    doc_input_mask = [1] * len(doc_input_ids)
                    # Pad up to the doc length
                    padding_length = max_seq_length - len(doc_input_ids)
                    zero_padding = [0] * padding_length
                    doc_input_ids += zero_padding
                    doc_input_mask += zero_padding
                    part_segment_ids += zero_padding
                    part_predict_mask += zero_padding
                    part_label_ids += [0] * padding_length  # [PAD] -> 0

                    assert len(doc_input_ids) == max_seq_length
                    assert len(doc_input_mask) == max_seq_length
                    assert len(part_segment_ids) == max_seq_length
                    assert len(part_predict_mask) == max_seq_length
                    assert len(part_label_ids) == max_seq_length

                    features.append(
                        InputFeatures(input_ids=doc_input_ids, input_mask=doc_input_mask, segment_ids=part_segment_ids,
                                      predict_mask=part_predict_mask, label_ids=part_label_ids, ex_id=example.guid,
                                      start_ix=start_ix))

                words_num = min(len(sent), max_seq_length - 2)  # 取min以防单句大于max_seq_length的情况
                part_segment_ids = [i] * words_num
                part_predict_mask = predict_mask[i][:max_seq_length - 2]
                part_label_ids = label_ids[i][:max_seq_length - 2]
                part_tokens = tokens[i][:max_seq_length - 2]
                start_ix = sent_index[i]
            else:
                part_segment_ids.extend(sent)
                part_predict_mask.extend(predict_mask[i])
                part_label_ids.extend(label_ids[i])
                part_tokens.extend(tokens[i])

        if part_segment_ids:
            part_segment_ids = [i - part_segment_ids[0] for i in part_segment_ids]
            part_segment_ids = [0] + part_segment_ids + [part_segment_ids[-1]]
            part_predict_mask = [0] + part_predict_mask + [0]
            part_label_ids = [0] + part_label_ids + [0]
            part_tokens = ['[CLS]'] + part_tokens + ['[SEP]']

            doc_input_ids = tokenizer.convert_tokens_to_ids(part_tokens)
            doc_input_mask = [1] * len(doc_input_ids)
            # Pad up to the doc length
            padding_length = max_seq_length - len(doc_input_ids)
            zero_padding = [0] * padding_length
            doc_input_ids += zero_padding
            doc_input_mask += zero_padding
            part_segment_ids += zero_padding
            part_predict_mask += zero_padding
            part_label_ids += [0] * padding_length  # [PAD] -> 0

            assert len(doc_input_ids) == max_seq_length
            assert len(doc_input_mask) == max_seq_length
            assert len(part_segment_ids) == max_seq_length
            assert len(part_predict_mask) == max_seq_length
            assert len(part_label_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=doc_input_ids, input_mask=doc_input_mask, segment_ids=part_segment_ids,
                              predict_mask=part_predict_mask, label_ids=part_label_ids, ex_id=example.guid,
                              start_ix=start_ix))

    return features, tokenize_info


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def create_tensor_data(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.ByteTensor([f.input_mask for f in features])
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_predict_mask = torch.ByteTensor([f.predict_mask for f in features])
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_predict_mask, all_label_ids)


def train():
    if config['train']['gradient_accumulation_steps'] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config['train']['gradient_accumulation_steps']))

    config['train']['batch_size'] = int(
        config['train']['batch_size'] / config['train']['gradient_accumulation_steps'])

    random.seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    if use_gpu:
        torch.cuda.manual_seed_all(config['train']['seed'])

    train_examples = processor.get_examples(data_dir, 'train')
    train_features, train_tokenize_info = convert_examples_to_features(train_examples, max_seq_length,
                                                                       tokenizer, label_list)
    if config['task']['ssl']:
        unlabeled_train_examples = processor.get_examples(data_dir, 'train_unlabeled')
        unlabeled_train_features, unlabeled_train_tokenize_info = convert_examples_to_features(unlabeled_train_examples,
                                                                                               max_seq_length,
                                                                                               tokenizer, label_list)

    num_train_steps = math.ceil(
        len(train_examples) / config['train']['batch_size'] / config['train']['gradient_accumulation_steps']) * \
                      config['train']['epochs']
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config['train']['learning_rate'],
                         warmup=config['train']['warmup_proportion'], t_total=num_train_steps)

    with codecs.open(os.path.join(config['task']['output_dir'], "train.tokenize_info"), 'w',
                     encoding='utf-8') as f:
        for item in train_tokenize_info:
            f.write(' '.join([str(num) for num in item]) + '\n')

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config['train']['batch_size'])
    logger.info("  Num steps = %d", num_train_steps)

    train_data = create_tensor_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])

    if config['task']['ssl']:
        unlabeled_train_data = create_tensor_data(unlabeled_train_features)
        unlabeled_train_sampler = SequentialSampler(unlabeled_train_data)
        unlabeled_data_loader = DataLoader(unlabeled_train_data, sampler=unlabeled_train_sampler,
                                           batch_size=config['train']['batch_size'])

    global_step = int(
        len(train_examples) / config['train']['batch_size'] / config['train'][
            'gradient_accumulation_steps'] * start_epoch)

    logger.info("***** Running training*****")
    if config['task']['ssl']:
        weight = torch.tensor(1., requires_grad=False).to(device)
    for epoch in trange(start_epoch, config['train']['epochs'], desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        if config['task']['ssl']:
            unlabeled_iter = iter(unlabeled_data_loader)
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            loss = model(input_ids, segment_ids, input_mask, predict_mask, label_ids)
            if config['task']['ssl'] and epoch > 2:
                batch = unlabeled_iter.next()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                _, probs = model(input_ids, segment_ids, input_mask, predict_mask)
                unlabeled_loss = -((probs.log() * probs).sum(-1) * predict_mask.float()).mean()
                loss = weight * unlabeled_loss + loss

                #print("unlabeled loss: %.3f; \nlabeled loss: %.3f; " % (unlabeled_loss.item(), loss.item()))

            if config['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if config['train']['gradient_accumulation_steps'] > 1:
                loss = loss / config['train']['gradient_accumulation_steps']

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % config['train']['gradient_accumulation_steps'] == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = config['train']['learning_rate'] * warmup_linear(global_step / num_train_steps,
                                                                                config['train']['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
        logger.info("memory usage: %.4f" % memory_usage_psutil())
        logger.info("epoch loss (mean word loss): %.4f" % (tr_loss / nb_tr_steps))
        train_loss_list.append(tr_loss / nb_tr_steps)

        if config['task']['ssl']:
            weight += 5.

        if config['dev']['do_every_epoch']:
            dev_loss_list.append(evaluate('dev', epoch))

        if config['test']['do_every_epoch']:
            evaluate('test', epoch)

        # # Save a checkpoint
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # torch.save({'epoch': epoch, 'model_state': model_to_save.state_dict(), 'max_seq_length': max_seq_length,
        #             'lower_case': lower_case, 'train_loss_list': train_loss_list, 'dev_loss_list': dev_loss_list},
        #            os.path.join(config['task']['output_dir'], 'checkpoint-%d' % epoch))

    draw(train_loss_list, dev_loss_list, config['train']['epochs'])

def evaluate(dataset, train_steps=None):
    examples = processor.get_examples(data_dir, dataset)
    examples_dict = {e.guid: e for e in examples}
    features, tokenize_info = convert_examples_to_features(examples, max_seq_length,
                                                           tokenizer, label_list)

    logger.info("***** Running Evaluation on %s set*****" % dataset)
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Num features = %d", len(features))
    logger.info("  Batch size = %d", config[dataset]['batch_size'])

    data = create_tensor_data(features)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler,
                            batch_size=config[dataset]['batch_size'])
    model.eval()
    predictions = []
    predict_masks = []
    nb_steps, nb_examples = 0, 0
    loss, accuracy = 0, 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        with torch.no_grad():
            tmp_loss = model(input_ids, segment_ids, input_mask, predict_mask, label_ids)
            outputs,_ = model(input_ids, segment_ids, input_mask, predict_mask)
        if not config['task']['cal_X_loss']:
            reshaped_predict_mask, _, _ = valid_first(predict_mask)
        else:
            reshaped_predict_mask = predict_mask
        masked_label_ids = torch.masked_select(label_ids, predict_mask)
        masked_outputs = torch.masked_select(outputs, reshaped_predict_mask)
        masked_label_ids = masked_label_ids.cpu().numpy()
        masked_outputs = masked_outputs.detach().cpu().numpy()

        def cal_accuracy(outputs, labels):
            return np.sum(outputs == labels)

        tmp_accuracy = cal_accuracy(masked_outputs, masked_label_ids)
        predictions.extend(outputs.detach().cpu().numpy().tolist())
        predict_masks.extend(reshaped_predict_mask.detach().cpu().numpy().tolist())
        if config['n_gpu'] > 1:
            tmp_loss = tmp_loss.mean()  # mean() to average on multi-gpu.

        loss += tmp_loss.item()
        accuracy += tmp_accuracy
        nb_examples += predict_mask.detach().cpu().numpy().sum()
        nb_steps += 1
    loss = loss / nb_steps
    accuracy = accuracy / nb_examples

    logger.info('eval_loss: %.4f; eval_accuracy: %.4f' % (loss, accuracy))
    if train_steps is not None:
        fn1 = "%s.predict_epoch_%s" % (dataset, train_steps)
        fn2 = "%s.mistake_epoch_%s" % (dataset, train_steps)
    else:
        fn1 = "%s.predict" % dataset
        fn2 = "%s.mistake" % dataset
    writer1 = codecs.open(os.path.join(config['task']['output_dir'], fn1), 'w', encoding='utf-8')
    writer2 = codecs.open(os.path.join(config['task']['output_dir'], fn2), 'w', encoding='utf-8')
    for feature, predict_line, predict_mask in zip(features, predictions, predict_masks):
        example = examples_dict[feature.ex_id]
        w1_sent = []
        word_idx = feature.start_ix
        mistake = False
        for index, label_id in enumerate(predict_line[:sum(predict_mask)]):
            if example.words[word_idx]=='[SEP]':
                word_idx += 1
                w1_sent.append("\n")
            line = ' '.join([example.words[word_idx], example.labels[word_idx], label_list[label_id]])
            w1_sent.append(line)
            if label_list[label_id] != example.labels[word_idx]:
                mistake = True
            word_idx += 1
        writer1.write('\n'.join(w1_sent) + '\n\n')
        if mistake: writer2.write('\n'.join(w1_sent) + '\n\n')
    writer1.close()
    writer2.close()
    return loss


def predict():
    examples = processor.get_examples(data_dir, "predict")
    examples_dict = {e.guid: e for e in examples}
    features, tokenize_info = convert_examples_to_features(examples, max_seq_length,
                                                                     tokenizer, label_list)

    logger.info("***** Running Evaluation on prediction set*****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Num features = %d", len(features))
    logger.info("  Batch size = %d", config['predict']['batch_size'])

    data = create_tensor_data(features)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler,
                                 batch_size=config['predict']['batch_size'])
    model.eval()
    predictions = []
    predict_masks = []
    nb_steps, nb_examples = 0, 0
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        with torch.no_grad():
            outputs, _ = model(input_ids, segment_ids, input_mask, predict_mask)

        if not config['task']['cal_X_loss']:
            reshaped_predict_mask, _, _ = valid_first(predict_mask)
        else:
            reshaped_predict_mask = predict_mask

        predictions.extend(outputs.detach().cpu().numpy().tolist())
        predict_masks.extend(reshaped_predict_mask.detach().cpu().numpy().tolist())

        nb_examples += predict_mask.detach().cpu().numpy().sum()
        nb_steps += 1

    writer1 = codecs.open(os.path.join(config['task']['output_dir'], "prediction_conll_results.txt"), 'w',
                          encoding='utf-8')
    writer2 = codecs.open(os.path.join(config['task']['output_dir'], "prediction_entities.txt"), 'w',
                          encoding='utf-8')
    last_example_id = None
    entity_types = list(set([i.split("-")[-1] for i in label_list]))
    entity_types.remove("O")
    entities = {i: [] for i in entity_types}
    r = re.compile("[^A-Za-z\-]")  # 过滤实体内的其他字符
    for feature, predict_line, predict_mask in zip(features, predictions, predict_masks):
        example = examples_dict[feature.ex_id]
        w1_sent = []
        if feature.ex_id != last_example_id:
            if last_example_id is not None:
                writer2.write(', '.join(['\t'.join(entity) for entity in entities.values()]) + "\n")
            entities = {i: [] for i in entity_types}
        entity = []
        pretype = None
        word_idx = feature.start_ix
        for index, label_id in enumerate(predict_line[:sum(predict_mask)]):
            if example.words[word_idx] == '[SEP]':
                word_idx += 1
                w1_sent.append("")
            line = ' '.join([example.words[word_idx], label_list[label_id]])
            if label_list[label_id] != "O":
                if label_list[label_id].startswith("B-"):
                    if pretype is not None:
                        if ' '.join(entity).lower():
                            entities[pretype].append(' '.join(entity))
                        entity = []
                    pretype = label_list[label_id].split("-")[1]
                    entity.append(re.sub(r, "", example.words[word_idx]))
                else:
                    if pretype is not None and label_list[label_id].split("-")[1] == pretype:
                        entity.append(re.sub(r, "", example.words[word_idx]))
            else:
                if pretype is not None:
                    if ' '.join(entity).lower():
                        entities[pretype].append(' '.join(entity))
                    entity = []
                    pretype = None
            w1_sent.append(line)
            word_idx += 1
        if feature.ex_id != last_example_id:
            writer1.write('-DOCSTART- O\n\n')
        writer1.write('\n'.join(w1_sent) + '\n\n')
        last_example_id = feature.ex_id
    writer2.write(', '.join(['\t'.join(entity) for entity in entities.values()]) + "\n")
    writer1.close()
    writer2.close()


def draw(train, dev,end):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    x = range(1, end + 1)
    plt.plot(x, train, color='green', marker='o')
    legend = ['train']
    if dev != []:
        plt.plot(x, dev, color='red', marker='+')
        legend.append('dev')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(legend)
    plt.savefig(os.path.join(config['task']['output_dir'], "loss.jpg"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1]) as f:
            config = yaml.load(f.read())

        data_dir = os.path.join(config['task']['data_dir'], config['task']['data_type'],
                                "semi" if config['task']['ssl'] else "full",
                                'BIOES' if config['task']['BIOES'] else 'BIO')
        if not os.path.exists(data_dir):
            raise ValueError("Data dir not found: %s" % data_dir)

        if config['use_cuda'] and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            use_gpu = True
        else:
            device = torch.device("cpu")
            use_gpu = False
        logger.info("device: {}".format(device))

        processors = {
            "ner": NERProcessor
        }
        task_name = config['task']['task_name'].lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)

        processor = processors[task_name]()
        label_list = processor.get_labels()

        # reconstruct the output_dirs
        if not os.path.exists(config['task']['output_dir']):
            os.makedirs(config['task']['output_dir'])
        path = os.path.join(config['task']['output_dir'], config['task']['data_type'])
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, '_'.join(
            [config['task']['embedder'], config['task']['encoder'], config['task']['decoder']]) + \
                            ("_reg" if config['task']['ssl'] else "") +
                            ("_doc" if config['task']['doc_level'] else ""))
        if not os.path.exists(path):
            os.makedirs(path)
        config['task']['output_dir'] = path
        logger.info("output dirs : " + config['task']['output_dir'])

        # load checkpoint if exists
        ckpts = [filename for filename in os.listdir(config['task']['output_dir']) if
                 re.fullmatch('checkpoint-\d+', filename)]

        if config['task']['checkpoint'] or ckpts:
            if config['task']['checkpoint']:
                model_file = config['task']['checkpoint']
            else:
                model_file = os.path.join(config['task']['output_dir'],
                                          sorted(ckpts, key=lambda x: int(x[len('checkpoint-'):]))[-1])
            logger.info('Load %s' % model_file)
            checkpoint = torch.load(model_file, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            max_seq_length = checkpoint['max_seq_length']
            lower_case = checkpoint['lower_case']
            model = NERModel(config['task']['bert_model_dir'], len(label_list), config['task']['layer_num'],
                             config['task']['embedder'], config['task']['encoder'], config['task']['decoder'],
                             config['task']['cal_X_loss'],checkpoint['model_state'])
            train_loss_list = checkpoint['train_loss_list']
            dev_loss_list = checkpoint['dev_loss_list']
        else:
            start_epoch = 0
            max_seq_length = config['task']['max_seq_length']
            lower_case = config['task']['lower_case']
            model = NERModel(config['task']['bert_model_dir'], len(label_list), config['task']['layer_num'],
                             config['task']['embedder'], config['task']['encoder'], config['task']['decoder'],
                             config['task']['cal_X_loss'])
            train_loss_list = []
            dev_loss_list = []

        tokenizer = BertTokenizer.from_pretrained(config['task']['bert_model_dir'], do_lower_case=lower_case)

        model.to(device)
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        if config['train']['do']:
            train()
        if config['dev']['do']:
            evaluate('dev')
        if config['test']['do']:
            evaluate('test')
        if config['predict']['do']:
            predict()
    else:
        print("Please specify the config file.")
