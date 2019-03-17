# -*- coding:utf8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import re
import codecs
import logging
import random
import yaml
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from models import *
from tools import valid_first

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN = DEV = TEST = "tiny"

TRAIN = "ai_data_train_labeled_140"
UNLABELED_TRAIN = "ai_data_train_unlabeled_1400"
DEV = "ai_data_dev46"
TEST = "ai_data_test46"


class BertForNER(BertPreTrainedModel):

    def __init__(self, config, num_labels, decoder):
        super(BertForNER, self).__init__(config)
        self.dropout_rate = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        #self.bilstm = BiLSTM(self.hidden_size,self.hidden_size, 1, self.dropout_rate)
        self.decoder = eval(decoder).create(num_labels, self.hidden_size, self.dropout_rate)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, predict_mask, label_ids=None):
        ''' return mean loss of words or preds'''
        hidden, _ = self.bert(input_ids, None, input_mask,
                                  output_all_encoded_layers=False)  # bert_layer: (batch_size, max_seq_len, hidden_size)
        #hidden = self.bilstm(hidden)
        return self.decoder(hidden, predict_mask, label_ids)


class InputExample(object):

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids,ex_id,start_ix):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids
        self.ex_id = ex_id
        self.start_ix = start_ix
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil, os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @staticmethod
    def create_examples_from_conll_format_file(data_file, set_type):
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


    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class CONLLProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, TRAIN + '.txt'), 'train')

    def get_dev_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, DEV + '.txt'), 'dev')

    def get_test_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, TEST + '.txt'), 'test')

    def get_unlabeled_train_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, UNLABELED_TRAIN + '.txt'),
                                                                    'train_unlabeled')

    @staticmethod
    def get_labels():
        #return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        return ['O', 'B-FIELD', 'I-FIELD', 'B-TEC', 'I-TEC', 'B-MISC', 'I-MISC']


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_list=[]):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    tokenize_info = []
    add_label = 'X'
    for (ex_index, example) in enumerate(examples):
        tokenize_count = [[]]
        tokens = [[]]
        segment_ids = [[]]
        sent_index = [0] # doc里面每个句子起始的index
        predict_mask = [[]]
        label_ids = [[]]
        sent_id = 0 # 当前的句子id
        for i, w in enumerate(example.words):
            if w == '[SEP]': # 遇到SEP表示一个句子结束，这里忽略句子间的分隔符
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
        doc_segment_ids = []
        doc_predict_mask = []
        doc_label_ids = []
        doc_tokens = []
        start_ix = 0
        for i,sent in enumerate(segment_ids):
            words_num += len(sent)
            if words_num>max_seq_length-2:
                doc_segment_ids = [0]+ doc_segment_ids + [doc_segment_ids[-1]]
                doc_predict_mask = [0] + doc_predict_mask + [0]
                doc_label_ids = [0] + doc_label_ids + [0]
                doc_tokens = ['[CLS]'] + doc_tokens + ['[SEP]']

                doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
                doc_input_mask = [1] * len(doc_input_ids)
                # Pad up to the doc length
                padding_length = max_seq_length - len(doc_input_ids)
                zero_padding = [0] * padding_length
                doc_input_ids += zero_padding
                doc_input_mask += zero_padding
                doc_segment_ids += zero_padding
                doc_predict_mask += zero_padding
                doc_label_ids += [0] * padding_length  # [PAD] -> 0

                assert len(doc_input_ids) == max_seq_length
                assert len(doc_input_mask) == max_seq_length
                assert len(doc_segment_ids) == max_seq_length
                assert len(doc_predict_mask) == max_seq_length
                assert len(doc_label_ids) == max_seq_length

                features.append(InputFeatures(input_ids=doc_input_ids, input_mask=doc_input_mask, segment_ids=doc_segment_ids,
                                              predict_mask=doc_predict_mask, label_ids=doc_label_ids, ex_id=example.guid, start_ix = start_ix))
                words_num = len(sent)
                doc_segment_ids = [0]*words_num
                doc_predict_mask = predict_mask[i]
                doc_label_ids = label_ids[i]
                doc_tokens = tokens[i]
                start_ix = sent_index[i]
            else:
                doc_segment_ids.extend([s-(doc_segment_ids[0] if doc_segment_ids!=[] else 0) for s in sent])
                doc_predict_mask.extend(predict_mask[i])
                doc_label_ids.extend(label_ids[i])
                doc_tokens.extend(tokens[i])

        doc_segment_ids = [0] + doc_segment_ids + [doc_segment_ids[-1]]
        doc_predict_mask = [0] + doc_predict_mask + [0]
        doc_label_ids = [0] + doc_label_ids + [0]
        doc_tokens = ['[CLS]'] + doc_tokens + ['[SEP]']

        doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        doc_input_mask = [1] * len(doc_input_ids)
        # Pad up to the doc length
        padding_length = max_seq_length - len(doc_input_ids)
        zero_padding = [0] * padding_length
        doc_input_ids += zero_padding
        doc_input_mask += zero_padding
        doc_segment_ids += zero_padding
        doc_predict_mask += zero_padding
        doc_label_ids += [0] * padding_length  # [PAD] -> 0

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(doc_predict_mask) == max_seq_length
        assert len(doc_label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=doc_input_ids, input_mask=doc_input_mask, segment_ids=doc_segment_ids,
                                      predict_mask=doc_predict_mask, label_ids=doc_label_ids,ex_id=example.guid,start_ix = start_ix))
    return features, tokenize_info


def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)


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

    train_examples = processor.get_train_examples(config['task']['data_dir'])
    unlabeled_train_examples = processor.get_unlabeled_train_examples(config['task']['data_dir'])

    num_train_steps = int(
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

    train_features, train_tokenize_info = convert_examples_to_features(train_examples, max_seq_length,
                                                                       tokenizer,label_list)
    unlabeled_train_features, unlabeled_train_tokenize_info = convert_examples_to_features(unlabeled_train_examples,
                                                                                           max_seq_length,
                                                                                           tokenizer,
                                                                                           label_list)

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

    unlabeled_train_data = create_tensor_data(unlabeled_train_features)
    unlabeled_train_sampler = SequentialSampler(unlabeled_train_data)
    unlabeled_data_loader = DataLoader(unlabeled_train_data, sampler=unlabeled_train_sampler,
                                       batch_size=config['train']['batch_size'])

    global_step = int(
        len(train_examples) / config['train']['batch_size'] / config['train'][
            'gradient_accumulation_steps'] * start_epoch)

    logger.info("***** Running training*****")
    weight = torch.tensor(1., requires_grad=False).to(device)
    train_loss_list = []
    dev_loss_list = []
    for epoch in trange(start_epoch, config['train']['epochs'], desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        unlabeled_iter = iter(unlabeled_data_loader)
        weight.fill_(1.)
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, predict_mask, label_ids)

            if epoch > 2:
                batch = unlabeled_iter.next()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                _, probs = model(input_ids, segment_ids, input_mask, predict_mask)
                unlabeled_loss = -((probs.log() * probs).sum(-1) * predict_mask.float()).mean()
                loss = weight * unlabeled_loss + loss

                print("unlabeled loss: %.3f; \nlabeled loss: %.3f; " % (unlabeled_loss.item(), loss.item()))

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
                weight += 1.
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

        # logger.info("memory usage: %.4f" % memory_usage_psutil())
        # logger.info("epoch loss (mean word loss): %.4f" % (tr_loss / nb_tr_steps))
        train_loss_list.append(tr_loss / nb_tr_steps)

        if config['dev']['do_every_epoch']:
            dev_loss_list.append(evaluate(config['dev']['dataset'], epoch))

        if config['test']['do_every_epoch']:
            evaluate(config['test']['dataset'], epoch)

        # Save a checkpoint
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model_to_save.state_dict(), 'max_seq_length': max_seq_length,
                    'lower_case': lower_case,'train_loss_list':train_loss_list, 'dev_loss_list':dev_loss_list},
                   os.path.join(config['task']['output_dir'], 'checkpoint-%d' % epoch))
    if not config['dev']['do_every_epoch'] and config['dev']['do_after_train']:
        evaluate(config['dev']['dataset'])
    if not config['test']['do_every_epoch'] and config['test']['do_after_train']:
        evaluate(config['test']['dataset'])

    draw(train_loss_list, dev_loss_list, config['train']['epochs'])

def evaluate(dataset, train_steps=None):
    if dataset == 'train':
        eval_examples = processor.get_train_examples(config['task']['data_dir'])
    elif dataset == 'dev':
        eval_examples = processor.get_dev_examples(config['task']['data_dir'])
    elif dataset == 'test':
        eval_examples = processor.get_test_examples(config['task']['data_dir'])
    else:
        raise ValueError("The dataset %s cannot be evaled." % dataset)
    eval_examples_dict = {e.guid:e for e in eval_examples}
    eval_features, eval_tokenize_info = convert_examples_to_features(eval_examples, max_seq_length,
                                                                     tokenizer,label_list)
    # with codecs.open(os.path.join(config['task']['output_dir'], "%s.tokenize_info" % dataset), 'w', encoding='utf-8') as f:
    #     for item in eval_tokenize_info:
    #         f.write(' '.join([str(num) for num in item]) + '\n')
    logger.info("***** Running Evaluation on %s set*****" % dataset)
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num features = %d", len(eval_features))
    logger.info("  Batch size = %d", config[dataset]['batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.ByteTensor([f.input_mask for f in eval_features])
    #all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_eval_mask = torch.ByteTensor([f.predict_mask for f in eval_features])
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_eval_mask, all_label_ids)
    # Run evalion for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=config[dataset]['batch_size'])
    model.eval()
    predictions = []
    predict_masks = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, predict_mask, label_ids)
            outputs,_ = model(input_ids, segment_ids, input_mask, predict_mask)
        reshaped_predict_mask, reshaped_label_ids, _ = valid_first(predict_mask, label_ids)
        masked_label_ids = torch.masked_select(label_ids, predict_mask)
        masked_outputs = torch.masked_select(outputs, reshaped_predict_mask)
        masked_label_ids = masked_label_ids.cpu().numpy()
        masked_outputs = masked_outputs.detach().cpu().numpy()

        def accuracy(outputs, labels):
            return np.sum(outputs == labels)

        tmp_eval_accuracy = accuracy(masked_outputs, masked_label_ids)
        predictions.extend(outputs.detach().cpu().numpy().tolist())
        predict_masks.extend(reshaped_predict_mask.detach().cpu().numpy().tolist())
        if config['n_gpu'] > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu.

        eval_loss += tmp_eval_loss.item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += predict_mask.detach().cpu().numpy().sum()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info('eval_loss: %.4f; eval_accuracy: %.4f' % (eval_loss,eval_accuracy))
    if train_steps is not None:
        fn1 = "%s.predict_epoch_%s " % (dataset, train_steps)
        fn2 = "%s.mistake_epoch_%s " % (dataset, train_steps)
    else:
        fn1 = "%s.predict " % dataset
        fn2 = "%s.mistake " % dataset
    writer1 = codecs.open(os.path.join(config['task']['output_dir'], fn1), 'w', encoding='utf-8')
    writer2 = codecs.open(os.path.join(config['task']['output_dir'], fn2), 'w', encoding='utf-8')
    for eval_feature, predict_line, predict_mask in zip(eval_features, predictions, predict_masks):
        example = eval_examples_dict[eval_feature.ex_id]
        w1_sent = []
        word_idx = eval_feature.start_ix
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
    return eval_loss

def draw(train, dev,end):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    x = range(1, end+1)
    plt.plot(x, train, color='green', marker='o')
    legend = ['train']
    if dev !=[]:
        plt.plot(x, dev, color='red', marker='+')
        legend.append('dev')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(legend)
    plt.title("Loss on "+config['task']['decoder'])
    plt.savefig(os.path.join(config['task']['output_dir'], config['task']['decoder'] + "_loss.jpg"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1]) as f:
            config = yaml.load(f.read())
        config['task']['output_dir'] = config['task']['output_dir'] + "_doc_reg"
        if config['use_cuda'] and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            use_gpu = True
        else:
            device = torch.device("cpu")
            use_gpu = False
        logger.info("device: {}".format(device))

        processors = {
            "conll": CONLLProcessor
        }

        task_name = config['task']['task_name'].lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)
        processor = processors[task_name]()
        label_list = processor.get_labels()

        if not os.path.exists(config['task']['output_dir']):
            os.makedirs(config['task']['output_dir'])

        ckpts = [filename for filename in os.listdir(config['task']['output_dir']) if
                 fullmatch('checkpoint-\d+', filename)]
        if config['task']['checkpoint'] or ckpts:
            if config['task']['checkpoint']:
                model_file = config['task']['checkpoint']
            else:
                model_file = os.path.join(config['task']['output_dir'],
                                          sorted(ckpts, key=lambda x: int(x[len('checkpoint-'):]))[-1])
            logging.info('Load %s' % model_file)
            checkpoint = torch.load(model_file, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            max_seq_length = checkpoint['max_seq_length']
            lower_case = checkpoint['lower_case']
            model = BertForNER.from_pretrained(config['task']['bert_model_dir'], state_dict=checkpoint['model_state'],
                                               num_labels=len(label_list), decoder=config['task']['decoder'])
            train_loss_list = checkpoint['train_loss_list']
            dev_loss_list = checkpoint['dev_loss_list']
        else:
            start_epoch = 0
            max_seq_length = config['task']['max_seq_length']
            lower_case = config['task']['lower_case']
            model = BertForNER.from_pretrained(config['task']['bert_model_dir'], num_labels=len(label_list),
                                               decoder=config['task']['decoder'])
            train_loss_list = []
            dev_loss_list = []

        tokenizer = BertTokenizer.from_pretrained(config['task']['bert_model_dir'], do_lower_case=lower_case)

        model.to(device)
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        if config['train']['do']:
            train()
        if config['test']['do']:
            evaluate(config['test']['dataset'])
    else:
        print("Please specify the config file.")
