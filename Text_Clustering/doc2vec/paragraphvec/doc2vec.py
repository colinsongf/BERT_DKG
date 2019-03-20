import multiprocessing
import os
import re
import signal
from math import ceil
from os.path import join
import random
from tqdm import trange, tqdm
from torchtext.data import Field, TabularDataset, Example, Dataset
from .models import *
from .loss import NegativeSampling
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

vec_dim = 200
dbow = True
num_epochs = 10
batch_size = 512
lr = 1e-3
window_size = 5
noise_num = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(file_path):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """
    file_path = join(file_path)
    text_field = Field(lower=True, tokenize=_tokenize_str)

    dataset = TabularDataset(
        path=file_path,
        format='csv',
        fields=[('text', text_field)],
        skip_header=True)

    text_field.build_vocab(dataset)
    return dataset


class MyDataset(Dataset):

    def __init__(self, data, **kwargs):
        text_field = Field(lower=True, tokenize=_tokenize_str)
        fields = [("text", text_field)]
        examples = []
        for text in data:
            examples.append(Example.fromlist([text], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)
        text_field.build_vocab(self)
        self.vocab = text_field.vocab
    
def _tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()


def get_noise_words(vocab, word_ix, num):
    noise_words = []
    while len(noise_words) != num:
        index = random.randint(0, len(vocab.stoi) - 1)
        if index != word_ix and index not in noise_words:
            noise_words.append(index)
    return noise_words


def get_train_dataset(dataset, vocab):
    examples = []
    for doc_id, data in enumerate(dataset.examples):
        for ix, word in enumerate(data.text):
            target_noise_ids = [vocab.stoi[word]] + get_noise_words(vocab, vocab.stoi[word], noise_num)
            context = ['<pad>'] * (window_size - ix) + data.text[max(ix - window_size, 0):ix] + data.text[ix + 1:min(
                ix + window_size + 1, len(data.text))] + ['<pad>'] * (window_size + ix + 1 - len(data.text))
            context_ids = [vocab.stoi[word] for word in context]
            examples.append([doc_id, context_ids, target_noise_ids])

    doc_ids = torch.LongTensor([example[0] for example in examples])
    context_ids = torch.LongTensor([example[1] for example in examples])
    target_noise_ids = torch.LongTensor([example[2] for example in examples])
    train_dataset = TensorDataset(doc_ids, context_ids, target_noise_ids)
    return train_dataset


def main(data):
    # dataset = load_dataset(file_path)
    dataset = MyDataset(data)
    print("doc num: %d" % (len(dataset)))
    vocab = dataset.fields['text'].vocab
    train_dataset = get_train_dataset(dataset, vocab)

    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    if dbow:
        model = DBOW(vec_dim, num_docs=len(dataset), num_words=len(vocab.itos))
    else:
        model = DM(vec_dim, num_docs=len(dataset), num_words=len(vocab.itos))

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch_i in trange(num_epochs, desc="Epoch"):
        loss = []
        for batch in tqdm(dataloader, desc="Iter"):
            batch = tuple(t.to(device) for t in batch)
            if dbow:
                x = model.forward(batch[0], batch[2])
            else:
                x = model.forward(batch[1], batch[0], batch[2])

            x = cost_func.forward(x)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer.step()

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)
        print("loss: {:.4f}".format(loss))

    return model._D.tolist()


if __name__ == "__main__":
    main("../data/temp.txt")
