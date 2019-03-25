
def get_lda_embed(data):
    import gensim
    from gensim.corpora.dictionary import Dictionary
    import sklearn.preprocessing as preprocessing
    docs = [gensim.utils.simple_preprocess(doc) for i, doc in enumerate(data)]
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in docs]

    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=50, alpha='auto', eval_every=5, update_every=5, chunksize=10000, passes=100)
    from gensim.test.utils import datapath
    import numpy as np
    temp_file = datapath("lda_model")
    lda.save(temp_file)
    #lda = gensim.models.ldamodel.LdaModel.load(temp_file)
    topics = lda.get_topics()
    X = []
    for doc in corpus:
        topic_probs = lda.get_document_topics(doc)
        X.append(np.sum([topics[pair[0]]*pair[1] for pair in topic_probs], axis=0))
    X = preprocessing.normalize(X)
    return X


def get_doc2vec_embed(dataset, hook):
    import gensim
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import sklearn.preprocessing as preprocessing
    docs = [TaggedDocument(gensim.utils.simple_preprocess(doc), [i]) for i, doc in enumerate(dataset.data)]
    doc2vec_dbow = Doc2Vec(dm=0, vector_size=300, min_count=2, max_count=1000)
    doc2vec_dm = Doc2Vec(dm=1, vector_size=300, min_count=2, max_count=1000)
    doc2vec_dbow.build_vocab(docs)
    doc2vec_dm.build_vocab(docs)
    doc2vec_dbow.train(docs, total_examples=doc2vec_dbow.corpus_count, epochs=10)
    doc2vec_dm.train(docs, total_examples=doc2vec_dm.corpus_count, epochs=10)

    X = [doc2vec_dbow.infer_vector(doc.words) for doc in docs]
    X = preprocessing.normalize(X)
    hook(dataset, X)
    return X

def get_bert_embed(data):
    import torch
    import gensim
    from pytorch_pretrained_bert import BertModel, BertTokenizer
    docs = [gensim.utils.simple_preprocess(doc) for i, doc in enumerate(data)]
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model.to(device)
    model.eval()
    X = []
    for i,doc in enumerate(docs):
        print("converting %d"%i)
        doc_tok = ['[CLS]']
        for word in doc:
            toks = tokenizer.tokenize(word)
            doc_tok.extend(toks)
        doc_tok = doc_tok[:511]+['[SEP]']
        doc_ids = tokenizer.convert_tokens_to_ids(doc_tok)
        tokens_tensor = torch.tensor([doc_ids]).to(device)
        with torch.no_grad():
            encoded_layers, pooled_output = model(tokens_tensor, output_all_encoded_layers=False)
        X.append(encoded_layers.sum(1).tolist()[0])
    return X


def get_finetuned_bert_embed(data):
    import torch
    import gensim
    import sklearn.preprocessing as preprocessing
    from torch.utils.data import TensorDataset
    import os
    from pytorch_pretrained_bert import BertModel, BertTokenizer
    from flair.data import Sentence
    import sklearn
    import numpy as np
    # if not os.path.exists("data/20news_conll.txt"):
    #     docs = [Sentence(doc) for doc in data]
    #     with open("data/20news_conll.txt", "w") as f:
    #         for doc in docs:
    #             f.write('\n'.join(["-DOCSTART-\n"]+[tok.text for tok in doc.tokens]+["\n\n"]))
    # docs = [Sentence(doc) for doc in data]
    docs = [gensim.utils.simple_preprocess(doc) for i, doc in enumerate(data)]
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    model = BertModel.from_pretrained('bert2vec/output_dir_lm_ai')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    model.to(device)
    model.eval()
    X = []
    for i, doc in enumerate(docs):
        print("converting %d" % i)
        doc_tok = ['[CLS]']
        for word in doc:
            toks = tokenizer.tokenize(word)
            doc_tok.extend(toks)
        doc_tok += ['[SEP]']
        doc_ids = tokenizer.convert_tokens_to_ids(doc_tok[:512])
        tokens_tensor = torch.tensor([doc_ids]).to(device)
        with torch.no_grad():
            encoded_layers, pooled_output = model(tokens_tensor, output_all_encoded_layers=True)
        X.append(np.concatenate([encoded_layer.sum(1).tolist()[0] for encoded_layer in encoded_layers[-4:]]))

    return preprocessing.normalize(X)


def get_bert2vec_embed(dataset, hook):
    import sklearn.preprocessing as preprocessing
    from bert2vec.run_lm_finetuning import main
    X = main(dataset, Args(), hook)
    X = preprocessing.normalize(X)
    return X

class Args(object):
    def __init__(self, train_file="./data/20news.txt", vocab="./bert2vec/vocab.txt", bert_config="./bert2vec/bert_config.json", vocab_size = 28000):
        self.train_file = train_file
        self.vocab = vocab
        self.bert_config = bert_config
        self.weighted = False
        if self.weighted:
            self.output_dir = "./output_bert_model_weighted_loss"
        else:
            self.output_dir = "./output_bert_model"
        self.max_seq_length = 512
        self.do_train = True
        self.train_batch_size = 256
        self.learning_rate = 3e-4
        self.num_train_epochs = 15.0
        self.warmup_proportion = 0.1
        self.no_cuda = False
        self.do_lower_case = True
        self.local_rank = -1
        self.seed = 42
        self.gradient_accumulation_steps = 1
        self.fp16 = False
        self.loss_scale = 0
        self.vocab_size = vocab_size
        self.mask_prob = -1


def get_tfidf_embed(dataset, hook):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(dataset.data)
    hook(dataset, X)
    return X


def get_doc2vec2_embed(dataset, hook):
    print("data len:%d" % len(dataset.data))
    from doc2vec.paragraphvec.doc2vec import main
    import sklearn.preprocessing as preprocessing
    X = main(dataset.data)
    X = preprocessing.normalize(X)
    hook(dataset, X)
    return X
