
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
    return X

def get_doc2vec_embed(data):
    import gensim
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import sklearn.preprocessing as preprocessing
    docs = [TaggedDocument(gensim.utils.simple_preprocess(doc), [i]) for i, doc in enumerate(data)]
    doc2vec_model = Doc2Vec(vector_size=200, min_count=1, max_count=1000)
    doc2vec_model.build_vocab(docs)
    doc2vec_model.train(docs, total_examples=doc2vec_model.corpus_count, epochs=10)
    doc2vec_X = [doc2vec_model.infer_vector(doc.words) for doc in docs]
    X = preprocessing.normalize(doc2vec_X)
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
    model = BertModel.from_pretrained('output_dir')
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
        X.append(np.concatenate([encoded_layer.sum(1).tolist()[0] for encoded_layer in encoded_layers]))

    return X
