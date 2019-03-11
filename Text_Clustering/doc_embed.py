

def get_doc2vec_embed(data):
    import gensim
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import sklearn.preprocessing as preprocessing
    docs = [TaggedDocument(gensim.utils.simple_preprocess(doc), [i]) for i, doc in enumerate(data)]
    doc2vec_model = Doc2Vec(vector_size=100, min_count=1, max_count=1000)
    doc2vec_model.build_vocab(docs)
    doc2vec_model.train(docs, total_examples=doc2vec_model.corpus_count, epochs=10)
    lda = gensim.models.ldamodel.LdaModel(corpus=docs, id2word=doc2vec_model.wv.index2word, num_topics=100, update_every=1, passes=1)

    doc2vec_X = [doc2vec_model.infer_vector(doc.words) for doc in docs]
    X = preprocessing.normalize(doc2vec_X)
    return X

def get_bert_embed(data):
    import torch
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
    docs = [Sentence(doc) for doc in data]
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    model = BertModel.from_pretrained('/home/yjc/.pytorch_pretrained_bert/bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('/home/yjc/.pytorch_pretrained_bert/bert-base-cased', do_lower_case=False)
    model.to(device)
    model.eval()
    X = []
    for doc in docs:
        doc_tok = ['[CLS]']
        for word in doc.tokens:
            toks = tokenizer.tokenize(word)
            doc_tok.extend(toks)
        doc_tok += ['[SEP]']
        doc_ids = tokenizer.convert_tokens_to_ids(doc_tok)
        tokens_tensor = torch.tensor([doc_ids]).to(device)
        with torch.no_grad():
            encoded_layers, pooled_output = model(tokens_tensor, output_all_encoded_layers=False)
        X.append(pooled_output.tolist()[0])
    return sklearn.preprocessing.normalize(X)




