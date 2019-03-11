
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

def get_bert_embed()