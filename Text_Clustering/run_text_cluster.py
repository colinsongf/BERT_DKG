from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans
import os
import logging
from optparse import OptionParser
import sys
from time import time
from doc_embed import *
import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--embed_type",dest="embed_type", default="get_doc2vec_embed")
op.add_option("--run_num",dest="run_num", type=int, default=1)
op.add_option("--doc_path", dest="doc_path", default="20newsgroup")
print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
def load_20news_data():
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)

    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()
    return dataset


class Mydataset(object):
    def __init__(self, data, entities=None):
        self.data = data
        self.entities = entities


def load_ai_data():
    docs = []
    with open("data/ai_data_sents3000.txt", encoding="utf8") as f:
        doc = []
        for line in f.readlines():
            if line == "\n":
                docs.append(" ".join(doc))
                doc = []
            else:
                doc.append(line.strip())
    print("%d docs" % len(docs))

    docs_entities = []
    path = "NER_projects/pt_bert_ner/no_X_version/output_predict_doc_reg/prediction_entities.txt"
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            for line in f.readlines():
                doc_entities = {"FIELD": set(), "TEC": set(), "MISC": set()}
                parts = line.strip("\n").split(", ")
                doc_entities['FIELD'] = parts[0].split("\t") if parts[0].split("\t") != [""] else []
                doc_entities['TEC'] = parts[1].split("\t") if parts[1].split("\t") != [""] else []
                doc_entities['MISC'] = parts[2].split("\t") if parts[2].split("\t") != [""] else []

                docs_entities.append(doc_entities)

    dataset = Mydataset(docs, docs_entities)
    print("%d docs_entities" % len(docs_entities))
    return dataset


def hook(dataset, X, metric=True, cluster_num=None):
    if metric:
        labels = dataset.target
        if cluster_num is None:
            true_k = np.unique(labels).shape[0]
        else:
            true_k = cluster_num
        f = open("result_%s.txt" % opts.embed_type, "a")
        vs = np.array([])
        nmis = np.array([])
        for i in range(opts.run_num):
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=False)
            km.fit(X)

            print("--------------The larger the better (%d)---------------------" % (i + 1), file=f)
            v = metrics.v_measure_score(labels, km.labels_)
            nmi = metrics.normalized_mutual_info_score(labels, km.labels_)
            vs = np.append(vs, v)
            nmis = np.append(nmis, nmi)
            print("V-measure: %0.3f" % v, file=f)
            print("Normalized Mutual Information: %0.3f"
                  % nmi, file=f)
            print("\n\n", file=f)
        v_var = vs.var()
        v_mean = vs.mean()
        nmi_var = nmis.var()
        nmi_mean = nmis.mean()
        print("V-measure: var: %0.4f; mean: %0.4f" % (v_var, v_mean), file=f)
        print("Normalized Mutual Information: var: %0.4f; mean: %0.4f" % (nmi_var, nmi_mean), file=f)
        f.close()

        print("V-measure: var: %0.4f; mean: %0.4f" % (v_var, v_mean))
        print("Normalized Mutual Information: var: %0.4f; mean: %0.4f" % (nmi_var, nmi_mean))
    else:
        if cluster_num is None:
            raise ValueError("While not given the true labels, the cluster num must be setted manually!")
        true_k = cluster_num
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=False)
        km.fit(X)


def run():
    if opts.doc_path == "20newsgroup":
        dataset = load_20news_data()
    else:
        dataset = load_ai_data()
    print("using embedding: %s" % opts.embed_type)
    print("run_num: %d" % opts.run_num)
    eval(opts.embed_type)(dataset, hook)


if __name__ == "__main__":
    run()
