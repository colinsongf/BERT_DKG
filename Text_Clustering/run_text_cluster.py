# -*- coding:utf8 -*-
from __future__ import print_function

import logging
import os
import random
import sys
from copy import deepcopy
from optparse import OptionParser

import numpy as np
from doc_embed import *
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--embed_type", dest="embed_type", default="get_MD2vec_embed")
op.add_option("--run_num", dest="run_num", type=int, default=10)
op.add_option("--cluster_num", dest="cluster_num", type=int, default=8)
op.add_option("--doc_path", dest="doc_path", default="ai")
op.add_option("--TEST_MODE", dest="TEST_MODE", default=False)
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
    with open("data/ai_abstracts.txt", encoding="utf8") as f:
        doc = []
        for line in f.readlines():
            if line == "\n":
                docs.append(" ".join(doc))
                doc = []
            else:
                doc.append(line.strip())
    print("%d docs" % len(docs))

    docs_entities = []
    # TODO: change the 'prediction_entities.txt' path
    path = "NER_projects/output/conll2003/BertEmbed_MultiAttn_CRFDecoder_reg_doc/prediction_entities.txt"
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            for line in f.readlines():
                doc_entities = {"FIELD": set(), "TEC": set(), "MISC": set()}
                parts = line.strip("\n").split(", ")
                doc_entities['FIELD'] = set(parts[0].split("\t")) if parts[0].split("\t") != [""] else set()
                doc_entities['TEC'] = set(parts[1].split("\t")) if parts[1].split("\t") != [""] else set()
                doc_entities['MISC'] = set(parts[2].split("\t")) if parts[2].split("\t") != [""] else set()

                docs_entities.append(doc_entities)

    dataset = Mydataset(docs, docs_entities)
    print("%d docs_entities" % len(docs_entities))
    return dataset


def hook_doc(dataset, X):
    if metric:
        labels = dataset.target
        true_k = np.unique(labels).shape[0]
        vs = np.array([])
        nmis = np.array([])
        dbs = np.array([])
        chs = np.array([])
        for i in range(opts.run_num):
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=False)
            km.fit(X)
            labels_ = km.labels_

            v = metrics.v_measure_score(labels, labels_)
            nmi = metrics.normalized_mutual_info_score(labels, labels_)
            vs = np.append(vs, v)
            nmis = np.append(nmis, nmi)
            db = metrics.davies_bouldin_score(X, labels)
            dbs = np.append(dbs, db)
            ch = metrics.calinski_harabaz_score(X, labels)
            chs = np.append(chs, ch)
        v_var = vs.var()
        v_mean = vs.mean()
        nmi_var = nmis.var()
        nmi_mean = nmis.mean()
        print("--------------The larger the better---------------------")
        print("V-measure: var: %0.4f; mean: %0.4f" % (v_var, v_mean))
        print("Normalized Mutual Information: var: %0.4f; mean: %0.4f" % (nmi_var, nmi_mean))
        print("Calinski-Harabaz score: var: %0.3f, mean: %0.3f"
              % (chs.var(), chs.mean()))
        print("--------------The lower the better---------------------")
        print("Davies-Bouldin score: var: %0.3f, mean: %0.3f"
              % (dbs.var(), dbs.mean()))
        return v_mean, nmi_mean, dbs.mean(), chs.mean()
    else:
        cluster_num = opts.cluster_num
        scs = np.array([])
        dbs = np.array([])
        for i in range(opts.run_num):
            if not opts.TEST_MODE:
                km = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', n_init=1,
                                     init_size=1000, batch_size=1000, verbose=False)
                km.fit(X)
                labels = km.labels_
            else:
                labels = np.ones([len(dataset.data)])
                labels = np.array(list(map(lambda x: random.randint(0, cluster_num - 1), labels)))

            sc = metrics.silhouette_score(X, labels, sample_size=1000)
            db = metrics.davies_bouldin_score(X, labels)
            scs = np.append(scs, sc)
            dbs = np.append(dbs, db)

        print("--------------The larger the better---------------------")
        print("Silhouette Coefficient: var:%0.3f, mean: %0.3f"
              % (scs.var(), scs.mean()))
        print()

        print("--------------The lower the better---------------------")
        print("Davies-Bouldin score: var: %0.3f, mean: %0.3f"
              % (dbs.var(), dbs.mean()))

        field_top = 10
        path = "cluster%d_field_top%d" % (cluster_num, field_top)
        if not os.path.exists(path):
            os.makedirs(path)

        import pandas as pd
        node_columns = ["lower_word", "normal_case", "entity_type", "degree"]
        edge_columns = ["from_id", "to_id", "nums"]
        df_nodes = pd.DataFrame(columns=node_columns)
        df_edges = pd.DataFrame(columns=edge_columns)

        cluster_df = pd.DataFrame()
        pre_fields = set()
        for cluster in range(cluster_num):
            fields = {}  # { lowercase entity: [normal case, nums, id]}
            id2field = {}
            tecs = {}
            id2tec = {}
            co_occurence = {}  # {(field_id, tec_id):nums}

            entities = np.array(dataset.entities)[labels == cluster]
            for entity in entities:
                f_ids = []
                t_ids = []
                for e in entity["FIELD"]:
                    key = func_get(e, fields)
                    if not key:
                        id2field[len(fields)] = e.lower()
                        fields[e.lower()] = [e.strip('-').strip(" "), 1, len(fields)]
                        f_ids.append(fields[e.lower()][-1])
                    else:
                        fields[key][1] += 1
                        f_ids.append(fields[key][-1])
                for t in entity["TEC"]:
                    key = func_get(t, tecs)
                    if not key:
                        id2tec[len(tecs)] = t.lower()
                        tecs[t.lower()] = [t, 1, len(tecs)]
                        t_ids.append(tecs[t.lower()][-1])
                    else:
                        tecs[key][1] += 1
                        t_ids.append(tecs[key][-1])

                for f in f_ids:
                    for t in t_ids:
                        if id2field[f] != id2tec[t]:
                            co_occurence.setdefault((f, t), 1)
                            co_occurence[(f, t)] += 1
            # 对于同一个词，保留次数多的那个类别，即要么FIELD，要么TEC
            confuse = set(id2field.values()).intersection(set(id2tec.values()))
            # confuse_manual = ['artificial',"pattern","iris","computer","support"] # 预定义的待删除实体
            confuse_manual = []
            delete_tecs = []
            delete_fields = []
            for word in list(confuse):
                if fields[word][1] > tecs[word][1]:
                    id2tec.pop(tecs[word][-1])
                    delete_tecs.append(tecs[word][-1])
                    tecs.pop(word)
                else:
                    id2field.pop(fields[word][-1])
                    delete_fields.append(fields[word][-1])
                    fields.pop(word)

            for word in confuse_manual:
                if word in fields:
                    delete_fields.append(fields[word][-1])
                    fields.pop(word)
                if word in tecs:
                    delete_tecs.append(tecs[word][-1])
                    tecs.pop(word)

            _co_occurence = deepcopy(co_occurence)
            for (f, t), n in _co_occurence.items():
                if f in delete_fields or t in delete_tecs:
                    co_occurence.pop((f, t))

            # 更新dataframe
            n_arr = np.array([[f[0], f[1][0], 0, f[1][1]] for f in fields.items()])
            n_arr = np.append(n_arr, np.array([[t[0], t[1][0], 1, t[1][1]] for t in tecs.items()]), axis=0)
            e_arr = np.array([[e[0][0], e[0][1], e[1]] for e in co_occurence.items()])

            n_df = pd.DataFrame(n_arr, columns=node_columns)
            e_df = pd.DataFrame(e_arr, columns=edge_columns)
            df_nodes = df_nodes.append(n_df,ignore_index=True)
            df_edges = df_edges.append(e_df,ignore_index=True)

            # 选择top3 field以及相应的tec
            fields_ = dict(sorted(fields.items(), key=lambda x: x[1][1], reverse=True)[:field_top])

            # 选择1%之后的几个
            # fields_ = dict(sorted(fields.items(), key=lambda x: x[1][1], reverse=True)[
            #                int(len(fields) * 0.01):int(len(fields) * 0.05) + field_top])

            used_fields = [i[1][-1] for i in fields_.items()]
            co_occurence_ = [i for i in co_occurence.items() if i[0][0] in used_fields]
            used_tecs = [i[0][1] for i in co_occurence.items() if i[0][0] in used_fields]
            with open(os.path.join(path, "cluster_%d.csv" % cluster), "w", encoding="utf8") as f:
                f.write('\n'.join(
                    ["Source,Target,Type,Weight"] + [','.join([str(f), str(t + len(fields)), "Directed", str(n)]) for
                                                     (f, t), n in co_occurence_]))
            with open(os.path.join(path, "nodes_%d.csv" % cluster), "w", encoding="utf8") as f:
                f.write('\n'.join(
                    ["Id,Label"] + [','.join([str(id), str(normal)]) for lower, [normal, num, id] in
                                    fields.items() if id in used_fields]))
                f.write("\n")
                f.write('\n'.join(
                    [','.join([str(id + len(fields)), str(normal)]) for lower, [normal, num, id] in
                     tecs.items() if id in used_tecs]))

            top_tec = sorted([(t[1][0], t[1][1]) for t in tecs.items()], key=lambda x: x[1], reverse=True)[:field_top]
            top_field = [(f[1][0], f[1][1]) for f in fields_.items()]
            cluster_df["cluster #%d (%d abstracts)" % (cluster, len(labels[labels == cluster]))] = np.array(
                [f[0] for f in top_field] + [t[0] for t in top_tec])
            cluster_df["nums #%d" % cluster] = np.array([f[1] for f in top_field] + [t[1] for t in top_tec])
            print('-' * 30 + "fields of cluster %d, docs:%d" % (cluster, len(labels[labels == cluster])) + '-' * 30)
            for i, (f, t) in enumerate(zip(top_field, top_tec)):
                print(str(f) + " ###### " + str(t))
            print("\n")

            # 用 networkx 画图
            # eages = [(f, t + len(fields), n) for (f, t), n in co_occurence_]
            # id2label = {id: normal for lower, [normal, num, id] in fields.items()}
            # id2label.update({id + len(fields): normal for lower, [normal, num, id] in tecs.items()})
            #
            # import networkx as nx
            # from matplotlib import pyplot as plt
            # plt.switch_backend('agg')
            # DG = nx.DiGraph()
            # DG.add_weighted_edges_from(eages)
            #
            # pos = nx.spring_layout(DG, k=0.15, iterations=100)
            #
            # id2labels = {id: id2label[id] for id in pos.keys()}
            #
            # d = nx.degree(DG)
            # d = [(d[node] + 1) * 50 for node in DG.nodes()]
            #
            # nx.draw_networkx_nodes(DG, pos, node_size=d)
            # nx.draw_networkx_edges(DG, pos)
            # nx.draw_networkx_labels(DG, pos, labels=id2labels)
            # plt.axis('off')
            # plt.savefig(os.path.join(path, "cluster_%d.png" % cluster))  # save as png
            # # plt.show()


        df_nodes.to_csv("nodes.csv",index_label="node_id")
        df_edges.to_csv("edges.csv",index_label="edge_id")
        cluster_df.to_csv("top%d_res.csv"%field_top,index=False)
        return dbs.mean(), scs.mean()

# 判断实体是否已存在
def func_get(e, entities):
    if e.lower() in entities:
        return e.lower()
    if e.lower()+"s" in entities: # svm = svms
        return e.lower()+"s"
    if e.lower().strip("s") in entities: # svms = svm
        return e.lower().strip("s")
    if ' '.join(e.lower().split(" ")[:-1]) in entities:
        return ' '.join(e.lower().split(" ")[:-1])
    return None

def run():
    global metric
    if opts.doc_path == "20newsgroup":
        dataset = load_20news_data()
        metric = True
    else:
        dataset = load_ai_data()
        metric = False
    print("using embedding: %s" % opts.embed_type)
    print("run_num: %d" % opts.run_num)
    if opts.TEST_MODE:
        hook_doc(dataset, None)
    else:
        eval(opts.embed_type)(dataset, hook_doc)


if __name__ == "__main__":
    run()
