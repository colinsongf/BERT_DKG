from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans

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

labels = dataset.target
true_k = np.unique(labels).shape[0]

#############################################################################
# get vectorized X
print("using embedding: %s" % opts.embed_type)
X = eval(opts.embed_type)(dataset.data)

# #############################################################################
# Do the actual clustering
f = open("result_%s.txt" % opts.embed_type, "a")
vs = np.array([])
nmis = np.array([])
for i in range(opts.run_num):
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))


    print("--------------The larger the better (%d)---------------------"%(i+1), file=f)
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
print("V-measure: var: %0.4f; mean: %0.4f" %(v_var, v_mean), file=f)
print("Normalized Mutual Information: var: %0.4f; mean: %0.4f" %(nmi_var, nmi_mean), file=f)
f.close()

print("V-measure: var: %0.4f; mean: %0.4f" %(v_var, v_mean))
print("Normalized Mutual Information: var: %0.4f; mean: %0.4f" %(nmi_var, nmi_mean))