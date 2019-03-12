from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
print(categories)
#import spacy
import codecs
import re
#nlp = spacy.load("en")
r1 = re.compile("[\n]>+")
r2 = re.compile("[ ]+")
with codecs.open("data/20news.txt","w",encoding='utf-8') as f:
    for doc in dataset.data:
        sents = re.sub(r1, "\n",doc).split("\n\n")
        reg_sents = [re.sub(r2, " ",sent.replace("\n"," ")).strip() for sent in sents]
        f.write('\n'.join(reg_sents)+"\n\n")

