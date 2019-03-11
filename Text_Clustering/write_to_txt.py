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
import spacy
nlp = spacy.load("en")
with open("data/20news.txt","w") as f:
    for doc in dataset.data:
        doc = nlp(doc)
        doc_str = []
        for sent in doc.sents:
            doc_str.append(sent.string.replace("\n"," "))
        f.write('\n'.join(doc_str)+"\n\n")

