import json
import spacy
import re

nlp = spacy.load('en')
r = re.compile("[\n]+")
defined_words = set(["Information retrieval","Information sciences","Information science"])
with open("data/ai_data_sents3000.txt", "w") as fw1:
    with open("data/ai_data_conll3000.txt", "w") as fw2:
        with open("/home/yjc/fc_out_academic.txt") as fr:
            len = 3000
            i = 1
            while i<len:
                line = fr.readline()
                if not line:
                    break
                ob = json.loads(line)
                if set(ob["entities"]).intersection(defined_words) !=set():
                    doc = ob['paperAbstract']
                    doc = re.sub(r, " ", doc)
                    doc = nlp(doc)
                    fw1_ = []
                    fw2.write("-DOCSTART-\n\n")
                    for sent in doc.sents:
                        fw1_.append(sent.text.strip())
                        fw2.write('\n'.join([word.text for word in sent])+"\n\n")
                    fw1.write('\n'.join(fw1_)+"\n\n")
                    fw2.write('\n')
                    i += 1
                    break




