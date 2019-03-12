import json
import spacy
import re

nlp = spacy.load('en')
r = re.compile("[\n]+")
with open("data/ai_data_sents.txt", "w") as fw:
    with open("home/yjc/fc_out_academic.txt") as fr:
        len = 10000
        i = 1
        while i<len:
            line = fr.readline()
            if not line:
                break
            doc = json.loads(line)['paperAbstract']
            doc = re.sub(r, " ", doc)
            doc = nlp(doc)
            fw.write('\n'.join([sent.text.strip() for sent in doc.sents])+'\n\n')
            i += 1





