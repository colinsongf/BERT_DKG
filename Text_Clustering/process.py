import json
import spacy
nlp = spacy.load('en')
with open("data/ai_data_sents.txt", "w") as fw:
    with open("~/fc_out_academic.txt") as fr:
        len = 10000
        i = 1
        while i<len:
            line = fr.readline()
            if not line:
                break
            doc = nlp(json.loads(line)['paperAbstract'])
            fw.write('\n'.join([sent.text.strip() for sent in doc.sents])+'\n\n')





