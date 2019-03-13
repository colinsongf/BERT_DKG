import json
import spacy
import re

nlp = spacy.load('en')
r = re.compile("[\n]+")
defined_words = set(["Information retrieval","Information sciences","Information science"])
with open("data/ai_data_sents3000.txt", "w") as fw1:
    with open("data/ai_data_train_labeded.txt", "w") as fw2:
        with open("data/ai_data_train_unlabeded.txt", "w") as fw3:
            with open("data/ai_data_test.txt", "w") as fw4:
                with open("/home/yjc/fc_out_academic.txt") as fr:
                    max_len = 3000
                    i = 1
                    while i<max_len:
                        line = fr.readline()
                        if not line:
                            break
                        ob = json.loads(line)
                        if set(ob["entities"]).intersection(defined_words) !=set():
                            doc = ob['paperAbstract']
                            doc = re.sub(r, " ", doc)
                            if i<=250:
                                fw_c = fw2
                            elif i<=2750:
                                fw_c = fw3
                            else:
                                fw_c = fw4
                            if len(doc) >20 and len(doc)<512:
                                doc = nlp(doc)
                                fw1_ = []
                                fw_c.write("-DOCSTART-\n\n")
                                for sent in doc.sents:
                                    if len(sent.text.strip())>10:
                                        fw1_.append(sent.text.strip())
                                        fw_c.write('\n'.join([word.text for word in sent])+"\n\n")
                                fw1.write('\n'.join(fw1_)+"\n\n")
                                fw_c.write('\n')
                                i += 1




