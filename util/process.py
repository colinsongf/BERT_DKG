import json
import re

import spacy


def json_to_conll():
    nlp = spacy.load('en')
    r1 = re.compile("[\n]+")
    r2 = re.compile("[\W]{3,}")
    r3 = re.compile('''[^\w,.?!\-;:'"<>]''')
    defined_words = set(["Information retrieval", "Information sciences", "Information science"])
    fc = open("data/ai_conference.txt", "w")
    with open("data/ai_data_sents_new.txt", "w") as fw1:
        with open("NER_projects/pt_bert_ner/semi-data/ai_data_to_predict_new.txt", "w") as fw2:
            with open("/home/yjc/fc_out_academic.txt") as fr:
                max_len = 8489
                i = 1
                while i < max_len:
                    line = fr.readline()
                    if not line:
                        break
                    ob = json.loads(line)
                    if True:
                    # if set(ob["entities"]).intersection(defined_words) != set():
                        doc = ob['paperAbstract']
                        doc = re.sub(r1, " ", doc)
                        doc = re.sub(r2, " ", doc)
                        fw_c = fw2
                        if len(doc.strip(" ")) > 20 and len(doc.strip(" ")) < 1000:
                            doc = nlp(doc)
                            fw1_ = []
                            fw_c.write("-DOCSTART- O\n\n")
                            for sent in doc.sents:
                                if len(sent.text.strip()) > 10:
                                    fw1_.append(sent.text.strip())
                                    fw_c.write('\n'.join([re.sub(r3, "", word.text) + " O" for word in sent if
                                                          re.sub(r3, "", word.text)]) + "\n\n")
                            fw1.write('\n'.join(fw1_) + "\n\n")
                            fw_c.write('\n')
                            fc.write(ob['journalName'] + "\n")
                            i += 1
                print(i)


json_to_conll()
