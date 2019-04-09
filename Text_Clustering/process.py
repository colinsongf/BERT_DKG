import json
import spacy
import re


def json_to_conll():
    nlp = spacy.load('en')
    r1 = re.compile("[\n]+")
    r2 = re.compile("[\W]{3,}")
    defined_words = set(["Information retrieval", "Information sciences", "Information science"])
    with open("data/ai_data_sents3000.txt", "w") as fw1:
        with open("data/ai_data_train_labeded.txt", "w") as fw2:
            with open("data/ai_data_train_unlabeded.txt", "w") as fw3:
                with open("data/ai_data_test.txt", "w") as fw4:
                    with open("/home/yjc/fc_out_academic.txt") as fr:
                        max_len = 3000
                        i = 1
                        while i < max_len:
                            line = fr.readline()
                            if not line:
                                break
                            ob = json.loads(line)
                            if set(ob["entities"]).intersection(defined_words) != set():
                                doc = ob['paperAbstract']
                                doc = re.sub(r1, " ", doc)
                                doc = re.sub(r2, " ", doc)
                                if i <= 250:
                                    fw_c = fw2
                                elif i <= 2750:
                                    fw_c = fw3
                                else:
                                    fw_c = fw4
                                if len(doc.strip(" ")) > 20 and len(doc.strip(" ")) < 1000:
                                    doc = nlp(doc)
                                    fw1_ = []
                                    fw_c.write("-DOCSTART- O\n\n")
                                    for sent in doc.sents:
                                        if len(sent.text.strip()) > 10:
                                            fw1_.append(sent.text.strip())
                                            fw_c.write('\n'.join([word.text + " O" for word in sent]) + "\n\n")
                                    fw1.write('\n'.join(fw1_) + "\n\n")
                                    fw_c.write('\n')
                                    i += 1
                        print(i)


def json_to_conll2():
    nlp = spacy.load('en')
    r1 = re.compile("[\n]+")
    r2 = re.compile("[\W]{3,}")
    r3 = re.compile("[\xa0 \t]")
    #defined_words = set(["Information retrieval", "Information sciences", "Information science"])
    defined_words = []
    with open("data/ai_data_sents_new.txt", "w") as fw1:
        with open("data/ai_data_to_predict_new.txt", "w") as fw2:
            with open("/home/yjc/fc_out_academic.txt") as fr:
                max_len = 8489
                i = 1
                while i < max_len:
                    line = fr.readline()
                    if not line:
                        break
                    ob = json.loads(line)
                    if set(ob["entities"]).intersection(defined_words) != set():
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
                            i += 1
                print(i)


def extract_from_json():
    r1 = re.compile("[\n]+")
    r2 = re.compile("[\W]{3,}")
    defined_words = set(["Information sciences", "Information science", "Information", "Information Management"])
    with open("data/ai_data_json", "w") as fw1:
        with open("/home/yjc/fc_out_academic.txt") as fr:
            max_len = 10000
            i = 1
            while i < max_len:
                line = fr.readline()
                if not line:
                    break
                ob = json.loads(line)
                if set(ob["entities"]).intersection(defined_words) != set():
                    fw1.write(line)
                    i += 1
            print(i)


json_to_conll2()
