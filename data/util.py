
def fun1():
    f1 = open("ai_data_train_labeled_140.txt","w",encoding="utf8")
    f2 = open("ai_data_dev46.txt","w",encoding="utf8")
    f3 = open("ai_data_test46.txt","w",encoding="utf8")
    n1 = 140
    n2 = 46
    n3 = 46
    fw =f1
    with open("ai_data_train_labeled.txt", encoding="utf8") as f:
        n = 0
        for line in f.readlines():
            if line.startswith("-DOCSTART-"):
                n += 1
                if n>n1:
                    fw = f2
                if n>n1+n2:
                    fw = f3
            fw.write(line)

def fun2():
    fw = open("ai_data_train_unlabeled_1400.txt","w",encoding="utf8")
    n1 = 1400
    with open("ai_data_train_unlabeled.txt", encoding="utf8") as f:
        n = 0
        for line in f.readlines():
            if line.startswith("-DOCSTART-"):
                n += 1
                if n>n1:
                    break
            if line=="\n" or not len(line.strip().split(" "))==1:
                fw.write(line)

def count_tokens_sents_docs():
    with open(r"D:\github\bishe\Text_Clustering\NER_projects\pt_bert_ner\data\conll2003\semi\BIOES\tiny.txt","r", encoding="utf8") as f:
        toks = 0
        sents = 0
        docs = 0
        for line in f.readlines():
            if line.strip() and not line.startswith("-DOCSTART-"):
                toks+=1
            if not line.strip():
                sents+=1
            if line.startswith("-DOCSTART-"):
                docs+=1
        print(toks)
        print(sents)
        print(docs)

def save2db():
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ai_data"]
    col = db["paper_infos"]
    col.drop()
    col = db["paper_infos"]
    with open("ai_conference.txt", "r", encoding="utf8") as f:
        conference = f.readlines()
        lists = [{'_id':i+1,'conference':c.strip(), 'fields':[], 'tecs':[]} for i,c in enumerate(conference)]
        col.insert_many(lists)
    with open(r"D:\github\bishe\Text_Clustering\NER_projects\pt_bert_ner\no_X_version\output_predict_new_doc_reg\prediction_entities.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
        fields = [set([i.strip() for i in l.strip().split(',')[0].split('\t')]) for l in lines]
        tecs = [set([i.strip() for i in l.strip().split(',')[1].split('\t')]) for l in lines]
        for i in range(len(lines)):
            col.update({'_id':i+1},{'$set':{'fields':list(fields[i]),'tecs':list(tecs[i])}})

count_tokens_sents_docs()