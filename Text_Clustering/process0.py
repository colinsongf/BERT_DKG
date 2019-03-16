import json
import re

r1 = re.compile("[\n]+")
r2 = re.compile("[\W]{3,}")
defined_words = set(["Information retrieval","Information sciences","Information science"])
with open("data/ai_data_json", "w") as fw1:
    with open("/home/yjc/fc_out_academic.txt") as fr:
        max_len = 3000
        i = 1
        while i<max_len:
            line = fr.readline()
            if not line:
                break
            ob = json.loads(line)
            if set(ob["entities"]).intersection(defined_words) !=set():
                fw1.write(line+"\n")
                i+=1
        print(i)




