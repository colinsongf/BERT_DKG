# MD2vec

# NER_projects
## bert_model
需下载预训练模型，解压后有如下三个：
- `bert_config.json`
- `pytorch_model.bin`
- `vocab.txt`

## data
- conll2003: conll 2003数据集
    - full: 未经拆分的全量数据
        - BIO: BIO标注格式，其中`tiny.txt`是用于调试的数据集
        - BIOES: BIOES标注格式，通过`bio2bioes.py`文件转换而来
            
        |文件名|文档数|句子数|词数|
        |----|-----|----|----|
        |train|946|14987|203621|
        |dev|216|3466|51362|
        |test|231|3684|46435|
        |tiny|2|30|658|
            
    - semi: 按照1:10的比例拆分训练数据得到
        - BIOES: 采用BIOES标注格式
        
        |文件名|文档数|句子数|词数|
        |----|-----|----|----|
        |train|86|1356|17028|
        |train_unlabeled|860|13631|186593|
        |dev|216|3466|51362|
        |test|231|3684|46435|
        |tiny|2|30|658|
        |tiny_unlabeled|20|235|3446|
            
- ai: 8488篇AI领域论文的摘要信息
    - semi:
        - BIOES:
        
        |文件名|文档数|句子数|词数|备注|
        |----|-----|----|----|---|
        |data_all|8488|1028088|59635|标注默认为'O'|
        |train|140|972|17074|人工标注|
        |train_unlabeled|1400|9760|168760|用于半监督学习的部分未标注数据，标注默认为'O'|
        |dev|46|335|5801|人工标注|
        |test|46|309|5413|人工标注|

## models
配置文件为`task_config.yaml`
1. `cal_X_loss`：可以选择是否将label为`X`的token的loss也计算进总loss中。
2. `ssl`：可以选择是否使用半监督熵正则化的方法将无标注数据也计算一个loss，然后融合到有标注数据的loss中。
当为`true`时，必须保证相应数据目录下有无标注数据`train_unlabeled.txt`文件。
3. `doc_level`：可以选择是否使用文档层面的输入形式。两种情况都要保证输入序列不超过`max_seq_length`参数以及
BERT预训练模型中`max_position_embeddings`参数。为`false`时，输入就是一个句子；
为`true`时，输入为同一篇文档中尽可能多的句子，当一整篇文档的所有句子不能作为输入时，会进行切分而不是直接丢弃多余的。

模型结构分为`embedder`,`encoder`,`decoder`：
1. `embedder`：可选择随机初始化`RandomEmbed`或者用BERT预训练模型`BertEmbed`；
2. `encoder`： 中间层，可选 `MultiAttn`（等价于BERT里面的`Multi-Head Attention`） 或者 `BiLSTM`
3. `decoder`：输出层，可选`SoftmaxDecoder`或者`CRFDecoder`

