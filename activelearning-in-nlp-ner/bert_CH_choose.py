import torch
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.device_count()

from torch.optim.sgd import SGD
from torchcrf import CRF
from tqdm import tqdm
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(1203)
torch.manual_seed(1203)
torch.cuda.manual_seed(1203)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import config
def build_corpus(path):
    """读取数据"""
    # assert split in ['train', 'dev', 'test']
    word_lists, tag_lists, unique_label = [], [], []
    for line in open(path, 'r', encoding='utf-8'):
        json_data = json.loads(line)
        word_lists.append(json_data['text'])
        tag_lists.append(json_data['labels'])
        unique_label.extend(json_data['labels'])
    return word_lists, tag_lists, unique_label
words, labels, unique_label = build_corpus('D:\E\A文本标注\英文标注\整理\新闻-中文-BIO.json')
test_words = words[int(len(words)*0.8):]
test_labels = labels[int(len(labels)*0.8):]
unique_label = list(set(unique_label))    # unique_label
# 一万条数据按MNLP计算结果选择前2千条数据
train_words, train_labels, unique_label2 = build_corpus('D:\project\\vscode\HF_BERT\\bert_ner\AL\新闻-中文-BIO-2000.json')

label2id = {v:k for k,v in enumerate(sorted(unique_label))}
id2label = {v:k for k,v in label2id.items()}

length_labels = len(unique_label)
lenght_train_texts = len(train_words)
# lenght_dev_texts = len(dev_words)
lenght_test_texts = len(test_words)
# bert-base-chinese路径
tokenizer = BertTokenizerFast.from_pretrained('D:\project\\vscode\HF_BERT\models\\bert-base-chinese\\', do_lower_case=True)

def align_label(texts, labels):
    # 首先tokenizer输入文本
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True, is_split_into_words = True)
  # 获取word_ids
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []
    # 采用上述的第一中方法来调整标签，使得标签与输入数据对其。
    for word_idx in word_ids:
        # 如果token不在word_ids内，则用 “-100” 填充
        if word_idx is None:
            label_ids.append(-100)
        # 如果token在word_ids内，且word_idx不为None，则从labels_to_ids获取label id
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(label2id[labels[word_idx]])
            except:
                label_ids.append(-100)
        # 如果token在word_ids内，且word_idx为None
        else:
            try:
                label_ids.append(label2id[labels[word_idx]] if False else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

# 构建自己的数据集类
class DataSequence(torch.utils.data.Dataset):
    def __init__(self, words, labels):
        word = words
        label = labels
        # tokenizer 向量化文本
        # data = []
        # label = []
        # for _ in df:
        #     data.append(_['text'])
        #     label.append(_['labels'])
        self.texts = [tokenizer(i, padding='max_length',max_length = 512,truncation=True, return_tensors="pt",is_split_into_words = True) for i in word]
        # 对齐标签
        self.labels = [align_label(i,j) for i,j in zip(word, label)]
        a = 1
        

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
    

class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('D:\project\\vscode\HF_BERT\models\\bert-base-chinese\\',
                                                               num_labels=length_labels)
        self.bert.resize_token_embeddings(len(tokenizer))
        # self.crf = CRF(num_tags=length_labels, batch_first=True)

    def forward(self, input_id, mask, label):
        _loss, logits = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        # out = self.crf.decode(emissions=logits[0][label!=-100].unsqueeze(0))
        # newl = label.unsqueeze(0)
        # loss = -self.crf(emissions=logits[0][label!=-100].unsqueeze(0), tags=label[label!=-100].unsqueeze(0))

        return _loss, logits

def predict(model, train_words, train_labels, test_words, test_labels):
    """主动学习部分：按MNLP算法选择前2000条数据

    Args:
        model (_type_): 模型
        train_words (_type_): 训练集
        train_labels (_type_): 训练集标签
        test_words (_type_): 测试集
        test_labels (_type_): 测试集标签

    Returns:
        []: [(MNLP分数, 样本index), (MNLP分数, 样本index), ...]
    """
    # 定义训练和验证集数据
    train_dataset = DataSequence(train_words, train_labels)
    # val_dataset = DataSequence(test_words, test_labels)
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1)
    # 判断是否使用GPU，如果有，尽量使用，可以加快训练速度
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义优化器
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # 开始训练循环
    logfile_name = 'bert_log.txt'
    model.eval()
    re = []
    # total_acc_val = 0
    # total_loss_val = 0
    for k,(train_data, train_label) in enumerate(train_dataloader):
        # 从train_data中获取mask和input_id
        train_label = train_label[0].to(device)
        # t = train_data['input_ids']
        mask = train_data['attention_mask'].squeeze(0).to(device)
        input_id = train_data['input_ids'].squeeze(0).to(device)
        # print(tokenizer.convert_ids_to_tokens(train_data['input_ids'][0][0]))
        # 梯度清零！！
        optimizer.zero_grad()
        # 输入模型训练结果：损失及分类概率
        loss, logits = model(input_id, mask, train_label)
        # 清楚无效token对应的结果
        logits_clean = logits[0][train_label != -100]
        label_clean = train_label[train_label != -100]
        # 获取概率值最大的预测
        # predictions = logits_clean.argmax(dim=1)
        # MNLP公式（核心）
        MNLP_score = torch.sum(torch.log(logits_clean.max(-1)[0]))/logits_clean.shape[0]    # MNLP
        re.append((MNLP_score.item(), k))
    return re

# 训练模型、测试模型部分（用于查看主动学习的效果，load用主动学习选择的数据训练得到的模型）
def train_loop(model, train_words, train_labels, test_words, test_labels):
    # 定义训练和验证集数据
    train_dataset = DataSequence(train_words, train_labels)
    val_dataset = DataSequence(test_words, test_labels)
    # 批量获取训练和验证集数据
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1)
    # 判断是否使用GPU，如果有，尽量使用，可以加快训练速度
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义优化器
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
    # 开始训练循环
    logfile_name = 'bert_log.txt'
    for epoch_num in range(epochs):
        y_true = []
        y_pred = []
        # 训练模型
        model.train()
        # 按批量循环训练模型
        for train_data, train_label in tqdm(train_dataloader):
            # 从train_data中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)
            # 梯度清零！！
            optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            loss, logits = model(input_id, mask, train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            # 获取概率值最大的预测
            predictions = logits_clean.argmax(dim=1)

            y_true.extend(label_clean.data.cpu())
            y_pred.extend(predictions.data.cpu())
            loss.backward()
            # 参数更新
            optimizer.step()
        log_file = open(logfile_name, 'a', encoding='utf-8')
        print('train-'+'Accuracy:'+"%.4f" % accuracy_score(y_true,y_pred)+',Precision:'+"%.4f" % precision_score(y_true,y_pred,average='macro')+',Recall:'+"%.4f" % recall_score(y_true, y_pred, average='macro')+',f1 score:'+"%.4f" % f1_score(y_true, y_pred, average='macro')
          , file=log_file)
        print('train-'+'Accuracy:'+"%.4f" % accuracy_score(y_true,y_pred)+',Precision:'+"%.4f" % precision_score(y_true,y_pred,average='macro')+',Recall:'+"%.4f" % recall_score(y_true, y_pred, average='macro')+',f1 score:'+"%.4f" % f1_score(y_true, y_pred, average='macro')
          )
        log_file.close()
        # 模型评估
        model.eval()
        y_true, y_pred = [], []
        for train_data, train_label in val_dataloader:
            # 从train_data中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)
            # 梯度清零！！
            # optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            loss, logits = model(input_id, mask, train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            # 获取概率值最大的预测
            predictions = logits_clean.argmax(dim=1)

            y_true.extend(label_clean.data.cpu())
            y_pred.extend(predictions.data.cpu())
        log_file = open(logfile_name, 'a', encoding='utf-8')
        print('test-'+'Accuracy:'+"%.4f" % accuracy_score(y_true,y_pred)+',Precision:'+"%.4f" % precision_score(y_true,y_pred,average='macro')+',Recall:'+"%.4f" % recall_score(y_true, y_pred, average='macro')+',f1 score:'+"%.4f" % f1_score(y_true, y_pred, average='macro')
          , file=log_file)
        print('test-'+'Accuracy:'+"%.4f" % accuracy_score(y_true,y_pred)+',Precision:'+"%.4f" % precision_score(y_true,y_pred,average='macro')+',Recall:'+"%.4f" % recall_score(y_true, y_pred, average='macro')+',f1 score:'+"%.4f" % f1_score(y_true, y_pred, average='macro')
          )
        log_file.close()
learning_rate = 1e-3
epochs = 7

# model = BertModel()
# train_loop(model, train_words, train_labels, test_words, test_labels)






model = torch.load('D:\project\\vscode\HF_BERT\\bert_ner\model_saved\ch_unbatch_7_1e-3.pt').cuda()
MNLP_score_list = predict(model, train_words, train_labels, test_words, test_labels)
MNLP_score_list_sorted = sorted(MNLP_score_list, reverse=True)
text_choose_idx = [_[1] for _ in MNLP_score_list_sorted[:2000]]
text_choose = [words[_] for _ in text_choose_idx]
label_choose = [labels[_] for _ in text_choose_idx]
file2 = open('新闻-中文-BIO-2000.json', 'a', encoding='utf-8')
for x,y in zip(text_choose,label_choose):
    d = dict()
    d['text'] = x
    d['labels'] = y
    s = json.dumps(d, ensure_ascii=False)
    file2.writelines(s)
    file2.writelines('\n')
file2.close()

# torch.save(model, 'ch_unbatch_7_1e-3.pt')
