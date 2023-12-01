# active_learning-in-ner
主动学习应用到命名实体识别中的小例子。

按照MNLP计算方法，选择前2千条（五分之一）数据作为新的训练数据，用新训练数据得到的模型在测试集上达到了89%的效果

### 一万条数据在测试集上的效果
```bash
train-Accuracy:0.9051,Precision:0.5575,Recall:0.4470,f1 score:0.4762
test-Accuracy:0.9548,Precision:0.6765,Recall:0.6510,f1 score:0.6280
train-Accuracy:0.9435,Precision:0.6392,Recall:0.5785,f1 score:0.5969
test-Accuracy:0.9552,Precision:0.6361,Recall:0.7013,f1 score:0.6595
train-Accuracy:0.9500,Precision:0.6438,Recall:0.6121,f1 score:0.6254
test-Accuracy:0.9548,Precision:0.6332,Recall:0.7183,f1 score:0.6614
train-Accuracy:0.9543,Precision:0.6591,Recall:0.6246,f1 score:0.6353
test-Accuracy:0.9561,Precision:0.6343,Recall:0.7145,f1 score:0.6643
train-Accuracy:0.9584,Precision:0.6777,Recall:0.6365,f1 score:0.6469
test-Accuracy:0.9547,Precision:0.6283,Recall:0.7211,f1 score:0.6599
train-Accuracy:0.9621,Precision:0.6854,Recall:0.6519,f1 score:0.6628
test-Accuracy:0.9518,Precision:0.5793,Recall:0.7005,f1 score:0.6156
train-Accuracy:0.9650,Precision:0.6916,Recall:0.6622,f1 score:0.6734
test-Accuracy:0.9549,Precision:0.6001,Recall:0.6900,f1 score:0.6312
```
### 主动学习算法选择的2千条数据（五分之一）在测试集上的效果
```bash
train-Accuracy:0.9041,Precision:0.4184,Recall:0.2714,f1 score:0.2940
test-Accuracy:0.9347,Precision:0.4706,Recall:0.4139,f1 score:0.4148
train-Accuracy:0.9684,Precision:0.5796,Recall:0.4824,f1 score:0.5018
test-Accuracy:0.9471,Precision:0.5307,Recall:0.4888,f1 score:0.4982
train-Accuracy:0.9781,Precision:0.5898,Recall:0.5556,f1 score:0.5696
test-Accuracy:0.9484,Precision:0.5138,Recall:0.5313,f1 score:0.5147
train-Accuracy:0.9820,Precision:0.6248,Recall:0.5805,f1 score:0.5841
test-Accuracy:0.9521,Precision:0.5545,Recall:0.5389,f1 score:0.5316
train-Accuracy:0.9839,Precision:0.6215,Recall:0.5976,f1 score:0.6040
test-Accuracy:0.9453,Precision:0.5465,Recall:0.5617,f1 score:0.5413
train-Accuracy:0.9853,Precision:0.6992,Recall:0.6117,f1 score:0.6206
test-Accuracy:0.9478,Precision:0.5544,Recall:0.5796,f1 score:0.5610
train-Accuracy:0.9867,Precision:0.7249,Recall:0.6379,f1 score:0.6559
test-Accuracy:0.9529,Precision:0.5927,Recall:0.6045,f1 score:0.5911
```

>**结论**:主动学习选择的部分数据（五分之一）可以达到原训练集89%的效果