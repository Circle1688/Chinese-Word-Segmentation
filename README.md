# Chinese-Word-Segmentation
基于双向GRU的通用新闻中文分词模型

## 数据集
分词-中文-新闻领域数据集PKU
https://modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku/summary
- 使用BIES标签体系标注
```
在 这 辞 旧 迎 新 的 美 好 时 刻 ， 我 祝 大 家 新 年 快 乐 ， 家 庭 幸 福 ！
S-CWS S-CWS B-CWS I-CWS I-CWS E-CWS S-CWS B-CWS E-CWS B-CWS E-CWS S-CWS S-CWS S-CWS B-CWS E-CWS B-CWS E-CWS B-CWS E-CWS S-CWS B-CWS E-CWS B-CWS E-CWS S-CWS
```

## 网络结构
双向GRU

## 训练
模型采用1张NVIDIA A6000训练
### 超参数
```
train_epochs=10
max_sequence_length=256
batch_size=125
learning_rate=5e-5
optimizer=AdamW
```
