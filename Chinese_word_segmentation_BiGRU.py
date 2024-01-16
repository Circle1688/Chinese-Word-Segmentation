import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from tqdm import tqdm

if torch.cuda.is_available():
    logger.info('use cuda')
else:
    logger.info('use cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WordSegmentationModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_gru_layers, vocab, dropout):
        super(WordSegmentationModel, self).__init__()
        self.num_classes = 5
        self.embedding = nn.Embedding(len(vocab), input_dim, padding_idx=0)
        self.biGru = nn.GRU(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_gru_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)
        self.classify = nn.Linear(hidden_size * 2, self.num_classes)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    def forward(self, x, y=None):
        # input_shape: (batch_size, sentence_length)
        x = self.embedding(x)  # output_shape: (batch_size, sentence_length, input_dim)
        outputs, _ = self.biGru(x)  # output_shape: (batch_size, sentence_length, hidden_size * 2)
        y_pred = self.classify(outputs)
        if y is not None:
            return self.loss(y_pred.view(-1, self.num_classes), y.view(-1))
        else:
            return y_pred

class CWSDataset(Dataset):
    def __init__(self, data_path, vocab, max_sequence_length):
        super(CWSDataset, self).__init__()
        self.vocab = vocab
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, 'r', encoding='utf8') as file:
            words = []
            labels = []
            for line in file:
                line = line.strip()
                if not line:
                    sequence = sentence_to_sequence(words, self.vocab)
                    label = labels_mapping(labels)
                    sequence, label = self.padding(sequence, label)
                    sequence = torch.LongTensor(sequence)
                    label = torch.LongTensor(label)
                    self.data.append([sequence, label])
                    words = []
                    labels = []
                    continue
                line_words = line.split('	')
                words.append(line_words[0])
                labels.append(line_words[1])

                # 部分数据，减少训练时间
                # if len(self.data) > 10000:
                #     break

    def padding(self, sequence, label):
        """
        用于词表转换后, 截断或者填充句子
        :param max_sequence: 最大截断数
        :param sequence: 句子序列
        :return:
        """
        if len(sequence) >= self.max_sequence_length:
            return (sequence[:self.max_sequence_length],
                    label[:self.max_sequence_length])
        else:
            return (sequence + [0] * (self.max_sequence_length - len(sequence)),
                    label + [-100] * (self.max_sequence_length - len(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['[UNK]']) for char in sentence]
    return sequence

def sequence_to_text(sequence, vocab):
    keys = list(vocab.keys())
    text = [keys[index] for index in sequence]
    return text

def padding_sequence(sequence, max_sequence_length):
    if len(sequence) >= max_sequence_length:
        return sequence[:max_sequence_length]
    else:
        return sequence + [0] * (max_sequence_length - len(sequence))
def labels_mapping(labels):
    """
    序列标注标签体系(B、I、E、S),四个标签分别表示单字处理单词的起始、中间、终止位置或者该单字独立成词
    - PAK 对应 -100
    - B-CWS 对应 0
    - I-CWS 对应 1
    - E-CWS 对应 2
    - S-CWS 对应 3
    :param labels:
    :return:
    """
    mapping = []
    for label in labels:
        if label == 'B-CWS':
            mapping.append(0)
        elif label == 'I-CWS':
            mapping.append(1)
        elif label == 'E-CWS':
            mapping.append(2)
        elif label == 'S-CWS':
            mapping.append(3)
    return mapping

def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            char = line.strip()
            vocab[char] = i
    return vocab

def build_dataset(data_path, vocab, max_sequence_length, batch_size):
    dataset = CWSDataset(data_path, vocab, max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_seg_list(sentence, label):
    tmp_string = ''
    seg_list = []
    for i, p in enumerate(label):
        tmp_string += sentence[i]
        if p == 2 or p == 3:
            seg_list.append(tmp_string)
            tmp_string = ''
    return seg_list

def evaluate(model, test_data_path, vocab, max_sequence_length, batch_size):
    model.eval()
    dataloader = build_dataset(test_data_path, vocab, max_sequence_length, batch_size)
    correct_all = []
    for x, y in tqdm(dataloader, desc='评估分词正确率'):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)
            for y_p, y_t, x_t in zip(y_pred, y, x):
                y_p = torch.argmax(y_p, dim=-1)

                y_t = y_t.tolist()
                while -100 in y_t:
                    y_t.remove(-100)  # 移除所有-100
                y_p = y_p.tolist()
                y_p = y_p[:len(y_t)]  # 截断

                # decode -> 词序列
                sentence = sequence_to_text(x_t, vocab)

                # 根据预测序列分词
                seg_t = get_seg_list(sentence, y_t)
                seg_p = get_seg_list(sentence, y_p)

                set_p = set(seg_p)
                set_t = set(seg_t)
                common = set_p.intersection(set_t)  # 求交集
                correct_all.append(len(list(common)) / len(y_t))  # 计算每个句子的分词正确率  正确数 / 总词数

    logger.info(f'平均正确率: {np.mean(correct_all)}')
    return np.mean(correct_all)

def predict(model_path, vocab_path, input_strings, max_sequence_length, char_dim, hidden_size, num_gru_layers):
    vocab = build_vocab(vocab_path)
    model = WordSegmentationModel(char_dim, hidden_size, num_gru_layers, vocab, dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for input_string in input_strings:
        x = sentence_to_sequence(input_string, vocab)
        x = padding_sequence(x, max_sequence_length)
        with torch.no_grad():
            x = torch.LongTensor([x])
            x = x.to(device)
            y_pred = model.forward(x)[0]
            result = torch.argmax(y_pred, dim=-1)
            result = result.tolist()[:len(input_string)]
            for i, p in enumerate(result):
                if p == 2 or p == 3:
                    print(input_string[i], end=' ')
                else:
                    print(input_string[i], end='')
            print()



def main():
    train_epochs = 10
    batch_size = 125  # 125
    learning_rate = 5e-5
    log_step = 100


    train_data_path = 'dataset/train.txt'
    test_data_path = 'dataset/test.txt'
    model_save_path = 'model.pth'
    vocab_path = 'vocab.txt'
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(train_data_path, vocab, max_sequence_length, batch_size)
    model = WordSegmentationModel(char_dim, hidden_size, num_gru_layers, vocab, dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print('=========开始训练=========')
    log = []
    steps = 0
    for epoch in range(train_epochs):
        model.train()
        watch_loss = []
        start = time.time()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            steps += 1
            if steps % log_step == 0:
                logger.info("=========\n第%d步平均loss:%f 耗时:%.2fs" % (steps, np.mean(watch_loss), time.time() - start))
                start = time.time()

        mean_loss = np.mean(watch_loss)

        logger.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model, test_data_path, vocab, max_sequence_length, batch_size)
        log.append([acc, mean_loss])
        if epoch != 0:
            log_display(log)  # 每轮更新一次曲线

    # log_display(log)

    torch.save(model.state_dict(), model_save_path)
    return





if __name__ == '__main__':
    max_sequence_length = 256
    char_dim = 100
    hidden_size = 200
    num_gru_layers = 6
    dropout = 0.1

    main()

    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict("model.pth",
            "vocab.txt",
            input_strings,
            max_sequence_length,
            char_dim,
            hidden_size,
            num_gru_layers)
