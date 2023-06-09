import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer


# 定义神经网络模型结构，
class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        # 定义第一层线性层，输入的维度是词向量维度，输出的维度是隐层维度，目的是将输入进行映射，进一步提取特征
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 非线性激活，增加训练非线性
        self.relu = nn.ReLU()
        # 第二个线性层，输入输入输出维度都是隐层维度，目的是让模型有可学习的参数
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 第三个线性层，输入维度都是隐层维度，输出是分类目标的维度，目的是得到分类结果
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def prepare_data(train_file):
    texts = []
    labels = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            texts.append(parts[0])
            labels.append(int(parts[1]))

    return texts, labels


# 自定义数据集类，可以根据数据集生成词表，并根据原始文本中的词在词表中出现的频率将原始文本变成词向量
# 比如“你好”，编码为[0,0,0,0,0,0,0,1,0,0,1],词表大小为向量的长度，1表示词表中对应位置的词出现了一次
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.vectorizer = CountVectorizer()

        # 将文本转换为词频向量，并把稀疏矩阵变稠密
        self.texts = self.vectorizer.fit_transform(self.texts).toarray()

    def __len__(self):
        return len(self.texts)

    # 可以根据id来获取对应数据和标签
    def __getitem__(self, index):
        text = torch.Tensor(self.texts[index])
        label = torch.Tensor([self.labels[index]])
        return text, label


if not os.path.exists('model'):
    os.makedirs('model')

# 判断是否有gpu加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型训练
def train_model(train_dataset, input_size, hidden_size, num_classes, batch_size, epoch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("数据加载完毕-------------")
    model = TextClassifier(input_size, hidden_size, num_classes)
    model.to(device)
    # 交叉熵损失函数是一个用于多分类问题的损失函数，可以将模型输出的概率分布与标签的索引进行比较，衡量两者之间的差异性，
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    batch_i = 0
    print("模型开始训练-------------")
    for epoch in range(epoch_size):
        start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_i % 10 == 0:
                print(f'loss{loss}')

            # 模型参数的梯度置零，以准备计算新的梯度。
            optimizer.zero_grad()
            # 反向传播，计算模型参数的梯度。
            loss.backward()
            # 更新模型的参数，根据计算的梯度值进行参数更新。
            optimizer.step()
            batch_i += 1

        # 每个epoch保存模型
        end = time.time()

        print(f'\nmodel_epoch{epoch}----time_cost{end - start}\n')
        model_path = os.path.join('model', f'model_epoch{epoch + 1}.pt')
        torch.save(model.state_dict(), model_path)


# 主函数
def main():
    train_file = './data/train.txt'
    input_size = 13137  # 词汇表大小
    hidden_size = 512  # 隐层大小
    num_classes = 2  # 分类类别数量
    batch_size = 24  # 一次输入的训练数据条数
    epoch_size = 10  # 所有训练数据都过一遍模型算一个epoch

    print("训练开始，数据加载-------------")
    # 将训练数据和标签加载出来
    texts, labels = prepare_data(train_file)
    train_dataset = TextDataset(texts, labels)

    # 模型训练
    train_model(train_dataset, input_size, hidden_size, num_classes, batch_size, epoch_size)


if __name__ == '__main__':
    main()
