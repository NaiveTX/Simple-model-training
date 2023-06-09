import torch
from sklearn.feature_extraction.text import CountVectorizer
from train import TextClassifier


def preprocess_text(text):
    # 文本预处理的函数
    # 可以根据自己的需求对文本进行预处理
    return text

def prepare_data(train_file):
    texts = []
    labels = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            texts.append(parts[0])
            labels.append(int(parts[1]))

    return texts, labels

def predict_single(text, model, vectorizer):
    model.eval()

    # 进行文本预处理
    text = preprocess_text(text)

    # 将文本转换为词频向量
    text_vector = vectorizer.transform([text]).toarray()
    inputs = torch.Tensor(text_vector)

    # 进行预测
    with torch.no_grad():
        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)

    # 返回预测结果
    return predicted_labels.item()


def main():
    # 加载训练好的模型参数
    model = TextClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('./model/model_epoch10.pt'))

    # 创建CountVectorizer对象，并加载之前保存的向量化器
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)

    # 进行单条文本的预测
    # text = '今年基金的行情一般吧'
    text = '2023年高考理科数学试题(上海卷)'
    predicted_label = predict_single(text, model, vectorizer)
    print('Predicted Label:', predicted_label)


if __name__ == '__main__':
    # 加载训练数据和相关参数
    train_file = './data/train.txt'
    texts, labels = prepare_data(train_file)

    # 定义模型参数
    input_size = 13137
    hidden_size = 512
    num_classes = 2

    # 执行预测
    main()
