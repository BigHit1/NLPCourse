import glob
import logging
import multiprocessing
import os

import jieba
import torch
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader


# 将文本分词后转化为分好词的文本

# 获取停用词列表


def getStopwords():
    with open('hit_stopwords.txt') as hit:
        hit_stopwords = hit.read().split('\n')
    with open('cn_stopwords.txt') as cn:
        cn_stopwords = cn.read().split('\n')
    return hit_stopwords + cn_stopwords


def getText():
    path_list = glob.glob(r"train/*")
    catagory = 0
    index = 0
    vocabulary = []
    for path in path_list:
        catagory += 1
        catagory_name = path.split('-')[1]
        text_list = glob.glob(f"{path}/*")
        for text_path in text_list:
            index += 1
            print(text_path)
            with open(text_path, encoding="gb18030", errors='ignore') as file:
                text = file.read()
                word_list = jieba.lcut(text, cut_all=True)

            stopwords = getStopwords()
            with open(f"train_vec/{index}.txt", 'w') as file:
                file.write(str(catagory) + '\n')
                file.write(catagory_name + '\n')
                subvocab = []
                for word in word_list:
                    if word != "\n" and word not in stopwords and word != " ":
                        file.write(word + " ")
                        subvocab.append(word)
            vocabulary.append(subvocab)
    print(len(vocabulary))

    with open('vocabulary.txt', 'w') as vocab:
        for sub in vocabulary:
            for word in sub:
                vocab.write(word + " ")
            vocab.write('\n')


# getText()


# 将词语转化为长度200的向量
def encoding():
    with open('vocabulary.txt') as vocab:
        raw_sentences = vocab.read().split('\n')
        raw_sentences = raw_sentences[0:len(raw_sentences) - 2]
    sentences = [s.split() for s in raw_sentences]

    model = Word2Vec(sentences, vector_size=200, window=10, min_count=10, workers=multiprocessing.cpu_count())
    model.save('vec.model')


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# encoding()
# model = Word2Vec.load('vec.model')
# wv = model.wv
# print(wv.similarity('艺术','科技'))

# 获取文本的平均长度
def getAverage():
    sum = 0
    min = 20000
    max = 0
    with open('vocabulary.txt') as vocab:
        raw_sentences = vocab.read().split('\n')
        raw_sentences = raw_sentences[0:len(raw_sentences) - 2]
    sentences = [s.split() for s in raw_sentences]
    i = 0
    for sentence in sentences:
        i += 1
        if len(sentence) == 0:
            print(i)
        if len(sentence) < min:
            min = len(sentence)
        if len(sentence) > max:
            max = len(sentence)
        sum += len(sentence)

    print(sum // len(sentences))
    print(min)
    print(max)


# getAverage()

# 制作dataload

class fudanDataSet(Dataset):

    def __init__(self):
        super().__init__()
        self.data_path = 'train_vec'
        self.data_size = [200, 200]
        self.wv = Word2Vec.load('vec.model').wv

    def __len__(self):
        return len(glob.glob(f"{self.data_path}/*"))

    # 根据index返回样本和标签的元组 样本是tensor
    def __getitem__(self, index):
        index += 1
        path = os.path.join(self.data_path, f'{index}.txt')
        with open(path) as text:
            remark = text.readline().split('\n')[0]
            remark_name = text.readline().split('\n')[0]
            raw_data = text.read()
        raw_data = raw_data.split()
        data = []
        i = 0
        for s in raw_data:
            try:
                self.wv[s]
            except:
                continue
            data.append(self.wv[s])
            i += 1
            if i == self.data_size[0]:
                break
        if i < self.data_size[0]:
            times = self.data_size[0] // i
            left = self.data_size[0] - times * i
            data = data[0:left] + data * times

        data = torch.Tensor(data)
        return data, remark



# dataset = fudanDataSet()
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for index, (data, remark) in enumerate(dataloader):
#     print(index)
#     print(data)
#     print(data.shape)
#     print(data.dtype)
#     print(remark)
def getTextTest():
    path_list = glob.glob(r"test/*")
    catagory = 0
    index = 0
    vocabulary = []
    for path in path_list:
        catagory += 1
        catagory_name = path.split('-')[1]
        text_list = glob.glob(f"{path}/*")
        for text_path in text_list:
            index += 1
            print(text_path)
            with open(text_path, encoding="gb18030", errors='ignore') as file:
                text = file.read()
                word_list = jieba.lcut(text, cut_all=True)

            stopwords = getStopwords()
            with open(f"test_vec/{index}.txt", 'w') as file:
                file.write(str(catagory) + '\n')
                file.write(catagory_name + '\n')
                subvocab = []
                for word in word_list:
                    if word != "\n" and word not in stopwords and word != " ":
                        file.write(word + " ")
                        subvocab.append(word)
            vocabulary.append(subvocab)
    print(len(vocabulary))

    with open('vocabulary_test.txt', 'w') as vocab:
        for sub in vocabulary:
            for word in sub:
                vocab.write(word + " ")
            vocab.write('\n')

# getTextTest()

class fudanDataSetTest(Dataset):

    def __init__(self):
        super().__init__()
        self.data_path = 'test_vec'
        self.data_size = [200, 200]
        self.wv = Word2Vec.load('vec.model').wv

    def __len__(self):
        return len(glob.glob(f"{self.data_path}/*"))

    # 根据index返回样本和标签的元组 样本是tensor
    def __getitem__(self, index):
        index += 1
        path = os.path.join(self.data_path, f'{index}.txt')
        with open(path) as text:
            remark = text.readline().split('\n')[0]
            remark_name = text.readline().split('\n')[0]
            raw_data = text.read()
        raw_data = raw_data.split()
        data = []
        i = 0
        for s in raw_data:
            try:
                self.wv[s]
            except:
                continue
            data.append(self.wv[s])
            i += 1
            if i == self.data_size[0]:
                break
        if i < self.data_size[0] and i != 0:
            times = self.data_size[0] // i
            left = self.data_size[0] - times * i
            data = data[0:left] + data * times
        if i == 0:
            data = torch.randn(self.data_size)

        data = torch.Tensor(data)
        return data, remark
# dataset = fudanDataSetTest()
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for index, (data, remark) in enumerate(dataloader):
#     print(index)
#     print(data)
#     print(data.shape)
#     print(data.dtype)
#     print(remark)