# -*- coding: utf-8 -*-

import jieba
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 从csv获取图书介绍数据
csv_path = './book_detail_20180921.csv'
book_df = pd.read_csv(csv_path)
book_df.dropna(inplace=True)
names = book_df['book_name'].values
introduces = book_df['introduction'].values
print(names)

# 准备数据
tagged_data = [TaggedDocument(words=list(jieba.cut(intro)), tags=[name]) 
              for name, intro in zip(names, introduces)]

# 定义模型
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size, alpha=alpha, 
                min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(tagged_data)

# 开始训练
for epoch in range(max_epochs):
    print('iteration: {}'.format(epoch))
    model.train(tagged_data, 
                total_examples=model.corpus_count, 
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save('d2v.model')
print('model saved!')