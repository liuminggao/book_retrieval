# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

model = Doc2Vec.load('d2v.model')
count = model.docvecs.count
names = [model.docvecs.index_to_doctag(i) for i in range(count)]
docvecs = model.docvecs.vectors_docs
print('example vector:')
print(names[0])
print(docvecs[0])
assert len(names) == len(docvecs)

tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
n2_docvecs = tsne.fit_transform(docvecs)

for i in range(count):
    plt.scatter(n2_docvecs[i, 0], n2_docvecs[i, 1])
    plt.annotate(names[i],
                 xy=(n2_docvecs[i, 0], n2_docvecs[i, 1]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()