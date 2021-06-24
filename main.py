import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

'''''
data = pd.read_csv('Reviews.csv')
data.drop(data.columns[[2, 3, 4]], axis = 1, inplace = True)
print(data)
count = CountVectorizer()
tfidf = TfidfVectorizer()

bag = count.fit_transform(data.Text)
bag_tfidf = tfidf.fit_transform(data.Text)

# print("Word bag = ",bag.toarray())
# print('\n')
# np.set_printoptions(precision = 2)
# # print(repr(count.vocabulary))
# print('tf-idf',bag_tfidf.toarray())

df = tfidf.fit_transform(data['Text'])
ind = np.argsort(tfidf.idf_)[::-1]
feat = tfidf.get_feature_names()
top_n = 10
top_features = [feat[i] for i in ind[:top_n]]
print(top_features)

'''''

data1 = pd.read_csv('spam.csv' )
data1.drop(data1.columns[[2, 3, 4]], axis = 1, inplace = True)
Ham = data1[data1.v1 == 'ham']
Spam = data1[data1.v1 == 'spam']

count = CountVectorizer()
tfidf = TfidfVectorizer()

# bag = count.fit_transform(Ham.v2)
# bag_tfidf = tfidf.fit_transform(Ham.v2)
#
# bag = count.fit_transform(Spam.v2)
# bag_tfidf = tfidf.fit_transform(Spam.v2)
# print("Word bag = ",bag.toarray())
# print('\n')
# np.set_printoptions(precision = 3)
# print('tf-idf',bag_tfidf.toarray())

df = tfidf.fit_transform(Ham['v2'])
ind = np.argsort(tfidf.idf_)[::-1]
feat = tfidf.get_feature_names()
top_n = 10
top_features = [feat[i] for i in ind[:top_n]]
print("Топ слов для спама, отмеченного как хамство\n")
print(top_features)

df_1 = tfidf.fit_transform(Spam['v2'])
ind = np.argsort(tfidf.idf_)[::-1]
feat = tfidf.get_feature_names()
top_n = 10
top_features = [feat[i] for i in ind[:top_n]]
print("Топ слов для спама\n")
print(top_features)