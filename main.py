from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from data_processing import iptdata
from knn import myknn

import methods


# Download datasets used by Kusner et al from https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
data_path = './data/bbcsport-emd_tr_te_split.mat'
#data_path = './data/amazon-emd_tr_te_split.mat'

# Download GloVe 6B tokens, 300d word embeddings from https://nlp.stanford.edu/projects/glove/
embeddings_path = './data/glove.6B.300d.txt'

data = iptdata(data_path, embeddings_path)

bow_data, class = data['BOW'], data['class']
topic_proportions = data['topic_proportions']
 ## specified in the paper using train_test_split to separate train and test sets.
bow_train, class_train, bow_test, class_test = train_test_split(bow_data, y)
topic_train, topic_test = train_test_split(topic_proportions)

cost_bow = data['cost_embeddings']
cost_topic = data['cost_topics']

method = HOTT
#method = HOFTT
#method = WMD
#method = WMDT20

# Compute test error
if method == HOTT or method == HOFTT:
    test_error = myknn(7,topic_train,class_train, topic_test, class_test, cost_topic, method)
else:
    test_error = myknn(7,normalize(bow_train, 'l1'),class_train,normalize(bow_test, 'l1'), class_test, cost_bow, method)
print("test error: ", test_error)
