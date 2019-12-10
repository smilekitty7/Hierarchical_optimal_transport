# load data adnd train the model using knn
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from dataprocessing import loader
from knn import myknn

import distances
import hott

# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'

# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
data_name = 'bbcsport-emd_tr_te_split.mat'
# data_name = 'amazon-emd_tr_te_split.mat'


data = loader(data_path + data_name, embeddings_path, p=1)

bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']
 ## specified in the paper using train_test_split to separate train and test sets.
bow_train, class_train, bow_test, class_test = train_test_split(bow_data, y)
topic_train, topic_test = train_test_split(topic_proportions)

cost_bow = data['cost_E']
cost_topic = data['cost_T']

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
