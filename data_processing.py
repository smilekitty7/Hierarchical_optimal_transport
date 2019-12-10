import numpy as np
import lda
import ot

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import scipy.io as sio
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

embeddings_path = 'glove.6B.300d.txt'
# load data from the WMD paper
# each data file contains the words, the embedding vector for each word, the bow vector for each word,


# Reduce vocabulary size by stemming and removing stop words.
def stem_vocab(vocab_BOW, vocab, vocab_embed):
    # do stemming, stop word removal, short word removal, infrequent word removal.

    # 1. stemming
    vocab_new = []
    emb = {}
    stemmer = SnowballStemmer("english")
    idx = {}

    for i, word in enumerate(vocab):
        word1 = stemmer.stem(word) # stemmed word
        if word1 in vocab_new:
            #print('yes')
            emb[word1].append(vocab_embed[word])
            idx[word1].append(i)
        else:
            vocab_new.append(word1)
            emb[word1] = [vocab_embed[word]]
            idx[word1] = [i]


    vocab_BOW_new = np.zeros((vocab_BOW.shape[0], len(vocab_new)))
    for i in range(len(vocab_new)):
        voc = vocab_new[i]
        vocab_BOW_per_doc = vocab_BOW[:, idx[voc]]
        #print(vocab_BOW_per_doc.shape)
        #print(np.sum(vocab_BOW_per_doc, axis=1).flatten())
        #print('---')
        vocab_BOW_new[:, i] = np.sum(vocab_BOW_per_doc, axis=1).flatten()

    for i in emb:
        if np.shape(emb[i])[0] > 1:
            new_emb = np.mean(emb[i], axis=0)
            emb[i] = np.array(new_emb)
        else:
            emb[i] = np.array(emb[i])

    # 2. stop words
    stop_words = set(stopwords.words('english'))

    # 3. word count < 5
    word_counts = vocab_BOW_new.sum(axis=0)

    # 4. word length < 3
    not_short = np.array([len(w) > 2 for w in vocab_new])

    judg = np.zeros(np.size(vocab_new))
    for i,w in enumerate(vocab_new):
        judg[i] = (w not in stop_words)

    judg = judg * not_short * (word_counts > 5)

    vocab_new2 = list(np.array(vocab_new)[judg==1])
    vocab_BOW_new2 = vocab_BOW_new[:,judg==1].astype(np.int)
    emb2 = emb

    w2 = np.array(vocab_new)[judg==0]
    for w in w2:
        del emb2[w]

    return (vocab_new2,
            emb2,
            vocab_BOW_new2)


def embeddings_new(vocab, vocab_BOW, embed_path):

    all_embed_vocab = {}
    with open(embed_path, 'r') as file:
        for line in file.readlines():
            word = line.split(' ')[0]
            embedding = [float(x) for x in line.split(' ')[1:]]
            all_embed_vocab[word] = embedding

    emb3 = {}
    new_vocab_idx = []
    vocab3 = []
    for i, w in enumerate(vocab):
        if w in all_embed_vocab:
            emb3[w] = all_embed_vocab[w]
            new_vocab_idx.append(i)
            vocab3.append(w)
    BOW3 = vocab_BOW[:, new_vocab_idx]
    return vocab3, emb3, BOW3


def iptdata(data_path,
           embeddings_path,
           T=70,
           glove_embeddings=True,
           stemming=True):
    data_all = sio.loadmat(data_path, squeeze_me=True, chars_as_strings=True)  # dict

    if 'Y' in data_all:
        y_all = data_all['Y'].astype(np.int)
    else:
        y_all = np.concatenate((data_all['yte'].astype(np.int), data_all['ytr'].astype(np.int)), axis=1)

    if 'X' in data_all:
        embed_all = data_all['X']
    else:
        embed_all = np.concatenate((data_all['xte'], data_all['xtr']), axis=1)

    if 'BOW_X' in data_all:
        BOW_all = data_all['BOW_X']
    else:
        BOW_all = np.concatenate((data_all['BOW_xte'], data_all['BOW_xtr']), axis=1)

    if 'words' in data_all:
        words_all = data_all['words']
    else:
        words_all = np.concatenate((data_all['words_tr'], data_all['words_te']), axis=1)

    vocab = []
    vocab_embed = {}

    l = len(words_all)
    for i in range(l):
        word_i = words_all[i]
        embed_i = embed_all[i]
        bow_i = BOW_all[i]
        w = len(word_i)
        for j in range(w):
            if type(word_i[j]) == str:
                if word_i[j] not in vocab:
                    vocab.append(word_i[j])
                    vocab_embed[word_i[j]] = embed_i[:, j]
            else:
                break

    vocab_BOW = np.zeros((l, len(vocab)), dtype=np.int)

    l = len(words_all)
    for i in range(l):
        word_i = words_all[i]
        bow_i = BOW_all[i]

        w = len(word_i)
        words_idx = []
        for j in range(w):
            if type(word_i[j]) == str:
                words_idx.append(vocab.index(word_i[j]))
            else:
                break

        vocab_BOW[i, words_idx] = bow_i.astype(np.int)


    if glove_embeddings:
        vocab, vocab_embed, vocab_BOW = embeddings_new(
            vocab, vocab_BOW, embeddings_path)

    if stemming:
        vocab, vocab_embed, vocab_BOW = stem_vocab(
            vocab_BOW, vocab, vocab_embed)

    ####################################################

    l1_BOW, l2_BOW = vocab_BOW.shape
    embed_dat = [[] for _ in range(l1_BOW)]
    for i in range(l2_BOW):
        for d in range(l1_BOW):
            if vocab_BOW[d, i] > 0:
                for _ in range(vocab_BOW[d, i]):
                    embed_dat[d].append(vocab_embed[vocab[i]])

    vocab_embed = []
    for doc_i in embed_dat:
        vocab_embed.append(np.array(doc_i))

    # Matrix of word embeddings
    embeddings = np.array([vocab_embed[w] for w in vocab])

    model = lda.LDA(n_topics=T, n_iter=1500, random_state=1)
    model.fit(vocab_BOW)
    topics = model.topic_word_
    lda_centers = np.matmul(topics, embeddings)
    n_top_words = 20
    topic_dict = {}
    topic_proportions = model.doc_topic_


    #cost_embeddings_cos = cosine_similarity(embeddings, embeddings)
    cost_embeddings = euclidean_distances(embeddings, embeddings)**1
    cost_topics = np.zeros((topics.shape[0], topics.shape[0]))
    cost_m = np.zeros((topics.shape[0], topics.shape[0]))

    for i in range(cost_topics.shape[0]):
        for j in range(i + 1, cost_topics.shape[1]):
            #print(i,j)
            # i_list = topic_dict[i].astype(bool)
            # j_list = topic_dict[j].astype(bool)
            #
            # topic_i = topics[i][i_list]
            # topic_j = topics[j][j_list]
            #
            # cost_e = cost_embeddings[i_list][:,j_list]
            # # np.ascontiguousarray(topic_i)
            # print(topic_i.flags['C_CONIGUOUS'])
            # # np.ascontiguousarray(topic_j)
            # print(topic_j.flags['C_CONTIGUOUS'])
            # cost_e = np.ascontiguousarray(cost_e)
            # print(cost_e.flags['C_CONTIGUOUS'])
            # cost_m[i,j] = ot.emd2(topic_i, topic_j, cost_e, numItermax=10000)
            cost_topics[i, j] = ot.emd2(topics[i], topics[j], cost_embeddings,numItermax=10000)
    cost_topics = cost_topics + transpose(cost_topics)

    output = {'BOW': vocab_BOW,
              'class': y_all - 1,
              'embeddings': embeddings,
              'topics': topics,
              'topic_proportions': topic_proportions,
              'cost_embeddings': cost_embeddings,
             'cost_topics': cost_topics}

    return output
