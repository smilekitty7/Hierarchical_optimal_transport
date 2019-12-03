import numpy as np
import lda
import ot

from sklearn.metrics.pairwise import euclidean_distances
import scipy.io as sio
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# load data from the WMD paper
# each data file contains the words, the embedding vector for each word, the bow vector for each word,


# Reduce vocabulary size by stemming and removing stop words.
def reduce_vocab(bow_data, vocab, embed_vocab, embed_aggregate='mean'):
    """Reduce vocabulary size by stemming and removing stop words.
    """
    vocab = np.array(vocab)
    short = np.array([len(w) > 2 for w in vocab])
    stop_words = set(stopwords.words('english'))
    stop = np.array([w not in stop_words for w in vocab])
    reduced_vocab = vocab[np.logical_and(short, stop)]
    reduced_bow_data = bow_data[:, np.logical_and(short, stop)]
    stemmer = SnowballStemmer("english")
    stemmed_dict = {}
    stemmed_idx_mapping = {}
    stemmed_vocab = []
    for i, w in enumerate(reduced_vocab):
        stem_w = stemmer.stem(w)
        if stem_w in stemmed_vocab:
            stemmed_dict[stem_w].append(w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)].append(i)
        else:
            stemmed_dict[stem_w] = [w]
            stemmed_vocab.append(stem_w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)] = [i]

    stemmed_bow_data = np.zeros((bow_data.shape[0], len(stemmed_vocab)),
                                dtype=np.int)
    for i in range(len(stemmed_vocab)):
        stemmed_bow_data[:, i] = reduced_bow_data[:, stemmed_idx_mapping[i]].sum(axis=1).flatten()

    word_counts = stemmed_bow_data.sum(axis=0)
    stemmed_reduced_vocab = np.array(stemmed_vocab)[word_counts > 5].tolist()
    stemmed_reduced_bow_data = stemmed_bow_data[:, word_counts > 5]

    stemmed_reduced_embed_vocab = {}
    for w in stemmed_reduced_vocab:
        old_w_embed = [embed_vocab[w_old] for w_old in stemmed_dict[w]]
        if embed_aggregate == 'mean':
            new_w_embed = np.mean(old_w_embed, axis=0)
        elif embed_aggregate == 'first':
            new_w_embed = old_w_embed[0]
        else:
            print('Unknown embedding aggregation')
            break
        stemmed_reduced_embed_vocab[w] = new_w_embed

    return (stemmed_reduced_vocab,
            stemmed_reduced_embed_vocab,
            stemmed_reduced_bow_data)


def change_embeddings(vocab, bow_data, embed_path):
    """Change embedding data if vocabulary has been reduced."""
    all_embed_vocab = {}
    with open(embed_path, 'r') as file:
        for line in file.readlines():
            word = line.split(' ')[0]
            embedding = [float(x) for x in line.split(' ')[1:]]
            all_embed_vocab[word] = embedding

    data_embed_vocab = {}
    new_vocab_idx = []
    new_vocab = []
    for i, w in enumerate(vocab):
        if w in all_embed_vocab:
            data_embed_vocab[w] = all_embed_vocab[w]
            new_vocab_idx.append(i)
            new_vocab.append(w)
    bow_data = bow_data[:, new_vocab_idx]
    return new_vocab, data_embed_vocab, bow_data
  
# loader
def loader(data_path,
           embeddings_path,
           p=1,
           K_lda=70,
           glove_embeddings=True,
           stemming=True):


    data_all = sio.loadmat('amazon-emd_tr_te_split.mat', squeeze_me=True, chars_as_strings=True)  # dict

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

    ####################################################
    # Use GLOVE word embeddings
    if glove_embeddings:
        vocab, vocab_embed, vocab_BOW = change_embeddings(
            vocab, vocab_BOW, embeddings_path)
    # Reduce vocabulary by removing short words, stop words, and stemming
    if stemming:
        vocab, vocab_embed, vocab_BOW = reduce_vocab(
            vocab_BOW, vocab, vocab_embed, embed_aggregate='mean')


    ####################################################



    l1_BOW, l2_BOW = vocab_BOW.shape
    embed_dat = [[] for _ in range(l1_BOW)]
    for i in range(l2_BOW):
        for d in range(l1_BOW):
            if vocab_BOW[d, i] > 0:
                for _ in range(vocab_BOW[d, i]):
                    embed_dat[d].append(vocab_embed[vocab[i]])

    embed_data = []
    for doc_i in embed_dat:
        embed_data.append(np.array(doc_i))


    # Matrix of word embeddings
    embeddings = np.array([vocab_embed[w] for w in vocab])


    model = lda.LDA(n_topics=K_lda, n_iter=1500, random_state=1)
    model.fit(vocab_BOW)
    topics = model.topic_word_
    lda_centers = np.matmul(topics, embeddings)
    n_top_words = 20
    for i, topic_dist in enumerate(topics):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]

    topic_proportions = model.doc_topic_

    cost_embeddings = euclidean_distances(embeddings, embeddings) ** p
    cost_topics = np.zeros((topics.shape[0], topics.shape[0]))

    for i in range(cost_topics.shape[0]):
        for j in range(i + 1, cost_topics.shape[1]):
            cost_topics[i, j] = ot.emd2(topics[i], topics[j], cost_embeddings, numItermax =10000)
    cost_topics = cost_topics + cost_topics.T

    output = {'X': vocab_BOW,
           'y': y_all-1,
           'embeddings': embeddings,
           'topics': topics,
           'proportions': topic_proportions,
           'cost_E': cost_embeddings,
           'cost_T': cost_topics}

    return output
