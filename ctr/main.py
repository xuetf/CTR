import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import io
import pandas as pd
from ctr.ctm import CollaborativeTopicModel
from sklearn.decomposition import LatentDirichletAllocation
from time import time
import gensim
import logging
from ctr.util import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def sklearn_lda(tf, raw_doc):
    tf_result = tf.transform(raw_doc)
    lda = LatentDirichletAllocation(n_components=200, max_iter=200,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0, batch_size=1000,
                                    verbose=True)
    t0 = time()
    W = lda.fit_transform(tf_result)  # theta doc-topic
    H = lda.components_  # beta topic-word
    print("done in %0.3fs." % (time() - t0))

    np.savetxt("../output/lda.theta", W)
    np.savetxt("../output/lda.beta", H)

    print_topics("../output/lda.beta", '../data/citeulike/vocab.dat', exp=False)


def gensim_lda(tf, raw_doc):
    tf_result = tf.transform(raw_doc)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(tf_result, documents_columns=False)
    id_map = dict((v, k) for k, v in tf.vocabulary_.items())

    # Use the gensim.models.ldamodel.LdaModel constructor to estimate
    # LDA model parameters on the corpus, and save to the variable `ldamodel`
    # Your code here:
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=200, id2word=id_map, chunksize=2000,
                                               passes=100, random_state=0)

    # Print Top 10 Topics / Word Distribution
    output = ldamodel.print_topics(20)
    print(output)

    np.savetxt("../output/gensim.beta", ldamodel.get_topics())
    np.savetxt("../output/gensim.theta", ldamodel.inference(corpus)[0])

    # lda = gensim.models.ldamodel.LdaModel.load(temp_file)
    ldamodel.save("../output/lda.model")


def print_topic(topic, vocabulary, exp=True, nwords=25):
    indices = list(range(len(vocabulary)))
    if exp:
        topic = np.exp(topic)
    topic = topic / topic.sum()
    indices.sort(key=lambda x: -topic[x])
    print(["{:}*{:.4f}".format(x, y) for (x, y) in zip(vocabulary[indices[0:nwords]], topic[indices[0:nwords]])])


def print_topics(beta_file, vocab_file, nwords=25, exp=True):
    # get the vocabulary
    vocabulary = np.array([line.rstrip('\n') for line in io.open(vocab_file, encoding='utf8')])
    topics = io.open(beta_file, 'r').readlines()
    # for each line in the beta file
    for topic_no,topic in enumerate(topics):
        print('topic %03d' % topic_no)
        topic = np.array(list(map(float, topic.split())))
        print_topic(topic, vocabulary, exp, nwords)
        print()


def print_doc_topic(doc_topic, exp=False, ntopic=20):
    indices = list(range(200))
    if exp: doc_topic = np.exp(doc_topic)
    doc_topic = doc_topic / doc_topic.sum()
    indices.sort(key=lambda x: -doc_topic[x])
    print(["Topic{:03d}*{:.4f}".format(x, y) for (x, y) in zip(indices[0:ntopic], doc_topic[indices[0:ntopic]])])


def print_doc_topics(theta_file, exp=False, ntopic=20):
    doc_topics = io.open(theta_file, 'r').readlines()
    # for each line in the beta file
    for doc_no, doc_topic in enumerate(doc_topics):
        print('doc %03d' % (doc_no+1))
        doc_topic = np.array(list(map(float, doc_topic.split())))
        print_doc_topic(doc_topic, exp=exp, ntopic=ntopic)
        print()
        if doc_no > 100: break


if __name__ == '__main__':
    vocab_data = [line.rstrip('\n') for line in io.open('../data/citeulike/vocab.dat', encoding='utf8')]
    raw_data = pd.read_csv('../data/citeulike/raw-data.csv', sep=',', encoding="ISO-8859-1")
    rating_data = pd.read_csv("../data/citeulike/user-info.csv")

    doc_ids, doc_word_ids, doc_word_cnts, ratings, raw_doc, tf_vec = raw_pd_to_cropus(vocab_data, raw_data, rating_data)

    # obtain beta,theta
    gensim_lda(tf_vec, raw_doc)
    print_doc_topics("../output/gensim.theta", exp=False)
    print('data prepared...')

    # train
    ctr = CollaborativeTopicModel(n_topic=200, n_voca=8000, doc_ids=doc_ids,
                                  doc_word_ids=doc_word_ids, doc_word_cnts=doc_word_cnts, ratings=ratings,
                                  beta_init='../output/gensim.beta', theta_init='../output/gensim.theta')
    ctr.fit(max_iter=1)

    # save ctr beta and theta
    ctr.save_theta_beta(beta_path='../output/ctr.beta',
                        theta_path='../output/ctr.theta', base_raw_doc_id=True)

    print_topics("../output/ctr.beta", '../data/citeulike/vocab.dat', exp=False)

    # new doc predict
    new_doc = "Researchers have access to large online archives of scientiﬁc articles. " \
              "As a consequence, ﬁnding relevant papers has become more difﬁcult. Newly formed online communities of researchers sharing citations provides a new way to solve this problem. In this paper, we develop an algorithm to recommend scientiﬁc articles to users of an online community. Our approach combines the merits of traditional collaborative ﬁltering and probabilistic topic modeling. It provides an interpretable latent structure for users and items, and can form recommendations about both existing and newly published articles. We study a large subset of data from CiteULike, a bibliography sharing service, and show that our algorithm provides a more effective recommender system than traditional collaborative ﬁltering."

    new_doc_ids, _ = raw_doc_to_cropus(tf_vec, new_doc)
    theta_init = doc_topic_distribution(new_doc, tf_vec)[0]
    r, theta = ctr.out_of_predict(0, new_doc_ids[0], theta_init=theta_init, return_theta=True)
    print_doc_topic(theta)
