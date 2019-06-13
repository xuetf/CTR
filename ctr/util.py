import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.test.utils import datapath
import os

class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """
    pass


def safe_log(a):
    r = np.ma.log(a)
    return r.filled(-10000)


def raw_pd_to_cropus(vocab_data, raw_doc_data, rating_data):
    ratings = [(u, d) for u, d, _ in rating_data.itertuples(index=False)]
    raw_doc = [title + " " + abstract for _, title, _, _, abstract in raw_doc_data.itertuples(index=False)]
    tf_vec = CountVectorizer(vocabulary=vocab_data)
    tf_vec.fit(raw_doc)
    doc_ids = raw_doc_data['doc.id'].values
    doc_word_ids, doc_word_cnts = raw_doc_to_cropus(tf_vec, raw_doc)
    return doc_ids,doc_word_ids,doc_word_cnts, ratings, raw_doc, tf_vec




def raw_doc_to_cropus(tf, raw_doc):
    if isinstance(raw_doc, str): raw_doc = [raw_doc]
    tf_result = tf.transform(raw_doc)
    doc_word_ids = []
    doc_word_cnts = []
    for i in range(tf_result.shape[0]):
        ind_from = tf_result.indptr[i]
        ind_to = tf_result.indptr[i + 1]
        doc_word_ids.append(tf_result.indices[ind_from: ind_to])
        doc_word_cnts.append(tf_result.data[ind_from: ind_to])
    return doc_word_ids, doc_word_cnts


# for new document, use the previous model to find the probability which topic it belongs to
def doc_topic_distribution(new_doc, tf, ldamodel=None, lda_file="output/gensim/lda.model", inference=True, save_theta_path=None, save_beta_path=None):
    if isinstance(new_doc, str): new_doc = [new_doc]
    if lda_file is not None:
        temp_file = datapath(os.path.join(os.path.dirname(__file__), lda_file))
        ldamodel = gensim.models.ldamodel.LdaModel.load(temp_file)

    # Fit and transform
    X = tf.transform(new_doc)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    if inference:
        theta = ldamodel.inference(corpus)[0] # gamma distribution
    else:
        theta = list(ldamodel[corpus])  # only show some topics

    beta = ldamodel.get_topics()

    if save_theta_path:
        np.savetxt(save_theta_path, theta)
    if save_beta_path:
        np.savetxt(save_beta_path, beta)

    return theta, beta