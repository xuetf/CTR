from .util import *
from .simplex_projection import euclidean_proj_simplex
from .trainset import Trainset
from .constant import *
import time
import numpy as np
import scipy.optimize
from six.moves import xrange
import pickle
import collections
import logging

logging.basicConfig(filename='../logs/ctr_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())),
                                                      format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.random.seed(2018)


class CollaborativeTopicModel:
    """
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
    """

    def __init__(self, n_topic, n_voca, doc_ids, doc_word_ids, doc_word_cnts, ratings, theta_init=None, beta_init=None,
                 resume=None, save_lag=3, save_format="../output/model_iter_{}.pkl", verbose=True):
        self.lambda_u = 0.01
        self.lambda_v = 10
        self.alpha = 1
        self.eta = 0.01
        self.alpha_smooth = 0.00
        self.beta_smooth = 0.01
        self.a = 1
        self.b = 0.01
        self.a_minus_b = self.a - self.b
        self.pseudo_count = 1.0

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.verbose = verbose
        self.start_iter = 0
        self.old_likelihood = -np.exp(50)
        self.save_lag = save_lag
        self.save_format = save_format

        self.construct_trainset(ratings) # 先处理，得到用户数量，物品数量，id转换

        # U = user_topic matrix, U x K
        self.U = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
                                               size=self.n_user)
        # V = item(doc)_topic matrix, V x K
        self.V = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_v),
                                               size=self.n_item)

        if theta_init:
            self.theta = np.loadtxt(theta_init).astype(float)
        else:
            self.theta = np.random.dirichlet(alpha=[1]*n_topic, size=[self.n_item]) + self.alpha_smooth
            # self.theta = np.random.random([n_item, n_topic])
        self.theta = self.theta / self.theta.sum(1)[:, np.newaxis]  # normalize along col

        if beta_init:
            self.beta = np.loadtxt(beta_init).astype(float).T # 转置
            if np.any(self.beta < 0): # 说明是log_beta
                self.beta = np.exp(self.beta)
        else:
            self.beta = np.random.dirichlet(alpha=[1]*n_voca, size=[n_topic]).T + self.beta_smooth # W * K 转置了！！注意
            # self.beta = np.random.random([n_voca, n_topic])  # every column is topic distribution over word
        self.beta = self.beta / self.beta.sum(0)  # normalize along row

        self.phi_sum = np.zeros([n_voca, n_topic])  # [word,topic] relation matrix

        self.pre_process_corpus(doc_ids, doc_word_ids, doc_word_cnts, ratings, theta_init)  # 这里还会调整参数的位置.C,R,theta

        if resume: # resume的不需要调整
            self.load_model(resume)

    def load_model(self, name='../output/model.pkl'):
        self.U, self.V, self.theta, self.beta, self.start_iter = pickle.load(open(name, 'rb'))
        self.start_iter += 1
        if self.verbose:
            print("load model successfully...")
            logging.info("load model successfully...")

    def save_model(self, iter, name='../output/model.pkl'):
        pickle.dump((self.U, self.V, self.theta, self.beta, iter), open(name, 'wb'))

    def save_theta_beta(self, theta_path='../output/ctr.theta', beta_path='../output/ctr.beta', base_raw_doc_id=True):
        np.savetxt(beta_path, self.beta.T)
        if base_raw_doc_id: # 调整顺序, theta按照self.doc_ids顺序排列。否则按照inner_id顺序排列
            theta_tmp = np.zeros_like(self.theta)
            for index, raw_doc_id in enumerate(self.doc_ids):
               inner_doc_id = self.trainset.to_inner_iid(raw_doc_id)
               theta_tmp[index, :] =  self.theta[inner_doc_id, :]
            np.savetxt(theta_path, theta_tmp)
        else:
            np.savetxt(theta_path, self.theta)

    def pre_process_corpus(self, doc_ids, doc_word_ids, doc_word_cnts, ratings, theta_init):
        '''
        :param doc_ids: [M * ()], each row is the word ids within an article
        :param doc_cnt: [M * ()], doc-word frequency matrix, align with doc_ids
        :param ratings: [(user,item)] list. each element is a tuple:(user,item) means user has item in his item list
        '''

        self.C = np.zeros([self.n_user, self.n_item]) + self.b
        self.R = np.zeros([self.n_user, self.n_item])  # user_size x item_size

        for ruid, riid in ratings:
            self.C[self.trainset.to_inner_uid(ruid), self.trainset.to_inner_iid(riid)] += self.a - self.b
            self.R[self.trainset.to_inner_uid(ruid), self.trainset.to_inner_iid(riid)] = 1

        # 调整doc_word_ids, doc_word_cnts的顺序
        self.doc_ids = doc_ids
        self.doc_word_ids = [0] * self.n_item
        self.doc_word_cnts = [0] * self.n_item

        for raw_iid, doc_word_id in zip(doc_ids, doc_word_ids): # 注意doc_ids必须赋上id (rid，doc_id)
            inner_iid = self.trainset.to_inner_iid(raw_iid)
            self.doc_word_ids[inner_iid] = doc_word_id

        for raw_iid, doc_word_cnt in zip(doc_ids, doc_word_cnts):  # 注意doc_ids必须赋上id(rid，doc_cnt)
            inner_iid = self.trainset.to_inner_iid(raw_iid)
            self.doc_word_cnts[inner_iid] = doc_word_cnt

        # 调整theta, 意味着theta是gen_sim按照文档顺序训练的各文档的主题分布。
        # 不调整，意味着是CTR模型保存的结果。
        if theta_init:
            theta_tmp = np.zeros((self.n_item, self.n_topic))
            for raw_iid, theta_i in zip(doc_ids, self.theta):
                inner_iid = self.trainset.to_inner_iid(raw_iid)
                theta_tmp[inner_iid, :] = theta_i
            self.theta = theta_tmp

    def construct_trainset(self, ratings):
        '''该方法保证只要ratings数据集不修改，则方法是幂等的'''
        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = collections.defaultdict(list) # the items has been rated by the user
        ir = collections.defaultdict(list) # the users has rated the item

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid in ratings:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append(iid)
            ir[iid].append(uid)

        self.n_user = len(ur)  # number of users
        self.n_item = len(ir)  # number of items
        self.n_ratings = len(ratings)

        trainset = Trainset(ur,
                            ir,
                            self.n_user,
                            self.n_item,
                            self.n_ratings,
                            raw2inner_id_users,
                            raw2inner_id_items)
        self.trainset = trainset

    def fit(self, max_iter=100):
        if self.verbose:
            print('Init Reconstruction error:{:.4f}, \tLikelihood:{:.4f}'.format(self.sqr_error(), self.likelihood()[0]))
            logging.info('Init Reconstruction error:{:.4f}, \tLikelihood:{:.4f}'.format(self.sqr_error(), self.likelihood()[0]))

        for iteration in xrange(self.start_iter, max_iter):
            if self.verbose:
                print ("iter:{} start...".format(iteration))
                logging.info("iter:{} start...".format(iteration))
            tic = time.clock()
            self.do_e_step()
            self.do_m_step()

            err, weight_err = self.sqr_error(is_weight=False), self.sqr_error(is_weight=True)
            like, converge = self.likelihood()

            if iteration % self.save_lag == 0:
                self.save_model(iteration, name=self.save_format.format(iteration))

            if self.verbose:
                logging.info('[ITER] {:3d},\tElapsed time:{:.2f},\tR error:{:.3f},\tWeighted R error:{:.3f},\tLikelihood:{:.4f},\tConverge:{:.10f}'
                      .format(iteration,time.clock() - tic, err,weight_err, like, converge))

                print('[ITER] {:3d},\tElapsed time:{:.2f},\tR error:{:.3f},\tWeighted R error:{:.3f},\tLikelihood:{:.4f},\tConverge:{:.10f}'
                      .format(iteration,time.clock() - tic, err,weight_err, like, converge))
            if iteration > min_iter and converge < converge_tol:
                if self.verbose:
                    print('has converged, end up...')
                    logging.info('has converged, end up...')
                break

    # reconstructing matrix for prediction
    def predict_item(self):
        return np.dot(self.U, self.V.T)

    # reconstruction error
    def sqr_error(self, is_weight=False):
        err = (self.R - self.predict_item()) ** 2
        if is_weight: err = self.C * err
        err = err.sum()
        return err

    def likelihood(self):
        # ui
        likelihood = 0.00
        for ui in self.U:
            likelihood += -0.5 * self.lambda_u * np.dot(ui.T, ui)

        # epsilon: v-\theta
        for vi, theta_i in zip(self.V, self.theta):
            likelihood += -0.5 * self.lambda_v * np.dot((vi-theta_i).T, vi-theta_i)

        # r
        err = (self.R - self.predict_item()) ** 2
        err = -0.5 * self.C * err
        likelihood += err.sum()

        # theta beta
        for vi in xrange(self.n_item):
            W = np.array(self.doc_word_ids[vi])
            word_beta = self.beta[W, :]
            phi = self.theta[vi, :] * word_beta + e
            phi = phi / phi.sum(1)[:, np.newaxis]
            likelihood += np.sum(np.sum(phi * (safe_log(self.theta[vi, :]) + safe_log(word_beta) - safe_log(phi))))

        converge = np.abs((likelihood - self.old_likelihood) / self.old_likelihood)

        if likelihood < self.old_likelihood:
            if self.verbose:
                logging.debug('likelihood is decreasing....')
                print ('likelihood is decreasing....')
        self.old_likelihood = likelihood

        return likelihood, converge

    def do_e_step(self):
        self.update_u_2()
        self.update_v_2()
        self.update_theta()

    def update_theta_j(self, gamma, v, theta_j):
        def f_simplex(x, v, gamma, lambda_v):
            return 0.5 * lambda_v * np.dot((v - x).T, v - x) - np.dot(gamma, safe_log(x))

        def df_simplex(x, v, gamma, lambda_v):
            g = -lambda_v * (x - v)
            g += gamma / x
            return -g

        opt_x_old = np.array(theta_j)  # 不能直接赋值，那样是引用
        f_old = f_simplex(theta_j, v, gamma, self.lambda_v)
        g = df_simplex(theta_j, v, gamma, self.lambda_v)
        ab_sum = np.sum(np.abs(g))
        if ab_sum > 1.0: g /= ab_sum
        theta_j = theta_j - 1.0 * g
        x_bar = euclidean_proj_simplex(theta_j)
        x_bar = x_bar / x_bar.sum()  # normalized
        x_bar = x_bar - opt_x_old
        r = np.dot(g, x_bar)
        r *= 0.5
        beta_lr = 0.5
        t = beta_lr
        iter = 0
        while iter < 100:
            theta_j = opt_x_old
            theta_j = t * x_bar + theta_j
            f_new = f_simplex(theta_j, v, gamma, self.lambda_v)
            if f_new > f_old + r * t:
                t = t * beta_lr
            else:
                break
            iter += 1

        # theta_j /= theta_j.sum()  # 数值问题，导致下面 np.sum(self.theta[vi, :])=1.000000000000000000002
        if np.any(theta_j < 0) or np.any(theta_j > 1):
            if self.verbose:
                print('something wrong happened....')
                logging.debug('something wrong happened....')
        return theta_j

    def update_phi(self, doc_word_id, theta, doc_cnt=None, is_new_doc=False):
        W = np.array(doc_word_id)
        word_beta = self.beta[W, :]
        phi = theta * word_beta  # W x K
        phi = phi / phi.sum(1)[:, np.newaxis]  # the optimal of phi is prop to this, so normalized

        gamma = np.sum(phi, axis=0)  # gamma相当于是这篇文档的beta分布值，phi_sum是在所有文档上求beta分布值

        if not is_new_doc: # upt phi if necessary
            self.phi_sum[W, :] += np.array(doc_cnt)[:, np.newaxis] * phi  # 列broadcast。phi_sum是为了在M-step更新beta用的
        return gamma

    def update_theta(self, iter=3):
        for vi in xrange(self.n_item):
            for _ in xrange(iter):
                gamma = self.update_phi(self.doc_word_ids[vi], self.theta[vi, :], self.doc_word_cnts[vi])
                self.theta[vi, :] = self.update_theta_j(gamma, self.V[vi, :], self.theta[vi, :])

    def new_doc_inference(self, doc_word_id, theta_init=None, iter=100):
        if theta_init is not None:
            theta = theta_init
        else:
            theta = np.random.dirichlet(alpha=[1] * self.n_topic, size=1) + self.alpha_smooth # 初始化
        theta /= theta.sum()

        for i in xrange(iter):
            gamma = self.update_phi(doc_word_id, theta, is_new_doc=True)
            theta = self.update_theta_j(gamma, theta, theta) # v=theta
        return theta

    def predict(self, ruid, rvid):
        inner_uid = self.trainset.to_inner_uid(ruid)
        inner_iid = self.trainset.to_inner_iid(rvid)
        if self.trainset.knows_user(inner_uid) and self.trainset.knows_item(inner_iid):
            return np.dot(self.U[inner_uid], self.V[inner_iid])
        else:
            raise PredictionImpossible('User and item are unknown!')

    def predict_many(self, ruid, rvids):
        """without check"""
        inner_uid = self.trainset.to_inner_uid(ruid)
        inner_iids = [self.trainset.to_inner_iid(rvid) for rvid in rvids]
        u = self.U[inner_uid]
        V = self.V[inner_iids, :]
        return np.dot(u, V.T)

    def out_of_predict(self, ruid, new_doc_word_id, theta_init=None, return_theta=False):
        inner_uid = self.trainset.to_inner_uid(ruid)
        if self.trainset.knows_user(inner_uid):
            theta = self.new_doc_inference(new_doc_word_id, theta_init=theta_init)
            if return_theta:
                return np.dot(self.U[inner_uid], theta.T), theta
            return np.dot(self.U[inner_uid], theta.T)
        else:
            raise PredictionImpossible('User:{} is unknown!'.format(ruid))

    def update_u(self):
        for ui in xrange(self.n_user):
            left = np.dot(self.V.T * self.C[ui, :], self.V) + self.lambda_u * np.identity(self.n_topic)

            self.U[ui, :] = np.linalg.solve(left, np.dot(self.V.T * self.C[ui, :], self.R[ui, :]))

    def update_u_2(self):
        ''' another implementation '''
        XX = np.zeros((self.n_topic, self.n_topic))
        for vi in xrange(self.n_item):
            v = self.V[vi, :]
            XX += np.matmul(v.reshape(-1, 1), v.reshape(1, -1))  # rank 1 multi

        XX *= self.b
        XX += self.lambda_u * np.identity(self.n_topic)

        for ui in xrange(self.n_user):
            A = np.array(XX)
            x = np.zeros(self.n_topic)
            for vi in self.trainset.ur[ui]:
                v = self.V[vi, :]
                A += self.a_minus_b * np.matmul(v.reshape(-1, 1), v.reshape(1, -1))  # rank 1 multi
                x += self.a * v
            self.U[ui, :] = np.linalg.solve(A, x)

    def update_v(self):
        for vi in xrange(self.n_item):
            left = np.dot(self.U.T * self.C[:, vi], self.U) + self.lambda_v * np.identity(self.n_topic)

            self.V[vi, :] = np.linalg.solve(left, np.dot(self.U.T * self.C[:, vi],
                                                            self.R[:, vi]) + self.lambda_v * self.theta[vi, :])

    def update_v_2(self):
        ''' another implementation '''
        XX = np.zeros((self.n_topic, self.n_topic))
        for ui in xrange(self.n_user):
            u = self.U[ui, :]
            XX += np.matmul(u.reshape(-1, 1), u.reshape(1, -1))  # rank 1 multi
        XX *= self.b

        for vi in xrange(self.n_item):
            theta_v = self.theta[vi, :]
            A = np.array(XX)
            x = np.zeros(self.n_topic)
            for ui in self.trainset.ir[vi]:
                u = self.U[ui, :]
                A += self.a_minus_b * np.matmul(u.reshape(-1, 1), u.reshape(1, -1))  # rank 1 multi
                x += self.a * u
            x += self.lambda_v * theta_v
            A += self.lambda_v * np.identity(self.n_topic)
            self.V[vi, :] = np.linalg.solve(A, x)

    def do_m_step(self):
        self.beta = self.phi_sum / self.phi_sum.sum(0)  # normalized each topic
        self.phi_sum = np.zeros([self.n_voca, self.n_topic]) + self.eta



