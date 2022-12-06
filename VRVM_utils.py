##  Sevin Mohammadi- sm4894
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import loggamma
from scipy.special import digamma
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
import pickle
import time
import random
from scipy.linalg import block_diag

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel,
    manhattan_distances,
    euclidean_distances,
    cosine_distances
)
import pickle
import traceback

class VI_BLR(object):
    """
    Variational Inference algorithm for Bayesian linear regression with unknown noise parameters
    y:= Normal(xTw,lambda_1)
    w := Normal(0,diag(alpha_1,...alpha_d)^-1)
    alpha_k := Gamma(a0,b0)
    lambda =: Gamma(e0,f0)

    Input:  q(alpha_k) := Gamma(a'_k,b'_k)  k={1,...,d}
            q(w) := Normal(mu',sigma')
            q(lambda) "= Gamma(e',f')
    """
    def __init__(self, x, x_new, y, kernel, multi_kernel=False,g = None, a0=1e-16, b0=1e-16, e0=1, f0=1): #a0=1e-6, b0=1e-6, e0=1e-16, f0=1e-16 #a0=1e-16, b0=1e-16, e0=1, f0=1.
        # (filename,line_number,function_name,text)=traceback.extract_stack()[-2] ### a0=1e-6, b0=1e-6, e0=1e-16, f0=1e-16 #a0=1e-6, b0=1e-6, e0=1e-16, f0=1e-16

        self.kernel = kernel       # 'rbf' or 'poly', or 'rbf_poly'
        self.multi_kernel = multi_kernel
        self.g = g 

        self.y = y
        self.xold = x 
        self.xMain = x
        self.mainDim = np.shape(self.xMain)[1]
        self.xNew = x_new
        self.rvs = x_new

        self.Xpruned = np.array([]).reshape(-1,self.mainDim )
        
        self.a0 = a0 
        self.b0 = b0 
        self.e0 = e0
        self.f0 = f0

        self.bias_used = True
        self.x = self._apply_kernel( self.xMain, self.xNew, kernel = self.kernel)
        # print(self.x)
        # exit()
        self.dim = np.shape(self.x)[1]
        self.n = np.shape(self.x)[0]

        
    def initialization(self, par_list=None):
        self.Lv =[]
        self.xxt = np.zeros([self.dim,self.dim])
        for i in range(self.n):
            self.xxt += np.outer(self.x[i],self.x[i])
        self.yx = np.zeros([1,self.dim])
        for i in range(self.n):
            self.yx += self.y[i]*self.x[i]
        
        # np.random.seed(50)
        self.a_p = np.ones((self.dim,1)) * (self.a0 + 0.5)
        self.b_p = np.ones((self.dim,1)) * self.b0 
        self.e_p = self.n/2 #+ self.dim/2
        self.f_p = 0.
        self.mu_p = np.zeros((self.dim,1)) 
        # self.mu_p = np.ones((self.dim,1))
        self.sigma_p = np.eye(self.dim)
    
    def train_model(self, prune_threshold ,prune=False):
        delta_L = np.NaN
        t=0
        while (delta_L > 10e-2 or math.isnan(delta_L)): #previously 10e-3 #10e-3
            # print("iteration ",t)
            if t % 10 == 0:
                print("iteration ",t)
                # alpha = pd.DataFrame(list(vi.a_p/vi.b_p),columns=['alpha'])
                # mu = pd.DataFrame(list(vi.mu_p),columns=['mu'])
                # a_p = pd.DataFrame(list(vi.a_p),columns=['a_p'])
                # b_p = pd.DataFrame(list(vi.b_p),columns=['b_p'])
                # df = pd.concat([a_p, b_p, alpha, mu],axis=1)
                # print(df)
                # print("*** e_p **** ", vi.e_p)
                # print("*** f_p **** " , vi.f_p)
                # # df['invlambda'] = vi.e_p/vi.f_p
                        
            # t1 = time.time()
            self.update_q_alpha()
            self.update_q_lambda()
            self.update_q_w()
            # print("one iteration takes {} sec".format(time.time() - t1))
            self.eval_l()
            if t>1:
                delta_L = np.abs(self.Lv[t]-self.Lv[t-1])
                # print(delta_L)
            t += 1
            if t > 50 and math.isnan(delta_L):
                break
            if t > 30:
                break
        # self.plot_ELBO(t)
        if prune:
            self._prune_while_training(prune_w_weight = True, weight_treshold = prune_threshold)#0.005 #last 0.05
        
    def _apply_kernel(self, x, y, kernel, train = False, bias = False, g = None, gama = None): #0.7
        # self.bias_used = bias_used
        """Apply the selected kernel function to the data."""
        if kernel == 'cosine_distances':
            phi = cosine_distances(x, y)
        elif kernel == 'euclidean_distances':
            phi = euclidean_distances(x, y)
        elif kernel == 'manhattan_distances':
            phi = manhattan_distances(x, y)
        elif kernel == 'univariate_linear_spline_kernel':
            phi = linear_kernel(x, y)
        elif kernel == 'linear':
            phi = linear_kernel(x, y)
        elif kernel == 'rbf':    
            if self.multi_kernel:
                pass
                # phi = []
                # final_shape = 0
                # if gama is None:
                #     gama = self.gamma_kernel 
                # for g in gama:
                #     phi_tmp = rbf_kernel(x, y, g) # y should be dic {g1:rv1, g2:rv2}
                #     final_shape += len(y)
                #     phi.append(phi_tmp)
                # phi = np.concatenate(phi, axis=1)
                # # if d is not None:
                # #     phi = np.concatenate([phi,d], axis=1) #.reshape(-1,2)
                # # print('***kernel size is*** :', phi.shape)
                # assert phi.shape[1] == len(self.gamma_kernel) * len(y) #+ 1
            else:
                print(self.g)
                phi = rbf_kernel(x, y, self.g)
        elif kernel == 'poly':
            print('kernel degree is {} and kernel gamma is {} '.format(self.degree_kernel,self.gamma_kernel))
            if train == True:
                phi = self.PHI[self.gamma_kernel]
            else:
                phi = polynomial_kernel(x, y, degree=self.degree_kernel , gamma=self.gamma_kernel, coef0=0) #gamma=10, coef0=0.5

        elif callable(kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
        else:
            raise ValueError("Kernel selection is invalid.")
        if self.bias_used or bias:
            # print(self.bias_used)
            phi = np.append( np.ones((phi.shape[0], 1)),phi, axis=1)
            self.bias_used = False
        # print("***** kernel is created  and kernel size is *** :", phi.shape)
        return phi
      
    def update_q_alpha(self):
        self.a_p = np.ones((self.dim,1)) * (self.a0 + 0.5)      
        self.b_p = 0.5*(np.square(self.mu_p) + np.diag(self.sigma_p).reshape(-1,1)) + self.b0

    def update_q_lambda(self):
        x = self.x
        y = self.y
        self.e_p = self.e0 + self.n/2
        tmp0 = np.sum(np.square(y))
        tmp1 = np.sum(y * x @ self.mu_p)
        tmp2 = np.sum(np.dot(x,self.sigma_p).T*x.T)
        tmp3 = np.sum(np.dot(x,self.mu_p @ self.mu_p.T)*x)
        tmp = tmp0 - 2*tmp1 + tmp2 +tmp3
        self.f_p = self.f0 + 0.5*tmp

    def update_q_w(self):
        diag = self.a_p/self.b_p
        tmp = np.diag(diag[:,0]) + self.e_p/self.f_p * self.xxt
        self.sigma_p = np.linalg.inv(tmp)
        self.mu_p = self.e_p/self.f_p * self.sigma_p @ self.yx.T
        
    
    # def update_q_alpha(self):
    #     self.a_p = np.ones((self.dim,1)) * (self.a0 + 0.5)      
    #     self.b_p = 0.5*(np.square(self.mu_p) + np.diag(self.sigma_p).reshape(-1,1)) + self.b0

    # def update_q_lambda(self):
    #     x = self.x
    #     y = self.y
    #     self.e_p = self.e0 + self.n/2
    #     tmp0 = np.sum(np.square(y))
    #     tmp1 = np.sum(y * x @ self.mu_p)
    #     tmp2 = np.sum(np.dot(x, self.sigma_p).T * x.T)
    #     tmp3 = np.sum(np.dot(x, self.mu_p @ self.mu_p.T) * x)
    #     tmp = tmp0 - 2 * tmp1 + tmp2 + tmp3
    #     self.f_p = self.f0 + 0.5 * tmp

    # def update_q_w(self):
        
    #     diag = self.a_p/self.b_p
    #     tmp = np.diag(diag[:,0]) + self.e_p/self.f_p * self.xxt
    #     self.sigma_p = np.linalg.inv(tmp)
    #     self.mu_p = self.e_p/self.f_p * self.sigma_p @ self.yx.T
    
    def _prune_while_training(self, prune_w_weight = False, weight_treshold = None, prune_w_alpha = False, alpha_treshold = None, args=None):
        """Remove basis functions based on weight values."""
        # keep_alpha = self.alpha_ < self.threshold_alpha
        if prune_w_alpha:
            keep_relevance = np.abs(self.a_p/self.b_p) < alpha_treshold
        elif prune_w_weight:
            keep_relevance = np.abs(self.mu_p) > weight_treshold
        else:
            keep_relevance = args
        # # keep_relevance[0] = True
        # # keep_relevance = keep_relevance.squeeze()
        # print(keep_mu_p)
        # print(keep_mu_p.shape)
        # if not np.any(keep_relevance):
        #     keep_relevance[0] = True
        #     if self.bias_used:
        #         keep_relevance[-1] = True

        # if self.bias_used:
        #     if not keep_relevance[-1]:
        #         self.bias_used = False
        #     keep_relevance = keep_relevance[keep_relevance[:-1]]
        # else:
        #     keep_relevance = keep_relevance[keep_relevance]
        
        self.a_p = self.a_p[keep_relevance]
        self.b_p = self.b_p[keep_relevance]
        # self.alpha_old = self.alpha_old[keep_mu_p]
        # self.e_p = self.e_p[keep_mu_p]
        # self.f_p = self.f_p[keep_mu_p]
        # self.x = self.x[keep_mu_p]
        # self.x = self.xold[keep_mu_p]
        print("kernel size before prunning ", self.x.shape)
        self.x = self.x[:,keep_relevance]
        print("kernel size after prunning ", self.x.shape)
        # self.yrelevance_ = self.y[keep_mu_p]
        self.sigma_p = self.sigma_p[np.ix_(keep_relevance, keep_relevance)]
        self.mu_p = self.mu_p[keep_relevance]
        self.Xpruned = self.xNew[keep_relevance[1:]]#np.append(self.Xpruned, self.xNew[keep_mu_p], axis=0).reshape(-1, self.mainDim)
        # self.ypruned = self.y[keep_relevance[1:]]
        # self.dim = np.shape(self.x)[1]
        # self.xxt = np.zeros([self.dim,self.dim])
        # for i in range(self.n):
        #     self.xxt += np.outer(self.x[i],self.x[i])
        # self.yx = np.zeros([1,self.dim])
        # for i in range(self.n):
        #     self.yx += self.y[i]*self.x[i]
        print("{} relevence vectors".format(len(self.mu_p)))  

    def _add_newData(self, x):
        # self.xold = np.append(self.xold, x, axis=0)
        self.xNew = np.append(self.Xpruned, x, axis = 0).reshape(-1,self.mainDim)
        x = self._apply_kernel(self.xMain, x, kernel=self.kernelType)
        # add_dim = np.shape(x)[1]
        # self.a_p = np.append(self.a_p, np.ones((add_dim,1)) * (self.a0 + 0.5)).reshape(-1,1)
        # self.b_p = np.append(self.b_p, np.ones((add_dim,1)) * self.b0 ).reshape(-1,1)
        # # self.e_p = self.n/2 #+ self.dim/2
        # # self.f_p = 0.
        # self.mu_p = np.append(self.mu_p, np.zeros((add_dim,1))).reshape(-1,1) 
        # # self.mu_p = np.ones((self.dim,1))
        # self.sigma_p = block_diag(self.sigma_p, np.eye(add_dim))
        # print("kernel size after prunning before data addition", self.x.shape)
        self.x = np.append(self.x, x, axis=1)
        # print("kernel size after prunning after data addition ", self.x.shape)
        self.dim = np.shape(self.x)[1]
        self.Lv =[]
        self.xxt = np.zeros([self.dim,self.dim])
        for i in range(self.n):
            self.xxt += np.outer(self.x[i],self.x[i])
        self.yx = np.zeros([1,self.dim])
        for i in range(self.n):
            self.yx += self.y[i]*self.x[i]
        
        # np.random.seed(50)
        self.a_p = np.ones((self.dim,1)) * (self.a0 + 0.5)
        self.b_p = np.ones((self.dim,1)) * self.b0 
        self.e_p = self.n/2 #+ self.dim/2
        self.f_p = 0.
        self.mu_p = np.zeros((self.dim,1)) 
        # self.mu_p = np.ones((self.dim,1))
        self.sigma_p = np.eye(self.dim)
        # self.xxt = np.zeros([self.dim,self.dim])
        # for i in range(self.n):
        #     self.xxt += np.outer(self.x[i],self.x[i])
        # self.yx = np.zeros([1,self.dim])
        # for i in range(self.n):
        #     self.yx += self.y[i]*self.x[i]
        
    def stemPlot_relevance(self, prune_w_weight=False, weight_treshold=None, prune_w_alpha = False, alpha_treshold=None):
        # keep_mu_p = np.abs(vi.mu_p) > mu_treshold
        if prune_w_weight==True:
            relevance = [self.mu_p[i] if np.abs(self.mu_p[i]) > weight_treshold else None for i in range(len(self.mu_p))]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.stem(self.mu_p)
            ax.stem(relevance,markerfmt='C2o')
            ax.set_xlabel('Input vectors')
            ax.set_ylabel('Weights')
            ax.legend(['Weights', 'Relevance vectors'])
            alpha = self.a_p/self.b_p#np.log10(self.a_p/self.b_p).squeeze()
            # print(alpha)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(self.a_p/self.b_p, bins=50)
            plt.show()
        elif prune_w_alpha==True:
            relevance = [self.a_p[i]/self.b_p[i] if np.abs(self.a_p[i]/self.b_p[i]) < alpha_treshold else None for i in range(len(self.a_p))]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.stem(self.a_p/self.b_p)
            ax.stem(relevance,markerfmt='C2o')
            mup = self.mu_p.squeeze()
            # print(alpha)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(mup, bins=50)
            plt.show()
        else:
            raise NotImplementedError
        
        # relevance = [vi.mu_p[i] if vi.a_p[i]/vi.b_p[i] < mu_treshold else None for i in range(len(vi.mu_p))]
        # print(relevance)
        # print(vi.mu_p.max(),vi.mu_p.min())
        # fig, ax = plt.subplots(figsize=(4, 4))
        # ax.stem(self.mu_p)
        # ax.stem(relevance,markerfmt='C2o')
        # alpha = np.log10(self.a_p/self.b_p).squeeze()
        # # print(alpha)
        # fig, ax = plt.subplots(figsize=(4, 4))
        # ax.hist(self.a_p/self.b_p, bins=50)
        # plt.show()

    def _prune(self,weight_treshold):
        """Remove basis functions based on weight values."""
        # keep_alpha = self.alpha_ < self.threshold_alpha
        keep_mu_p = np.abs(self.mu_p) > weight_treshold
        keep_mu_p = keep_mu_p.squeeze()
        # print(keep_mu_p)
        # print(keep_mu_p.shape)
        # if not np.any(keep_mu_p):
        #     keep_mu_p[0] = True
        #     if self.bias_used:
        #         keep_mu_p[-1] = True

        # if self.bias_used:
        #     if not keep_mu_p[-1]:
        #         self.bias_used = False
        #     self.relevance_ = self.relevance_[keep_mu_p[:-1]]
        # else:
        #     self.relevance_ = self.relevance_[keep_mu_p]

        self.a_p = self.a_p[keep_mu_p]
        self.b_p = self.b_p[keep_mu_p]
        # self.alpha_old = self.alpha_old[keep_mu_p]
        # self.e_p = self.e_p[keep_mu_p]
        # self.f_p = self.f_p[keep_mu_p]
        # self.x = self.x[keep_mu_p]
        self.Xpruned = self.Xpruned[keep_mu_p]
        self.Xrelevance_ = self.x[:,keep_mu_p]
        # self.yrelevance_ = self.y[keep_mu_p]
        self.sigma_p = self.sigma_p[np.ix_(keep_mu_p, keep_mu_p)]
        self.mu_p = self.mu_p[keep_mu_p]
        print("{} relevence vectors".format(len(self.mu_p)))



    def posterior_predictive(self, x_test, prune=False, args=None):
        self.x_stars = x_test
        # self.ds = d
        self.bias_used = True
        # print( self.Xpruned.shape)
        # print(self.x.shape)
        # exit()
        # simple = True #False # i changed this
        if prune is False:
            # x_test = self.test_PHI[self.gamma_rbf][kk]
            print(self.rvs)
            # exit()
            x_test = self._apply_kernel( x_test, self.rvs, kernel = self.kernel, g = self.g,  train=False)
        else:
            # print(x_test.shape)
            # print(self.Xpruned.shape)
            # x_test = self._apply_kernel( x_test, self.rvs, kernel = self.kernelType, train=False)
            x_test = self._apply_kernel( x_test, self.rvs, kernel = self.kernel, train=False)


            self.mu_p = self.mu_p[args].reshape(-1,1)
            self.a_p = self.a_p[args]
            self.b_p = self.b_p[args]
            self.sigma_p = self.sigma_p[np.ix_(args.squeeze(), args.squeeze())]
            # print(self.mu_p)
            # exit()
            print("kernel size before prunning ", x_test.shape)
            x_test = x_test[:,args].squeeze() #reshape(2202,2213)
            print("kernel size after prunning ", x_test.shape)
            # print(x_test.shape)
            # exit()
        # x_test = np.append( np.ones((x_test.shape[0], 1)),x_test, axis=1)
        # print(x_test.shape)
        diag = self.a_p/self.b_p 
        # print("final kernel size ", self.x.shape)
        print("final kernel size ", x_test.shape)
        # X = self.x #self.Xrelevance_
        # y = self.y
        # print(X.shape)
        # print(y.shape)
        # exit()
        # self.M_alpha = np.diag(diag[:,0])
        # self.wHat = np.linalg.inv(X.T @ X) @ X.T @ y
        # tmp_ = np.linalg.inv(X.T @ X + self.M_alpha )
        # self.wBar = tmp_ @ X.T @ y
        # print(self.wBar)
        # print( self.wBar)
        # self.R = (y - X @ self.wHat).T @ (y - X @ self.wHat) + self.wHat.T @ X.T @ X @ self.wHat - self.wBar.T @ (X.T @ X +self.M_alpha) @ self.wBar
        # self.R = (y - X @ self.wHat).T @ (y - X @ self.wHat) + self.wHat.T @ X.T @ X @ tmp_ @ self.M_alpha @ self.wHat 
        # self.K = X.T @ X + self.M_alpha
        
        # y_predt = np.dot(x_test, self.wBar) 
        # print(y_predn)
        self.y_predn = np.dot(x_test, self.mu_p) 
        # vart = []
        self.varn = []
        for i in range(len(x_test)):
            x_new = x_test[i,:].T
            # M = x_new @ x_new.T + self.K
            # tmp = 1- x_new.T @ np.linalg.inv(M) @ x_new
            # sigma = (2* self.f0 + self.R)/((2*self.e0 + self.n) * tmp)
            # tmp_vart = (2* self.e0 + self.n) * sigma/(2*self.e0 + self.n - 2)
            tmp_varn = self.f_p/self.e_p +  x_new.T @ self.sigma_p @ x_new
            # vart.append(tmp_vart)
            self.varn.append(tmp_varn)
        # vart = np.array(vart).reshape(len(x_test),1)
        self.varn = np.array(self.varn).reshape(len(x_test),1)
        # print(var)
        # print(y_pred)
        return self.y_predn, self.varn
    
    def healing(self):
        self.y_predn_healed = []
        self.var_pred_healed = []
        diag = self.b_p/self.a_p 
        self.dim = self.x.shape[0]
        tmp1 = self.y - np.dot(self.x,self.mu_p) * self.e_p/self.f_p
        tmp2 = np.linalg.inv(self.f_p/self.e_p * np.eye(self.dim) + self.x @ np.diag(diag[:,0]) @ self.x.T)
        inds1 = np.where(self.x_stars>=-10)[0]
        inds2 = np.where(self.x_stars<-10)[0]
        for ii in range(len(self.x_stars)):
            if ii in inds1:
                self.x_star = self.x_stars[ii].reshape(-1,1)
                d = None#self.ds[ii]
                y_pred = self.y_predn[ii]
                var_pred = self.varn[ii]
                self.phi_star = self._apply_kernel(self.rvs, self.x_star, d, kernel = self.kernelType, train=False, gama=[4])
                self.phi_star_x_star = self._apply_kernel(self.x_star, self.x_star, d, kernel = self.kernelType, train=False,  gama=[4])
                self.phi_x_star = self._apply_kernel(self.x_star, self.rvs, d, kernel = self.kernelType, train=False,bias = True, gama=None)
                self.q_star = np.dot(self.phi_star.T, tmp1)
                self.e_star = self.phi_star_x_star - self.e_p/self.f_p * self.phi_x_star @ self.sigma_p @ self.x.T @ self.phi_star 
                self.s_star = self.phi_star.T @ (tmp2) @ self.phi_star
                self.alpha_star = 0.1
                y_pred_star = y_pred + self.e_star * self.q_star/(self.alpha_star + self.s_star)
                var_pred_star = var_pred + self.e_star * self.e_star /(self.alpha_star + self.s_star)
                assert self.e_star * self.e_star /(self.alpha_star + self.s_star) >= 0.
                self.y_predn_healed.append(y_pred_star[0])
                self.var_pred_healed.append(var_pred_star[0])
            else:
                y_pred = self.y_predn[ii]
                var_pred = self.varn[ii]
                self.y_predn_healed.append(y_pred)
                self.var_pred_healed.append(var_pred)
        # print(self.y_predn_healed)
        # print(self.var_pred_healed)
        return self.y_predn_healed, self.var_pred_healed,self.y_predn_healed, self.var_pred_healed

    
    def predict(self): #, logdist = False
        y_predict = self.x @ self.mu_p
        # if logdist == False:
        #     y_predict = self.x @ self.mu_p
        # else: 
        #     y_predict = np.exp(self.x @ self.mu_p + 0.5 * self.f_p/self.e_p)
        return y_predict
    def eval_l(self,para_lis=None):
        self.L = 0.
        x = self.x
        y = self.y
        n = self.n
        d = self.dim
        ep = self.e_p
        fp = self.f_p
        ap = self.a_p
        bp = self.b_p
        e0 = self.e0
        f0= self.f0
        a0 = self.a0
        b0 = self.b0
        mup = self.mu_p
        sigmap = self.sigma_p

        self.L += -n/2 * np.log(2*np.pi)
        self.L += n/2*(ep - np.log(fp) + loggamma(ep) + (1-ep) * digamma(ep))
        self.L += -0.5 * ep/fp * np.sum(np.square(y)) + np.sum(y * x @ mup) * ep/fp
        self.L += -0.5 * ep/fp * (np.sum(np.dot(x,sigmap).T*x.T) + np.sum(np.dot(x,mup @ mup.T)*x))
        self.L += -0.5 * d * np.log(2*np.pi) 
        self.L += e0 * np.log(f0) - loggamma(e0) 
        self.L += (e0 -1) * (ep - np.log(fp) + loggamma(ep) + (1-ep) * digamma(ep)) - f0*ep/fp
        self.L -= -d /2 * np.log(2*np.pi) -0.5 * np.linalg.slogdet(sigmap)[1] - 0.5 * d
        self.L -= ep * np.log(fp)- loggamma(ep) + (ep -1) * (ep - np.log(fp) + loggamma(ep) +(1- ep) * digamma(ep)) - ep
        for i in range(d):
            self.L += 0.5 * (ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i]))
            self.L += -0.5* (ap[i]/bp[i] * (sigmap[i,i] + mup[i]**2))
            self.L += a0 * np.log(b0) - loggamma(a0)
            self.L += (a0 - 1) * (ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i])) -b0 * ap[i]/bp[i]
            self.L -= ap[i] * np.log(bp[i]) - loggamma(ap[i])
            self.L -= (ap[i]-1)*(ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i])) - ap[i]
        self.Lv.append(self.L[0])
        # print(self.L)       
    # def eval_l(self,para_lis=None):
    #     self.L = 0.
    #     x = self.x
    #     y = self.y
    #     n = self.n
    #     d = self.dim
    #     ep = self.e_p
    #     fp = self.f_p
    #     ap = self.a_p
    #     bp = self.b_p
    #     e0 = self.e0
    #     f0= self.f0
    #     a0 = self.a0
    #     b0 = self.b0
    #     mup = self.mu_p
    #     sigmap = self.sigma_p

    #     self.L += -n/2 * np.log(2*np.pi)
    #     self.L += n/2*(ep - np.log(fp) + loggamma(ep) + (1-ep) * digamma(ep))
    #     self.L += -0.5 * ep/fp * np.sum(np.square(y)) + np.sum(y * x @ mup) * ep/fp
    #     self.L += -0.5 * ep/fp * (np.sum(np.dot(x,sigmap).T*x.T) + np.sum(np.dot(x,mup @ mup.T)*x))
    #     self.L += -0.5 * d * np.log(2*np.pi) 
    #     self.L += e0 * np.log(f0) - loggamma(e0) 
    #     self.L += (e0 -1) * (ep - np.log(fp) + loggamma(ep) + (1-ep) * digamma(ep)) - f0*ep/fp
    #     self.L -= -d /2 * np.log(2*np.pi) -0.5 * np.linalg.slogdet(sigmap)[1] - 0.5 * d
    #     self.L -= ep * np.log(fp)- loggamma(ep) + (ep -1) * (ep - np.log(fp) + loggamma(ep) +(1- ep) * digamma(ep)) - ep
    #     for i in range(d):
    #         self.L += 0.5 * (ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i]))
    #         self.L += -0.5* (ap[i]/bp[i] * (sigmap[i,i] + mup[i]**2))
    #         self.L += a0 * np.log(b0) - loggamma(a0)
    #         self.L += (a0 - 1) * (ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i])) -b0 * ap[i]/bp[i]
    #         self.L -= ap[i] * np.log(bp[i]) - loggamma(ap[i])
    #         self.L -= (ap[i]-1)*(ap[i] - np.log(bp[i]) + loggamma(ap[i]) + (1-ap[i]) * digamma(ap[i])) - ap[i]
    #     self.Lv.append(self.L[0])
    
############################plots#############
    def plot_inverse_lambda(self):
        e_p = self.e_p
        f_p = self.f_p
        def invgamma(x,e_p,f_p):
            y = (f_p**e_p)/gamma(e_p) * x**(-e_p-1) * np.exp(-f_p / x)
            return y
        x = np.linspace(50000,200000, 100).reshape(1,100)
        y = invgamma(x,e_p,f_p).reshape(1,100)
        plt.plot(x,y)
        plt.show()

    def plot_lambda(self):
        x = np.arange(0,200000,1000)
        e_p = self.e_p
        f_p = self.f_p
        e0 = self.e0
        f0 = self.f0
        def gammaFunc(x, e_p, f_p):
            y = f_p**e_p/gamma(e_p) * x**(e_p-1) * np.exp(-f_p * x)
            return y
        def invgammaFunc(x, e_p, f_p):
            y = f_p**e_p/gamma(e_p) * x**(-e_p-1) * np.exp(-f_p / x)
            return y
        def log_invgammaFunc(x, e_p, f_p):
            # y = f_p**e_p/gamma(e_p) * x**(-e_p-1) * np.exp(-f_p / x)
            y = e_p * np.log(f_p)-loggamma(e_p) + (-e_p-1) * np.log(x) - f_p / x
            return y
        # y_p = log_invgammaFunc(x,e_p,f_p)
        # y_0 = log_invgammaFunc(x,e0,f0)
        # y=[]
        # for i in range(len(x)):
        #     y.append(inv_f_y(x[i],e_p,f_p))
        fig , ax = plt.subplots(figsize=(5,5))
        ax.plot(list(x),list(y_0))
        ax.plot(list(x),list(y_p))
        plt.show()
    
    def plot_alpha(self):
        x = np.arange(0,200000,1000)
        a_p = self.a_p
        b_p = self.b_p
        a0 = self.a0
        b0 = self.b0
        def gammaFunc(x, a, b):
            y = b**a/gamma(a) * x**(a-1) * np.exp(-b * x)
            return y
        def invgammaFunc(x, a, b):
            y = b**a/gamma(a) * x**(-a-1) * np.exp(-a / x)
            return y
        def log_invgammaFunc(x, a, b):
            y = a * np.log(b)-loggamma(a) + (-a-1) * np.log(x) - b / x
            return y
        
        fig , ax = plt.subplots(figsize=(5,5))        
        y_0 = log_invgammaFunc(x,a0,b0)
        ax.plot(x,y_0)
        for d in range(self.dim):
            y_p = log_invgammaFunc(x,a_p[d,0],b_p[d,0])
            ax.plot(x,y_p)
        plt.show()

    def plot_alpha_VS_k(self):
        k = np.arange(1,self.dim+1,1).tolist()
        a_p = self.a_p
        b_p = self.b_p
        std_of_features = np.sqrt(self.b_p/self.a_p)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        ax[0].stem(std_of_features[:5])
        ax[1].stem(std_of_features[5:])
        plt.show()
    
    def plot_feature_dist(self, f_ind):
        x = np.arange(10,100,0.1)
        mu = self.mu_p[f_ind,0]
        var = np.sqrt(self.sigma_p[f_ind,f_ind])
        y = norm.pdf(x,mu,var)
        line, = plt.plot(x, y, color="k", alpha=0.2)
        return line
    
    def plot_TTime_dist(self):
        x = np.arange(0,8000,10)
        mu = self.mu_p[f_ind,0]
        var = np.sqrt(self.sigma_p[f_ind,f_ind])
        y = norm.pdf(x,mu,var)
        line, = plt.plot(x, y, color="k", alpha=0.2)
        return line
        

    def plot_ELBO(self, numItr,save=False):
        x = np.arange(1,numItr+1,1).tolist()
        # print(self.Lv)
        plt.plot(x,self.Lv)
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        # plt.ylim([-240000,-210000])
        # plt.show()
        if save:
            plt.savefig('elbow.png')
            plt.close()
        else:
            plt.show()
        
    def save(self):
        """save class as self.name.txt"""
        file = open(self.name+'.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open(self.name+'.txt','r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)

    def __str__(self):
        return str(self.__dict__)
