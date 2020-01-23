#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 02:03:04 2019

@author: linyizi
"""

class LDA_original:
    
    @staticmethod
    def _convergence_(new, old, epsilon = 1.0e-3):
        '''
        Check convergence.
        '''
        delta = abs(new - old)
        return np.all(delta) < epsilon
    
   
    @staticmethod
    def _normalization_col(x):
        '''
        Normalize a matrix. 
        Each element is divided by the corresponding column sum.
        '''
        return x/np.sum(x,0)
    
    
    @staticmethod
    def _accumulate_Phi(beta, Phi, doc):
        '''
        This function accumulates the effect of Phi_new from all documents after e step.
        beta is V*k matrix.
        Phi is N_d * k matrix.
        Return updated beta.
        '''
        row_index = list(doc.keys())
        word_count = list(doc.values())
        for i in range(len(row_index)):
            beta[row_index[i],:] = word_count[i] * Phi[i,:]

        return beta
    
    def __init__(self, k, max_em_iter=50, max_alpha_iter=50, max_Estep_iter=50):
        self._k = k
        self._max_em_iter = max_em_iter
        self._max_alpha_iter = max_alpha_iter
        self._max_Estep_iter = max_Estep_iter
        
    def initializaiton(self, V):
        '''
        Initialize alpha and beta. 
        alpha is a k-dim vector. beta is V*k matrix.
        '''
        
        k = self._k
        
        alpha = np.random.uniform(size = k)
        
        alpha_new = alpha/np.sum(alpha)
    
        beta = np.random.dirichlet(alpha_new, V)
        
        return alpha_new, beta
    
    def Estep(self, doc, alpha, beta, N_d):
        '''
        E step for a document, which calculate the posterior parameters.
        beta_old and alpha-old is coming from previous iteration.
        Return Phi and gamma  of a document.
        '''
        
        k = self._k
        max_iter = self._max_Estep_iter
        
        gamma_old = [alpha[i] + N_d/k  for i in range(k)] 
        row_index = list(doc.keys())
        word_count = np.array(list(doc.values()))
    
        for i in range(max_iter):
            # Update Phi
            Phi = np.zeros((N_d, k))
            for i in range(N_d):
                for j in range(k):
                    Phi[i,j] = beta[row_index[i],j] * np.exp(digamma(gamma_old[j]))
                Phi[i,:] = Phi[i,:]/np.sum(Phi[i,:])
            
            #Update gamma
            Phi_sum = np.zeros(k)
            for j in range(k):
                z = 0
                for i in range(N_d):
                    z += Phi[i,j] * word_count[i]
                Phi_sum[j] = z
        
            gamma_new = alpha + Phi_sum
        
            # Converge or not
            if (i>0) & self._convergence_(gamma_new, gamma_old):
                break
            else:
                gamma_old = gamma_new.copy()
    
            
        return gamma_new, Phi
    
    def newton_raphson(self, alpha_old, gamma_matrix):
        '''
        This function uses New Raphson method to update alpha in the M step.
        alpha_old is a k-dim vector.
        gamma_matrix is a M * k matrix which stores all gamma from M documents.
        Return updated alpha.
        '''

        k = self._k
        max_iter = self._max_alpha_iter
        
        M = gamma_matrix.shape[0]
        pg = np.sum(digamma(gamma_matrix), 0) - np.sum(digamma(np.sum(gamma_matrix, 1)))
        alpha_new = alpha_old.copy()
    
        for t in range(max_iter):
        
            alpha_sum = np.sum(alpha_old)
            g = M * (digamma(alpha_sum) - digamma(alpha_old)) + pg
            h = -M * polygamma(1, alpha_old)
            z = M * polygamma(1, alpha_sum)
            c = np.sum(g/h)/(z**(-1.0) + np.sum(h**(-1.0)))
        
            delta = (g-c)/h
            alpha_new -= delta
        
            if np.any(alpha_new) < 0:
                alpha_new = self.newton_raphson(alpha_old/10, gamma_matrix)
                return alpha_new
        
            if (t > 1) & self._convergence_(delta, np.zeros((1,k))):
                break
            else:
                alpha_old = alpha_new.copy()
            
        return alpha_new
    
    def fit(self, doc, vocabulary):
        '''
        Latent Dirichlet Allocation Model.
        doc is a set of documents, each document is a dictionary.
        vocabulary contains the words in all documents.
        Return updated alpha and beta.
        '''
            
        k = self._k
        max_iter = self._max_em_iter
            
        N_d = [len(d) for d in doc] # Get the length of each document.
        V = len(vocabulary) # Get the length of vocabulary
        M = len(doc) # Get the document number.
    
       # Initialize alpha, beta and the statistics od gamma
        alpha_new, beta_new = self.initializaiton(V)
        gamma_matrix = np.zeros((M, k))
    
        for iter in range(max_iter):
            beta_old = beta_new.copy()
            alpha_old = alpha_new.copy()
            # E step
            for i in range(M):
                gamma, Phi = self.Estep(doc[i], alpha_old, beta_old, N_d[i])
                beta_new = self._accumulate_Phi(beta_new, Phi, doc[i])
                gamma_matrix[i,:] = gamma
        
            # M step
            alpha_new = self.newton_raphson(alpha_old, gamma_matrix)
            beta_new = self._normalization_col(beta_new)
        
            # check convergence
            if self._convergence_(alpha_new, alpha_old) & self._convergence_(np.sum(beta_new,0), np.sum(beta_old,0)):
                break
        
        return alpha_new, beta_new