import numpy as np
import numba
from scipy.special import digamma, polygamma


class LDA_OPT():
    
    @staticmethod
    def _convergence_(new, old, epsilon = 1.0e-3):
        '''
        Check convergence.
        '''
        return np.all(np.abs(new - old)) < epsilon
    
    
    @staticmethod
    def _normalization(x):
        '''
        Normalize the input.
        '''
        return x/np.sum(x)
    
  
    @staticmethod
    def _normalization_row(x):
        '''
        Normalize the matrix by row.
        '''
        return x/np.sum(x,1)[:,None]
    

    @staticmethod
    def _normalization_col(x):
        '''
        Normalize a matrix. 
        Each element is divided by the corresponding column sum.
        '''
        return x/np.sum(x,0)
    
    
    @staticmethod
    @numba.jit()
    def _accumulate_Phi(beta, Phi, doc):
        '''
        This function accumulates the effect of Phi_new from all documents after e step.
        beta is V*k matrix.
        Phi is N_d * k matrix.
        Return updated beta.
        '''
        
        beta[list(doc.keys()),:] += np.diag(list(doc.values())) @ Phi
    
        return beta
    
    
    def __init__(self, k, max_em_iter=90, max_alpha_iter=90, max_Estep_iter=90):
        '''
        Initialize a class.
        k is the topic class.
        max_em_iter, max_alpha_iter and max_Estep_iter are maximum iteration of EM algorithm, Newton-Raphson method and Estep respectively.
        '''
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
        np.random.seed(12345)
        alpha = self._normalization(np.random.uniform(size = k))
    
        beta = np.random.dirichlet(alpha, V)
        
        return alpha, beta
    
    
    def Estep(self, doc, alpha, beta, N_d):
        '''
        E step for a document, which calculate the posterior parameters.
        beta_old and alpha-old is coming from previous iteration.
        Return Phi and gamma  of a document.
        '''
        
        k = self._k
        max_iter = self._max_Estep_iter
        
        gamma_old = alpha + np.ones(k) * N_d/k
        row_index = list(doc.keys())
        word_count = np.array(list(doc.values()))
    
        for i in range(max_iter):
            # Update Phi
            Phi_exp = np.exp(digamma(gamma_old))
            Phi = beta[row_index,:] @ np.diag(Phi_exp)
            Phi_new = self._normalization_row(Phi)
        
            # Update gamma
            Phi_sum = Phi_new.T @ word_count[:,None] # k-dim
            gamma_new = alpha + Phi_sum.T[0]
        
            # Converge or not
            if (i>0) & self._convergence_(gamma_new, gamma_old):
                break
            else:
                gamma_old = gamma_new.copy()
    
            
        return gamma_new, Phi_new
    

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
    
    
    
    @numba.jit()
    def E(self, doc, alpha_old, beta_old, beta_new, gamma_matrix, N_d, M):
        '''
        Get $\gamma$ and $Phi$ for all documents and calculate the statistics for M step.
        '''
        for i in range(M):
            gamma, Phi = self.Estep(doc[i], alpha_old, beta_old, N_d[i])
            beta_new = self._accumulate_Phi(beta_new, Phi, doc[i])
            gamma_matrix[i,:] = gamma
        return beta_new, gamma_matrix
    

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
            beta_new, gamma_matrix = self.E(doc, alpha_old, beta_old, beta_new, gamma_matrix, N_d, M)
        
            # M step
            alpha_new = self.newton_raphson(alpha_old, gamma_matrix)
            beta_new = self._normalization_col(beta_new)
        
            # check convergence
            if self._convergence_(alpha_new, alpha_old) & self._convergence_(np.sum(beta_new,0), np.sum(beta_old,0)):
                break
        
        return alpha_new/np.sum(alpha_new), beta_new
