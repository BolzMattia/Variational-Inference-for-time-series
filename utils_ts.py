import numpy as np
from dateutil import parser
import pandas as pd
import scipy.stats as sps

def inv_softplus(x,_limit_upper=30,_limit_lower=1e-12):
    '''
    Returns y (float32), s.t. softplus(y)=x    
    '''
    if isinstance(x,np.float) or isinstance(x,np.int):
        if x<_limit_upper:
            ret=np.log(np.exp(x)-1)
        else:
            ret=x
    else:
        ret=np.zeros(x.shape,dtype=np.float32)
        under_limit=x<_limit_upper
        over_limit=np.logical_not(under_limit)
        ret[under_limit]=np.float32(np.log(np.exp(x[under_limit])-1+_limit_lower))    
        ret[over_limit]=x[over_limit]        
    return ret

def safe_softplus(x, limit=10):
    ret=x
    _under_limit=x<limit
    ret[_under_limit]=np.log(1.0 + np.exp(x[_under_limit]))
    return ret

def lagify(y,p):
    '''
    Taken time series y (vertical), returns columns with the last p lags of y.
    Returns both y and ylag, aligned so that ylag sees just until yesterday.
    '''
    T,N=y.shape
    ylag=np.ones([T,N*p+1])
    for pp in range(p):
        ylag[pp+1:T,N*pp+1:pp*N+N+1]=y[:T-pp-1,:]
    return np.float32(y[p:,:]),np.float32(ylag[p:,:])

def VAR_data_generation(T,N,par_p,cov_wn,const_terms):
    '''
    generates T x N data, with par_p VAR structure, cov_wn noise covariance and a vector of constant terms cont_terms.
    
    '''
    p=int(par_p.shape[0]/N)
    eps=np.random.multivariate_normal(np.zeros(N),cov_wn,size=T)
    y=np.zeros([T,N])
    last_y=np.zeros([p,N])
    ylag=np.zeros([T,N*p+1])
    for t in range(T):
        ylag[t]=np.concatenate([np.ones([1,1]),last_y.reshape(1,-1)],axis=1)
        y[t,:]=const_terms+ np.matmul(last_y.reshape(1,-1),par_p)+eps[t]
        last_y[:p-1]=last_y[1:]
        last_y[p-1]=y[t]
    return y,ylag

def spiral_indexes(N):
    '''
    return the indexes of a line vector that corresponds to the elements of a triangular matrix.
    spiral means that the elements in the matrix are inserted using a spiral sequence (as tensorflow.fill_triangular does).
    '''
    spiral_matrix=np.zeros([N,N],dtype=np.int)
    spiral_line_tril=np.zeros(int(N*(N+1)/2),dtype=np.int)
    last_num=0
    ln=0
    for n in range(N):
        if (n%2)==0:
            #assigns the inverted rows
            val_n=N-int(n/2)
            spiral_matrix[N-1-int(n/2),:N-int(n/2)]=np.flip(last_num+np.arange(val_n))

            #print(ln,ln+N-int(n/2))
            qn=N**2-int(n/2)*N
            inds=(np.arange(qn-N,qn-int(n/2)))
            spiral_line_tril[ln:ln+N-int(n/2)]=np.flip(inds)
            last_num+=val_n
            ln+=N-(int(n/2))
        else:
            #assign the rows
            val_n=int((n+1)/2)
            spiral_matrix[int((n-1)/2),:int((n+1)/2)]=last_num+np.arange(val_n)
            last_num+=val_n

            qn=(val_n-1)*N#int(val_n*(val_n-1)/2)
            inds=np.arange(qn,qn+val_n)
            spiral_line_tril[ln:ln+val_n]=inds
            ln+=val_n
    return spiral_matrix[np.diag_indices(N)],spiral_matrix[np.tril_indices(N,-1)],spiral_matrix[np.tril_indices(N)],spiral_matrix,spiral_line_tril


def fromMat2diag_udiag(mat):
    '''
    Given a matrix returns the diagonal and the strictly lower triangular part of Cholesky(mat).
    The strict lower matrix returned is normalized per the diagonal elements corresponding.
    '''
    N=mat.shape[0]
    cholmat=np.linalg.cholesky(mat)
    choldiag=np.diag(cholmat)
    normmat=np.tile(np.reshape(choldiag,[1,N]),[N,1])
    choludiag=(cholmat/normmat)[np.tril_indices(N,-1)]
    return choldiag, choludiag

def arctanh(x):
    '''
    returns arctanh(x), doesn't check for nans.
    '''   
    ret=0.5*np.log((1+x)/(1-x))
    if (np.sum(np.isnan(ret))>0):
        print(x)
        ret[np.isnan(ret)]=0.0
    return ret

class indexes_librarian:
    '''
    A single class that collects different set of indexes, useful to gather ndarrays.
    '''
    def __init__(self,N):
        self.spiral_diag,self.spiral_udiag,self.spiral_tril,self.spiral_matrix,self.spiral_line=spiral_indexes(N)
        self.diag=np.diag_indices(N)
        self.udiag=np.tril_indices(N,-1)
        self.tril=np.tril_indices(N)
        
def from_daily2_monthly(y,log_returns=False):  
    '''
    Transform the pandas Dataframe with time-index to a montly series. log_returns parameter controls if log-returns must be computed.
    '''
    ind_dates=np.zeros(y.shape[0],dtype=np.int)
    last_date=None
    jj=0
    for ii in range(y.index.shape[0]):
        date_ii=parser.parse(y.index[ii])
        if ii==0 or not date_ii.month==last_date.month:
            ind_dates[jj]=ii
            jj+=1
        last_date=date_ii
    ind_dates=ind_dates[:jj]
    ret=y.iloc[ind_dates,:].values
    if log_returns:
        ret=np.log(ret[1:,:])-np.log(ret[:-1,:])
    ret=pd.DataFrame(ret,y.index[ind_dates[1:]])
    return ret

def init_BetaPdfLowVariance_fromPoint(x,b=10.0,_min_a=1e-1):
    '''
    Given x, a ndarray of observed vaues from different Beta distributions, returns a pair of parameters a,b that corresponds to Beta distributions with expected value equal to x and variance controlled by b (bigger the b, lower the variance).
    '''
    xb=np.ones(x.shape,dtype=np.float32)*b
    xa=xb*(x)/(1.0-x)
    if isinstance(x,np.float):
        if xa/xb<_min_a:
            xa=_min_a*xb
    else:
        under_min=xa/xb<_min_a
        xa[under_min]=_min_a*xb[under_min]
    return xa,xb

def init_GammaPdfLowVariance_fromPoint(x,b=10.0):
    '''
    Given x, a ndarray of observed vaues from different Gamma distributions, returns a pair of parameters a,b that corresponds to Gamma distributions with expected value equal to x and variance controlled by b (bigger the b, lower the variance).
    '''
    xb=np.ones(x.shape,dtype=np.float32)*b
    xa=xb*x
    return xa,xb

def view_stats(x,axis=None):
    if axis is None:
        print(f'min: {np.min(x)}\nmean: {np.mean(x)}\nmax: {np.max(x)}\nstd: {np.std(x)}')    
    else:
        print(f'min: {np.min(x,axis=axis)}\nmean: {np.mean(x,axis=axis)}\nmax: {np.max(x,axis=axis)}\nstd: {np.std(x,axis=axis)}')
