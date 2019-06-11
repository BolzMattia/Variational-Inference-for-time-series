from abstract_classes import Dynamics_abstract
from utils_ts import inv_softplus,indexes_librarian
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
class Dynamics_cholVcv_OUbrownian(Dynamics_abstract):
    '''Class used to represents an Ornstein-Uhlenbeck dynamics for the components of a triangular matrix.
    '''
    def __init__(self):
        #the likelihood function of the dynamics
        def llkl_fun(cholVcv,line_cholVcv,W,OU_theta=None,OU_mu_line=None,**kwargs):
            [MC,T,N,_]=cholVcv.get_shape().as_list()
            if not OU_mu_line is None:#if there is an active Ornstein-Uhlenbeck dynamics
                dvcv_mean=tf.reshape(OU_theta,[MC,1])*OU_mu_line
            else:#if there is not OU dynamics
                dvcv_mean=np.zeros([MC,num_tril],dtype=np.float32)
            pdf_dvcv = tfp.distributions.MultivariateNormalTriL(dvcv_mean, scale_tril=W)
            if OU_mu_line is None:
                dvcv=tf.transpose(line_cholVcv[:,1:,:]-line_cholVcv[:,:-1,:],[1,0,2])                
            else:
                dvcv=tf.transpose(line_cholVcv[:,1:,:]+(tf.reshape(OU_theta,[MC,1,1])-1.0)*line_cholVcv[:,:-1,:],[1,0,2])
            p0value_dvcv=pdf_dvcv.log_prob(dvcv)
            p0value=tf.reduce_sum(p0value_dvcv,axis=0)#the prior value, for every particle
            return p0value
        #inherits from Dynamics_abstract
        Dynamics_abstract.__init__(self,llkl_fun)
    def evolve_states(self,sess,cholVcv,W, OU_theta, OU_mu, OU_mu_line, line_cholVcv,**kwargs):        
        dt=1#by now only one-step-ahead forecast is implemented
        if len(cholVcv.shape)==4:#if True, I have multiple MC particles
            [MC,T,N,_]=cholVcv.shape
            num_tril=int(N*(N+1)/2)
            mcholVcv=cholVcv[:,-1]
            mOU_theta=OU_theta.reshape([MC,1])
            mOU_mu=OU_mu
            mOU_mu_line=OU_mu_line
            mW=W
            mline_cholVcv=line_cholVcv[:,-1]
            #adds the diffusion components
            dcVcv=(mW@np.random.randn(MC,num_tril,1)).reshape([MC,num_tril])
            #the predicted line of elements of the cholesky decomposition
            newline_cholVcv=(mline_cholVcv*(1.0-mOU_theta)+dcVcv+mOU_theta*mOU_mu_line).reshape([MC,dt,num_tril])            
        else:#just a single particle
            MC=1
            [T,N,_]=cholVcv.shape
            num_tril=int(N*(N+1)/2)
            mcholVcv=cholVcv[-1]
            mOU_theta=OU_theta
            mOU_mu=OU_mu
            mOU_mu_line=OU_mu_line
            mW=W
            mline_cholVcv=line_cholVcv[-1]
            #adds the diffusion components
            dcVcv=mW@np.random.randn(num_tril,1).reshape([num_tril])
            #the predicted line of elements of the cholesky decomposition
            newline_cholVcv=mline_cholVcv*(1.0-mOU_theta)+dcVcv+mOU_theta*mOU_mu_line
            newline_cholVcv=newline_cholVcv.reshape([1,dt,num_tril])     
        #this method reshapes the cholesky matrix, transforming the diagonal with softplus
        new_vcv,_,_=self._shape_vcv(newline_cholVcv,MC=MC,T=dt,N=N)
        #evaluates the tensor
        fc_vcv=np.float32(sess.run(new_vcv))
        #return a dictionary with the cholesky decomposition of the vcv
        ret={'cholVcv':fc_vcv}    
        return ret
    def _shape_vcv(self,sam_not_shaped,MC,T,N): 
        '''This function reshapes the components of the triangular matrix, sam_not_shaped,
        to a proper cholesky-decomposition of a positive matrix.
        It applies Softplus to the diagonal elements.
        '''
        _tol=1e-12#to avoid numerical instabilities, this constant is added to diagonal
        min_diag=inv_softplus(_tol)
        #this is used to extract the elements that goes into the diagonal of the matrix
        ind_precomputed=indexes_librarian(N)
        sam_diag_line = tf.nn.softplus(tf.gather(sam_not_shaped,ind_precomputed.spiral_diag,axis=2))
        sam_udiag_line = tf.gather(sam_not_shaped,ind_precomputed.spiral_udiag,axis=2)
        sam_diag = tf.linalg.diag(sam_diag_line+min_diag)
        sam_udiag=tfp.distributions.fill_triangular(sam_not_shaped)
        bijsoftplus=tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
        sam=bijsoftplus.forward(tfp.distributions.fill_triangular(sam_not_shaped))
        return sam,sam_diag_line,sam_udiag_line
class Dynamics_staticMean(Dynamics_abstract):
    '''Dummy class, it corresponds to a static mean vector for the observations, that does vary in time.'''
    def __init__(self):
        def llkl_fun(**kwargs):
            return np.float32(0.0)
        Dynamics_abstract.__init__(self,llkl_fun)
    def evolve_states(self,sess,Ey,**kwargs):
        return {'Ey':Ey[:,-1:]}
class Dynamics_DCC(Dynamics_abstract):
    '''Used to extract the forecasted vcv matrix by a DCC.
    Null dynamics on the time-varying parameters is used.
    '''
    def __init__(self):
        def llkl_fun(**kwargs):
            return np.float32(0.0)
        Dynamics_abstract.__init__(self,llkl_fun)
    def evolve_states(self,sess,Vcv_forecast,**kwargs):
        return {'Vcv':Vcv_forecast}
class Dynamics_IndependentJumps(Dynamics_abstract):
    '''The dynamics of independent jumps with fixed occurence rate=Jumps_rate and fixed distribution=N(Jumps_mean,Jumps_vcv)
    '''
    def __init__(self):
        def llkl_fun(Jumps_rate,Jumps_mean,Jumps_vcv,Jumps_softId_t,Jumps_values_t,**kwargs):    
            MC,T,N=Jumps_values_t.get_shape().as_list()
            mean_jumps_ids_t=tf.reduce_mean(Jumps_softId_t,axis=1)
            #The jumps occured Id can be softId, not taking exactly 1 or 0 values.
            #This evaluation mediates over time, to enables to compute the bernoulli density function anyway.
            p0value_softId_t=tf.reduce_mean(mean_jumps_ids_t*Jumps_rate+(1-mean_jumps_ids_t)*(1-Jumps_rate),axis=1)
            pdf_values=tfp.distributions.MultivariateNormalTriL(tf.reshape(Jumps_mean,[MC,1,N]),tf.reshape(Jumps_vcv,[MC,1,N,N]))
            p0value_values_t=tf.reduce_mean(pdf_values.log_prob(Jumps_values_t),axis=1)
            p0value=p0value_softId_t+p0value_values_t
            return p0value
        Dynamics_abstract.__init__(self,llkl_fun)
    def evolve_states(self,sess,Jumps_rate,Jumps_mean,Jumps_vcv,**kwargs):
        '''Samples the jumps occurence and their values. Then multiply them to have the value of E[y].'''
        if len(Jumps_rate.shape)==1:
            Jumps_rate=Jumps_rate.reshape([1,1])
            Jumps_mean=Jumps_mean.reshape([1,-1])
            [MC,N]=Jumps_mean.shape
            Jumps_vcv=Jumps_vcv.reshape([1,N,N])
        else:
            [MC,N]=Jumps_mean.shape
        ret={}
        rate_np,mean_np,vcv_np=Jumps_rate,Jumps_mean,Jumps_vcv@Jumps_vcv.transpose([0,2,1])
        Jumps_id=np.float32(np.random.binomial(1,rate_np))
        Jumps_values=np.zeros([MC,N],dtype=np.float32)
        for mc in range(MC):
            Jumps_values[mc]=np.random.multivariate_normal(mean_np[mc],vcv_np[mc])
        Ey=Jumps_id*Jumps_values
        Ey=Ey.reshape([MC,1,-1])
        ret={'Ey':Ey}
        return ret
    
class Dynamics_merge(Dynamics_abstract):
    '''This class merges a list of dynamics objects inheriting from Dynamics_abstract.
    Usually the dynamics works on different sets of parameters.    
    '''
    def __init__(self,dynamics2merge):
        self.merged=dynamics2merge        
        def llkl_fun(**kwargs):
            ret=None
            for dyn in self.merged:
                if ret is None:
                    ret=dyn.llkl_fun(**kwargs)
                else:
                    ret=ret+dyn.llkl_fun(**kwargs)
            return ret
        Dynamics_abstract.__init__(self,llkl_fun)
    def evolve_states(self,sess,**kwargs):
        '''It evolves the parameters according to all the dynamics and merges the evolved parameters.'''
        ret=None
        for dyn in self.merged:
            if ret is None:
                ret=dyn.evolve_states(sess,**kwargs)
            else:
                ret={**ret,**(dyn.evolve_states(sess,**kwargs))}
        return ret