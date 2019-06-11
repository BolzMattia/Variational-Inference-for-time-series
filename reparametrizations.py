import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from abstract_classes import VIsampler_abstract
from utils_ts import inv_softplus,init_BetaPdfLowVariance_fromPoint,fromMat2diag_udiag,indexes_librarian

class VIsampler_NullVector(VIsampler_abstract):
    '''Dummy sampler, it just gives a null vector for the mean, with constantly null entropy. Contains no lambda'''
    def __init__(self,name,MC,T,N,**kwargs):
        Ey=tf.constant(np.zeros([MC,T,N],dtype=np.float32))
        self.T,self.N=T,N
        states={'Ey':Ey}
        #entropy
        entropy=0.0
        #implements the interface
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
    def forecast(self,sess,Ey=None,**kwargs):
        '''Forecasts the mean with the null vector'''
        return {'Ey':np.zeros([1,1,self.N],dtype=np.float32)}
    def diagnostic(self,sess):
        print('Null mean')
class VIsampler_Beta(VIsampler_abstract):
    '''Reparametrization of a set of independent Beta distributions.
    The lambda are the parameters a,b of every Beta distribution.
    The entropy is exactly computed, not MC estimated.
    '''
    def __init__(self,name,MC,init_a,init_b,**kwargs):        
        lambda_a=tf.nn.softplus(tf.get_variable(name+"_lambda_a",dtype=tf.float32,initializer=inv_softplus(init_a)))
        lambda_b=tf.nn.softplus(tf.get_variable(name+"_lambda_b",dtype=tf.float32,initializer=inv_softplus(init_b)))
        pdf_beta=tfp.distributions.Beta(lambda_a,lambda_b,name=name+'_pdf_beta')
        samples=pdf_beta.sample([MC])
        states={name:samples}
        #entropy
        entropy=tf.reduce_sum(pdf_beta.entropy(),name=name+'_entropy')
        self.samples=samples
        #implements the interface
        VIsampler_abstract.__init__(self,name,MC,states,entropy)

class VIsampler_unionOfSamplers(VIsampler_abstract):
    '''Given a list of reparametrizations, it returns a reparametrization over the independent cartesian products of such reparametrizations.
    The entropy is the sum of the entropies of the single elements.
    The samplers just merges the parameters.
    '''
    def __init__(self,name,samplers):
        states={}
        entropy=0.0
        MC=None
        for sampler in samplers:
            states={**states,**sampler.states}
            entropy=entropy+sampler.entropy
            if MC is None:
                MC=sampler.MC
            else:
                assert MC==sampler.MC
        self.samplers=samplers
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
    def diagnostic(self,sess):
        '''It runs the diagnostic of every sampler.'''
        for sampler in self.samplers:
            sampler.diagnostic(sess)
    def forecast(self,sess,**kwargs):
        '''It demands the forecast to the single samplers.'''
        ret={}        
        for sampler in self.samplers:
            ret={**ret,**sampler.forecast(sess,**kwargs)}
        return ret   
        
class VIsampler_MultivariateGaussian(VIsampler_abstract):
    '''Reparametrize a time-varying multivariate gaussian distribution, with independent distribution for different times.
    lambda will be the cholesky components of the covariance matrix, at every time, and the mean vector at every time.
    The entropy is exact.
    init_mean is a vector T,N
    init_vcv is a vector T,N,N representing positive-definite matrix for every T
    samples have the dimensions: MC,T,N'''
    def __init__(self,name,MC,init_mean,init_vcv,**kwargs):
        if len(init_mean.shape)==2:
            [T,N]=init_mean.shape
        else:
            T,N=None,init_mean.shape[0]
        num_tril=int(N*(N+1)/2)        
        ind=indexes_librarian(N)
        lambda_mean=tf.get_variable(name+'_mean',dtype=tf.float32,initializer=init_mean)
        _cholvcv=np.linalg.cholesky(init_vcv)
        if T is None:
            diag_cholvcv=_cholvcv[ind.diag]
            _cholvcv[ind.diag]=inv_softplus(diag_cholvcv)
            init_lambda_vcv=_cholvcv[ind.tril]            
            init_lambda_vcv[ind.spiral_diag]=_cholvcv[ind.diag]
            init_lambda_vcv[ind.spiral_udiag]=_cholvcv[ind.udiag]
            T=1
        else:
            init_lambda_vcv=np.zeros([T,num_tril],dtype=np.float32)
            for t in range(T):
                _cholvcv_temp=_cholvcv[t]
                diag_cholvcv=_cholvcv_temp[ind.diag]
                _cholvcv_temp[ind.diag]=inv_softplus(diag_cholvcv)
                _init_lambda_vcv=_cholvcv_temp[ind.tril]
                init_lambda_vcv[t,ind.spiral_diag]=_cholvcv_temp[ind.diag]
                init_lambda_vcv[t,ind.spiral_udiag]=_cholvcv_temp[ind.udiag]
        lambda_vcv=tf.get_variable(name+'_std',dtype=tf.float32,initializer=init_lambda_vcv)
        diagSoftPlus=tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
        vcv=diagSoftPlus.forward(tfp.distributions.fill_triangular(lambda_vcv))
        pdf=tfp.distributions.MultivariateNormalTriL(lambda_mean,scale_tril=vcv)
        samples=pdf.sample(MC)
        states={name:samples}
        self.lambda_mean=lambda_mean
        self.lambda_vcv=lambda_vcv
        self.vcv=vcv
        entropy=tf.reduce_mean(pdf.entropy())*T
        self.pdf=pdf
        self.samples=samples
        self.N=N
        #implements the interface
        VIsampler_abstract.__init__(self,name,MC,states,entropy)

class Lambda_tvp_gumbel:
    '''Class that collects the lambda of bernoulli distributions for every time.
    init_tvp_success is a T,N vector with the success probability.
    It preprocess the success probability, creating variables with the logit of init_tvp_success.
    '''
    def __init__(self,name,init_tvp_success):
        [T,N]=init_tvp_success.shape
        init_J_digital=np.reshape(init_tvp_success,[T,N,1])
        init_J_digital=np.concatenate([init_J_digital,1-init_J_digital],axis=2)
        init_lambda_J_logit=np.float32(np.log(init_J_digital/(1-init_J_digital)))
        #the lambdas
        logit_probs=tf.get_variable(name+"_lambda_J_logit",dtype=tf.float32,initializer=init_lambda_J_logit)
        self.logit_probs=logit_probs
        
class VIsampler_tvpBernoulli_soft(VIsampler_abstract):
    '''Class that reparametrize time varying bernoulli variables, using the Gumbel softmax trick.
    init_success is a T,N vector with the success probabilities.
    temperature_Gumbel is the temperature parameter contrilling for the variance of the samples gradient.
    samples have dimensions MC,T,N
    The entropy is exact, while the samples are approximated!
    '''
    def __init__(self,name,MC,init_success,temperature_Gumbel=1.0,**kwargs):
        [T,N]=init_success.shape
        lambda_logitprobs=Lambda_tvp_gumbel(name+'_lambda',init_success)
        pdf_softId=tfp.distributions.RelaxedOneHotCategorical(temperature_Gumbel, lambda_logitprobs.logit_probs,name=name+'_pdf_JumpsId')
        samples_yesNo=pdf_softId.sample(MC)
        samples=samples_yesNo[:,:,:,0]
        states={name:samples}
        #applies sigmoid to reconstruct the probabilities from the logits.
        success_probs=tf.nn.sigmoid(lambda_logitprobs.logit_probs)
        entropy=-tf.reduce_mean(success_probs*tf.log(success_probs)+(1-success_probs)*tf.log(1-success_probs))*T
        #implements the interface
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
        self.samples=samples

class Lambda_CholeskyWishart:
    '''Contains the lambda that reparametrize of a Wishart matrix. More properly: its Cholesky-decomposition.
    init_posDefM is a positive definite matrix, with dimensions N,N.
    init_df are the degree of freedom of the Wishart distribution, with dimensions 1.
    applies the inverse of softplus to the diagonal elements, so that lambdas can vary with no constraints.'''
    def __init__(self,name,init_posDefM,init_df):
        N=init_posDefM.shape[0]
        init_lambda_mean,init_lambda_df=self.preProcessing_init(init_posDefM,init_df)
        self.N=init_lambda_mean.shape[0]
        self._mean_preExp=tf.get_variable(name+"_mean",dtype=tf.float32,initializer=init_lambda_mean)
        bij_softplus4diag=tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
        self.mean=bij_softplus4diag.forward(self._mean_preExp)
        self._df=tf.get_variable(name+"_df",dtype=tf.float32,initializer=np.float32(init_lambda_df))
        self.df=N+tf.nn.relu(self._df)
    def preProcessing_init(self,init_mean,init_df):
        '''Uses softplus to allow the lamdba to vary over real numbers.
        This for both the degree of freedom and the diagonal elements of the Cholesky-decomposition.'''
        N=init_mean.shape[0]
        init_lambda_mean=np.float32(init_mean)
        init_lambda_df=np.float32(init_df-N)
        init_lambda_mean=np.linalg.cholesky(init_lambda_mean)
        ind_diag=np.diag_indices(N)
        init_lambda_mean[ind_diag]=inv_softplus(init_lambda_mean[ind_diag])
        return init_lambda_mean,init_lambda_df
class VIsampler_WishartMatrix:
    '''It reparametrize a Wishart matrix.
    init_mean is the matrix mean, init_df its degrees of freedom.
    line_needed=True if the elements of the matrix (with the diagonal transformed with inverse softplus) will be needed outside the class.
    samples are triangular matrixes with dimensions MC,N,N.
    The entropy is exact.
    '''
    def __init__(self,name,MC,init_mean,init_df,line_needed=False):
        N=init_mean.shape[0]
        lambda_cholesky=Lambda_CholeskyWishart(name+'_lambda',init_mean,init_df)            
        #the samples
        pdf_wishart = tfp.distributions.Wishart(lambda_cholesky.df,scale_tril=lambda_cholesky.mean,input_output_cholesky=True,name=name+'wish_sampler')
        _tol=1e-12
        eye_tol=_tol*np.eye(N,dtype=np.float32)
        #Notice: the sample will be normalized by the sqrt of the degrees of freedom!
        samples=pdf_wishart.sample(MC)/tf.sqrt(lambda_cholesky.df)+eye_tol        
        #the entropy
        entropy=tf.add(pdf_wishart.entropy(),-0.5*lambda_cholesky.df,name=name+'_entropy')  #bze      
        states={name:samples}
        if line_needed:
            ind=indexes_librarian(N)
            samples_line=tf.gather(tf.reshape(samples,[MC,N**2]),ind.spiral_line,axis=1)
            states[name+'_line']=samples_line
            self.samples_line=samples_line
        else:
            self.samples_line=None
        #implements the interface
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
        self.samples=samples
class VIsampler_brownian_line:
    '''It reparametrizes a brownian motion, with shift different at every time and standard deviations different at every time.
    The brownian motion components are correlated via a time-fixed covariance matrix, initialized as init_W (N,N positive matrix).
    The correlations between components is decided by init_W, that is a positive definite covariance matrix of the brownian motion components.
    The lambdas are the shift and the standard deviations at every time, with the correlation matrix W driving the brownian motion.
    init_std could be a vector with dimensions T,N.
    '''
    def __init__(self,name,MC,T,N,init_std=1.0,init_W=None):
        if init_W is None:
            init_W=np.eye(N)
        num_tril=int(N*(N+1)/2)
        init_lambda_mean=np.zeros([1,T,N],dtype=np.float32)
        init_cholW=np.float32(np.linalg.cholesky(init_W))
        #correction for the standard deviations implied by init_W and the time diffusion of the brownian motion is applied. Otherwise the final samplers would not have init_std deviations.
        correction_W=np.tile(np.diag(init_cholW).reshape([1,1,N]),[1,T,1])
        correction_t=np.tile((np.sqrt(1+np.arange(T))).reshape([1,T,1]),[1,1,N])
        init_lambda_std=np.float32(init_std/correction_t/correction_W)
        lambda_mean=tf.get_variable(name=name+'_lambda_mean',initializer=init_lambda_mean)
        _lambda_std=tf.get_variable(name=name+'_lambda_std',initializer=inv_softplus(init_lambda_std))
        init_lambda_W=np.zeros(num_tril,dtype=np.float32)
        ind=indexes_librarian(N)
        init_lambda_W[ind.spiral_udiag]=init_cholW[np.tril_indices(N,-1)]
        init_lambda_W[ind.spiral_diag]=init_cholW[np.diag_indices(N)]
        
        lambda_W=tf.get_variable(name=name+'_lambda_W',initializer=init_lambda_W)
        W=tfp.distributions.fill_triangular(lambda_W)
        lambda_std=tf.nn.softplus(_lambda_std)
        
        pdf_brownian=tfp.distributions.MultivariateNormalTriL(init_lambda_mean[0,0],scale_tril=W)
        samples_brownian=tf.cumsum(pdf_brownian.sample([MC,T]),axis=1)
        samples=lambda_mean+samples_brownian*lambda_std
        self.samples=samples
        self.entropy=(pdf_brownian.entropy()+tf.reduce_sum(tf.log(lambda_std))+0.5*np.sum(np.log(1+np.arange(T))))#/(np.ones(MC,dtype=np.float32))
        self.W=W
    
class VIsampler_CholVcvBrownianDynamics(VIsampler_abstract):
    '''This class reparametrize a set of time-varying triangular matrixes with positive diagonal.
    The lambdas are: the covariance matrix driving the brownian motion components,
    the shifts of the components at every time,
    The standard deviations of the component at every time.
    The entropy is exact.
    Samples have the dimensions MC,T,N,N. 
    init_vcv_mean is the mean of the T,N,N POSITIVE DEFINITE MATRIX (not the Cholesky-decomposition).
    init_vcv_std are the initial standard deviations of the components of the Cholesky (with inverse softplus applied to diagonal elements).
    '''
    def __init__(self,name,MC,init_vcv_mean,init_vcv_std,init_W_mean,**kwargs):
        [T,N,_]=init_vcv_mean.shape
        num_tril=int(N*(N+1)/2)
        init_vcv_line=self.vcv2line(init_vcv_mean).reshape([1,T,num_tril])
        bw=VIsampler_brownian_line(name+'_blank',MC,T,num_tril,init_std=init_vcv_std,init_W=init_W_mean)
        line_cholVcv=bw.samples+init_vcv_line
        cholVcv=self.shape_line2vcv(line_cholVcv)
        states={'cholVcv':cholVcv,'line_cholVcv':line_cholVcv}
        VIsampler_abstract.__init__(self,name,MC,states,bw.entropy)
        self.samples=cholVcv
        self.samples_line=line_cholVcv
    def shape_line2vcv(self, line_vcv):
        '''Takes a line vector with the components and shapes the triangular matrix.
        It applies softplus to the diagonal elements.
        '''
        diagSoftPlus=tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
        vcv=diagSoftPlus.forward(tfp.distributions.fill_triangular(line_vcv))
        return vcv
    def vcv2line(self, init_vcv_mean):
        '''It preprocess the desired mean for the final covariance matrix,
        giving back a line with the triangular components of its cholesky decomposition. With the diagonal elements transformed with the inverse of sofplus.
        '''
        [T,N,_]=init_vcv_mean.shape
        num_tril=int(N*(N+1)/2)
        ind=indexes_librarian(N)
        init_vcv_chol=np.linalg.cholesky(init_vcv_mean)
        init_vcv_line=np.zeros([T,num_tril],dtype=np.float32)
        init_vcv_line[:,ind.spiral_diag]=inv_softplus(init_vcv_chol[:,ind.diag[0],ind.diag[1]])
        init_vcv_line[:,ind.spiral_udiag]=init_vcv_chol[:,ind.udiag[0],ind.udiag[1]]
        return init_vcv_line

    
class VIsampler_cholVcvBrownianDynamics_OU(VIsampler_abstract):
    '''Sampler that combines a preparametrization for a time-varying Cholesky-decomposition of a covariance matrix
    with parameters of an Ornstein-Uhlenbeck process for the components of the triangular matrix,
    together with an independent reparametrization of W,
    a covariance matrix driving the diffusion components of the elements of the Cholesky-decomposition of the vcv.
    The posterior of W will be assumed Wishart distributed, and reparametrized accordingly.
    inverse softplus is applied to the diagonal elements.
    init_vcv_mean are the center of the distribution of the time-varying matrix.
    init_vcv_std the standard deviations of the Cholesky-decomposition components of the triangular samples.
    init_W_mean the center of the Wishart distributed W.
    Assuming the OU process of the shape:
    dv=(mu-x)theta+W*epsilon
    init_OU_mu the mean of Wishart distribution that reparametrizes mu.
    init_OU_df the degrees of freedom of the Wishart distribution that reparametrizes mu.
    init_OU_rates is a 2 dim vector, with the parameters of the Beta distribution that reparametrize theta.
    '''
    def __init__(self,name,MC,init_vcv_mean,init_vcv_std,init_W_mean,init_W_df,init_OU_rates,init_OU_mu,init_OU_mu_df,init_vcv_df=None,LD_reparametrize=True,NNet_cholVcv=False,x=None,**kwargs):
        [T,N,_]=init_vcv_mean.shape
        if NNet_cholVcv:
            #if true, uses a Neural networks to reparametrize the Cholesky decomposition
            cholVcv=VIsampler_CholVcv_NNwishart(name+'_cholVcv',MC,x,init_vcv_mean=init_vcv_mean,init_vcv_df=init_vcv_df,line_needed=True)
        else:
            #it uses a shifted, stretched brownian motion
            cholVcv=VIsampler_CholVcvBrownianDynamics(name+'_vcv',MC,init_vcv_mean,init_vcv_std,init_W_mean)
        W=VIsampler_WishartMatrix(name+'_W',MC,init_W_mean,init_W_df)
        OU_rate=VIsampler_Beta(name+'_OU_rate',MC,init_OU_rates[0],init_OU_rates[1])
        OU_mean=VIsampler_WishartMatrix(name+'_OU_mu',MC,init_OU_mu,init_OU_mu_df,line_needed=True)
        self.entropy_cholVcv=cholVcv.entropy
        self.entropy_OU_rate=OU_rate.entropy
        self.entropy_OU_mean=OU_mean.entropy
        self.entropy_W=W.entropy
        entropy=cholVcv.entropy+OU_rate.entropy+OU_mean.entropy+W.entropy
        self.OU_rate=OU_rate
        self.OU_mean=OU_mean
        self.cholVcv=cholVcv
        self.W=W
        states={'cholVcv':cholVcv.samples,'line_cholVcv':cholVcv.samples_line,'W':W.samples,'OU_theta':OU_rate.samples,'OU_mu':OU_mean.samples,'OU_mu_line':OU_mean.samples_line}
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
    def diagnostic(self,sess):
        '''Prints the entropies of the distributions of the parameters.'''
        cholVcv_entropy,OU_mu_entropy,W_entropy,OU_rate_entropy=sess.run([self.entropy_cholVcv,self.entropy_OU_mean,self.entropy_W,self.entropy_OU_rate])
        print(f'Entropies:\ncholVcv {cholVcv_entropy}, OU mu {OU_mu_entropy}, W {W_entropy}, OU rate {OU_rate_entropy}')        

class VIsampler_multivariateGARCH(VIsampler_abstract):
    '''
    The class used to sample from a set of N-univariate GARCH parameters.
    Assuming the GARCH components as: s_t^2=mu+alpha*s_{t-1}^2+beta*epsilon^2.
    s_0 and mu are parametrized by gamma distributions, independent over single components and with respect to each other.
    alpha is parametrized by a beta distribution, for every component.
    beta_n/(1-alpha_n) is reparametrized as a beta distribution, conditionally independent from alpha.
    init_mu,init_t0 are two components vectors with the parameters of the gamma reparametrization.
    init_alpha, init_beta are two components vectors with the parameters of the beta reparametrization.
    Any of the parameters family can be fixed to its mean value, by putting ""_fix=True.
    Entropy is exact.
    '''
    def __init__(self,
                 name,
                 MC,
                 init_mu,
                 init_alpha,
                 init_beta,
                 init_t0,
                 t0_fix=True,
                 mu_fix=True,
                 alpha_fix=False,
                 beta_fix=False):
        [N,_]=init_mu.shape
        assert 2==_
        assert (N,2)==init_alpha.shape
        assert (N,2)==init_beta.shape
        init_mu=np.float32(init_mu)
        init_alpha=np.float32(init_alpha)
        init_beta=np.float32(init_beta)
        init_t0=np.float32(init_t0)
        #lambda
        lambda_t0=tf.nn.softplus(tf.get_variable(name+"_lambda_t0",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_t0)))
        lambda_mu=tf.nn.softplus(tf.get_variable(name+"_lambda_mu",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_mu)))
        lambda_alpha=tf.nn.softplus(tf.get_variable(name+"_lambda_alpha",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_alpha)))
        lambda_beta=tf.nn.softplus(tf.get_variable(name+"_lambda_beta",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_beta)))
        #samples
        pdf_gamma_t0=tfp.distributions.Gamma(lambda_t0[:,0],lambda_t0[:,1],name=name+'_pdf_t0')
        pdf_gamma_mu=tfp.distributions.Gamma(lambda_mu[:,0],lambda_mu[:,1],name=name+'_pdf_mu')
        pdf_beta_alpha=tfp.distributions.Beta(lambda_alpha[:,0],lambda_alpha[:,1],name=name+'_pdf_alpha')
        pdf_beta_beta=tfp.distributions.Beta(lambda_beta[:,0],lambda_beta[:,1],name=name+'_pdf_beta')
        if t0_fix:
            sam_t0=tf.constant(np.tile((init_t0[:,0]/init_t0[:,1]).reshape([1,N]),[MC,1]))#pdf_gamma_mu.sample([MC])
            entropy_t0=0.0
        else:
            sam_t0=pdf_gamma_t0.sample([MC])
            entropy_t0=pdf_gamma_t0.entropy()
        if mu_fix:
            sam_mu=tf.constant(np.tile((init_mu[:,0]/init_mu[:,1]).reshape([1,N]),[MC,1]))#pdf_gamma_mu.sample([MC])
            entropy_mu=0.0
        else:
            sam_mu=pdf_gamma_mu.sample([MC])
            entropy_mu=pdf_gamma_mu.entropy()
        if alpha_fix:            
            sam_alpha=tf.constant(np.tile((init_alpha[:,0]/(init_alpha[:,0]+init_alpha[:,1])).reshape([1,N]),[MC,1]))#pdf_gamma_mu.sample([MC])
            entropy_alpha=0.0
        else:
            sam_alpha=pdf_beta_alpha.sample([MC])
            entropy_alpha=pdf_beta_alpha.entropy()
        if beta_fix:            
            sam_beta=tf.constant(np.tile((init_beta[:,0]/(init_beta[:,0]+init_beta[:,1])).reshape([1,N]),[MC,1]))*(1.0-sam_alpha)
            entropy_beta=0.0
        else:
            sam_beta_unitary=pdf_beta_beta.sample([MC])
            sam_beta=sam_beta_unitary*(1.0-sam_alpha)
            entropy_beta=pdf_beta_beta.entropy()+tf.log(1.0-sam_alpha)#MC estimated
        #entropy
        states={'GARCH_t0':sam_t0,'GARCH_mus':sam_mu,'GARCH_alphas':sam_alpha,'GARCH_betas':sam_beta}
        entropy=tf.reduce_mean(tf.add(entropy_t0+entropy_mu,entropy_alpha+entropy_beta,name=name+'_entropy'))*N
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
        self.entropy=entropy
        self.lambda_t0=lambda_t0
        self.lambda_mu=lambda_mu
        self.lambda_alpha=lambda_alpha
        self.lambda_beta=lambda_beta
        self.sam_t0=sam_t0
        self.sam_mu=sam_mu
        self.sam_alpha=sam_alpha
        self.sam_beta=sam_beta
        
class VIsampler_MatrixMA(VIsampler_abstract):
    '''
    The class used to sample the correlation coefficients of the DCC.
    a's posterior is reparametrized as a Beta distribution.
    b/(1-a) is reparametrized as Beta distribution, conditionally independent from a.
    t0 components are parametrized as independent gaussians.
    init_a is two component vector with the initial parameters of the Beta distribution.
    init_b is two component vector with the initial parameters of the Beta distribution.
    init_t0 is a (N,N,2) vector s.t. init_t0(:,:,0) are the means of the correlation matrix components at time 0,
    while init_t0(:,:,1) are the standard deviations of the correlation matrix components at time 0.
    Any element can be fixed to its mean by setting ""_fix=True.
    Entropy is exact.
    '''
    def __init__(self,
                 name,
                 MC,
                 init_a,
                 init_b,
                 init_t0,
                 fix_t0=True,
                 fix_a=False,
                 fix_b=False):
        N=init_t0.shape[0]
        init_a=np.float32(init_a)
        init_b=np.float32(init_b)
        init_t0=np.copy(init_t0)
        ind_diag_N=np.diag_indices(N)
        init_t0[ind_diag_N[0],ind_diag_N[1],0]=utils_ts.inv_softplus(init_t0[ind_diag_N[0],ind_diag_N[1],0])
        #lambda
        lambda_t0_mean=tf.get_variable(name+"_lambda_t0_mean",dtype=tf.float32,initializer=init_t0[:,:,0])
        lambda_t0_std=tf.nn.softplus(tf.get_variable(name+"_lambda_t0_std",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_t0[:,:,1])))
        lambda_a=tf.nn.softplus(tf.get_variable(name+"_lambda_alpha",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_a)))
        lambda_b=tf.nn.softplus(tf.get_variable(name+"_lambda_beta",dtype=tf.float32,initializer=utils_ts.inv_softplus(init_b)))
        #samples
        pdf_gaussian_t0=tfp.distributions.Normal(lambda_t0_mean,lambda_t0_std,name=name+'_pdf_rho')
        pdf_beta_a=tfp.distributions.Beta(lambda_a[:,0],lambda_a[:,1],name=name+'_pdf_a')
        pdf_beta_b=tfp.distributions.Beta(lambda_b[:,0],lambda_b[:,1],name=name+'_pdf_b')
        if fix_t0:
            sam_t0=tf.constant(np.tile((init_t0[:,:,0]).reshape([1,N,N]),[MC,1,1]))#pdf_gamma_mu.sample([MC])
            entropy_t0=0.0
        else:
            sam_t0_posneg=pdf_gaussian_t0.sample([MC])
            bij_softplus=tfp.bijectors.TransformDiagonal(diag_bijector=tfp.bijectors.Softplus())
            sam_t0=bij_softplus.forward(sam_t0_posneg)
            entropy_t0=tf.reduce_mean(pdf_gaussian_t0.entropy())*N*N
        if fix_a:
            sam_a=tf.constant(np.tile((init_a[:,0]/(init_a[:,0]+init_a[:,1])).reshape([1,1,1]),[MC,1,1]))#pdf_gamma_mu.sample([MC])
            entropy_a=0.0
        else:
            sam_a=pdf_beta_a.sample([MC])
            entropy_a=pdf_beta_a.entropy()
        if fix_b:
            sam_b=tf.constant(np.tile((init_b[:,0]/(init_b[:,0]+init_b[:,1])).reshape([1,1,1]),[MC,1,1]))*(1.0-sam_a)
            entropy_b=0.0
        else:
            sam_b_unitary=pdf_beta_b.sample([MC])
            sam_b=sam_b_unitary*(1.0-sam_a)
            entropy_b=pdf_beta_b.entropy()+tf.log(1.0-sam_a)
        #entropy
        states={'DCC_t0':sam_t0,'DCC_a':sam_a,'DCC_b':sam_b}
        entropy=tf.add(entropy_t0,entropy_a+entropy_b,name=name+'entropy')
        VIsampler_abstract.__init__(self,name,MC,states,entropy)        
        self.entropy=entropy
        self.lambda_t0_mean=lambda_t0_mean
        self.lambda_t0_std=lambda_t0_std
        self.lambda_a=lambda_a
        self.lambda_b=lambda_b
        self.sam_t0=sam_t0
        self.sam_a=sam_a
        self.sam_b=sam_b

        
class VIsampler_DCC(VIsampler_abstract):
    '''
    The class used to sample all the parameters needed for a DCC estimation.
    Collects the reparametrizations of the GARCH and correlations parameters.
    It also computes the tensors representing the covariance matrix at any time.
    The entropy is the exact entropy of the GARCH and correlations parameters, being the dynamics of the Vcv static, given y.
    samples are MC,T,N,N matrixes s.t. (i,j,:,:) is positive definite, for every i,j.
    '''
    def __init__(self,name,MC,y,init_garch_t0,init_mu,init_alpha,init_beta,init_R0,init_a,init_b):
        [T,N]=y.shape
        y_batch=np.tile(y.reshape([1,T,N]),[MC,1,1])
        #the correlation matrix on the whole data y.
        corr_y_batch=np.tile((np.corrcoef(y.transpose([1,0]))).reshape([1,N,N]),[MC,1,1])        
        #Samples of parameters
        Rdynamics=VIsampler_MatrixMA('R_dynamics',MC,init_a,init_b,init_R0)        
        Ddynamics=VIsampler_multivariateGARCH('D_dynamics',MC,init_mu,init_alpha,init_beta,init_garch_t0)
        #Vcv construction:
        #y-->epsilon
        conditional_sigmas=[]
        conditional_sigmas.append(Ddynamics.sam_t0)
        for t in range(T):
            new_sigmas=Ddynamics.sam_mu+conditional_sigmas[t]*Ddynamics.sam_beta+np.square(y_batch[:,t])*Ddynamics.sam_alpha
            conditional_sigmas.append(new_sigmas)    
        sigmas=tf.sqrt(tf.stack(conditional_sigmas,axis=1))
        diag_sigmas=tf.linalg.diag(sigmas)
        epsilon_batch=y_batch/sigmas[:,:T]
        #epsilon-->q
        Rdynamics.sam_t0=tf.constant(np.float32(corr_y_batch))#TO IMPLEMENT A WISHART SAMPLING FOR R0
        conditional_correlations=[]
        conditional_correlations.append(Rdynamics.sam_t0)
        epsilon_cross_batch=tf_VectorsCrossProduct(epsilon_batch)
        for t in range(T):
            new_correlations=Rdynamics.sam_t0*(1-Rdynamics.sam_a-Rdynamics.sam_b)+epsilon_cross_batch[:,t]*Rdynamics.sam_a+conditional_correlations[t]*Rdynamics.sam_b
            conditional_correlations.append(new_correlations)
        correlations=tf.stack(conditional_correlations,axis=1)
        #q-->rho
        diag_correlations=tf.matrix_diag_part(correlations)
        diag_normalization=tf.sqrt(tf_VectorsCrossProduct(diag_correlations))
        correlations_normalized=correlations/diag_normalization        
        #conditional DCC matrix
        vcv=tf.matmul(diag_sigmas,tf.matmul(correlations_normalized,diag_sigmas))        
        #exposed fields
        self.sam=vcv[:,:-1]
        self.sam_forecast=vcv[:,-1:]#the last computed vcv will be the forecast for the following day.
        states={'Vcv':self.sam,'Vcv_forecast':self.sam_forecast,'R_unnormalized':correlations,**Rdynamics.states,**Ddynamics.states}
        #states={'Vcv':self.sam,'a':Rdynamics.sam_a,'b':Rdynamics.sam_b,'alpha':Ddynamics.sam_alpha,'beta':Ddynamics.sam_beta,'mu':Ddynamics.sam_mu,'R0':Rdynamics.sam_t0,'R_unnormalized':correlations}
        entropy=tf.reduce_mean(Rdynamics.entropy+Ddynamics.entropy,axis=(1,2))
        self.diag_sigmas=diag_sigmas
        self.correlations_normalized=correlations_normalized
        self.Rdynamics=Rdynamics
        self.Ddynamics=Ddynamics
        VIsampler_abstract.__init__(self,name,MC,states,entropy)        
    def forecast(self,Vcv_forecast,**kwargs):
        '''The last computed matrix, using the last observation of y, is the forecast for the tomorrow Vcv.'''
        ret={'Vcv':Vcv_forecast}
        return ret    
    def diagnostic(self,sess):
        '''plots the 10th,50th,90th percentiles of the covariance matrix.'''
        vcv_np,alpha_np=sess.run([self.sam,self.Ddynamics.sam_alpha])
        plt.plot(np.quantile(vcv_np[:,:,0,0],0.1,axis=0))
        plt.plot(np.quantile(vcv_np[:,:,0,0],0.5,axis=0))        
        plt.plot(np.quantile(vcv_np[:,:,0,0],0.9,axis=0))        
        plt.show()
    
class VIsampler_IndependentJumps_andMoments(VIsampler_abstract):
    '''
    This classes reprametrizes the posterior of jumps occurence, jumps values and its unconditional moments.
    the posterior for the occurrence, at given time for a given tile will be parametrized by a Bernoulli.
    the posterior for the values occured at any given time for any given title will be gaussian, with every component independent from each other.
    the posterior for the jump unconditional mean will be a multivariate gaussian.
    the posterior for the jump unconditional covariance matrix will be a Wishart.
    init_success_t is a T,N vector, s.t. every component is the initial posterior probabilities that a jump occurred for that element, at that time.
    init_jumps_t_mean is a T,N vector with the posterior mean of a jump at that time.
    init_jumps_t_vcv is a T,N vector with the standard deviations of the posterior distribution of the jumps, at any time, for any component.
    init_success_Beta is a 2 vector with the parameters of the Beta distribution for the probability of a jump occurring at a single time, for asingle title.
    init_jumpsMean_mean is a N vector, with the unconditional mean of the jump values.
    init_jumpsMean_vcv is a N,N vector, with the unconditional covariance of the jump mean values.
    init_jumpsVcv_mean is a N,N positive matrix, with the mean of the unconditional jump values covariance matrix.
    init_jumpsVcv_df is a 1 vector, with the degrees of freedom of the unconditional jump values covariance matrix.
    Entropy is exact.
    '''
    def __init__(self,
                 name,
                 MC,
                 init_success_t,
                 init_jumps_t_mean,
                 init_jumps_t_vcv,
                 init_success_Beta,
                 init_jumpsMean_mean,
                 init_jumpsMean_vcv,
                 init_jumpsVcv_mean,
                 init_jumpsVcv_df,
                 **kwargs):
        jumps_rate=VIsampler_Beta(name+'_rate',MC,init_success_Beta[0],init_success_Beta[1])
        jumps_softId_t=VIsampler_tvpBernoulli_soft(name+'_softId',MC,init_success_t)
        jumps_values_t=VIsampler_MultivariateGaussian(name+'_values',MC,init_jumps_t_mean,init_jumps_t_vcv)
        jumps_mean=VIsampler_MultivariateGaussian(name+'_mean',MC,init_jumpsMean_mean,init_jumpsMean_vcv)
        jumps_vcv=VIsampler_WishartMatrix(name+'_vcv',MC,init_jumpsVcv_mean,init_jumpsVcv_df)
        Ey=jumps_values_t.samples*jumps_softId_t.samples
        states={'Ey':Ey,'Jumps_rate':jumps_rate.samples,'Jumps_mean':jumps_mean.samples,'Jumps_vcv':jumps_vcv.samples,'Jumps_softId_t':jumps_softId_t.samples,'Jumps_values_t':jumps_values_t.samples}
        entropy=jumps_rate.entropy+jumps_mean.entropy+jumps_vcv.entropy+jumps_softId_t.entropy+jumps_values_t.entropy
        VIsampler_abstract.__init__(self,name,MC,states,entropy)
        self.jumps_mean=jumps_mean    
    def diagnostic(self,sess):
        '''Plots 10th,50th,90th quantiles of the jumps values samples.'''
        Ey=sess.run(self.states['Ey'])
        plt.plot(np.quantile(Ey,0.1,axis=0))
        plt.plot(np.quantile(Ey,0.5,axis=0))
        plt.plot(np.quantile(Ey,0.9,axis=0))
        plt.show()
