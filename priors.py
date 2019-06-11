from abstract_classes import p0_abstract
from utils_ts import inv_softplus,indexes_librarian
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class p0_unionOfp0s(p0_abstract):
    '''This class just glues different priors together.
    The log-prior resulting will have as value the sum of the values of the single log-priors.
    '''
    def __init__(self,name,p0s):
        def p0_fun(**kwargs):
            p0value=0.0
            for p0 in p0s:#for every prior, adds its log-value to the final value
                p0value=p0value+p0.p0_fun(**kwargs)
            return p0value
        p0_abstract.__init__(self,name,p0_fun)

def prior_normalWishart(name,init_normal_mean,init_normal_vcv,init_W_mean,init_W_df):
    '''A prior function on a pair of (vector,positive matrix).
    The prior uses two independent distributions, a multivariate gaussian over the vector and a Wishart distributions over the covariance matrix.
    vector ~ Normal(init_normal_mean,init_normal_vcv)
    matrix ~ Wishart(init_W_df,init_W_mean/init_W_df)
    '''
    N=init_normal_mean.shape[0]
    assert N==init_W_mean.shape[0]
    assert N==init_W_mean.shape[1]
    assert N<=init_W_df
    prior_normal=tfp.distributions.MultivariateNormalFullCovariance(np.float32(init_normal_mean),np.float32(init_normal_vcv),name=name+'_normal')
    prior_W=tfp.distributions.Wishart(np.float32(init_W_df),np.float32(init_W_mean/init_W_df),input_output_cholesky=True,name=name+'_W')    
    def prior_var(samples_normal,samples_W):
        prior_t0=prior_normal.log_prob(samples_normal)+prior_W.log_prob(samples_W)
        return prior_t0
    return prior_var

def prior_Beta(name,a,b):
    '''Beta distribution prior with parameters a,b.
    '''
    pdf_p=tfp.distributions.Beta(a,b,name=name)
    def prior(sam):
        return tf.reduce_mean(pdf_p.log_prob(sam),axis=1)
    return prior

class p0_DCC(p0_abstract):
    '''
    A prior over a set of DCC parameters.
    Assuming GARCH components of the kind: sigma_t^2=mu+alpha*(sigma_{t-1}^2)+beta*epsilon_t^2.
    A Gamma distribution is used for mu. A Beta distribution is used for alpha, and another Beta distribution for beta/(1-alpha).
    Concerning the correlations coefficients a,b, two independent Beta distributions are considered.
    '''
    def __init__(self,name,N,fitGammaAlpha_mu,fitGammaBeta_mu,fitBetaA_alpha, fitBetaB_alpha,fitBetaA_beta, fitBetaB_beta,fitBetaA_a,fitBetaB_a,fitBetaA_b,fitBetaB_b,**kwargs):
        pdf_alphas=tfp.distributions.Beta(np.float32(fitBetaA_alpha),np.float32(fitBetaB_alpha))
        pdf_betas=tfp.distributions.Beta(np.float32(fitBetaA_beta),np.float32(fitBetaB_beta))
        pdf_mus=tfp.distributions.Gamma(np.float32(fitGammaAlpha_mu),np.float32(fitGammaBeta_mu))
        pdf_a=tfp.distributions.Beta(np.float32(fitBetaA_a),np.float32(fitBetaB_a))
        pdf_b=tfp.distributions.Beta(np.float32(fitBetaA_b),np.float32(fitBetaB_b))
        def p0_fun(GARCH_alphas,
            GARCH_betas,
            GARCH_mus,
            DCC_a,
            DCC_b,
            **kwargs):
            p0value_alphas = pdf_alphas.log_prob(GARCH_alphas)
            p0value_betas = pdf_betas.log_prob(GARCH_betas/(1.0-GARCH_alphas))
            p0value_mus = pdf_mus.log_prob(GARCH_mus)
            p0value_a = pdf_a.log_prob(DCC_a)
            p0value_b = pdf_b.log_prob(DCC_b)
            #the values of the densities
            p0value=tf.reduce_mean(p0value_alphas+p0value_betas+p0value_mus,axis=(1))*N+tf.reduce_mean(p0value_a+p0value_b,axis=(1,2))
            return p0value
        p0_abstract.__init__(self,name,p0_fun)
class p0_cholVcvOUW_WishartBetaWishart(p0_abstract):
    '''Prior for the cholesky-decomposition of the covariance matrix and the hyperparameters determining its dynamics.
    This are the Ornstein-Uhlenebeck parameters, and the diffusion matrix W.
    The prior used will be a wishart distribution over Vcv_0, with parameters: init_vcv_mean,init_vcv_df.
    A beta distribution over the theta rate of the OU dynamics, with parameters: init_OU_rates[0],init_OU_rates[1]
    A Wishart distribution over the mu center of the Ornstein-Uhlenbeck process, with parameters: init_OU_mu,init_OU_mu_df
    A Wishart distribution for the W diffusion matrix, with parameters: init_W_mean,init_W_df
    The prior distribution over vcv_t will then be univoquely determined.
    '''
    def __init__(self,name,init_vcv_mean,init_vcv_df,init_W_mean,init_W_df,init_OU_rates=None,init_OU_mu=None,init_OU_mu_df=None,**kwargs):
        T,N,_=init_vcv_mean.shape
        num_tril=int(N*(N+1)/2)
        ind_precomputed=indexes_librarian(N)
        #the distribution of vcv_0
        pdf_vcv0=tfp.distributions.Wishart(init_vcv_df[0],init_vcv_mean[0],input_output_cholesky=True,name=name+'_pdf_vcv0')            
        if not init_OU_mu is None:
            #distributions for the OU parameters
            pdf_OU_mu=tfp.distributions.Wishart(init_OU_mu_df,init_OU_mu,name=name+'_OU_mu')
            pdf_OU_theta=tfp.distributions.Beta(init_OU_rates[0],init_OU_rates[1],name=name+'_OU_theta') 
        pdf_W=tfp.distributions.Wishart(init_W_df,init_W_mean,input_output_cholesky=True,name=name+'_pdf_W')
        def p0_fun(cholVcv,line_cholVcv,OU_theta,OU_mu_line,W,**kwargs):            
            [MC,_,_,_]=cholVcv.get_shape().as_list()
            name_MC=name+'_MC'
            #the values of the densities
            p0value_vcv0=pdf_vcv0.log_prob(cholVcv[:,0,:])
            p0value_W=pdf_W.log_prob(W)
            #the final value of the prior
            p0value=p0value_vcv0+p0value_W
            return p0value
        p0_abstract.__init__(self,name,p0_fun)


class p0_independentGaussianJumps(p0_abstract):
    '''A prior over independent jumps events, with multivariate gaussian components.
    The jumps rate of occurence will have a Beta prior distribution with parameters init_success_Beta[0],init_success_Beta[1]
    The jumps mean will have a multivariate gaussian distribution N(init_jumpsMean_mean,init_jumpsMean_vcv).
    The covariance matrix of the jumps distribution will have a Wishart prior W(init_jumpsVcv_df,init_jumpsVcv_mean/init_jumpsVcv_df).
    The three distribution are independent.
    '''
    def __init__(self,name,
                 init_success_Beta,
                 init_jumpsMean_mean,
                 init_jumpsMean_vcv,
                 init_jumpsVcv_mean,
                 init_jumpsVcv_df,
                 **kwargs):
        p0_jump_rate=prior_Beta(name+'_jumps_rate',init_success_Beta[0],init_success_Beta[1])
        p0_jumps_moments=prior_normalWishart(name+'_jumps_moments',init_jumpsMean_mean,init_jumpsMean_vcv,init_jumpsVcv_mean,init_jumpsVcv_df)
        def p0_fun(Jumps_rate,Jumps_mean,Jumps_vcv,**kwargs):          
            p0value_moments=p0_jumps_moments(Jumps_mean,Jumps_vcv)
            p0value_rates=p0_jump_rate(Jumps_rate)            
            p0value=p0value_moments+p0value_rates
            return p0value
        p0_abstract.__init__(self,name,p0_fun)
