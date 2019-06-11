from abstract_classes import VIwrapper_abstract
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import scipy.stats as sps
from utils_ts import inv_softplus,init_BetaPdfLowVariance_fromPoint,fromMat2diag_udiag,indexes_librarian
from reparametrizations import VIsampler_cholVcvBrownianDynamics_OU,VIsampler_DCC,VIsampler_IndependentJumps_andMoments,VIsampler_NullVector,VIsampler_unionOfSamplers
from priors import p0_cholVcvOUW_WishartBetaWishart,p0_independentGaussianJumps,p0_unionOfp0s
from dynamics import Dynamics_cholVcv_OUbrownian,Dynamics_IndependentJumps,Dynamics_merge,Dynamics_staticMean
from models import BayesianModel_MeanCholVcv,BayesianModel_MeanVcv
from sklearn.covariance import LedoitWolf as LedoitWolf
def filter_W_fromVcv(vcv,variance_perc=1.0):
    '''vcv is a filtered value for the Vcv, with shapes T,N,N.
    It filters init_W,init_df that are the initial distribution parameters for W's posterior.
    W is the diffusion matrix of the components of the cholesky-decomposition of vcv.
    It filters also init_vcv_std, the standard deviations of this components' posteriors.
    '''
    [T,N,_]=vcv.shape
    num_tril=int(N*(N+1)/2)
    chol_vcv=np.zeros([T,int(N*(N+1)/2)])
    ind=indexes_librarian(N)
    for t in range(T):
        cvcv=np.linalg.cholesky(vcv[t])                
        chol_vcv[t,ind.spiral_diag]=inv_softplus(cvcv[ind.diag[0],ind.diag[1]])
        chol_vcv[t,ind.spiral_udiag]=cvcv[ind.udiag[0],ind.udiag[1]]        
    cov = LedoitWolf().fit(chol_vcv[1:,:]-chol_vcv[:-1,:])
    init_W=cov.covariance_
    try:
        np.linalg.cholesky(init_W)
    except:
        #adds a constant term if init_W is singular
        print('W resulted singular, a correction term (I*1e-4) is added')
        init_W+=np.eye(num_tril)*1e-4
    init_df=np.max([4*num_tril/variance_perc,num_tril])
    init_W*=2
    init_vcv_std=np.abs(chol_vcv)*0.1/N*variance_perc
    #init_vcv_std=np.tile(np.reshape(np.abs(vcv).mean(axis=0),[1,N,N]),[T,1,1])/np.sqrt(N)*variance_perc    
    return np.float32(init_W),np.float32(init_df),np.float32(init_vcv_std)

def filter_vcv_from_Y(y,time_window=None,exclude_center=False, unilateral=0,use_LedoitWolf=True):
    '''Given data y, with shape T,N, filters the vcv of Y, time by time, with a rolling window.
    unilateral=-1, uses a left side rolling window, =0 a centered rolling windows,=1 a right side rolling window.
    exclude_center=true will exclude the central observation while computing its vcv (shoul be used to filter jumps).
    '''
    [T,N]=y.shape
    if time_window is None:
        time_window=2*N
    if time_window>=T:
        vcv=np.tile(np.reshape(np.cov(y.transpose([1,0])),[1,N,N]),[T,1,1])
    else:
        vcv=np.zeros([T,N,N],dtype=np.float32)
        for t in range(T):
            tl=t-time_window
            tcl=t
            tcr=t
            tr=t+time_window
            if exclude_center:
                tcl-=1
            if unilateral==-1:
                y_select=y[np.max([0,tl]):np.max([time_window,tcl])]
            elif unilateral==0:
                y_select=np.concatenate([y[np.max([0,tl]):np.min([T,tcl])],y[np.max([0,tcr]):np.min([T,tr])]],axis=0)
            elif unilateral==1:
                y_select=y[np.max([0,np.min([T-time_window,tcr])]):np.min([T,tr])]           
            #computes the cov of this wime-window
            if use_LedoitWolf:
                cov = LedoitWolf().fit(y_select)
                vcv[t]=cov.covariance_                
            else:
                y_select=y_select.transpose([1,0])
                vcv[t]=np.cov(y_select)
    return vcv
def y2init_jumps(y,vcv=None,variance_perc=1.0):
    '''Given data y and a filtering of their vcv, it filters:
    a posterior distribution for the jump probability and values (as independent gaussian) at every time.
    It also filters a distribution for the jumps unconditional moments and unconditional success rate.
    (multivariate gaussian for the mean and Wishart for the covariance).
    '''
    [T,N]=y.shape
    if vcv is None:
        #if vcv is None, it filters out using a rolling window, excluding the center observation (that could be a jump)
        vcv=filter_vcv_from_Y(y,exclude_center=True,unilateral=-1)
    _MC=10000#particles used to determine quantiles of distribution density
    _q=0.05#the quantile used to filter out what could be a jump
    outliers=np.zeros(T)
    pdf_y=np.zeros(T)
    pdf_artificial=np.zeros([T,_MC])#the density values simulated
    for t in range(T):
        pdf_conditional_vcv=sps.multivariate_normal(mean=np.zeros(N),cov=vcv[t])
        pdf_y[t]=pdf_conditional_vcv.logpdf(y[t])    
        pdf_artificial[t]=pdf_conditional_vcv.logpdf(pdf_conditional_vcv.rvs(size=_MC))
        #checks which quantile the density of y belongs, with respect to the simulated ones
        qy=np.mean(pdf_y[t]>pdf_artificial[t])
        #if it lower then _q, corrected for false positives rate, it will be treated like a jump
        outliers[t]=1*(qy<(_q/np.sqrt(T))) 
    p_success=np.float32(np.mean(outliers))#the overall mean of jumps
    if p_success<=2/T:#if too rare jumps are filtered, it uses defaults values as filters
        print('jumps filtering: correction for mean applied')
        p_success=np.float32(1/T)
        init_jumpsMean_mean=np.zeros(N,dtype=np.float32)
        init_success_t=1/T*np.ones([T,N],dtype=np.float32)
        init_jumps_t_mean=np.zeros([T,N],dtype=np.float32)
        init_jumps_t_vcv=np.tile((vcv.mean(axis=0)*p_success).reshape([1,N,N]),[T,1,1])    
        init_jumpsMean_vcv=vcv.mean(axis=0)/T
    else:
        oT=np.sum(outliers)#how many jumps
        is_outlier=outliers>0#casts to boolean which observations are probably jumps
        y_select=y[is_outlier]#selects the probably jumps observations
        init_jumpsMean_mean=np.mean(y_select,axis=0)#their mean will be the unconditional mean of the jumps
        init_success_t=p_success*np.ones([T,N],dtype=np.float32)
        init_success_t[is_outlier]=0.51#the filtered jumps are 0.51 probably jumps
        init_jumps_t_mean=np.zeros([T,N],dtype=np.float32)
        init_jumps_t_mean[is_outlier]=y_select#the jumps will have posterior mean equal to the observations
        init_jumps_t_vcv=np.tile((vcv[is_outlier].mean(axis=0)/T).reshape([1,N,N]),[T,1,1])#the mean of the vcv for the jumps will be the posterior covariance for jumps
        init_jumpsMean_vcv=(vcv[is_outlier].mean(axis=0)/T)
    
    init_jumpsVcv_df=2*N
    init_success_Beta=init_BetaPdfLowVariance_fromPoint(p_success*np.ones(1,dtype=np.float32))
    init_jumpsVcv_mean=vcv.mean(axis=0)/T    
    ret={'init_success_t':init_success_t,'init_jumps_t_mean':init_jumps_t_mean,'init_jumps_t_vcv':init_jumps_t_vcv,'init_success_Beta':init_success_Beta,'init_jumpsMean_mean':init_jumpsMean_mean,'init_jumpsMean_vcv':init_jumpsMean_vcv,'init_jumpsVcv_mean':init_jumpsVcv_mean,'init_jumpsVcv_df':init_jumpsVcv_df}
    return ret
def init_OUrate_from_vcv(init_vcv,df=20):
    '''Filters a probable value for the Ornstein-Uhlenbeck rate theta,
    by using linear regression of vcv_t components on vcv_{t-1} components'''
    from sklearn.linear_model import LinearRegression
    xl=init_vcv[:-1].reshape(-1,1)
    yl=init_vcv[1:].reshape(-1)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(xl, yl)    
    init_OU_rates=np.float32([1+(1-lr.coef_[0])*df,1+df])
    return init_OU_rates
    
def y2init_OUVcv(y,init=None,variance_perc=1.0,use_LedoitWolf=True):
    '''Given data y, filters all the initial elements to initialize a brownian reparametrization of a Cholesky-decomposition of vcv'''
    [T,N]=y.shape
    num_tril=int(N*(N+1)/2)
    if init is None:
        init={}
    if not 'init_vcv_mean' in init:
        init_vcv=filter_vcv_from_Y(y,use_LedoitWolf=use_LedoitWolf)
    else:
        init_vcv=init['init_vcv_mean']
    init_vcv_df=4*N*np.ones(T,dtype=np.float32)
    init_W,init_W_df,init_vcv_std=filter_W_fromVcv(init_vcv,variance_perc)
    init_OU_rates=init_OUrate_from_vcv(init_vcv,N)
    init_OU_mu=np.float32(init_vcv.mean(axis=0))
    init_OU_mu_df=np.max([4*num_tril*variance_perc,num_tril])
    ret={'init_vcv_mean':np.float32(init_vcv),'init_vcv_std':np.float32(init_vcv_std),'init_vcv_df':np.float32(init_vcv_df),'init_W_mean':np.float32(init_W),'init_W_df':np.float32(init_W_df),'init_OU_rates':np.float32(init_OU_rates),'init_OU_mu':np.float32(init_OU_mu),'init_OU_mu_df':np.float32(init_OU_mu_df)}
    return ret

#wrappers
class VIwrapper_JumpsOUdiff(VIwrapper_abstract):
    '''This class is used to glue together the steps needed to inference a OU-diffusion+jumps model.
    Namely: initialize the variational parameters and the sampler, creates the prior, creates the model, then initialize the SGD optimizations process.
    y are data with dimensions T,N; MC is the number of MC-particles to use during SGD.
    init is a dictionary with the initialization points for the prior and for the variational parameters.
    The initialization parameters not supplied will be filtered from data.
    jumps=True introduce independent jumps in the model.
    variance_perc controls how much inflate the variance of the initial variational distibution.
    learning_rate,train_epochs_limit_min,train_epochs_limit_max are tuning parameters for the SGD run.
    '''
    def __init__(self,name,y,MC,init=None,jumps=True,init_LedoitWolf_vcv=True,variance_perc=1.0,learning_rate=None,train_epochs_limit_min=10,train_epochs_limit_max=1000):
        [T,N]=y.shape
        if learning_rate is None:
            learning_rate=1/T/N
        if init is None:
            init={}
        #completes the initialization parameters missing, filtering it from data
        init={**init,**y2init_OUVcv(y,init,variance_perc=variance_perc,use_LedoitWolf=init_LedoitWolf_vcv)}
        if jumps:            
            init_jumps=y2init_jumps(y,variance_perc=variance_perc)
            init={**init_jumps,**init}
        #initialize the reparametrization of the covariance matrix
        cholVcv=VIsampler_cholVcvBrownianDynamics_OU(name+'_cholVcv',MC,**init,LD_reparametrize=False)
        #initialize the prior
        p0_cholVcv=p0_cholVcvOUW_WishartBetaWishart(name+'_p0',**init)
        #selects the dynamics for the parameters
        dyn_cholVcv=Dynamics_cholVcv_OUbrownian()
        if jumps:
            #initialize the reparametrization of jumps
            Ey=VIsampler_IndependentJumps_andMoments(name+'_jumps',MC,**init)
            #and the jumps prior
            p0_Jumps=p0_independentGaussianJumps(name+'_p0Jumps',**init)
            #put together the covariance prior and the jumps prior (independents)
            p0=p0_unionOfp0s(name+'_p0',[p0_Jumps,p0_cholVcv])
            #selects the dynamics for the jumps
            dyn_Ey=Dynamics_IndependentJumps()
        else:
            #without jumps, null mean with static dynamics is selected
            Ey=VIsampler_NullVector(name+'_Ey',MC,T,N)
            dyn_Ey=Dynamics_staticMean()
            p0=p0_cholVcv
        #the final reparametrization is the independent product of the mean and covariance matrix reparametrizations
        sampler=VIsampler_unionOfSamplers(name+'_sam_final',[cholVcv,Ey])
        #the model dynamics, mixing the dynamics for the cholVCV with the dynamics of E[y]
        dynamics=Dynamics_merge([dyn_cholVcv,dyn_Ey])
        #the complete model
        model=BayesianModel_MeanCholVcv(name+'_model',p0.p0_fun,dynamics)
        #implements the wrapper interface
        VIwrapper_abstract.__init__(self,name+'_wrapper',y,MC,sampler,model,learning_rate=learning_rate,train_epochs_limit_min=train_epochs_limit_min,train_epochs_limit_max=train_epochs_limit_max)
        self.init=init
        
from utils_ts import init_BetaPdfLowVariance_fromPoint,init_GammaPdfLowVariance_fromPoint
class VIwrapper_DCC(VIwrapper_abstract):
    '''
    The class used to fit a variational infered DCC to data y.
    It does the necessary operations: initialize the reparametrization of the parameters, it creates the prior and the model, initialize the SGD optimizer.
    y is a T,N vector, MC the number of MC-particles to use during SGD optimization.
    coef_garch_dcc,coef_corr_dcc are the initialization points, both for the prior and the variational parameters.
    variance_perc will be used to decide how to inflate the prior and the variational parameters around this filtered values.
    If not supplied, the R package rmgarch will be used to fit a simple DCC and its parameters will be used, instead.
    p0 is a prior function, if not supplied will be created an empirical prior using the above mentioned values.
    learning_rate,train_epochs_limit_min,train_epochs_limit_max are tuning parameter for the SGD optimizer.
    '''
    def __init__(self,name,y,MC,variance_perc=1.0,coef_garch_dcc=None,coef_corr_dcc=None,p0=None,learning_rate=None,train_epochs_limit_min=10,train_epochs_limit_max=100):
        [T,N]=y.shape
        if learning_rate is None:
            learning_rate=1/T/N**2
        if coef_corr_dcc is None:
            #runs the rmgarch package from R
            from r_DCC import dcc_fit_forecast
            dcc_vcv_train,dcc_vcv_forecast,dcc_y_hat,dcc_fit=dcc_fit_forecast(y,1,y_scale=1)
            coef_garch_dcc,coef_corr_dcc=dcc_fit.coef()
        self.coef_garch_dcc,self.coef_corr_dcc=coef_garch_dcc,coef_corr_dcc
        #preprocess the variatoinal pramaters initialization points
        init_VI=self.initialization_VI_point(y,variance_perc,coef_garch_dcc,coef_corr_dcc,dcc_vcv_train)
        if p0 is None:
            #creates the prior function, if not supplied
            init_p0=self.initialization_p0(variance_perc,coef_garch_dcc,coef_corr_dcc)
            p0=p0_DCC(name+'_p0',N,*init_p0)
        else:
            init_p0=None
        #null mean is used (ARMA component is not implemented yet, left to future release)
        Ey=VIsampler_NullVector(name+'_Ey',MC,T,N)
        #the covariance matrix reparametrization
        samDCC_Vb=VIsampler_DCC(name+'_DCC',MC,y,*init_VI)
        #the Mean+Vcv reparametrization
        sampler=VIsampler_unionOfSamplers(name+'_vb_DCC_model',[Ey,samDCC_Vb])
        #dynamics of the parameters (although for DCC it is not needed, dummy dynamics will be used)
        dyn_Ey=Dynamics_staticMean()
        dyn_Vcv=Dynamics_DCC()
        dynamics=Dynamics_merge([dyn_Vcv,dyn_Ey])
        #the full model (prior+dynamics)
        model=BayesianModel_MeanVcv(name+'_model',p0.p0_fun,dynamics)        
        #the wrapper interface
        VIwrapper_abstract.__init__(self,name,y_train,MC,sampler,model,learning_rate=learning_rate,train_epochs_limit_max=train_epochs_limit_max,train_epochs_limit_min=train_epochs_limit_min)        
        #saves the initialitation point of the rmgarch package (from R)        
        self.dcc_vcv_train=dcc_vcv_train
        self.dcc_vcv_forecast=dcc_vcv_forecast
        self.dcc_y_hat=dcc_y_hat
        self.dcc_fit=dcc_fit
    def initialization_p0(self,variance_perc,coef_garch_dcc,coef_corr_dcc,_degree_of_freedom=2.0,_min_Beta_a=1e-1):
        '''Computes the prior, given a single value of the DCC parameters and a dispersion parameter variance_perc'''
        fitGammaAlpha_mu, fitGammaBeta_mu=init_GammaPdfLowVariance_fromPoint(coef_garch_dcc[:,3],_degree_of_freedom)        
        fitBetaA_alpha, fitBetaB_alpha=init_BetaPdfLowVariance_fromPoint(coef_garch_dcc[:,4],_degree_of_freedom)        
        fitBetaA_beta, fitBetaB_beta=init_BetaPdfLowVariance_fromPoint(coef_garch_dcc[:,5]/(1.0-coef_garch_dcc[:,4]),_degree_of_freedom)
        fitBetaA_a,fitBetaB_a=init_BetaPdfLowVariance_fromPoint(coef_corr_dcc[0],_degree_of_freedom)
        fitBetaA_b,fitBetaB_b=init_BetaPdfLowVariance_fromPoint(coef_corr_dcc[1],_degree_of_freedom)
        return fitGammaAlpha_mu,fitGammaBeta_mu,fitBetaA_alpha, fitBetaB_alpha,fitBetaA_beta, fitBetaB_beta,fitBetaA_a,fitBetaB_a,fitBetaA_b,fitBetaB_b
    def initialization_VI_point(self,y,perc_variance_TN,coef_garch_dcc,coef_corr_dcc,dcc_vcv_train):
        '''Computes the initial variational parameters, given a single value of the DCC parameters, data y and a dispersion parameter variance_perc'''
        [T,N]=y.shape
        const_obsv=T*N*perc_variance_TN
        const_obsv2=T*N*perc_variance_TN
        _=init_BetaPdfLowVariance_fromPoint(coef_corr_dcc[0],const_obsv2)
        init_a=np.zeros([1,2,1],dtype=np.float32)
        init_a[0,0]=_[0]
        init_a[0,1]=_[1]
        _=init_BetaPdfLowVariance_fromPoint(coef_corr_dcc[1]/(1.0-coef_corr_dcc[0]),const_obsv2)
        init_b=np.zeros([1,2,1],dtype=np.float32)
        init_b[0,0]=_[0]
        init_b[0,1]=_[1]
        
        init_mu=np.zeros([N,2],dtype=np.float32)
        _=init_GammaPdfLowVariance_fromPoint(coef_garch_dcc[:,3],const_obsv)
        init_mu[:,0]=_[0]
        init_mu[:,1]=_[1]

        init_garch_t0=np.zeros([N,2],dtype=np.float32)
        _=init_GammaPdfLowVariance_fromPoint(np.diag(dcc_vcv_train[0]),const_obsv)
        init_garch_t0[:,0]=_[0]
        init_garch_t0[:,1]=_[1]

        _=init_BetaPdfLowVariance_fromPoint(coef_garch_dcc[:,4],const_obsv)
        init_alpha=np.ones([N,2],dtype=np.float32)
        init_alpha[:,0]=_[0]
        init_alpha[:,1]=_[1]

        _=init_BetaPdfLowVariance_fromPoint(coef_garch_dcc[:,5]/(1.0-coef_garch_dcc[:,4]),const_obsv)
        init_beta=np.ones([N,2],dtype=np.float32)
        init_beta[:,0]=_[0]
        init_beta[:,1]=_[1]
        
        init_R0=np.concatenate([np.eye(N).reshape([N,N,1]),np.ones([N,N]).reshape([N,N,1])],axis=2)
        corr_y=np.corrcoef(y.transpose([1,0])).reshape([N,N,1])
        init_R0=np.concatenate([corr_y,np.abs(corr_y)/(const_obsv)],axis=2)
        init_R0=np.float32(init_R0)
        return init_garch_t0,init_mu,init_alpha,init_beta,init_R0,init_a,init_b