import tensorflow_probability as tfp
from abstract_classes import BayesianModel_abstract
#models
class BayesianModel_MeanCholVcv(BayesianModel_abstract):
    '''Class that represents a multivariate state-space model, with states that are the mean of the observed vector and the cholesky-decomposition of its covariance matrix.    
    '''
    def __init__(self,name,p0_fun,dynamics):        
        def llkl_fun(y,Ey,cholVcv,**kwargs):
            #The likelihood function, needs E[y] and cholVcv. **kwargs represents other parameters to ignore.
            pdf_y=tfp.distributions.MultivariateNormalTriL(Ey,scale_tril=cholVcv,name=name+'_pdf')
            y_llkl=pdf_y.log_prob(y)
            return y_llkl
        #inherits from BayesianModel_abstract
        BayesianModel_abstract.__init__(self,name,llkl_fun,p0_fun,dynamics)
class BayesianModel_MeanVcv(BayesianModel_abstract):
    '''Class that represents a multivariate state-space model, with states that are the mean of the observed vector and its covariance matrix.
    '''
    def __init__(self,name,p0_fun,dynamics):
        def llkl_fun(y,Ey,Vcv,**kwargs):
            #The likelihood function, needs E[y] and Vcv. **kwargs represents other parameters to ignore.
            pdf_y=tfp.distributions.MultivariateNormalFullCovariance(Ey,Vcv,name=name+'_pdf')
            y_llkl=pdf_y.log_prob(y)
            return y_llkl
        #inherits from BayesianModel_abstract
        BayesianModel_abstract.__init__(self,name,llkl_fun,p0_fun,dynamics)