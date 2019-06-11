# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:57:01 2019

@author: matti
"""

#import os


#r_path = r'C:/Users/matti/Documents/R/win-library/3.5'
#r_path = r'C:/Program Files/R/R-3.5.1/library'
#r_path = r'C:/Bolz/Python/envs/tensorflow/lib/R'
#r_path = r'C:/Program Files/R/R-3.5.1'
#r_path='C:/PROGRA~1/R/R-3.5.1'

#os.environ['R_HOME'] = r_path

from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import pandas as pd
import numpy as np
# import R's "base" package
base = importr('base')#,lib_loc=r_path

# import R's "utils" package
utils = importr('utils')
try:
    rmgarch=importr('rmgarch')
except:
    utils.install_packages('rmgarch')
    print('installing rmgarch package')
    rmgarch=importr('rmgarch')

#the get method will be used to extract methods form S4 classes
#getmethod = R.baseenv.get("getMethod")
r_coef_method=R["coef"]
#r_S4_coef=getmethod("coef",signature = StrVector(["lmList", ]),where = "package:lme4")
    
#used to convert a vector into it's string definition in R
def mat2rSyntax(mat):
    return str(mat).replace('[','c(').replace(']',')')
#the id of the last uGarch_spec created (will be used to create variables in R)
id_uGarch_spec=-1

class uGarch_spec:
    def __init__(self, arma_orders=None, garch_orders=None, variance_model='sGARCH', distribution_model='norm'):
        global id_uGarch_spec
        id_uGarch_spec+=1
        if arma_orders is None:
            arma_orders=[1,1]
        str_arma_orders=mat2rSyntax(arma_orders)
        if garch_orders is None:
            garch_orders=[1,1]
        str_garch_orders=mat2rSyntax(garch_orders)        
        self.R_name='uGarch_spec' + str(id_uGarch_spec)
        R(f'{self.R_name} = ugarchspec(mean.model = list(armaOrder = {str_arma_orders}), variance.model = list(garchOrder = {str_garch_orders}, model = \'{variance_model}\'), distribution.model = \'{distribution_model}\')')
        
class DCC_fit:
    def __init__(self, y, spec_uGarch=None, nums_uGarch=None, DCC_order=None, out_of_sample=0, DCC_distribution='mvnorm'):        
        global id_uGarch_spec        
        id_uGarch_spec+=1
        [T,N]=y.shape
        #creates the default uGarch_spec for every column, if not provided
        if spec_uGarch==None:
            spec_uGarch=[uGarch_spec()]
            nums_uGarch=[y.shape[1]]
        assert len(spec_uGarch)==len(nums_uGarch)
        #creates the object multispec (resemble the R one, it contains the multispecification for every Garch)
        str_vec_spec=''
        for spc in range(len(nums_uGarch)):
            str_vec_spec+=f'replicate({nums_uGarch[spc]},{spec_uGarch[spc].R_name}),'
        str_vec_spec=str_vec_spec.rstrip(',')
        self.R_uGarch_multispec_name='uGarchmulti_spec'+str(id_uGarch_spec)
        R(f'{self.R_uGarch_multispec_name}=multispec(c({str_vec_spec}))')
        self.R_uGarch_multispec=R[self.R_uGarch_multispec_name]
        #creates the global DCC specification object (dccspec, in R)
        if DCC_order is None:
            DCC_order=[1,1]
        str_DCC_order=mat2rSyntax(DCC_order)
        self.R_DCC_spec_name='DCC_spec' + str(id_uGarch_spec)
        self.R_DCC_spec=R(f'{self.R_DCC_spec_name} = dccspec(uspec = {self.R_uGarch_multispec_name}, dccOrder = {str_DCC_order}, distribution = \'{DCC_distribution}\')')                
        #fits the data with the DCC_spec
        self.R_DCCfit_name='DCC_fit'+str(id_uGarch_spec)
        R_DCCfit_func=R('dccfit')
        pandas2ri.activate()
        if isinstance(y, pd.DataFrame):            
            R_y = pandas2ri.py2ri(y)
            self.fit=R_DCCfit_func(self.R_DCC_spec,R_y,out_of_sample)
        else:
            rpy2.robjects.numpy2ri.activate()
            self.fit=R_DCCfit_func(self.R_DCC_spec,y,out_of_sample)
        #creates the empty fields that will be extracted from the R fit object
        self.fit_rcor=None
        self._chol_vcv=None
        self.out_of_sample=out_of_sample
        #7 refers to: (mu,ar,ma,omega,alpha,beta,gamma) coefficients of the ARMA+GARCH of the title
        _coef_=r_coef_method(self.fit)
        self.coef_=np.array(_coef_[:-2]).reshape(N,6)
        self.global_coef_=np.array(_coef_[-2:])
        self.N=N
    def coef(self):
        return self.coef_,self.global_coef_
    
    def corr(self):
        if self.fit_rcor is None:
            rcor_func=R['rcor']
            self.fit_corr=np.asarray(rcor_func(self.fit,'R'))
        return self.fit_corr
    
    def cov(self):
        if self.fit_rcor is None:
            rcov_func=R['rcov']
            self.fit_cov=np.asarray(rcov_func(self.fit))
        return self.fit_cov
    
    
    def chol_vcv(self):
        if self._chol_vcv is None:
            self._chol_vcv=np.linalg.cholesky(self.cov().transpose(2,0,1))
        return self._chol_vcv
    
    def chol_corr(self):
        if self._chol_corr is None:
            self._chol_corr=np.linalg.cholesky(self.corr().transpose(2,0,1))
        return self._chol_corr
    
    def forecast(self, n_ahead = 1, n_roll = None):
        N=self.N
        forecast_func=R['dccforecast']
        fitted_func=R['fitted']
        rcov_func=R['rcov']
        if n_roll is None:
            n_roll=self.out_of_sample
        _forecast=forecast_func(self.fit,n_ahead,n_roll)
        _y_hat=(fitted_func(_forecast))
        y_hat=np.asarray(_y_hat).reshape(n_roll+1,n_ahead,N)
        _vcv_hat=(rcov_func(_forecast))
        vcv_hat=np.zeros([n_ahead,n_roll+1,N,N])
        for t in range(n_roll+1):
            vcv_hat[:,t,:,:]=np.asarray(_vcv_hat[t]).reshape([N,N,n_ahead]).transpose([2,0,1])
        if n_ahead==1:
            vcv_hat=vcv_hat[0]
            y_hat=y_hat[0]
        return y_hat,vcv_hat

def dcc_fit_forecast(y,dt,y_scale=1):    
    dcc_filter=DCC_fit(y*y_scale,out_of_sample=0)
    dcc_y_hat,dcc_vcv_hat=dcc_filter.forecast(dt,0)
    dcc_y_hat/=y_scale
    dcc_vcv_hat/=y_scale**2
    vcv_is=dcc_filter.cov().transpose([2,0,1])/y_scale**2
    print(f'dcc covariance statistics:\nmin: {np.min(vcv_is)}\nmean: {np.mean(vcv_is)}\nmax: {np.max(vcv_is)}\nstd: {np.std(vcv_is)}')
    return vcv_is,dcc_vcv_hat,dcc_y_hat,dcc_filter
    
    
if __name__ == '__main__':
    qui=DCC_fit(np.random.randn(100,2))
    quo=qui.coef_()  
