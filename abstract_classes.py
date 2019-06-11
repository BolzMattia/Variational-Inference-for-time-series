#abstract classes, the interfaces
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import datetime
import time
from utils_ts import inv_softplus,indexes_librarian,view_stats

class VIsampler_abstract:
    '''This class represents the abstract interface that a sampler must implement to fit the package guidelines.
    states represents a dictionary with the parameters tensors.
    entropy represents an MC tensors with an estimates of the entropy of the sampler (or the exact value, if available)'''
    def __init__(self,name,MC,states,entropy):
        self.name=name
        self.MC=MC
        self.states=states
        self.entropy=entropy
    def get_samples(self,sess,size=None):
        '''samples the states, evaluating the tensors and returning a dictionary with the numpy ndarray.'''
        if size is None:
            size=self.MC
        states=self.states
        ret={}
        for state in states:#for every state initializes the array that will contains the samples
            ret[state]=np.zeros([size,*states[state].get_shape().as_list()[1:]],dtype=np.float32)            
        ii=0
        while size>0:#cycles until size samples are obtained.
            di=np.min([self.MC,size])
            sam=sess.run(self.states)
            for state in states:
                ret[state][ii:ii+di]=sam[state][:di]
            size-=di
            ii+=di
        return ret    
    def diagnostic(self,sess):
        '''This functions will be called to evaluate the behaviour of the sampler while training.
        THIS FUNCTION SHOULD BE OVERRIDED WHEN INHERITED.
        '''
        states_np=sess.run(self.states)
        for state in states_np:
            #just prints the values
            print(f'state {state}:n')
            view_stats(states_np[state])
class p0_abstract:
    '''
    The abstract class used to represent a prior distribution.
    name is a string with the proper name of the prior.
    p0_fun is a function that takes the tensors (representing MC particles of parameters) as inputs and gives a tensor with the log-value of their prior.
    '''
    def __init__(self,name,p0_fun):
        self.name=name
        self.p0_fun=p0_fun
        
class Dynamics_abstract:
    '''The class that represent the requested dynamic of the parameters.
    llkl_fun is a function that takes the tensors (representing MC particles of parameters) as inputs and gives a tensor with the log-value of their likelihood.'''
    def __init__(self,llkl_fun):
        self.llkl_fun=llkl_fun
    def evolve_states(self,sess,**kwargs):
        '''This function takes MC parameters particles, and evolve the time-varying ones accordingly.
        Gives back the results as numpy ndarray.
        sess is an active tensorflo session, used to evaluate/evolve the parameters.
         **kwargs represents the parameters to evolve.
         THIS FUNCTION SHOULD BE OVERRIDED WHEN INHERITED,
         RETURNING A DICTONARY WITH THE STATES-PARAMETERS AT THE NEXT TIME-STEP.
         '''
        ret=sess.run(kwargs)
        return ret
    
class BayesianModel_abstract:
    '''
    The abstract class used to represent a model for data (likelihood and prior).
    llkl_fun is a function that takes y and tensors (representing MC particles of parameters) as inputs, and gives the value of the log-likelihood of y.
    p0_fun does the same but gives back the prior log-value for each MC particle.
    dynamics is an object inheriting from Dynamics_abstract.
    '''
    def __init__(self,name,llkl_fun,p0_fun,dynamics):        
        self.name=name
        self.llkl_fun=llkl_fun
        self.p0_fun=p0_fun
        self.dynamics=dynamics
    def forecast_logProb(self,y,sess,**kwargs):
        '''Evaluates the log-probability of y, given a set of parameters.
        y is the data to forecast.
        sess is an active tensorflow session.
        **kwargs represents the parameters.'''
        _parameters=self.dynamics.evolve_states(sess,**kwargs)
        _logProb=self._get_LogProb(y,sess,**_parameters)
        return _logProb
    def _get_LogProb(self,y,sess,**kwargs):
        '''Evaluates the log-likelihood of y, using the parameters in **kwargs and the active session sess.
        Private method not intended for outside calls.'''
        llkl=sess.run(self.llkl_fun(y,**kwargs))
        if len(llkl.shape)==2:            
            MC=llkl.shape[0]
            ret=tf.reduce_logsumexp(llkl,axis=0)-np.log(MC)
        else:
            ret=llkl
        return sess.run(ret)
class EstimatorVariationalBayes:
    '''This class collects a model a sampler and data y to run the SGD optimization.
    model inherits from BayesianModel_abstract.
    sampler inherits from VIsampler_abstract.
    '''
    def __init__(self,y,model,sampler):
        self.p0=model.p0_fun(**sampler.states)#the prior tensor
        self.dynamics=model.dynamics.llkl_fun(**sampler.states)#the dynamics of parameters tensor
        self.llkl_y=model.llkl_fun(y,**sampler.states)#the likelihood of data tensor
        self.llkl=tf.reduce_sum(self.llkl_y,axis=1)+self.dynamics#the whole likelihood
        p1=self.p0+self.llkl#posterior
        self.KL=-sampler.entropy-p1#-ELBO=-KL+Evidence
        self.p1=p1
        self.model=model
        self.sampler=sampler
        self.llkl_test_values=[]
        self.llkl_train_values=[]
    def set_optimizer_KL(self,vars2optimize=None,learning_rate=0.01,learning_beta=0.9,train_epochs_limit_max=1000,train_epochs_limit_min=10):
        '''Functions that creates the optimizer with the given tuning parameters.'''
        self.optim_KL=optimizer_SGD_VI(self.KL,self.p1,self.llkl_y,self.sampler.states,vars2optimize,learning_rate,learning_beta,train_epochs_limit_max,train_epochs_limit_min)
    def run_inference(self,sess,diagnostic=False,KL=True,audit=None,y2forecast=None):
        '''Run the SGD inference.
        sess is an active tf session.
        diagnostic=True will run the sampler diagnostic function whenever a new maximum for ELBO is reached (minimum for the KL).
        audit is an instance f the audit class, used to log the tensorflow computational graph at every minimum, if None, the last state is kept.
        if y_forecast is provided, the log probability of y forecasted at every step.
        '''
        optim=self.optim_KL
        while optim.min_is_recent() and not optim.max_train_reached():            
            print(f'train round: {optim.ii}')
            t0=time.time()
            _step,_states,_llkl=optim.step(sess,audit=audit)            
            if not y2forecast is None:#
                self.llkl_train_values.append(self.model.forecast_logProb(y2forecast,sess,**_states))
                _=sess.run(tf.reduce_logsumexp(_llkl,axis=0)-np.log(_llkl.shape[0])).mean()
                self.llkl_test_values.append(_)                
            t1=time.time()
            print(f'step time: {t1-t0}')
            if _step and diagnostic:
                #maximum of elbo is reached and diagnostic=True, so runs the sampler diagnostic.
                self.sampler.diagnostic(sess)
        if not audit is None:
            #it restores the tf state at the last maximum of the ELBO.            
            audit.restore_tensorflow(sess)
        return self.optim_KL.get_ELBO(),self.llkl_train_values,self.llkl_test_values
    def sample_log_lkl(self,sess):
        '''samples the log_lkl of the model on the training set.'''
        _llkl=sess.run(tf.reduce_logsumexp(self.llkl_y,axis=0))-np.log(self.sampler.MC)
        return np.sum(_llkl)

class VIwrapper_abstract:
    '''Class used to connect a reparametrization, a model and tuning parameters for SGD, creating and off-the-shelf forecasters of data.
    name is a string with the specific name of the instance.
    y is a 2d array of data.
    MC is the number of MC particles to use at every step.
    sampler is an object that inherits from VIsampler_abstract.
    '''
    def __init__(self,name,y,MC,sampler,model,learning_rate=0.1,train_epochs_limit_min=10,train_epochs_limit_max=1000):
        estimatorVI=EstimatorVariationalBayes(name+'_estimatorVI',y,model,sampler)        
        estimatorVI.set_optimizer_KL(learning_rate=learning_rate,learning_beta=1-learning_rate,train_epochs_limit_max=train_epochs_limit_max,train_epochs_limit_min=train_epochs_limit_min)
        #implements the interface
        self.name=name
        self.MC=MC
        self.y=y
        self.model=model
        self.sampler=sampler
        self.estimatorVI=estimatorVI
    def run_inference(self,sess,audit=None,diagnostic=False,y2forecast=None):
        '''This functions runs the SGD inference.'''
        ELBO,train_llkl,test_llkl=self.estimatorVI.run_inference(sess,diagnostic=diagnostic,audit=audit,y2forecast=y2forecast)
        if not audit is None:
            audit.restore_tensorflow(sess)
        #returns the elbo
        return ELBO,train_llkl,test_llkl
    def forecast_VI(self,sess,size=None, **kwargs):
        '''Samples parameters using the sampler and evolving the time-varying ones according to the parameters dynamics.'''
        if size is None:
            size=self.MC
        _states=self.sampler.get_samples(sess=sess,size=size)
        ret=self.sampler.forecast(sess=sess,size=size,**_states, **kwargs)
        return ret
    def forecast_MAP(self,sess, **kwargs):
        '''Returns the MAP parameters sampled during inference'''
        _states,_MAP=self.estimatorVI.optim_KL.get_MAP()
        ret=self.sampler.forecast(sess=sess,**_states, **kwargs)
        return ret,_MAP
    def forecast_maxLkl(self,sess, **kwargs):
        '''Returns the parameters with the maximum likelihood, sampled during the inference'''
        _states,_maxLkl=self.estimatorVI.optim_KL.get_maxllkl()
        ret=self.sampler.forecast(sess=sess,**_states, **kwargs)
        return ret,_maxLkl
    def get_logPy(self, y, sess, size_MC=None,VI=True,MAP=True,maxLkl=True, **kwargs):
        '''Given data y and an active tf session, it returns the logProbability obtained forecasting y.
        It's possible to select which method to use: MAP, max-likelihood or VI with parameters particles sampled from the reparametrized sampler (using size_MC particles).
        **kwargs is used in case additional parameters are requested (such as regressors x).
        '''
        ret_test={}#returns a dictonary with keys the inference methods and values the log(P(y)) of y
        ret_train={}#returns a dictonary with keys the inference methods and values the log(P(y)) on the train set
        if VI:
            #samples from the approximated posterior
            _samples_VI=self.forecast_VI(sess=sess,size=size_MC, **kwargs)
            #forecasts using the samples
            ret_VI=self.model.get_LogProb(y=y,sess=sess,**_samples_VI)
            ret_test['VI']=ret_VI
            ret_train['VI']=self.estimatorVI.sample_log_lkl(sess)
        if MAP:
            #takes the MAP parameters visited during inference
            _samples_MAP,_MAP=self.forecast_MAP(sess=sess, **kwargs)
            #forecasts using MAP parameters
            ret_MAP=self.model.get_LogProb(y=y,sess=sess,**_samples_MAP)
            ret_test['MAP']=ret_MAP
            ret_train['MAP']=_MAP            
        if maxLkl:
            #takes the max likelihood parameters, visited during inference
            _samples_maxLkl,_maxLkl=self.forecast_maxLkl(sess=sess, **kwargs)
            #forecasts using the max likelihood parameters
            ret_maxLkl=self.model.get_LogProb(y=y,sess=sess,**_samples_maxLkl)        
            ret_test['maxLkl']=ret_maxLkl
            ret_train['maxLkl']=_maxLkl
        return ret_test,ret_train
class VIwrapper_abstract:
    '''Class used to connect a reparametrization, a model and tuning parameters for SGD, creating and off-the-shelf forecasters of data.
    name is a string with the specific name of the instance.
    y is a 2d array of data.
    MC is the number of MC particles to use at every step.
    sampler is an object that inherits from VIsampler_abstract.
    '''
    def __init__(self,name,y,MC,sampler,model,learning_rate=0.1,train_epochs_limit_min=10,train_epochs_limit_max=1000):
        estimatorVI=EstimatorVariationalBayes(y,model,sampler)        
        estimatorVI.set_optimizer_KL(learning_rate=learning_rate,learning_beta=1-learning_rate,train_epochs_limit_max=train_epochs_limit_max,train_epochs_limit_min=train_epochs_limit_min)
        #the interface
        self.name=name
        self.MC=MC
        self.y=y
        self.model=model
        self.sampler=sampler
        self.estimatorVI=estimatorVI
    def run_inference(self,sess,audit=None,diagnostic=False,y2forecast=None):
        '''This functions runs the SGD inference.'''
        ELBO,train_llkl,test_llkl=self.estimatorVI.run_inference(sess,diagnostic=diagnostic,audit=audit,y2forecast=y2forecast)
        if not audit is None:
            audit.restore_tensorflow(sess)
        #returns the elbo
        return ELBO,train_llkl,test_llkl
    def get_logPy(self, y, sess, size_MC=None,VI=True,MAP=True,maxLkl=True, **kwargs):
        '''Given data y and an active tf session, it returns the logProbability obtained forecasting y.
        It's possible to select which method to use: MAP, max-likelihood or VI with parameters particles sampled from the reparametrized sampler (using size_MC particles).
        **kwargs is used in case additional parameters are requested (such as regressors x).
        '''
        ret_test={}#returns a dictonary with keys the inference methods and values the log(P(y)) of y
        ret_train={}#returns a dictonary with keys the inference methods and values the log(P(y)) on the train set
        if VI:
            if size_MC is None:
                size_MC=self.MC
            #samples from the approximated posterior
            _statesVI=self.sampler.get_samples(sess=sess,size=size_MC)  
            #forecasts using the samples
            ret_test['VI']=self.model.forecast_logProb(y,sess,**_statesVI)            
            ret_train['VI']=np.sum(self.estimatorVI.sample_log_lkl(sess))
        if MAP:
            #takes the MAP parameters visited during inference
            _statesMAP,_MAP=self.estimatorVI.optim_KL.get_MAP()
            #forecasts using MAP parameters
            ret_test['MAP']=self.model.forecast_logProb(y,sess,**_statesMAP)
            ret_train['MAP']=_MAP
        if maxLkl:
            #takes the max likelihood parameters, visited during inference
            _statesLkl,_maxLkl=self.estimatorVI.optim_KL.get_maxllkl()
            #forecasts using the max likelihood parameters
            ret_test['maxLkl']=self.model.forecast_logProb(y,sess,**_statesLkl)            
            ret_train['maxLkl']=_maxLkl
        return ret_test,ret_train

class optimizer_SGD_VI:
    def __init__(self,ELBO,p1,llkl,states,vars2optimize=None,learning_rate=0.01,learning_beta=0.9,train_epochs_limit_max=1000,train_epochs_limit_min=10):
        '''Class used to run an SGD optimization of the Elbo. It also estimates maximum likelihood and maximum-a-posteriori (MAP).'''
        self.learning_rate=learning_rate
        self.learning_beta=learning_beta #The momentum parameters for the Adam optimizer
        self.train_epochs_limit_min=np.min([train_epochs_limit_min,train_epochs_limit_max])
        self.train_epochs_limit_max=np.max([train_epochs_limit_min,train_epochs_limit_max])
        self.reset_training()
        self.ELBO=tf.reduce_mean(ELBO)
        self.states=states
        self.p1=p1
        self.llkl=llkl
        if not vars2optimize is None:
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_beta).minimize(self.ELBO,var_list=vars2optimize)
        else:
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.learning_beta).minimize(self.ELBO)
    def step(self, sess, batch_train=None,audit=None):
        '''Makes an SGD steps, verify if any particle is the new MAP or a new maximum-likelihood'''
        self.ii+=1
        _elbo,_p1,_llkl,_,_states=sess.run([self.ELBO,self.p1,self.llkl,self.optimizer,self.states],feed_dict=batch_train)            
        if np.isnan(_elbo):
            raise Exception(f'loss function gave NaN result')
        _is_min=True
        if self.ii==0:
            self.ii_min=self.ii
            self.MAP=-np.Inf
            self.max_lkl=-np.Inf
        elif _elbo<self.values[self.ii_min]:
            self.ii_min=self.ii
        else:
            _is_min=False
        if _is_min and not audit is None:
            audit.log_tensorflow(sess)
        self.values[self.ii]=_elbo
        self.is_min_value[self.ii]=_is_min
        #saves the minimum of the point loss
        _llkl_globalTime=np.sum(_llkl,axis=1)
        _llkl_max=np.max(_llkl_globalTime)
        if _llkl_max>self.max_lkl:#if true, it is the new max-likelihood
            self.particle_max_llkl=np.argmax(_llkl_globalTime)
            self.max_lkl=_llkl_globalTime[self.particle_max_llkl]
            self.states_maxllkl=_states
        _p1_max=np.max(_p1)
        if _p1_max>self.MAP:#if true, it is the new MAP
            self.MAP=_p1_max
            self.particle_MAP=np.argmax(_p1)
            self.MAP_llkl=np.sum(_llkl[self.particle_MAP])
            self.states_MAP=_states
        return _is_min,_states,_llkl
    def min_is_recent(self):
        '''Gives true if the max of the ELBO is more then train_epochs_limit_min in the past training time'''
        return np.abs(self.ii-self.ii_min)<self.train_epochs_limit_min
    def max_train_reached(self):
        '''Gives true if the maximum training time is reached'''
        return self.ii>=self.train_epochs_limit_max-1
    def reset_training(self):
        '''Reset the training time, initializing internal variables'''
        self.values=np.zeros(self.train_epochs_limit_max)
        self.is_min_value=np.zeros(self.train_epochs_limit_max)
        self.ii=-1
        self.ii_min=-1
    def get_ELBO(self):
        '''Returns the historical ELBO values, obtained during training'''
        if self.ii>=0:
            return self.values[:self.ii+1]
        else:
            return None
    def get_MAP(self):
        '''Returns the MAP parameters and value'''
        states_MAP=self.states_MAP
        particle_MAP=self.particle_MAP
        _states={}
        for st in states_MAP:
            _states[st]=(states_MAP[st])[particle_MAP:particle_MAP+1]
        return _states,self.MAP_llkl
    def get_maxllkl(self):
        '''Returns the max-likelihood parameters and value'''
        states_maxllkl=self.states_maxllkl
        particle_max_llkl=self.particle_max_llkl
        _states={}
        for st in states_maxllkl:
            _states[st]=(states_maxllkl[st])[particle_max_llkl:particle_max_llkl+1]
        return _states,self.max_lkl
    