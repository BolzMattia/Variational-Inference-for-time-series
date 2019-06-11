###UTILS_TF
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time


class audit:
    '''
    Class used to save the tensorflow session and to create the folders where to save other files.
    '''
    def __init__(self,name,save_path,save_jpynb=None):
        import shutil
        import os
        now_string=datetime.datetime.now().strftime('%Y%m%d %H%M%S')
        folder2save = f'{save_path}\\{name} {now_string}'
        if not os.path.exists(folder2save):
            os.makedirs(folder2save)
        self.folder2save=folder2save
        self.saver=tf.train.Saver()
        self.file2save=self.folder2save + '\\tf'
        if not save_jpynb is None:
            source_path=os.getcwd()+save_jpynb
            if os.path.exists(source_path):
                shutil.copy2(source_path,folder2save + '\\' + save_jpynb)
        self.tf_saver=tf.train.Saver()        
    def log_tensorflow(self,sess):
        self.tf_saver.save(sess,self.file2save)
        print(f'tensorflow saved at:\n{self.file2save}')
    def restore_tensorflow(self,sess):
        self.tf_saver.restore(sess, self.file2save)

def tf_VectorsCrossProduct(x):
    '''Creates cross-product of tensors'''
    len_shape_x=len(x.shape)
    rx=tf.reshape(x,[*x.shape,1])
    cross_x=tf.matmul(rx,tf.transpose(rx,np.concatenate([np.arange(len_shape_x-1),[len_shape_x,len_shape_x-1]])))
    return cross_x