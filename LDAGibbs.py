#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma
from collections import OrderedDict
import pandas as pd
from numba import jit

class DataPreProcessing(object):
    def __init__(self):
        self.docs_count=0
        self.words_count=0
        self.docs=[]
        self.word2id=OrderedDict()
        self.id2word=OrderedDict()
        
def preprocessing(corpus):
    word_index=0
    dpre=DataPreProcessing()
    for sentence in corpus:
        s=[]
        for word in sentence:
            if word in dpre.word2id.keys():
                s.append(dpre.word2id[word])
            else:
                dpre.word2id[word]=word_index
                s.append(word_index)
                word_index+=1
        dpre.docs.append(s)
    dpre.docs_count=len(dpre.docs)
    dpre.words_count=len(dpre.word2id)
    dpre.id2word={v:k for k,v in dpre.word2id.items()}
    return dpre



class LDAModel(object):
    """
    Latent Dirichlet Allocation
    implementation of `Latent Dirichlet Allocation` by David M.Blei, et al. (2003)
    """
    def __init__(self,dpre,K,alpha=0.1,beta=0.01,max_iter=100,seed=1):
        #initial var
        self.dpre=dpre
        self.K=K
        self.V=dpre.words_count
        self.alpha=alpha
        self.beta=beta
        self.max_iter=max_iter
        self.seed=seed
        
        self.dt=np.zeros([self.dpre.docs_count,self.K],dtype=int)   #docs*topics
        self.tw=np.zeros([self.K,self.V],dtype=int)  #topics*words
        self.dtsum=self.dt.sum(axis=1)    
        self.twsum=self.tw.sum(axis=1)
        self.Z_assignment=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #topic assignment for each word for each doc
        
        #output var:
        self.theta=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])
        self.phi=np.array([[0.0 for y in range(self.V)] for x in range(self.K)])      

    @jit    
    def initializeModel(self):
        np.random.seed(self.seed)
        for m in range(self.dpre.docs_count):
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                k=np.random.randint(0,self.K) 
                self.dt[m,k]+=1
                self.dtsum[m]+=1
                self.tw[k,self.dpre.docs[m][n]]+=1 
                self.twsum[k]+=1
                self.Z_assignment[m][n]=k
        
    @jit
    def inferenceModel(self):   
        #Gibbs sampling over burn-in period and sampling period
        self.initializeModel()
        
        cur_iter=0
        while cur_iter<=self.max_iter:
            for m in range(self.dpre.docs_count):
                for n in range(len(self.dpre.docs[m])):   #n is word's index
                    self.sample(m,n)
            #print(cur_iter)
            cur_iter+=1
        
        self.updateEstimatedParameters()
 
    @jit
    def sample(self,m,n):
        topic=self.Z_assignment[m][n]
        word=self.dpre.docs[m][n]
                    
        self.dt[m,topic]-=1
        self.dtsum[m]-=1
        self.tw[topic,word]-=1
        self.twsum[topic]-=1

        p=(self.tw[:,word]+self.beta)/(self.twsum+self.V*self.beta)\
           *(self.dt[m,:]+self.alpha)/(self.dtsum[m]+self.K*self.alpha)
           
        topic=np.random.multinomial(1,p/p.sum()).argmax()

        self.dt[m,topic]+=1
        self.dtsum[m]+=1
        self.tw[topic,word]+=1
        self.twsum[topic]+=1
        self.Z_assignment[m][n]=topic 
           
        
    @jit
    def updateEstimatedParameters(self):
        for m in range(self.dpre.docs_count):
            self.theta[m]=(self.dt[m]+self.alpha)/(self.dtsum[m]+self.alpha*self.K)
        for k in range(self.K):
            self.phi[k]=(self.tw[k]+self.beta)/(self.twsum[k]+self.beta*self.V)

    def print_tw(self,topN=10):
        topics=[]
        for k in range(self.K):
            topic=[]
            index=self.phi[k].argsort()[::-1][:topN]
            for ix in index:
                prob=("%.3f"%self.phi[k,ix])
                word=self.dpre.id2word[ix]
                topic.append((prob,word))
            topics.append(topic)
        return topics
    
    def print_dt(self):
        return self.theta
            
def test():
    corpus=[['apple','orange','banana'],            
            ['banana','orange'],
            ['cat','dog'],
            ['dog','tiger'],
            ['tiger','cat'],
            ['computer','AI','network'],
            ['computer','IR','algorithm'],
            ['computer','IR','algorithm','network'],
            ['apple','orange'],
            ['algorithm','process','ML'],
            ['dog','tiger']]
    dpre=preprocessing(corpus)
    
    K=3
    model=LDAModel(dpre,K)
    model.inferenceModel()  
    
    
if __name__=='__main__':
     test()

