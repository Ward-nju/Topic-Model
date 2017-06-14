#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma
from collections import OrderedDict
import pandas as pd

class DataPreProcessing(object):
    def __init__(self):
        self.docs_count=0
        self.words_count=0
        self.labels_count=0
        self.docs=[]
        self.labels=[]
        self.word2id=OrderedDict()
        self.id2word=OrderedDict()
        self.label2id=OrderedDict()
        self.id2label=OrderedDict()
        
def preprocessing(corpus,labels):
    """
    corpus: type list, like \n
            [['red','apple','banana','yellow','blue'],
             ['apple','iphone','computer',],
             ['banana','orange','pear'],
             ['apple','intel','ms']] 
    labels: type list, like [['fruit','color'],['tech'],['fruit'],['tech']] same length with corpus
    return: DataPreProcessing object
    """
    if len(corpus)!=len(labels):
        print('errors occur:corpus and labels have different length!')
    else:
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
        label_index=0
        for label in labels:
            llist=[]
            for l in label:
                if l in dpre.label2id.keys():
                    llist.append(dpre.label2id[l])
                else:
                    dpre.label2id[l]=label_index
                    llist.append(label_index)
                    label_index+=1
            dpre.labels.append(llist)    
            
        dpre.docs_count=len(dpre.docs)
        dpre.words_count=len(dpre.word2id)
        dpre.labels_count=label_index
        dpre.id2word={v:k for k,v in dpre.word2id.items()}
        dpre.id2label={v:k for k,v in dpre.label2id.items()}
        return dpre


class LabeledLDAModel(object):
    """
    Labeled LDA
    implementation of `Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora` by Ramage D, et al. (2009)
    """
    def __init__(self,dpre,alpha=0.1,beta=0.01,max_iter=100,seed=1,converge_criteria=0.001):
        #initial var
        self.dpre=dpre
        self.K=dpre.labels_count
        self.alpha=alpha
        self.beta=beta
        self.max_iter=max_iter
        self.seed=seed
        self.converge_criteria=converge_criteria
        
        dt=np.zeros([self.dpre.docs_count,self.K],dtype=int)   #docs*topics
        tw=np.zeros([self.K,self.dpre.words_count],dtype=int)  #topics*words
        dtsum=dt.sum(axis=1)    
        twsum=tw.sum(axis=1)
        Z=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #topic assignment for each word for each doc
        
        #initialization
        np.random.seed(self.seed)
        for m in range(self.dpre.docs_count):
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                k=np.random.randint(0,self.K)
                dt[m,k]+=1
                dtsum[m]+=1
                tw[k,self.dpre.docs[m][n]]+=1
                twsum[k]+=1
                Z[m][n]=k
        
        #output var:
        self.theta=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])
        self.phi=np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])      

                  
        #Gibbs sampling over burn-in period and sampling period
        converge=False
        cur_iter=0
        
        while not converge and (cur_iter<=self.max_iter):
            for m in range(self.dpre.docs_count):
                labels=self.dpre.labels[m]  #read doc[m]'s labels
                for n in range(len(self.dpre.docs[m])):   #n is word's index
                    topic=Z[m][n]
                    word=self.dpre.docs[m][n]
                    
                    dt[m,topic]-=1
                    dtsum[m]-=1
                    tw[topic,word]-=1
                    twsum[topic]-=1

                    #在这里进行了限制，只有labels对应的主题有概率，其余为0
                    p=np.array([0.0 for x in range(self.K)])
                    for k in labels:
                        p[k]=(tw[k,word]+self.beta)/(twsum[k]+self.dpre.words_count*self.beta)*(dt[m,k]+self.alpha)/(dtsum[m]+len(labels)*self.alpha) 
                    topic=np.random.multinomial(1,p/p.sum()).argmax()
                    
                    dt[m,topic]+=1
                    dtsum[m]+=1
                    tw[topic,word]+=1
                    twsum[topic]+=1
                    Z[m][n]=topic    
            #print(cur_iter)
            cur_iter+=1
        #output
        for m in range(self.dpre.docs_count):
            self.theta[m]=(dt[m]+self.alpha)/(dtsum[m]+self.alpha*self.K)
        for k in range(self.K):
            self.phi[k]=(tw[k]+self.beta)/(twsum[k]+self.beta*self.dpre.words_count)
    
    def print_topics(self,topN=10):
        for k in range(self.K):
            s=''
            ids=self.phi[k].argsort()[::-1][:topN]
            for id in ids:
                prob=("%.3f"%self.phi[k,id])
                word=self.dpre.id2word[id]
                s+=str(prob)+'*'+word+' + '
            print(self.dpre.id2label[k]+':  '+s)
    
                
if __name__=='__main__':
    corpus=[['red','apple','banana','yellow','blue'],
            ['apple','iphone','computer',],
            ['banana','orange','pear'],
            ['apple','intel','ms']]
    labels=[['fruit','color'],['tech'],['fruit'],['tech']]
    dpre=preprocessing(corpus,labels)
    K=dpre.labels_count
    model=LabeledLDAModel(dpre)      
