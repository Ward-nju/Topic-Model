#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma
from collections import OrderedDict
import pandas as pd
from numba import jit
from datetime import datetime


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count=0
        self.words_count=0
        self.authors_count=0
        self.docs=[]
        self.authors=[]
        self.word2id=OrderedDict()
        self.id2word=OrderedDict()
        self.author2id=OrderedDict()
        self.id2author=OrderedDict()
        
def preprocessing(corpus,authors):
    if len(corpus)!=len(authors):
        print('errors occur:corpus and authors have different length!')
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
        author_index=0
        for author in authors:
            alist=[]
            for a in author:
                if a in dpre.author2id.keys():
                    alist.append(dpre.author2id[a])
                else:
                    dpre.author2id[a]=author_index
                    alist.append(author_index)
                    author_index+=1
            dpre.authors.append(alist)    
            
        dpre.docs_count=len(dpre.docs)
        dpre.words_count=len(dpre.word2id)
        dpre.authors_count=author_index
        dpre.id2word={v:k for k,v in dpre.word2id.items()}
        dpre.id2author={v:k for k,v in dpre.author2id.items()}
        return dpre


class ATM(object):
    """
    Author Topic Model
    implementation of `The Author-Topic Model for Authors and Documents` by Rosen-Zvi, et al. (2004)
    """
    def __init__(self,dpre,K,alpha=0.1,beta=0.01,max_iter=100,seed=1):
        #initial var
        self.dpre=dpre
        self.A=dpre.authors_count
        self.K=K
        self.V=dpre.words_count
        self.alpha=alpha
        self.beta=beta
        self.max_iter=max_iter
        self.seed=seed
        
        self.at=np.zeros([self.A,self.K],dtype=int)   #authors*topics
        self.tw=np.zeros([self.K,self.V],dtype=int)  #topics*words
        self.atsum=self.at.sum(axis=1)    
        self.twsum=self.tw.sum(axis=1)
        
        self.Z_assigment=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #topic assignment for each word for each doc
        self.A_assigment=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #author assignment for each word for each doc
        
        #output var:
        self.theta=np.array([[0.0 for y in range(self.K)] for x in range(self.A)])
        self.phi=np.array([[0.0 for y in range(self.V)] for x in range(self.K)])      

    @jit    
    def initializeModel(self):
        #initialization
        print('init start:',datetime.now())
        np.random.seed(self.seed)

        for m in range(self.dpre.docs_count):
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                #选主题
                #k=np.random.multinomial(1,[1/self.K]*self.K).argmax()
                k=np.random.randint(low=0,high=self.K)
                
                #选作者
                if len(self.dpre.authors[m])==1:    #这篇文章只有一个作者，那就是TA
                    a=self.dpre.authors[m][0]
                else:   #若有多个作者，随机选择一个
                    idx=np.random.randint(low=0,high=len(self.dpre.authors[m]))
                    a=self.dpre.authors[m][idx]
                    
                self.at[a,k]+=1
                self.atsum[a]+=1
                self.tw[k,self.dpre.docs[m][n]]+=1
                self.twsum[k]+=1
                self.Z_assigment[m][n]=k
                self.A_assigment[m][n]=a
                
        print('init finish:',datetime.now())
        
        
    @jit
    def inferenceModel(self): 
        self.initializeModel()
        
        print('inference start:',datetime.now())
        
        cur_iter=0        
        while cur_iter<=self.max_iter:
            for m in range(self.dpre.docs_count):
                for n in range(len(self.dpre.docs[m])):   #n is word's index
                    self.sample(m,n)
            print(cur_iter,datetime.now())
            cur_iter+=1            
        
        print('inference finish:',datetime.now())
        
        self.updateEstimatedParameters()
        
        

    @jit
    def sample(self,m,n):
        old_topic=self.Z_assigment[m][n]
        old_author=self.A_assigment[m][n]
        word=self.dpre.docs[m][n]
        authors_set=self.dpre.authors[m]
            
        self.at[old_author,old_topic]-=1
        self.atsum[old_author]-=1
        self.tw[old_topic,word]-=1
        self.twsum[old_topic]-=1

        section1=(self.tw[:,word]+self.beta)/(self.twsum+self.V*self.beta)
        section2=(self.at[authors_set,:]+self.alpha)/(self.atsum[authors_set].repeat(self.K).reshape(len(authors_set),self.K)+self.K*self.alpha)
        p=section1*section2
        
        p=p.reshape(len(authors_set)*self.K)
        index=np.random.multinomial(1,p/p.sum()).argmax()
        
        new_author=authors_set[int(index/self.K)]
        new_topic=index%self.K
        """
        p=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.authors_count)])
        for a in self.dpre.authors[m]:   #!
            for k in range(self.K):
                p[a,k]=(tw[k,word]+self.beta)/(twsum[k]+self.dpre.words_count*self.beta) \
                        *(at[a,k]+self.alpha)/(atsum[a]+self.K*self.alpha)
                    #print(p)
        p=p.reshape(self.dpre.authors_count*self.K)
        index=np.random.multinomial(1,p/p.sum()).argmax()
        author=int(index/self.K)
        topic=index%self.K
        """            
        self.at[new_author,new_topic]+=1
        self.atsum[new_author]+=1
        self.tw[new_topic,word]+=1
        self.twsum[new_topic]+=1
        self.Z_assigment[m][n]=new_topic     
        self.A_assigment[m][n]=new_author 
    
    @jit
    def updateEstimatedParameters(self):
        for a in range(self.A):
            self.theta[a]=(self.at[a]+self.alpha)/(self.atsum[a]+self.alpha*self.K)
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
            
            
    
    def symmetric_KL_divergence(self,i,j):
        #caculate symmetric KL divergence between author i and j
        #i,j: author name or author id
        if type(i)!=int or type(j)!=int:
            i=self.dpre.author2id[i]
            j=self.dpre.author2id[j]
        sKL=0
        for k in range(self.K):
            sKL+=self.theta[i,k]*np.log(self.theta[i,k]/self.theta[j,k]) \
                +self.theta[j,k]*np.log(self.theta[j,k]/self.theta[i,k])
        return sKL


if __name__=='__main__':
    corpus=[['computer','medical','DM','algorithm','drug'],
            ['computer','AI','DM','algorithm'],
            ['art','beauty','architectural'],
            ['drug','medical','hospital']]
    authors=[['Tom','Amy'],['Tom'],['Ward'],['Amy']]
    dpre=preprocessing(corpus,authors)
    K=3
    model=ATM(dpre,K,max_iter=100)
    model.inferenceModel()

