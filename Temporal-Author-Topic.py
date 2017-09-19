#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#from scipy.special import gamma
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from numba import jit
import pickle

class DataPreProcessing(object):
    def __init__(self):
        self.docs_count=0
        self.words_count=0
        self.authors_count=0
        self.years_count=0
        
        self.docs=[]
        self.authors=[]
        self.years=[]
        
        self.word2id=OrderedDict()
        self.id2word=OrderedDict()
        self.author2id=OrderedDict()
        self.id2author=OrderedDict()
        self.year2id=OrderedDict()

              
def preprocessing(corpus,authors,years):
    if len(corpus)==len(authors) and len(corpus)==len(years):
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
        year_index=0
        for year in sorted(set(years),reverse=False):
            dpre.year2id[year]=year_index
            year_index+=1
            
        for year in years:  
            dpre.years.append(dpre.year2id[year])
            
        dpre.docs_count=len(dpre.docs)
        dpre.words_count=len(dpre.word2id)
        dpre.authors_count=author_index
        dpre.years_count=len(set(years))
        
        dpre.id2word={v:k for k,v in dpre.word2id.items()}
        dpre.id2author={v:k for k,v in dpre.author2id.items()}
        
        return dpre
    else:
        print('errors occur: different length!')
    


class Temporal_Author_Topic_Model(object):
    """
    Temporal Author Topic Model
    implementation of `Exploiting Temporal Authors Interests via Temporal-Author-Topic Modeling` by Ali Daud, et al. (2009)
    """
    def __init__(self,dpre,K,alpha=0.1,beta=0.01,gam=0.1,max_iter=100,seed=1):
        self.dpre=dpre
        self.A=dpre.authors_count #number of authors
        self.K=K #number of topics
        self.V=dpre.words_count #number of words
        self.Y=dpre.years_count #number of timestamps
        
        self.alpha=alpha
        self.beta=beta
        self.gam=gam
        self.max_iter=max_iter
        self.seed=seed
        
        self.at=np.zeros([self.A,self.K],dtype=int)  #authors*topics 
        self.tw=np.zeros([self.K,self.V],dtype=int)  #topics*words
        self.ty=np.zeros([self.K,self.Y],dtype=int)   #topics*years
        
        self.atsum=self.at.sum(axis=1)    
        self.twsum=self.tw.sum(axis=1)
        self.tysum=self.ty.sum(axis=1)
        
        #topic assignment for each word for each doc
        self.Z_assigment=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])
        #author assignment for each word for each doc
        self.A_assigment=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    
        
        #output var
        self.theta=np.zeros([self.A,self.K],dtype=float)
        self.phi=np.zeros([self.K,self.V],dtype=float)
        self.psi=np.zeros([self.K,self.Y],dtype=float)
    
    
    @jit    
    def initializeModel(self):
        print('init start:',datetime.now())
        np.random.seed(self.seed) 
        for m in range(self.dpre.docs_count):
            year=self.dpre.years[m]
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                #选主题
                #k=np.random.multinomial(1,[1/self.K]*self.K).argmax() 与下面等价
                k=np.random.randint(low=0,high=self.K)
                
                #选作者
                if len(self.dpre.authors[m])==1:    #这篇文章只有一个作者，那就是TA
                    a=self.dpre.authors[m][0]
                else:   #若有多个作者，随机选择一个
                    idx=np.random.randint(low=0,high=len(self.dpre.authors[m]))
                    a=self.dpre.authors[m][idx]
                """
                p_a=np.array([0.0 for x in range(self.A)])  
                for x in self.dpre.authors[m]:  #保证了这篇文章对应的单词只能出自这篇文章的作者
                    p_a[x]=1/len(self.dpre.authors[m])  
                a=np.random.multinomial(1,p_a).argmax()
                """          
                self.at[a,k]+=1
                self.atsum[a]+=1
                self.tw[k,self.dpre.docs[m][n]]+=1
                self.twsum[k]+=1
                self.ty[k,year]+=1
                self.tysum[k]+=1
                
                self.Z_assigment[m][n]=k
                self.A_assigment[m][n]=a
        print('init finish:',datetime.now())
    
    
    
    @jit
    def inferenceModel(self):    
        #Gibbs sampling over burn-in period and sampling period
        self.initializeModel()
        
        print('inference start:',datetime.now())
        
        cur_iter=0        
        while cur_iter<=self.max_iter:
            #i=0
            for m in range(self.dpre.docs_count):
                N=len(self.dpre.docs[m])
                for n in range(N):   #n is word's index
                    self.sample(m,n)
                #print(i)
                #i+=1
            print(cur_iter,datetime.now())
            cur_iter+=1     
            
        print('inference finish:',datetime.now())
        
        self.updateEstimatedParameters()    
    
    
    
    @jit
    def sample(self,m,n):
        old_topic=self.Z_assigment[m][n]
        old_author=self.A_assigment[m][n]
        word=self.dpre.docs[m][n]
        year=self.dpre.years[m]
        authors_set=self.dpre.authors[m] #第m篇文章的作者集合
                    
        self.at[old_author,old_topic]-=1
        self.atsum[old_author]-=1
        self.tw[old_topic,word]-=1
        self.twsum[old_topic]-=1
        self.ty[old_topic,year]-=1
        self.tysum[old_topic]-=1

        section1=(self.tw[:,word]+self.beta)/(self.twsum+self.V*self.beta)
        section2=(self.at[authors_set,:]+self.alpha)/(self.atsum[authors_set].repeat(self.K).reshape(len(authors_set),self.K)+self.K*self.alpha)
        section3=(self.ty[:,year]+self.gam)/(self.tysum+self.Y*self.gam)
        p=section1*section2*section3
        p=p.reshape(len(authors_set)*self.K)
        index=np.random.multinomial(1,p/p.sum()).argmax()
        new_author=authors_set[int(index/self.K)]
        new_topic=index%self.K
        """
        #use loop, so slowly
        
        p=np.zeros([self.A,self.K],dtype=float)
        for a in self.dpre.authors[m]:   #!
            for k in range(self.K):
                p[a,k]=(self.tw[k,word]+self.beta)/(self.twsum[k]+self.V*self.beta) \
                    *(self.at[a,k]+self.alpha)/(self.atsum[a]+self.K*self.alpha) \
                    *(self.ty[k,year]+self.gam)/(self.tysum[k]+self.Y*self.gam)
        p=p.reshape(self.A*self.K)
        index=np.random.multinomial(1,p/p.sum()).argmax()
        new_author=int(index/self.K)
        new_topic=index%self.K
        """         
           
        self.at[new_author,new_topic]+=1
        self.atsum[new_author]+=1
        self.tw[new_topic,word]+=1
        self.twsum[new_topic]+=1
        self.ty[new_topic,year]+=1
        self.tysum[new_topic]+=1
        self.Z_assigment[m][n]=new_topic     
        self.A_assigment[m][n]=new_author    

    @jit
    def updateEstimatedParameters(self):
        for a in range(self.A):
            self.theta[a]=(self.at[a]+self.alpha)/(self.atsum[a]+self.alpha*self.K)
        for k in range(self.K):
            self.phi[k]=(self.tw[k]+self.beta)/(self.twsum[k]+self.beta*self.V)
        for k in range(self.K):
            self.psi[k]=(self.ty[k]+self.gam)/(self.tysum[k]+self.gam*self.Y)
                    
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
    
    def print_at(self):
        return self.theta
    
    def print_ty(self):
        return self.psi
       
def test():
    corpus=[['computer','medical','DM','algorithm','drug'],
            ['computer','AI','DM','algorithm'],
            ['art','beauty','architectural','architector'],
            ['drug','medical','hospital'],
            ['computer','AI','SVM','socialnetwork'],
            ['art','beauty','architectural','building'],
            ['architectural','building','architector']]
    authors=[['Tom','Amy'],['Tom'],['Tom'],['Amy'],['Tom'],['Tom'],['Tom']]
    years=[2004,2004,2005,2008,2004,2005,2005]
    dpre=preprocessing(corpus,authors,years)
    K=3
    model=Temporal_Author_Topic_Model(dpre,K,max_iter=100)
    model.inferenceModel()
    a=model.print_at()
    b=model.print_tw()
    c=model.print_ty()


if __name__=='__main__':
    test()
