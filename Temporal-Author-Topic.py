#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#from scipy.special import gamma
from collections import OrderedDict
import pandas as pd

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


class Temporal_Author_Topic_Model(object):
    """
    Author Topic Model
    implementation of `Exploiting Temporal Authors Interests via Temporal-Author-Topic Modeling` by Ali Daud, et al. (2009)
    """
    def __init__(self,dpre,K,alpha=0.1,beta=0.01,gam=0.1,max_iter=100,seed=1):
        #initial var
        self.dpre=dpre
        self.K=K
        self.alpha=alpha
        self.beta=beta
        self.gam=gam
        self.max_iter=max_iter
        self.seed=seed
        
        at=np.zeros([self.dpre.authors_count,self.K],dtype=int)   #authors*topics
        tw=np.zeros([self.K,self.dpre.words_count],dtype=int)  #topics*words
        ty=np.zeros([self.K,self.dpre.years_count],dtype=int)   #topics*years
        
        atsum=at.sum(axis=1)    
        twsum=tw.sum(axis=1)
        tysum=ty.sum(axis=1)
        
        Z=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #topic assignment for each word for each doc
        A=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #author assignment for each word for each doc
        #Y=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])   #year assigment for each word for each doc
        
        #initialization
        np.random.seed(self.seed)
        for m in range(self.dpre.docs_count):
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                #选主题
                k=np.random.multinomial(1,[1/self.K]*self.K).argmax()
                
                #选作者
                p_a=np.array([0.0 for x in range(self.dpre.authors_count)])  
                for x in self.dpre.authors[m]:  #保证了这篇文章对应的单词只能出自这篇文章的作者
                    p_a[x]=1/len(self.dpre.authors[m])  
                a=np.random.multinomial(1,p_a).argmax()
                          
                at[a,k]+=1
                atsum[a]+=1
                tw[k,self.dpre.docs[m][n]]+=1
                twsum[k]+=1
                ty[k,self.dpre.years[m]]+=1
                tysum[k]+=1
                
                Z[m][n]=k
                A[m][n]=a
                #Y[m][n]=
        
        #output var:
        self.theta=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.authors_count)])
        self.phi=np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])      
        self.psi=np.array([[0.0 for y in range(self.dpre.years_count)] for x in range(self.K)])

        #Gibbs sampling over burn-in period and sampling period
        cur_iter=0
        
        while cur_iter<=self.max_iter:
            for m in range(self.dpre.docs_count):
                for n in range(len(self.dpre.docs[m])):   #n is word's index
                    topic=Z[m][n]
                    author=A[m][n]
                    word=self.dpre.docs[m][n]
                    
                    at[author,topic]-=1
                    atsum[author]-=1
                    tw[topic,word]-=1
                    twsum[topic]-=1
                    ty[topic,self.dpre.years[m]]-=1
                    tysum[topic]-=1

                    p=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.authors_count)])
                    for a in self.dpre.authors[m]:   #!
                        for k in range(self.K):
                            p[a,k]=(tw[k,word]+self.beta)/(twsum[k]+self.dpre.words_count*self.beta) \
                                    *(at[a,k]+self.alpha)/(atsum[a]+self.K*self.alpha) \
                                    *(ty[k,self.dpre.years[m]]+self.gam)/(tysum[k]+self.dpre.years_count*self.gam)
                    #print(p)
                    p=p.reshape(self.dpre.authors_count*self.K)
                    index=np.random.multinomial(1,p/p.sum()).argmax()
                    author=int(index/self.K)
                    topic=index%self.K
                    
                    at[author,topic]+=1
                    atsum[author]+=1
                    tw[topic,word]+=1
                    twsum[topic]+=1
                    ty[topic,self.dpre.years[m]]+=1
                    tysum[topic]+=1
                    Z[m][n]=topic     
                    A[m][n]=author 
            #print(cur_iter)
            cur_iter+=1            
        #output
        for a in range(self.dpre.authors_count):
            self.theta[a]=(at[a]+self.alpha)/(atsum[a]+self.alpha*self.K)
        for k in range(self.K):
            self.phi[k]=(tw[k]+self.beta)/(twsum[k]+self.beta*self.dpre.words_count)
        for k in range(self.K):
            self.psi[k]=(ty[k]+self.gam)/(tysum[k]+self.gam*self.dpre.years_count)
    
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
       

if __name__=='__main__':
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

