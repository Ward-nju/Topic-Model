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
    def __init__(self,dpre,K,alpha=0.1,beta=0.01,max_iter=100,seed=1,converge_criteria=0.001):
        #initial var
        self.dpre=dpre
        self.K=K
        self.alpha=alpha
        self.beta=beta
        self.max_iter=max_iter
        self.seed=seed
        self.converge_criteria=converge_criteria
        
        at=np.zeros([self.dpre.authors_count,self.K],dtype=int)   #authors*topics
        tw=np.zeros([self.K,self.dpre.words_count],dtype=int)  #topics*words
        atsum=at.sum(axis=1)    
        twsum=tw.sum(axis=1)
        Z=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #topic assignment for each word for each doc
        A=np.array([[0 for y in range(len(self.dpre.docs[x]))] for x in range(self.dpre.docs_count)])    #author assignment for each word for each doc
        
        #initialization
        np.random.seed(self.seed)
        for m in range(self.dpre.docs_count):
            for n in range(len(self.dpre.docs[m])):   #n is word's index
                k=np.random.multinomial(1,[1/self.K]*self.K).argmax()
                
                p_a=np.array([0.0 for x in range(self.dpre.authors_count)])  #!
                for x in self.dpre.authors[m]:
                    p_a[x]=1/len(self.dpre.authors[m])
                a=np.random.multinomial(1,p_a).argmax()
                
                at[a,k]+=1
                atsum[a]+=1
                tw[k,self.dpre.docs[m][n]]+=1
                twsum[k]+=1
                Z[m][n]=k
                A[m][n]=a
        
        #output var:
        self.theta=np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.authors_count)])
        self.phi=np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])      

        #Gibbs sampling over burn-in period and sampling period
        converge=False
        cur_iter=0
        
        while not converge and (cur_iter<=self.max_iter):
            for m in range(self.dpre.docs_count):
                for n in range(len(self.dpre.docs[m])):   #n is word's index
                    topic=Z[m][n]
                    author=A[m][n]
                    word=self.dpre.docs[m][n]
                    
                    at[author,topic]-=1
                    atsum[author]-=1
                    tw[topic,word]-=1
                    twsum[topic]-=1

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
                    
                    at[author,topic]+=1
                    atsum[author]+=1
                    tw[topic,word]+=1
                    twsum[topic]+=1
                    Z[m][n]=topic     
                    A[m][n]=author 
            #print(cur_iter)
            cur_iter+=1            
        #output
        for a in range(self.dpre.authors_count):
            self.theta[a]=(at[a]+self.alpha)/(atsum[a]+self.alpha*self.K)
        for k in range(self.K):
            self.phi[k]=(tw[k]+self.beta)/(twsum[k]+self.beta*self.dpre.words_count)
    
    def print_topics(self,topN=10):
        for k in range(self.K):
            s=''
            index=self.phi[k].argsort()[::-1][:topN]
            for ix in index:
                prob=("%.3f"%self.phi[k,ix])
                word=self.dpre.id2word[ix]
                s+=str(prob)+'*'+word+' + '
            print('topic'+str(k)+':  '+s)
            
    def perplexity(self,dpre_test=None):
        #when split the corpus into traing\test set: the authors in testing set should be in trainging set,too.
        if dpre_test==None:
            dpre_test=self.dpre
        N=0
        p=0.0
        for m in range(dpre_test.docs_count):   
            p_d=1.0
            for n in range(len(dpre_test.docs[m])):
                authors=dpre_test.authors[m]  #author ids in testing set
                word=dpre_test.id2word[dpre_test.docs[m][n]]    #word
                
                if word in self.dpre.word2id.keys(): #training set has "word"
                    w_id=self.dpre.word2id[word]  #"word" id in training set
                    p_w=0.0  #probablity for single word
                    for a in authors:
                        a_id=self.dpre.author2id[dpre_test.id2author[a]]    #author id in training set
                        p_w+=np.dot(self.theta[a_id,:],self.phi[:,w_id])
                    p_w=p_w/len(authors)  #avg probabily for "word" given by the set of "authors"
                    p_d*=p_w  #probablity for documents
                else:  #no way to caculate the probablity for unseen word
                    pass
            p+=np.log(p_d)
            N+=len(dpre_test.docs[m])
        perplexity=np.exp(-p/N)
        return perplexity
    
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

