# Topic-Model 
#### 2017.06.13
第一次用Github，一边学习一边更新，其实网上关于这部分的代码很多，但是自己写能加深认识和理解。  
目前完成了  
LDA: `Latent Dirichlet Allocation` by David M.Blei, et al. (2003)  
Labeled-LDA: `Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora` by Ramage D, et al. (2009)  
Author-Topic-Model: `The Author-Topic Model for Authors and Documents` by Rosen-Zvi, et al. (2004)  
都是采用Gibbs Sampling实现。

接下来的任务：</br>
1.代码里只设置了最大迭代次数，如何判断马尔科夫链的收敛，计算公式？</br>
2.主题数确定：perplexity p(w|T)如何计算</br>
3.推断</br>


#### 2017.09.16
好久没看主题模型这方面的了，根据Boss要求，更新了一个主题模型：  
`Exploiting Temporal Authors Interests via Temporal-Author-Topic Modeling` by Ali Daud, et al. (2009)
虽然代码可读性有点小瑕疵，不过经过试验，代码能完成基本的任务，暂时先这样吧~

#### 2017.09.19
去掉了Temporal-Author-Topic Modeling中的一个循环，并采用numba的@jit加速，效率有显著提升。  
其他的准备也据此修改。
