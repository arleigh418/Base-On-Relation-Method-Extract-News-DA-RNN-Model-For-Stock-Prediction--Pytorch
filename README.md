# Base-On-Relation-Method-Extract-News-DA-RNN-Model-For-Stock-Prediction


Run model.py.


# Update
### 2020/5/6
1.How I deal with article vector, you may can follow this relation work :https://github.com/arleigh418/How-Much-News-Should-We-Extract-For-Stock-Price-Prediction/tree/master/Stage1_Replace%20Company%20Name%20Train%20Embedding

2.How to use ta package and add each article vector you may can refer : https://github.com/arleigh418/How-Much-News-Should-We-Extract-For-Stock-Price-Prediction/tree/master/Stage2_2Count%20TA%20%26%20Merge%20Stock%20Price%20And%20Article

### 2019/11/30
### Add data_prepare.py
1.As somebody need, I provide three function to show how we count vector, cos and getting sum of each day vector.If cos similar is not achieve your target(e.g. cos>0.7), then we also use top 30% to get more similar article with target center article, like below code:
```
np.percentile({article use} , {per}, interpolation='midpoint')
```
You can even use 
```
np.percentile({article use} , {per}, interpolation='linear')
```
2.Not only add each day vector to present one day news imformation vector, we also get average to present one day vector to test(the excel file we provided is avg method). If you are interested, you can try it by yourself.

(I will provide avg method, I can't find avg method code now QQ)

3.I highly suggest you to clean each article(news) by Stopword or other method.

# Reference
1.This porject is referenced this paper:
#### A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction (DARNN)_ Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell, A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction, IJCAI, 2017.

2.This model modify from here:  https://github.com/Zhenye-Na/DA-RNN

3.We use FASTTEXT to deal with news data ,the method that training fasttext and count relation between news(Cosine) you can reference this   : https://github.com/arleigh418/Word-Embedding-With-Gensim   (Cosine method is in doc2vec_count_cos.py)

--> tips:We add each word's vector from article to present one article vector .

4.We use the 'ta' package to count technical analysis in our data,please reference here: https://github.com/bukosabino/ta

5.Stock price come from : https://finance.yahoo.com/quote/2330.TW?p=2330.TW

# How we do it ? 
Please check Description.pdf.

# Others

1.GOODONE.pkl is the model we train.
2.We know that there are still much left for improvement,we are trying to make this model better. If you have any question , please contact me for free.

