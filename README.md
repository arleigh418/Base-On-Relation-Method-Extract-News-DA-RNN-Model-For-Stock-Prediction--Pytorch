# Base-On-Word-Embedding-DA-RNN-Model-For-Stock-Prediction


### Run model.py 

This porject is referenced this paper:
#### A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction (DARNN)_ Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell, A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction, IJCAI, 2017.

#### How we do it? Please check Description.pdf

This model based on:  https://github.com/Zhenye-Na/DA-RNN

We use FASTTEXT to deal with news data ,the method that training fasttext and count relation between news(Cosine) you can reference this   : https://github.com/arleigh418/Word-Embedding-With-Gensim   (Cosine method is in doc2vec_count_cos.py)

We use the 'ta' package to count technical analysis in our data,please reference here: https://github.com/bukosabino/ta


We provide documentation to description how we do and our data come from. We know that there are still much left for improvement,we are trying to make this model better.

GOODONE.pkl is the model we train.
