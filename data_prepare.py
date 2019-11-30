import numpy as np 
from gensim.models import FastText

def cos_sim(vector1, vector2):  
    cos=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return cos

def count_vector(model, words): 
    vect_list = []
    vec = np.zeros(500) 
    for w in words:
       
        try:    
           vec = vec + model.wv[w]
        except:
            continue
    return vec


def sum_of_each_day_vector(news_public_day , df_vector):
    Each_date_vector = []
    Each_date =[]
    for i in range(len(news_public_day)):
        if i==0:
            vector_sum = np.zeros(500)
            vector_sum = vector_sum+df_vector[i]
        elif news_public_day[i] == news_public_day[i-1]:
            vector_sum = vector_sum+df_vector[i]
        elif news_public_day[i] != news_public_day[i-1]:
            Each_date.append(news_public_day[i-1])
            Each_date_vector.append(vector_sum)
            vector_sum = np.zeros(500)
            vector_sum = vector_sum+df_vector[i]
        elif news_public_day[i] != news_public_day[i-1] and news_public_day[i] != news_public_day[i-2] :
            print('Unknown Error')
            exit()
    return Each_date_vector,Each_date