Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd

Entertain=pd.read_csv("D:/assignments of data science/11 recommendation engine 2 & network analytics/recommendation engine/Entertainment.csv")

Entertain.shape 
Entertain.columns
Entertain.Category

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")

Entertain["Category"].isnull().sum() 
Entertain["Category"] = Entertain["Category"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(Entertain.Category)
tfidf_matrix.shape 

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

Entertain_index = pd.Series(Entertain.index, index = Entertain['Titles']).drop_duplicates()

Entertain_id = Entertain_index["Sabrina (1995)"]
Entertain_id

def get_recommendations(Name, topN):    
    Entertain_id = Entertain_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[Entertain_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    Entertain_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    ET_similar_show = pd.DataFrame(columns=["name", "Score"])
    ET_similar_show["name"] = Entertain.loc[Entertain_idx, "Titles"]
    ET_similar_show["Score"] = anime_scores
    ET_similar_show.reset_index(inplace = True)  
    print (ET_similar_show)
   

get_recommendations("Babe (1995)", topN = 10)
Entertain_index["Babe (1995)"]
