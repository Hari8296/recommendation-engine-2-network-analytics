Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd

Gaming=pd.read_csv("D:/assignments of data science/11 recommendation engine 2 & network analytics/recommendation engine/game.csv")

Gaming1 = Gaming.drop_duplicates(subset='game', keep='first', inplace=False)
Gaming1.shape 
Gaming1.columns
Gaming1.game 

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(stop_words = "english")

Gaming1["game"].isnull().sum() 

tfidf_matrix = tfidf.fit_transform(Gaming1.game) 
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

Gaming_index = pd.Series(Gaming1.index, index = Gaming1['game']).drop_duplicates()

Gaming_id = Gaming_index["Grand Theft Auto IV"]
Gaming_id

def get_recommendations(Name, topN):    
    Gaming_id = Gaming_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[Gaming_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    Gaming_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    ET_similar_show = pd.DataFrame(columns=["name", "Score"])
    ET_similar_show["name"] = Gaming1.loc[Gaming_idx, "game"]
    ET_similar_show["Score"] = anime_scores
    ET_similar_show.reset_index(inplace = True)  
    print (ET_similar_show)

get_recommendations("Super Mario Galaxy 2", topN = 5)
Gaming_index["Super Mario Galaxy 2"]