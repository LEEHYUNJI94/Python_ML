import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

'''
1.분석에 필요한 컬럼만 선언
2.merge시 join 키값이 되도록 컬럼명(movieId) 변경
'''
#movie data
movie_df=pd.read_csv("movies_metadata.csv")
movie_df = movie_df[['id', 'original_title', 'original_language', 'genres']]
movie_df=movie_df.rename(columns={'id':'movieId'})
movie_df=movie_df[movie_df["original_language"]=="en"]

# rate data
rate_df=pd.read_csv("ratings_small.csv")
rate_df=rate_df[['userId', 'movieId', 'rating']]


movie_df['movieId'] =pd.to_numeric(movie_df['movieId']  , errors="coerce")
rate_df['movieId']  =pd.to_numeric(rate_df['movieId'] , errors="coerce")


'''
1.json.loads :  json 디코딩 (문자열을 python타입으로 변경)
2.genres 컬럼 중 name만 append
3.apply : pandas객체 혹은 행,열에 대해 함수 적용
apply(func, axis=0, raw=False...)  
'''
def json_parse(genre_str):
    genre_loads=json.loads(genre_str.replace('\'','"'))

    genres_list=[]
    for genre in genre_loads:
        genres_list.append(genre['name'])
    return genres_list

movie_df['genres']=movie_df['genres'].apply(json_parse)



# merge movie_df + rate_df
merge_dfs=pd.merge(movie_df,rate_df,on='movieId',how='inner')

pivot_table=merge_dfs.pivot_table(index='userId',columns='original_title',values='rating')

'''
Pearson
가중치 0.1
'''

GENRE_WEIGHT=0.1

def PearsonR(s1,s2):
    s1_d=s1 - s1.mean()
    s2_d=s2 - s2.mean()
    return np.sum(s1_d*s2_d)/np.sqrt(np.sum(s1_d**2)*np.sum(s2_d**2))

def recommend(input_movie,pivot_table,n,similar_genre=True):
    result=[]

    # input_moive와 같다면 skip  
    for title in pivot_table.columns:
        if title == input_movie:
            continue
    
        cor = PearsonR(pivot_table[input_movie],pivot_table[title])
        input_genres=movie_df[movie_df['original_title']==input_movie]['genres'].iloc[0]

    # input_genre와 비교
        if similar_genre and len(input_genres) > 0:
            temp_genres=movie_df[movie_df['original_title']==title]['genres'].iloc[0]
            same_count=np.sum(np.isin(input_genres,temp_genres))
            cor += (GENRE_WEIGHT * same_count )

        if np.isnan(cor):
            continue
        else:
            result.append((title,'{:.2f}'.format(cor),temp_genres))
    #cor 역정렬
    result.sort(key=lambda r:r[1], reverse=True)

    return result[:n]

#추천 알고리즘 TEST
use_recommend=recommend('Zardoz',pivot_table,10,similar_genre=True)
pd.DataFrame(use_recommend, columns=['Title','Correlation','Genre'])
