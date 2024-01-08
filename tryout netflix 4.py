import pandas as pd #to read a particular data set
import numpy as np #to create an array

## import dataset to python
netflix = pd.read_csv("/Users/zahirahrahim/Downloads/netflix.csv")
print(netflix.head(10))

## delete the variables that we don't need because we will only use the show_id
## title, type and description column
netflix_cleaned_df= netflix.drop(columns=['director','cast','country','date_added',
                                         'release_year','rating','duration'])
print(netflix_cleaned_df.head(6))

## to see the description of the new data frame
netflix_cleaned_df.info()

### ~~ CONTENT BASED RECOMMENDATION ~~
## this project is focusing on giving recommendations based on the content in the 
## show's plot given in the description column. 
## so the shows that will appear in the 'more like this' section 
## will have similar content  the plot summaries with the movie

netflix_cleaned_df.head(1)['description']

## when we want to create a recoomendation engine, for each movie we have to create a 
## a vector of matrix because while applying recommendation system that usually 
##based on pairwise similarity.

##tfidf is atechnic to create a document matrix from the sentences in description colum
from sklearn.feature_extraction.text import TfidfVectorizer

## data cleaning
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                      analyzer='word',token_pattern=r'\w{1,}', ##remove unnecessary words and symbols
                      ngram_range=(1,3), #take the combintion of 1-3 different words
                      stop_words='english') ## remove unnecessary words like the etc

## fillyng NA with empty srings for the description columns
netflix_cleaned_df['description'] = netflix_cleaned_df['description'].fillna('')

## creating the vector
## tfv_matrix is a sparse matrix because most of the values in the ,atrix =0
tfv_matrix= tfv.fit_transform(netflix_cleaned_df['description'])

tfv_matrix.shape
   
## finding simarity values
from sklearn.metrics.pairwise import sigmoid_kernel

sig=sigmoid_kernel(tfv_matrix,tfv_matrix)          

sig[0]

indices = pd.Series(netflix_cleaned_df.index, index=netflix_cleaned_df['title']).drop_duplicates()

def give_rec(title,sig=sig):
    
    # get the index corresponding to the original title
    idx= indices[title]
    
    # get the pairwise similiraty scores
    sig_scores= list(enumerate(sig[idx]))
    
    #sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    # Scores of the 5 most similar shows
    sig_scores= sig_scores[1:6]
    
    #movie indices
    movie_indices= [i[0] for i in sig_scores]
    
    # top 10 most similar movies
    return netflix_cleaned_df['title'].iloc[movie_indices]

give_rec('My Little Pony: A New Generation')

