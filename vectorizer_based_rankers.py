import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer

class OverviewBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.df = movies[['id', 'title', 'overview']]
        self.df.loc[:, 'overview'] = self.df['overview'].fillna(value='')
        tfidf = TfidfVectorizer(stop_words='english', strip_accents='unicode')
        self.tfidf_matrix = tfidf.fit_transform(self.df['overview'])
        self.train_ratings = train_ratings

    def rank(self, user:int, test_ids: np.ndarray) -> np.ndarray:
        train_movies = self.train_ratings[self.train_ratings['userId'] == user]
        train_movie_ratings = train_movies['rating']
        train_movie_ids = train_movies['movieId']
        train_movie_indices = self.df[self.df['id'].isin(train_movie_ids)].index
        train_movies_tfidf = self.tfidf_matrix[train_movie_indices]

        test_movie_indices = self.df[self.df['id'].isin(test_ids)].index
        test_movies_tfidf = self.tfidf_matrix[test_movie_indices]

        sim = cosine_similarity(train_movies_tfidf, test_movies_tfidf)
        adjusted_sim = train_movie_ratings.to_numpy().reshape(-1, 1) * sim

        return np.sum(adjusted_sim, axis=0)

class KeywordsBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        stemmer = SnowballStemmer('english')

        self.df = movies[['id', 'title', 'keywords']]
        self.df.loc[:, 'keywords'] = self.df['keywords'] \
            .apply(lambda x: " ".join([
                stemmer.stem(kwrd).replace(" ", "").lower() for kwrd in x
                ])
            )
        cntvec = CountVectorizer(stop_words='english', strip_accents='unicode')
        self.cntvec_matrix = cntvec.fit_transform(self.df['keywords'])
        self.train_ratings = train_ratings

    def rank(self, user:int, test_ids: np.ndarray) -> np.ndarray:
        train_movies = self.train_ratings[self.train_ratings['userId'] == user]
        train_movie_ratings = train_movies['rating']
        train_movie_ids = train_movies['movieId']
        train_movie_indices = self.df[self.df['id'].isin(train_movie_ids)].index
        train_movies_tfidf = self.cntvec_matrix[train_movie_indices]

        test_movie_indices = self.df[self.df['id'].isin(test_ids)].index
        test_movies_tfidf = self.cntvec_matrix[test_movie_indices]

        sim = cosine_similarity(train_movies_tfidf, test_movies_tfidf)
        adjusted_sim = train_movie_ratings.to_numpy().reshape(-1, 1) * sim

        return np.sum(adjusted_sim, axis=0)