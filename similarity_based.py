import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

class OverviewBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.df = movies[['id', 'title', 'overview']]
        self.df.loc[:, 'overview'] = self.df['overview'].fillna(value='')
        tfidf = TfidfVectorizer()
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
                    stemmer.stem(kwrd).replace(" ", "").lower() for kwrd in literal_eval(x)
                ])
            )
        cntvec = CountVectorizer()
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


class GenreBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.df = movies[['id', 'genres']]
        self.df.loc[:, 'genres'] = self.df['genres'].apply(literal_eval) 

        mlb = MultiLabelBinarizer()
        mlb_genres = mlb.fit_transform(self.df['genres'])

        self.df = self.df.assign(**{
            f"genre-{genre.lower()}": mlb_genres[:, i] 
            for i, genre in enumerate(mlb.classes_)
        })
        self.df = self.df.drop(columns=['genres'])
        self.train_ratings = train_ratings

    def rank(self, user:int, test_ids: np.ndarray) -> np.ndarray:
        train_movies = self.train_ratings[self.train_ratings['userId'] == user]
        train_movie_ratings = train_movies['rating']
        train_movie_ids = train_movies['movieId']
        train_movie_indices = self.df[self.df['id'].isin(train_movie_ids)].index
        train_movies_genres = self.df.filter(regex='genre').to_numpy()[train_movie_indices]

        test_movie_indices = self.df[self.df['id'].isin(test_ids)].index
        test_movies_genres = self.df.filter(regex='genre').to_numpy()[test_movie_indices]

        sim = cosine_similarity(train_movies_genres, test_movies_genres)
        adjusted_sim = train_movie_ratings.to_numpy().reshape(-1, 1) * sim

        return np.sum(adjusted_sim, axis=0)


class CastBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        stemmer = SnowballStemmer('english')

        self.df = movies[['id', 'title', 'cast']]
        self.df.loc[:, 'cast'] = self.df['cast'] \
            .apply(lambda x: " ".join(literal_eval(x))
            )
        cntvec = CountVectorizer()
        self.cntvec_matrix = cntvec.fit_transform(self.df['cast'])
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


class CrewBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        stemmer = SnowballStemmer('english')

        self.df = movies[['id', 'title', 'crew']]
        self.df.loc[:, 'crew'] = self.df['crew'] \
            .apply(lambda x: " ".join(literal_eval(x))
            )
        cntvec = CountVectorizer()
        self.cntvec_matrix = cntvec.fit_transform(self.df['crew'])
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
    

class SoupBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        stemmer = SnowballStemmer('english')

        self.df = movies[['id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew']]
        self.df.loc[:, 'keywords'] = self.df['keywords'] \
            .apply(lambda x: " ".join([
                    stemmer.stem(kwrd).replace(" ", "").lower() for kwrd in literal_eval(x)
                ])
            )
        self.df.loc[:, 'genres'] = self.df['genres'] \
            .apply(lambda x: " ".join(literal_eval(x)))
        self.df.loc[:, 'cast'] = self.df['cast'] \
            .apply(lambda x: " ".join(literal_eval(x)))
        self.df.loc[:, 'crew'] = self.df['crew'] \
            .apply(lambda x: " ".join(literal_eval(x)))

        self.df = self.df.assign(
            soup=self.df['keywords'] + \
                self.df['genres'] + ' ' + \
                self.df['cast'] + ' ' + \
                self.df['crew']
        )

        cntvec = CountVectorizer()
        self.cntvec_matrix = cntvec.fit_transform(self.df['soup'])
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