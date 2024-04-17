import pandas as pd
import numpy as np

class VoteAverageBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.movies = movies
        self.train_ratings = train_ratings

    def rank(self, user: int, query_movie_ids: np.ndarray, **kwargs) -> np.ndarray:
        return self.movies[self.movies['id'].isin(query_movie_ids)]['vote_average']
    
class WeightedVoteBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.movies = movies
        v = self.movies['vote_count']
        r = self.movies['vote_average']
        m = v.quantile(0.9)
        c = r.mean()
        self.movies['weighted_vote'] = (v/(v+m) * r) + (m/(v+m) * c)
        self.train_ratings = train_ratings

    def rank(self, user: int, query_movie_ids: np.ndarray, **kwargs) -> np.ndarray:
        return self.movies[self.movies['id'].isin(query_movie_ids)]['weighted_vote']