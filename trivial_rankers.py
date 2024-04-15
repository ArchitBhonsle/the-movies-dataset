import pandas as pd
import numpy as np

class VoteAverageBasedRanker():
    def __init__(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        self.movies = movies
        self.train_ratings = train_ratings

    def rank(self, user: int, query_movie_ids: np.ndarray) -> np.ndarray:
        return self.movies[self.movies['id'].isin(query_movie_ids)]['vote_average']