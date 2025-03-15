from abc import ABC
import os
from typing import List

import pandas as pd
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var

from pyspark.sql import SparkSession, DataFrame
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from src.dataset.movielens.loader import Loader
from src.utils import defaults as d
from src.utils.enums import MovieLensDataset, MovieLensType


class AbstractModel(ABC):
    def __init__(self, model_name, dataset: MovieLensDataset, spark: SparkSession = None):
        self.model_name = model_name
        self.dataset = dataset
        self.spark = spark

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save_to_tmp_file(self, df: pd.DataFrame, save_dir:str, name: str):
        file_location = self.get_path(save_dir=save_dir, data_path='data/tmp', name=name)
        df.to_csv(self.get_path(save_dir=save_dir, data_path = 'data/tmp', name=name), index=False)
        return file_location

    def prepare_data_pandas(self, columns: List[str]) -> pd.DataFrame:
        loader = Loader()
        df_ratings = loader.load_pandas(dataset=self.dataset, ml_type=MovieLensType.RATINGS)
        df_movies = loader.load_pandas(dataset=self.dataset, ml_type=MovieLensType.MOVIES)
        return df_ratings.merge(df_movies, on=d.idf_item)[columns]

    def prepare_data_spark(self, columns: List[str]) -> DataFrame:
        loader = Loader(self.spark)
        df_ratings = loader.load_spark(dataset=self.dataset, ml_type=MovieLensType.RATINGS)
        df_movies = loader.load_spark(dataset=self.dataset, ml_type=MovieLensType.MOVIES)

        df = df_ratings.join(df_movies, on=d.idf_item, how='inner') \
            .select(columns)
        return df

    def get_path(self, save_dir, data_path: str = 'data', name: str = None):
        save_dir = os.path.join('..', data_path, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        if name:
            return os.path.join(save_dir, name)
        return os.path.join(save_dir, self.model_name)

    def map_at_k(self, test_df, top_k, predictions_df):
        return map_at_k(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item,
                        col_rating=d.idf_rating, col_prediction=d.idf_prediction,
                        relevancy_method="top_k", k=top_k)

    def ndcg_at_k(self, test_df, top_k, predictions_df):
        return ndcg_at_k(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item,
                         col_rating=d.idf_rating, col_prediction=d.idf_prediction,
                         relevancy_method="top_k", k=top_k)

    def precision_at_k(self, test_df, top_k, predictions_df):
        return precision_at_k(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item,
                              col_rating=d.idf_rating, col_prediction=d.idf_prediction,
                              relevancy_method="top_k", k=top_k)

    def recall_at_k(self, test_df, top_k, predictions_df):
        return recall_at_k(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item,
                           col_rating=d.idf_rating, col_prediction=d.idf_prediction,
                           relevancy_method="top_k", k=top_k)

    def rsquared(self, test_df, predictions_df):
        return rsquared(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item, col_rating=d.idf_rating,
                        col_prediction=d.idf_prediction)

    def rmse(self, test_df, predictions_df):
        return rmse(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item, col_rating=d.idf_rating,
                    col_prediction=d.idf_prediction)

    def mae(self, test_df, predictions_df):
        return mae(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item, col_rating=d.idf_rating,
                   col_prediction=d.idf_prediction)

    def exp_var(self, test_df, predictions_df):
        return exp_var(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item, col_rating=d.idf_rating,
                       col_prediction=d.idf_prediction)

    def at_k_metrics(self, test_df, top_k, predictions_df) -> pd.DataFrame:
        return pd.DataFrame({
            d.idf_map: [self.map_at_k(test_df, top_k, predictions_df)],
            d.idf_ndcg: [self.ndcg_at_k(test_df, top_k, predictions_df)],
            d.idf_precision: [self.precision_at_k(test_df, top_k, predictions_df)],
            d.idf_recall: [self.recall_at_k(test_df, top_k, predictions_df)]
        })

    def metrics(self, test_df, predictions_df) -> pd.DataFrame:
        return pd.DataFrame({
            d.idf_r2: [self.rmse(test_df, predictions_df)],
            d.idf_rmse: [self.mae(test_df, predictions_df)],
            d.idf_mae: [self.exp_var(test_df, predictions_df)],
            d.idf_exp_var: [self.rsquared(test_df, predictions_df)]
        })

    def spark_ranking_evaluation(self, test_df, top_k, predictions_df):
        return SparkRankingEvaluation(test_df, predictions_df, k=top_k, col_user=d.idf_user, col_item=d.idf_item,
                                      col_rating=d.idf_rating, col_prediction=d.idf_prediction,
                                      relevancy_method="top_k")

    def spark_rating_evaluation(self, test_df, predictions_df):
        return SparkRatingEvaluation(test_df, predictions_df, col_user=d.idf_user, col_item=d.idf_item,
                                     col_rating=d.idf_rating, col_prediction=d.idf_prediction)

    def spark_ranking_metrics(self, test_df, top_k, predictions_df) -> pd.DataFrame:
        ranking_eval = self.spark_ranking_evaluation(test_df, top_k, predictions_df)
        return pd.DataFrame({
            d.idf_map: [ranking_eval.map_at_k()],
            d.idf_ndcg: [ranking_eval.ndcg_at_k()],
            d.idf_precision: [ranking_eval.precision_at_k()],
            d.idf_recall: [ranking_eval.recall_at_k()]
        })

    def spark_rating_metrics(self, test_df, predictions_df) -> pd.DataFrame:
        rating_eval = self.spark_rating_evaluation(test_df, predictions_df)
        return pd.DataFrame({
            d.idf_r2: [rating_eval.rmse()],
            d.idf_rmse: [rating_eval.mae()],
            d.idf_mae: [rating_eval.exp_var()],
            d.idf_exp_var: [rating_eval.rsquared()]
        })
