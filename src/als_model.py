import os
import gc

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS, ALSModel
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation

from src.dataset.movielens.loader import Loader
from src.utils.enums import MovieLensDataset, MovieLensType
from src.utils import defaults as d
from src.utils.spark_session_utils import create_spark_session


class SparkAlsModel:
    def __init__(self, spark: SparkSession, dataset: MovieLensDataset, rank: int, max_iter: int,
                 reg_param: float, test_size: float, top_k: int, seed: int):
        self.spark = spark
        self.dataset = dataset
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.test_size = test_size
        self.top_k = top_k
        self.seed = seed

        # Prepara os dados: carrega ratings e movies e faz o merge
        self.df = self._prepare_data()
        # Divide os dados em treino e teste
        self.train_df, self.test_df = self.df.randomSplit([1 - self.test_size, self.test_size], seed=self.seed)

        # Garante que os usuários do teste estejam presentes no treino
        train_users = [row[d.idf_user] for row in self.train_df.select(d.idf_user).distinct().collect()]
        self.test_df = self.test_df.filter(col(d.idf_user).isin(train_users))

        self.model_name = f"movielens_als_model_rank_{self.rank}_maxiter_{self.max_iter}.model"


    def _prepare_data(self) -> DataFrame:
        loader = Loader(self.spark)
        df_ratings = loader.load_spark(dataset=self.dataset, ml_type=MovieLensType.RATINGS)
        df_movies = loader.load_spark(dataset=self.dataset, ml_type=MovieLensType.MOVIES)

        df = df_ratings.join(df_movies, on=d.idf_item, how='inner') \
            .select(d.idf_user, d.idf_item, d.idf_rating, d.idf_title)
        return df

    def train(self):
        als = ALS(
            maxIter=self.max_iter,
            rank=self.rank,
            regParam=self.reg_param,
            userCol=d.idf_user,
            itemCol=d.idf_item,
            ratingCol=d.idf_rating,
            coldStartStrategy="drop",
            seed=self.seed
        )
        self.model = als.fit(self.train_df)
        self.save(self.model)

    def predict(self) -> DataFrame:
        model = self.load()

        # Get the cross join of all user-item pairs and score them.
        users = self.train_df.select(d.idf_user).distinct()
        items = self.train_df.select(d.idf_item).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)

        # Remove seen items.
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            self.train_df.alias("train"),
            (dfs_pred[d.idf_user] == self.train_df[d.idf_user]) & (dfs_pred[d.idf_item] == self.train_df[d.idf_item]),
            how='outer'
        )

        top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{d.idf_rating}"].isNull()) \
            .select('pred.' + d.idf_user, 'pred.' + d.idf_item, 'pred.' + d.idf_prediction)


        return top_all

    def evaluate(self):
        # Gera as predições para avaliação de ranking
        predictions_df = self.predict()

        # Avaliação de ranking
        rank_eval = SparkRankingEvaluation(
            self.train_df, predictions_df, k=self.top_k,
            col_user=d.idf_user, col_item=d.idf_item,
            col_rating=d.idf_rating, col_prediction=d.idf_prediction,
            relevancy_method="top_k"
        )

        #TODO: Chegar pq essas metricas estao vazias
        eval_map = rank_eval.map_at_k()
        eval_ndcg = rank_eval.ndcg_at_k()
        eval_precision = rank_eval.precision_at_k()
        eval_recall = rank_eval.recall_at_k()

        # Gera predições para avaliação de ratings usando o modelo treinado
        prediction_ratings = self.model.transform(self.train_df)
        prediction_ratings.cache().show()  # Para visualização, se necessário

        rating_eval = SparkRatingEvaluation(
            self.train_df, prediction_ratings,
            col_user=d.idf_user, col_item=d.idf_item,
            col_rating=d.idf_rating, col_prediction=d.idf_prediction
        )
        eval_rmse = rating_eval.rmse()
        eval_mae = rating_eval.mae()
        eval_exp_var = rating_eval.exp_var()
        eval_r2 = rating_eval.rsquared()

        # Monta o dicionário de resultados como um DataFrame
        results_dict = {
            "Metric": ["MAP", "nDCG@K", "Precision@K", "Recall@K", "R2", "RMSE", "MAE", "Explained Variance"],
            "Value": [eval_map, eval_ndcg, eval_precision, eval_recall, eval_r2, eval_rmse, eval_mae, eval_exp_var]
        }
        results_df = pd.DataFrame(results_dict)
        return results_df


    def save(self, model: ALSModel):
        model_path = self._get_model_path()
        model.write().overwrite().save(model_path)
        print(f"Model saved to {model_path}")

    def load(self) -> ALSModel:
        model_path = self._get_model_path()
        return ALSModel.load(model_path)

    def _get_model_path(self):
        save_dir = os.path.join('..', 'data', 'als')
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, self.model_name)


if __name__ == '__main__':
    spark = create_spark_session("ALS")

    model = SparkAlsModel(
        spark=spark,
        dataset=MovieLensDataset.ML_100K,
        rank=10,
        max_iter=15,
        reg_param=0.05,
        test_size=0.25,
        top_k=10,
        seed=42
    )
    model.train()
    result = model.evaluate()
    print(result)
    spark.stop()
    gc.collect()