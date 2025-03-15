import gc

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS, ALSModel

from src.abstract_model import AbstractModel
from src.utils.enums import MovieLensDataset
from src.utils import defaults as d
from src.utils.spark_session_utils import create_spark_session


class SparkAlsModel(AbstractModel):
    def __init__(self, spark: SparkSession, dataset: MovieLensDataset, rank: int, max_iter: int,
                 reg_param: float, validate_size: float, top_k: int, seed: int):
        super().__init__(
            dataset = dataset,
            model_name=f"movielens_als_model_rank_{rank}_maxiter_{max_iter}.model",
            spark = spark
        )

        self.model = None
        self.spark = spark
        self.dataset = dataset
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.top_k = top_k
        self.seed = seed

        # Prepara os dados: carrega ratings e movies e faz o merge
        self.df = self.prepare_data_spark([d.idf_user, d.idf_item, d.idf_rating, d.idf_title])

        # Divide os dados em treino e teste
        self.train_df, self.test_df = self.df.randomSplit([1 - validate_size, validate_size], seed=self.seed)

        # Garante que os usuários do teste estejam presentes no treino
        train_users = [row[d.idf_user] for row in self.train_df.select(d.idf_user).distinct().collect()]
        self.test_df = self.test_df.filter(col(d.idf_user).isin(train_users))

        self.model_name = f"movielens_als_model_rank_{self.rank}_maxiter_{self.max_iter}.model"

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

        ranking_metrics = self.spark_ranking_metrics(self.test_df, self.top_k, predictions_df)
        rating_metrics = self.spark_rating_metrics(self.test_df, predictions_df)

        return pd.concat([ranking_metrics, rating_metrics], axis=1)


    def save(self, model: ALSModel):
        model_path = self.get_model_path('als')
        model.write().overwrite().save(model_path)
        print(f"Model saved to {model_path}")

    def load(self) -> ALSModel:
        model_path = self.get_model_path('als')
        return ALSModel.load(model_path)

if __name__ == '__main__':
    spark = create_spark_session("ALS")

    model = SparkAlsModel(
        spark=spark,
        dataset=MovieLensDataset.ML_100K,
        rank=10,
        max_iter=15,
        reg_param=0.05,
        validate_size=0.25,
        top_k=10,
        seed=42
    )
    model.train()
    result = model.evaluate()
    print(result)
    spark.stop()
    gc.collect()