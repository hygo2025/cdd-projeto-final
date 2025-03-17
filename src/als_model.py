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
                 reg_param: float, alpha:float, validate_size: float, top_k: int, seed: int):
        print("Inicializando SparkAlsModel...")
        super().__init__(
            dataset=dataset,
            model_name=f"movielens_als_model_rank_{rank}_maxiter_{max_iter}.model",
            spark=spark
        )
        self.model = None
        self.spark = spark
        self.dataset = dataset
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.top_k = top_k
        self.alpha = alpha
        self.seed = seed

        print("Carregando e preparando os dados com prepare_data_spark...")
        self.df = self.prepare_data_spark([d.idf_user, d.idf_item, d.idf_rating, d.idf_title])
        print(f"Dados carregados: {self.df.count()} linhas.")

        print("Dividindo os dados em conjuntos de treino e teste...")
        self.train_df, self.test_df = self.df.randomSplit([1 - validate_size, validate_size], seed=self.seed)
        print(f"Conjunto de treino: {self.train_df.count()} linhas; Conjunto de teste: {self.test_df.count()} linhas.")

        print("Filtrando o conjunto de teste para manter apenas usuários presentes no treino...")
        train_users = [row[d.idf_user] for row in self.train_df.select(d.idf_user).distinct().collect()]
        self.test_df = self.test_df.filter(col(d.idf_user).isin(train_users))
        print(f"Após filtragem, teste possui: {self.test_df.count()} linhas.\n")

        self.model_name = f"movielens_als_model_rank_{self.rank}_maxiter_{self.max_iter}.model"

    def train(self):
        print("Iniciando treinamento do modelo ALS...")
        als = ALS(
            maxIter=self.max_iter,
            rank=self.rank,
            regParam=self.reg_param,
            alpha=self.alpha,
            userCol=d.idf_user,
            itemCol=d.idf_item,
            ratingCol=d.idf_rating,
            coldStartStrategy="drop",
            seed=self.seed
        )
        self.model = als.fit(self.train_df)
        print("Treinamento concluído. Salvando o modelo...")
        return self.model

    def predict(self) -> DataFrame:
        print("Iniciando predição com o modelo ALS...")
        model = self.train()
        print("Obtendo usuários e itens distintos do conjunto de treino...")
        users = self.train_df.select(d.idf_user).distinct()
        items = self.train_df.select(d.idf_item).distinct()
        print("Realizando cross join entre usuários e itens...")
        user_item = users.crossJoin(items)
        print("Gerando predições utilizando o modelo ALS...")
        dfs_pred = model.transform(user_item)

        print("Removendo itens já vistos (presentes no treino)...")
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            self.train_df.alias("train"),
            (col("pred." + d.idf_user) == col("train." + d.idf_user)) &
            (col("pred." + d.idf_item) == col("train." + d.idf_item)),
            how='outer'
        )
        top_all = dfs_pred_exclude_train.filter(
            dfs_pred_exclude_train[f"train.{d.idf_rating}"].isNull()
        ).select('pred.' + d.idf_user, 'pred.' + d.idf_item, 'pred.' + d.idf_prediction)
        print("Predição concluída.\n")
        return top_all

    def evaluate(self):
        print("Iniciando avaliação do modelo ALS...")
        print("Gerando predições para avaliação...")
        predictions_df = self.predict()
        print("Calculando métricas de avaliação de ranking...")
        ranking_metrics = self.spark_ranking_metrics(self.test_df, self.top_k, predictions_df)
        print("Calculando métricas de avaliação de rating...")
        rating_metrics = self.spark_rating_metrics(self.test_df, predictions_df)
        print("Avaliação concluída.\n")
        return pd.concat([ranking_metrics, rating_metrics], axis=1)

    def save(self, model: ALSModel):
        model_path = self.get_path('als')
        model.write().overwrite().save(model_path)
        print(f"Modelo salvo em: {model_path}\n")

    def load(self) -> ALSModel:
        model_path = self.get_path('als')
        print(f"Carregando modelo do caminho: {model_path}...")
        model = ALSModel.load(model_path)
        print("Modelo carregado com sucesso.\n")
        return model

if __name__ == '__main__':
    print("Criando sessão Spark...")
    spark = create_spark_session("ALS")
    print("Sessão Spark criada.\n")

    print("Inicializando SparkAlsModel...")
    model = SparkAlsModel(
        spark=spark,
        dataset=MovieLensDataset.ML_100K,
        rank=10,
        max_iter=20,
        reg_param=0.05,
        alpha=0.1,
        validate_size=0.25,
        top_k=10,
        seed=42
    )
    print("Treinando modelo ALS...")
    model.train()
    print("Avaliando modelo ALS...")
    result = model.evaluate()
    print("Resultados da avaliação:")
    print(result)
    spark.stop()
    gc.collect()
