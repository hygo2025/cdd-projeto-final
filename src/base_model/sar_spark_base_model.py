from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.connect.session import SparkSession

from src.base_model.similarity.cooccurrence_matrix import cooccurrence_matrix
from src.base_model.similarity.cosine_similarity import cosine_similarity
from src.utils.enums import SimilarityType
from pyspark.sql.window import Window
from src.utils import defaults as d


class SarSparkBaseModel:
    def __init__(self,
                 spark: SparkSession,
                 rating_threshold=0,
                 top_k=10,
                 threshold=1,
                 similarity_type=SimilarityType.COSINE
                 ):
        self.spark = spark
        self.rating_threshold = rating_threshold
        self.top_k = top_k
        self.threshold = threshold
        self.similarity_type = similarity_type

        # Atributos gerados durante o fit
        self.user_affinity = None  # Matriz de afinidade usuário-item (interações filtradas)
        self.item_frequencies = None  # Frequência (popularidade) de cada item
        self.item_similarity = None  # Matriz de similaridade entre itens

    def fit(self, interactions: DataFrame):
        print("Building user affinity sparse matrix")
        temp_df = interactions.filter(F.col(d.idf_rating) >= self.rating_threshold)
        temp_df = temp_df.select(d.idf_user, d.idf_item, d.idf_rating).distinct()
        self.user_affinity = temp_df.select(d.idf_user, d.idf_item, d.idf_rating)

        self.item_frequencies = self.user_affinity.groupBy(d.idf_item).agg(F.count(d.idf_user).alias("item_count"))

        print("Calculating item similarity")
        if self.similarity_type == SimilarityType.COCCURRENCE:
            print("Using co-occurrence based similarity")
            co_occurrence = cooccurrence_matrix(df=temp_df, threshold=self.threshold)
            self.item_similarity = co_occurrence
        elif self.similarity_type == SimilarityType.COSINE:
            print("Using cosine similarity")
            self.item_similarity = cosine_similarity(
                df=self.user_affinity,
            )

        del temp_df
        print("Done training")

    def recommend(self, user_id) -> DataFrame:
        user_items_df = self.user_affinity.filter(F.col(d.idf_user) == user_id).select(d.idf_item).distinct()
        user_items = [row[d.idf_item] for row in user_items_df.collect()]

        # Filtra a matriz de similaridade para os pares onde o item consumido aparece como "i1"
        sim_df = self.item_similarity.filter(F.col("i1").isin(user_items))
        recs = sim_df.groupBy("i2").agg(F.sum(self.get_target_column()).alias("score"))
        recs = recs.filter(~F.col("i2").isin(user_items))
        recommendations = recs.orderBy(F.col("score").desc()).limit(self.top_k)
        return recommendations

    def get_target_column(self):
        if self.similarity_type == SimilarityType.COCCURRENCE:
            return d.idf_cosine
        elif self.similarity_type == SimilarityType.COSINE:
            return d.idf_cosine
        else:
            return None

    def recommend_k_items(self, test: DataFrame, sort_top_k: bool = True, remove_seen: bool = False) -> DataFrame:
        """
        Recomenda os top K itens para cada usuário presente no DataFrame de teste,
        retornando um DataFrame com as colunas [d.idf_user, d.idf_prediction],
        onde d.idf_prediction é uma lista de itens recomendados (ordenados por score).

        Args:
            test (DataFrame): DataFrame de teste contendo a coluna d.idf_user.
            sort_top_k (bool): Flag para ordenar os top K resultados.
            remove_seen (bool): Flag para remover itens que o usuário já viu (do treinamento).

        Returns:
            DataFrame: com as colunas [d.idf_user, d.idf_prediction]
        """
        top_k = self.top_k
        # Seleciona os usuários de teste (distinct)
        test_users = test.select(d.idf_user).distinct()

        # Alias para a afinidade do usuário (treino) e para a matriz de similaridade
        ua = self.user_affinity.select(d.idf_user, d.idf_item).distinct().alias("ua")
        sim = self.item_similarity.alias("sim")

        # Junta os usuários de teste com os itens com os quais já interagiram (do treino)
        user_items = test_users.join(ua, on=d.idf_user, how="inner")

        # Junta com a matriz de similaridade: une o item que o usuário consumiu (ua[d.idf_item])
        # com o item candidato (sim["i1"])
        joined = user_items.join(sim, user_items[d.idf_item] == sim["i1"], how="inner")

        # Agrega as similaridades para obter a pontuação (prediction) para cada (usuário, item candidato)
        scores = joined.groupBy(
            user_items[d.idf_user].alias(d.idf_user),
            sim["i2"].alias(d.idf_item)
        ).agg(F.sum(F.col(d.idf_cosine)).alias("prediction"))

        # Se remove_seen=True, remove os itens que o usuário já viu no treinamento
        if remove_seen:
            seen = self.user_affinity.select(d.idf_user, d.idf_item).distinct()
            scores = scores.join(seen, on=[d.idf_user, d.idf_item], how="left_anti")

        # Define uma janela para particionar por usuário e ordenar os itens por score decrescente
        window_spec = Window.partitionBy(d.idf_user).orderBy(F.col("prediction").desc())
        scores = scores.withColumn("rn", F.row_number().over(window_spec)) \
            .filter(F.col("rn") <= top_k) \
            .drop("rn")

        if sort_top_k:
            scores = scores.orderBy(d.idf_user, F.col("prediction").desc())

        # Agrupa por usuário e coleta os itens recomendados em uma lista, igual ao ALS
        top_k_items = scores.groupBy(d.idf_user) \
            .agg(F.collect_list(d.idf_item).alias(d.idf_prediction))

        return top_k_items
