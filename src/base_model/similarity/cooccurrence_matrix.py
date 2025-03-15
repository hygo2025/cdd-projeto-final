from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.utils import defaults as d

def cooccurrence_matrix(df: DataFrame, threshold: int) -> DataFrame:
    """
    Calcula a matriz de co-ocorrência a partir do DataFrame de interações.

    Args:
        df (DataFrame): DataFrame contendo, ao menos, as colunas definidas em col_user e col_item.
        threshold (int): Valor mínimo de co-ocorrência para manter o par.
        col_user (str): Nome da coluna que representa o usuário.
        col_item (str): Nome da coluna que representa o item.

    Returns:
        DataFrame: com as colunas "i1", "i2" e "co_count", mantendo apenas os pares com co_count >= threshold.
    """
    col_user, col_item, col_result = d.idf_user, d.idf_item, d.idf_cooccurrence

    # Seleciona interações únicas
    interactions = df.select(col_user, col_item).distinct()

    # Realiza self-join para obter pares de itens por usuário usando as colunas informadas
    joined = interactions.alias("a").join(
        interactions.alias("b"), on=[col_user]
    ).filter(F.col("a." + col_item) != F.col("b." + col_item)) \
        .select(F.col("a." + col_item).alias("i1"), F.col("b." + col_item).alias("i2"))

    # Agrupa os pares e conta as co-ocorrências
    co_occurrence_df = joined.groupBy("i1", "i2").agg(F.count("*").alias(col_result))

    # Filtra os pares com contagem abaixo do threshold
    co_occurrence_df = co_occurrence_df.filter(F.col(col_result) >= threshold)

    return co_occurrence_df
