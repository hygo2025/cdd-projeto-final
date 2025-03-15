import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.utils import defaults as d


def cosine_similarity(df: DataFrame, n_partitions: int = 200) -> DataFrame:
    """
    Calcula a similaridade do cosseno baseada na contagem de co-ocorrência dos itens:

        similarity = count(i1,i2) / (sqrt(count(i1)) * sqrt(count(i2)))

    Os pares são obtidos com i1 <= i2 e o resultado é reparticionado para melhorar a performance.
    """

    col_user, col_item, col_result = d.idf_user, d.idf_item, d.idf_cosine

    # Obtém os pares de itens por usuário
    pairs = (
        df.select(col_user, F.col(col_item).alias("i1"))
        .join(
            df.select(F.col(col_user).alias("_user"), F.col(col_item).alias("i2")),
            (F.col(col_user) == F.col("_user")) & (F.col("i1") <= F.col("i2"))
        )
        .select(col_user, "i1", "i2")
    )

    # Conta a ocorrência de cada item
    item_count = df.groupBy(col_item).count()

    # Calcula a co-ocorrência: número de usuários que consumiram o par (i1, i2)
    cooccurrence = pairs.groupBy("i1", "i2").count()

    # Calcula a raiz quadrada das contagens para cada item
    item_count_i = item_count.select(F.col(col_item).alias("i1"),
                                     F.pow(F.col("count"), 0.5).alias("i1_sqrt_count"))

    item_count_j = item_count.select(F.col(col_item).alias("i2"),
                                     F.pow(F.col("count"), 0.5).alias("i2_sqrt_count"))

    # Junta os DataFrames para calcular a similaridade do cosseno
    similarity_df = (
        cooccurrence.join(item_count_i, on="i1")
        .join(item_count_j, on="i2")
        .select(
            "i1",
            "i2",
            (F.col("count") / (F.col("i1_sqrt_count") * F.col("i2_sqrt_count"))).alias(col_result)
        )
        .repartition(n_partitions, "i1", "i2")
    )

    return similarity_df