import itertools
import pandas as pd
from src.utils.enums import MovieLensDataset
from src.als_model import SparkAlsModel

def evaluate_als(spark):
    # Definindo os grids de parâmetros
    max_iter_options = [20]
    rank_options = [10, 20, 30, 40]
    reg_param_options = [0.05]
    alpha_options = [0.1]

    # Parâmetros fixos
    validate_size = 0.25
    top_k = 10
    seed = 42
    dataset = MovieLensDataset.ML_1M

    # Cria um DataFrame vazio para armazenar os resultados
    df_results = pd.DataFrame()

    # Itera por todas as combinações de parâmetros
    for max_iter, rank, reg_param, alpha in itertools.product(
            max_iter_options, rank_options, reg_param_options, alpha_options
    ):
        print(f"\nTestando: maxIter={max_iter}, rank={rank}, regParam={reg_param}, alpha={alpha}")

        # Instanciando o modelo ALS com a configuração atual
        als_model = SparkAlsModel(
            spark=spark,
            dataset=dataset,
            max_iter=max_iter,
            rank=rank,
            reg_param=reg_param,
            alpha=alpha,
            validate_size=validate_size,
            top_k=top_k,
            seed=seed
        )

        # Avaliando o modelo
        result = als_model.evaluate()

        # Adiciona as informações desejadas
        result['algorithm'] = 'als'
        result['version'] = f"max_iter_{max_iter}_rank_{rank}_reg_param_{reg_param}_alpha_{alpha}"

        df_results = pd.concat([df_results, result], ignore_index=True)
        print(f"Resultado: {result}")


    return df_results

if __name__ == '__main__':
    from src.utils.spark_session_utils import create_spark_session

    spark = create_spark_session("ALS")
    als_result = evaluate_als(spark)
    print(als_result)