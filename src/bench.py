import itertools
import pandas as pd

from src.sar_model import SarModel
from src.utils.enums import MovieLensDataset, SimilarityType
from src.utils.spark_session_utils import create_spark_session

from src.ncf_model import NcfModel
from src.bivae_model import BivaeModel
from src.light_gcn_model import LightGcnModel
from src.als_model import SparkAlsModel
from src.fastai_model import FastAiModel

def evaluate_fastai():
    # Parâmetros a serem testados
    n_factors_options = [40, 50]
    epochs_options = [1, 2]

    # Parâmetros fixos
    test_size = 0.25
    top_k = 10
    seed = 42
    dataset = MovieLensDataset.ML_1M

    # Cria um DataFrame vazio para armazenar os resultados
    df_results = pd.DataFrame()

    # Itera por todas as combinações de parâmetros
    for n_factors, epochs in itertools.product(n_factors_options, epochs_options):
        print(f"\nTestando: n_factors={n_factors}, epochs={epochs}")

        # Instancia o modelo FastAiModel com a configuração atual
        model = FastAiModel(
            dataset=dataset,
            n_factors=n_factors,
            test_size=test_size,
            epochs=epochs,
            top_k=top_k,
            seed=seed
        )

        # Avalia o modelo
        result = model.evaluate()

        # Adiciona informações de identificação
        result['algorithm'] = 'fastai'
        result['version'] = f"n_factors_{n_factors}_epochs_{epochs}"

        # Converte o dicionário de resultado em um DataFrame de uma linha e concatena
        df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)

        print(f"Resultado: {result}")

    return df_results

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

def evaluate_lightgcn():
    # Parâmetros a serem testados
    n_layers_options = [2, 3, 4]
    lr_options = [0.005, 0.01]
    epochs_options = [1, 5]

    # Parâmetros fixos
    batch_size = 1024
    top_k = 10
    test_size = 0.2
    seed = 42
    dataset = MovieLensDataset.ML_1M

    # Cria um DataFrame vazio para armazenar os resultados
    df_results = pd.DataFrame()

    # Itera por todas as combinações de parâmetros
    for n_layers, lr, epochs in itertools.product(n_layers_options, lr_options, epochs_options):
        print(f"\nTestando: n_layers={n_layers}, lr={lr}, epochs={epochs}")

        # Instancia o modelo LightGcnModel com a configuração atual
        model = LightGcnModel(
            dataset=dataset,
            n_layers=n_layers,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            top_k=top_k,
            test_size=test_size,
            seed=seed
        )

        # Avalia o modelo
        result = model.evaluate()

        # Adiciona informações de identificação
        result['algorithm'] = 'lightgcn'
        result['version'] = f"n_layers_{n_layers}_lr_{lr}_epochs_{epochs}"

        # Converte o dicionário de resultado em um DataFrame de uma linha e concatena
        df_results = pd.concat([df_results, result], ignore_index=True)

        print(f"Resultado: {result}")

    return df_results

def evaluate_bivae():
    # Parâmetros a serem testados
    latent_dim_options = [50, 100]
    epochs_options = [300, 500]
    lr_options = [0.001]

    # Parâmetros fixos
    top_k = 10
    batch_size = 1024
    encoder_dims = [100]
    act_func = 'tanh'
    likelihood = 'pois'
    test_size = 0.25
    seed = 42
    dataset = MovieLensDataset.ML_100K

    # Cria um DataFrame vazio para armazenar os resultados
    df_results = pd.DataFrame()

    # Itera por todas as combinações de parâmetros
    for latent_dim, epochs, lr in itertools.product(
        latent_dim_options, epochs_options, lr_options
    ):
        print(f"\nTestando: latent_dim={latent_dim}, epochs={epochs}, lr={lr}")

        # Instancia o modelo BivaeModel com a configuração atual
        model = BivaeModel(
            dataset=dataset,
            top_k=top_k,
            batch_size=batch_size,
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            act_func=act_func,
            likelihood=likelihood,
            epochs=epochs,
            lr=lr,
            test_size=test_size,
            seed=seed
        )

        # Avalia o modelo
        result = model.evaluate()

        # Adiciona informações de identificação
        result['algorithm'] = 'bivae'
        result['version'] = f"latent_dim_{latent_dim}_epochs_{epochs}_lr_{lr}"

        # Converte o dicionário de resultado em um DataFrame de uma linha e concatena
        df_results = pd.concat([df_results, result], ignore_index=True)

        print(f"Resultado: {result}")

    return df_results

def evaluate_ncf():
    # Parâmetros a serem testados
    n_factors_options = [4]
    batch_size_options = [256, 512]
    lr_options = [1e-3]
    epochs_options = [10, 15]

    # Parâmetros fixos
    layer_sizes = [16, 8, 4]
    top_k = 10
    test_size = 0.25
    seed = 42
    dataset = MovieLensDataset.ML_1M

    # Cria um DataFrame vazio para armazenar os resultados
    df_results = pd.DataFrame()

    # Itera por todas as combinações de parâmetros
    for n_factors, batch_size, lr, epochs in itertools.product(
        n_factors_options, batch_size_options, lr_options, epochs_options
    ):
        print(f"\nTestando: n_factors={n_factors}, batch_size={batch_size}, lr={lr}, epochs={epochs}")

        # Instancia o modelo NCF com a configuração atual
        model = NcfModel(
            dataset=dataset,
            n_factors=n_factors,
            batch_size=batch_size,
            lr=lr,
            layer_sizes=layer_sizes,
            epochs=epochs,
            top_k=top_k,
            test_size=test_size,
            seed=seed
        )

        # Avalia o modelo
        result = model.evaluate()

        # Adiciona informações de identificação
        result['algorithm'] = 'ncf'
        result['version'] = f"n_factors_{n_factors}_batch_size_{batch_size}_lr_{lr}_epochs_{epochs}"

        df_results = pd.concat([df_results, result], ignore_index=True)

        print(f"Resultado: {result}")

    return df_results

def evaluate_sar():
    top_k = 10
    validate_size = 0.25
    time_decay_coefficient = 30
    seed = 42
    df_results = pd.DataFrame()


    # Iterando sobre todas as similaridades definidas na enumeração
    for similarity in SimilarityType:
        print(f"\nTestando similaridade: {similarity.value}")

        # Instanciando o modelo com a similaridade atual
        sar_model = SarModel(
            dataset=MovieLensDataset.ML_1M,
            top_k=top_k,
            validate_size=validate_size,
            time_decay_coefficient=time_decay_coefficient,
            similarity_type=similarity,
            seed=seed
        )

        # Avaliando o modelo
        result = sar_model.evaluate()

        result['version'] = f"similarity_{similarity.value}_top_k_{top_k}_validate_size_{validate_size}_time_decay_coefficient_{time_decay_coefficient}"
        result['algorithm'] = 'sar'

        df_results = pd.concat([df_results, result], ignore_index=True)

        print(f"Resultado para similaridade {similarity.value}:")
        print(result)
    return df_results


if __name__ == '__main__':
    full_results = pd.DataFrame()

    # print("Iniciando avaliação do modelo SAR")
    # sar_results = evaluate_sar()
    # full_results = pd.concat([full_results, sar_results], ignore_index=True)
    # full_results.to_csv('bench/01_sar.csv', index=False)
    #
    # print("Iniciando avaliação do modelo NCF")
    # ncf_result = evaluate_ncf()
    # full_results = pd.concat([full_results, ncf_result], ignore_index=True)
    # full_results.to_csv('bench/02_sar_ncf.csv', index=False)

    # print("Iniciando avaliação do modelo Bivae")
    # bivae_result = evaluate_bivae()
    # full_results = pd.concat([full_results, bivae_result], ignore_index=True)
    # full_results.to_csv('bench/03_sar_ncf_bivae.csv', index=False)

    print("Iniciando avaliação do modelo LightGCN")
    lightgcn_result = evaluate_lightgcn()
    full_results = pd.concat([full_results, lightgcn_result], ignore_index=True)
    full_results.to_csv('bench/04_sar_ncf_bivae_lightgcn.csv', index=False)

    print("Iniciando avaliação do modelo FastAI")
    fastai_result = evaluate_fastai()
    full_results = pd.concat([full_results, fastai_result], ignore_index=True)
    full_results.to_csv('bench/05_sar_ncf_bivae_lightgcn_fastai.csv', index=False)

    print("Iniciando avaliação do modelo ALS")
    spark = create_spark_session("ALS")
    als_result = evaluate_als(spark)
    full_results = pd.concat([full_results, als_result], ignore_index=True)
    full_results.to_csv('bench/06_sar_ncf_bivae_lightgcn_fastai_als.csv', index=False)
