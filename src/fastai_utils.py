import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *
from sklearn.model_selection import train_test_split

from src.utils.enums import MovieLensDataset, MovieLensType
from src.dataset.movielens.loader import Loader

from src.utils import defaults as d
from src.utils.utils import get_torch_device


def cartesian_product(*arrays):
    """
    Calcula o produto cartesiano dos arrays de entrada.

    Cada linha da saída é uma combinação com um elemento de cada array.

    Parâmetros:
        *arrays: um ou mais arrays 1-D (listas, np.array, etc.).

    Retorna:
        numpy.ndarray: array 2-D onde cada linha representa uma combinação única.
    """
    # Converte todos os arrays para np.array (caso ainda não sejam)
    arrays = [np.asarray(a) for a in arrays]
    # Cria os grids com indexação 'ij' para manter a ordem correta
    grids = np.meshgrid(*arrays, indexing='ij')
    # Empilha os grids na última dimensão e remodela para 2-D
    return np.stack(grids, axis=-1).reshape(-1, len(arrays))


def score(
    learner,
    test_df: pd.DataFrame,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Gera as predições para todas as combinações de usuários e itens fornecidas,
    e, se top_k > 0, reduz o resultado para os top_k itens recomendados por usuário.

    Args:
        learner: Modelo treinado.
        test_df (pd.DataFrame): DataFrame contendo as combinações de usuário e item.
        top_k (int, opcional): Número de itens a recomendar por usuário.
            Se None ou <= 0, retorna todas as predições.

    Returns:
        pd.DataFrame: DataFrame com as predições ordenadas.
    """
    # Cria uma cópia para não modificar o DataFrame original
    df = test_df.copy()

    # Substitui valores não conhecidos pelo modelo por NaN
    total_users, total_items = learner.dls.classes.values()
    df.loc[~df[d.idf_user].isin(total_users), d.idf_user] = np.nan
    df.loc[~df[d.idf_item].isin(total_items), d.idf_item] = np.nan

    # Mapeia os IDs para os índices das embeddings
    u = learner._get_idx(df[d.idf_user], is_item=False)
    m = learner._get_idx(df[d.idf_item], is_item=True)

    # Empilha os índices e prepara o tensor
    x = torch.column_stack((u, m))

    device = get_torch_device()
    x = x.to(device)
    learner.model.to(device)
    learner.model.eval()  # Garante que o modelo está em modo de avaliação

    # Realiza a predição sem computar gradientes
    with torch.no_grad():
        pred = learner.model(x).detach().cpu().numpy()

    # Cria o DataFrame de scores e ordena por usuário e predição
    scores = pd.DataFrame({
        d.idf_user: df[d.idf_user],
        d.idf_item: df[d.idf_item],
        d.idf_prediction: pred
    })
    scores.sort_values([d.idf_user, d.idf_prediction], ascending=[True, False], inplace=True)

    # Se top_k for informado e maior que zero, mantém apenas os top_k itens por usuário
    if top_k is not None and top_k > 0:
        top_scores = scores.groupby(d.idf_user).head(top_k).reset_index(drop=True)
    else:
        top_scores = scores

    return top_scores