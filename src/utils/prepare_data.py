import pandas as pd
import ast
import json
from tqdm import tqdm
import gc

tqdm.pandas()  # Inicializa tqdm para pandas


def parse_genres_fast(genre_entry):
    """
    Tenta converter a string para uma lista de dicionários usando json.loads.
    Se falhar, utiliza ast.literal_eval como fallback.
    """
    if not isinstance(genre_entry, str) or not genre_entry.strip():
        return []
    try:
        # Substitui aspas simples por aspas duplas para formar um JSON válido.
        json_compatible = genre_entry.replace("'", '"')
        return json.loads(json_compatible)
    except Exception:
        try:
            return ast.literal_eval(genre_entry)
        except Exception:
            return []


def extract_genres_flags(genres_list, all_genres):
    """
    Para uma lista de gêneros (lista de dicionários), retorna um dicionário onde
    as chaves são 'genre-{nome}' e os valores são True se o gênero estiver presente e False caso contrário.
    """
    if not isinstance(genres_list, list):
        return {f'genre-{genre}': False for genre in all_genres}
    present = {g['name'].lower().replace(" ", "_") for g in genres_list if isinstance(g, dict)}
    return {f'genre-{genre}': (genre in present) for genre in all_genres}


def process_data(links_path, movies_metadata_path, ratings_path, output_path):
    # Definindo as colunas necessárias para cada CSV
    links_usecols = ['movieId', 'tmdbId']
    movies_usecols = ['id', 'adult', 'budget', 'genres', 'original_language', 'original_title',
                      'overview', 'popularity', 'production_companies', 'release_date', 'runtime',
                      'title', 'vote_average', 'vote_count']
    ratings_usecols = ['movieId', 'userId', 'rating', 'timestamp']

    # Leitura dos CSVs com uso seletivo de colunas e low_memory
    links = pd.read_csv(links_path, usecols=links_usecols, low_memory=True)
    movies_metadata = pd.read_csv(movies_metadata_path, usecols=movies_usecols, low_memory=True)
    ratings = pd.read_csv(ratings_path, usecols=ratings_usecols, low_memory=True)

    # Converter colunas numéricas e eliminar registros nulos
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links.dropna(subset=['tmdbId'], inplace=True)
    links['tmdbId'] = links['tmdbId'].astype('int32')

    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
    movies_metadata.dropna(subset=['id'], inplace=True)
    movies_metadata['id'] = movies_metadata['id'].astype('int32')

    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce').astype('int32')
    ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce').astype('int32')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce').astype('float32')
    ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='coerce').astype('int32')

    # Merge: links, movies_metadata e ratings
    merge_df = pd.merge(links, movies_metadata, left_on='tmdbId', right_on='id', how='left')
    data = pd.merge(merge_df, ratings, on='movieId', how='left')
    print("Data shape:", data.shape)

    # Seleciona somente as colunas finais desejadas
    data = data[['movieId', 'userId', 'rating', 'timestamp', 'adult', 'budget',
                 'genres', 'original_language', 'original_title', 'overview',
                 'popularity', 'production_companies', 'release_date', 'runtime',
                 'title', 'vote_average', 'vote_count']]

    # Processa a coluna 'genres' convertendo os dados para lista de dicionários
    data['genres'] = data['genres'].progress_apply(parse_genres_fast)

    # Extrai todos os nomes de gêneros presentes no DataFrame
    all_genres = set()
    for genres in tqdm(data['genres'], desc="Extraindo gêneros"):
        if isinstance(genres, list):
            for genre in genres:
                all_genres.add(genre['name'].lower().replace(" ", "_"))

    # Cria, para cada linha, um dicionário com as flags de cada gênero
    genre_flags = data['genres'].progress_apply(lambda x: extract_genres_flags(x, all_genres))
    genre_flags_df = pd.DataFrame(genre_flags.tolist())

    # Concatena as colunas de gêneros ao DataFrame original e descarta a coluna 'genres'
    data = pd.concat([data, genre_flags_df], axis=1)
    data.drop(columns=['genres'], inplace=True)

    # Exporta o DataFrame para CSV
    data.to_csv(output_path, index=False)

    # Libera memória excluindo variáveis intermediárias e coletando o lixo
    del links, movies_metadata, ratings, merge_df, genre_flags, genre_flags_df, data
    gc.collect()


if __name__ == '__main__':
    print("Processing small data...")
    links_small_path = '../../data/imdb_movielens/links_small.csv'
    movies_metadata_path = '../../data/imdb_movielens/movies_metadata.csv'
    ratings_small_path = '../../data/imdb_movielens/ratings_small.csv'
    output_small_path = '../data/imdb_movielens/small_data.csv'

    process_data(links_small_path, movies_metadata_path, ratings_small_path, output_small_path)

    print("Processing full data...")
    links_path = '../../data/imdb_movielens/links.csv'
    movies_metadata_path = '../../data/imdb_movielens/movies_metadata.csv'
    ratings_path = '../../data/imdb_movielens/ratings.csv'
    output_full_path = '../data/imdb_movielens/full_data.csv'

    process_data(links_path, movies_metadata_path, ratings_path, output_full_path)
