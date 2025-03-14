import pandas as pd
import ast
from tqdm import tqdm

tqdm.pandas()

def parse_genres(genre_entry):
    try:
        if isinstance(genre_entry, str):
            return ast.literal_eval(genre_entry)
    except Exception:
        return []
    return genre_entry

def process_data(links_path, movies_metadata_path, ratings_path, output_path):
    # Leitura dos arquivos CSV
    links = pd.read_csv(links_path)
    movies_metadata = pd.read_csv(movies_metadata_path)
    ratings = pd.read_csv(ratings_path)

    # Remover registros com 'id' nulo em movies_metadata
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
    movies_metadata = movies_metadata.dropna(subset=['id'])
    movies_metadata['id'] = movies_metadata['id'].astype(int)

    # Remover registros com 'tmdbId' nulo em links
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    # Merge: links, movies_metadata e ratings
    merge_df = pd.merge(links, movies_metadata, left_on='tmdbId', right_on='id', how='left')
    data = pd.merge(merge_df, ratings, on='movieId', how='left')

    print("Data shape:", data.shape)

    # Seleciona as colunas desejadas
    data = data[['movieId', 'userId', 'rating', 'timestamp', 'adult', 'budget',
                 'genres', 'original_language', 'original_title', 'overview',
                 'popularity', 'production_companies', 'release_date', 'runtime',
                 'title', 'vote_average', 'vote_count']]

    # Processa a coluna 'genres' convertendo os dados para lista de dicionários
    data['genres_list'] = data['genres'].progress_apply(parse_genres)

    # Extrai todos os nomes de gêneros presentes no dataframe
    all_genres = set()
    for genres in tqdm(data['genres_list'], desc="Extraindo gêneros"):
        if isinstance(genres, list):
            for genre in genres:
                all_genres.add(genre['name'].lower().replace(" ", "_"))

    # Cria uma coluna booleana para cada gênero encontrado, ex: 'genre-animation'
    for genre in tqdm(all_genres, desc="Criando colunas de gêneros"):
        data[f'genre-{genre}'] = data['genres_list'].apply(
            lambda x: any(g['name'].lower().replace(" ", "_") == genre for g in x) if isinstance(x, list) else False
        )

    data.drop(columns=['genres_list', 'genres'], inplace=True)

    # Exporta o dataframe para CSV
    data.to_csv(output_path, index=False)

if __name__ == '__main__':
    print("Processing small data...")
    links_small_path = '../data/imdb_movielens/links_small.csv'
    movies_metadata_path = '../data/imdb_movielens/movies_metadata.csv'
    ratings_small_path = '../data/imdb_movielens/ratings_small.csv'
    output_path = '../data/imdb_movielens/small_data.csv'

    process_data(links_small_path, movies_metadata_path, ratings_small_path, output_path)

    print("Processing full data...")
    links_path = '../data/imdb_movielens/links.csv'
    movies_metadata_path = '../data/imdb_movielens/movies_metadata.csv'
    ratings_path = '../data/imdb_movielens/ratings.csv'
    output_path = '../data/imdb_movielens/full_data.csv'

    process_data(links_path, movies_metadata_path, ratings_path, output_path)
