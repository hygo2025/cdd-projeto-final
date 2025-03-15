import pandas as pd
import ast
import json
from tqdm import tqdm

tqdm.pandas()

def parse_genres(genre_entry):
    if not isinstance(genre_entry, str) or not genre_entry.strip():
        return []
    try:
        json_compatible = genre_entry.replace("'", '"')
        return json.loads(json_compatible)
    except Exception:
        try:
            return ast.literal_eval(genre_entry)
        except Exception:
            return []


def extract_genres_flags(genres_list, all_genres):
    if not isinstance(genres_list, list):
        return {f'genre-{genre}': False for genre in all_genres}
    present = {g['name'].lower().replace(" ", "_") for g in genres_list if isinstance(g, dict)}
    return {f'genre-{genre}': (genre in present) for genre in all_genres}


def process_data(links_path, movies_metadata_path, output_path):
    links_usecols = ['movieId', 'tmdbId']
    movies_usecols = ['id', 'adult', 'budget', 'genres', 'original_language', 'original_title',
                      'overview', 'popularity', 'release_date', 'runtime',
                      'title', 'vote_average', 'vote_count']

    links = pd.read_csv(links_path, usecols=links_usecols, low_memory=True)
    movies_metadata = pd.read_csv(movies_metadata_path, usecols=movies_usecols, low_memory=True)

    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links.dropna(subset=['tmdbId'], inplace=True)
    links['tmdbId'] = links['tmdbId'].astype('int32')

    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
    movies_metadata.dropna(subset=['id'], inplace=True)
    movies_metadata['id'] = movies_metadata['id'].astype('int32')

    data = pd.merge(links, movies_metadata, left_on='tmdbId', right_on='id', how='left')

    data['genres'] = data['genres'].progress_apply(parse_genres)

    all_genres = set()
    for genres in tqdm(data['genres'], desc="Extraindo gêneros"):
        if isinstance(genres, list):
            for genre in genres:
                all_genres.add(genre['name'].lower().replace(" ", "_"))

    genre_flags = data['genres'].progress_apply(lambda x: extract_genres_flags(x, all_genres))
    genre_flags_df = pd.DataFrame(genre_flags.tolist())

    data = pd.concat([data, genre_flags_df], axis=1)
    data.drop(columns=['genres'], inplace=True)

    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    print("Processing small data...")
    links_small_path = '../../data/imdb_movielens/links_small.csv'
    movies_metadata_path = '../../data/imdb_movielens/movies_metadata.csv'
    output_small_path = '../../data/imdb_movielens/small_data.csv'

    process_data(links_small_path, movies_metadata_path, output_small_path)

    print("Processing full data...")
    links_path = '../../data/imdb_movielens/links.csv'
    movies_metadata_path = '../../data/imdb_movielens/movies_metadata.csv'
    output_full_path = '../../data/imdb_movielens/full_data.csv'

    process_data(links_path, movies_metadata_path, output_full_path)
