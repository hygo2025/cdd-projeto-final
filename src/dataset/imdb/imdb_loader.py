import os

import pandas as pd
import ast
import json
from tqdm import tqdm
import requests
import zipfile
from src.utils import defaults as d

tqdm.pandas()


def download_file(url: str = d.imdb_url, base_path: str = None):
    paths = d.get_imdb_paths(base_path)
    output_path = paths['imdb_output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"O arquivo já existe em {output_path}. Pulando o download.")
        return

    print("Iniciando download do arquivo...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\rProgresso: {percent:.2f}%", end="")
    print("\nDownload concluído.")


def extract_zip(base_path: str = None):
    paths = d.get_imdb_paths(base_path)
    output_path = paths['imdb_output_path']
    extract_to = paths['imdb_extract_to']

    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"O conteúdo já foi extraído em {extract_to}. Pulando a extração.")
        return

    print(f"Iniciando extração de {output_path} para {extract_to}...")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extração concluída.")

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

    links = pd.read_csv(links_path, usecols=links_usecols)
    movies_metadata = pd.read_csv(movies_metadata_path, usecols=movies_usecols)

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

def load_films_data() -> pd.DataFrame:
    paths = d.get_imdb_paths()
    links_path= paths['imdb_links_path']
    movies_metadata_path = paths['imdb_movies_metadata_path']
    output_full_path = paths['imdb_output_full_path']

    if not os.path.exists(output_full_path):
        download_file()
        extract_zip()
        process_data(links_path, movies_metadata_path, output_full_path)

    return pd.read_csv(output_full_path)
