import os
from typing import Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
import pandas as pd

from src.utils import defaults as d
from src.dataset.movielens.downloader import Downloader
from src.utils.enums import MovieLensType, MovieLensDataset
from config import settings


def load_schema(ml_type: MovieLensType) -> Optional[StructType]:
    if ml_type == MovieLensType.RATINGS:
        return StructType([
            StructField(d.idf_user, IntegerType(), True),
            StructField(d.idf_item, IntegerType(), True),
            StructField(d.idf_rating, DoubleType(), True),
            StructField(d.idf_timestamp, LongType(), True)
        ])
    elif ml_type == MovieLensType.MOVIES:
        return StructType([
            StructField(d.idf_item, IntegerType(), True),
            StructField(d.idf_title, StringType(), True),
            StructField(d.idf_genres, StringType(), True)
        ])


def filter_invalid_coluns(df: DataFrame, ml_type: MovieLensType) -> DataFrame:
    if ml_type == MovieLensType.RATINGS:
        df = df.filter(col(d.idf_user).isNotNull())
        df = df.filter(col(d.idf_item).isNotNull())
        df = df.filter(col(d.idf_rating).isNotNull())

    return df


def define_schema_pandas(df: pd.DataFrame, ml_type: MovieLensType) -> pd.DataFrame:
    if ml_type == MovieLensType.RATINGS:
        df = df.rename(columns={
            'userId': d.idf_user,
            'movieId': d.idf_item,
            'rating': d.idf_rating,
            'timestamp': d.idf_timestamp
        })
    if ml_type == MovieLensType.MOVIES:
        df = df.rename(columns={
            'movieId': d.idf_item,
            'title': d.idf_title,
            'genres': d.idf_genres
        })
    return df


class Loader:
    def __init__(self, spark: SparkSession = None,
                 download_folder=settings.get("DOWNLOAD_FOLDER", os.environ.get("DOWNLOAD_FOLDER")),
                 extract_folder=settings.get("DOWNLOAD_FOLDER", os.environ.get("DOWNLOAD_FOLDER"))):
        self.extract_folder = extract_folder
        self.download_folder = download_folder
        self.spark = spark

    def _get_file_path(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> str:
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        file_name = ml_type.value
        if dataset == MovieLensDataset.ML_1M:
            file_name = file_name.replace('.csv', '.dat')

        return os.path.join(dataset_folder, file_name)

    def _ensure_dataset(self, dataset: MovieLensDataset):
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        if not os.path.exists(dataset_folder):
            print(f"Dataset {dataset.name} não encontrado em {dataset_folder}. Iniciando download e extração...")
            downloader = Downloader(download_folder=self.download_folder, extract_folder=self.extract_folder)
            downloader.download_and_extract_dataset(dataset)
        else:
            self.logger.info(f"Dataset {dataset.name} já existe em {dataset_folder}.")

    def load_pandas(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> pd.DataFrame:
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")

        if dataset == MovieLensDataset.ML_1M and ml_type == MovieLensType.RATINGS:
            df = pd.read_csv(file_path, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"], encoding='ISO-8859-1')
        elif dataset == MovieLensDataset.ML_1M and ml_type == MovieLensType.MOVIES:
            df = pd.read_csv(file_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding='ISO-8859-1')
        else:
            df = pd.read_csv(file_path)
        return define_schema_pandas(df, ml_type)

    def load_spark(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> DataFrame:
        if self.spark is None:
            raise ValueError("SparkSession não foi inicializada.")
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")

        schema = load_schema(ml_type)

        if schema is not None:
            df = self.spark.read.csv(file_path, header=False, schema=schema)
        else:
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)

        return filter_invalid_coluns(df=df, ml_type=ml_type)
