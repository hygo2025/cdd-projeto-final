
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

import joblib

from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.sar import SAR

from src.abstract_model import AbstractModel
from src.utils.enums import MovieLensDataset, SimilarityType
from src.utils import defaults as d


class SarModel(AbstractModel):
    def __init__(self,
                 dataset: MovieLensDataset,
                 top_k:int,
                 validate_size: float,
                 time_decay_coefficient: int,
                 similarity_type:SimilarityType,
                 seed:int):
        super().__init__(
            dataset = dataset,
            model_name=f"movielens_sar_model_time_decay_coefficient_{time_decay_coefficient}_similarity_type_{similarity_type}_top_k_{top_k}.model"
        )

        self.dataset = dataset
        self.top_k = top_k
        self.similarity_type = similarity_type
        self.seed = seed
        self.time_decay_coefficient = time_decay_coefficient

        # Prepara os dados: carrega ratings e movies e faz o merge
        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_timestamp, d.idf_title])
        self.df[d.idf_rating] = self.df[d.idf_rating].astype(np.float32)

        # Divide os dados em treino e teste
        self.train_df, self.test_df = python_stratified_split(
            self.df, ratio=1-validate_size, col_user=d.idf_user, col_item=d.idf_item, seed=seed
        )


    def train(self):
        model = SAR(
            similarity_type=self.similarity_type.value,
            time_decay_coefficient=self.time_decay_coefficient,
            timedecay_formula=True,
            col_user=d.idf_user,
            col_item=d.idf_item,
            col_rating=d.idf_rating,
            col_timestamp=d.idf_timestamp,
            col_prediction=d.idf_prediction,
        )
        model.fit(self.train_df)
        self.save(model)

    def predict(self) -> DataFrame:
        model = self.load()
        top_all = model.recommend_k_items(self.train_df, top_k=self.top_k, remove_seen=True)

        top_k_with_titles = top_all.join(
            self.df[[d.idf_item, d.idf_title]].drop_duplicates().set_index(d.idf_item),
            on=d.idf_item,
            how="inner",
        ).sort_values(by=[d.idf_user, d.idf_prediction], ascending=False)

        return top_k_with_titles

    def evaluate(self):
        # Gera as predições para avaliação de ranking
        predictions_df = self.predict()
        return self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)

    def save(self, model: SAR):
        model_path = self.get_model_path('sar')
        joblib.dump(value=model, filename=model_path)
        print(f"Model saved to {model_path}")

    def load(self) -> SAR:
        model_path = self.get_model_path('sar')
        model = joblib.load(model_path)
        return model

if __name__ == '__main__':
    model = SarModel(
        dataset=MovieLensDataset.ML_100K,
        top_k=10,
        validate_size=0.25,
        time_decay_coefficient=30,
        similarity_type=SimilarityType.COSINE,
        seed=42
    )
    model.train()
    result = model.evaluate()
    print(result)