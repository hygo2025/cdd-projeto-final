import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *
from sklearn.model_selection import train_test_split

from recommenders.datasets.python_splitters import python_chrono_split

from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from src.abstract_model import AbstractModel
from src.utils.enums import MovieLensDataset, MovieLensType
from src.dataset.movielens.loader import Loader

from recommenders.models.deeprec.deeprec_utils import prepare_hparams

from src.utils import defaults as d
from src.utils.utils import get_torch_device
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var
from recommenders.models.fastai.fastai_utils import cartesian_product, score
import joblib
from tqdm import tqdm
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split


class LightGcnModel(AbstractModel):
    def __init__(self,
                 dataset: MovieLensDataset,
                 n_layers: int,
                 batch_size: int,
                 lr: float,
                 epochs: int,
                 top_k: int,
                 test_size: float,
                 seed: int):
        super().__init__(
            dataset=dataset,
            model_name=f"model_batch_size_{batch_size}_epochs_{epochs}_lr_{lr}_n_layers_{n_layers}_seed_{seed}",
        )
        self.dataset = dataset
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k
        self.test_size = test_size
        self.seed = seed

        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_timestamp])

        self.train_df, self.test_df = python_stratified_split(
            self.df,
            ratio=1 - self.test_size,
            col_user=d.idf_user,
            col_item=d.idf_item,
        )

    def train(self):
        data = ImplicitCF(train=self.train_df,
                          test=self.test_df,
                          col_user=d.idf_user,
                          col_item=d.idf_item,
                          col_rating=d.idf_rating,
                          seed=self.seed)
        hparams = prepare_hparams('./resource/lightgcn.yaml',
                                  n_layers=self.n_layers,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  learning_rate=self.lr,
                                  eval_epoch=5,
                                  top_k=self.top_k,
                                  save_model=True,
                                  save_epoch=1,
                                  MODEL_DIR=f"{self.get_path('lightgcn')}/model",
                                  )
        model = LightGCN(hparams, data, seed=self.seed)
        model.fit()
        # self.save(model)
        return model

    def predict(self):
        print("Iniciando o processo de predição...")
        model = self.train()
        return model.recommend_k_items(self.train_df, top_k=self.top_k, remove_seen=True)

    def evaluate(self):
        print("Iniciando o processo de avaliação...")
        predictions_df = self.predict()
        print("Calculando métricas de avaliação (top K)...")
        metrics_at_k = self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)
        print("Avaliação concluída.")
        return metrics_at_k

    def save(self, model: LightGCN):
        model_path = self.get_path('lightgcn')
        model.infer_embedding(user_file=f"{model_path}/user_embeddings.csv",
                              item_file=f"{model_path}/item_embeddings.csv")

        print(f"Modelo salvo em: {model_path}")

    def load(self) -> NCF:
        pass


if __name__ == '__main__':
    model = LightGcnModel(
        dataset=MovieLensDataset.ML_1M,
        n_layers=3,
        batch_size=1024,
        lr=0.005,
        epochs=1,
        top_k=10,
        test_size=0.2,
        seed=42
    )
    # Para treinar ou predição, descomente conforme necessário:
    # model.train()
    # model.predict()
    result = model.evaluate()
    print("Resultados da avaliação:")
    print(result)
