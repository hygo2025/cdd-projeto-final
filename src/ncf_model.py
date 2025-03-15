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

from src.utils import defaults as d
from src.utils.utils import get_torch_device
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var
from recommenders.models.fastai.fastai_utils import cartesian_product, score
import joblib

class NcfModel(AbstractModel):
    def __init__(self,
                 dataset: MovieLensDataset,
                 n_factors:int,
                 batch_size: int,
                 lr: float,
                 layer_sizes: List[int],
                 epochs: int,
                 top_k: int,
                 test_size: float,
                 seed: int):
        super().__init__(
            dataset=dataset,
            model_name=f"movielens_model_batch_size_{batch_size}_epochs_{epochs}.pkl"
        )
        self.dataset = dataset
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.lr = lr
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.top_k = top_k
        self.test_size = test_size
        self.seed = seed

        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_timestamp])
        self.train_df, self.test_df = python_chrono_split(
            self.df,
            ratio=1-self.test_size,
            col_user=d.idf_user,
            col_item=d.idf_item,
            col_timestamp=d.idf_timestamp,
        )

        self.test_df = self.test_df[self.test_df[d.idf_user].isin(self.train_df[d.idf_user].unique())]
        self.test_df = self.test_df[self.test_df[d.idf_item].isin(self.train_df[d.idf_item].unique())]

    def _build_model(self):
        data = NCFDataset(train_file=self.save_to_tmp_file(df=self.train_df, save_dir='ncf', name='train.csv'),
                          test_file=self.save_to_tmp_file(df=self.test_df, save_dir='ncf', name='test.csv'),
                          col_user=d.idf_user,
                          col_item=d.idf_item,
                          col_rating=d.idf_rating,
                          seed=self.seed)
        model = NCF(
            n_users=data.n_users,
            n_items=data.n_items,
            n_factors=self.n_factors,
            layer_sizes=self.layer_sizes,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            verbose=10,
            seed=self.seed
        )
        return model, data
    def train(self):
        model, data = self._build_model()
        model.fit(data)
        self.save(model)
        return model

    def predict(self):
        model = self.train()

        users, items, preds = [], [], []
        item = list(self.train_df[d.idf_item].unique())
        for user in self.train_df[d.idf_user].unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={d.idf_user: users, d.idf_item: items, d.idf_prediction: preds})

        merged = pd.merge(self.train_df, all_predictions, on=[d.idf_user, d.idf_item], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop(d.idf_rating, axis=1)

        return all_predictions

    def evaluate(self):
        predictions_df = self.predict()
        metrics_at_k = self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)
        return metrics_at_k

    def save(self, model: NCF):
        model_path = self.get_path('ncf')
        model.save(model_path)
        print(f"Model saved to {model_path}")

    def load(self) -> NCF:
        #TODO: Essa funcao nao esta funcionando corretamente
        model_path = self.get_path('ncf')
        model, _ = self._build_model()
        model.load(neumf_dir=model_path)
        return model


if __name__ == '__main__':
    model = NcfModel(
        dataset=MovieLensDataset.ML_1M,
        n_factors=42,
        batch_size=256,
        lr=1e-3,
        layer_sizes=[16,8,4],
        epochs=1,
        top_k=10,
        test_size=0.2,
        seed=42)
    # model.train()
    # model.predict()
    result = model.evaluate()
    print(result)
