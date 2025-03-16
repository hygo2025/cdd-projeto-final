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
from recommenders.datasets.python_splitters import python_random_split
from recommenders.models.cornac.cornac_utils import predict_ranking
import cornac


# Parameters

# Recommend top k items
TOP_K = 10
# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"
# Metrics to use for evaluation
RANKING_METRICS = [ evaluator.ndcg_at_k.__name__, evaluator.precision_at_k.__name__ ]
RATING_METRICS = [ evaluator.rmse.__name__, evaluator.mae.__name__ ]
# Use session hook to evaluate model while training
EVALUATE_WHILE_TRAINING = True

RANDOM_SEED = SEED  # Set seed for deterministic result

# Train and test set pickle file paths. If provided, use them. Otherwise, download the MovieLens dataset.
DATA_DIR = None
TRAIN_PICKLE_PATH = None
TEST_PICKLE_PATH = None
EXPORT_DIR_BASE = os.path.join("outputs", "model")
# Model checkpoints directory. If None, use temp-dir.
MODEL_DIR = None

#### Hyperparameters
MODEL_TYPE = "wide_deep"
STEPS = 50000  # Number of batches to train
BATCH_SIZE = 32
# Wide (linear) model hyperparameters
LINEAR_OPTIMIZER = "adagrad"
LINEAR_OPTIMIZER_LR = 0.0621  # Learning rate
LINEAR_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
LINEAR_L2_REG = 0.0
LINEAR_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
# DNN model hyperparameters
DNN_OPTIMIZER = "adadelta"
DNN_OPTIMIZER_LR = 0.1
DNN_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
DNN_L2_REG = 0.0
DNN_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
# Layer dimensions. Defined as follows to make this notebook runnable from Hyperparameter tuning services like AzureML Hyperdrive
DNN_HIDDEN_LAYER_1 = 0     # Set 0 to not use this layer
DNN_HIDDEN_LAYER_2 = 64    # Set 0 to not use this layer
DNN_HIDDEN_LAYER_3 = 128   # Set 0 to not use this layer
DNN_HIDDEN_LAYER_4 = 512   # Note, at least one layer should have nodes.
DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if h > 0]
DNN_USER_DIM = 32          # User embedding feature dimension
DNN_ITEM_DIM = 16          # Item embedding feature dimension
DNN_DROPOUT = 0.8
DNN_BATCH_NORM = 1

class WideDeepModel(AbstractModel):
    def __init__(self,
                 dataset: MovieLensDataset,
                 top_k: int,
                 batch_size: int,
                 latent_dim: int,
                 encoder_dims: List[int],
                 act_func: str,
                 likelihood: str,
                 epochs: int,
                 lr: float,
                 test_size: float,
                 seed: int):
        super().__init__(
            dataset=dataset,
            model_name=f"model_top_k_{top_k}_latent_dim_{latent_dim}_act_func_{act_func}_likelihood_{likelihood}_epochs_{epochs}_lr_{lr}_seed_{seed}",
        )
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.act_func = act_func
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k
        self.likelihood = likelihood
        self.seed = seed
        self.test_size = test_size
        self.batch_size = batch_size

        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_timestamp])

        self.train_df, self.test_df = python_stratified_split(
            self.df,
            ratio=1 - self.test_size,
            col_user=d.idf_user,
            col_item=d.idf_item,
        )

    def train(self):
        train_set = cornac.data.Dataset.from_uir(self.train_df.itertuples(index=False), seed=self.seed)

        print('Number of users: {}'.format(train_set.num_users))
        print('Number of items: {}'.format(train_set.num_items))

        model = cornac.models.BiVAECF(
            k=self.latent_dim,
            encoder_structure=self.encoder_dims,
            act_fn=self.act_func,
            likelihood=self.likelihood,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            seed=self.seed,
            use_gpu=torch.cuda.is_available(),
            verbose=True
        )

        model.fit(train_set)
        return model

    def predict(self):
        print("Iniciando o processo de predição...")
        model = self.train()
        return predict_ranking(model, self.train_df,
                                usercol=d.idf_user,
                                itemcol=d.idf_item,
                                predcol=d.idf_prediction,
                                remove_seen=True)

    def evaluate(self):
        print("Iniciando o processo de avaliação...")
        predictions_df = self.predict()
        print("Calculando métricas de avaliação (top K)...")
        metrics_at_k = self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)
        print("Avaliação concluída.")
        return metrics_at_k

    def save(self, model: LightGCN):
        pass

    def load(self) -> NCF:
        pass


if __name__ == '__main__':
    model = BivaeModel(
        dataset=MovieLensDataset.ML_1M,
        top_k=10,
        batch_size=1024,
        latent_dim=50,
        encoder_dims=[100],
        act_func='tanh',
        likelihood='pois',
        epochs=100,
        lr=0.005,
        test_size=0.2,
        seed=42
    )
    # Para treinar ou predição, descomente conforme necessário:
    # model.train()
    # model.predict()
    result = model.evaluate()
    print("Resultados da avaliação:")
    print(result)
