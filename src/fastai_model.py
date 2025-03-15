import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *
from sklearn.model_selection import train_test_split

from src.abstract_model import AbstractModel
from src.utils.enums import MovieLensDataset, MovieLensType
from src.dataset.movielens.loader import Loader

from src.utils import defaults as d
from src.utils.utils import get_torch_device
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var
from recommenders.models.fastai.fastai_utils import cartesian_product, score


class FastAiModel(AbstractModel):
    def __init__(self, dataset: MovieLensDataset, n_factors: int, test_size: float, epochs: int, top_k: int, seed: int):
        super().__init__(
            dataset=dataset,
            model_name=f"movielens_model_n_factors_{n_factors}_epochs_{epochs}_.pkl"
        )

        self.seed = seed
        self.n_factors = n_factors
        self.dataset = dataset
        self.test_size = test_size
        self.epochs = epochs
        self.top_k = top_k
        self.rating_range = [0, 5.5]

        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_title])
        self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)

        self.test_df = self.test_df[self.test_df[d.idf_user].isin(self.train_df[d.idf_user])]

    def _prepare(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train(self):
        self._prepare()
        data = CollabDataLoaders.from_df(self.train_df, user_name=d.idf_user, item_name=d.idf_item,
                                         rating_name=d.idf_rating, valid_pct=0)
        learn = collab_learner(data, n_factors=self.n_factors, y_range=self.rating_range, wd=1e-1)
        learn.fit_one_cycle(self.epochs, lr_max=5e-3)
        self.save(learn)

    def predict(self):
        self._prepare()
        learner = self.load()

        # Obtém os usuários e itens que o modelo conhece (geralmente, os do treinamento)
        total_users, total_items = learner.dls.classes.values()
        # Remove o primeiro token especial (se for o caso)
        total_users = total_users[1:]
        total_items = total_items[1:]

        # Seleciona os usuários do conjunto de teste que estão presentes no modelo
        test_users = self.test_df[d.idf_user].unique()
        test_users = np.intersect1d(test_users, total_users)

        # Cria o produto cartesiano com os usuários válidos e todos os itens conhecidos
        users_items = cartesian_product(np.array(test_users), np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=[d.idf_user, d.idf_item])

        # Certifique-se de que os tipos estão consistentes (por exemplo, inteiros)
        users_items[d.idf_user] = users_items[d.idf_user].astype(int)
        users_items[d.idf_item] = users_items[d.idf_item].astype(int)

        # Remove pares que já estão presentes no conjunto de treinamento
        # OBS: Assumindo que self.train_df já possui user_id e movie_id como inteiros
        training_removed = pd.merge(
            users_items,
            self.train_df,  # já deve estar com os tipos corretos
            on=[d.idf_user, d.idf_item],
            how='left'
        )
        # Pega apenas os pares que não estavam no treinamento (rating NaN)
        training_removed = training_removed[training_removed[d.idf_rating].isna()][[d.idf_user, d.idf_item]]

        # Adicionalmente, garanta que os usuários estejam entre os usuários do modelo
        training_removed = training_removed[training_removed[d.idf_user].isin(total_users)]

        # Calcula as pontuações para os pares filtrados
        top_k_scores = score(
            learner,
            test_df=training_removed,
            user_col=d.idf_user,
            item_col=d.idf_item,
            prediction_col=d.idf_prediction,
            top_k=self.top_k
        )

        return top_k_scores

    def evaluate(self):
        predictions_df = self.predict()

        predictions_df = predictions_df.astype({
            'user_id': 'int64',
            'movie_id': 'int64',
            'prediction': 'float64'
        })

        metrics_at_k = self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)
        metrics = self.metrics(test_df=self.test_df, predictions_df=predictions_df)

        return pd.concat([metrics_at_k, metrics], axis=1)

    def save(self, learn):
        model_path = self._get_model_path()
        learn.export(model_path)
        print(f"Modelo exportado com sucesso em {model_path}")

    def load(self):
        model_path = self._get_model_path()
        return load_learner(model_path)

    def _get_model_path(self):
        save_dir = os.path.join('..', 'data', 'fastai')
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, self.model_name)


if __name__ == '__main__':
    model = FastAiModel(dataset=MovieLensDataset.ML_100K, n_factors=42, test_size=0.2, epochs=1, top_k=10, seed=42)
    model.train()
    # model.predict()
    result = model.evaluate()
    print(result)
