import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *
from sklearn.model_selection import train_test_split

from src.abstract_model import AbstractModel
from src.utils.ms import MovieLensDataset, MovieLensType
from src.dataset.movielens.loader import Loader

from src.utils import defaults as d
from src.utils.utils import get_torch_device
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var
from recommenders.models.fastai.fastai_utils import cartesian_product, score
from tqdm import tqdm
import numpy as np
import joblib

class FastAiModel(AbstractModel):
    def __init__(self, dataset: MovieLensDataset, n_factors: int, test_size: float, epochs: int, top_k: int, seed: int):
        print("Inicializando o modelo FastAiModel...")
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

        print("Preparando os dados com prepare_data_pandas...")
        self.df = self.prepare_data_pandas([d.idf_user, d.idf_item, d.idf_rating, d.idf_title])
        print("Dividindo os dados em treino e teste...")
        self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)
        print(f"Conjunto de treino: {len(self.train_df)} linhas; Conjunto de teste: {len(self.test_df)} linhas.")

        print("Filtrando o conjunto de teste para incluir apenas usuários presentes no treino...")
        self.test_df = self.test_df[self.test_df[d.idf_user].isin(self.train_df[d.idf_user])]
        print("Inicialização concluída.\n")

    def _prepare(self):
        print("Configurando sementes para reprodutibilidade...")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        print("Sementes configuradas.\n")

    def train(self):
        print("Iniciando treinamento do modelo FastAi...")
        self._prepare()
        print("Criando DataLoaders de colaboração a partir do conjunto de treino...")
        data = CollabDataLoaders.from_df(self.train_df, user_name=d.idf_user, item_name=d.idf_item, rating_name=d.idf_rating, valid_pct=0)
        print("Criando o learner do modelo de colaboração...")
        learn = collab_learner(data, n_factors=self.n_factors, y_range=self.rating_range, wd=1e-1)
        print("Iniciando o ciclo de treinamento...")
        learn.fit_one_cycle(self.epochs, lr_max=5e-3)
        # self.save(learn)
        print("Treinamento concluído.\n")
        return learn

    def predict(self):
        print("Iniciando processo de predição...")
        self._prepare()
        print("Carregando o modelo treinado...")
        learner = self.load()

        print("Obtendo usuários e itens conhecidos pelo modelo...")
        total_users, total_items = learner.dls.classes.values()
        # Remove o token especial
        total_users = total_users[1:]
        total_items = total_items[1:]
        print(f"Total de usuários conhecidos: {len(total_users)}; Total de itens conhecidos: {len(total_items)}")

        print("Filtrando usuários do conjunto de teste que estão presentes no modelo...")
        test_users = self.test_df[d.idf_user].unique()
        test_users = np.intersect1d(test_users, total_users)
        print(f"Número de usuários de teste filtrados: {len(test_users)}")

        print("Criando produto cartesiano entre usuários válidos e itens conhecidos...")
        users_items = cartesian_product(np.array(test_users), np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=[d.idf_user, d.idf_item])

        print("Garantindo consistência dos tipos de dados...")
        users_items[d.idf_user] = users_items[d.idf_user].astype(int)
        users_items[d.idf_item] = users_items[d.idf_item].astype(int)

        print("Removendo pares já presentes no conjunto de treino...")
        training_removed = pd.merge(
            users_items,
            self.train_df,
            on=[d.idf_user, d.idf_item],
            how='left'
        )
        training_removed = training_removed[training_removed[d.idf_rating].isna()][[d.idf_user, d.idf_item]]
        training_removed = training_removed[training_removed[d.idf_user].isin(total_users)]
        print(f"Total de pares para predição: {len(training_removed)}")

        print("Calculando pontuações para os pares filtrados...")
        top_k_scores = score(
            learner,
            test_df=training_removed,
            user_col=d.idf_user,
            item_col=d.idf_item,
            prediction_col=d.idf_prediction,
            top_k=self.top_k
        )
        print("Predição concluída.\n")
        return top_k_scores

    def evaluate(self):
        print("Iniciando avaliação do modelo FastAi...")
        predictions_df = self.predict()
        print("Convertendo tipos dos dados de predição...")
        predictions_df = predictions_df.astype({
            'user_id': 'int64',
            'movie_id': 'int64',
            'prediction': 'float64'
        })
        print("Calculando métricas de avaliação (top K)...")
        metrics_at_k = self.at_k_metrics(test_df=self.test_df, top_k=self.top_k, predictions_df=predictions_df)
        print("Avaliação concluída.\n")
        return metrics_at_k

    def save(self, learn):
        model_path = self.get_path('fastai')
        learn.export(model_path)
        print(f"Modelo exportado com sucesso em: {model_path}\n")

    def load(self):
        model_path = self.get_path('fastai')
        print(f"Carregando modelo do caminho: {model_path}")
        model = load_learner(model_path)
        print("Modelo carregado com sucesso.\n")
        return model


if __name__ == '__main__':
    print("Executando FastAiModel...")
    model = FastAiModel(dataset=MovieLensDataset.ML_1M, n_factors=40, test_size=0.25, epochs=1, top_k=10, seed=42)
    print("Treinando modelo FastAi...")
    model.train()
    print("Avaliando modelo FastAi...")
    result = model.evaluate()
    print("Resultados da avaliação:")
    print(result)
