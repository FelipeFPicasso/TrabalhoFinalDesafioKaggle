"""Preparação dos dados: caminhos, engenharia de atributos, pré-processamento e validação.

As regras daqui não usam o alvo. Tudo o que é aprendido dos dados (medianas, escala,
categorias do one-hot) fica dentro do Pipeline, que só vê o treino de cada dobra.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pastas do projeto (criadas se não existirem)
BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "data"
GRAFICOS = BASE / "graficos"
RESULTADOS = BASE / "resultados"
SUBMISSOES = BASE / "submissoes"
for _p in (GRAFICOS, RESULTADOS, SUBMISSOES):
    _p.mkdir(exist_ok=True)

SEED = 42  # semente única do projeto
ALVO = "Transported"
GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Colunas finais, por tipo de tratamento
NUMERICAS = ["Age", "TamGrupo", "CabineNum", "GastoTotal_log"] + [f"{g}_log" for g in GASTOS]
BINARIAS = ["CryoSleep", "VIP", "Sozinho", "SemGasto"]
CATEGORICAS = ["HomePlanet", "Destination", "Deck", "Side", "CabineRegiao"]
FAMILIA = ["TamFamilia"]  # atributo opcional (sobrenome); só entra se for pedido

# Chave geral: False = projeto original (sem o atributo de família).
# Se a ablação (02c) mostrar ganho e a dupla quiser adotá-lo, troque para True e
# rode tudo de novo (./run_all.sh).
USAR_FAMILIA = False


def carregar_dados():
    treino = pd.read_csv(DADOS / "train.csv")
    teste = pd.read_csv(DADOS / "test.csv")
    return treino, teste


def _para_float(serie):
    # True/False (ou "True"/"False") -> 1.0/0.0, mantendo os ausentes como NaN
    return serie.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})


def contagem_sobrenomes(treino, teste):
    """Quantas vezes cada sobrenome aparece em treino + teste juntos.

    Usa só os nomes, nunca o alvo. A contagem é feita nos dois arquivos juntos para
    que o atributo tenha a mesma escala no treino e no teste.
    """
    nomes = pd.concat([treino["Name"], teste["Name"]]).dropna()
    return nomes.str.split().str[-1].value_counts()


def engenharia_atributos(df, contagem_sobrenome=None):
    """Cria os atributos a partir de regras determinísticas (sem parâmetros aprendidos)."""
    d = df.copy()

    # Grupo de viagem: prefixo do PassengerId
    d["Grupo"] = d["PassengerId"].str.split("_").str[0].astype(int)
    d["TamGrupo"] = d.groupby("Grupo")["PassengerId"].transform("count")
    d["Sozinho"] = (d["TamGrupo"] == 1).astype(float)

    # Cabine = deck/número/lado
    cab = d["Cabin"].str.split("/", expand=True)
    d["Deck"] = cab[0]
    d["CabineNum"] = pd.to_numeric(cab[1], errors="coerce")
    d["Side"] = cab[2]

    # Planeta e lado da cabine: completa com o valor do resto do grupo
    for col in ["HomePlanet", "Side"]:
        valor_grupo = d.groupby("Grupo")[col].transform(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        d[col] = d[col].fillna(valor_grupo)

    d["CryoSleep"] = _para_float(d["CryoSleep"])
    d["VIP"] = _para_float(d["VIP"])

    # Quem está em CryoSleep ou tem menos de 13 anos não gasta: ausente vira 0
    sem_gasto_certo = (d["CryoSleep"] == 1) | (d["Age"] < 13)
    for g in GASTOS:
        d.loc[sem_gasto_certo & d[g].isna(), g] = 0.0

    # CryoSleep ausente mas com gasto > 0 só pode ser False
    gastou = d[GASTOS].fillna(0).sum(axis=1) > 0
    d.loc[d["CryoSleep"].isna() & gastou, "CryoSleep"] = 0.0

    # Gastos em log (muito assimétricos) e indicador de gasto zero
    for g in GASTOS:
        d[f"{g}_log"] = np.log1p(d[g])
    d["GastoTotal_log"] = np.log1p(d[GASTOS].sum(axis=1, min_count=1))
    d["SemGasto"] = (d[GASTOS].fillna(0).sum(axis=1) == 0).astype(float)

    # Região da cabine: faixas de 300 números
    d["CabineRegiao"] = (d["CabineNum"] // 300).astype("Int64").astype(str).replace("<NA>", np.nan)

    # Tamanho da "família" = nº de pessoas com o mesmo sobrenome (opcional)
    if contagem_sobrenome is not None:
        sobrenome = d["Name"].str.split().str[-1]
        d["TamFamilia"] = sobrenome.map(contagem_sobrenome)

    return d


class Selecionar:
    """Escolhe, na hora do ajuste, só as colunas da lista que existem em X.

    Assim o mesmo pré-processador serve com ou sem o atributo de família.
    """

    def __init__(self, colunas):
        self.colunas = colunas

    def __call__(self, X):
        return [c for c in self.colunas if c in X.columns]


def preprocessador(escalar=True):
    """Imputação + escala (numéricas), imputação (binárias) e one-hot (categóricas)."""
    passos_num = [("imputa", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("escala", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos_num), Selecionar(NUMERICAS + FAMILIA)),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARIAS),
        ("cat", Pipeline([
            ("imputa", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ], verbose_feature_names_out=False)


def preparar_xy(df, contagem_sobrenome=None):
    """Devolve X (colunas finais), y (ou None no teste) e o grupo de cada linha."""
    d = engenharia_atributos(df, contagem_sobrenome)
    colunas = NUMERICAS + BINARIAS + CATEGORICAS
    if contagem_sobrenome is not None:
        colunas = colunas + FAMILIA
    X = d[colunas]
    y = d[ALVO].astype(int) if ALVO in d else None
    return X, y, d["Grupo"].values


def preparar_dados(usar_familia=None):
    """Carrega os CSVs e devolve (treino, teste, X, y, grupos, X_teste)."""
    if usar_familia is None:
        usar_familia = USAR_FAMILIA
    treino, teste = carregar_dados()
    contagem = contagem_sobrenomes(treino, teste) if usar_familia else None
    X, y, grupos = preparar_xy(treino, contagem)
    X_teste, _, _ = preparar_xy(teste, contagem)
    return treino, teste, X, y, grupos, X_teste


def validacao_agrupada(n_splits=5, seed=SEED):
    """CV estratificada em que um grupo de viagem nunca é dividido entre dobras."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
