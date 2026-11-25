from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "data"
GRAFICOS = BASE / "graficos"
RESULTADOS = BASE / "resultados"
SUBMISSOES = BASE / "submissoes"
for _p in (GRAFICOS, RESULTADOS, SUBMISSOES):
    _p.mkdir(exist_ok=True)

SEED = 42
ALVO = "Transported"
GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

NUMERICAS = ["Age", "TamGrupo", "CabineNum", "GastoTotal_log"] + [f"{g}_log" for g in GASTOS]
BINARIAS = ["CryoSleep", "VIP", "Sozinho", "SemGasto"]
CATEGORICAS = ["HomePlanet", "Destination", "Deck", "Side", "CabineRegiao"]

def carregar_dados():
    treino = pd.read_csv(DADOS / "train.csv")
    teste = pd.read_csv(DADOS / "test.csv")
    return treino, teste

def _para_float(serie):
    return serie.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

def engenharia_atributos(df):
    d = df.copy()

    d["Grupo"] = d["PassengerId"].str.split("_").str[0].astype(int)
    d["TamGrupo"] = d.groupby("Grupo")["PassengerId"].transform("count")
    d["Sozinho"] = (d["TamGrupo"] == 1).astype(float)

    cab = d["Cabin"].str.split("/", expand=True)
    d["Deck"] = cab[0]
    d["CabineNum"] = pd.to_numeric(cab[1], errors="coerce")
    d["Side"] = cab[2]

    for col in ["HomePlanet", "Side"]:
        valor_grupo = d.groupby("Grupo")[col].transform(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        d[col] = d[col].fillna(valor_grupo)

    d["CryoSleep"] = _para_float(d["CryoSleep"])
    d["VIP"] = _para_float(d["VIP"])

    sem_gasto_certo = (d["CryoSleep"] == 1) | (d["Age"] < 13)
    for g in GASTOS:
        d.loc[sem_gasto_certo & d[g].isna(), g] = 0.0

    gastou = d[GASTOS].fillna(0).sum(axis=1) > 0
    d.loc[d["CryoSleep"].isna() & gastou, "CryoSleep"] = 0.0

    for g in GASTOS:
        d[f"{g}_log"] = np.log1p(d[g])
    d["GastoTotal_log"] = np.log1p(d[GASTOS].sum(axis=1, min_count=1))
    d["SemGasto"] = (d[GASTOS].fillna(0).sum(axis=1) == 0).astype(float)

    d["CabineRegiao"] = (d["CabineNum"] // 300).astype("Int64").astype(str).replace("<NA>", np.nan)

    return d

def preprocessador(escalar=True):
    passos_num = [("imputa", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("escala", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos_num), NUMERICAS),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARIAS),
        ("cat", Pipeline([
            ("imputa", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ], verbose_feature_names_out=False)

def preparar_xy(df):
    d = engenharia_atributos(df)
    X = d[NUMERICAS + BINARIAS + CATEGORICAS]
    y = d[ALVO].astype(int) if ALVO in d else None
    return X, y, d["Grupo"].values

def validacao_agrupada(n_splits=5, seed=SEED):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
