"""Definição dos classificadores e reconstrução dos modelos a partir dos hiperparâmetros salvos."""
import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preparacao import RESULTADOS, SEED, preprocessador

# Nomes das técnicas (são as chaves usadas nos CSVs de resultados)
BASELINE = "Baseline (classe majoritária)"
LR = "Regressão Logística"
LGBM = "LightGBM (boosting de árvores)"
SVM = "SVM (kernel RBF)"
MLP = "Rede Neural (MLP)"
ENSEMBLE = "Ensemble (votação suave)"
KNN = "k-NN (vizinhos mais próximos)"
MLP_LIMIAR = "MLP + limiar ajustado"
STACKING = "Stacking (regressão logística)"

BASES = [LR, LGBM, SVM, MLP]  # os quatro modelos que entram nos ensembles


def novo_classificador(tecnica):
    """Classificador de cada técnica, com os parâmetros fixos do projeto."""
    if tecnica == BASELINE:
        return DummyClassifier(strategy="most_frequent")
    if tecnica == LR:
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED)
    if tecnica == LGBM:
        return LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
    if tecnica == SVM:
        return SVC(kernel="rbf", random_state=SEED)
    if tecnica == MLP:
        return MLPClassifier(early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=20, max_iter=1000, random_state=SEED)
    if tecnica == KNN:
        return KNeighborsClassifier(n_jobs=1)
    raise ValueError(f"técnica desconhecida: {tecnica}")


def pipe(modelo):
    """Pré-processamento + modelo no mesmo Pipeline (evita vazamento de dados)."""
    return Pipeline([("prep", preprocessador()), ("clf", modelo)])


def hiperparametros_salvos():
    """Lê os melhores hiperparâmetros de resultados.csv (e de resultados_extras.csv, se existir)."""
    saidas = {}
    for nome_arq in ("resultados.csv", "resultados_extras.csv"):
        arq = RESULTADOS / nome_arq
        if arq.exists():
            df = pd.read_csv(arq)
            saidas.update({r.tecnica: json.loads(r.params) for r in df.itertuples()})
    return saidas


def reconstruir(tecnica, params):
    """Recria o Pipeline de uma técnica com os hiperparâmetros já encontrados (sem treinar)."""
    est = pipe(novo_classificador(tecnica))
    p = dict(params)
    if "clf__hidden_layer_sizes" in p:  # no JSON a tupla vira lista
        p["clf__hidden_layer_sizes"] = tuple(p["clf__hidden_layer_sizes"])
    return est.set_params(**p)
