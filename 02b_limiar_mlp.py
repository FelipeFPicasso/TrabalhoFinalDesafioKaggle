"""Ajuste do limiar de decisão da MLP (comparação antes x depois).

Rodar depois de 02_modelos.py, pois usa os hiperparâmetros salvos em resultados.csv.
O limiar é escolhido só com previsões fora da dobra (CV agrupada) na parte de treino.
A validação reservada e o Kaggle não participam da escolha.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.preparacao import (GRAFICOS, RESULTADOS, SEED, SUBMISSOES, carregar_dados,
                            preparar_xy, preprocessador, validacao_agrupada)

LIMIARES = np.round(np.arange(0.30, 0.7001, 0.005), 3)


def prever(prob, limiar):
    return (prob > limiar).astype(int)


def escolher_limiar(y_real, prob):
    """Limiar de maior acurácia; em caso de empate, o mais próximo de 0,5."""
    acc = np.array([accuracy_score(y_real, prever(prob, t)) for t in LIMIARES])
    empatados = LIMIARES[np.isclose(acc, acc.max())]
    return float(empatados[np.argmin(np.abs(empatados - 0.5))]), acc


treino, teste = carregar_dados()
X, y, grupos = preparar_xy(treino)
X_teste, _, _ = preparar_xy(teste)

# Mesma divisão de 02_modelos.py
idx_tr, idx_val = next(validacao_agrupada().split(X, y, grupos))
X_tr, y_tr, g_tr = X.iloc[idx_tr], y.iloc[idx_tr], grupos[idx_tr]
X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]

res = pd.read_csv(RESULTADOS / "resultados.csv")
salvos = json.loads(res.loc[res.tecnica == "Rede Neural (MLP)", "params"].iloc[0])
params = {k.replace("clf__", ""): v for k, v in salvos.items()}
params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

modelo = Pipeline([
    ("prep", preprocessador()),
    ("clf", MLPClassifier(early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=20, max_iter=1000, random_state=SEED,
                          **params)),
])

# 1) Probabilidades fora da dobra, somente no treino
cv = validacao_agrupada()
dobras = list(cv.split(X_tr, y_tr, g_tr))
prob_oof = cross_val_predict(modelo, X_tr, y_tr, cv=dobras, method="predict_proba")[:, 1]
limiar, acc_oof = escolher_limiar(y_tr, prob_oof)

# 2) CV depois do ajuste: o limiar de cada dobra é escolhido nas outras quatro,
#    para que a dobra avaliada não influencie o próprio limiar
cv_antes, cv_depois, limiares_dobra = [], [], []
for _, avaliacao in dobras:
    resto = np.setdiff1d(np.arange(len(y_tr)), avaliacao)
    t, _ = escolher_limiar(y_tr.iloc[resto], prob_oof[resto])
    limiares_dobra.append(t)
    cv_antes.append(accuracy_score(y_tr.iloc[avaliacao], prever(prob_oof[avaliacao], 0.5)))
    cv_depois.append(accuracy_score(y_tr.iloc[avaliacao], prever(prob_oof[avaliacao], t)))

# 3) Antes x depois com o modelo treinado em todo o treino
modelo_tr = clone(modelo).fit(X_tr, y_tr)
prob_tr = modelo_tr.predict_proba(X_tr)[:, 1]
prob_val = modelo_tr.predict_proba(X_val)[:, 1]

# 4) Modelo final nos dados rotulados completos, com o mesmo limiar
final = clone(modelo).fit(X, y)
prob_teste = final.predict_proba(X_teste)[:, 1]

# Confere se o "antes" reproduz a MLP original
originais = pd.read_csv(RESULTADOS / "predicoes_validacao.csv")["Rede Neural (MLP)"].values
dif_val = int((prever(prob_val, 0.5) != originais).sum())
sub_original = pd.read_csv(SUBMISSOES / "04_rede_neural.csv").Transported.values
dif_teste = int((prever(prob_teste, 0.5) != sub_original).sum())
print(f"Diferenças em relação à MLP original: validação={dif_val}, teste={dif_teste}")
if dif_val or dif_teste:
    print("AVISO: o resultado original não foi reproduzido; confira as versões das bibliotecas.")

linhas = []
for nome, t, cv_acc in [("antes", 0.5, cv_antes), ("depois", limiar, cv_depois)]:
    pv, pt = prever(prob_val, t), prever(prob_teste, t)
    acertos = (pv == y_val.values).astype(float)
    rng = np.random.default_rng(SEED)
    boot = [acertos[rng.integers(0, len(acertos), len(acertos))].mean() for _ in range(2000)]
    linhas.append(dict(
        versao=nome, limiar=t,
        acc_treino=accuracy_score(y_tr, prever(prob_tr, t)),
        acc_cv_media=np.mean(cv_acc), acc_cv_dp=np.std(cv_acc),
        acc_validacao=accuracy_score(y_val, pv),
        ic95_inf=np.percentile(boot, 2.5), ic95_sup=np.percentile(boot, 97.5),
        revocacao_false=recall_score(y_val, pv, pos_label=0),
        revocacao_true=recall_score(y_val, pv, pos_label=1),
        taxa_true_validacao=pv.mean(), taxa_true_teste=pt.mean()))
tabela = pd.DataFrame(linhas)
tabela.to_csv(RESULTADOS / "limiar_mlp.csv", index=False)

# McNemar exato entre as duas versões na validação
a = prever(prob_val, limiar) == y_val.values
b = prever(prob_val, 0.5) == y_val.values
so_depois, so_antes = int((a & ~b).sum()), int((~a & b).sum())
p = binomtest(so_depois, so_depois + so_antes, 0.5).pvalue if so_depois + so_antes else 1.0
pd.DataFrame([dict(so_depois_acerta=so_depois, so_antes_acerta=so_antes, p_valor=p)]) \
    .to_csv(RESULTADOS / "limiar_mlp_mcnemar.csv", index=False)

pd.DataFrame({"PassengerId": teste.PassengerId,
              "Transported": prever(prob_teste, limiar).astype(bool)}) \
    .to_csv(SUBMISSOES / "06_rede_neural_limiar.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(LIMIARES, acc_oof, label="CV no treino (fora da dobra)")
ax.axvline(0.5, c="gray", ls=":", label="Limiar padrão (0,5)")
ax.axvline(limiar, c="k", ls="--", label=f"Limiar escolhido ({limiar:.3f})")
ax.set_xlabel("Limiar de decisão")
ax.set_ylabel("Acurácia (maior é melhor)")
ax.set_title("MLP: acurácia em função do limiar de decisão")
ax.legend()
fig.tight_layout()
fig.savefig(GRAFICOS / "17_mlp_limiar.png", dpi=150)
plt.close(fig)

print(f"Limiar escolhido: {limiar:.3f} | limiares por dobra: {limiares_dobra}")
print(tabela.round(4).T.to_string(header=False))
print(f"McNemar: só depois acerta={so_depois}, só antes acerta={so_antes}, p={p:.4f}")
