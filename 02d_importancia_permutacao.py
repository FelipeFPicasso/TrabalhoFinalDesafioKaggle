import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from src.modelos import BASES, KNN, hiperparametros_salvos, reconstruir
from src.preparacao import (GASTOS, GRAFICOS, RESULTADOS, SEED, preparar_dados,
                            validacao_agrupada)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid", context="notebook")
N_REPETICOES = 20

_, _, X, y, grupos, _ = preparar_dados()
idx_tr, idx_val = next(validacao_agrupada().split(X, y, grupos))
X_tr, y_tr, X_val, y_val = X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_val], y.iloc[idx_val]

GRUPOS_ATRIBUTOS = {
    "Gastos a bordo": [f"{g}_log" for g in GASTOS] + ["GastoTotal_log", "SemGasto"],
    "CryoSleep": ["CryoSleep"],
    "Cabine (deck, lado, nº, região)": ["Deck", "Side", "CabineNum", "CabineRegiao"],
    "Planeta de origem": ["HomePlanet"],
    "Destino": ["Destination"],
    "Idade": ["Age"],
    "Grupo de viagem": ["TamGrupo", "Sozinho"],
    "VIP": ["VIP"],
}
if "TamFamilia" in X.columns:
    GRUPOS_ATRIBUTOS["Família (sobrenome)"] = ["TamFamilia"]

salvos = hiperparametros_salvos()
modelos = {}
for n in [n for n in BASES + [KNN] if n in salvos]:
    modelos[n] = reconstruir(n, salvos[n]).fit(X_tr, y_tr)


def queda_por_permutacao(modelo, colunas, rng):
    base = accuracy_score(y_val, modelo.predict(X_val))
    quedas = []
    for _ in range(N_REPETICOES):
        Xp = X_val.copy()
        ordem = rng.permutation(len(Xp))
        for c in colunas:
            Xp[c] = X_val[c].values[ordem]
        quedas.append(base - accuracy_score(y_val, modelo.predict(Xp)))
    return np.mean(quedas), np.std(quedas)


linhas = []
for nome, modelo in modelos.items():
    print(f"=== {nome}")
    rng = np.random.default_rng(SEED)
    for grupo, cols in GRUPOS_ATRIBUTOS.items():
        m, s = queda_por_permutacao(modelo, cols, rng)
        linhas.append(dict(modelo=nome, grupo=grupo, queda_media=m, queda_dp=s))
        print(f"  {grupo}: queda de {m * 100:.2f} p.p. (dp {s * 100:.2f})")

imp = pd.DataFrame(linhas)
imp.to_csv(RESULTADOS / "importancia_permutacao.csv", index=False)

larga = imp.pivot(index="grupo", columns="modelo", values="queda_media")[list(modelos)]
larga = larga.loc[larga.mean(axis=1).sort_values(ascending=False).index]
sp = larga.corr(method="spearman")
sp.to_csv(RESULTADOS / "importancia_permutacao_spearman.csv")
print("\nConcordância entre modelos (Spearman):\n" + sp.round(2).to_string())

fig, ax = plt.subplots(figsize=(9.5, 5))
rotulos = [n.split(" (")[0].replace(" ", "\n", 1) for n in larga.columns]
sns.heatmap(larga * 100, annot=True, fmt=".1f", cmap="Blues", ax=ax,
            xticklabels=rotulos, cbar_kws={"label": "Queda de acurácia (p.p.)"})
ax.set_ylabel(""); ax.set_xlabel("")
ax.set_title("Importância por permutação na validação\n"
             f"(média de {N_REPETICOES} repetições; maior = o modelo depende mais do grupo)")
fig.tight_layout(); fig.savefig(GRAFICOS / "18_importancia_permutacao.png", dpi=150)
plt.close(fig)
print("\nConcluído (permutação).")
