import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from src.modelos import BASES, KNN, hiperparametros_salvos, reconstruir
from src.preparacao import GRAFICOS, RESULTADOS, SEED, preparar_dados, validacao_agrupada

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(SEED)

salvos = hiperparametros_salvos()
tecnicas = [n for n in BASES + [KNN] if n in salvos]

conjuntos = {}
for usar in (False, True):
    _, _, X, y, grupos, _ = preparar_dados(usar_familia=usar)
    conjuntos[usar] = (X, y, grupos)
X0, y0, g0 = conjuntos[False]
idx_tr, idx_val = next(validacao_agrupada().split(X0, y0, g0))
cv_interna = validacao_agrupada()

linhas, difs_dobras = [], {}
for nome in tecnicas:
    print(f"=== {nome}")
    cv, val = {}, {}
    for usar, (X, y, g) in conjuntos.items():
        est = reconstruir(nome, salvos[nome])
        X_tr, y_tr, g_tr = X.iloc[idx_tr], y.iloc[idx_tr], g[idx_tr]
        cv[usar] = cross_val_score(est, X_tr, y_tr, groups=g_tr, cv=cv_interna,
                                   scoring="accuracy", n_jobs=-1)
        est.fit(X_tr, y_tr)
        val[usar] = accuracy_score(y.iloc[idx_val], est.predict(X.iloc[idx_val]))
    dif = cv[True] - cv[False]
    difs_dobras[nome] = dif
    linhas.append(dict(tecnica=nome, cv_sem=cv[False].mean(), cv_com=cv[True].mean(),
                       dif_cv_media=dif.mean(), dif_cv_dp=dif.std(),
                       dobras_com_ganho=int((dif > 0).sum()),
                       val_sem=val[False], val_com=val[True], dif_val=val[True] - val[False]))
    print(f"  CV: {cv[False].mean():.4f} -> {cv[True].mean():.4f} "
          f"(dif {dif.mean():+.4f}, ganho em {int((dif > 0).sum())}/5 dobras) | "
          f"validação: {val[False]:.4f} -> {val[True]:.4f}")

tab = pd.DataFrame(linhas)
tab.to_csv(RESULTADOS / "ablacao_familia.csv", index=False)
print("\n" + tab.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.3))
for i, (nome, dif) in enumerate(difs_dobras.items()):
    ax.scatter(np.full(len(dif), i), dif, color="#2a6fbb", alpha=0.6, zorder=3)
    ax.hlines(dif.mean(), i - 0.25, i + 0.25, color="k", lw=2, zorder=4)
ax.axhline(0, c="gray", ls="--", lw=1)
ax.set_xticks(range(len(difs_dobras)))
ax.set_xticklabels([n.split(" (")[0] for n in difs_dobras], fontsize=9)
ax.set_ylabel("Acurácia com família − sem família")
ax.set_title("Efeito do atributo de família na CV (um ponto por dobra; traço = média)")
fig.tight_layout(); fig.savefig(GRAFICOS / "19_ablacao_familia.png", dpi=150); plt.close(fig)
print("\nConcluído (ablação). Para adotar o atributo: USAR_FAMILIA = True em src/preparacao.py.")
