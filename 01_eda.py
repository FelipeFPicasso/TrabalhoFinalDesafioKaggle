import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preparacao import ALVO, GASTOS, GRAFICOS, RESULTADOS, carregar_dados

sns.set_theme(style="whitegrid", context="notebook")
treino, teste = carregar_dados()
linhas = []

def log(texto=""):
    print(texto)
    linhas.append(str(texto))

log(f"Treino: {treino.shape[0]} linhas x {treino.shape[1]} colunas")
log(f"Teste:  {teste.shape[0]} linhas x {teste.shape[1]} colunas")
log("\nTipos:\n" + treino.dtypes.to_string())
log("\nDescrição numérica:\n" + treino.describe().T.round(2).to_string())
log(f"\nLinhas duplicadas (sem PassengerId): {treino.drop(columns='PassengerId').duplicated().sum()}")

taxa = treino[ALVO].mean()
log(f"\nAlvo Transported: {treino[ALVO].sum()} True / {(~treino[ALVO]).sum()} False "
    f"-> {taxa:.4f} de positivos (classes balanceadas)")

fig, ax = plt.subplots(figsize=(5, 3.5))
cont = treino[ALVO].value_counts().sort_index()
ax.bar(["False", "True"], cont.values, color=["#8c8c8c", "#2a6fbb"])
for i, v in enumerate(cont.values):
    ax.text(i, v + 50, f"{v} ({v / len(treino):.1%})", ha="center")
ax.set_ylabel("Passageiros")
ax.set_title("Distribuição do alvo (Transported)")
ax.set_ylim(0, cont.max() * 1.15)
fig.tight_layout(); fig.savefig(GRAFICOS / "01_alvo.png", dpi=150); plt.close(fig)

aus = pd.DataFrame({
    "treino_%": treino.drop(columns=ALVO).isna().mean() * 100,
    "teste_%": teste.isna().mean() * 100,
}).round(2)
log("\nValores ausentes (%):\n" + aus.to_string())
n_aus_linha = treino.drop(columns=ALVO).isna().sum(axis=1)
log(f"Linhas com ao menos um ausente: {(n_aus_linha > 0).mean():.1%}; "
    f"com 2 ou mais: {(n_aus_linha > 1).mean():.1%}")

fig, ax = plt.subplots(figsize=(7, 4))
aus.drop(index="PassengerId").plot.barh(ax=ax, color=["#2a6fbb", "#e08a2c"])
ax.set_xlabel("% de valores ausentes")
ax.set_title("Valores ausentes por coluna")
fig.tight_layout(); fig.savefig(GRAFICOS / "02_ausentes.png", dpi=150); plt.close(fig)

d = treino.copy()
d["Grupo"] = d.PassengerId.str[:4]
d["TamGrupo"] = d.groupby("Grupo").PassengerId.transform("count")
cab = d.Cabin.str.split("/", expand=True)
d["Deck"], d["CabineNum"], d["Side"] = cab[0], pd.to_numeric(cab[1]), cab[2]
d["GastoTotal"] = d[GASTOS].sum(axis=1, min_count=1)

log("\nTaxa de Transported por categoria:")
cats = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axes.ravel(), cats):
    tab = d.groupby(c)[ALVO].agg(["mean", "size"])
    log(f"  {c}: " + "; ".join(f"{k}={m:.2f} (n={int(n)})" for k, (m, n) in tab.iterrows()))
    ax.bar(tab.index.astype(str), tab["mean"], color="#2a6fbb")
    ax.axhline(taxa, ls="--", c="k", lw=1)
    for i, (m, n) in enumerate(tab.values):
        ax.text(i, m + 0.02, f"n={int(n)}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title(c); ax.set_ylabel("Taxa de Transported")
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Taxa de transportados por categoria (tracejado: taxa geral)")
fig.tight_layout(); fig.savefig(GRAFICOS / "03_categoricas_vs_alvo.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(d, x="Age", hue=ALVO, bins=40, multiple="layer", ax=axes[0])
axes[0].set_title("Idade por classe")
faixa = pd.cut(d.Age, [-1, 4, 12, 17, 25, 35, 50, 80])
tab = d.groupby(faixa, observed=True)[ALVO].agg(["mean", "size"])
axes[1].bar(tab.index.astype(str), tab["mean"], color="#2a6fbb")
axes[1].axhline(taxa, ls="--", c="k", lw=1)
axes[1].set_ylabel("Taxa de Transported"); axes[1].set_title("Taxa por faixa etária")
fig.tight_layout(); fig.savefig(GRAFICOS / "04_idade.png", dpi=150); plt.close(fig)
log("\nTaxa por faixa etária:\n" + tab.round(3).to_string())

log("\nGastos: % de zeros, assimetria e média por classe")
for g in GASTOS + ["GastoTotal"]:
    s = d[g].dropna()
    m = d.groupby(ALVO)[g].mean()
    log(f"  {g}: zeros={np.mean(s == 0):.1%}, skew={s.skew():.1f}, max={s.max():.0f}, "
        f"média False={m[False]:.0f}, True={m[True]:.0f}")
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, g in zip(axes.ravel(), GASTOS + ["GastoTotal"]):
    sns.histplot(d.assign(v=np.log1p(d[g])), x="v", hue=ALVO, bins=30, ax=ax, element="step")
    ax.set_xlabel(f"log(1 + {g})"); ax.set_title(g)
fig.suptitle("Gastos a bordo (escala log1p) por classe")
fig.tight_layout(); fig.savefig(GRAFICOS / "05_gastos.png", dpi=150); plt.close(fig)

sem_gasto = d.GastoTotal == 0
log(f"Taxa de Transported com gasto total = 0: {d.loc[sem_gasto, ALVO].mean():.3f}; "
    f"com gasto > 0: {d.loc[d.GastoTotal > 0, ALVO].mean():.3f}")

cryo = d.CryoSleep == True
log(f"\nSoma dos gastos dos passageiros em CryoSleep: {d.loc[cryo, GASTOS].sum().sum():.0f}")
log(f"Soma dos gastos de menores de 13 anos: {d.loc[d.Age < 13, GASTOS].sum().sum():.0f}")
log(f"CryoSleep ausente e gasto > 0: {(d.CryoSleep.isna() & (d.GastoTotal > 0)).sum()} linhas")

for c in ["HomePlanet", "Side", "Destination", "Deck"]:
    n = d.dropna(subset=[c]).groupby("Grupo")[c].nunique()
    log(f"Grupos com mais de um valor de {c}: {(n > 1).mean():.1%}")
log(f"Grupos em comum entre treino e teste: "
    f"{len(set(d.Grupo) & set(teste.PassengerId.str[:4]))}")
multi = d[d.TamGrupo > 1].groupby("Grupo")[ALVO].mean()
log(f"Grupos com 2+ pessoas em que todos têm o mesmo alvo: {multi.isin([0, 1]).mean():.1%}")
tab = d.groupby("TamGrupo")[ALVO].agg(["mean", "size"])
log("\nTaxa por tamanho do grupo:\n" + tab.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(tab.index.astype(str), tab["mean"], color="#2a6fbb")
axes[0].axhline(taxa, ls="--", c="k", lw=1)
axes[0].set_xlabel("Tamanho do grupo"); axes[0].set_ylabel("Taxa de Transported")
axes[0].set_title("Taxa por tamanho do grupo de viagem")
bins = d.dropna(subset=["CabineNum", "Side"]).assign(faixa=lambda x: (x.CabineNum // 100) * 100)
tb = bins.groupby(["faixa", "Side"])[ALVO].mean().unstack()
tb.plot(ax=axes[1], marker="o")
axes[1].axhline(taxa, ls="--", c="k", lw=1)
axes[1].set_xlabel("Número da cabine (faixas de 100)"); axes[1].set_ylabel("Taxa de Transported")
axes[1].set_title("Taxa por número e lado da cabine")
fig.tight_layout(); fig.savefig(GRAFICOS / "06_grupo_cabine.png", dpi=150); plt.close(fig)

num = d[["Age", "TamGrupo", "CabineNum"]].copy()
for g in GASTOS + ["GastoTotal"]:
    num[g] = np.log1p(d[g])
num["CryoSleep"] = d.CryoSleep.map({True: 1, False: 0})
num["VIP"] = d.VIP.map({True: 1, False: 0})
num[ALVO] = d[ALVO].astype(int)
corr = num.corr()
log("\nCorrelação com o alvo:\n" + corr[ALVO].drop(ALVO).sort_values().round(3).to_string())
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlação de Pearson (gastos em log1p)")
fig.tight_layout(); fig.savefig(GRAFICOS / "07_correlacao.png", dpi=150); plt.close(fig)

(RESULTADOS / "eda_resumo.txt").write_text("\n".join(linhas), encoding="utf-8")
print("\nEDA concluída.")
