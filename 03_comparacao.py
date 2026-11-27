import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preparacao import BASE, GRAFICOS, RESULTADOS

res = pd.read_csv(RESULTADOS / "resultados.csv")
arq_extras = RESULTADOS / "resultados_extras.csv"
if arq_extras.exists():
    res = pd.concat([res, pd.read_csv(arq_extras)], ignore_index=True)

arq_kaggle = BASE / "kaggle_scores.csv"
if arq_kaggle.exists():
    kag = pd.read_csv(arq_kaggle, dtype={"kaggle_publico": float, "kaggle_privado": float})
    res = res.merge(kag[["submissao", "kaggle_publico", "kaggle_privado"]], on="submissao",
                    how="left")
else:
    res["kaggle_publico"] = np.nan
    res["kaggle_privado"] = np.nan

res["gap_treino_validacao"] = res.acc_treino - res.acc_validacao
res["gap_validacao_kaggle"] = res.acc_validacao - res.kaggle_publico


def fmt(v, casas=4):
    return "—" if pd.isna(v) else f"{v:.{casas}f}".replace(".", ",")


linhas = ["| Técnica | Treino | CV (média ± dp) | Validação | IC 95% val. | AUC val. | "
          "Kaggle público | Ajuste (s) |",
          "|:-----------------|-------:|-------------:|-------:|-----------:|------:|"
          "-------:|------:|"]
for _, r in res.iterrows():
    linhas.append(
        f"| {r.tecnica} | {fmt(r.acc_treino, 3)} | {fmt(r.acc_cv_media, 3)} ± {fmt(r.acc_cv_dp, 3)} | "
        f"{fmt(r.acc_validacao, 3)} | {fmt(r.ic95_inf, 3)}–{fmt(r.ic95_sup, 3)} | "
        f"{fmt(r.auc_validacao, 3)} | {fmt(r.kaggle_publico, 4)} | "
        f"{fmt(r.tempo_ajuste_final_s, 2)} |")
tabela = "\n".join(linhas)
(RESULTADOS / "tabela_comparativa.md").write_text(tabela + "\n", encoding="utf-8")
res.to_csv(RESULTADOS / "comparacao_completa.csv", index=False)
print(tabela)

series = [("acc_treino", "Treino", "#9ecae1"), ("acc_cv_media", "CV (5 dobras)", "#4292c6"),
          ("acc_validacao", "Validação", "#08519c"), ("kaggle_publico", "Kaggle público", "#e6550d")]
tem_kaggle = res.kaggle_publico.notna().any()
if not tem_kaggle:
    series = series[:3]
x = np.arange(len(res)); w = 0.8 / len(series)
fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(res)), 5))
for i, (col, rot, cor) in enumerate(series):
    vals = res[col].values
    barras = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=rot, color=cor)
    topo = res.ic95_sup.values if col == "acc_validacao" else vals
    for b, v, t in zip(barras, vals, topo):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, t + 0.004, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=7, rotation=90)
ax.errorbar(x + (2 - (len(series) - 1) / 2) * w, res.acc_validacao,
            yerr=[res.acc_validacao - res.ic95_inf, res.ic95_sup - res.acc_validacao],
            fmt="none", ecolor="k", capsize=3, lw=1)
ax.set_xticks(x); ax.set_xticklabels([textwrap.fill(t, 16) for t in res.tecnica], fontsize=8.5)
ax.set_ylabel("Acurácia (maior é melhor)"); ax.set_ylim(0.45, 0.95)
ax.axhline(res.acc_validacao.iloc[0], c="gray", ls=":", lw=1)
ax.set_title("Acurácia por técnica: treino, validação cruzada, validação"
             + (" e Kaggle (público)" if tem_kaggle else "")
             + "\n(barras de erro: IC 95% bootstrap da validação; pontilhado: baseline)")
ax.legend(loc="upper right", ncol=len(series), fontsize=9)
fig.tight_layout(); fig.savefig(GRAFICOS / "20_comparacao.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
m = res[~res.tecnica.str.startswith("Baseline")]
ax.scatter(m.tempo_ajuste_final_s, m.acc_validacao, s=60, color="#08519c")
for _, r in m.iterrows():
    ax.annotate(r.tecnica.split(" (")[0], (r.tempo_ajuste_final_s, r.acc_validacao),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("Tempo de ajuste no treino completo (s, escala log)")
ax.set_ylabel("Acurácia na validação (maior é melhor)")
ax.set_title("Desempenho x custo computacional")
fig.tight_layout(); fig.savefig(GRAFICOS / "21_desempenho_custo.png", dpi=150); plt.close(fig)
print("\nFiguras 20_comparacao.png e 21_desempenho_custo.png geradas.")
if not tem_kaggle:
    print("AVISO: kaggle_scores.csv sem pontuações — preencha após submeter e rode de novo.")
