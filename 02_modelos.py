import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score,
                                     learning_curve)
from sklearn.neural_network import MLPClassifier

from src.avaliacao import linha_resultado, mcnemar_exato, slug
from src.modelos import BASELINE, ENSEMBLE, LGBM, LR, MLP, SVM, novo_classificador, pipe
from src.preparacao import (GRAFICOS, RESULTADOS, SEED, SUBMISSOES, preparar_dados,
                            validacao_agrupada)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(SEED)

treino, teste, X, y, grupos, X_teste = preparar_dados()

idx_tr, idx_val = next(validacao_agrupada().split(X, y, grupos))
X_tr, y_tr, g_tr = X.iloc[idx_tr], y.iloc[idx_tr], grupos[idx_tr]
X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
assert not set(g_tr) & set(grupos[idx_val]), "grupo dividido entre treino e validação"
print(f"Treino: {len(idx_tr)} | Validação: {len(idx_val)} | Teste Kaggle: {len(X_teste)}")
print(f"Taxa de positivos -> treino {y_tr.mean():.3f}, validação {y_val.mean():.3f}")

cv_interna = validacao_agrupada()

TECNICAS = [
    (BASELINE, pipe(novo_classificador(BASELINE)), {}, "grade"),
    (LR, pipe(novo_classificador(LR)),
     {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
      "clf__penalty": ["l1", "l2"]}, "grade"),
    (LGBM, pipe(novo_classificador(LGBM)),
     {"clf__n_estimators": [100, 200, 400, 800],
      "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
      "clf__num_leaves": [7, 15, 31, 63],
      "clf__min_child_samples": [10, 20, 50, 100],
      "clf__subsample": [0.7, 0.85, 1.0], "clf__subsample_freq": [1],
      "clf__colsample_bytree": [0.5, 0.7, 1.0],
      "clf__reg_lambda": [0, 1, 5, 10]}, "aleatoria"),
    (SVM, pipe(novo_classificador(SVM)),
     {"clf__C": [0.1, 0.3, 1, 3, 10, 30],
      "clf__gamma": [0.003, 0.01, 0.03, 0.1, 0.3]}, "grade"),
    (MLP, pipe(novo_classificador(MLP)),
     {"clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
      "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
      "clf__learning_rate_init": [1e-3, 3e-3]}, "grade"),
]

resultados, melhores, buscas, prob_val, pred_val = [], {}, {}, {}, {}


def avaliar(nome, modelo_tr, tempo_busca, cv_media, cv_dp, params):
    pv = modelo_tr.predict(X_val)
    pred_val[nome] = pv
    prob = None
    if hasattr(modelo_tr, "predict_proba") and not nome.startswith("Baseline"):
        prob_val[nome] = prob = modelo_tr.predict_proba(X_val)[:, 1]
    final = clone(modelo_tr)
    t0 = time.perf_counter(); final.fit(X, y); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); pt = final.predict(X_teste); t_pred = time.perf_counter() - t0
    arquivo = SUBMISSOES / f"{len(resultados):02d}_{slug(nome)}.csv"
    linha = linha_resultado(nome, arquivo, teste.PassengerId, y_tr, modelo_tr.predict(X_tr),
                            y_val, pv, prob, cv_media, cv_dp, tempo_busca, t_fit, t_pred,
                            pt, params)
    resultados.append(linha)
    print(f"  treino={linha['acc_treino']:.4f}  cv={cv_media:.4f}±{cv_dp:.4f}  "
          f"validação={linha['acc_validacao']:.4f} "
          f"[{linha['ic95_inf']:.3f}; {linha['ic95_sup']:.3f}]  -> {arquivo.name}")
    return final


for nome, est, grade, tipo in TECNICAS:
    print(f"\n=== {nome}")
    if tipo == "aleatoria":
        busca = RandomizedSearchCV(est, grade, n_iter=60, cv=cv_interna, scoring="accuracy",
                                   n_jobs=-1, random_state=SEED, return_train_score=True)
    else:
        busca = GridSearchCV(est, grade or {}, cv=cv_interna, scoring="accuracy", n_jobs=-1,
                             return_train_score=True)
    t0 = time.perf_counter()
    busca.fit(X_tr, y_tr, groups=g_tr)
    t_busca = time.perf_counter() - t0
    i = busca.best_index_
    cv_m, cv_s = busca.cv_results_["mean_test_score"][i], busca.cv_results_["std_test_score"][i]
    print(f"  melhores hiperparâmetros: {busca.best_params_}  ({t_busca:.1f}s de busca)")
    modelo = busca.best_estimator_
    if nome.startswith("SVM"):
        modelo = clone(modelo).set_params(clf__probability=True).fit(X_tr, y_tr)
    melhores[nome] = modelo
    buscas[nome] = busca
    avaliar(nome, modelo, t_busca, cv_m, cv_s, busca.best_params_)

nome = ENSEMBLE
print(f"\n=== {nome}")
membros = [(slug(n), clone(m)) for n, m in melhores.items() if not n.startswith("Baseline")]
ens = VotingClassifier(membros, voting="soft", n_jobs=-1)
t0 = time.perf_counter()
cv_ens = cross_val_score(ens, X_tr, y_tr, groups=g_tr, cv=cv_interna, scoring="accuracy")
ens.fit(X_tr, y_tr)
t_ens = time.perf_counter() - t0
melhores[nome] = ens
avaliar(nome, ens, t_ens, cv_ens.mean(), cv_ens.std(), {"membros": [m for m, _ in membros],
                                                        "pesos": "iguais"})

res = pd.DataFrame(resultados)
res.to_csv(RESULTADOS / "resultados.csv", index=False)
print("\n" + res[["tecnica", "acc_treino", "acc_cv_media", "acc_validacao", "auc_validacao",
                  "tempo_busca_s", "tempo_ajuste_final_s"]].round(4).to_string(index=False))

with open(RESULTADOS / "relatorio_classificacao.txt", "w", encoding="utf-8") as f:
    for n, pv in pred_val.items():
        f.write(f"===== {n}\n")
        f.write(classification_report(y_val, pv, target_names=["False", "True"], digits=4,
                                      zero_division=0))
        f.write(f"Matriz de confusão (linhas=real, colunas=previsto):\n"
                f"{confusion_matrix(y_val, pv)}\n\n")

melhor = res.loc[res.acc_validacao.idxmax(), "tecnica"]
linhas_mc = []
for n, pv in pred_val.items():
    if n == melhor:
        continue
    n01, n10, p = mcnemar_exato(pred_val[melhor] == y_val.values, pv == y_val.values)
    linhas_mc.append(dict(comparacao=f"{melhor} vs {n}", so_melhor_acerta=n01,
                          so_outro_acerta=n10, p_valor=p))
pd.DataFrame(linhas_mc).to_csv(RESULTADOS / "mcnemar_validacao.csv", index=False)
print("\nMcNemar:\n" + pd.DataFrame(linhas_mc).round(4).to_string(index=False))

nomes_mod = [n for n in pred_val if not n.startswith("Baseline")]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, n in zip(axes.ravel(), pred_val):
    cm = confusion_matrix(y_val, pred_val[n])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["False", "True"], yticklabels=["False", "True"])
    ax.set_title(f"{n}\nacurácia = {accuracy_score(y_val, pred_val[n]):.4f}", fontsize=10)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
fig.suptitle(f"Matrizes de confusão na validação (n = {len(y_val)})")
fig.tight_layout(); fig.savefig(GRAFICOS / "10_matrizes_confusao.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
for n in nomes_mod:
    fpr, tpr, _ = roc_curve(y_val, prob_val[n])
    ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_val, prob_val[n]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
ax.set_xlabel("Taxa de falsos positivos"); ax.set_ylabel("Taxa de verdadeiros positivos")
ax.set_title("Curvas ROC na validação"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(GRAFICOS / "11_roc.png", dpi=150); plt.close(fig)

cvr = lambda n: pd.DataFrame(buscas[n].cv_results_)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
d = cvr(LR)
for pen, g in d.groupby("param_clf__penalty"):
    g = g.sort_values("param_clf__C")
    axes[0].errorbar(g.param_clf__C.astype(float), g.mean_test_score, g.std_test_score,
                     marker="o", capsize=3, label=f"{pen} (validação CV)")
    axes[0].plot(g.param_clf__C.astype(float), g.mean_train_score, ls=":",
                 label=f"{pen} (treino)")
axes[0].set_xscale("log"); axes[0].set_xlabel("C (inverso da regularização)")
axes[0].set_ylabel("Acurácia (maior é melhor)"); axes[0].legend(fontsize=8)
axes[0].set_title("Regressão Logística")
d = cvr(SVM)
piv = d.pivot_table(index="param_clf__C", columns="param_clf__gamma", values="mean_test_score")
sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=axes[1], annot_kws={"size": 8})
axes[1].set_title("SVM RBF — acurácia média na CV"); axes[1].set_xlabel("gamma")
axes[1].set_ylabel("C")
d = cvr(MLP)
d = d[d.param_clf__learning_rate_init == buscas[MLP].best_params_["clf__learning_rate_init"]]
piv = d.pivot_table(index=d.param_clf__hidden_layer_sizes.astype(str),
                    columns="param_clf__alpha", values="mean_test_score")
sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=axes[2], annot_kws={"size": 8})
axes[2].set_title("MLP — acurácia média na CV"); axes[2].set_xlabel("alpha (L2)")
axes[2].set_ylabel("Camadas ocultas")
fig.tight_layout(); fig.savefig(GRAFICOS / "12_busca_hiperparametros.png", dpi=150)
plt.close(fig)

d = cvr(LGBM)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
sc = axes[0].scatter(d.mean_train_score, d.mean_test_score,
                     c=np.log10(d.param_clf__num_leaves.astype(float)), cmap="viridis")
axes[0].plot([0.78, 0.95], [0.78, 0.95], "k--", lw=1)
axes[0].set_xlabel("Acurácia de treino (CV)"); axes[0].set_ylabel("Acurácia de validação (CV)")
axes[0].set_title("LightGBM: 60 configurações sorteadas")
cb = fig.colorbar(sc, ax=axes[0]); cb.set_label("log10(num_leaves)")
imp = pd.Series(melhores[LGBM].named_steps["clf"].booster_.feature_importance("gain"),
                index=melhores[LGBM].named_steps["prep"].get_feature_names_out())
imp = (imp / imp.sum()).sort_values().tail(15)
axes[1].barh(imp.index, imp.values, color="#2a6fbb")
axes[1].set_xlabel("Fração do ganho total"); axes[1].set_title("LightGBM: 15 atributos mais importantes")
fig.tight_layout(); fig.savefig(GRAFICOS / "13_lightgbm.png", dpi=150); plt.close(fig)
imp.to_csv(RESULTADOS / "lightgbm_importancia.csv")

lr = melhores[LR]
coef = pd.Series(lr.named_steps["clf"].coef_[0],
                 index=lr.named_steps["prep"].get_feature_names_out())
coef.sort_values().to_csv(RESULTADOS / "regressao_logistica_coeficientes.csv")
top = coef.reindex(coef.abs().sort_values().tail(18).index)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top.index, top.values, color=np.where(top.values > 0, "#2a6fbb", "#c44e52"))
ax.axvline(0, c="k", lw=1)
ax.set_xlabel("Coeficiente (log-odds de Transported por desvio-padrão / categoria)")
ax.set_title("Regressão Logística: 18 maiores coeficientes em módulo")
fig.tight_layout(); fig.savefig(GRAFICOS / "14_lr_coeficientes.png", dpi=150); plt.close(fig)

mlp_best = melhores[MLP]
prep = clone(mlp_best.named_steps["prep"]).fit(X_tr)
Xt_tr, Xt_val = prep.transform(X_tr), prep.transform(X_val)
p = {k.replace("clf__", ""): v for k, v in buscas[MLP].best_params_.items()}
mlp = MLPClassifier(random_state=SEED, **p)
hist = []
for ep in range(1, 301):
    mlp.partial_fit(Xt_tr, y_tr, classes=[0, 1])
    hist.append(dict(epoca=ep,
                     perda_treino=log_loss(y_tr, mlp.predict_proba(Xt_tr)),
                     perda_validacao=log_loss(y_val, mlp.predict_proba(Xt_val)),
                     acc_treino=accuracy_score(y_tr, mlp.predict(Xt_tr)),
                     acc_validacao=accuracy_score(y_val, mlp.predict(Xt_val))))
hist = pd.DataFrame(hist); hist.to_csv(RESULTADOS / "mlp_historico.csv", index=False)
ep_es = mlp_best.named_steps["clf"].n_iter_
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(hist.epoca, hist.perda_treino, label="Treino")
axes[0].plot(hist.epoca, hist.perda_validacao, label="Validação")
axes[0].set_ylabel("Log loss (menor é melhor)")
axes[1].plot(hist.epoca, hist.acc_treino, label="Treino")
axes[1].plot(hist.epoca, hist.acc_validacao, label="Validação")
axes[1].set_ylabel("Acurácia (maior é melhor)")
for ax in axes:
    ax.axvline(ep_es, c="k", ls="--", lw=1, label=f"Parada antecipada ({ep_es} épocas)")
    ax.set_xlabel("Época"); ax.legend()
fig.suptitle(f"MLP {p['hidden_layer_sizes']}, alpha={p['alpha']}: curvas por época")
fig.tight_layout(); fig.savefig(GRAFICOS / "15_mlp_curvas.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
curvas = []
for ax, n in zip(axes, [k for k in nomes_mod if not k.startswith("Ensemble")]):
    est = clone(melhores[n])
    if n.startswith("SVM"):
        est.set_params(clf__probability=False)
    tam, s_tr, s_te = learning_curve(est, X_tr, y_tr, groups=g_tr, cv=cv_interna,
                                     train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
                                     scoring="accuracy", n_jobs=-1, shuffle=True,
                                     random_state=SEED)
    for t, a, b in zip(tam, s_tr.mean(1), s_te.mean(1)):
        curvas.append(dict(tecnica=n, n_treino=t, acc_treino=a, acc_validacao=b))
    ax.plot(tam, s_tr.mean(1), "o-", label="Treino")
    ax.fill_between(tam, s_te.mean(1) - s_te.std(1), s_te.mean(1) + s_te.std(1), alpha=0.2)
    ax.plot(tam, s_te.mean(1), "o-", label="Validação (CV)")
    ax.set_title(n, fontsize=10); ax.set_xlabel("Exemplos de treino")
axes[0].set_ylabel("Acurácia (maior é melhor)"); axes[0].legend()
fig.suptitle("Curvas de aprendizado (CV agrupada de 5 dobras na parte de treino)")
fig.tight_layout(); fig.savefig(GRAFICOS / "16_curvas_aprendizado.png", dpi=150); plt.close(fig)
pd.DataFrame(curvas).to_csv(RESULTADOS / "curvas_aprendizado.csv", index=False)

pd.DataFrame({"PassengerId": treino.PassengerId.iloc[idx_val].values, "real": y_val.values,
              **{n: v for n, v in pred_val.items()}}) \
    .to_csv(RESULTADOS / "predicoes_validacao.csv", index=False)
print("\nConcluído. Submissões em submissoes/, resultados em resultados/, figuras em graficos/.")
