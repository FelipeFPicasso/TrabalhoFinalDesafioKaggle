import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score

from src.avaliacao import linha_resultado, mcnemar_exato
from src.modelos import (BASES, ENSEMBLE, KNN, MLP, MLP_LIMIAR, STACKING, SVM,
                         hiperparametros_salvos, novo_classificador, pipe, reconstruir)
from src.preparacao import (GRAFICOS, RESULTADOS, SEED, SUBMISSOES, preparar_dados,
                            validacao_agrupada)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(SEED)

treino, teste, X, y, grupos, X_teste = preparar_dados()
idx_tr, idx_val = next(validacao_agrupada().split(X, y, grupos))
X_tr, y_tr, g_tr = X.iloc[idx_tr], y.iloc[idx_tr], grupos[idx_tr]
X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
ytr, yval = y_tr.values, y_val.values
cv_interna = validacao_agrupada()
dobras = list(cv_interna.split(X_tr, y_tr, g_tr))

salvos = hiperparametros_salvos()
res_orig = pd.read_csv(RESULTADOS / "resultados.csv").set_index("tecnica")
melhor_orig = res_orig.acc_validacao.idxmax()

linhas, pred_val, prob_val = [], {}, {}


def registrar(nome, arquivo, pred_tr, pv, prob, cv_m, cv_s, t_busca, t_fit, t_pred, pt, params):
    linha = linha_resultado(nome, SUBMISSOES / arquivo, teste.PassengerId, ytr, pred_tr, yval,
                            pv, prob, cv_m, cv_s, t_busca, t_fit, t_pred, pt, params)
    linhas.append(linha)
    pred_val[nome] = np.asarray(pv)
    print(f"  treino={linha['acc_treino']:.4f}  cv={cv_m:.4f}±{cv_s:.4f}  "
          f"validação={linha['acc_validacao']:.4f} [{linha['ic95_inf']:.3f}; "
          f"{linha['ic95_sup']:.3f}]  -> {arquivo}")


def proba_oof(modelo, Xa, ya, ga):
    return cross_val_predict(modelo, Xa, ya, groups=ga, cv=cv_interna, n_jobs=-1,
                             method="predict_proba")[:, 1]


print("Ajustando os 4 modelos-base com os hiperparâmetros do 02_modelos.py...")
base_tr, p_val = {}, {}
for n in BASES:
    m = reconstruir(n, salvos[n])
    if n == SVM:
        m.set_params(clf__probability=True)
    base_tr[n] = m.fit(X_tr, y_tr)
    p_val[n] = m.predict_proba(X_val)[:, 1]
    pred_val[n] = m.predict(X_val)
    dif = abs(accuracy_score(y_val, pred_val[n]) - res_orig.loc[n, "acc_validacao"])
    if dif > 1e-9:
        print(f"  AVISO: {n} não reproduziu o 02 (diferença {dif:.4f}). "
              f"Rode o 02_modelos.py de novo antes deste script.")
pred_val[ENSEMBLE] = np.mean([p_val[n] for n in BASES], axis=0) > 0.5

print("Gerando probabilidades fora da dobra (OOF) no treino...")
oof = {n: proba_oof(base_tr[n], X_tr, y_tr, g_tr) for n in BASES}

print(f"\n=== {KNN}")
grade_knn = {"clf__n_neighbors": [5, 9, 15, 25, 35, 51, 75, 101],
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]}
busca = GridSearchCV(pipe(novo_classificador(KNN)), grade_knn, cv=cv_interna,
                     scoring="accuracy", n_jobs=-1)
t0 = time.perf_counter(); busca.fit(X_tr, y_tr, groups=g_tr); t_busca = time.perf_counter() - t0
i = busca.best_index_
print(f"  melhores hiperparâmetros: {busca.best_params_}  ({t_busca:.1f}s de busca)")
knn = busca.best_estimator_
prob_val[KNN] = knn.predict_proba(X_val)[:, 1]
final = clone(knn)
t0 = time.perf_counter(); final.fit(X, y); t_fit = time.perf_counter() - t0
t0 = time.perf_counter(); pt = final.predict(X_teste); t_pred = time.perf_counter() - t0
registrar(KNN, "06_knn.csv", knn.predict(X_tr), knn.predict(X_val), prob_val[KNN],
          busca.cv_results_["mean_test_score"][i], busca.cv_results_["std_test_score"][i],
          t_busca, t_fit, t_pred, pt, busca.best_params_)

print(f"\n=== {MLP_LIMIAR}")
LIMIARES = np.round(np.arange(0.30, 0.701, 0.01), 2)


def acc_por_limiar(p, yy):
    return np.array([np.mean((p >= t) == yy) for t in LIMIARES])


def escolher_limiar(p, yy):
    a = acc_por_limiar(p, yy)
    suave = np.convolve(np.pad(a, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return float(LIMIARES[np.argmax(suave)])


t0 = time.perf_counter()
p_oof_mlp = oof[MLP]
limiar = escolher_limiar(p_oof_mlp, ytr)
t_busca_lim = time.perf_counter() - t0
print(f"  limiar escolhido por CV: {limiar:.2f} (padrão: 0,50)")

acc_lim, acc_05 = [], []
for _, te in dobras:
    resto = np.setdiff1d(np.arange(len(ytr)), te)
    t_k = escolher_limiar(p_oof_mlp[resto], ytr[resto])
    acc_lim.append(np.mean((p_oof_mlp[te] >= t_k) == ytr[te]))
    acc_05.append(np.mean((p_oof_mlp[te] >= 0.5) == ytr[te]))
print(f"  CV por dobra: limiar 0,50 -> {np.mean(acc_05):.4f} | limiar escolhido -> "
      f"{np.mean(acc_lim):.4f}")

pv_antes, pv_depois = p_val[MLP] >= 0.5, p_val[MLP] >= limiar
final_mlp = clone(base_tr[MLP])
t0 = time.perf_counter(); final_mlp.fit(X, y); t_fit = time.perf_counter() - t0
t0 = time.perf_counter(); p_teste_mlp = final_mlp.predict_proba(X_teste)[:, 1]
t_pred = time.perf_counter() - t0
pt = p_teste_mlp >= limiar
registrar(MLP_LIMIAR, "07_mlp_limiar.csv",
          base_tr[MLP].predict_proba(X_tr)[:, 1] >= limiar, pv_depois, p_val[MLP],
          np.mean(acc_lim), np.std(acc_lim), t_busca_lim, t_fit, t_pred, pt,
          {"limiar": limiar, **salvos[MLP]})

ad = pd.DataFrame([
    dict(versao="antes (limiar 0,50)", limiar=0.5, acc_validacao=np.mean(pv_antes == yval),
         recall_False=recall_score(yval, pv_antes, pos_label=0),
         recall_True=recall_score(yval, pv_antes, pos_label=1),
         precisao_True=precision_score(yval, pv_antes, zero_division=0),
         pct_True_validacao=pv_antes.mean(), pct_True_teste=(p_teste_mlp >= 0.5).mean()),
    dict(versao=f"depois (limiar {limiar:.2f})", limiar=limiar,
         acc_validacao=np.mean(pv_depois == yval),
         recall_False=recall_score(yval, pv_depois, pos_label=0),
         recall_True=recall_score(yval, pv_depois, pos_label=1),
         precisao_True=precision_score(yval, pv_depois, zero_division=0),
         pct_True_validacao=pv_depois.mean(), pct_True_teste=pt.mean())])
ad.to_csv(RESULTADOS / "mlp_limiar_antes_depois.csv", index=False)
print("\n" + ad.round(4).to_string(index=False))
print(f"  (taxa real de True: treino {ytr.mean():.3f}, validação {yval.mean():.3f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
a = acc_por_limiar(p_oof_mlp, ytr)
axes[0].plot(LIMIARES, a, "o-", ms=3, alpha=0.6, label="OOF no treino (bruta)")
axes[0].plot(LIMIARES, np.convolve(np.pad(a, 2, mode="edge"), np.ones(5) / 5, mode="valid"),
             lw=2, label="OOF no treino (suavizada)")
axes[0].axvline(0.5, c="gray", ls=":", label="Limiar padrão 0,50")
axes[0].axvline(limiar, c="k", ls="--", label=f"Limiar escolhido {limiar:.2f}")
axes[0].set_xlabel("Limiar de decisão"); axes[0].set_ylabel("Acurácia (maior é melhor)")
axes[0].set_title("MLP: acurácia na CV por limiar"); axes[0].legend(fontsize=8)
for rot, p_ in [("OOF no treino", p_oof_mlp), ("Validação", p_val[MLP]),
                ("Teste Kaggle", p_teste_mlp)]:
    axes[1].plot(LIMIARES, [(p_ >= t).mean() for t in LIMIARES], label=rot)
axes[1].axhline(ytr.mean(), c="gray", ls=":", label="% real de True no treino")
axes[1].axvline(limiar, c="k", ls="--")
axes[1].set_xlabel("Limiar de decisão"); axes[1].set_ylabel("Fração prevista como True")
axes[1].set_title("MLP: quantos 'True' o limiar produz"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(GRAFICOS / "17_limiar_mlp.png", dpi=150); plt.close(fig)

print(f"\n=== {STACKING}")
Z_tr = np.column_stack([oof[n] for n in BASES])
Z_val = np.column_stack([p_val[n] for n in BASES])
meta = LogisticRegression(max_iter=1000, random_state=SEED)
cv_meta = cross_val_score(meta, Z_tr, y_tr, groups=g_tr, cv=cv_interna, scoring="accuracy")
meta.fit(Z_tr, y_tr)
coef = pd.Series(meta.coef_[0], index=BASES)
coef["(intercepto)"] = meta.intercept_[0]
coef.to_csv(RESULTADOS / "stacking_coeficientes.csv", header=["coeficiente"])
print("  pesos do meta-modelo:\n" + coef.round(3).to_string())

prob_val[STACKING] = meta.predict_proba(Z_val)[:, 1]
Z_ins = np.column_stack([base_tr[n].predict_proba(X_tr)[:, 1] for n in BASES])

t0 = time.perf_counter()
base_full = {n: clone(base_tr[n]).fit(X, y) for n in BASES}
Z_full = np.column_stack([proba_oof(base_tr[n], X, y, grupos) for n in BASES])
meta_full = LogisticRegression(max_iter=1000, random_state=SEED).fit(Z_full, y)
t_fit = time.perf_counter() - t0
t0 = time.perf_counter()
Z_teste = np.column_stack([base_full[n].predict_proba(X_teste)[:, 1] for n in BASES])
pt = meta_full.predict(Z_teste)
t_pred = time.perf_counter() - t0
registrar(STACKING, "08_stacking.csv", meta.predict(Z_ins), meta.predict(Z_val),
          prob_val[STACKING], cv_meta.mean(), cv_meta.std(), np.nan, t_fit, t_pred, pt,
          {"meta": "LogisticRegression", "entradas": BASES, "oof": "CV agrupada de 5 dobras"})

extras = pd.DataFrame(linhas)
extras.to_csv(RESULTADOS / "resultados_extras.csv", index=False)
print("\n" + extras[["tecnica", "acc_treino", "acc_cv_media", "acc_validacao",
                     "auc_validacao"]].round(4).to_string(index=False))

pares = [(MLP_LIMIAR, MLP), (STACKING, ENSEMBLE), (KNN, melhor_orig),
         (MLP_LIMIAR, melhor_orig), (STACKING, melhor_orig)]
vistos, linhas_mc = set(), []
for a_, b_ in pares:
    if a_ == b_ or (a_, b_) in vistos:
        continue
    vistos.add((a_, b_))
    n01, n10, p_ = mcnemar_exato(pred_val[a_] == yval, pred_val[b_] == yval)
    linhas_mc.append(dict(comparacao=f"{a_} vs {b_}", so_primeiro_acerta=n01,
                          so_segundo_acerta=n10, p_valor=p_))
pd.DataFrame(linhas_mc).to_csv(RESULTADOS / "mcnemar_extras.csv", index=False)
print("\nMcNemar:\n" + pd.DataFrame(linhas_mc).round(4).to_string(index=False))

pd.DataFrame({"PassengerId": treino.PassengerId.iloc[idx_val].values, "real": yval,
              **pred_val}).to_csv(RESULTADOS / "predicoes_validacao_extras.csv", index=False)
print("\nConcluído (extras). Submissões 06, 07 e 08 em submissoes/.")
