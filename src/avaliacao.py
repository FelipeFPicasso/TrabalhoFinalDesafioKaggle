import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.preparacao import SEED


def slug(nome):
    s = nome.lower().split(" (")[0] if not nome.startswith("Baseline") else "baseline"
    for a, b in [("ã", "a"), ("í", "i"), ("é", "e"), (" ", "_")]:
        s = s.replace(a, b)
    return s


def linha_resultado(nome, arquivo, ids_teste, y_tr, pred_tr, y_val, pred_val, prob_val,
                    cv_media, cv_dp, tempo_busca, t_fit, t_pred, pred_teste, params):
    acc_tr = accuracy_score(y_tr, pred_tr)
    acc_val = accuracy_score(y_val, pred_val)
    auc_val = roc_auc_score(y_val, prob_val) if prob_val is not None else np.nan
    rng = np.random.default_rng(SEED)
    acertos = (np.asarray(pred_val) == np.asarray(y_val)).astype(float)
    boot = [acertos[rng.integers(0, len(acertos), len(acertos))].mean() for _ in range(2000)]
    pd.DataFrame({"PassengerId": ids_teste, "Transported": np.asarray(pred_teste).astype(bool)}) \
        .to_csv(arquivo, index=False)
    return dict(tecnica=nome, acc_treino=acc_tr, acc_cv_media=cv_media, acc_cv_dp=cv_dp,
                acc_validacao=acc_val, ic95_inf=np.percentile(boot, 2.5),
                ic95_sup=np.percentile(boot, 97.5), auc_validacao=auc_val,
                tempo_busca_s=tempo_busca, tempo_ajuste_final_s=t_fit,
                tempo_previsao_teste_s=t_pred, taxa_true_teste=np.mean(pred_teste),
                submissao=arquivo.name, params=json.dumps(params, default=str))


def mcnemar_exato(acerta_a, acerta_b):
    n01 = int(np.sum(acerta_a & ~acerta_b))
    n10 = int(np.sum(~acerta_a & acerta_b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p
