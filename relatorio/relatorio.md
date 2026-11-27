---
title: "Spaceship Titanic: comparação de técnicas de aprendizado supervisionado"
subtitle: "Trabalho Prático — Desafio Kaggle — Aprendizado de Máquina"
author:
  - "Felipe Picasso"
  - "Guilherme Reginato"
date: "Dezembro de 2026"
---

# Identificação

| | |
|---|---|
| **Integrantes** | Felipe Picasso; Guilherme Reginato |
| **Desafio** | Spaceship Titanic — <https://www.kaggle.com/competitions/spaceship-titanic> (grupo *Getting Started*, em andamento) |
| **Tarefa / métrica oficial** | Classificação binária de `Transported` / acurácia |
| **Código-fonte** | <https://github.com/FelipeFPicasso/TrabalhoFinalDesafioKaggle> |
| **Vídeo** | [LINK DO VÍDEO — YouTube não listado ou Google Drive] |
| **Ambiente** | Python 3.10, scikit-learn 1.7.2, LightGBM 4.7.0, pandas 2.3.3; CPU de 12 núcleos, sem GPU |

# Resumo

O desafio pede prever quais passageiros da nave *Spaceship Titanic* foram transportados para outra dimensão (`Transported`), a partir de 8 693 passageiros rotulados e 4 277 de teste. A análise exploratória mostrou que o alvo é balanceado (50,4% de positivos), que cerca de 2% de cada coluna está ausente e que os sinais mais fortes são `CryoSleep` e os gastos a bordo, que seguem regras determinísticas (quem está em criossono ou tem menos de 13 anos não gasta nada). Também mostrou que passageiros do mesmo grupo de viagem compartilham planeta de origem e lado da cabine, e que nenhum grupo aparece ao mesmo tempo no treino e no teste. Por isso, todas as técnicas foram avaliadas com o mesmo protocolo: validação estratificada e agrupada por grupo de viagem (80% treino, 20% validação, semente 42), ajuste de hiperparâmetros por validação cruzada agrupada de 5 dobras apenas no treino e a acurácia como métrica. Comparamos um baseline de classe majoritária (0,494 na validação) com Regressão Logística (0,751), LightGBM (0,786), SVM com kernel RBF (0,780), uma rede neural MLP (0,791), um k-NN (0,766) e dois ensembles, por votação suave (0,785) e por *stacking* (0,792). LightGBM, SVM, MLP e os dois ensembles ficaram estatisticamente empatados na validação (intervalos de confiança de ±2 pontos e teste de McNemar com p > 0,1), todos acima da regressão logística, que sofre de subajuste, e o k-NN ficou abaixo da MLP (p = 0,002). Dois experimentos de melhoria não trouxeram ganho: ajustar o limiar de decisão da MLP por validação cruzada (limiar 0,53) piorou a validação (0,783), e um atributo de família pelo sobrenome não melhorou os modelos de forma consistente. A importância por permutação mostrou que todos os modelos dependem sobretudo dos gastos a bordo. No Kaggle, as pontuações públicas foram [PREENCHER].

# 1. Introdução

O *Spaceship Titanic* é um desafio de introdução do Kaggle. No ano de 2912, a nave colidiu com uma anomalia espaço-temporal e cerca de metade dos passageiros foi "transportada" para outra dimensão. Para cada passageiro há dados demográficos, de cabine, de viagem e de gastos em serviços a bordo, e o objetivo é prever o valor booleano `Transported`. A avaliação oficial é a **acurácia** sobre o conjunto de teste. Como o desafio está em andamento, o Kaggle mostra apenas a pontuação pública (calculada sobre parte do teste); a pontuação privada não é divulgada enquanto a competição está aberta.

O objetivo deste trabalho não é a posição no ranking, e sim aplicar o ciclo completo de aprendizado de máquina e **comparar criticamente técnicas de famílias diferentes** sob um protocolo único: um modelo linear (Regressão Logística), um ensemble de árvores por *gradient boosting* (LightGBM), um método de margem máxima com kernel (SVM RBF), uma rede neural (MLP) e um método baseado em instâncias (k-NN), além de dois ensembles (votação suave e *stacking*) e de um teste de ajuste do limiar de decisão da MLP. O relatório segue a ordem: dados e EDA (Seção 2), preparação (Seção 3), metodologia (Seção 4), tentativas (Seção 5), comparação e discussão (Seção 6) e conclusão (Seção 7).

# 2. Dados e análise exploratória

Os arquivos `train.csv` (8 693 linhas × 14 colunas) e `test.csv` (4 277 linhas × 13 colunas) foram baixados da aba *Data* do desafio. As colunas são:

- `PassengerId` no formato `gggg_pp`: `gggg` identifica o grupo de viagem (família ou amigos) e `pp` o passageiro no grupo;
- categóricas: `HomePlanet` (Earth, Europa, Mars), `Destination` (3 exoplanetas), `Cabin` (`deck/número/lado`, com lado P ou S);
- booleanas: `CryoSleep` (passageiro em animação suspensa) e `VIP`;
- numéricas: `Age` e cinco gastos — `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`;
- `Name` (texto) e o alvo `Transported`.

Todo o código da EDA está em `01_eda.py`, e o resumo numérico completo em `resultados/eda_resumo.txt`. Os principais achados:

**A1 — Alvo balanceado.** 4 378 transportados contra 4 315 não transportados (50,36%, Figura 1). A acurácia é uma métrica adequada e não é necessário reamostrar as classes. O baseline de classe majoritária acerta cerca de 50%.

![Figura 1 — Distribuição do alvo no treino.](graficos/01_alvo.png){width=45%}

**A2 — Ausentes espalhados.** Todas as colunas, exceto `PassengerId` e o alvo, têm entre 2,1% e 2,5% de ausentes, com proporções semelhantes no teste (Figura 2). Os ausentes estão espalhados: 24% das linhas têm algum ausente, mas só 2,5% têm dois ou mais. Descartar linhas perderia um quarto dos dados, então a opção foi imputar.

![Figura 2 — Porcentagem de valores ausentes por coluna, no treino e no teste.](graficos/02_ausentes.png){width=70%}

**A3 — `CryoSleep` é o atributo isolado mais forte.** 82% dos passageiros em criossono foram transportados, contra 33% dos demais (Figura 3). `HomePlanet` (Europa 66%, Earth 42%), `Destination` (55 Cancri e 61%), o deck (B 73%, C 68%, E 36%) e o lado da cabine (S 56%, P 45%) também separam as classes. `VIP` atinge só 199 passageiros (2,3%) e tem pouco efeito.

![Figura 3 — Taxa de transportados por categoria; a linha tracejada é a taxa geral (50,4%) e *n* é o número de passageiros.](graficos/03_categoricas_vs_alvo.png)

**A4 — Gastos muito assimétricos e com efeitos de sinais opostos.** Entre 62% e 66% dos valores de cada gasto são zero, com assimetria entre 6 e 13 e máximos acima de 14 000 (Figura 4). Quem não gastou nada foi transportado em 78,6% dos casos, contra 29,9% de quem gastou. Os gastos em `RoomService`, `Spa` e `VRDeck` são bem maiores entre os não transportados (médias de 389, 564 e 544 contra 63, 62 e 69). Já `FoodCourt` e `ShoppingMall` têm média ligeiramente *maior* entre os transportados. Na escala log1p, `GastoTotal` tem correlação de −0,47 com o alvo, a mesma magnitude de `CryoSleep` (+0,47) (Figura 7).

![Figura 4 — Distribuição dos gastos em escala log(1 + x) por classe.](graficos/05_gastos.png)

**A5 — Regras determinísticas nos dados.** A soma dos gastos dos 3 037 passageiros em criossono é exatamente 0, e a dos menores de 13 anos também. Além disso, 119 passageiros têm `CryoSleep` ausente mas gastaram algo, o que implica `CryoSleep = False`.

**A6 — Idade.** Crianças de até 4 anos foram transportadas em 77% dos casos, e as de 5 a 12 anos em 61%. A partir dos 18 anos a taxa fica estável, perto de 46–49% (Figura 5). O efeito é não linear: a correlação linear de `Age` com o alvo é de apenas −0,08.

![Figura 5 — Idade por classe e taxa de transportados por faixa etária.](graficos/04_idade.png)

**A7 — Estrutura de grupos.** Em 100% dos grupos de viagem, todos os membros têm o mesmo `HomePlanet` e o mesmo lado de cabine, o que permite imputar esses ausentes a partir de outro membro do grupo. Quem viaja sozinho (55% dos passageiros) foi transportado em 45% dos casos, contra 54% a 64% em grupos de 2 a 6 pessoas (Figura 6, esquerda). **Nenhum dos 3 063 grupos do teste aparece entre os 6 217 grupos do treino.** Ou seja, o Kaggle separou treino e teste por grupo. Como passageiros do mesmo grupo são parecidos (em 43,6% dos grupos com 2 ou mais pessoas, todos têm o mesmo alvo), uma validação aleatória colocaria membros do mesmo grupo nos dois lados e daria uma estimativa otimista. Esse achado definiu o protocolo da Seção 4.

**A8 — Número da cabine.** A taxa de transportados oscila de forma não monotônica com o número da cabine, com padrões diferentes para os lados P e S (Figura 6, direita). Um modelo linear não captura essa relação diretamente.

![Figura 6 — Taxa de transportados por tamanho do grupo (esquerda) e por faixa do número da cabine e lado (direita).](graficos/06_grupo_cabine.png)

![Figura 7 — Correlação de Pearson entre atributos numéricos (gastos em log1p) e o alvo.](graficos/07_correlacao.png){width=70%}

**A9 — Sobrenome e família.** O `Name` não serve como atributo direto, mas o sobrenome poderia indicar parentesco entre grupos diferentes. Os cerca de 13 mil passageiros (treino e teste) têm só 2 406 sobrenomes distintos, e 93,8% dos passageiros do treino têm um sobrenome que aparece em dois ou mais grupos de viagem. A taxa de transportados cai de 62% (sobrenome único, n = 133) para 48% (mais de 5 pessoas com o mesmo sobrenome, n = 5 535), mas quase dois terços dos passageiros (64%) estão nessa última faixa, e a correlação com o tamanho do grupo é de apenas 0,25. Ou seja, o sobrenome parece indicar mais o quanto ele é comum do que parentesco, e ficou a dúvida de se o atributo ajudaria os modelos. Por isso ele foi avaliado numa ablação (Seção 6) e **não foi incluído** no pré-processamento.

# 3. Preparação dos dados

A preparação foi dividida em duas partes, para evitar vazamento de dados (`src/preparacao.py`):

**(a) Regras determinísticas, sem parâmetros aprendidos** (`engenharia_atributos`). Usam apenas os atributos da própria linha ou do seu grupo, nunca o alvo, e são aplicadas separadamente ao treino e ao teste. Como os grupos não se repetem entre arquivos (A7) e a validação mantém cada grupo inteiro de um lado só, o resultado é o mesmo que se seria obtido na previsão real.

| Operação | Justificativa (EDA) | Efeito |
|---|---|---|
| `Grupo`, `TamGrupo`, `Sozinho` a partir do `PassengerId` | A7 | 3 novos atributos |
| `Cabin` → `Deck`, `CabineNum`, `Side`; `CabineRegiao` = faixa de 300 números (categórica) | A3, A8 | relação não linear do número da cabine disponível também para o modelo linear |
| `HomePlanet` e `Side` ausentes ← valor de outro membro do grupo | A7 (100% constantes no grupo) | ausentes: 201 → 111 e 199 → 99 |
| Gastos ausentes ← 0 se `CryoSleep` = True ou idade < 13 | A5 | ausentes nos gastos: 943 → 537 |
| `CryoSleep` ausente ← False se gastou algo | A5 | ausentes: 217 → 98 |
| `log1p` de cada gasto e do `GastoTotal`; indicador `SemGasto` | A4 | reduz a assimetria e o peso de valores extremos (importante para LR, SVM e MLP) |
| Descartar `Name`, `PassengerId`, `Cabin`, `Grupo` | identificadores e texto livre | evita memorizar identificadores |

**(b) Transformações com parâmetros aprendidos** (`preprocessador`). Ficam dentro de um `ColumnTransformer` encadeado ao classificador em um `Pipeline`. Assim, em cada dobra da validação cruzada e na validação final, medianas, modas, médias e desvios são calculados **somente com os dados de treino daquela dobra**, conforme as boas práticas do scikit-learn [3]:

- 9 numéricas (`Age`, `TamGrupo`, `CabineNum`, 5 gastos em log e o total em log): imputação pela mediana e padronização (`StandardScaler`);
- 4 binárias (`CryoSleep`, `VIP`, `Sozinho`, `SemGasto`): imputação pela moda;
- 5 categóricas (`HomePlanet`, `Destination`, `Deck`, `Side`, `CabineRegiao`): imputação pela moda e *one-hot* (`handle_unknown="ignore"`).

O resultado são 36 atributos. O mesmo pré-processamento é usado em todas as técnicas, para que as diferenças de desempenho venham do modelo e não da preparação. A padronização não altera as árvores do LightGBM, que são invariantes a transformações monotônicas. Não foi feita remoção de outliers (a transformação log já reduz o efeito dos gastos extremos) nem reamostragem (as classes são balanceadas).

**Atributo de família (testado, não adotado).** O `src/preparacao.py` também sabe criar `TamFamilia` (número de passageiros com o mesmo sobrenome, contado em treino e teste juntos, sem usar o alvo), mas ele fica desligado por padrão (`USAR_FAMILIA = False`). A ablação da Seção 6 mostrou que ele não melhora os modelos de forma consistente, e por isso o conjunto final continua com os 36 atributos.

# 4. Metodologia experimental

**Protocolo único** (`02_modelos.py` e `02b_tentativas_extras.py`), igual para todas as técnicas:

1. **Divisão treino/validação.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; a primeira dobra (1 697 passageiros, 20%) é a validação e as demais (6 996) formam o treino. A divisão é estratificada pelo alvo e agrupada por grupo de viagem, e o código verifica que nenhum grupo fica nos dois lados. Assim, a validação reproduz a separação feita pelo Kaggle (A7).
2. **Ajuste de hiperparâmetros dentro de cada técnica.** É feito por busca em grade (`GridSearchCV`) ou aleatória (`RandomizedSearchCV`, 60 sorteios, no LightGBM), com validação cruzada **agrupada de 5 dobras somente no treino** e acurácia como critério. A validação e o ranking público do Kaggle não foram usados para escolher hiperparâmetros.
3. **Avaliação.** O melhor modelo de cada técnica, reajustado em todo o treino, é avaliado **uma única vez** na validação. Reportamos a acurácia de treino, a acurácia média ± desvio-padrão da validação cruzada (da configuração escolhida), a acurácia de validação com intervalo de confiança de 95% por *bootstrap* (2 000 reamostragens), a AUC e os tempos de execução. Diferenças entre o melhor modelo e os demais são testadas com o teste exato de McNemar nas mesmas 1 697 previsões [6].
4. **Submissão.** Com os mesmos hiperparâmetros, o modelo é reajustado nos 8 693 passageiros rotulados e prevê o `test.csv`. Cada técnica gera um arquivo em `submissoes/`, enviado ao Kaggle.

**Métrica.** Acurácia, a métrica oficial. A AUC e a precisão/revocação por classe são apenas complementares.

**Baseline.** `DummyClassifier(strategy="most_frequent")`, que prevê sempre `True` (a classe majoritária no treino). Ele confere que o alvo foi definido corretamente (`Transported`, e não um identificador) e serve de piso: qualquer modelo útil deve superar ~50%.

**Tentativas extras** (`02b_tentativas_extras.py`). Usam a mesma partição e a mesma CV e reaproveitam os hiperparâmetros já escolhidos, sem tocar na validação. O limiar da MLP e o meta-modelo do *stacking* são ajustados só com probabilidades *fora da dobra* (OOF) do treino: cada probabilidade vem de um modelo que não viu aquele passageiro.

**Reprodutibilidade.** A semente 42 é usada em todas as divisões, buscas e modelos, e o LightGBM roda com `n_jobs=1` dentro de cada ajuste. `run_all.sh` executa tudo do início ao fim e regenera todas as tabelas, figuras e submissões deste relatório (o texto em si é montado por `04_relatorio.py`).

# 5. Tentativas

As faixas completas de busca estão no Apêndice A, e as figuras de diagnóstico de cada técnica nas Figuras 8 a 13. O Apêndice A também lista os hiperparâmetros escolhidos.

## 5.1 Baseline — classe majoritária

Prevê sempre `True`: acurácia de 0,506 no treino e 0,494 na validação (a validação tem 49,4% de positivos). A matriz de confusão (Figura 8) mostra revocação 1 para `True` e 0 para `False`. Serve como referência mínima.

## 5.2 Técnica 1 — Regressão Logística (modelo linear)

**Por quê.** É o modelo linear clássico para classificação binária: rápido, com probabilidades bem calibradas e coeficientes interpretáveis. Ele indica quanto do problema é separável linearmente com os atributos construídos.

**Ajuste.** Grade de `C` ∈ {0,001, …, 100} × penalidade {L1, L2}, com solver `liblinear` (20 combinações, 5,8 s). A melhor foi **C = 0,1 com L2**. A curva da Figura 10 (esquerda) fica plana para C ≥ 0,03 (0,781–0,782 na CV), e o treino fica apenas ~0,4 ponto acima da validação cruzada em todas as regularizações: o modelo não sobreajusta, e mais flexibilidade (C maior) não ajuda. Isso indica **subajuste por viés**. A curva de aprendizado (Figura 13) confirma: treino e validação ficam próximos de 0,78–0,79 desde 558 exemplos, e mais dados não ajudariam.

**Resultado.** Treino 0,784, CV 0,782 ± 0,009, validação **0,751** [0,730; 0,770], AUC 0,833. Os coeficientes (Figura 11) confirmam a EDA: os maiores efeitos positivos são `HomePlanet_Europa` (+0,79), `CryoSleep` (+0,76), `CabineRegiao_3` (+0,71) e `Deck_C` (+0,68). Os maiores negativos são `Spa_log` (−0,70), `VRDeck_log` (−0,63), `Deck_G` (−0,60) e `RoomService_log` (−0,53). `FoodCourt_log` (+0,44) e `ShoppingMall_log` (+0,24) têm sinal positivo, como antecipado em A4. Isso só aparece porque a regressão é multivariada: dado o gasto total, gastar em praça de alimentação e no shopping, e não em spa ou VR, está associado a ser transportado.

## 5.3 Técnica 2 — LightGBM (ensemble de árvores por *gradient boosting*)

**Por quê.** Ensembles de árvores capturam interações e relações não monotônicas (como deck × lado × número da cabine, A8) sem engenharia manual, e costumam ser o estado da arte em dados tabulares. O LightGBM [4] constrói as árvores folha a folha, com histogramas.

**Ajuste.** Busca aleatória de 60 combinações em 8 hiperparâmetros (27 s). A escolhida foi **400 árvores, taxa de aprendizado 0,1, `num_leaves` = 7, `min_child_samples` = 20, `colsample_bytree` = 0,7, `reg_lambda` = 10**: árvores rasas e regularização L2 forte. A Figura 12 (esquerda) mostra que as configurações com mais folhas atingem acurácia de treino maior, mas não de validação: o ganho de capacidade vira sobreajuste.

**Resultado.** Treino 0,853, CV **0,816** ± 0,009 (a maior), validação 0,786 [0,766; 0,806], AUC **0,885** (a maior). É a técnica com maior distância entre treino e validação (6,7 pontos). Mesmo assim, a curva de aprendizado (Figura 13) mostra a validação ainda subindo com mais dados (0,792 → 0,815) e o treino descendo (0,953 → 0,857): é um modelo de alta variância que se beneficiaria de mais exemplos. Pela importância por ganho (Figura 12, direita), `GastoTotal_log` (22% do ganho total) domina, seguido de `Spa_log` (9%), `CryoSleep` (9%), `FoodCourt_log`, `VRDeck_log` e `SemGasto` (8% cada). Os atributos criados na preparação estão entre os mais usados.

## 5.4 Técnica 3 — SVM com kernel RBF

**Por quê.** É um classificador de margem máxima. O kernel RBF gera uma fronteira não linear suave baseada em similaridade entre passageiros, uma abordagem bem diferente de árvores (partições) e de redes (composição de funções). Depende da padronização, que o pipeline garante.

**Ajuste.** Grade de `C` ∈ {0,1, …, 30} × `gamma` ∈ {0,003, …, 0,3} (30 combinações, 59 s, a busca mais cara). O mapa da Figura 10 (centro) mostra a troca clássica entre os dois parâmetros: a acurácia é máxima em uma diagonal (C = 3 com γ = 0,03; C = 30 com γ = 0,01) e cai quando γ é grande (fronteira irregular, sobreajuste: 0,765 com C = 30 e γ = 0,3) ou quando C e γ são pequenos (fronteira rígida demais, subajuste: 0,743). A escolhida foi **C = 3, γ = 0,03**.

**Resultado.** Treino 0,825, CV 0,810 ± 0,011, validação 0,780 [0,759; 0,798], AUC 0,868. É a técnica com menor distância entre treino e CV entre as não lineares (1,6 ponto). Sua curva de aprendizado é a mais estável (treino ~0,83 constante, validação subindo até 0,809). As probabilidades usadas na curva ROC e no ensemble vêm do escalonamento de Platt (`probability=True`), ativado só no modelo final. O custo é o maior: 11,0 s para ajustar nos 8 693 passageiros, contra 0,3 s do LightGBM, e o custo do SVM cresce de forma aproximadamente quadrática com o número de exemplos.

## 5.5 Técnica 4 — Rede neural (MLP)

**Por quê.** Um perceptron multicamadas aprende combinações não lineares dos atributos por retropropagação. É a família das redes neurais, diferente das três anteriores.

**Ajuste.** `MLPClassifier` (ReLU, Adam, entropia cruzada) com parada antecipada (10% do treino separados internamente, paciência de 20 épocas). Grade de arquitetura {(32), (64), (128), (64, 32), (128, 64)} × regularização L2 `alpha` {10⁻⁴, …, 1} × taxa de aprendizado {0,001, 0,003} (50 combinações, 33 s). A escolhida foi **uma camada oculta de 64 neurônios, alpha = 0,1, taxa 0,003**. A Figura 10 (direita) mostra diferenças pequenas entre as configurações (0,797–0,808): redes mais profundas não ajudaram em 7 mil exemplos tabulares.

**Curvas por época.** Para obter as curvas de perda (Figura 9), a configuração escolhida foi treinada por 300 épocas sem parada antecipada, medindo a *log loss* e a acurácia no treino e na validação. A perda de validação atinge o mínimo na época 32 (0,439) e depois sobe devagar (0,449 na época 300), enquanto a perda de treino continua caindo (0,348 → 0,298) e a acurácia de treino sobe de 0,83 para 0,87. É um sobreajuste típico. A acurácia de validação, porém, fica estável perto de 0,785, porque a perda piora por excesso de confiança nas previsões erradas, não por mais erros. A parada antecipada do modelo escolhido interrompeu o treino na época 63, perto da região ótima.

**Resultado.** Treino 0,838, CV 0,808 ± 0,006 (o menor desvio entre dobras), validação **0,791** [0,771; 0,810] (a maior), AUC 0,877. Ela é a mais assimétrica entre as classes: revocação de 0,824 para `False` e 0,758 para `True`. No teste, prevê apenas 45,6% de `True`, contra 51,9% a 52,4% das outras técnicas e 50,4% no treino. Isso levantou a hipótese de que o limiar de 0,5 da MLP estivesse deslocado para `False`, o que a favoreceria nesta validação (49,4% de positivos) e poderia prejudicá-la no teste. Testamos a hipótese na Seção 5.9.

![Figura 8 — Matrizes de confusão de todas as técnicas na validação (1 697 passageiros).](graficos/10_matrizes_confusao.png)

![Figura 9 — MLP escolhida: log loss (esquerda; menor é melhor) e acurácia (direita; maior é melhor) por época no treino e na validação. A linha tracejada marca onde a parada antecipada interrompeu o modelo usado na submissão.](graficos/15_mlp_curvas.png)

![Figura 10 — Ajuste de hiperparâmetros na validação cruzada agrupada (eixos: acurácia, maior é melhor). Esquerda: Regressão Logística (treino pontilhado, validação com barra de ±1 desvio-padrão). Centro: SVM, C × gamma. Direita: MLP, arquitetura × alpha.](graficos/12_busca_hiperparametros.png)

![Figura 11 — Regressão Logística: os 18 maiores coeficientes em módulo (atributos padronizados; azul aumenta e vermelho reduz a chance de `Transported`).](graficos/14_lr_coeficientes.png){width=70%}

![Figura 12 — LightGBM: acurácia de treino × validação das 60 configurações sorteadas (esquerda; acima da diagonal não há configurações, e quanto mais à direita, mais sobreajuste) e importância relativa dos atributos (direita).](graficos/13_lightgbm.png)

![Figura 13 — Curvas de aprendizado: acurácia (maior é melhor) de treino e de validação cruzada em função do número de exemplos; a faixa sombreada é ±1 desvio-padrão entre as dobras.](graficos/16_curvas_aprendizado.png)

## 5.6 Técnica 5 — k-NN (vizinhos mais próximos)

**Por quê.** O k-NN é um método baseado em instâncias: não aprende uma função, só guarda o treino e classifica cada passageiro pelo voto dos *k* mais parecidos. É uma família diferente das quatro anteriores (modelo linear, boosting de árvores, margem com kernel e rede neural).

**Ajuste.** `KNeighborsClassifier` com o mesmo pré-processamento (atributos padronizados e *one-hot*), porque a distância exige atributos na mesma escala. Grade de *k* ∈ {5, 9, 15, 25, 35, 51, 75, 101} × pesos {uniforme, por distância} × métrica de Minkowski com p ∈ {1, 2} (32 combinações, 4,9 s), pela mesma CV agrupada. A escolhida foi ***k* = 25, pesos uniformes e distância euclidiana (p = 2)**.

**Resultado.** Treino 0,808, CV 0,791 ± 0,005, validação **0,766** [0,746; 0,787], AUC 0,856. Ficou acima da regressão logística (0,751), mas abaixo das outras técnicas não lineares: contra a MLP, o k-NN acerta sozinho 69 passageiros e a MLP, 112 (McNemar, p = 0,002). Uma explicação provável, que não testamos, é que a distância euclidiana dá o mesmo peso a todos os 36 atributos, inclusive aos pouco informativos (como `Destination` e as categorias de `Deck`), enquanto LightGBM e MLP conseguem aprender a ignorá-los. Ajustar é quase instantâneo (0,04 s), porque o método só guarda os dados.

## 5.7 Tentativa extra — Ensemble por votação suave

**Por quê.** Se os erros das técnicas não coincidem, a média das probabilidades pode reduzir a variância. O ensemble usa as quatro técnicas acima, com os hiperparâmetros já escolhidos e pesos iguais. Não houve busca adicional, para não usar a validação na escolha dos pesos.

**Resultado.** Treino 0,833, CV 0,814 ± 0,012, validação 0,785 [0,764; 0,804], AUC 0,878. Ficou entre os seus membros, sem superar o melhor. O teste de McNemar mostra que os erros das técnicas não lineares coincidem em grande parte: a MLP e o LightGBM discordam em apenas 173 dos 1 697 passageiros (10%), o que deixa pouco espaço para a média ajudar. Além disso, a regressão logística, que é mais fraca, entra com o mesmo peso das outras.

## 5.8 Tentativa extra — Ensemble por *stacking*

**Por quê.** Na votação suave todas as técnicas têm o mesmo peso, inclusive a mais fraca. No *stacking*, um meta-modelo (regressão logística com parâmetros padrão) aprende quanto confiar em cada uma. As entradas são as probabilidades de `True` das quatro técnicas geradas *fora da dobra* no treino (CV agrupada de 5 dobras), para que ele não aprenda com previsões de modelos que já viram o passageiro. Não houve busca adicional.

**Resultado.** Treino 0,856, CV 0,818 ± 0,009, validação **0,792** [0,771; 0,810], AUC 0,888: os maiores valores de CV, validação e AUC, mas por margens pequenas. Contra a votação suave (76 contra 64 discordâncias, p = 0,35) e contra a MLP (65 contra 64, p = 1,00) não há diferença detectável. Os pesos do meta-modelo são +4,79 (LightGBM), +1,96 (MLP), −0,37 (regressão logística) e −0,56 (SVM). Ele se apoia quase só no LightGBM e na MLP, e os pesos negativos pequenos não significam que os outros dois atrapalham: com entradas tão correlacionadas, eles servem para corrigir pequenas diferenças. O ganho pequeno tem a mesma explicação da votação, já que os erros dos modelos coincidem em grande parte (Seção 5.7). Dois cuidados: o "treino" de 0,856 não é comparável ao das outras técnicas (usa as probabilidades dos modelos-base nos próprios passageiros de treino), e a CV do *stacking* é um pouco otimista, porque os hiperparâmetros dos modelos-base foram escolhidos nas mesmas dobras. O ajuste final leva 27 s.

## 5.9 Tentativa extra — MLP com limiar de decisão ajustado

**Por quê.** A MLP prevê menos `True` no teste (45,6%, contra cerca de 52% das outras), e a Seção 5.5 levantou a hipótese de um limiar deslocado. Se fosse isso, escolher o limiar por validação cruzada deveria ajudar.

**Como.** Calculamos as probabilidades fora da dobra da MLP no treino e escolhemos, entre 0,30 e 0,70 (passo de 0,01), o limiar de maior acurácia na curva suavizada (média móvel de 5 pontos). A validação não participa da escolha. Para ter uma estimativa honesta, também escolhemos o limiar em 4 dobras e o aplicamos na quinta.

**Resultado.** O limiar escolhido foi **0,53**, acima de 0,50 e não abaixo, como a hipótese previa. Na CV por dobra, a acurácia mal muda (0,8075 com 0,50 e 0,8068 com 0,53; Figura 14, esquerda). Na validação o limiar ajustado piorou (Tabela 2; McNemar p = 0,029, com 11 acertos exclusivos do limiar ajustado contra 25 do padrão). Esse p está perto do limite e vem de um relatório com vários testes, então não o lemos como prova forte, mas também não há indício de que o ajuste ajude.

| | Limiar 0,50 (padrão) | Limiar 0,53 (escolhido na CV) |
|:-----------------------------------|-------:|-------:|
| Acurácia na validação | 0,7914 | 0,7831 |
| Revocação de `False` (validação) | 0,824 | 0,837 |
| Revocação de `True` (validação) | 0,758 | 0,728 |
| % previsto como `True` na validação | 46,4% | 44,3% |
| % previsto como `True` no teste | 45,6% | 43,5% |

*Tabela 2 — MLP antes e depois do ajuste do limiar. A proporção real de `True` é 50,6% no treino e 49,4% na validação.*

![Figura 14 — MLP: acurácia na CV agrupada por limiar de decisão (esquerda; maior é melhor), com o limiar padrão e o escolhido, e fração de passageiros previstos como `True` em cada limiar no treino (fora da dobra), na validação e no teste (direita).](graficos/17_limiar_mlp.png){width=90%}

**Interpretação.** O limiar escolhido empurra ainda mais as previsões para `False` (no teste, 45,6% → 43,5%), o contrário do que a hipótese pedia. Na CV o limiar de 0,5 já estava perto do ideal, então a menor taxa de `True` no teste não é um defeito de limiar que se corrija assim. Pode refletir uma diferença de composição entre treino e teste (não verificamos) ou a MLP ser mais conservadora. Mantivemos a MLP com limiar 0,50 como a oficial. A versão com limiar ajustado é uma tentativa extra, com submissão própria (`07_mlp_limiar.csv`), e um resultado negativo. Não ajustamos o limiar de novo olhando a validação ou o Kaggle, o que seria usar o conjunto de avaliação para escolher hiperparâmetros.

# 6. Comparação e discussão

{{TABELA_COMPARATIVA}}

*Tabela 1 — Resultados de todas as tentativas no mesmo protocolo. Treino: acurácia do modelo escolhido nos 6 996 passageiros de treino. CV: média ± desvio-padrão da acurácia nas 5 dobras agrupadas do treino, para a configuração escolhida. Validação: acurácia nos 1 697 passageiros reservados, com IC 95% por bootstrap. Kaggle público: pontuação do arquivo de submissão (modelo reajustado nos 8 693 passageiros). Ajuste: tempo de um ajuste nos 8 693 passageiros (os tempos de busca estão na Seção 5). Os tempos dependem da máquina e variam entre execuções; as métricas são idênticas. O ajuste do stacking inclui o reajuste dos quatro modelos-base e as probabilidades fora da dobra.*

![Figura 15 — Acurácia (maior é melhor) de treino, validação cruzada, validação e Kaggle público para cada técnica. As barras de erro são o IC 95% da validação, e a linha pontilhada é o baseline.](graficos/20_comparacao.png)

![Figura 16 — Acurácia de validação × tempo de ajuste no treino completo (escala log).](graficos/21_desempenho_custo.png){width=70%}

**Ranking e significância.** Na validação, a ordem foi Stacking (0,792) ≈ MLP (0,791) > LightGBM (0,786) > Ensemble por votação (0,785) > MLP com limiar ajustado (0,783) > SVM (0,780) > k-NN (0,766) ≫ Regressão Logística (0,751) ≫ baseline (0,494). Na validação cruzada a ordem entre as não lineares é outra (Stacking 0,818 > LightGBM 0,816 > Ensemble 0,814 > SVM 0,810 > MLP 0,808 > MLP com limiar 0,807 > k-NN 0,791). Isso mostra que as diferenças entre Stacking, MLP, LightGBM, votação e SVM, de 0,1 a 1,2 ponto, são menores que o ruído de estimativa: o IC 95% da validação tem ±2 pontos, e o teste de McNemar da MLP contra o LightGBM (p = 0,54), o SVM (p = 0,11), a votação (p = 0,38) e o stacking (p = 1,00) não é significativo. **A conclusão estatisticamente defensável é que essas cinco técnicas estão empatadas.** Duas diferenças são claras: a regressão logística fica abaixo de todas as outras em acurácia (para a MLP, 150 contra 81 discordâncias, p < 0,001), e o k-NN fica abaixo da MLP (69 contra 112, p = 0,002). A diferença entre a MLP e a sua versão com limiar ajustado é discutida na Seção 5.9.

**Por que o modelo linear perdeu.** A EDA já mostrava relações que um hiperplano não representa bem: o efeito da idade concentrado nas crianças (A6), a oscilação da taxa com o número da cabine (A8) e principalmente interações. Por exemplo, gastar zero tem um significado para quem está em criossono (é obrigatório) e outro para quem está acordado. A engenharia de atributos (faixas de cabine, `SemGasto`, log dos gastos) recuperou parte disso, mas a regressão logística continua com treino ≈ validação ≈ 0,78 na CV, em qualquer regularização e tamanho de treino. É um caso claro de viés: o limite é a forma do modelo, não a quantidade de dados.

**Treino × validação: sobreajuste de cada técnica.** As distâncias entre treino e CV são 0,2 ponto (LR), 1,6 (SVM), 1,7 (k-NN), 3,0 (MLP) e 3,7 (LightGBM). A do stacking (3,8) não é comparável, porque a sua acurácia de treino é inflada por construção (Seção 5.8). Mesmo com a regularização escolhida pela própria CV, o LightGBM ainda memoriza parte do treino. Como a busca usou a acurácia de validação para escolher os hiperparâmetros, ela aceitou esse sobreajuste porque ele não piorava a generalização. A curva de aprendizado mostra que, com mais dados, LightGBM e SVM ainda melhorariam, enquanto LR e MLP estão perto do platô.

**Validação cruzada × validação reservada.** Em todas as técnicas, inclusive o baseline, a acurácia na validação reservada ficou de 1,1 a 3,1 pontos abaixo da média da CV. Duas causas se somam. (i) *Viés de seleção*: a média de CV reportada é a da melhor configuração entre 20 a 60 testadas, portanto otimista. Esse efeito é pequeno na LR (poucas configurações quase iguais) e maior nas outras. (ii) *Variação entre partições*: a validação reservada é uma única dobra de 1 697 passageiros e parece ser mais difícil que a média. A queda da LR (3,1 pontos), que quase não tem viés de seleção, indica que (ii) predomina. Por isso as comparações usam sempre a **mesma** partição para todas as técnicas, e diferenças menores que o IC não são interpretadas.

**Validação × Kaggle.** [PREENCHER APÓS AS SUBMISSÕES — comparar a coluna "Kaggle público" da Tabela 1 com "Validação" e "CV" nos nove arquivos. Esperamos pontuações públicas entre a validação e a CV (≈0,78–0,82), porque o modelo final é treinado com 25% mais dados (8 693 contra 6 996), o que, pelas curvas de aprendizado, beneficia sobretudo o LightGBM e o SVM. Se a ordem das técnicas no Kaggle diferir da validação, isso é coerente com o empate estatístico acima: a pontuação pública usa só parte do teste, e diferenças de 0,5 ponto equivalem a poucas dezenas de passageiros. Comparar `04_rede_neural.csv` com `07_mlp_limiar.csv`: a CV indicou que o limiar ajustado não ajuda, e o Kaggle dirá se isso se confirma no teste (sem usar essa nota para escolher o limiar). Comparar também `05_ensemble.csv` com `08_stacking.csv`.]

**Custo computacional e interpretabilidade.** A regressão logística é a mais barata entre as técnicas com treino de verdade (0,07 s) e a mais interpretável (coeficientes com sinal e magnitude, Figura 11), mas perde cerca de 3 a 4 pontos de acurácia. O k-NN ajusta em 0,04 s, só guardando os dados, e perde em acurácia para as outras não lineares. O LightGBM tem o melhor equilíbrio: entre os melhores em acurácia, AUC alta (0,885; só o stacking, com 0,888, é maior), ajuste em 0,3 s e importâncias de atributos diretas. O SVM empata em acurácia, mas é 35 vezes mais lento para ajustar, escala mal com mais dados e é o menos interpretável. A MLP é rápida para prever, mas sensível à inicialização e ao limiar, e precisa de parada antecipada. A votação soma o custo de todos (11,8 s, dominado pelo SVM) sem ganho de acurácia, e o stacking leva 27 s, cerca de 85 vezes o tempo do LightGBM. **Para este problema, recomendaríamos o LightGBM**: ele empata com o melhor resultado, tem o menor custo entre os modelos não lineares, AUC alta e explicações por atributo. O stacking tem os maiores números, mas sem diferença detectável e com mais custo e menos interpretabilidade.

**Importância por permutação.** Coeficientes e importâncias por ganho (Figuras 11 e 12) não se comparam entre modelos. Por isso medimos, na validação, quanto a acurácia cai quando um grupo de atributos é embaralhado (média de 20 repetições; atributos correlacionados, como os gastos e `SemGasto`, são embaralhados juntos).

![Figura 17 — Queda de acurácia na validação (em pontos percentuais; maior significa que o modelo depende mais do grupo) ao embaralhar cada grupo de atributos, para cada modelo.](graficos/18_importancia_permutacao.png){width=70%}

Em todos os modelos os **gastos a bordo** dominam, com quedas de 18,6 pontos (LightGBM) a 23,4 (k-NN). O resto é bem menor: cabine (1,7 na regressão logística a 5,2 na MLP), planeta de origem (0,4 a 2,0), `CryoSleep` (0,2 a 1,5) e, abaixo de 1 ponto, destino, idade, grupo de viagem e VIP. Os modelos concordam no ranking (correlação de Spearman entre 0,83 e 0,98 para regressão logística, LightGBM, SVM e MLP, e entre 0,74 e 0,83 com o k-NN), e a cabine pesa mais nos modelos não lineares do que na regressão logística, o que é coerente com a relação não monotônica do número da cabine (A8). Dois cuidados: (i) o `CryoSleep` parece pouco importante, apesar da correlação de 0,47 com o alvo, porque quem está em criossono não gasta (A5) e os modelos recuperam essa informação pelos gastos; (ii) embaralhar os gastos cria combinações impossíveis (criossono com gasto), o que provavelmente exagera a queda desse grupo. Os valores medem a dependência de cada modelo, não a importância do grupo para o problema.

**Atributo de família (ablação).** Para decidir sobre o atributo `TamFamilia` (A9), comparamos cada técnica com e sem ele, com os mesmos hiperparâmetros, as mesmas dobras e a mesma validação.

| Técnica | CV sem | CV com | Dif. na CV (média) | Dobras com ganho | Val. sem | Val. com | Dif. na val. |
|:--------------------|------:|------:|-------:|-----:|------:|------:|-------:|
| Regressão Logística | 0,7819 | 0,7828 | +0,0009 | 3/5 | 0,7507 | 0,7525 | +0,0018 |
| LightGBM | 0,8156 | 0,8155 | −0,0001 | 2/5 | 0,7861 | 0,7820 | −0,0041 |
| SVM (kernel RBF) | 0,8096 | 0,8066 | −0,0030 | 0/5 | 0,7796 | 0,7767 | −0,0029 |
| Rede Neural (MLP) | 0,8075 | 0,8027 | −0,0049 | 0/5 | 0,7914 | 0,7820 | −0,0094 |
| k-NN | 0,7907 | 0,7887 | −0,0020 | 2/5 | 0,7661 | 0,7672 | +0,0012 |

*Tabela 3 — Acurácia com e sem o atributo de família. "Dobras com ganho" conta em quantas das 5 dobras da CV o atributo melhorou a acurácia.*

O atributo não ajuda de forma consistente. Na CV só a regressão logística melhora (+0,0009, menos que o desvio-padrão da diferença entre dobras, 0,0015), o SVM e a MLP pioram em todas as 5 dobras, e o LightGBM e o k-NN ficam iguais. Na validação a MLP perde 0,9 ponto. Uma explicação plausível está em A9: com 2 406 sobrenomes para cerca de 13 mil passageiros, `TamFamilia` mede sobretudo o quão comum o sobrenome é. Os hiperparâmetros não foram refeitos com o atributo, o que favorece um pouco a versão sem ele. Mantivemos os 36 atributos originais.

**Limitações.** (i) A avaliação final usa uma única partição de validação. Uma validação cruzada aninhada completa daria intervalos menores, com um custo cerca de 5 vezes maior. (ii) A busca de hiperparâmetros foi limitada (por exemplo, 60 sorteios no LightGBM). O limiar de decisão foi ajustado só para a MLP, e o k-NN usa a distância sem ponderar os atributos. (iii) A CV do *stacking* é um pouco otimista, porque os hiperparâmetros dos modelos-base foram escolhidos nas mesmas dobras, e a estimativa por dobra do limiar é aproximada (as probabilidades fora da dobra já tinham sido geradas pela CV completa). (iv) O atributo de família foi testado sem refazer a busca de hiperparâmetros e só como contagem de sobrenomes. Não usamos os rótulos de outros membros do grupo, o que exigiria cuidado redobrado com vazamento e não está disponível no teste, cujos grupos são novos. (v) A importância por permutação é afetada pela correlação entre atributos e por combinações impossíveis criadas ao embaralhar.

# 7. Conclusão

Aplicamos o ciclo completo de aprendizado de máquina ao *Spaceship Titanic*. A EDA guiou a preparação, com regras determinísticas de imputação, transformação log dos gastos e atributos de grupo e de cabine, e também o protocolo, com validação agrupada por grupo de viagem porque o Kaggle separa treino e teste dessa forma. Todas as transformações com parâmetros ficaram dentro de pipelines ajustados só no treino. Com protocolo único, métrica oficial e baseline (0,494), as cinco técnicas distintas e os dois ensembles atingiram, na validação, 0,751 (Regressão Logística), 0,766 (k-NN), 0,780 (SVM RBF), 0,785 (votação suave), 0,786 (LightGBM), 0,791 (MLP) e 0,792 (stacking). LightGBM, SVM, MLP e os ensembles ficaram estatisticamente empatados e cerca de 3 a 4 pontos acima do modelo linear, que é limitado pelo viés, e o k-NN ficou abaixo da MLP. Dois experimentos de melhoria não deram resultado: o limiar de decisão ajustado por validação cruzada (0,53) piorou a MLP na validação (de 0,791 para 0,783), e o atributo de família não melhorou os modelos de forma consistente. A importância por permutação mostrou que todos os modelos dependem sobretudo dos gastos a bordo. As curvas de aprendizado e as distâncias entre treino e validação mostram perfis diferentes: LightGBM e SVM ainda ganhariam com mais dados, e MLP e LightGBM sobreajustam mais. Considerando acurácia, custo e interpretabilidade, o LightGBM é a escolha recomendada. O stacking tem os maiores números, mas sem diferença detectável e com mais custo e complexidade. No Kaggle, as submissões obtiveram [PREENCHER]. Como trabalho futuro, sugerimos validação cruzada aninhada, ponderação dos atributos no k-NN e atributos de família que usem relações mais específicas que o sobrenome.

# Referências

[1] KAGGLE. *Spaceship Titanic — Predict which passengers are transported to an alternate dimension*. 2022. Disponível em: <https://www.kaggle.com/competitions/spaceship-titanic>. Acesso em: nov. 2026.

[2] PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.

[3] SCIKIT-LEARN DEVELOPERS. *Common pitfalls and recommended practices* e *Cross-validation: evaluating estimator performance*. Disponível em: <https://scikit-learn.org/stable/common_pitfalls.html> e <https://scikit-learn.org/stable/modules/cross_validation.html>. Acesso em: nov. 2026.

[4] KE, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In: *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.

[5] HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. (Regressão logística, SVM, redes neurais e boosting; caps. 4, 10, 11 e 12.)

[6] DIETTERICH, T. G. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, v. 10, n. 7, p. 1895–1923, 1998.

[7] PLATT, J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, p. 61–74, 1999.

[8] KAGGLE LEARN. *Intro to Machine Learning* e *Intermediate Machine Learning*. Disponível em: <https://www.kaggle.com/learn>.

# Apêndice A — Espaços de busca e hiperparâmetros escolhidos

| Técnica | Espaço de busca | Busca | Escolhidos |
|---|---|---|---|
| Regressão Logística | C ∈ {0,001; 0,003; 0,01; 0,03; 0,1; 0,3; 1; 3; 10; 100}; penalidade ∈ {L1, L2}; solver liblinear | grade (20) | C = 0,1; L2 |
| LightGBM | n_estimators ∈ {100, 200, 400, 800}; learning_rate ∈ {0,01; 0,02; 0,05; 0,1}; num_leaves ∈ {7, 15, 31, 63}; min_child_samples ∈ {10, 20, 50, 100}; subsample ∈ {0,7; 0,85; 1}; colsample_bytree ∈ {0,5; 0,7; 1}; reg_lambda ∈ {0, 1, 5, 10} | aleatória (60 de 9 216) | 400; 0,1; 7; 20; 1,0; 0,7; 10 |
| SVM RBF | C ∈ {0,1; 0,3; 1; 3; 10; 30}; gamma ∈ {0,003; 0,01; 0,03; 0,1; 0,3} | grade (30) | C = 3; gamma = 0,03 |
| MLP | camadas ∈ {(32), (64), (128), (64, 32), (128, 64)}; alpha ∈ {10⁻⁴; 10⁻³; 10⁻²; 10⁻¹; 1}; learning_rate_init ∈ {0,001; 0,003}; Adam, ReLU, early stopping (10%, paciência 20) | grade (50) | (64); 0,1; 0,003 |
| Ensemble | votação suave das 4 técnicas acima, pesos iguais | — | — |
| k-NN | k ∈ {5, 9, 15, 25, 35, 51, 75, 101}; pesos ∈ {uniforme, distância}; p de Minkowski ∈ {1, 2} | grade (32) | k = 25; uniforme; p = 2 |
| MLP com limiar | limiar ∈ {0,30; 0,31; …; 0,70}, escolhido na curva OOF suavizada (média móvel de 5 pontos) | varredura (41) | 0,53 |
| Stacking | regressão logística (parâmetros padrão) sobre as probabilidades OOF de RL, LightGBM, SVM e MLP (CV agrupada de 5 dobras) | — | pesos +4,79 (LightGBM), +1,96 (MLP), −0,37 (RL), −0,56 (SVM); intercepto −2,87 |

Precisão, revocação e F1 por classe de todas as técnicas do `02_modelos.py`, na validação, estão em `resultados/relatorio_classificacao.txt`. As previsões de validação estão em `resultados/predicoes_validacao.csv`, e o teste de McNemar em `resultados/mcnemar_validacao.csv`. As tentativas extras geram `resultados/resultados_extras.csv`, `resultados/mcnemar_extras.csv`, `resultados/mlp_limiar_antes_depois.csv`, `resultados/stacking_coeficientes.csv` e `resultados/predicoes_validacao_extras.csv`. A ablação do atributo de família está em `resultados/ablacao_familia.csv`, e a importância por permutação em `resultados/importancia_permutacao.csv` e `resultados/importancia_permutacao_spearman.csv`.

# Apêndice B — Submissões no Kaggle

[INSERIR AQUI a captura de tela da página *My Submissions* (<https://www.kaggle.com/competitions/spaceship-titanic/submissions>) mostrando os nove arquivos enviados (`00_baseline.csv` … `08_stacking.csv`), com datas e pontuações.]

# Apêndice C — Uso de ferramentas de IA generativa

[REVISAR E AJUSTAR para refletir o uso real.] Foram usados assistentes de IA da Anthropic como apoio, em duas etapas. (1) O **Claude Code** na escrita dos scripts Python do pipeline original (estrutura do pipeline, figuras e geração do relatório) e numa primeira versão deste texto, a partir das instruções do trabalho. [CONFERIR se foi isso mesmo.] (2) O **Claude (conversa no claude.ai)** na revisão do projeto e na implementação das melhorias: tentativas extras (k-NN, limiar da MLP e *stacking*), ablação do atributo de família, importância por permutação, notebook da EDA, comentários no código e redação das seções correspondentes deste relatório. A dupla verificou o resultado da seguinte forma: executou todos os scripts em sequência com os dados reais do Kaggle, sem erros; [DESCREVER o que a dupla de fato fez além disso — por exemplo: conferiu que cada número das tabelas e do texto coincide com `resultados/`; leu o código de cada etapa e verificou a ausência de vazamento; conferiu as figuras; submeteu os arquivos ao Kaggle]. Nenhum notebook público do Kaggle nem código de terceiros foi copiado; as bibliotecas usadas estão citadas nas referências.
