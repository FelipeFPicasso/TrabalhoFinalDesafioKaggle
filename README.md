# Spaceship Titanic — comparação de técnicas de aprendizado supervisionado

Trabalho prático da disciplina de Aprendizado de Máquina sobre o desafio
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) do Kaggle: prever quais
passageiros foram transportados para outra dimensão (classificação binária, métrica oficial: acurácia).

O projeto faz a análise exploratória dos dados, prepara os atributos sem vazamento de dados e compara,
sob um mesmo protocolo de validação agrupada por grupo de viagem, um baseline e sete tentativas:
Regressão Logística, LightGBM, SVM com kernel RBF, rede neural MLP, k-NN, um ensemble por votação suave
e um ensemble por stacking, além da MLP com limiar de decisão ajustado. Gera os arquivos de submissão
para o Kaggle, as figuras, as tabelas comparativas e (à parte) o relatório.

## Como executar

1. **Ambiente** (Python 3.10 ou superior), a partir da pasta do projeto:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # no Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Dados:** baixe `train.csv` e `test.csv` da [página do desafio](https://www.kaggle.com/competitions/spaceship-titanic/data)
   e coloque em `data/` (a pasta não vai para o repositório).
3. **Pipeline completo** (alguns minutos; as sementes são fixas, então os resultados se repetem):
   ```bash
   ./run_all.sh                       # sem .venv: PY=python ./run_all.sh
   ```
   O script roda, nesta ordem:

   | Etapa | O que faz | Principais saídas |
   |---|---|---|
   | `01_eda.py` | Análise exploratória | `graficos/01` a `08`, `resultados/eda_resumo.txt` |
   | `02_modelos.py` | Baseline, Regressão Logística, LightGBM, SVM, MLP e ensemble por votação | `submissoes/00` a `05`, `resultados/resultados.csv`, `graficos/10` a `16` |
   | `02b_tentativas_extras.py` | k-NN, MLP com limiar ajustado e stacking | `submissoes/06` a `08`, `resultados/resultados_extras.csv`, `graficos/17` |
   | `02c_ablacao_familia.py` | Mede o efeito do atributo de família (sobrenome) | `resultados/ablacao_familia.csv`, `graficos/19` |
   | `02d_importancia_permutacao.py` | Importância por permutação em cada modelo | `resultados/importancia_permutacao.csv`, `graficos/18` |
   | `03_comparacao.py` | Tabela e gráficos comparativos | `resultados/tabela_comparativa.md`, `graficos/20` e `21` |

   Os scripts `02b`, `02c` e `02d` usam os hiperparâmetros salvos pelo `02_modelos.py`, então a ordem importa.
4. **Submissão ao Kaggle** (precisa de `kaggle.json` configurado; o Kaggle limita o número de submissões por dia):
   ```bash
   ./submeter.sh 06_knn.csv 07_mlp_limiar.csv 08_stacking.csv   # só as informadas; sem argumentos envia todas
   ```
   Depois copie as pontuações pública e privada da página *My Submissions* para `kaggle_scores.csv` e rode
   `./run_all.sh` de novo (ou só o `03_comparacao.py`) para a tabela e o gráfico incluírem o Kaggle.
5. **Relatório** (fora do `run_all.sh`; precisa do LibreOffice, comando `soffice`):
   ```bash
   .venv/bin/python 04_relatorio.py ML_Kaggle_Picasso_Reginato
   ```
   Monta `relatorio/relatorio.md` em `.docx` e `.pdf`, com a tabela comparativa mais recente.

## Notebook da EDA

`01_eda.ipynb` tem a mesma análise de `01_eda.py`, em células, para apresentar no vídeo. Para abrir:
`pip install notebook` e `jupyter notebook` na pasta do projeto (precisa ser dela para achar `src/`),
depois *Run All*. O `01_eda.py` é a fonte: para regenerar o notebook, `pip install jupytext` e
`jupytext --to ipynb 01_eda.py`.

## Atributo de família (opcional)

`TamFamilia` (nº de pessoas com o mesmo sobrenome, contado em treino + teste, sem usar o alvo) não é usado
por padrão. O `02c_ablacao_familia.py` mede o ganho. Para adotá-lo em todo o projeto, troque
`USAR_FAMILIA = True` em `src/preparacao.py` e rode `./run_all.sh` de novo.

## Estrutura

```
01_eda.py / 01_eda.ipynb        análise exploratória
02_modelos.py                   técnicas principais
02b_tentativas_extras.py        k-NN, limiar da MLP, stacking
02c_ablacao_familia.py          ablação do atributo de família
02d_importancia_permutacao.py   importância por permutação
03_comparacao.py                tabela e figuras comparativas
04_relatorio.py                 relatório em docx/pdf
src/                            preparação (preparacao.py), modelos (modelos.py) e avaliação (avaliacao.py)
data/  graficos/  resultados/  submissoes/  relatorio/
kaggle_scores.csv               pontuações do Kaggle (preenchidas à mão)
```
