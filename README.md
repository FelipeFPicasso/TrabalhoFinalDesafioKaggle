# Spaceship Titanic — comparação de técnicas de aprendizado supervisionado

Trabalho prático da disciplina de Aprendizado de Máquina sobre o desafio
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) do Kaggle: prever quais
passageiros foram transportados para outra dimensão (classificação binária, métrica oficial: acurácia).

O projeto faz a análise exploratória dos dados, prepara os atributos sem vazamento de dados e compara,
sob um mesmo protocolo de validação agrupada por grupo de viagem, um baseline, Regressão Logística,
LightGBM, SVM com kernel RBF, rede neural MLP e um ensemble por votação suave. Gera os arquivos de
submissão para o Kaggle, as figuras, as tabelas comparativas e o relatório.

Os dados (`train.csv` e `test.csv`) devem ser baixados da página do desafio e colocados em `data/`.
Dependências em `requirements.txt`; execução completa com `./run_all.sh`.
