🌿 Sistema de Monitoramento Ambiental Industrial
Disciplina: Inteligência Artificial
Alunos: Douglas Hebert e Lucas de Souza Morais

📌 Descrição do Projeto

Este projeto implementa um Sistema de Monitoramento Ambiental Industrial, utilizando dados reais de autos de infração do IBAMA.

A solução inclui:

📥 Ingestão e pré-processamento de dados

🧽 Limpeza automática do dataset (84 colunas)

🤖 Modelos de IA, incluindo:

Classificação (RandomForestClassifier)

Regressão (RandomForestRegressor)

Detecção de Anomalias (IsolationForest)

📊 Dashboard interativo em HTML

🔎 Análises exploratórias

🧱 Pipeline modularizado (src/)

O objetivo é demonstrar como aplicar IA em dados ambientais para:
✔️ apoiar fiscalização
✔️ detectar comportamentos anômalos
✔️ gerar insights ambientais

📁 Estrutura do Repositório

projeto-ambiental/
│
├── data/
│ ├── raw/ — dados brutos (CSV do IBAMA)
│ └── processed/ — dados limpos (parquet)
│
├── models/ — modelos treinados (.joblib)
│
├── src/
│ ├── data_ingestion.py
│ ├── preprocessing.py
│ ├── inspect_parquet.py
│ ├── model.py
│ └── generate_dashboard.py
│
├── dashboard.html
├── requirements.txt
├── .gitignore
└── README.md

🚀 Como Executar o Projeto
1️⃣ Instale as dependências

pip install -r requirements.txt

2️⃣ Coloque o CSV do IBAMA em:

data/raw/auto_infracao_2024.csv

3️⃣ Execute o pré-processamento:

python src/preprocessing.py
Gera arquivos em data/processed:

clean_autuacoes.parquet

sample_for_dashboard.parquet

4️⃣ Treine os modelos:

python src/model.py
Gera:

preprocessor.joblib

rf_clf.joblib

rf_reg.joblib

iso_forest.joblib

5️⃣ Gere o Dashboard:

python src/generate_dashboard.py
Gera:

dashboard.html

Abra no navegador.

🤖 Modelos Utilizados
Random Forest Classifier

Usado para classificar infrações com base em:

tipo

gravidade

bioma

UF

histórico

Random Forest Regressor

Prediz valores de multa com base em:

bioma afetado

tipo de infração

estado

campos numéricos e categóricos

Isolation Forest

Detecta infrações fora do padrão — útil para identificar anomalias ambientais.

📊 Dashboard

O dashboard apresenta:

Gráficos Plotly

Mapa de coordenadas (Leaflet)

Indicadores

Lista de anomalias

Filtros por UF, Bioma e Ano

Arquivo final: dashboard.html

🧪 Tecnologias Utilizadas

Python 3

Pandas

Scikit-Learn

Numpy

FastParquet

Plotly

Folium

Joblib

📅 Última Atualização

2025-02-28
