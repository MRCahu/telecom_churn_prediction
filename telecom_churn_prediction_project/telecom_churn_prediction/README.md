# Previsão de Churn na Telecom X

Este projeto visa prever a evasão de clientes utilizando dados históricos da Telecom X. O dataset `TelecomX_Data.json` contém registros de 7.267 clientes em formato JSON hierárquico, processado para aplicação de modelos de Machine Learning.

## Estrutura do Projeto
- `data/` - dados brutos e processados
- `notebooks/` - scripts de pré-processamento e modelagem
- `models/` - modelos treinados
- `reports/` - relatórios e figuras

## Criação do Ambiente
```bash
python -m venv .venv
source .venv/bin/activate
# instale as dependências utilizando o arquivo na raiz do repositório
pip install -r ../../requirements.txt
```

## Execução
```bash
python notebooks/01_data_preprocessing.py
python notebooks/02_exploratory_analysis.py
python notebooks/03_model_development.py
pytest
```

## Métricas Finais
A **Regressão Logística** obteve acurácia de **0.8010** e F1-Score de **0.5795** no conjunto de teste.

## Figuras Principais
![Matriz de Confusão](telecom_churn_prediction_project/telecom_churn_prediction/reports/confusion_matrix_Regressão_Logística.png)

![Importância das Features](telecom_churn_prediction_project/telecom_churn_prediction/reports/feature_importance.png)

Detalhes adicionais estão em `reports/report.md`.
