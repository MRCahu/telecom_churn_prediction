# 📉 Previsão de Evasão de Clientes (Churn) – Telecom X

Este projeto tem como objetivo prever a evasão de clientes da **Telecom X** utilizando técnicas de *Machine Learning*. A base de dados, composta por **7.267 registros** em formato JSON hierárquico, foi transformada em um dataset estruturado para a construção de modelos preditivos.

Todo o pipeline de ciência de dados foi contemplado: desde o pré-processamento e análise exploratória, até o desenvolvimento, comparação e avaliação de modelos com foco em desempenho.

---

## 🧠 Tecnologias Utilizadas

- Python 3.11+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- Pytest
- GitHub Actions (CI/CD)

---

## 📁 Estrutura do Projeto

```text
.
├── .github/workflows/                   # Integração contínua (CI)
├── src/                                 # Código-fonte principal
│   ├── data_preprocess.py
│   ├── train_and_evaluate.py
│   └── utils.py
├── telecom_churn_prediction_project/
│   └── telecom_churn_prediction/
├── data/
│   ├── TelecomX_Data.json               # Dados originais
│   └── processed_data.csv               # Dados tratados
├── models/                              # Modelos treinados
├── notebooks/                           # Notebooks de análise e modelagem
│   ├── 01_data_preprocessing.py
│   ├── 02_exploratory_analysis.py
│   ├── 03_model_development.py
│   └── telecom_churn_prediction_notebook.ipynb
├── reports/                             # Relatórios, gráficos e métricas
│   ├── confusion_matrix_Random_Forest.png
│   ├── confusion_matrix_Regressão_Logística.png
│   ├── exploratory_analysis.png
│   ├── feature_importance.png
│   ├── logistic_regression_coefficients.csv
│   ├── models_comparison.csv
│   ├── random_forest_importance.csv
│   └── report.md
├── tests/                               # Testes automatizados
│   ├── test_data_preprocess.py
│   └── test_train_and_evaluate.py
├── LICENSE
├── README.md
├── requirements.txt
└── todo.md
````

---

## ⚙️ Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/MRCahu/telecom_churn_prediction.git
cd telecom_churn_prediction
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
.venv\Scripts\activate            # Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline de Execução

Após configurar o ambiente, execute:

```bash
python notebooks/01_data_preprocessing.py
python notebooks/02_exploratory_analysis.py
python notebooks/03_model_development.py
pytest
```

---

## 🧪 Testes

Os testes com `pytest` cobrem o pipeline de dados e funções auxiliares.

```bash
pytest
```

---

## 📊 Resultados e Avaliação dos Modelos

| Modelo              | Acurácia | F1-Score |
| ------------------- | -------- | -------- |
| Regressão Logística | 0.8010   | 0.5795   |
| Random Forest       | 0.7950   | 0.5712   |
| KNN                 | 0.7431   | 0.4846   |

📄 Arquivo com os resultados: [![models_comparison.csv](reports/models_comparison.csv)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/models_comparison.csv)

---

## 📈 Visualizações

### Matriz de Confusão – Regressão Logística

[![Matriz Logística](reports/confusion_matrix_logistica.png)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/confusion_matrix_logistic.png)

### Matriz de Confusão – Random Forest

[![Matriz RF](reports/confusion_matrix_random_forest.png)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/confusion_matrix_Random_Forest.png)

### Importância das Variáveis

[![Importância](reports/feature_importance.png)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/feature_importance.png)

### Análise Exploratória

[![EDA](reports/exploratory_analysis.png)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/exploratory_analysis.png)

📘 Relatório completo: [![report.md](reports/report.md)](https://github.com/MRCahu/telecom_churn_prediction/blob/main/telecom_churn_prediction_project/telecom_churn_prediction/reports/report.md)

---

## 📂 Arquivos Complementares

* `logistic_regression_coefficients.csv` – Coeficientes da Regressão Logística
* `random_forest_importance.csv` – Importância das variáveis no Random Forest
* `models_comparison.csv` – Tabela comparativa de desempenho

---

## 👤 Autor

**Mauro Roberto Barbosa Cahu**
📍 Recife/PE – Brasil
[🔗 LinkedIn](https://www.linkedin.com/in/mauro-cahu-159a05273)
[💻 GitHub](https://github.com/MRCahu)
📫 [maurocahu@gmail.com](mailto:maurocahu@gmail.com)

---

## 📌 Status do Projeto

* ✅ MVP funcional
* 🧪 Testes automatizados implementados
* 🛠️ Planejamento de API com FastAPI
* 📊 Integração futura com Streamlit para visualização interativa

---

## 📄 Licença

Este projeto está licenciado sob os termos da [Licença MIT](LICENSE).
