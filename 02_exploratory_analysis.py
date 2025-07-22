from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils import get_project_root

# Carregar os dados processados
df = pd.read_csv(get_project_root() / "data" / "processed_data.csv")

# Verificar a proporção de churn
churn_counts = df['Churn_Yes'].value_counts()
print("Proporção de Churn:")
print(f"Não churn (0): {churn_counts[0]} ({churn_counts[0]/len(df)*100:.2f}%)")
print(f"Churn (1): {churn_counts[1]} ({churn_counts[1]/len(df)*100:.2f}%)")

# Calcular a matriz de correlação
correlation_matrix = df.corr()

# Criar visualizações
plt.figure(figsize=(20, 16))

# Matriz de correlação
plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Correlação com Churn
plt.subplot(2, 2, 2)
churn_corr = correlation_matrix['Churn_Yes'].sort_values(ascending=False)
churn_corr = churn_corr[churn_corr.index != 'Churn_Yes']  # Remove a própria variável
top_corr = churn_corr.head(10)
bottom_corr = churn_corr.tail(10)
combined_corr = pd.concat([top_corr, bottom_corr])

plt.barh(range(len(combined_corr)), combined_corr.values)
plt.yticks(range(len(combined_corr)), combined_corr.index)
plt.xlabel('Correlação com Churn')
plt.title('Top 10 Correlações Positivas e Negativas com Churn')
plt.grid(axis='x', alpha=0.3)

# Distribuição de MonthlyCharges por Churn
plt.subplot(2, 2, 3)
churn_0 = df[df['Churn_Yes'] == 0]['MonthlyCharges']
churn_1 = df[df['Churn_Yes'] == 1]['MonthlyCharges']
plt.hist(churn_0, alpha=0.7, label='Não Churn', bins=30, density=True)
plt.hist(churn_1, alpha=0.7, label='Churn', bins=30, density=True)
plt.xlabel('MonthlyCharges')
plt.ylabel('Densidade')
plt.title('Distribuição de MonthlyCharges por Churn')
plt.legend()

# Distribuição de tenure por Churn
plt.subplot(2, 2, 4)
churn_0_tenure = df[df['Churn_Yes'] == 0]['tenure']
churn_1_tenure = df[df['Churn_Yes'] == 1]['tenure']
plt.hist(churn_0_tenure, alpha=0.7, label='Não Churn', bins=30, density=True)
plt.hist(churn_1_tenure, alpha=0.7, label='Churn', bins=30, density=True)
plt.xlabel('Tenure (meses)')
plt.ylabel('Densidade')
plt.title('Distribuição de Tenure por Churn')
plt.legend()

plt.tight_layout()
plt.savefig(get_project_root() / "reports" / "exploratory_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Distribuição de churn por tipo de contrato
contract_type = np.where(df['Contract_One year'] == 1, 'One year',
                         np.where(df['Contract_Two year'] == 1, 'Two year', 'Month-to-month'))
contract_churn = pd.crosstab(contract_type, df['Churn_Yes'], normalize='index')
contract_churn.plot(kind='bar', stacked=True, figsize=(6, 4))
plt.xlabel('Tipo de Contrato')
plt.ylabel('Proporção')
plt.title('Distribuição de Churn por Tipo de Contrato')
plt.legend(['Não Churn', 'Churn'], loc='upper right')
plt.tight_layout()
plt.savefig('/home/ubuntu/telecom_churn_prediction/reports/churn_by_contract.png', dpi=300, bbox_inches='tight')
plt.show()

# Estatísticas descritivas por grupo de churn
print("\nEstatísticas descritivas por grupo de churn:")
print("\nNão Churn (0):")
print(df[df['Churn_Yes'] == 0][['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].describe())
print("\nChurn (1):")
print(df[df['Churn_Yes'] == 1][['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].describe())

# Salvar as correlações mais importantes
print("\nTop 15 correlações com Churn:")
print(churn_corr.head(15))
print("\nBottom 15 correlações com Churn:")
print(churn_corr.tail(15))
