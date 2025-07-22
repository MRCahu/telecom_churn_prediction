from pathlib import Path
import pandas as pd

from src.utils import get_project_root

# Carregar os dados
data_path = get_project_root() / "data" / "TelecomX_Data.json"
df = pd.read_json(data_path)

# Achatando as colunas aninhadas
def flatten_json_col(df, col_name):
    col_df = pd.json_normalize(df[col_name])
    df = df.drop(col_name, axis=1)
    df = pd.concat([df, col_df], axis=1)
    return df

df = flatten_json_col(df, 'customer')
df = flatten_json_col(df, 'phone')
df = flatten_json_col(df, 'internet')

# Achatando a coluna 'account' primeiro para acessar 'Charges'
df_account = pd.json_normalize(df['account'])
df = df.drop('account', axis=1)
df = pd.concat([df, df_account], axis=1)

# Renomear as colunas 'Charges.Monthly' e 'Charges.Total' para 'MonthlyCharges' e 'TotalCharges'
df = df.rename(columns={'Charges.Monthly': 'MonthlyCharges', 'Charges.Total': 'TotalCharges'})

# Remover a coluna de ID do cliente
df = df.drop('customerID', axis=1)

# Tratar valores ausentes na coluna 'Churn'
df['Churn'] = df['Churn'].replace('', 'No')

# Converter a coluna 'TotalCharges' para numérica, tratando erros
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Codificar variáveis categóricas usando one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Exibir as primeiras linhas do dataframe processado
print("Primeiras 5 linhas do DataFrame processado:")
print(df_encoded.head().to_markdown(index=False, numalign='left', stralign='left'))

# Salvar o dataframe processado
processed_path = get_project_root() / "data" / "processed_data.csv"
df_encoded.to_csv(processed_path, index=False)

print(f"\nDataFrame processado e salvo em {processed_path}")
