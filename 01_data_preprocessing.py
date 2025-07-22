import pandas as pd

# Carregar os dados
file_path = '/home/ubuntu/telecom_churn_prediction/data/TelecomX_Data.json'
df = pd.read_json(file_path)

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
df_encoded.to_csv('/home/ubuntu/telecom_churn_prediction/data/processed_data.csv', index=False)

print("\nDataFrame processado e salvo em /home/ubuntu/telecom_churn_prediction/data/processed_data.csv")


