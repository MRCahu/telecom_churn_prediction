import pandas as pd
from pathlib import Path

# Paths inside the repository
DATA_DIR = Path('telecom_churn_prediction_project/telecom_churn_prediction/data')
RAW_JSON = DATA_DIR / 'TelecomX_Data.json'
EXPECTED_PROCESSED = DATA_DIR / 'processed_data.csv'


def preprocess_data(json_path: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_json(json_path)

    def flatten_json_col(df, col_name):
        col_df = pd.json_normalize(df[col_name])
        df = df.drop(col_name, axis=1)
        df = pd.concat([df, col_df], axis=1)
        return df

    df = flatten_json_col(df, 'customer')
    df = flatten_json_col(df, 'phone')
    df = flatten_json_col(df, 'internet')

    df_account = pd.json_normalize(df['account'])
    df = df.drop('account', axis=1)
    df = pd.concat([df, df_account], axis=1)

    df = df.rename(columns={'Charges.Monthly': 'MonthlyCharges',
                            'Charges.Total': 'TotalCharges'})

    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].replace('', 'No')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df_encoded.to_csv(output_csv, index=False)
    return df_encoded


def test_processed_csv_creation(tmp_path):
    output_csv = tmp_path / 'processed.csv'
    df = preprocess_data(RAW_JSON, output_csv)

    assert output_csv.exists(), 'Processed CSV was not created.'

    expected_cols = pd.read_csv(EXPECTED_PROCESSED, nrows=0).columns.tolist()
    assert list(df.columns) == expected_cols

