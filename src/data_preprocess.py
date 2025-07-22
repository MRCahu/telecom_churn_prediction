import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw JSON data from the given path."""
    logging.info("Loading raw data from %s", path)
    df = pd.read_json(path)
    logging.debug("Raw data shape: %s", df.shape)
    return df


def flatten_json_col(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Flatten a nested JSON column and concatenate with the DataFrame."""
    logging.debug("Flattening column %s", col_name)
    col_df = pd.json_normalize(df[col_name])
    df = df.drop(col_name, axis=1)
    return pd.concat([df, col_df], axis=1)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing steps to the raw DataFrame."""
    df = flatten_json_col(df, 'customer')
    df = flatten_json_col(df, 'phone')
    df = flatten_json_col(df, 'internet')

    account_df = pd.json_normalize(df['account'])
    df = df.drop('account', axis=1)
    df = pd.concat([df, account_df], axis=1)

    df = df.rename(columns={'Charges.Monthly': 'MonthlyCharges',
                            'Charges.Total': 'TotalCharges'})
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['Churn'] = df['Churn'].replace('', 'No')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logging.info("Preprocessing completed. Data shape: %s", df.shape)
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save the processed DataFrame to CSV."""
    logging.info("Saving processed data to %s", path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    raw_path = 'TelecomX_Data.json'
    processed_path = 'processed_data.csv'
    df_raw = load_raw_data(raw_path)
    df_processed = preprocess(df_raw)
    save_processed(df_processed, processed_path)


if __name__ == "__main__":
    main()
