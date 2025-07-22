import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = Path('telecom_churn_prediction_project/telecom_churn_prediction/data')
PROCESSED_CSV = DATA_DIR / 'processed_data.csv'


def train_and_evaluate(df: pd.DataFrame):
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(random_state=42, solver='liblinear')
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log_reg = log_reg.predict(X_test_scaled)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        'log_reg': {
            'accuracy': accuracy_score(y_test, y_pred_log_reg),
            'precision': precision_score(y_test, y_pred_log_reg),
            'recall': recall_score(y_test, y_pred_log_reg),
            'f1': f1_score(y_test, y_pred_log_reg),
        },
        'rf': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
        },
    }
    return log_reg, rf, metrics


def test_train_and_evaluate_models():
    df = pd.read_csv(PROCESSED_CSV)
    log_reg, rf, metrics = train_and_evaluate(df)

    assert isinstance(log_reg, LogisticRegression)
    assert isinstance(rf, RandomForestClassifier)

    for name in ('log_reg', 'rf'):
        m = metrics[name]
        assert 0.7 <= m['accuracy'] <= 1.0
        assert 0.4 <= m['precision'] <= 1.0
        assert 0.3 <= m['recall'] <= 1.0
        assert 0.4 <= m['f1'] <= 1.0

