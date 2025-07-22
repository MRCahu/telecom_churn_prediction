import logging
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed CSV data and split it into train and test sets."""
    logging.info("Loading processed data from %s", path)
    df = pd.read_csv(path)
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logging.debug("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[LogisticRegression, RandomForestClassifier, StandardScaler]:
    """Train Logistic Regression and Random Forest models."""
    logging.info("Training models")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    log_reg_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    logging.info("Models trained")
    return log_reg_model, rf_model, scaler


def evaluate_models(log_reg_model: LogisticRegression, rf_model: RandomForestClassifier,
                    scaler: StandardScaler, X_test: pd.DataFrame, y_test: pd.Series,
                    feature_names: pd.Index) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Evaluate models and return metrics and auxiliary data."""
    logging.info("Evaluating models")
    y_pred_log_reg = log_reg_model.predict(scaler.transform(X_test))
    y_pred_rf = rf_model.predict(X_test)

    metrics = pd.DataFrame([
        {
            'Modelo': 'Regressao Logistica',
            'Acuracia': accuracy_score(y_test, y_pred_log_reg),
            'Precisao': precision_score(y_test, y_pred_log_reg),
            'Recall': recall_score(y_test, y_pred_log_reg),
            'F1-Score': f1_score(y_test, y_pred_log_reg)
        },
        {
            'Modelo': 'Random Forest',
            'Acuracia': accuracy_score(y_test, y_pred_rf),
            'Precisao': precision_score(y_test, y_pred_rf),
            'Recall': recall_score(y_test, y_pred_rf),
            'F1-Score': f1_score(y_test, y_pred_rf)
        }
    ])

    cm_log = confusion_matrix(y_test, y_pred_log_reg)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    log_reg_coef = pd.DataFrame({'Feature': feature_names, 'Coefficient': log_reg_model.coef_[0]})
    log_reg_coef['Abs_Coefficient'] = log_reg_coef['Coefficient'].abs()
    log_reg_coef_sorted = log_reg_coef.sort_values('Abs_Coefficient', ascending=False)

    rf_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
    rf_importance_sorted = rf_importance.sort_values('Importance', ascending=False)

    logging.info("Evaluation complete")
    return metrics, cm_log, cm_rf, log_reg_coef_sorted, rf_importance_sorted


def save_reports(metrics: pd.DataFrame, cm_log: np.ndarray, cm_rf: np.ndarray,
                 log_reg_coef: pd.DataFrame, rf_importance: pd.DataFrame,
                 output_dir: str = 'reports') -> None:
    """Save evaluation reports and figures."""
    logging.info("Saving reports to %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics.to_csv(os.path.join(output_dir, 'models_comparison.csv'), index=False)
    log_reg_coef.to_csv(os.path.join(output_dir, 'logistic_regression_coefficients.csv'), index=False)
    rf_importance.to_csv(os.path.join(output_dir, 'random_forest_importance.csv'), index=False)

    for cm, name in zip([cm_log, cm_rf], ['Regressao_Logistica', 'Random_Forest']):
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusao - {name.replace("_", " ")}')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    top_log = log_reg_coef.head(10)
    plt.barh(range(len(top_log)), top_log['Abs_Coefficient'])
    plt.yticks(range(len(top_log)), top_log['Feature'])
    plt.xlabel('Valor Absoluto do Coeficiente')
    plt.title('Top 10 Variaveis - Regressao Logistica')
    plt.gca().invert_yaxis()

    plt.subplot(2, 1, 2)
    top_rf = rf_importance.head(10)
    plt.barh(range(len(top_rf)), top_rf['Importance'])
    plt.yticks(range(len(top_rf)), top_rf['Feature'])
    plt.xlabel('Importancia')
    plt.title('Top 10 Variaveis - Random Forest')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Reports saved")


def main() -> None:
    data_path = 'processed_data.csv'
    X_train, X_test, y_train, y_test = load_data(data_path)
    log_reg_model, rf_model, scaler = train_models(X_train, y_train)
    metrics, cm_log, cm_rf, log_reg_coef, rf_importance = evaluate_models(
        log_reg_model, rf_model, scaler, X_test, y_test, X_train.columns
    )
    save_reports(metrics, cm_log, cm_rf, log_reg_coef, rf_importance)


if __name__ == "__main__":
    main()
