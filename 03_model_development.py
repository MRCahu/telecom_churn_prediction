from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils import get_project_root

# Carregar os dados processados
df = pd.read_csv(get_project_root() / "data" / "processed_data.csv")

# Separar features (X) e target (y)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Configuração de validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline para Regressão Logística com normalização
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(random_state=42, solver='liblinear'))
])

# Avaliação basal com cross_val_score
baseline_scores = cross_val_score(
    log_reg_pipeline, X_train, y_train, cv=cv, scoring='f1')
print(f"F1-Score médio (CV) - Regressão Logística: {baseline_scores.mean():.4f}")

pd.DataFrame({
    'fold': np.arange(1, len(baseline_scores) + 1),
    'f1_score': baseline_scores
}).to_csv(
    '/home/ubuntu/telecom_churn_prediction/reports/baseline_cv_scores.csv',
    index=False)

# GridSearchCV para otimização de hiperparâmetros da Regressão Logística
param_grid = {
    'log_reg__C': [0.01, 0.1, 1, 10],
    'log_reg__penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(
    log_reg_pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_log_reg_model = grid_search.best_estimator_
y_pred_log_reg = best_log_reg_model.predict(X_test)

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(
    '/home/ubuntu/telecom_churn_prediction/reports/gridsearch_logistic_regression_results.csv',
    index=False)
with open('/home/ubuntu/telecom_churn_prediction/reports/gridsearch_logistic_regression_best_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f, indent=2)

# Modelo Random Forest (não sensível à escala)

# 2. Random Forest (não sensível à escala)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Função para avaliar e imprimir métricas
def evaluate_model(model_name, y_true, y_pred):
    print(f"\n--- Avaliação do Modelo: {model_name} ---")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusão:")
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.savefig(get_project_root() / "reports" / f"confusion_matrix_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Avaliar os modelos
evaluate_model("Regressão Logística", y_test, y_pred_log_reg)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Curvas ROC para comparar os modelos
from sklearn.metrics import roc_curve, auc
y_prob_log_reg = best_log_reg_model.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log_reg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_log = auc(fpr_log, tpr_log)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Regressão Logística (AUC={roc_auc_log:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('/home/ubuntu/telecom_churn_prediction/reports/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise da importância das variáveis
print("\n--- Análise da Importância das Variáveis ---")

# Regressão Logística - Coeficientes
print("\nRegressão Logística - Top 10 Coeficientes (Importância):")
feature_names = X.columns
log_reg_coef = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': best_log_reg_model.named_steps['log_reg'].coef_[0]
})
log_reg_coef['Abs_Coefficient'] = np.abs(log_reg_coef['Coefficient'])
log_reg_coef_sorted = log_reg_coef.sort_values('Abs_Coefficient', ascending=False)
print(log_reg_coef_sorted.head(10))

# Random Forest - Importância das Features
print("\nRandom Forest - Top 10 Features Mais Importantes:")
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
})
rf_importance_sorted = rf_importance.sort_values('Importance', ascending=False)
print(rf_importance_sorted.head(10))

# Visualizar importância das variáveis
plt.figure(figsize=(15, 10))

# Regressão Logística
plt.subplot(2, 1, 1)
top_10_log_reg = log_reg_coef_sorted.head(10)
plt.barh(range(len(top_10_log_reg)), top_10_log_reg['Abs_Coefficient'])
plt.yticks(range(len(top_10_log_reg)), top_10_log_reg['Feature'])
plt.xlabel('Valor Absoluto do Coeficiente')
plt.title('Top 10 Variáveis Mais Importantes - Regressão Logística')
plt.gca().invert_yaxis()

# Random Forest
plt.subplot(2, 1, 2)
top_10_rf = rf_importance_sorted.head(10)
plt.barh(range(len(top_10_rf)), top_10_rf['Importance'])
plt.yticks(range(len(top_10_rf)), top_10_rf['Feature'])
plt.xlabel('Importância')
plt.title('Top 10 Variáveis Mais Importantes - Random Forest')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(get_project_root() / "reports" / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# Comparação dos modelos
print("\n--- Comparação dos Modelos ---")
models_comparison = pd.DataFrame({
    'Modelo': ['Regressão Logística', 'Random Forest'],
    'Acurácia': [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_rf)],
    'Precisão': [precision_score(y_test, y_pred_log_reg), precision_score(y_test, y_pred_rf)],
    'Recall': [recall_score(y_test, y_pred_log_reg), recall_score(y_test, y_pred_rf)],
    'F1-Score': [f1_score(y_test, y_pred_log_reg), f1_score(y_test, y_pred_rf)]
})
print(models_comparison)

# Salvar os resultados
reports_dir = get_project_root() / "reports"
models_comparison.to_csv(reports_dir / "models_comparison.csv", index=False)
log_reg_coef_sorted.to_csv(reports_dir / "logistic_regression_coefficients.csv", index=False)
rf_importance_sorted.to_csv(reports_dir / "random_forest_importance.csv", index=False)

print(f"\nResultados salvos em {reports_dir}")

