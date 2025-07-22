import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados processados
df = pd.read_csv("/home/ubuntu/telecom_churn_prediction/data/processed_data.csv")

# Separar features (X) e target (y)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalização dos dados para modelos sensíveis à escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos
# 1. Regressão Logística (sensível à escala)
log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
log_reg_model.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

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
    plt.savefig(f'/home/ubuntu/telecom_churn_prediction/reports/confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Avaliar os modelos
evaluate_model("Regressão Logística", y_test, y_pred_log_reg)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Curvas ROC para comparar os modelos
from sklearn.metrics import roc_curve, auc
y_prob_log_reg = log_reg_model.predict_proba(X_test_scaled)[:, 1]
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
    'Coefficient': log_reg_model.coef_[0]
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
plt.savefig('/home/ubuntu/telecom_churn_prediction/reports/feature_importance.png', dpi=300, bbox_inches='tight')
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
models_comparison.to_csv('/home/ubuntu/telecom_churn_prediction/reports/models_comparison.csv', index=False)
log_reg_coef_sorted.to_csv('/home/ubuntu/telecom_churn_prediction/reports/logistic_regression_coefficients.csv', index=False)
rf_importance_sorted.to_csv('/home/ubuntu/telecom_churn_prediction/reports/random_forest_importance.csv', index=False)

print("\nResultados salvos em /home/ubuntu/telecom_churn_prediction/reports/")

