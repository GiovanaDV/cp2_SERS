'''
 Exercício 2 – Regressão (Eólica)
 Título: Previsão de potência de turbinas eólicas
 Dataset sugerido: Wind Turbine Scada Dataset – Kaggle
 (https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)
 Com base nos dados de operação de turbinas eólicas (velocidade do vento, ângulo do
 rotor, densidade do ar, etc.), treine modelos de regressão para prever a potência gerada
 (kW). Compare o desempenho de três algoritmos do Scikit-learn, como Regressão
 Linear, Regressão de Árvores e Random Forest Regressor.
 • Separe treino e teste (80/20).
 • Normalize os dados, se necessário.
 • Avalie com RMSE e R²
'''

# ===== IMPORTANDO BIBLIOTECAS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ===== CARREGANDO OS DADOS =====
df = pd.read_csv('C:\\Users\\giova\\PycharmProjects\\cp2_SERS\\PARTE2\\datasets_parte2\\T1.csv')

print("=" * 60)
print("ANÁLISE DO DATASET")
print("=" * 60)
print(f"Dimensões do dataset: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head())
print(f"\nColunas disponíveis:")
print(df.columns.tolist())
print(f"\nInformações gerais:")
print(df.info())

# ===== PREPARANDO OS DADOS =====
# Identificando a coluna de potência
coluna_potencia = None
for col in ['ActivePower', 'LV ActivePower (kW)', 'Active_Power', 'Power', 'power']:
    if col in df.columns:
        coluna_potencia = col
        break

if coluna_potencia is None:
    print("\nAVISO: Coluna de potência não encontrada. Procurando por 'kW' no nome...")
    for col in df.columns:
        if 'kW' in col or 'power' in col.lower():
            coluna_potencia = col
            break

if coluna_potencia is None:
    print("Usando última coluna numérica como potência")
    coluna_potencia = df.select_dtypes(include=[np.number]).columns[-1]

print(f"\n{'=' * 60}")
print(f"Coluna de potência identificada: {coluna_potencia}")

# Removendo valores faltantes
df_limpo = df.dropna()

# Estatísticas da potência
print(f"\nEstatísticas da Potência:")
print(f"  Mínimo: {df_limpo[coluna_potencia].min():.2f} kW")
print(f"  Máximo: {df_limpo[coluna_potencia].max():.2f} kW")
print(f"  Média: {df_limpo[coluna_potencia].mean():.2f} kW")
print(f"  Mediana: {df_limpo[coluna_potencia].median():.2f} kW")

# Separando X (variáveis independentes) e y (variável dependente)
X = df_limpo.drop(columns=[coluna_potencia])

# Remove colunas não numéricas
colunas_originais = X.columns.tolist()
X = X.select_dtypes(include=[np.number])

if len(colunas_originais) > len(X.columns):
    removidas = set(colunas_originais) - set(X.columns)
    print(f"\nColunas não numéricas removidas: {removidas}")

y = df_limpo[coluna_potencia].values

print(f"\nTotal de amostras: {len(df_limpo)}")
print(f"Número de variáveis preditoras: {X.shape[1]}")
print(f"Variáveis usadas: {X.columns.tolist()}")

# Dividindo em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nAmostras de treino: {len(X_train)} (80%)")
print(f"Amostras de teste: {len(X_test)} (20%)")

# ===== NORMALIZANDO OS DADOS =====
print(f"\n{'=' * 60}")
print("NORMALIZANDO OS ATRIBUTOS")
print("=" * 60)

# Normalização com StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dados normalizados com StandardScaler (média=0, desvio=1)")

# ===== TREINANDO OS MODELOS =====
print(f"\n{'=' * 60}")
print("TREINAMENTO DOS MODELOS")
print("=" * 60)

# 1. REGRESSÃO LINEAR
print("\n1. Treinando Regressão Linear...")
modelo_linear = LinearRegression()
modelo_linear.fit(X_train_scaled, y_train)
y_pred_linear = modelo_linear.predict(X_test_scaled)

# 2. ÁRVORE DE REGRESSÃO
print("2. Treinando Árvore de Regressão...")
modelo_arvore = DecisionTreeRegressor(random_state=42, max_depth=15)
modelo_arvore.fit(X_train_scaled, y_train)
y_pred_arvore = modelo_arvore.predict(X_test_scaled)

# 3. RANDOM FOREST REGRESSOR
print("3. Treinando Random Forest Regressor...")
modelo_forest = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
modelo_forest.fit(X_train_scaled, y_train)
y_pred_forest = modelo_forest.predict(X_test_scaled)

# ===== AVALIANDO OS MODELOS =====
print(f"\n{'=' * 60}")
print("RESULTADOS - COMPARAÇÃO DOS MODELOS")
print("=" * 60)


# Função para calcular métricas
def avaliar_modelo(y_real, y_previsto, nome_modelo):
    r2 = r2_score(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mae = mean_absolute_error(y_real, y_previsto)

    print(f"\n{nome_modelo}:")
    print(f"  R² (coeficiente de determinação): {r2:.4f}")
    print(f"  RMSE (erro médio quadrático):     {rmse:.2f} kW")
    print(f"  MAE (erro absoluto médio):        {mae:.2f} kW")

    return r2, rmse, mae


# Avaliando cada modelo
r2_linear, rmse_linear, mae_linear = avaliar_modelo(y_test, y_pred_linear, "REGRESSÃO LINEAR")
r2_arvore, rmse_arvore, mae_arvore = avaliar_modelo(y_test, y_pred_arvore, "ÁRVORE DE REGRESSÃO")
r2_forest, rmse_forest, mae_forest = avaliar_modelo(y_test, y_pred_forest, "RANDOM FOREST")

# ===== COMPARAÇÃO FINAL =====
print(f"\n{'=' * 60}")
print("RESUMO COMPARATIVO")
print("=" * 60)

resultados = pd.DataFrame({
    'Modelo': ['Regressão Linear', 'Árvore de Regressão', 'Random Forest'],
    'R²': [r2_linear, r2_arvore, r2_forest],
    'RMSE': [rmse_linear, rmse_arvore, rmse_forest],
    'MAE': [mae_linear, mae_arvore, mae_forest]
})

print(resultados.to_string(index=False))

# Identificando o melhor modelo (maior R²)
melhor_idx = resultados['R²'].idxmax()
melhor_modelo = resultados.loc[melhor_idx, 'Modelo']

print(f"\n MELHOR MODELO: {melhor_modelo}")
print(f"   (possui o maior R² = {resultados.loc[melhor_idx, 'R²']:.4f})")

# ===== VISUALIZAÇÃO DOS RESULTADOS =====
print(f"\n{'=' * 60}")
print("Gerando gráficos de comparação...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Gráfico 1: Regressão Linear
axes[0].scatter(y_test, y_pred_linear, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Potência Real (kW)')
axes[0].set_ylabel('Potência Prevista (kW)')
axes[0].set_title(f'Regressão Linear\nR² = {r2_linear:.4f}\nRMSE = {rmse_linear:.2f} kW')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Árvore de Regressão
axes[1].scatter(y_test, y_pred_arvore, alpha=0.3, s=10, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Potência Real (kW)')
axes[1].set_ylabel('Potência Prevista (kW)')
axes[1].set_title(f'Árvore de Regressão\nR² = {r2_arvore:.4f}\nRMSE = {rmse_arvore:.2f} kW')
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Random Forest
axes[2].scatter(y_test, y_pred_forest, alpha=0.3, s=10, color='orange')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Potência Real (kW)')
axes[2].set_ylabel('Potência Prevista (kW)')
axes[2].set_title(f'Random Forest\nR² = {r2_forest:.4f}\nRMSE = {rmse_forest:.2f} kW')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacao_eolica.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'comparacao_eolica.png'")
plt.show()

# ===== ANÁLISE DAS VARIÁVEIS MAIS IMPORTANTES (RANDOM FOREST) =====
if melhor_modelo == 'Random Forest':
    print(f"\n{'=' * 60}")
    print("VARIÁVEIS MAIS IMPORTANTES - Random Forest")
    print("=" * 60)

    importancias = modelo_forest.feature_importances_
    indices = np.argsort(importancias)[::-1][:5]  # Top 5

    print("\nTop 5 variáveis mais importantes:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {X.columns[idx]}: {importancias[idx]:.4f}")

