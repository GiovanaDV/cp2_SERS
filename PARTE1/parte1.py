'''
Parte 1 – Regressão (Appliances Energy Prediction)
Você receberá um conjunto de dados contendo informações ambientais de uma residência
(temperatura, umidade, pressão, hora do dia etc.) e o consumo de energia dos eletrodomésticos
(Wh).
• Seu desafio será prever o consumo de energia a partir das variáveis ambientais.
• Teste diferentes modelos (ex.: Regressão Linear, Árvore de Regressão, Random Forest) e
avalie o desempenho com métricas como R², RMSE e MAE.
• Compare os resultados e discuta qual modelo melhor explica o consumo.
'''

# importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# carregando dados csv
df = pd.read_csv('C:\\Users\\giova\\PycharmProjects\\cp2_SERS\\PARTE1\\datasets_parte1\\energydata_complete.csv')

# preparando dados
# Variável alvo: Appliances (consumo de energia)
# Variáveis preditoras: temperaturas, umidades, etc.

colunas_relevantes = [col for col in df.columns if col != 'Appliances' and col != 'date']
colunas_relevantes = colunas_relevantes[:10]  # Pegando as 10 primeiras para simplificar

df_limpo = df[['Appliances'] + colunas_relevantes].dropna()

# Separando X (variáveis independentes) e y (variável dependente)
X = df_limpo[colunas_relevantes].values
y = df_limpo['Appliances'].values

print(f"\n{'=' * 60}")
print(f"Variáveis selecionadas para previsão: {colunas_relevantes}")
print(f"Total de amostras após limpeza: {len(df_limpo)}")

# Dividindo em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Amostras de treino: {len(X_train)}")
print(f"Amostras de teste: {len(X_test)}")

# treinando modelos
print(f"\n{'=' * 60}")
print("TREINAMENTO DOS MODELOS")
print("=" * 60)

# 1. REGRESSÃO LINEAR
print("\n1. Treinando Regressão Linear...")
modelo_linear = LinearRegression()
modelo_linear.fit(X_train, y_train)
y_pred_linear = modelo_linear.predict(X_test)

# 2. ÁRVORE DE DECISÃO
print("2. Treinando Árvore de Decisão...")
modelo_arvore = DecisionTreeRegressor(random_state=42, max_depth=10)
modelo_arvore.fit(X_train, y_train)
y_pred_arvore = modelo_arvore.predict(X_test)

# 3. RANDOM FOREST
print("3. Treinando Random Forest...")
modelo_forest = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
modelo_forest.fit(X_train, y_train)
y_pred_forest = modelo_forest.predict(X_test)

# avaliando modelos
# Função para calcular métricas
def avaliar_modelo(y_real, y_previsto, nome_modelo):
    r2 = r2_score(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mae = mean_absolute_error(y_real, y_previsto)

    print(f"\n{nome_modelo}:")
    print(f"  R² (coeficiente de determinação): {r2:.4f}")
    print(f"  RMSE (erro médio quadrático):     {rmse:.2f} Wh")
    print(f"  MAE (erro absoluto médio):        {mae:.2f} Wh")

    return r2, rmse, mae


# Avaliando cada modelo
r2_linear, rmse_linear, mae_linear = avaliar_modelo(y_test, y_pred_linear, "REGRESSÃO LINEAR")
r2_arvore, rmse_arvore, mae_arvore = avaliar_modelo(y_test, y_pred_arvore, "ÁRVORE DE DECISÃO")
r2_forest, rmse_forest, mae_forest = avaliar_modelo(y_test, y_pred_forest, "RANDOM FOREST")

# comparando modelos
resultados = pd.DataFrame({
    'Modelo': ['Regressão Linear', 'Árvore de Decisão', 'Random Forest'],
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
