'''
Parte 2 – Classificação (Smart Grid Stability)
Você receberá um conjunto de dados simulados de uma rede elétrica inteligente, com variáveis
como potência ativa, potência reativa, tensão e corrente. O dataset indica se a rede está estável ou
instável.
• Seu desafio será classificar a estabilidade da rede com base nas variáveis fornecidas.
• Treine modelos de classificação (ex.: Árvore de Decisão, KNN, Regressão Logística) e avalie o
desempenho com acurácia, matriz de confusão e F1-score.
• Compare os resultados e discuta qual modelo é mais confiável para detectar instabilidade.
'''

# ===== IMPORTANDO BIBLIOTECAS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CARREGANDO OS DADOS =====
# Substitua pelo caminho ou URL do seu dataset de Smart Grid
df = pd.read_csv('C:\\Users\\giova\\PycharmProjects\\cp2_SERS\\datasets_parte1\\smart_grid_stability_augmented.csv')

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
# Variável alvo: stabf ou stab (estável=1 ou instável=0)
# Ajuste o nome da coluna alvo conforme seu dataset

# Identificando a coluna alvo (geralmente é 'stabf' ou 'stab')
coluna_alvo = None
for col in ['stabf', 'stab', 'stability', 'stable']:
    if col in df.columns:
        coluna_alvo = col
        break

if coluna_alvo is None:
    print("\nAVISO: Não encontrei coluna de estabilidade. Usando última coluna.")
    coluna_alvo = df.columns[-1]

print(f"\n{'=' * 60}")
print(f"Coluna alvo (estabilidade): {coluna_alvo}")

# Verificando distribuição das classes
print(f"\nDistribuição das classes:")
print(df[coluna_alvo].value_counts())

# Removendo valores faltantes
df_limpo = df.dropna()

# Separando X (variáveis independentes) e y (variável dependente)
X = df_limpo.drop(columns=[coluna_alvo]).values
y = df_limpo[coluna_alvo].values

# Se y for numérico contínuo, converte para binário (estável/instável)
if y.dtype == 'float64':
    y = (y >= 0).astype(int)  # 1 = estável, 0 = instável

print(f"Total de amostras após limpeza: {len(df_limpo)}")
print(f"Número de variáveis preditoras: {X.shape[1]}")

# Dividindo em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Amostras de treino: {len(X_train)}")
print(f"Amostras de teste: {len(X_test)}")

# ===== TREINANDO OS MODELOS =====
print(f"\n{'=' * 60}")
print("TREINAMENTO DOS MODELOS")
print("=" * 60)

# 1. ÁRVORE DE DECISÃO
print("\n1. Treinando Árvore de Decisão...")
modelo_arvore = DecisionTreeClassifier(random_state=42, max_depth=10)
modelo_arvore.fit(X_train, y_train)
y_pred_arvore = modelo_arvore.predict(X_test)

# 2. KNN (K-NEAREST NEIGHBORS)
print("2. Treinando KNN (K=5)...")
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train, y_train)
y_pred_knn = modelo_knn.predict(X_test)

# 3. REGRESSÃO LOGÍSTICA
print("3. Treinando Regressão Logística...")
modelo_logistica = LogisticRegression(random_state=42, max_iter=1000)
modelo_logistica.fit(X_train, y_train)
y_pred_logistica = modelo_logistica.predict(X_test)

# ===== AVALIANDO OS MODELOS =====
print(f"\n{'=' * 60}")
print("RESULTADOS - COMPARAÇÃO DOS MODELOS")
print("=" * 60)


# Função para calcular métricas
def avaliar_modelo(y_real, y_previsto, nome_modelo):
    acuracia = accuracy_score(y_real, y_previsto)
    f1 = f1_score(y_real, y_previsto, average='weighted')
    conf_matrix = confusion_matrix(y_real, y_previsto)

    print(f"\n{nome_modelo}:")
    print(f"  Acurácia:  {acuracia:.4f} ({acuracia * 100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Matriz de Confusão:")
    print(f"    {conf_matrix}")

    return acuracia, f1, conf_matrix


# Avaliando cada modelo
acc_arvore, f1_arvore, cm_arvore = avaliar_modelo(y_test, y_pred_arvore, "ÁRVORE DE DECISÃO")
acc_knn, f1_knn, cm_knn = avaliar_modelo(y_test, y_pred_knn, "KNN")
acc_logistica, f1_logistica, cm_logistica = avaliar_modelo(y_test, y_pred_logistica, "REGRESSÃO LOGÍSTICA")

# ===== COMPARAÇÃO FINAL =====
print(f"\n{'=' * 60}")
print("RESUMO COMPARATIVO")
print("=" * 60)

resultados = pd.DataFrame({
    'Modelo': ['Árvore de Decisão', 'KNN', 'Regressão Logística'],
    'Acurácia': [acc_arvore, acc_knn, acc_logistica],
    'F1-Score': [f1_arvore, f1_knn, f1_logistica]
})

print(resultados.to_string(index=False))

# Identificando o melhor modelo (maior acurácia)
melhor_idx = resultados['Acurácia'].idxmax()
melhor_modelo = resultados.loc[melhor_idx, 'Modelo']

print(f"\n🏆 MELHOR MODELO: {melhor_modelo}")
print(f"   (possui a maior acurácia = {resultados.loc[melhor_idx, 'Acurácia']:.4f})")

# ===== VISUALIZAÇÃO DAS MATRIZES DE CONFUSÃO =====
print(f"\n{'=' * 60}")
print("Gerando gráficos das matrizes de confusão...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Matriz 1: Árvore de Decisão
sns.heatmap(cm_arvore, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Previsto')
axes[0].set_ylabel('Real')
axes[0].set_title(f'Árvore de Decisão\nAcurácia = {acc_arvore:.4f}')

# Matriz 2: KNN
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_xlabel('Previsto')
axes[1].set_ylabel('Real')
axes[1].set_title(f'KNN\nAcurácia = {acc_knn:.4f}')

# Matriz 3: Regressão Logística
sns.heatmap(cm_logistica, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_xlabel('Previsto')
axes[2].set_ylabel('Real')
axes[2].set_title(f'Regressão Logística\nAcurácia = {acc_logistica:.4f}')

plt.tight_layout()
plt.savefig('comparacao_classificacao.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'comparacao_classificacao.png'")
plt.show()

# ===== RELATÓRIO DETALHADO DO MELHOR MODELO =====
print(f"\n{'=' * 60}")
print(f"RELATÓRIO DETALHADO - {melhor_modelo}")
print("=" * 60)

if melhor_modelo == 'Árvore de Decisão':
    print(classification_report(y_test, y_pred_arvore, target_names=['Instável', 'Estável']))
elif melhor_modelo == 'KNN':
    print(classification_report(y_test, y_pred_knn, target_names=['Instável', 'Estável']))
else:
    print(classification_report(y_test, y_pred_logistica, target_names=['Instável', 'Estável']))

print(f"\n{'=' * 60}")
print("ANÁLISE COMPLETA!")
print("=" * 60)