'''
 Exercício 1 – Classificação (Solar)
 Título: Previsão de nível de radiação solar
 Dataset sugerido: Solar Radiation Prediction Dataset – Kaggle
 (https://www.kaggle.com/datasets/dronio/SolarEnergy)
 Utilize o dataset de radiação solar para treinar um modelo supervisionado para classificar
 períodos em Alta Radiação e Baixa Radiação (crie a variável-alvo a partir de um limiar,
 por exemplo, a mediana da radiação). Compare o desempenho de três algoritmos do
 Scikit-learn, como Árvore de Decisão, Random Forest e Support Vector Machine
 (SVM).
 • Separe dados em treino e teste (70/30).
 • Normalize os atributos contínuos, se necessário.
 • Avalie com acurácia e matriz de confusão.
'''

# ===== IMPORTANDO BIBLIOTECAS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CARREGANDO OS DADOS =====
df = pd.read_csv('C:\\Users\\giova\\PycharmProjects\\cp2_SERS\\PARTE2\\datasets_parte2\\SolarPrediction.csv')

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
# Identificando a coluna de radiação solar (geralmente 'Radiation' ou similar)
coluna_radiacao = None
for col in ['Radiation', 'radiation', 'Solar_Radiation', 'solar_radiation']:
    if col in df.columns:
        coluna_radiacao = col
        break

if coluna_radiacao is None:
    print("\nAVISO: Coluna de radiação não encontrada. Usando primeira coluna numérica.")
    coluna_radiacao = df.select_dtypes(include=[np.number]).columns[0]

print(f"\n{'=' * 60}")
print(f"Coluna de radiação identificada: {coluna_radiacao}")

# Removendo valores faltantes
df_limpo = df.dropna()

# Criando a variável alvo (Alta/Baixa Radiação) usando a MEDIANA
mediana_radiacao = df_limpo[coluna_radiacao].median()
print(f"Mediana da radiação: {mediana_radiacao:.2f}")

# 1 = Alta Radiação (>= mediana), 0 = Baixa Radiação (< mediana)
df_limpo['Classe_Radiacao'] = (df_limpo[coluna_radiacao] >= mediana_radiacao).astype(int)

print(f"\nDistribuição das classes criadas:")
print(df_limpo['Classe_Radiacao'].value_counts())
print(f"  0 = Baixa Radiação (< {mediana_radiacao:.2f})")
print(f"  1 = Alta Radiação (>= {mediana_radiacao:.2f})")

# Separando X (variáveis independentes) e y (variável dependente)
# Remove a coluna de radiação original e a classe criada
X = df_limpo.drop(columns=[coluna_radiacao, 'Classe_Radiacao'])

# Remove colunas não numéricas
X = X.select_dtypes(include=[np.number])

y = df_limpo['Classe_Radiacao'].values

print(f"\nTotal de amostras: {len(df_limpo)}")
print(f"Número de variáveis preditoras: {X.shape[1]}")
print(f"Variáveis usadas: {X.columns.tolist()}")

# Dividindo em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nAmostras de treino: {len(X_train)} (70%)")
print(f"Amostras de teste: {len(X_test)} (30%)")

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

# 1. ÁRVORE DE DECISÃO
print("\n1. Treinando Árvore de Decisão...")
modelo_arvore = DecisionTreeClassifier(random_state=42, max_depth=10)
modelo_arvore.fit(X_train_scaled, y_train)
y_pred_arvore = modelo_arvore.predict(X_test_scaled)

# 2. RANDOM FOREST
print("2. Treinando Random Forest...")
modelo_forest = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
modelo_forest.fit(X_train_scaled, y_train)
y_pred_forest = modelo_forest.predict(X_test_scaled)

# 3. SVM (SUPPORT VECTOR MACHINE)
print("3. Treinando SVM...")
modelo_svm = SVC(kernel='rbf', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)
y_pred_svm = modelo_svm.predict(X_test_scaled)

# ===== AVALIANDO OS MODELOS =====
print(f"\n{'=' * 60}")
print("RESULTADOS - COMPARAÇÃO DOS MODELOS")
print("=" * 60)


# Função para calcular métricas
def avaliar_modelo(y_real, y_previsto, nome_modelo):
    acuracia = accuracy_score(y_real, y_previsto)
    conf_matrix = confusion_matrix(y_real, y_previsto)

    print(f"\n{nome_modelo}:")
    print(f"  Acurácia:  {acuracia:.4f} ({acuracia * 100:.2f}%)")
    print(f"  Matriz de Confusão:")
    print(f"    {conf_matrix}")

    return acuracia, conf_matrix


# Avaliando cada modelo
acc_arvore, cm_arvore = avaliar_modelo(y_test, y_pred_arvore, "ÁRVORE DE DECISÃO")
acc_forest, cm_forest = avaliar_modelo(y_test, y_pred_forest, "RANDOM FOREST")
acc_svm, cm_svm = avaliar_modelo(y_test, y_pred_svm, "SVM")

# ===== COMPARAÇÃO FINAL =====
print(f"\n{'=' * 60}")
print("RESUMO COMPARATIVO")
print("=" * 60)

resultados = pd.DataFrame({
    'Modelo': ['Árvore de Decisão', 'Random Forest', 'SVM'],
    'Acurácia': [acc_arvore, acc_forest, acc_svm]
})

print(resultados.to_string(index=False))

# Identificando o melhor modelo (maior acurácia)
melhor_idx = resultados['Acurácia'].idxmax()
melhor_modelo = resultados.loc[melhor_idx, 'Modelo']

print(f"\n MELHOR MODELO: {melhor_modelo}")
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

# Matriz 2: Random Forest
sns.heatmap(cm_forest, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_xlabel('Previsto')
axes[1].set_ylabel('Real')
axes[1].set_title(f'Random Forest\nAcurácia = {acc_forest:.4f}')

# Matriz 3: SVM
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_xlabel('Previsto')
axes[2].set_ylabel('Real')
axes[2].set_title(f'SVM\nAcurácia = {acc_svm:.4f}')

plt.tight_layout()
plt.savefig('comparacao_solar.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'comparacao_solar.png'")
plt.show()

# ===== RELATÓRIO DETALHADO DO MELHOR MODELO =====
print(f"\n{'=' * 60}")
print(f"RELATÓRIO DETALHADO - {melhor_modelo}")
print("=" * 60)

if melhor_modelo == 'Árvore de Decisão':
    print(classification_report(y_test, y_pred_arvore, target_names=['Baixa Radiação', 'Alta Radiação']))
elif melhor_modelo == 'Random Forest':
    print(classification_report(y_test, y_pred_forest, target_names=['Baixa Radiação', 'Alta Radiação']))
else:
    print(classification_report(y_test, y_pred_svm, target_names=['Baixa Radiação', 'Alta Radiação']))

