PARTE 01 


- Ex 01

Análise dos resultados:

Foram testados três modelos de aprendizado de máquina para prever o consumo de energia de eletrodomésticos a partir de variáveis ambientais (temperatura, umidade, etc.). 

1. **Random Forest** apresentou o melhor desempenho com R² de 0.32
2.  Regressão Linear (R² = 0.13)
3.  Árvore de Decisão (R² = 0.12).

Embora o R² de 0.32 indique que o modelo explica apenas 32% da variação no consumo de energia, ele ainda foi superior aos demais. 
O Random Forest também obteve os menores erros de previsão (RMSE = 82.30 Wh e MAE = 44.53 Wh), mostrando previsões mais precisas.
Os valores relativamente baixos de R² em todos os modelos sugerem que o consumo de energia dos eletrodomésticos é influenciado por 
outros fatores não presentes no dataset (como horário de uso, comportamento dos moradores, tipo de eletrodoméstico ligado), sendo difícil de prever apenas com dados ambientais

--> Random Forest venceu porque é mais "inteligente" e consegue encontrar padrões complicados, mas mesmo assim nenhum modelo 
foi perfeito porque faltam informações importantes no dataset (tipo: quando as pessoas ligam os aparelhos).

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Ex 02

<img width="1500" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/f8af0ea6-38ae-4028-b4d5-87220fdd8bc1" />

Análise dos resultados: 

Foram testados três modelos de classificação para prever a estabilidade de uma rede elétrica inteligente com base em variáveis como potência, tensão e corrente. 

1. **Árvore de Decisão** obteve desempenho perfeito com 100% de acurácia e F1-Score de 1.0, classificando corretamente todos os 12.000 casos de teste, tanto redes estáveis quanto instáveis. 
2. Regressão Logística também apresentou excelente desempenho (96,3% de acurácia)
3. KNN teve a menor performance (83,3% de acurácia)
 
O resultado perfeito da Árvore de Decisão indica que existe um padrão muito claro e bem definido nos dados que separa redes estáveis de instáveis, permitindo que o modelo aprenda perfeitamente as 
"regras" de decisão. O KNN teve pior desempenho porque depende da distância entre pontos e pode ser confundido por dados muito próximos das fronteiras de decisão, 
mostrando-se menos eficaz para esse tipo de problema.

--> A Árvore de Decisão acertou 100% porque os dados de rede elétrica seguem padrões físicos muito claros e previsíveis, criando "regras" perfeitas que o modelo conseguiu aprender completamente. 
É o modelo mais confiável para detectar instabilidade nesta rede.

