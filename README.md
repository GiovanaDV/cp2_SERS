**Integrantes**
Allan de Souza Cardoso RM 561721

Eduardo Bacelar Rudner RM 564925

Giovana Dias Valentini RM  562390

Júlia Borges Paschoalinoto RM 564725

Raquel Amaral de Oliveira RM 566491


**PARTE 01** 


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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


**PARTE 02** 

- Ex 01

<img width="1500" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/e30a8483-a0bf-411f-ac7e-d6a5e6ebdb43" />

Análise dos resultados:

Foram testados três modelos de classificação para prever níveis de radiação solar (Alta ou Baixa) a partir de variáveis climáticas. 
A variável alvo foi criada usando a mediana da radiação como limiar, dividindo os dados em duas classes equilibradas. 

1. **Random Forest** apresentou o melhor desempenho com 87,99% de acurácia
2. Árvore de Decisão (87,76%)
3. SVM (84,95%)
 
Analisando a matriz de confusão do Random Forest, observa-se que o modelo teve melhor desempenho em identificar Baixa Radiação (97% de recall) do que Alta Radiação (79% de recall), indicando que ele é mais conservador e tende a classificar casos duvidosos como "baixa radiação". Os três modelos apresentaram desempenho satisfatório (acima de 84%), demonstrando que é possível prever níveis de radiação solar com boa confiabilidade usando dados climáticos.

--> Random Forest venceu com 88% de acertos porque é mais robusto ao combinar várias árvores. Ele é especialmente bom em detectar baixa radiação (97% de acerto), sendo o modelo mais confiável para prever condições solares.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Ex 02

<img width="1500" height="400" alt="3" src="https://github.com/user-attachments/assets/6eb2666e-3a5e-45de-a90c-3871390192a7" />

Análise dos resultados:

Foram testados três modelos de regressão para prever a potência gerada por turbinas eólicas (em kW) a partir de variáveis operacionais como velocidade do vento, direção do vento e ângulo do rotor. 

1. **Random Forest** apresentou o melhor desempenho com R² de 0.9051, explicando 90,5% da variação na potência gerada
2. Regressão Linear (R² = 0.9007)
3. Árvore de Regressão (R² = 0.8577).

O Random Forest também obteve os menores erros de previsão, com RMSE de 402,38 kW e MAE de 163,21 kW, indicando que suas previsões são, em média, mais precisas. A análise das variáveis mais importantes revelou que a **Velocidade do Vento** (48,93% de importância) e a **Curva de Potência Teórica** (47,20%) são os fatores dominantes na previsão, enquanto a Direção do Vento tem influência menor (3,86%). Isso faz sentido fisicamente, pois a potência de uma turbina eólica é diretamente proporcional ao cubo da velocidade do vento (P ∝ v³), tornando esta a variável mais crítica. Todos os três modelos apresentaram R² acima de 85%, demonstrando que é altamente viável prever a potência de turbinas eólicas a partir de dados operacionais, sendo o Random Forest o mais confiável para aplicações práticas.

--> Random Forest venceu com 90,5% de explicação porque captura melhor a relação não-linear entre vento e potência. A velocidade do vento é disparadamente o fator mais importante (49%), confirmando a física das turbinas eólicas. O modelo é altamente confiável para prever geração de energia.



















  
