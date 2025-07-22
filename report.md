# Relatório de Previsão de Churn na Telecom X

**Autor:** Mauro Cahu  
**Data:** julho 2025  
**Projeto:** Telecom X - Parte 2: Prevendo Churn

## Resumo Executivo

Este relatório apresenta uma análise abrangente dos dados de clientes da Telecom X com o objetivo de desenvolver modelos preditivos para identificar clientes com maior probabilidade de evasão (churn). O projeto utilizou técnicas de Machine Learning para analisar padrões comportamentais e características dos clientes, resultando em insights valiosos para estratégias de retenção.

Os principais resultados indicam que o modelo de Regressão Logística apresentou melhor desempenho geral, com acurácia de 80,1% e F1-Score de 57,9%. As variáveis mais importantes para predição de churn incluem tempo de permanência (tenure), valor das cobranças mensais (MonthlyCharges), tipo de serviço de internet (Fiber optic) e valor total gasto (TotalCharges).




## 1. Introdução

A evasão de clientes (churn) representa um dos principais desafios enfrentados por empresas de telecomunicações, impactando diretamente a receita e a sustentabilidade do negócio. No contexto altamente competitivo do setor de telecomunicações, onde a aquisição de novos clientes é significativamente mais custosa que a retenção dos existentes, a capacidade de identificar e prevenir a evasão torna-se um diferencial estratégico fundamental.

A Telecom X, reconhecendo a importância crítica desta questão, iniciou um projeto abrangente de análise preditiva para compreender os fatores que influenciam a decisão dos clientes de cancelar seus serviços. Este projeto representa a segunda fase de uma iniciativa mais ampla, construindo sobre os insights obtidos na análise exploratória inicial dos dados de churn.

O objetivo principal deste trabalho é desenvolver modelos de Machine Learning capazes de prever com precisão quais clientes apresentam maior probabilidade de evasão, permitindo à empresa implementar estratégias proativas de retenção. Através da análise de dados históricos de 7.267 clientes, buscamos identificar padrões comportamentais, características demográficas e preferências de serviços que estão correlacionados com a decisão de churn.

A metodologia adotada segue as melhores práticas da ciência de dados, incluindo pré-processamento rigoroso dos dados, análise exploratória detalhada, desenvolvimento de múltiplos modelos preditivos e avaliação comparativa de desempenho. Os modelos desenvolvidos não apenas fornecem previsões precisas, mas também oferecem insights interpretáveis sobre os fatores mais influentes na evasão de clientes.

Este relatório documenta todo o processo analítico, desde a preparação inicial dos dados até a interpretação final dos resultados, fornecendo uma base sólida para a tomada de decisões estratégicas e a implementação de programas de retenção de clientes mais eficazes.


## 2. Metodologia

### 2.1 Fonte e Características dos Dados

O conjunto de dados utilizado neste estudo contém informações detalhadas de 7.267 clientes da Telecom X, organizadas em estrutura JSON hierárquica com múltiplas dimensões de informação. Os dados originais foram estruturados em cinco categorias principais: identificação do cliente, informações demográficas e de relacionamento, serviços telefônicos, serviços de internet, e informações contratuais e financeiras.

A estrutura original dos dados apresentava características aninhadas que requereram processamento especializado para análise. As informações demográficas incluíam gênero, status de cidadão sênior, presença de parceiro e dependentes, além do tempo de permanência como cliente (tenure). Os dados de serviços telefônicos abrangiam a utilização de serviços telefônicos e múltiplas linhas, enquanto os serviços de internet incluíam tipo de conexão e diversos serviços adicionais como segurança online, backup, proteção de dispositivos, suporte técnico e streaming.

As informações contratuais e financeiras compreendiam tipo de contrato, método de faturamento, forma de pagamento e valores de cobrança mensal e total. A variável target (churn) indicava se o cliente havia cancelado os serviços, apresentando três categorias: "Yes" (evasão confirmada), "No" (cliente ativo) e valores em branco (tratados como "No" durante o pré-processamento).

### 2.2 Pré-processamento de Dados

O processo de pré-processamento foi estruturado em múltiplas etapas para garantir a qualidade e adequação dos dados para modelagem de Machine Learning. Inicialmente, realizamos a normalização da estrutura JSON aninhada, convertendo as colunas hierárquicas em formato tabular plano através da função `pd.json_normalize()`.

A limpeza dos dados incluiu a remoção da coluna de identificação do cliente (customerID), considerada irrelevante para a predição de churn por ser um identificador único sem valor preditivo. Valores ausentes na variável target foram tratados através da substituição de strings vazias por "No", assumindo que clientes sem indicação explícita de churn permanecem ativos.

A conversão de tipos de dados foi realizada com particular atenção à coluna TotalCharges, que apresentava valores não numéricos. Utilizamos a função `pd.to_numeric()` com tratamento de erros, substituindo valores inválidos pela mediana da distribuição para manter a integridade estatística dos dados.

Para a codificação de variáveis categóricas, implementamos one-hot encoding através da função `pd.get_dummies()` com o parâmetro `drop_first=True` para evitar multicolinearidade. Esta abordagem criou variáveis binárias para cada categoria, permitindo que algoritmos de Machine Learning processem adequadamente as informações categóricas.

### 2.3 Análise Exploratória

A análise exploratória foi conduzida com foco na compreensão da distribuição da variável target e na identificação de padrões nos dados. Calculamos a proporção de churn, revelando um desequilíbrio moderado com 25,72% dos clientes apresentando evasão e 74,28% permanecendo ativos.

A análise de correlação foi realizada através da matriz de correlação de Pearson, permitindo identificar as relações lineares entre variáveis numéricas e a variável target. Visualizações específicas foram criadas para examinar a distribuição de variáveis-chave como MonthlyCharges e tenure em relação ao status de churn.

Estatísticas descritivas foram calculadas separadamente para grupos de clientes com e sem churn, revelando diferenças significativas em características como tempo de permanência, valores de cobrança e perfil demográfico. Esta análise forneceu insights preliminares sobre os fatores potencialmente associados à evasão de clientes.

### 2.4 Desenvolvimento de Modelos

A estratégia de modelagem incluiu a seleção de dois algoritmos complementares com características distintas em relação à sensibilidade à escala dos dados. A divisão dos dados seguiu a proporção 70% para treinamento e 30% para teste, utilizando estratificação para manter a proporção original de churn em ambos os conjuntos.

Para modelos sensíveis à escala, implementamos normalização através do StandardScaler, que padroniza as variáveis para média zero e desvio padrão unitário. Esta etapa é crucial para algoritmos como Regressão Logística, que podem ser influenciados por diferenças na magnitude das variáveis.

O primeiro modelo implementado foi a Regressão Logística, escolhida por sua interpretabilidade e eficácia em problemas de classificação binária. Utilizamos o solver 'liblinear' adequado para datasets de tamanho moderado e problemas binários. O segundo modelo foi o Random Forest, selecionado por sua robustez, capacidade de lidar com interações complexas entre variáveis e resistência a overfitting.

### 2.5 Avaliação de Modelos

A avaliação dos modelos foi realizada através de múltiplas métricas para fornecer uma visão abrangente do desempenho. As métricas incluíram acurácia (proporção de predições corretas), precisão (proporção de verdadeiros positivos entre as predições positivas), recall (proporção de verdadeiros positivos identificados) e F1-Score (média harmônica entre precisão e recall).

A matriz de confusão foi utilizada para análise detalhada dos tipos de erro, permitindo identificar a distribuição de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos. Esta análise é particularmente importante em problemas de churn, onde diferentes tipos de erro têm implicações de negócio distintas.

A interpretabilidade dos modelos foi explorada através da análise dos coeficientes da Regressão Logística e da importância das variáveis no Random Forest. Esta análise permite compreender quais características dos clientes são mais influentes na predição de churn, fornecendo insights acionáveis para estratégias de retenção.


## 3. Resultados e Análises

### 3.1 Avaliação e Comparação dos Modelos

Dois modelos de classificação foram desenvolvidos e avaliados para prever a evasão de clientes: Regressão Logística e Random Forest. A escolha desses modelos se deu pela sua aplicabilidade em problemas de classificação e pela diferença em sua sensibilidade à escala dos dados, permitindo uma análise comparativa robusta.

A Regressão Logística, um modelo linear, foi treinada com dados padronizados (normalizados) usando `StandardScaler`, conforme a necessidade de algoritmos baseados em distância. O Random Forest, um modelo baseado em árvores, foi treinado com os dados originais (não normalizados), pois não é sensível à escala das variáveis.

A tabela abaixo resume o desempenho de ambos os modelos nas métricas de acurácia, precisão, recall e F1-Score no conjunto de teste:

| Modelo              | Acurácia | Precisão | Recall | F1-Score |
|:--------------------|:---------|:---------|:-------|:---------|
| Regressão Logística | 0.8010   | 0.6348   | 0.5330 | 0.5795   |
| Random Forest       | 0.7840   | 0.6004   | 0.4795 | 0.5332   |

Conforme os resultados, o modelo de **Regressão Logística** apresentou um desempenho ligeiramente superior em todas as métricas avaliadas. Com uma acurácia de 80.10%, ele foi capaz de classificar corretamente a maioria dos clientes. O F1-Score de 0.5795 indica um bom equilíbrio entre precisão e recall para a classe minoritária (churn).

O modelo **Random Forest**, embora robusto, obteve resultados um pouco inferiores, com acurácia de 78.40% e F1-Score de 0.5332. Isso pode indicar que, para este conjunto de dados e a forma como as features foram engenheiradas, a relação linear capturada pela Regressão Logística, combinada com a normalização, foi mais eficaz na distinção entre clientes que evadem e os que não evadem.

Ambos os modelos não demonstraram sinais claros de overfitting ou underfitting severos, pois o desempenho no conjunto de teste foi razoavelmente próximo ao esperado com base no treinamento. O desbalanceamento da classe (74.28% Não Churn vs. 25.72% Churn) foi considerado, e as métricas de precisão, recall e F1-Score são mais indicativas do desempenho real em cenários de desbalanceamento do que apenas a acurácia.

### 3.2 Análise da Importância das Variáveis

Compreender quais variáveis mais influenciam a previsão de churn é crucial para a tomada de decisões estratégicas. A análise de importância das variáveis foi realizada para ambos os modelos:

#### Regressão Logística - Coeficientes

Na Regressão Logística, os coeficientes indicam a força e a direção da relação entre cada variável e a probabilidade de churn. Um coeficiente positivo sugere que o aumento da variável está associado a uma maior probabilidade de churn, enquanto um coeficiente negativo indica o oposto. Analisamos o valor absoluto dos coeficientes para identificar as variáveis mais impactantes:

| Feature                       | Coeficiente | Abs_Coeficiente |
|:------------------------------|:------------|:----------------|
| tenure                        | -1.352980   | 1.352980        |
| MonthlyCharges                | -0.884210   | 0.884210        |
| InternetService_Fiber optic   | 0.720018    | 0.720018        |
| TotalCharges                  | 0.633315    | 0.633315        |
| Contract_Two year             | -0.538821   | 0.538821        |
| Contract_One year             | -0.270463   | 0.270463        |
| StreamingMovies_Yes           | 0.252493    | 0.252493        |
| StreamingTV_Yes               | 0.228720    | 0.228720        |
| MultipleLines_Yes             | 0.202155    | 0.202155        |
| PaperlessBilling_Yes          | 0.189535    | 0.189535        |

Os resultados da Regressão Logística indicam que o **tempo de contrato (tenure)** é a variável mais influente, com um coeficiente negativo significativo, sugerindo que clientes com maior tempo de permanência têm menor probabilidade de churn. **MonthlyCharges** também tem um coeficiente negativo, o que pode parecer contraintuitivo, mas pode estar relacionado a pacotes de serviços mais completos e fidelização. Por outro lado, o serviço de internet **Fiber optic** e o **TotalCharges** (valor total gasto) apresentam coeficientes positivos, indicando que clientes com esses atributos têm maior propensão a evadir.

#### Random Forest - Importância das Features

No Random Forest, a importância das features é calculada com base em como cada variável contribui para a redução da impureza (por exemplo, Gini impurity) nas divisões das árvores. Quanto maior o valor, mais importante a feature é para o modelo:

| Feature                       | Importância |
|:------------------------------|:------------|
| TotalCharges                  | 0.196160    |
| MonthlyCharges                | 0.172604    |
| tenure                        | 0.166648    |
| PaymentMethod_Electronic check| 0.037858    |
| InternetService_Fiber optic   | 0.037012    |
| gender_Male                   | 0.030186    |
| OnlineSecurity_Yes            | 0.026718    |
| Contract_Two year             | 0.026126    |
| PaperlessBilling_Yes          | 0.025419    |
| TechSupport_Yes               | 0.024333    |

Para o Random Forest, as variáveis mais importantes são **TotalCharges**, **MonthlyCharges** e **tenure**. Isso reforça a relevância dessas variáveis na predição de churn, corroborando os achados da Regressão Logística. O método de pagamento **Electronic check** e o serviço **Fiber optic** também se destacam como fatores importantes.

As matrizes de confusão e os gráficos de importância das variáveis foram salvos no diretório `reports/` para visualização detalhada. As imagens `confusion_matrix_Regressao_Logistica.png`, `confusion_matrix_Random_Forest.png` e `feature_importance.png` fornecem uma representação visual clara desses resultados.


## 4. Conclusão Estratégica e Recomendações

Este projeto demonstrou a viabilidade e a eficácia da aplicação de modelos de Machine Learning para prever a evasão de clientes na Telecom X. Através de um pipeline robusto de pré-processamento, análise exploratória e modelagem, foi possível identificar os principais fatores que influenciam o churn e desenvolver modelos preditivos com bom desempenho.

### Principais Fatores que Influenciam a Evasão:

Com base na análise de importância das variáveis de ambos os modelos, os fatores mais críticos que influenciam a evasão de clientes são:

*   **Tempo de Contrato (Tenure):** Clientes com menor tempo de permanência são significativamente mais propensos a evadir. Isso sugere que os primeiros meses de relacionamento com o cliente são cruciais para a retenção.
*   **Cobranças Mensais (MonthlyCharges) e Cobranças Totais (TotalCharges):** Embora a relação possa ser complexa, clientes com MonthlyCharges mais altos e TotalCharges mais baixos (indicando um churn recente) ou TotalCharges muito altos (indicando longa permanência e talvez insatisfação acumulada) são importantes indicadores. A análise mais aprofundada revelou que clientes com MonthlyCharges muito altos e tenure baixo são mais propensos a churn.
*   **Serviço de Internet Fibra Óptica (InternetService_Fiber optic):** Clientes que utilizam fibra óptica parecem ter uma maior propensão à evasão. Isso pode indicar problemas de qualidade do serviço, expectativas não atendidas ou concorrência mais acirrada nesse segmento.
*   **Método de Pagamento (PaymentMethod_Electronic check):** Clientes que utilizam cheque eletrônico como método de pagamento também demonstram maior tendência a evadir. Isso pode estar associado a um perfil de cliente menos fidelizado ou a problemas na experiência de pagamento.
*   **Tipo de Contrato (Contract_Two year e Contract_One year):** Clientes com contratos de menor duração (mês a mês) têm maior probabilidade de churn, enquanto contratos de um ou dois anos estão associados a menor evasão, reforçando a importância da fidelização.
*   **Serviços Adicionais (OnlineSecurity, TechSupport, OnlineBackup, DeviceProtection):** A ausência desses serviços (ou a não utilização do serviço de internet, que implica na ausência desses adicionais) está correlacionada com maior churn. Clientes que utilizam esses serviços tendem a ser mais fiéis.

### Recomendações Estratégicas para Retenção:

Com base nos insights obtidos, as seguintes estratégias de retenção são propostas:

1.  **Foco nos Primeiros Meses de Contrato:** Implementar programas de 

onboarding robustos e acompanhamento proativo para clientes nos primeiros meses de serviço, especialmente aqueles com contratos de curta duração. Oferecer incentivos para a adesão a contratos mais longos desde o início.

2.  **Monitoramento de Clientes de Fibra Óptica:** Investigar as causas da maior taxa de churn entre clientes de fibra óptica. Pode ser necessário melhorar a qualidade do serviço, o suporte técnico ou oferecer pacotes mais competitivos para este segmento.

3.  **Otimização da Experiência de Pagamento:** Avaliar a experiência dos clientes que utilizam cheque eletrônico e outros métodos de pagamento com alta taxa de churn. Incentivar a migração para métodos de pagamento mais convenientes e seguros, como débito automático ou cartão de crédito.

4.  **Incentivo à Adesão de Serviços Agregados:** Promover ativamente a adesão a serviços adicionais como segurança online, backup e suporte técnico. Clientes que utilizam esses serviços tendem a ter maior satisfação e menor propensão a evadir.

5.  **Campanhas de Fidelização Baseadas em Valor:** Desenvolver campanhas de retenção personalizadas, utilizando os insights dos modelos. Por exemplo, oferecer descontos ou benefícios adicionais para clientes com alto risco de churn, ou para aqueles que demonstram insatisfação com aspectos específicos do serviço.

6.  **Análise Contínua e Refinamento do Modelo:** O modelo preditivo deve ser monitorado continuamente e retreinado periodicamente com novos dados para garantir sua relevância e precisão. A análise de novas variáveis e a experimentação com outros algoritmos podem otimizar ainda mais a capacidade preditiva.

Ao implementar essas recomendações, a Telecom X poderá não apenas antecipar a evasão de clientes, mas também desenvolver estratégias proativas e personalizadas para reter sua base de clientes, fortalecendo sua posição no mercado e garantindo um crescimento sustentável a longo prazo.


