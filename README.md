# Prevenção de Churn em uma Empresa de Telecomunicações

<p align="center">
  <img width="600" height="300" src="imagem01.png">
</p>

## Descrição do Projeto:

A empresa de telecomunicações enfrenta desafios significativos relacionados à retenção de clientes, uma vez que a taxa de churn tem impactos diretos nos resultados financeiros e na reputação da empresa. O objetivo deste projeto de ciência de dados é desenvolver um modelo preditivo que avalie a probabilidade de um cliente se tornar churn. Isso permitirá à empresa implementar estratégias proativas de retenção, personalizadas para cada cliente, visando reduzir a perda de clientes e maximizar a satisfação e lealdade.

## Motivação:

Espera-se que o projeto forneça à empresa uma ferramenta preditiva robusta que permita antecipar e abordar proativamente casos de churn. Com a implementação bem-sucedida do modelo, a empresa estará melhor posicionada para otimizar a retenção de clientes, promovendo a satisfação e lealdade, e mitigando os impactos negativos associados ao churn.

## Conjunto de Dados:

Os dados utilizados neste projeto foram originalmente disponibilizados na [plataforma de ensino da IBM Developer](https://developer.ibm.com/), e tratam de um problema típico de uma companhia de telecomunicações. O dataset completo pode ser encontrado [neste link](https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv).
O dataset conta com mais de sete mil linhas e vinte e uma colunas. Sendo que, cada linha representa um cliente, cada coluna contém os atributos do cliente descritos na coluna Metadados.

O conjunto de dados inclui informações sobre:
- Clientes que saíram no último mês – a coluna é chamada de Churn
- Serviços para os quais cada cliente se inscreveu – telefone, várias linhas, internet, segurança online, backup online, proteção de dispositivos, suporte técnico e streaming de TV e filmes
- Informações da conta do cliente – há quanto tempo eles são clientes, contrato, forma de pagamento, cobrança sem papel, cobranças mensais e cobranças totais
- Informações demográficas sobre os clientes – sexo, faixa etária e se eles têm parceiros e dependentes

## Principais Técnicas/Algoritmos Utilizados:

### Data Preparation:
**Bibliotecas:**
- pandas
- numpy
- train_test_split (sklearn.model_selection)

**Principais Técnicas/Algoritmos:**
- Divisão de dados em conjuntos de treinamento e teste (train_test_split)

### Feature Selection:
**Bibliotecas:**
- seaborn
- pandas
- numpy
- matplotlib.pyplot
- RandomForestClassifier (sklearn.ensemble)
- GradientBoostingClassifier (sklearn.ensemble)
- DecisionTreeClassifier (sklearn.tree)

**Principais Técnicas/Algoritmos:**
- Visualização de correlações (seaborn)
- Seleção de características com base em importância (Random Forest, Gradient Boosting, Decision Tree)

### Modelagem:
**Bibliotecas:**
- seaborn
- matplotlib.pyplot
- RandomForestClassifier (sklearn.ensemble)
- GradientBoostingClassifier (sklearn.ensemble)
- DecisionTreeClassifier (sklearn.tree)

**Principais Técnicas/Algoritmos:**
- Modelos de Machine Learning para classificação (Random Forest, Gradient Boosting, Decision Tree)

### Código de Escoragem:
**Bibliotecas:**
- matplotlib.pyplot
- pandas
- numpy
- itertools
- confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score (sklearn.metrics)
- pickle
- train_test_split (sklearn.model_selection)

**Principais Técnicas/Algoritmos:**
- Avaliação de desempenho do modelo por meio de métricas como Matriz de Confusão, Curva ROC, Curva Precision-Recall, AUC-ROC
- Salvamento e carga de modelos treinados (pickle)

## Pré-requisitos:

As bibliotecas necessárias para este projeto estão no arquivo “requiriments.txt”

## Estrutura do Projeto:

### Objetivo do Projeto:
### Aquisição dos Dados
### Preparação dos Dados
  - Tratamento de Nulos
  - Substituição dos valores nulos pela moda para variáveis categóricas.
  - Eliminação de variáveis com mais de 70% de nulos.
  - Tratamento de Variáveis Categóricas
  - Aplicação do método LabelEncoder para variáveis de alta cardinalidade.
  - Aplicação do método OneHotEncoder para variáveis de baixa cardinalidade.
  - Normalização
  - Validação Cruzada Holdout 70/30
  - Metadados
  - Salvando Tabelas Pós-Preparação dos Dados
### Feature Selection
  - Lendo dados pós processo de data prep
  - Treinar modelo com algoritmo Random Forest
  - Obter importância das variáveis
  - Correlação de Pearson
  - Salvando abt para treinamento dos modelos
### Modelagem
  - Importar bibliotecas e definir funções
  - Separando uma amostra de 70% para treinar o modelo e 30% para testar o modelo
  - Treinamento modelos
    - Árvore de Decisão
    - Regressão Logística
    - Random Forest
    - LightGBM
  - Modelo Campeão Escolhido
  - Salvando artefatos dos modelos
### Código de Escoragem
  - Bibliotecas necessárias para o projeto
  - Leitura dos dados a serem escorados
  - Separar 70% dos dados para treino e 30% para validação
  - Carregar lista de variáveis que foram excluídas por excesso de nulos
  - Retirar 'Unnamed: 0','customerID' e 'Churn' das tabelas (para escoragem não é necessário e em produção não teremos target)
  - Carregar os encoders e a lista de colunas
  - Label Encoder
  - OneHot Encoder
  - Carregar o scaler
  - Carregar lista de variáveis que passaram pelo Feature Selection (utilizadas no treinamento do modelo)
  - Carregando modelo campeão
  - Escorando base de treino e teste
  - Salvando como arquivo csv

## Resultados:

### Modelo Campeão Escolhido:
O modelo escolhido é o LightGBM, destaca-se pela ordenação eficaz das taxas de score, proporcionando uma identificação eficiente dos clientes mais propensos ou menos propensos a se tornarem um Churn.
Seu bom desempenho na métrica AUC reforça a confiabilidade na capacidade do modelo de distinguir entre casos positivos e negativos.
Ao visualizar a matriz de confusão, o LightGBM foi o modelo que apresentou melhor desempenho com verdadeiros negativos e falsos positivos. Ou seja, pensando que a partir do resultado do modelo, o negócio tomaria investimento em marketing para atuar nos possíveis "churns". Este modelo fornece uma predição que irá economizar com investimentos equivocados feitos em clientes que não têm chance de se tornar churn, mas o modelo classificou como churn - falsos positivos. Dentre os modelos treinados, o LightGBM foi o que apresentou melhor desempenho nessa classificação.
Além disso, a consistência nas métricas de KS, Gini e AUC para as bases de treinamento e teste sugere que o modelo não tenha overfitting, garantindo robustez e generalização para novos dados. Essa escolha é respaldada por sua eficácia e confiabilidade na predição de casos de diabetes.

### Conclusão:
Por fim, ao concluir o projeto, o modelo desenvolvido revela sua capacidade em prever a probabilidade de um cliente se tornar um churn. Essa informação possibilita a implementação de estratégias preventivas e a retenção de clientes antes que eles encerrem o contrato.
O modelo demonstra uma sólida capacidade de predição, sendo eficaz na identificação de clientes mais ou menos propensos a se tornarem churn.
Os resultados indicam que o LightGBM é o modelo mais eficiente para essa tarefa, destacando-se pela ordenação eficaz das taxas de score. Sua métrica AUC sólida reforça a confiança na habilidade do modelo em distinguir entre clientes propensos e não propensos ao churn.
O modelo é eficiente ao ter um bom desempenho com falsos positivos, evitando investimentos desnecessários em estratégias de retenção para esses casos.
A consistência nas métricas de KS, Gini e AUC entre as bases de treinamento e teste sugere robustez e generalização para novos dados. Com isso, o modelo pode ser usado como uma ferramenta valiosa para a tomada de decisões estratégicas no negócio.

## Contato:

Quer falar comigo? Conecte-se comigo no [LinkedIn](https://www.linkedin.com/in/rafaelsdomingos/) ou se preferir, escreva um e-mail para rafael.salomaod96@gmail.com.
Também estou presente no [Medium](https://medium.com/@rafael.salomaod), compartilhando aprendizados ao longo dos meus estudos em Big Data & Analytics.
