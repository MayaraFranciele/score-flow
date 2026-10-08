# Relatório do Projeto ScoreFlow

## Alunos
- Allan Reis da Conceição
- Gabriel Noriler Souza
- Giovana Alves Duarte de Sena
- Daniel Barbosa Alves
- Mayara Franciele Santos da Silva

## 1. Introdução
O projeto ScoreFlow foi desenvolvido com o objetivo de criar uma aplicação de Machine Learning para avaliação de risco de inadimplência de crédito. A proposta central do trabalho foi unir conceitos de ciência de dados, modelagem preditiva e desenvolvimento web em uma solução funcional e visualmente organizada.

A aplicação permite que o usuário insira características financeiras e pessoais de um cliente e, com base em um modelo treinado, obtenha uma previsão de risco. O sistema foi pensado como uma interface de apoio à decisão para cenários de análise de crédito, mesmo em ambiente acadêmico.

## 2. Objetivo do Projeto
O objetivo principal foi construir uma solução completa que abrangesse desde a geração de dados até a implantação de um modelo em uma interface web. Em termos práticos, o projeto busca:

- gerar dados sintéticos de crédito;
- preparar e transformar as variáveis;
- comparar diferentes modelos de classificação;
- selecionar o melhor modelo para a tarefa;
- exportar o modelo treinado;
- desenvolver uma interface para uso em tempo real;
- apresentar a previsão de risco em formato acessível e visual.

## 3. Problemática
A análise de crédito exige que instituições financeiras avaliem o risco de inadimplência antes de conceder empréstimos ou limites de crédito. Esse processo combina variáveis sociais, financeiras e comportamentais, exigindo uma decisão que minimize perdas e maximize a segurança do processo.

Neste contexto, o uso de modelos de aprendizado de máquina permite automatizar e padronizar a classificação do risco, reduzindo a margem de subjetividade e acelerando a tomada de decisão.

## 4. Dados
O conjunto de dados foi gerado artificialmente para reproduzir perfis de clientes com características comuns em cenários de concessão de crédito. A base inclui variáveis como:

- idade;
- renda mensal;
- valor do empréstimo;
- prazo do financiamento;
- taxa de juros;
- score de crédito;
- tempo de emprego;
- dívida total;
- comprometimento da renda;
- finalidade do crédito;
- posse de imóvel;
- região;
- tipo de emprego;
- entre outras variáveis relevantes.

Além das variáveis descritoras, a base também possui o alvo binário de inadimplência, que indica se o cliente deixou de honrar o pagamento.

## 5. Metodologia
A metodologia aplicada ao projeto foi organizada em etapas:

### 5.1 Geração dos Dados
Foi implementada uma função para gerar uma base sintética com padrões estatísticos coerentes com um cenário de crédito. A geração buscou simular clientes com diferentes níveis de risco e perfis financeiros variados.

### 5.2 Preparação e Engenharia de Features
Durante o processamento, foram criadas e ajustadas variáveis relevantes para o modelo. Um exemplo importante foi o cálculo do comprometimento de renda, que representa a proporção da renda comprometida com o pagamento do crédito.

### 5.3 Tratamento dos Dados
A etapa de preparação incluiu:

- tratamento de valores ausentes;
- padronização da estrutura da base;
- separação entre variáveis numéricas e categóricas;
- uso de transformadores adequados para cada tipo de dado.

### 5.4 Treinamento dos Modelos
Os modelos avaliados incluíram:

- Regressão Logística;
- Random Forest;
- Gradient Boosting.

A comparação foi feita por meio de validação cruzada, com métricas como ROC AUC, precisão, recall e F1-score. O modelo com melhor desempenho foi escolhido para compor a solução final.

### 5.5 Ajuste de Limiar
Após a escolha do melhor modelo, foi aplicado ajuste de limiar para melhor equilibrar a classificação entre clientes de baixo e alto risco, especialmente considerando o impacto de falsos negativos e falsos positivos.

### 5.6 Exportação do Modelo
O modelo treinado e os metadados relacionados (como limiar e colunas esperadas) foram serializados em um arquivo `.pkl`, permitindo que a aplicação web carregasse o artefato diretamente.

## 6. Arquitetura da Solução
A solução foi estruturada em três blocos principais:

### 6.1 Treinamento
O arquivo `treinar.py` é responsável por:

- gerar a base de dados;
- treinar modelos;
- comparar desempenhos;
- salvar o melhor modelo.

### 6.2 Aplicação Web
O arquivo `app.py` implementa a API e a interface do Flask. Ele carrega o modelo treinado e recebe os dados do formulário para prever a probabilidade de inadimplência.

### 6.3 Interface do Usuário
Os arquivos em `templates/` e `static/` dão suporte à interface gráfica. A tela foi desenvolvida em estilo dark mode corporativo, com painel de análise, resultado de risco e campos de entrada.

## 7. Implementação do Modelo
A aplicação utiliza um pipeline de processamento e classificação para transformar os dados de entrada e calcular a probabilidade de inadimplência. O fluxo principal é:

1. coleta dos dados do formulário;
2. conversão dos valores para o formato esperado;
3. criação de um DataFrame com a ordem correta das features;
4. aplicação do modelo treinado;
5. retorno do nível de risco e da probabilidade.

A resposta final da aplicação inclui:

- risco do cliente (`Low` ou `High`);
- probabilidade estimada;
- limiar de decisão do modelo;
- mensagem interpretativa.

## 8. Análise exploratória e conclusões
A análise exploratória foi realizada com foco em verificar a qualidade da base e confirmar padrões relevantes do comportamento do risco de inadimplência. Inicialmente, foi calculada a proporção de clientes inadimplentes para verificar o equilíbrio da variável alvo. Em seguida, foram identificadas colunas com valores ausentes e a distribuição da variável alvo foi plotada para observar o comportamento da amostra.

Também foram gerados gráficos para comparar a inadimplência por score de crédito, por finalidade do empréstimo e por posse de imóvel. Esses gráficos permitiram identificar que clientes com menor score e com maior comprometimento financeiro tendem a apresentar maior risco de default. A coluna `id_contrato` foi removida antes do treinamento porque ela funciona como identificador do registro e não agrega informação útil para a previsão, além de representar um risco de vazamento de informação para o modelo.

Além disso, foram produzidos os principais gráficos esperados para a etapa exploratória, incluindo a distribuição da classe alvo, a análise de valores nulos e as comparações por score, finalidade e posse de imóvel.

## 9. Pipeline sem vazamento e comparação por validação cruzada
A construção do pipeline foi feita com separação estratificada 80/20 entre treino e teste, garantindo que a amostra de teste não fosse usada para decidir o algoritmo ou o limiar. O pré-processamento foi realizado dentro do próprio Pipeline, aplicando imputação para valores numéricos e categóricos e OneHotEncoder para variáveis categóricas. O procedimento foi estruturado para evitar vazamento de informação entre treino e teste.

Os modelos avaliados foram:

- Regressão Logística;
- Random Forest;
- Gradient Boosting.

A comparação foi feita com validação cruzada estratificada de 5 folds, e os resultados foram resumidos em média e desvio padrão do ROC AUC. Essa abordagem permite uma comparação mais robusta entre algoritmos sem sobreajustar ao conjunto de teste.

## 10. Avaliação no teste e interpretação no contexto do negócio
O melhor modelo foi selecionado com base na validação cruzada e, em seguida, foi avaliado no conjunto de teste. Para esse modelo, foram calculadas a matriz de confusão e as métricas de acurácia, precisão, recall, F1-score e ROC AUC. A análise de performance foi interpretada no contexto do negócio, em que a decisão de crédito envolve trade-off entre segurança e inclusão financeira.

Os falsos positivos e falsos negativos foram analisados como parte do processo de decisão. Um falso positivo ocorre quando o modelo classifica um cliente como inadimplente quando ele na prática não o seria. Isso pode gerar perda de oportunidade para o cliente e reduzir a competitividade do negócio. Já o falso negativo representa um caso em que o cliente é classificado como adimplente, mas acaba não honrando o compromisso. Esse erro tende a gerar maior custo financeiro para a instituição, pois implica em risco de crédito não identificado.

Em geral, no contexto de crédito, os falsos negativos costumam ser mais caros para o negócio, porque o crédito foi concedido a um cliente de maior risco, aumentando a probabilidade de perda financeira. Por isso, a escolha do limiar e a análise do custo do erro devem considerar esse impacto econômico.

## 11. Análise de limiar com custo e recomendação justificada
A análise de limiar foi feita com base no conjunto de treino usando `cross_val_predict`, evitando uso do conjunto de teste para ajustar o critério de decisão. Foram avaliados diferentes limiares e calculado o custo total de cada opção, considerando o custo do falso positivo e do falso negativo. O custo financeiro foi definido com base na hipótese de que um falso negativo gera maior impacto para a organização.

A recomendação foi feita com foco no menor custo total e no melhor equilíbrio entre precisão e recall. Isso garante que a decisão do modelo seja explicada não apenas por desempenho estatístico, mas também por impacto operacional e financeiro para a instituição.

A curva ROC também foi gerada para complementar a interpretação do modelo, mostrando a capacidade discriminatória do classificador ao longo dos diferentes limiares.

## 12. Integração ao sistema funcionando
A integração do modelo ao sistema foi implementada em Flask, e o projeto inclui a aplicação web com interface gráfica para entrada de dados e previsão de risco. O arquivo `config.py` foi ajustado para refletir o conjunto real de variáveis que compõem o problema e que são aceitas no formulário.

A interface permite inserir dados do cliente e receber em tempo real a classificação de risco. A aplicação foi desenvolvida de forma a manter a lógica da modelagem fiel ao conjunto de features que o modelo realmente entende, evitando inconsistências entre interface e pipeline de treino.

## 13. Reflexão ética e LGPD
A implementação deste projeto foi pensada com atenção à responsabilidade ética em modelos de decisão automatizada. O modelo não deve ser usado para discriminar clientes com base em atributos sensíveis, como raça, religião, origem étnica, orientação sexual, estado de saúde ou outras características protegidas por lei. Em um cenário real, o uso de variáveis sensíveis seria inapropriado e poderia levar a decisões injustas ou ilegais.

Além disso, a base sintética utilizada não contém dados reais de pessoas e foi criada exclusivamente para fins acadêmicos. Mesmo assim, em aplicações reais, a coleta e o uso de dados pessoais devem seguir princípios da LGPD, incluindo minimização de dados, consentimento, transparência, segurança da informação e direito de explicação ao titular.

A utilização do modelo deve sempre considerar a necessidade de monitoramento contínuo, revisão humana das decisões e avaliação de impactos para grupos vulneráveis. Em outras palavras, a IA deve funcionar como suporte à decisão e não como substituto de critérios éticos e humanos.

## 14. Desafios extras e análise adicional
Além dos requisitos base, o projeto também contemplou itens relevantes de aprofundamento:

- a feature `comprometimento_renda` foi criada e incorporada ao conjunto de variáveis para capturar a carga financeira do cliente;
- o efeito dessa feature foi comparado para verificar seu impacto na capacidade preditiva do modelo;
- hiperparâmetros foram ajustados com `GridSearchCV`;
- o parâmetro `class_weight='balanced'` foi testado para avaliar seu impacto em precisão e recall.

Essas otimizações ajudam a demonstrar que o modelo foi validado com cuidado e que o projeto buscou equilíbrio entre desempenho estatístico e aplicabilidade prática.

## 15. Resultados e conclusão
Os resultados obtidos mostram que o projeto consegue distinguir adequadamente clientes com maior e menor risco de inadimplência, mantendo uma abordagem consistente com as boas práticas de Machine Learning. A combinação de pipeline robusto, avaliação cruzada, análise de limiar, interface funcional e documentação técnica permite que o trabalho seja entregado como uma solução end-to-end, alinhada às exigências da avaliação.

O projeto cumpre, portanto, os critérios de análise exploratória, validação, avaliação de negócio, análise de limiar, integração funcional e reflexão ética, constituindo uma solução completa e coerente para o contexto de crédito.

## 16. Benefícios da Solução
A solução oferece diversos benefícios, como:

- automatização da análise de risco;
- padronização da decisão de crédito;
- interface amigável para uso em aplicações acadêmicas e demonstrações;
- integração entre Machine Learning e front-end;
- possibilidade de extensão para dados reais em projetos futuros.

## 17. Limitações
Como trata-se de uma base sintética, o projeto possui algumas limitações naturais:

- os dados não representam clientes reais de uma instituição financeira;
- os padrões podem ser mais simples do que o observado em dados reais;
- o modelo é apropriado para fins de demonstração e estudo acadêmico.

Mesmo assim, a arquitetura e o processo seguem padrões de desenvolvimento coerentes com aplicações reais de Machine Learning.

## 18. Referências
- Scikit-learn Documentation
- Flask Documentation
- Pandas Documentation
- Seaborn Documentation
- Machine Learning para análise de risco de crédito
- Lei Geral de Proteção de Dados (LGPD)

## 19. Observação Final
Este relatório foi desenvolvido como complemento do projeto em GitHub. O repositório público deve ser usado como entrega principal, enquanto este documento funciona como suporte técnico e acadêmico da solução final.
