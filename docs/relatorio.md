# Relatório do Projeto ScoreFlow

## Alunos
- [Nome do Aluno 1]
- [Nome do Aluno 2]
- [Nome do Aluno 3]

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

## 8. Resultados
A etapa de comparação entre modelos mostrou que, para a base sintética, a Regressão Logística obteve o melhor desempenho dentre os modelos avaliados. A modelagem adotada foi capaz de distinguir clientes com maior e menor risco de inadimplência com uma performance aceitável para fins acadêmicos e demonstrativos.

Os principais indicadores observados foram:

- ROC AUC;
- acurácia;
- precisão;
- recall;
- F1-score.

Esses resultados indicam que o modelo consegue capturar padrões relevantes de risco e entregar previsões consistentes para o contexto do projeto.

## 9. Benefícios da Solução
A solução oferece diversos benefícios, como:

- automatização da análise de risco;
- padronização da decisão de crédito;
- interface amigável para uso em aplicações acadêmicas e demonstrações;
- integração entre Machine Learning e front-end;
- possibilidade de extensão para dados reais em projetos futuros.

## 10. Limitações
Como trata-se de uma base sintética, o projeto possui algumas limitações naturais:

- os dados não representam clientes reais de uma instituição financeira;
- os padrões podem ser mais simples do que o observado em dados reais;
- o modelo é apropriado para fins de demonstração e estudo acadêmico.

Mesmo assim, a arquitetura e o processo seguem padrões de desenvolvimento coerentes com aplicações reais de Machine Learning.

## 11. Conclusão
O projeto ScoreFlow cumpriu o objetivo de desenvolver uma solução completa de Machine Learning aplicada ao contexto de risco de crédito. A combinação de geração de dados, treinamento de modelo, ajuste de limiar e interface web demonstrou de forma prática como uma solução de IA pode ser transformada em um produto acessível e visualmente funcional.

Além disso, o projeto reforça a importância de entender o problema de negócio, preparar corretamente os dados, validar modelos e entregar uma solução com boa experiência de uso. O resultado final representa um bom exemplo de pipeline end-to-end em Machine Learning para fins acadêmicos.

## 12. Referências
- Scikit-learn Documentation
- Flask Documentation
- Pandas Documentation
- Seaborn Documentation
- Machine Learning para análise de risco de crédito

## 13. Observação Final
Este relatório foi desenvolvido como complemento do projeto em GitHub. O repositório público deve ser usado como entrega principal, enquanto este documento pode servir como suporte acadêmico e de apresentação do trabalho.
