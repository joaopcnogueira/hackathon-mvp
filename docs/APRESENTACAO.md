# Hackathon 2026 — Plataforma AutoML

## Por que AutoML?

Cientistas de dados gastam aproximadamente **80% do tempo** em tarefas repetitivas: limpeza de dados, tratamento de valores nulos, seleção de variáveis, experimentação com algoritmos ([Forbes/CrowdFlower, 2016](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/)). O AutoML (Automated Machine Learning) surge para **democratizar o acesso à inteligência artificial**, permitindo que profissionais de negócio treinem modelos sem escrever código.

O desafio deste hackathon é construir uma **plataforma que automatize esse processo de ponta a ponta**.

---

## O Desafio

Desenvolver uma **plataforma AutoML no-code** (point & click) que permita a qualquer usuário — técnico ou não — treinar modelos de machine learning de forma intuitiva e profissional.

### A plataforma deve ser capaz de:

**1. Upload e Análise de Dados**
- Aceitar arquivos CSV e Excel
- Detectar automaticamente tipos de colunas (numéricas, categóricas, datas)
- Identificar valores nulos e problemas nos dados
- Exibir estatísticas descritivas do dataset

**2. Pré-processamento Automático**
- Tratamento de valores faltantes (imputação)
- Codificação de variáveis categóricas
- Normalização/padronização de features
- Divisão treino/teste de forma correta (evitando data leakage)

**3. Treinamento de Modelos**

A plataforma deve suportar **três tipos de problemas**:

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Classificação** | Prever categorias (sim/não, classes) | Cliente vai comprar? Vai cancelar? |
| **Regressão** | Prever valores numéricos contínuos | Qual será o custo? Qual o preço? |
| **Séries Temporais** | Prever valores futuros baseados em histórico | Qual a demanda do próximo mês? |

> **Mínimo de 3 algoritmos** para cada tipo de problema

**4. Avaliação e Comparação**
- Calcular métricas apropriadas para cada tipo de problema
- Comparar performance entre diferentes algoritmos
- Exibir ranking dos melhores modelos

**5. Deploy do Modelo**
- **API REST**: predições em tempo real
- **Batch**: processar múltiplos registros de uma vez
- Download de artefatos (modelo treinado)

---

## Cronograma

| Data | Atividade |
|------|-----------|
| **02/02** | Viagem de ida (manhã) + Abertura do Hackathon (tarde) |
| **03/02** | Desenvolvimento |
| **04/02** | Desenvolvimento |
| **05/02** | Desenvolvimento |
| **06/02** | Apresentação (manhã) + Avaliação e premiação (tarde) |
| **07/02** | Viagem de retorno |

---

## Datasets

Todos os times utilizarão os **mesmos datasets** para garantir comparabilidade:

| Tema | Dataset | Link para Download | Objetivo |
|------|---------|-------------------|----------|
| Classificação | Online Shoppers | [UCI Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | Prever se visitante comprará |
| Regressão | Insurance | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | Prever custo de seguro |
| Séries Temporais | Daily Temperature | [Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) | Prever valores futuros |

> **Importante**: Durante a avaliação, a plataforma será testada também com **datasets ocultos** para verificar a robustez da solução.

---

## Critérios de Avaliação

### Funcionalidade
Cada tema (classificação, regressão, séries temporais) será avaliado em **10 itens funcionais**, incluindo:

- Upload do dataset funciona corretamente
- Detecção automática do tipo de problema
- Identificação correta da coluna target
- Pré-processamento executa sem erros
- Treinamento completa com sucesso
- Métricas são exibidas corretamente
- Comparativo entre modelos
- Deploy via API funciona
- Deploy via Batch funciona
- Interface intuitiva e responsiva

**Nota Funcionalidade** = Classificação + Regressão + Séries Temporais

### Performance (critério de desempate)
Em caso de empate funcional, a performance dos modelos será considerada:

| Tipo | Métrica | Critério |
|------|---------|----------|
| Classificação | F1-Score | Maior é melhor |
| Regressão | R² Score | Maior é melhor |
| Séries Temporais | MAPE | Menor é melhor |

**Nota Performance** = (Classificação + Regressão + Séries Temporais) / 3

---

## Entregáveis

- **Plataforma funcional**: Interface no-code intuitiva e responsiva
- **Documentação**: Instruções mínimas de uso
- **Demonstração ao vivo**: Fluxo completo (upload → treino → deploy)
- **API disponível**: Para teste durante a avaliação

---

## Inspirações

Plataformas que podem servir de referência visual e funcional:

- **Orange Data Mining** — Interface visual com blocos conectáveis
- **Azure Machine Learning Studio** — Fluxo guiado e profissional
- **Google AutoML** — Simplicidade e foco no usuário
- **H2O Driverless AI** — Automação inteligente de features

---

## Dicas para os Times

1. **Foco na experiência do usuário** — A plataforma é para quem *não* sabe programar
2. **Evitem data leakage** — O split treino/teste deve ocorrer *antes* do pré-processamento
3. **Métricas corretas para cada problema** — Não usem métricas de classificação em regressão
4. **Testem com os datasets fornecidos** — Mas lembrem que haverá datasets surpresa
5. **Priorizem funcionalidade básica** — Melhor uma plataforma simples que funciona do que uma complexa quebrada

---

**Boa sorte a todos os times!**