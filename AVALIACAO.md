# Esquema de Avaliacao do Hackathon

## Visao Geral

A avaliacao consiste em **2 etapas**:

| Etapa | Finalidade | Resultado |
|-------|-----------|-----------|
| 1. Funcionalidade | Avaliar se a solucao funciona | **Nota Final (0-10)** |
| 2. Performance | Ranquear times e desempate | **Posicao no Ranking** |

> A **nota final** é determinada pela Funcionalidade. A Performance serve para **ranking e desempate** entre times com mesma nota.

---

## Distribuicao dos Datasets

### Para os Times (Desenvolvimento)

| Tema | Dataset | Arquivo |
|------|---------|---------|
| Classificacao | Online Shoppers | `online_shoppers.csv` |
| Regressao | Insurance | `insurance.csv` |
| Series Temporais | Daily Temperature | `daily_temperature.csv` |

### Para Avaliacao (Manter Ocultos)

| Tema | Etapa | Dataset | Arquivo |
|------|-------|---------|---------|
| Classificacao | Funcionalidade | Adult Census (34k) | `adult_census_funcionalidade.csv` |
| Classificacao | Performance | Adult Census (14k) | `adult_census_performance.csv` |
| Regressao | Funcionalidade | California Housing (14k) | `california_housing_funcionalidade.csv` |
| Regressao | Performance | California Housing (6k) | `california_housing_performance.csv` |
| Series Temporais | Funcionalidade | Air Passengers (132 meses) | `air_passengers_funcionalidade.csv` |
| Series Temporais | Performance | Air Passengers (12 meses) | `air_passengers_performance.csv` |

---

## Etapa 1: Funcionalidade (Nota Final)

Cada tema possui **10 itens** de checklist. Cada item vale **0 ou 1 ponto**.

A nota final e a **media dos 3 temas**.

```
NOTA FINAL = (Classificacao + Regressao + Series Temporais) / 3
```

---

### Classificacao (0-10 pontos)

| # | Item | Pontos (0/1) |
|---|------|:------------:|
| 1 | Upload do dataset CSV funciona | |
| 2 | Detecta automaticamente que e problema de classificacao | |
| 3 | Identifica corretamente a coluna target | |
| 4 | Pre-processamento executa sem erros | |
| 5 | Treinamento dos modelos completa | |
| 6 | Exibe metricas de avaliacao (Accuracy, F1, etc) | |
| 7 | Permite visualizar comparativo entre modelos | |
| 8 | Deploy do modelo via API funciona (previsao unitaria) | |
| 9 | Deploy do modelo via Batch funciona (previsao em lote) | |
| 10 | Interface responde sem travar/crashar | |
| | **TOTAL CLASSIFICACAO** | **/10** |

---

### Regressao (0-10 pontos)

| # | Item | Pontos (0/1) |
|---|------|:------------:|
| 1 | Upload do dataset CSV funciona | |
| 2 | Detecta automaticamente que e problema de regressao | |
| 3 | Identifica corretamente a coluna target | |
| 4 | Pre-processamento executa sem erros | |
| 5 | Treinamento dos modelos completa | |
| 6 | Exibe metricas de avaliacao (R2, MAE, RMSE, etc) | |
| 7 | Permite visualizar comparativo entre modelos | |
| 8 | Deploy do modelo via API funciona (previsao unitaria) | |
| 9 | Deploy do modelo via Batch funciona (previsao em lote) | |
| 10 | Interface responde sem travar/crashar | |
| | **TOTAL REGRESSAO** | **/10** |

---

### Series Temporais (0-10 pontos)

| # | Item | Pontos (0/1) |
|---|------|:------------:|
| 1 | Upload do dataset CSV funciona | |
| 2 | Detecta automaticamente que e problema de serie temporal | |
| 3 | Identifica corretamente a coluna de data e valores | |
| 4 | Pre-processamento executa sem erros | |
| 5 | Treinamento dos modelos completa | |
| 6 | Exibe metricas de avaliacao (MAPE, MAE, etc) | |
| 7 | Permite visualizar previsao vs dados reais | |
| 8 | Deploy do modelo via API funciona (previsao unitaria) | |
| 9 | Deploy do modelo via Batch funciona (previsao em lote) | |
| 10 | Interface responde sem travar/crashar | |
| | **TOTAL SERIES TEMPORAIS** | **/10** |

---

### Calculo da Nota Final

| Tema | Pontos |
|------|--------|
| Classificacao | /10 |
| Regressao | /10 |
| Series Temporais | /10 |
| **NOTA FINAL** | **(C + R + S) / 3 = /10** |

---

## Etapa 2: Performance (Ranking e Desempate)

A performance **nao altera a nota final**. Serve para:

1. **Ranquear os times** apos a avaliacao de funcionalidade
2. **Desempatar** times com a mesma nota de funcionalidade

> **Importante:** O Deploy via Batch (item 9 da funcionalidade) e essencial para a avaliacao de performance, pois e atraves dele que os datasets de performance serao processados em lote para coleta das metricas.

### Metricas Utilizadas

| Tema | Metrica | Melhor Valor |
|------|---------|--------------|
| Classificacao | F1-Score | Maior |
| Regressao | R2 Score | Maior |
| Series Temporais | MAPE | Menor |

### Como Funciona o Ranking

1. Times sao ordenados primeiro pela **nota de funcionalidade** (maior para menor)
2. Em caso de empate, compara-se a **performance** nos datasets de teste:
   - Soma-se a posicao nos 3 rankings de performance
   - Menor soma = melhor colocacao

---

## Ficha de Avaliacao

### Time: _______________

#### Funcionalidade - Classificacao

- [ ] 1. Upload do dataset CSV funciona
- [ ] 2. Detecta automaticamente que e problema de classificacao
- [ ] 3. Identifica corretamente a coluna target
- [ ] 4. Pre-processamento executa sem erros
- [ ] 5. Treinamento dos modelos completa
- [ ] 6. Exibe metricas de avaliacao
- [ ] 7. Permite visualizar comparativo entre modelos
- [ ] 8. Deploy do modelo via API funciona (previsao unitaria)
- [ ] 9. Deploy do modelo via Batch funciona (previsao em lote)
- [ ] 10. Interface responde sem travar/crashar

**Total Classificacao: ___/10**

#### Funcionalidade - Regressao

- [ ] 1. Upload do dataset CSV funciona
- [ ] 2. Detecta automaticamente que e problema de regressao
- [ ] 3. Identifica corretamente a coluna target
- [ ] 4. Pre-processamento executa sem erros
- [ ] 5. Treinamento dos modelos completa
- [ ] 6. Exibe metricas de avaliacao
- [ ] 7. Permite visualizar comparativo entre modelos
- [ ] 8. Deploy do modelo via API funciona (previsao unitaria)
- [ ] 9. Deploy do modelo via Batch funciona (previsao em lote)
- [ ] 10. Interface responde sem travar/crashar

**Total Regressao: ___/10**

#### Funcionalidade - Series Temporais

- [ ] 1. Upload do dataset CSV funciona
- [ ] 2. Detecta automaticamente que e problema de serie temporal
- [ ] 3. Identifica corretamente a coluna de data e valores
- [ ] 4. Pre-processamento executa sem erros
- [ ] 5. Treinamento dos modelos completa
- [ ] 6. Exibe metricas de avaliacao
- [ ] 7. Permite visualizar previsao vs dados reais
- [ ] 8. Deploy do modelo via API funciona (previsao unitaria)
- [ ] 9. Deploy do modelo via Batch funciona (previsao em lote)
- [ ] 10. Interface responde sem travar/crashar

**Total Series Temporais: ___/10**

#### Resultado

| Componente | Pontos |
|------------|--------|
| Classificacao | /10 |
| Regressao | /10 |
| Series Temporais | /10 |
| **NOTA FINAL** | **/10** |

#### Performance (para ranking)

| Tema | Metrica | Valor | Posicao |
|------|---------|-------|---------|
| Classificacao | F1-Score | | |
| Regressao | R2 | | |
| Series Temporais | MAPE | | |

---

## Planilha de Ranking Final

### Ranking por Nota de Funcionalidade

| Posicao | Time | Classificacao | Regressao | Series | Nota Final | Performance (desempate) |
|---------|------|---------------|-----------|--------|------------|-------------------------|
| 1o | | /10 | /10 | /10 | | |
| 2o | | /10 | /10 | /10 | | |
| 3o | | /10 | /10 | /10 | | |
| 4o | | /10 | /10 | /10 | | |
| 5o | | /10 | /10 | /10 | | |

### Ranking de Performance (para desempate)

#### Classificacao (F1-Score)

| Pos | Time | F1-Score |
|-----|------|----------|
| 1o | | |
| 2o | | |
| 3o | | |

#### Regressao (R2)

| Pos | Time | R2 |
|-----|------|-----|
| 1o | | |
| 2o | | |
| 3o | | |

#### Series Temporais (MAPE)

| Pos | Time | MAPE |
|-----|------|------|
| 1o | | |
| 2o | | |
| 3o | | |

---

## Exemplo de Avaliacao

### Time Alpha

**Funcionalidade:**
- Classificacao: 9/10 (falhou no item 8)
- Regressao: 10/10
- Series Temporais: 8/10 (falhou nos itens 4 e 9)

**Nota Final: (9 + 10 + 8) / 3 = 9.0**

### Time Beta

**Funcionalidade:**
- Classificacao: 10/10
- Regressao: 9/10 (falhou no item 7)
- Series Temporais: 8/10 (falhou nos itens 3 e 6)

**Nota Final: (10 + 9 + 8) / 3 = 9.0**

### Desempate pela Performance

| Tema | Time Alpha | Time Beta |
|------|------------|-----------|
| Classificacao (F1) | 0.82 (2o) | 0.85 (1o) |
| Regressao (R2) | 0.91 (1o) | 0.88 (2o) |
| Series (MAPE) | 8% (1o) | 12% (2o) |
| **Soma posicoes** | **4** | **5** |

**Resultado: Time Alpha vence o desempate (menor soma = 4)**

---

## Checklist do Avaliador

### Antes do Hackathon
- [ ] Entregar datasets de desenvolvimento aos times
- [ ] Manter datasets de avaliacao ocultos
- [ ] Preparar ambiente para executar solucoes

### Durante a Avaliacao
- [ ] Para cada time, rodar os 3 datasets de funcionalidade
- [ ] Marcar os 10 itens de cada tema (0 ou 1)
- [ ] Calcular media dos 3 temas
- [ ] Rodar os 3 datasets de performance
- [ ] Coletar metricas (F1, R2, MAPE)
- [ ] Montar ranking de performance para desempate

### Apos a Avaliacao
- [ ] Ordenar times pela nota de funcionalidade
- [ ] Aplicar desempate por performance onde necessario
- [ ] Divulgar ranking final

---

## Resumo Visual

```
┌─────────────────────────────────────────────────────────────────┐
│                      AVALIACAO DO HACKATHON                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FUNCIONALIDADE (determina a nota)                             │
│   ─────────────────────────────────                             │
│   Classificacao:    10 itens (0/1 cada) = /10                   │
│   Regressao:        10 itens (0/1 cada) = /10                   │
│   Series Temporais: 10 itens (0/1 cada) = /10                   │
│                                                                 │
│   NOTA FINAL = Media dos 3 temas = /10                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PERFORMANCE (ranking e desempate)                             │
│   ─────────────────────────────────                             │
│   - Nao altera a nota final                                     │
│   - Usado para ordenar times com mesma nota                     │
│   - Metricas: F1-Score, R2, MAPE                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
