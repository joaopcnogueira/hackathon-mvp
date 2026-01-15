# Preferências para Escrita de Código

## Nomenclatura

- Use nomes de variáveis, métodos e classes descritivos e autoexplicativos que representem exatamente o que são ou fazem (ex: `calcularTotalPedido()` em vez de `calc()`, `usuarioAtivo` em vez de `ua`)
- Evite abreviações obscuras; prefira nomes mais longos e claros
- Nomes de booleanos devem indicar uma condição (ex: `estaAtivo`, `possuiPermissao`, `isValid`)
- Nomes de variáveis, funções, métodos e classes, sempre escreva em inglês

## Comentários e Documentação

- Documente sempre funções, métodos e classes com descrição do propósito, parâmetros e retorno
- A documentação de funções, métodos e classes, sempre escreva em português do Brasil
- Use comentários no código apenas quando agregarem valor real:
  - Explique o "porquê", não o "o quê" (o código já mostra o que está fazendo)
  - Adicione comentários quando houver detalhes não óbvios ou comportamentos específicos (ex: "trata vírgula como separador decimal")
  - Explique fórmulas ou cálculos complexos (ex: "KS: máxima diferença entre TPR e FPR, em %")
- Evite comentários redundantes que apenas repetem o que o código já diz claramente
- Confie na clareza do código: nomes descritivos devem tornar comentários desnecessários na maioria dos casos

## Estrutura e Organização

- Mantenha funções e métodos curtos, com uma única responsabilidade
- Classes devem ser coesas e focadas em um propósito específico
- Prefira composição e funções pequenas a classes monolíticas
- Organize o código de forma que possa ser lido de cima para baixo, como uma narrativa

## Legibilidade

- O código deve ser lido como um livro: claro, fluido e autoexplicativo
- Prefira código legível a código "esperto" ou excessivamente conciso
- Use tipagem explícita quando disponível na linguagem

## Condicionais

- Evite `else` quando possível — prefira early return ou guard clauses
- Evite ifs aninhados; extraia para funções auxiliares ou use early return
- Abstraia condições complexas para variáveis com nomes descritivos (ex: `const userCanEdit = user.isActive && user.hasPermission('edit')`)

## Boas Práticas Gerais

- Siga as convenções e padrões da linguagem em uso
- Evite magic numbers e strings — extraia para constantes nomeadas ou Enums quando fizer sentido
- Mantenha a estrutura simples e fácil de entender, sem over-engineering
- Escreva código fácil de manter e testar
- Trate erros de forma clara e previsível
- Aplique o princípio DRY (Don't Repeat Yourself), mas sem abstrações prematuras

## Explicações

- Ao apresentar código, explique de forma simples e direta o que foi feito e por quê
- Use linguagem acessível, evitando jargões desnecessários
- Se houver decisões de design importantes, justifique brevemente

---

### Padrão de Commits para Features

**NOTA**: Este padrão se aplica apenas a **novas features/métricas**. Para refatorações e ajustes em features existentes, não é necessário seguir este padrão rígido.

Cada **nova feature** deve ser dividida em **3 commits** seguindo este padrão:

**Commit 1: Model e Migration**
```
CRED-319 feat(<escopo>): adicionar model e migration

- Cria model Mcm<Nome> para armazenar <descrição>.
- Campo 'value' usa <tipo> para <explicação>.
- Migration auto-gerada com schema dinâmico e índice composto.
```

**Commit 2: Camadas de Negócio**
```
CRED-319 feat(<escopo>): implementar camadas de negócio

- Adiciona validator, service, repository e use case para <Nome>.
- <Lógica principal do cálculo>.
- <Detalhes específicos da implementação>.
- Baseado na função <nome_função> do código original (linha X).
```
