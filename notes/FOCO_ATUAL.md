# Foco atual: comparação RFM × DiffMeans em amostra pequena

*Escrito em 2026-10-06. Documento de orientação: o que está sendo feito agora, por quê e
com que critério.*

A versão técnica de referência, com todos os detalhes do protocolo, está em
`revision_20261002/IV-small-n_design.md`. Este arquivo explica o mesmo conteúdo passo a passo.
Se os dois discordarem, o design é a fonte de verdade e este arquivo precisa ser corrigido.

**Status:** desenho escrito, experimento **não rodado**. Em 2026-10-08 ficaram decididos três
pontos: os hiperparâmetros do RFM (opção A, seção 4.4), o tratamento de $T$ como análise
secundária (seção 4.4.1) e a correção de Holm com p-valores (seção 4.6). Os três já estão no
design.

---

## 0. Em uma frase

> No GuppyLM, com $n$ exemplos por classe ($n = 1, 2, 4, \dots, 32$), o RFM extrai uma direção
> de conceito melhor que o DiffMeans? A diferença muda de sinal conforme $n$ cresce?

Isso responde à pergunta **P1** do `PROPOSITO.md`: o RFM encontra algo que os métodos simples
não encontram?

---

## 1. Por que este experimento, e não outro

Com todos os dados de treino, os três conceitos atuais não discriminam os métodos:

- **Monitoring:** RFM, DiffMeans, LogReg e PCA corrigido dão AUROC entre 0,99 e 1,00. É um
  efeito de teto. Quando todo método acerta tudo, a métrica não ordena nada.
- **Steering:** nenhuma direção produz texto coerente (Check III). Também não ordena nada.
- **Os conceitos foram escolhidos para serem fáceis.** O docstring de
  `rfm_guppy/concepts.py` diz *"chosen for semantic separability and lexical concreteness"*.

Reduzir $n$ torna a tarefa mais difícil sem trocar de modelo nem de conceito. O teste de
viabilidade (DiffMeans, bloco 3, AUROC de teste, mediana de 200 subamostras) mostra onde há
espaço para comparar:

| conceito | $n=1$ | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=32$ |
|---|---|---|---|---|---|---|
| `food` | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| `valence` | 0,782 | 0,875 | 0,938 | 0,975 | 0,988 | 0,993 |
| `env_social` | 0,922 | 0,975 | 0,994 | 0,999 | 1,000 | 1,000 |

Leitura:

- **`food` é trivial**: um exemplo por classe já basta. Ele sai da comparação e fica só como
  controle.
- **`valence` e `env_social` ficam abaixo do teto para $n \le 16$.** É nessa faixa que a
  comparação tem informação.

---

## 2. A proposição: todos os métodos procuram no mesmo lugar

### 2.1 A ideia sem fórmulas

Imagine que você tem só 2 pontos de treino em $\mathbb{R}^{384}$, um de cada classe. A única
informação que eles carregam sobre "para onde aponta o conceito" é o segmento que liga um ao
outro. Qualquer método razoável deveria responder "a direção desse segmento". A proposição
mostra que isso não é só intuição. Para o RFM, o DiffMeans, o PCA e a LogReg, é um fato
exato: **nenhum deles consegue apontar para fora do espaço gerado pelas diferenças entre os
pontos de treino.**

Com $N$ pontos, esse espaço tem dimensão no máximo $N - 1$. Com $n = 8$ por classe ($N = 16$),
a direção escolhida fica num subespaço de dimensão $\le 15$ dentro de $\mathbb{R}^{384}$.
Quando $n$ é pequeno, a pergunta "o RFM acha uma direção diferente?" fica mais precisa: **qual
direção cada método escolhe dentro desse subespaço pequeno?**

### 2.2 Os objetos

Fixe um bloco $\ell$ e omita-o da notação. A subamostra de treino tem $N = 2n$ ativações
$a^{(1)}, \dots, a^{(N)} \in \mathbb{R}^{d}$, com $d = 384$, e rótulos $y \in \{0,1\}^N$.

**O subespaço das diferenças:**

$$
\mathcal{D} := \operatorname{span}\{\, a^{(i)} - a^{(j)} : 1 \le i, j \le N \,\}.
$$

**Ativações centradas:** $\tilde a^{(i)} = a^{(i)} - \bar a$, com $\bar a$ a média da
subamostra. Este é o primeiro passo do RFM no artigo e no código.

**Lema 0.** $\mathcal{D} = \operatorname{span}\{\tilde a^{(1)}, \dots, \tilde a^{(N)}\}$ e
$\dim \mathcal{D} \le N - 1$.

*Por quê.*

- $\tilde a^{(i)}$ é a média das diferenças $a^{(i)} - a^{(j)}$ sobre $j$, logo está em $\mathcal{D}$.
- Na outra direção, $a^{(i)} - a^{(j)} = \tilde a^{(i)} - \tilde a^{(j)}$.
- Os $\tilde a^{(i)}$ somam zero, então são $N$ vetores linearmente dependentes, e a dimensão
  cai para $N - 1$.

O lema também vale com `normalize=True`. Normalizar multiplica cada $\tilde a^{(i)}$ por um
escalar positivo, e isso não muda o span.

**O RFM.** O kernel é Laplaciano–Mahalanobis,
$K_M(x,z) = \exp\!\big(-\lVert x - z\rVert_M / L\big)$, com
$\lVert u\rVert_M = \sqrt{u^\top M u}$. Começa-se com $M_0 = I$. A cada iteração:

1. *Ajuste:* resolve-se a kernel ridge regression,
   $\alpha_t = (K_{M_t} + \lambda I)^{-1} y$, e o preditor é
   $f_t(z) = \sum_i \alpha_{t,i} K_{M_t}(z, \tilde a^{(i)})$.
2. *Gradiente:* calcula-se $\nabla f_t$ em cada ponto de treino.
   Para $z \ne \tilde a^{(i)}$,
   $$
   \nabla_z K_{M}(z, \tilde a^{(i)}) = -\frac{K_M(z, \tilde a^{(i)})}{L\,\lVert z - \tilde a^{(i)}\rVert_M}\; M\,(z - \tilde a^{(i)}).
   $$
   Em $z = \tilde a^{(i)}$, o kernel tem um "bico" e não é diferenciável. O código descarta esse
   termo (`rfm.py`, `_agop_step`, `D < eps` dá `W = 0`), e a proposição adota a mesma convenção.
3. *AGOP:* centram-se os gradientes, $g_j - \bar g$, e forma-se
   $$
   M_{t+1} = \frac1N \sum_j (g_j - \bar g)(g_j - \bar g)^\top.
   $$

A direção RFM é o autovetor principal de $M_T$, com o sinal escolhido pela correlação com $y$.

### 2.3 O enunciado

> **Proposição 1.** Para todo $t \ge 1$, $\operatorname{range}(M_t) \subseteq \mathcal{D}$.
> Logo $\operatorname{rank}(M_t) \le 2n - 1$, e, se $M_T \ne 0$, a direção RFM pertence a
> $\mathcal{D}$. O resultado vale para quaisquer $y$, $L$, $\lambda$, $T$ e com ou sem
> normalização.

### 2.4 A prova, passo a passo

A afirmação provada por indução é: **$M_t$ leva $\mathcal{D}$ em $\mathcal{D}$.**

1. **Base ($t = 0$).** $M_0 = I$ leva qualquer coisa nela mesma.
2. **Os gradientes ficam em $\mathcal{D}$.** Pela fórmula do passo 2, o gradiente em
   $\tilde a^{(j)}$ é uma soma de termos (escalar) $\times\, M_t(\tilde a^{(j)} - \tilde a^{(i)})$.
   A diferença $\tilde a^{(j)} - \tilde a^{(i)}$ está em $\mathcal{D}$, e pela hipótese de
   indução $M_t$ a mantém em $\mathcal{D}$. Somas de vetores de $\mathcal{D}$ ficam em
   $\mathcal{D}$.
3. **Centrar não atrapalha.** $\bar g$ é média de vetores de $\mathcal{D}$, então
   $g_j - \bar g \in \mathcal{D}$.
4. **A AGOP só produz vetores de $\mathcal{D}$.** Para qualquer $x$,
   $$
   M_{t+1}x = \frac1N \sum_j \langle g_j - \bar g, x\rangle\,(g_j - \bar g),
   $$
   uma combinação dos $g_j - \bar g$. Então
   $\operatorname{range}(M_{t+1}) \subseteq \mathcal{D}$, o que fecha a indução.
5. **O autovetor está em $\mathcal{D}$.** Se $M_T v = \mu v$ com $\mu > 0$, então
   $v = \mu^{-1} M_T v$ é uma saída de $M_T$, e portanto está em $\mathcal{D}$. $\square$

**Como ler o resultado.** Depois da primeira iteração, $M_t$ zera tudo que é ortogonal a
$\mathcal{D}$. A distância $\lVert\cdot\rVert_{M_t}$ fica cega para essas direções. O RFM não
tem mecanismo para "inventar" uma direção que os dados de treino não sugerem.

### 2.5 Os baselines também

> **Corolário 1.** DiffMeans, PCA das diferenças pareadas (centrado ou não) e regressão
> logística com penalidade $\ell_2$ produzem direções em $\mathcal{D}$.

- **DiffMeans:**
  $\hat\mu_+ - \hat\mu_- = \frac{1}{n^2}\sum_{i \in +}\sum_{j \in -}(a^{(i)} - a^{(j)})$,
  uma média de diferenças.
- **PCA:** cada linha da matriz de diferenças pareadas está em $\mathcal{D}$, e o vetor singular
  é uma combinação das linhas.
- **LogReg:** decomponha $w = w_{\mathcal{D}} + w_\perp$. A componente $w_\perp$ não muda
  nenhuma previsão no treino, porque é ortogonal a todos os $\tilde a^{(i)}$, mas aumenta a
  penalidade $\lVert w\rVert^2$. O ótimo, portanto, tem $w_\perp = 0$. Isso vale para o ótimo
  exato. O solver é aproximado, por isso o notebook mede o resíduo fora de $\mathcal{D}$.

### 2.6 O caso extremo: um exemplo por classe

> **Corolário 2.** Com $n = 1$ e `normalize=False`, os quatro métodos devolvem exatamente a
> mesma direção, $u = (a^+ - a^-)/\lVert a^+ - a^-\rVert$. Logo $\Delta(1) = 0$.

*Por quê.*

- Com dois pontos, $\mathcal{D}$ é a reta gerada por $u$. Cada método devolve $+u$ ou $-u$.
- A orientação pela correlação com $y$ escolhe $+u$ em todos.
- Falta garantir que o RFM não devolva a matriz nula. O cálculo explícito, feito no design,
  mostra que $M_t = c_t\,\delta\delta^\top$ com $c_t > 0$ para todo $t$, em que
  $\delta = a^+ - a^-$.

**Para que serve:** é um teste de sanidade. Se o código não der direções idênticas com
$n = 1$, há bug.

### 2.7 O que a proposição não diz

- Ela **não** diz que os métodos dão a mesma direção para $n \ge 2$. Diz só que procuram no
  mesmo subespaço.
- Ela **não** diz nada sobre qual direção é melhor. Isso é o experimento.
- Ela só tem conteúdo enquanto $2n - 1 < 384$, isto é, $n \le 192$. Com mais dados que isso,
  $\mathcal{D}$ pode ser o espaço inteiro.

---

## 3. A previsão teórica (hipótese, não teorema)

### 3.1 A referência clássica

Para duas classes gaussianas com a mesma covariância $\Sigma$, a melhor direção linear é a de
Fisher:

$$
w^\star = \Sigma^{-1}(\mu_+ - \mu_-).
$$

O DiffMeans usa $I$ no lugar de $\Sigma$. Ele acerta quando $\Sigma \propto I$ e erra quando
as classes estão "esticadas" em direções diferentes de $\mu_+ - \mu_-$.

Com poucos dados, porém, estimar $\Sigma$ é o problema. Com $2n < 384$ pontos, a covariância
amostral nem é invertível. Bickel & Levina (2004, *Bernoulli* 10(6)) mostram que, quando a
dimensão cresce mais rápido que o número de amostras, a regra de Fisher estimada se degrada
até o nível do chute aleatório, enquanto regras que ignoram as correlações continuam boas.
**Pendente: conferir o enunciado exato e as hipóteses na fonte antes de citar na
dissertação.**

### 3.2 A ponte para o nosso caso (heurística)

- O DiffMeans é um estimador simples. Não tem hiperparâmetros, e seu erro cai como $1/n$.
- O RFM usa mais estrutura dos dados (a matriz $M_t$ e os coeficientes $\alpha_t$). Com poucos
  pontos, essa estrutura é estimada com muito ruído.
- **Trade-off viés–variância:** com $n$ pequeno, o estimador simples deve vencer. Com $n$
  grande, o estimador mais rico pode vencer, se houver estrutura além de $\mu_+ - \mu_-$ para
  aproveitar.

### 3.3 As hipóteses, em forma precisa

Defina, para o conceito $c$ e a métrica $m$ (AUROC de teste):

$$
\Delta_c(n) := \mathbb{E}_S\big[\, m(v_{\text{RFM}}(S)) - m(v_{\text{DM}}(S)) \,\big],
$$

em que $S$ é uma subamostra de treino com $n$ exemplos por classe.

| Nome | Enunciado | Status |
|---|---|---|
| **Teorema** | $\Delta_c(1) = 0$ | provado (Corolário 2) |
| **H3a** | Para $n \in \{2, 4\}$, $\Delta_c(n) \le 0$: o DiffMeans vence em amostra pequena | conjectura |
| **H3b** | $\Delta_c$ muda de sinal: negativo para $n$ pequeno, positivo para $n$ grande, com cruzamento em $n^\star$ | conjectura |
| **H-artigo** | Existe $n$ com $\Delta_c(n) > 0$: o RFM vence em algum regime no Guppy | afirmação em teste |

H3a e H-artigo podem valer juntas. Esse é o cenário H3b, e nele o número interessante é
$n^\star$.

---

## 4. O desenho formal

### 4.1 Pergunta

Como $\Delta_c(n)$ varia com $n$, para RFM contra DiffMeans? Os contrastes RFM contra LogReg e
RFM contra PCA são secundários.

### 4.2 Material

- **Conceitos:**
  - `valence`, pool de 60 por classe, teste com 20 por classe;
  - `env_social`, pool de 180 por classe, teste com 60 por classe;
  - `food` só como controle de teto.
- **Bloco:** o bloco 3, fixado antes de rodar. Os outros 5 blocos entram como análise
  secundária, reportados sem seleção.
- **Dados:** os do cache, `rfm_guppy/cache/`. Nenhuma inferência nova no modelo.

### 4.3 Amostragem

- **Grade:** $n \in \{1, 2, 4, 8, 16, 32\}$, mais o $n$ máximo (60 ou 180) como ponto de
  referência.
- **Repetições:** $R = 200$ subamostras por $(c, n)$, sem reposição, com semente fixa.
- **Pareamento:** a mesma subamostra alimenta os quatro métodos. Assim cada $\Delta$ é uma
  diferença pareada, com muito menos ruído.

### 4.4 Métodos e hiperparâmetros (decidido em 2026-10-08: opção A)

O artigo **não** usa $L$ fixo. Ele escolhe $(L, T, \text{normalize})$ por busca em
$L \in \{1, 10, 100\}$ e $T \in \{1,\dots,10\}$, numa validação de 20% do treino, com
$\lambda = 10^{-3}$. Essa busca não roda com $n = 2$.

**Opção A (adotada):** congelar os $(L, T)$ que a busca do artigo escolhe com o treino
completo, com **`normalize=False` fixo para todos os conceitos e blocos**. Os mesmos valores
valem para todo $n$ e toda subamostra.

**Isonomia (decidido em 2026-10-08).** Todos os conceitos passam pela mesma regra, sem escolha
de normalize por conceito: busca em $L \in \{1,10,100\}$ e $T \in \{1,\dots,10\}$, ajuste no
`train` completo, seleção por $|\rho|$ no `val` do cache. Em código, é
`extract_concept_vector(..., sweep={"normalize": [False]})`. Escolheu-se `False` porque:

- o Corolário 2 só está provado para `False`, e assim o C0 cobre os três conceitos;
- a busca original já tinha escolhido `False` em `valence` e `env_social`, em todos os blocos.

O código varre `False` primeiro e só troca de melhor com desigualdade estrita. Por isso, onde
o cache já tinha `False`, a busca restrita devolve os mesmos $(L, T)$. Só `food` nos blocos 1–5
muda, porque lá o cache tinha `True`. O passo 0 do notebook recalcula tudo, confere a igualdade
com o cache e só então abre o teste. No bloco 3:

| | $L$ | $T$ | normalize | $\lambda$ |
|---|---|---|---|---|
| `valence` | 10 | 10 | não | $10^{-3}$ |
| `env_social` | 10 | 3 | não | $10^{-3}$ |
| `food` | passo 0 | passo 0 | não | $10^{-3}$ |

**Opção B (descartada):** $L = 10$ para tudo, com $T$ fixado *a priori*.

Os outros métodos:

- **DiffMeans:** sem hiperparâmetros.
- **PCA:** sem centragem, como no Apêndice H do artigo, com pareamento aleatório dentro de $S$.
- **LogReg:** $C$ recalculado no passo 0, com a mesma regra para todos os conceitos: ajuste no
  `train` completo, seleção por $|\rho|$ no `val` do cache, grade $\{1000, 100, 1, 0.1\}$,
  empate resolvido pela primeira na ordem da grade. `logreg_vector` não devolve o $C$, então o
  notebook reproduz o laço e o registra.

**Viés declarado:** congelar hiperparâmetros escolhidos com o treino completo favorece o RFM e
a LogReg. Se o RFM não vencer mesmo assim, o resultado é robusto. Se vencer, a vantagem é um
limite superior.

#### 4.4.1 Sensibilidade a $T$ (decidido em 2026-10-08)

O teste primário **não muda**. Ele usa o $T$ congelado da opção A e as 10 células com Holm.
Escolher $T$ depois de ver os dados quebraria o pré-registro.

Ficam fixados antes de rodar, como análises **secundárias**:

1. **Curva $\Delta_c(n, T)$ para $T = 1, \dots, 10$.** É descritiva e não envolve seleção. Mostra
   se a conclusão do teste primário depende do $T$ escolhido. É a versão mínima.
2. **$T$ oráculo:** $\Delta_c^{\max}(n) := \max_{T} \Delta_c(n, T)$, com um $T$ por célula
   $(c, n)$ e não um por subamostra. O $T$ é escolhido no teste, então isso é um **limite
   superior do RFM**:
   - **pode reforçar H3a.** Se o limite superior do IC de $\Delta_c^{\max}(n)$ for $< 0$
     para $n = 2$ ou $4$, o DiffMeans vence mesmo com o melhor $T$ para o RFM;
   - **não pode sustentar H-artigo.** Uma vitória do RFM com $T$ oráculo pode ser só efeito
     de seleção no teste.

**Fica de fora:** escolher $T$ por validação cruzada leave-one-out dentro de $S$. Com $n = 2$,
cada ajuste teria 3 pontos, e a AUROC de um único ponto de validação não é definida. Só faria
sentido para $n \ge 4$, ao custo de +2–3 h.

**Notas:**

- Pela Proposição 1, o C0 vale para qualquer $T$. Pelo Corolário 2, com $n = 1$ todo $T$ dá a
  mesma direção, então $\Delta_c(1, T) = 0$ para todo $T$.
- `rfm_fit` (`rfm.py`) devolve só $M_T$. Para obter a curva com um único ajuste, o notebook
  chama `_agop_step` num laço e guarda $M_1, \dots, M_{10}$. O custo estimado é de +1 h.
- Só $T$ varia. $L$ fica fixo, porque cada $L$ exigiria um novo ajuste completo.

### 4.5 Métricas

- **Primária:** AUROC no teste da projeção $\langle a(x), v\rangle$.
- **Secundária:** $|\rho|$ no teste.
- **Descritivas:**
  - $|\cos(v_{\text{RFM}}, v_{\text{DM}})|$: as direções são as mesmas?
  - o posto de $M_T$;
  - o resíduo fora de $\mathcal{D}$, que verifica a proposição na prática.

### 4.6 Estatística

- **Estimador:** $\hat\Delta_c(n)$ é a média das $R$ diferenças pareadas.
- **IC de 95%:** bootstrap percentil sobre as $R$ diferenças.
- **Correção de Holm, só para H-artigo**, sobre as 10 células: 2 conceitos ×
  $n \in \{2,4,8,16,32\}$. H-artigo diz que *existe* uma célula com $\Delta > 0$. Por isso
  precisa de controle sobre a grade inteira.
  1. *p-valor por célula:* $p_{c,n} = \big(1 + \#\{b : \hat\Delta^{*}_b \le 0\}\big)/(1 + B)$,
     em que $\hat\Delta^{*}_b$ é a média da $b$-ésima reamostragem bootstrap e $B = 10.000$.
     É unilateral, para $H_0: \Delta_c(n) \le 0$.
  2. *Holm:* ordenam-se os p-valores, $p_{(1)} \le \dots \le p_{(10)}$. A célula de posição $i$
     é declarada com $\Delta > 0$ se $p_{(j)} \le 0{,}025/(10 - j + 1)$ para todo $j \le i$.
  3. O nível 0,025 unilateral equivale ao limite inferior do IC bilateral de 95%.
- **Sem correção:** H3a, H3b, a curva $\Delta_c(n, T)$ e o $T$ oráculo (4.4.1) usam o IC de
  95% célula a célula. H3a e H3b falam de células específicas, não da existência de uma.
- **O IC não cobre** a variabilidade do conjunto de teste, que tem só 40 exemplos em `valence`.

### 4.7 Critérios de decisão, fixados antes de rodar

| | Critério | Se acontecer |
|---|---|---|
| **C0** | Nos três conceitos, todos com `normalize=False`. Com $n=1$: direções idênticas ($1 - \lvert\cos\rvert < 10^{-8}$). Para todo $n$: resíduo fora de $\mathcal{D}$ $< 10^{-6}$. No passo 0: $(L, T)$ recalculados iguais aos do cache onde ele tinha `False` | Se falhar, **para tudo** e procura o bug |
| **H3a** | Refutada se, para $n = 2$ ou $4$, o limite inferior do IC de 95% (sem correção) de $\Delta$ for $> 0$ | O RFM é mais eficiente em amostra que o previsto |
| **H-artigo** | Sustentada se ao menos uma célula for declarada com $\Delta > 0$ pelo Holm com p-valores (4.6). Senão, refutada no Guppy | — |
| **H3b** | Sustentada se, no mesmo conceito, uma célula tiver limite superior $< 0$ e outra, com $n$ maior, limite inferior $> 0$ (IC de 95%, sem correção) | Reportar $n^\star$ |
| **H3a, reforço** (secundário) | Para $n = 2$ ou $4$, o limite superior do IC de $\Delta_c^{\max}(n)$ é $< 0$ (4.4.1) | O DiffMeans vence mesmo contra o RFM com o melhor $T$ |

### 4.8 Como ler cada resultado possível

| O que aparecer | O que significa |
|---|---|
| $\Delta \le 0$ em toda a grade | O RFM não extrai direção melhor que o DiffMeans no Guppy, nem em amostra pequena |
| $\Delta < 0$ e depois $\Delta > 0$ | Cruzamento: o RFM só compensa com dados suficientes |
| $\Delta > 0$ já com $n$ pequeno | O RFM é mais eficiente em amostra, contra a previsão clássica |
| $\Delta \approx 0$ com IC estreito | Equivalência prática neste regime |

### 4.9 Limitações que já se sabe que existem

1. **São só 2 conceitos.** Não há inferência *entre* conceitos.
2. **As classes de `valence` juntam 4 categorias cada.** Com $n$ pequeno, a subamostra pode
   não cobrir todas. Isso é registrado, mas não estratificado.
3. **O pool de `valence` é pequeno (60).** Com $n = 32$, as subamostras se sobrepõem muito, e
   o IC fica otimista.
4. **Os hiperparâmetros congelados** favorecem o RFM (seção 4.4).
5. **Um modelo, conceitos de tópico.** Nada se estende a modelos grandes sem hipótese extra.

---

## 5. O que está fora deste experimento, de propósito

- **Steering.** A métrica está quebrada, porque nenhuma direção gera texto coerente. Escalar
  essa métrica só multiplicaria o problema.
- **Novos conceitos** (*one-vs-one* entre as 60 categorias, bipartições aleatórias). São a
  extensão natural se este experimento render. Não entram agora.
- **Conceitos lidos só do prompt do usuário.** Há poucos textos únicos por categoria. A ideia
  vai para `Perguntas de pesquisa`.

---

## 6. Próximos passos

- [x] Decidir a opção A ou B da seção 4.4: opção A.
- [x] Decidir como tratar $T$: curva $\Delta(n,T)$ e $T$ oráculo como secundários (4.4.1).
- [x] Definir a correção de Holm com p-valores e o nível dos ICs de H3a e H3b (4.6).
- [x] Passar as três decisões para `revision_20261002/IV-small-n_design.md` (§3.3, §3.3.1,
  §3.5, §3.6, §3.8).
- [x] Isonomia: `normalize=False` fixo para todos os conceitos, e $C$ da LogReg recalculado
  no `val` do cache (§4.4).
- [ ] Ajustes menores do design: $n_{\max}$ não determinístico por causa do PCA, script de
  viabilidade ausente.
- [ ] Notebook `revision_20261002/IV-small-n.ipynb`, começando pelo teste C0 (~4–5 h).
- [ ] Escrever `revision_20261002/IV-small-n.md` com o resultado, no formato dos Checks I–III.
- [ ] Conferir Bickel & Levina (2004) na fonte.
- [ ] Redigir a Proposição 1 e os Corolários 1–2 para a dissertação, com a prova completa
  (base: `revision_20261002/IV-small-n_design.md`, §1).

---

## Glossário de símbolos

| Símbolo | Significado |
|---|---|
| $d = 384$ | dimensão do residual stream do GuppyLM |
| $\ell$ | bloco do transformer (0 a 5); aqui, fixo em 3 |
| $a^{(i)}$ | ativação do último token do exemplo $i$ no bloco $\ell$ |
| $n$ | exemplos de treino **por classe** na subamostra |
| $N = 2n$ | total de exemplos de treino na subamostra |
| $S$ | uma subamostra de treino balanceada |
| $\mathcal{D}$ | span das diferenças entre pontos de treino; $\dim \le N-1$ |
| $M_t$ | matriz AGOP na iteração $t$ do RFM |
| $L,\ \lambda,\ T$ | largura do kernel, regularização ridge, número de iterações |
| $v_{\text{RFM}},\ v_{\text{DM}}$ | direções unitárias extraídas por RFM e DiffMeans |
| $\Delta_c(n)$ | diferença esperada de AUROC de teste, RFM menos DiffMeans |
| $n^\star$ | tamanho em que $\Delta$ muda de sinal, se mudar |
