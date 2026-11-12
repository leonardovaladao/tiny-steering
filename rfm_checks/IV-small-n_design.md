# Check IV — Comparação em amostra pequena (F3): proposição, previsão e desenho

Rodada de revisão de 2026-10. Documento de desenho, escrito **antes** de rodar o experimento.
Notebook previsto: `IV-small-n.ipynb`.

Notação geral: $d = 384$ é a dimensão do residual stream, $\ell$ é o bloco, e
$a^{(i)} \in \mathbb{R}^d$ é a ativação do último token do exemplo $i$ no bloco $\ell$, que
fica fixo e é omitido da notação. Uma subamostra de treino com $n$ exemplos por classe tem
$N = 2n$ pontos e rótulos $y \in \{0,1\}^N$, com $n$ uns e $n$ zeros.

---

## 1. Proposição: todos os métodos escolhem a direção no mesmo subespaço

### 1.1 Objetos

**Pré-processamento.** Seja $\bar a = \frac1N \sum_{i=1}^N a^{(i)}$ a média da subamostra e
$\tilde a^{(i)} = a^{(i)} - \bar a$. Com `normalize=True`, usa-se
$b^{(i)} = \tilde a^{(i)} / \lVert \tilde a^{(i)} \rVert_2$; com `normalize=False`,
$b^{(i)} = \tilde a^{(i)}$. Isso segue `rfm_guppy/rfm.py:_preprocess` e o Apêndice A do artigo.

**Subespaço das diferenças.**

$$
\mathcal{D} := \operatorname{span}\{\, a^{(i)} - a^{(j)} : 1 \le i, j \le N \,\} \subseteq \mathbb{R}^d .
$$

**Lema 0.** $\mathcal{D} = \operatorname{span}\{\tilde a^{(1)}, \dots, \tilde a^{(N)}\}$ e
$\dim \mathcal{D} \le N - 1$. Se nenhum $\tilde a^{(i)}$ é nulo, também vale
$\operatorname{span}\{b^{(i)} - b^{(j)}\} \subseteq \operatorname{span}\{b^{(i)}\} = \mathcal{D}$.

*Prova.* A identidade $\tilde a^{(i)} = \frac1N \sum_{j}(a^{(i)} - a^{(j)})$ dá
$\tilde a^{(i)} \in \mathcal{D}$. Na outra direção,
$a^{(i)} - a^{(j)} = \tilde a^{(i)} - \tilde a^{(j)}$. Como $\sum_i \tilde a^{(i)} = 0$, os
$N$ vetores $\tilde a^{(i)}$ são linearmente dependentes, e por isso $\dim \le N-1$. Os
$b^{(i)}$ são múltiplos escalares não nulos dos $\tilde a^{(i)}$, logo
$\operatorname{span}\{b^{(i)}\} = \mathcal{D}$, e as diferenças entre eles ficam nesse mesmo
span. $\square$

**Kernel.** Para $M \in \mathbb{R}^{d\times d}$ simétrica com $M \succeq 0$ e $L > 0$:

$$
\lVert u \rVert_M := \sqrt{u^\top M u}, \qquad
K_M(x, z) := \exp\!\Big(-\tfrac{1}{L}\,\lVert x - z \rVert_M\Big).
$$

Quando $M$ é singular, $\lVert\cdot\rVert_M$ é apenas uma seminorma. Para $M \succeq 0$, vale
$\lVert u \rVert_M = 0 \iff M^{1/2}u = 0 \iff Mu = 0$.

**RFM (artigo, eq. (1) e Steps 1–2; código `_agop_step`).** Começa-se com $M_0 = I$ e
$\lambda > 0$. Para $t = 0, 1, \dots, T-1$:

1. *Kernel ridge regression.* $\alpha_t = \big(K_{M_t}(B, B) + \lambda I_N\big)^{-1} y$, com
   $K_{M_t}(B,B)_{ij} = K_{M_t}(b^{(i)}, b^{(j)})$ e $f_t(z) = \sum_i \alpha_{t,i}\, K_{M_t}(z, b^{(i)})$.
2. *Gradiente, com a convenção do código.* O kernel Laplaciano não é diferenciável em
   $z = b^{(i)}$. Por isso o código descarta os termos com distância nula
   (`D < eps` $\Rightarrow$ `W = 0`). Formalmente,
   $$
   g_t(z) := -\frac{1}{L} \sum_{i \,:\, \lVert z - b^{(i)}\rVert_{M_t} > 0}
   \alpha_{t,i}\, \frac{K_{M_t}(z, b^{(i)})}{\lVert z - b^{(i)} \rVert_{M_t}}\; M_t\,\big(z - b^{(i)}\big).
   $$
   Em todo ponto $z$ com $\lVert z - b^{(i)}\rVert_{M_t} > 0$ para todo $i$, $f_t$ é
   diferenciável e $g_t(z) = \nabla f_t(z)$.
3. *AGOP com gradientes centrados.* Com $g_j := g_t(b^{(j)})$ e $\bar g := \frac1N \sum_j g_j$,
   $$
   M_{t+1} = \frac{1}{N} \sum_{j=1}^N (g_j - \bar g)(g_j - \bar g)^\top .
   $$

A direção RFM é $v_{\text{RFM}} = \pm\, u_1(M_T)$, em que $u_1$ é um autovetor unitário
associado ao maior autovalor de $M_T$. O sinal é escolhido pela correlação de Pearson com
$y$ no treino.

### 1.2 Enunciado

> **Proposição 1.** Nas definições acima, com `normalize` igual a `True` ou `False`, vale
> $\operatorname{range}(M_t) \subseteq \mathcal{D}$ para todo $t \ge 1$. Em consequência:
>
> (a) $\operatorname{rank}(M_t) \le N - 1 = 2n - 1$;
>
> (b) todo autovetor de $M_t$ com autovalor não nulo pertence a $\mathcal{D}$;
>
> (c) se $M_T \neq 0$, então $v_{\text{RFM}} \in \mathcal{D}$.
>
> O resultado não depende de $y$, de $L$, de $\lambda$ nem de $T$.

### 1.3 Prova

Vamos mostrar, por indução em $t \ge 0$, a afirmação

$$
(\mathrm{I}_t): \qquad M_t\,\mathcal{D} \subseteq \mathcal{D}.
$$

*Base.* $M_0 = I$, então $(\mathrm{I}_0)$ vale trivialmente.

*Passo.* Suponha $(\mathrm{I}_t)$. Pelo Lema 0, $b^{(j)} - b^{(i)} \in \mathcal{D}$ para
quaisquer $i, j$, e então $M_t(b^{(j)} - b^{(i)}) \in \mathcal{D}$. Pela definição de
$g_t$, cada $g_j = g_t(b^{(j)})$ é combinação linear de vetores da forma
$M_t(b^{(j)} - b^{(i)})$, logo $g_j \in \mathcal{D}$. Também
$\bar g \in \mathcal{D}$ e $g_j - \bar g \in \mathcal{D}$.

Para qualquer $x \in \mathbb{R}^d$,

$$
M_{t+1}\, x = \frac{1}{N} \sum_j \big\langle g_j - \bar g,\, x \big\rangle\, (g_j - \bar g) \in \mathcal{D},
$$

então $\operatorname{range}(M_{t+1}) \subseteq \mathcal{D}$. Em particular,
$M_{t+1}\mathcal{D} \subseteq \mathcal{D}$, que é $(\mathrm{I}_{t+1})$. O argumento também
provou $\operatorname{range}(M_{t+1}) \subseteq \mathcal{D}$ para todo $t \ge 0$, que é a
afirmação principal.

*(a)* Segue de $\dim \mathcal{D} \le N-1$ (Lema 0).
*(b)* Se $M_t u = \mu u$ com $\mu \neq 0$, então $u = \mu^{-1} M_t u \in \operatorname{range}(M_t) \subseteq \mathcal{D}$.
*(c)* $M_T \succeq 0$ e $M_T \neq 0$ implicam que o maior autovalor é positivo, e (b) se
aplica. Trocar o sinal não tira o vetor de $\mathcal{D}$. $\square$

**Observação 1.** Como $M_t$ é simétrica, $\ker(M_t) = \operatorname{range}(M_t)^\perp \supseteq \mathcal{D}^\perp$.
Então, a partir de $t = 1$, a distância $\lVert\cdot\rVert_{M_t}$ ignora toda direção
ortogonal ao *span* afim dos dados de treino. O RFM não tem como "descobrir" uma direção
fora de $\mathcal{D}$.

**Observação 2.** A restrição só tem conteúdo enquanto $2n - 1 < d$, isto é, $n \le 192$.
Na grade do experimento ($n \le 32$) temos $\dim\mathcal{D} \le 63$ dentro de $\mathbb{R}^{384}$.

### 1.4 Corolário: os baselines também ficam em $\mathcal{D}$

> **Corolário 1.** Para a mesma subamostra:
>
> (i) **DiffMeans.** $\hat\mu_+ - \hat\mu_- \in \mathcal{D}$.
>
> (ii) **PCA das diferenças pareadas**, centrado ou não. O primeiro vetor singular à direita
> está em $\mathcal{D}$.
>
> (iii) **Regressão logística com penalidade $\ell_2$**, sobre os dados centrados, com ou sem
> intercepto penalizado. O minimizador exato $w^\star$ está em $\mathcal{D}$.

*Prova.*

(i) Escrevendo $P$ e $Q$ para os índices positivos e negativos, com $|P| = |Q| = n$:

$$
\hat\mu_+ - \hat\mu_- = \frac{1}{n^2} \sum_{i \in P} \sum_{j \in Q} \big(a^{(i)} - a^{(j)}\big) \in \mathcal{D}.
$$

(ii) Cada linha $D_k = a^{(\pi_+(k))} - a^{(\pi_-(k))}$ da matriz de diferenças está em
$\mathcal{D}$. A versão centrada subtrai a média das linhas, que também está em
$\mathcal{D}$. O primeiro vetor singular à direita é $v = \sigma_1^{-1} D^\top u_1$, uma
combinação das linhas, logo $v \in \mathcal{D}$.

(iii) O objetivo tem a forma
$J(w, b) = \tfrac12 \lVert w \rVert^2 + \tfrac{\beta}{2} b^2 + C \sum_i \ell\big(y_i,\, \langle w, \tilde a^{(i)}\rangle + b\big)$,
com $\beta \in \{0, 1\}$ (o `liblinear` penaliza o intercepto, o que corresponde a $\beta = 1$).
Decomponha $w = w_{\mathcal{D}} + w_\perp$ com $w_\perp \in \mathcal{D}^\perp$. Como
$\tilde a^{(i)} \in \mathcal{D}$, os termos de perda dependem só de $w_{\mathcal{D}}$, e
$\lVert w \rVert^2 = \lVert w_{\mathcal{D}}\rVert^2 + \lVert w_\perp\rVert^2$. Então
$J(w_{\mathcal{D}}, b) < J(w, b)$ sempre que $w_\perp \neq 0$, e o minimizador tem
$w_\perp = 0$. $\square$

*Ressalva numérica.* (iii) vale para o minimizador exato. O solver devolve uma aproximação,
então o notebook verifica a fração $\lVert P_{\mathcal{D}^\perp} w \rVert / \lVert w \rVert$,
que deve ficar no nível do erro de ponto flutuante.

### 1.5 Corolário: com um exemplo por classe, os quatro métodos coincidem

> **Corolário 2.** Seja $n = 1$, com pontos $a^{+} \neq a^{-}$, e
> $u := (a^+ - a^-) / \lVert a^+ - a^- \rVert_2$. Para todo $L > 0$, $\lambda > 0$ e
> $T \ge 1$, com `normalize=False`:
> $$
> v_{\text{RFM}} = v_{\text{DM}} = v_{\text{PCA}} = v_{\text{LogReg}} = u .
> $$
> Em particular, $\Delta(1) = 0$ exatamente, para qualquer métrica calculada a partir da direção.

*Prova.*

*Passo 1: $\mathcal{D}$ é uma reta.* Com $N = 2$, temos $\mathcal{D} = \operatorname{span}\{u\}$.

*Passo 2: $M_T \neq 0$.* Escreva $\delta := \tilde a^{+} - \tilde a^{-} = a^+ - a^-$ e suponha
$M_t = c_t\, \delta\delta^\top$ com $c_t > 0$ (para $t = 0$ vale a variante $M_0 = I$, e o
cálculo abaixo é o mesmo). Então $\lVert\delta\rVert_{M_t} > 0$ e
$\kappa := K_{M_t}(\tilde a^+, \tilde a^-) \in (0, 1)$. Com $y = (1, 0)$:

$$
\alpha_t = \frac{1}{(1+\lambda)^2 - \kappa^2}\begin{pmatrix} 1+\lambda \\ -\kappa \end{pmatrix}.
$$

Com $s := \kappa \big/ \big(L\, \lVert\delta\rVert_{M_t}\, [(1+\lambda)^2 - \kappa^2]\big) > 0$,
os gradientes são $g_+ = s\,\kappa\, M_t\delta$ e $g_- = s\,(1+\lambda)\, M_t\delta$. Depois de
centrar,

$$
g_\pm - \bar g = \pm\, \tfrac{s}{2}\,(\kappa - 1 - \lambda)\, M_t \delta
\quad\Longrightarrow\quad
M_{t+1} = \Big(\tfrac{s\,(1 + \lambda - \kappa)}{2}\Big)^{2} (M_t\delta)(M_t\delta)^\top .
$$

Como $1 + \lambda - \kappa > 0$ e $M_t\delta \neq 0$ (pois $\lVert\delta\rVert_{M_t} > 0$),
obtém-se $M_{t+1} = c_{t+1}\,\delta\delta^\top$ com $c_{t+1} > 0$. Por indução,
$M_T \neq 0$ para todo $T \ge 1$. O caso $y = (0, 1)$ é simétrico.

*Passo 3: as quatro direções são $\pm u$.* Pela Proposição 1(c) e pelo Corolário 1, as quatro
direções são vetores unitários em $\operatorname{span}\{u\}$, ou seja, $\pm u$.

*Passo 4: a orientação escolhe $+u$.* Todos os métodos orientam pelo sinal da correlação de
Pearson entre $\langle \tilde a^{(i)}, v \rangle$ e $y_i$ no treino. Com $v = u$, as projeções
são $\pm \lVert\delta\rVert/2$, e a correlação com $y = (1, 0)$ é $+1$. Logo todas as
direções orientadas são $+u$. $\square$

**Uso no experimento.** O Corolário 2 vira um teste de sanidade da implementação. Se, com
$n = 1$, as direções do código não coincidirem a menos de erro de ponto flutuante, há um bug
ou uma etapa de pré-processamento fora do modelo acima.

---

## 2. Previsão teórica

Esta seção separa o que é **teorema** do que é **hipótese**.

### 2.1 O que já é teorema

- Para qualquer $n$, as quatro direções estão em $\mathcal{D}$, com $\dim\mathcal{D} \le 2n-1$
  (Proposição 1 e Corolário 1). Em amostra pequena, a comparação entre métodos é sobre
  **qual direção cada um escolhe dentro de $\mathcal{D}$**.
- Com $n = 1$, $\Delta(1) = 0$ (Corolário 2). Qualquer diferença entre os métodos só pode
  aparecer a partir de $n = 2$.

### 2.2 Referência clássica: Fisher e o regime $n \ll d$

Para duas classes gaussianas $\mathcal{N}(\mu_\pm, \Sigma)$ com $\Sigma \succ 0$ e prioris
iguais, a regra de Bayes é linear, com direção

$$
w^\star = \Sigma^{-1}(\mu_+ - \mu_-).
$$

O DiffMeans troca $\Sigma$ por $I$: $v_{\text{DM}} \propto \hat\mu_+ - \hat\mu_-$. Ele
coincide com $w^\star$ quando $\Sigma \propto I$, ou quando $\mu_+ - \mu_-$ é autovetor de
$\Sigma$.

Com $N = 2n < d$, a covariância amostral $\hat\Sigma$ tem posto $\le N - 2$ e não é
invertível. A regra de Fisher *plug-in* precisa então de inversa generalizada ou de
regularização. Bickel & Levina (2004, *Bernoulli* 10(6)) estudam esse regime. Sob as
hipóteses deles, quando $d/n \to \infty$, a regra de Fisher com inversa generalizada tem
erro que tende a $1/2$, enquanto a regra de independência, que ignora as correlações, pode
manter erro próximo do ótimo. **Antes de citar na dissertação, conferir na fonte o enunciado
exato e as hipóteses do teorema.**

**Ponte para o nosso caso (heurística, não teorema).**

- O DiffMeans é um estimador sem hiperparâmetros. Sob amostragem independente,
  $\operatorname{Cov}(\hat\mu_+ - \hat\mu_-) = (\Sigma_+ + \Sigma_-)/n$, então o erro de
  estimação cai como $1/n$.
- O RFM e a LogReg usam estrutura de segunda ordem dos dados: o RFM pela matriz $M_t$ e pelos
  coeficientes $\alpha_t$, a LogReg pela curvatura da perda. Com $N$ pequeno, essa estrutura
  é estimada com muito ruído.
- A expectativa é um *trade-off* viés–variância. Para $n$ pequeno, o DiffMeans (viesado,
  mas de baixa variância) vence. Para $n$ grande, métodos que estimam mais estrutura podem
  vencer, se houver estrutura além de $\mu_+ - \mu_-$ para aproveitar.

### 2.3 Hipóteses

Para um conceito $c$ e uma métrica $m$ (seção 3.4):

$$
\Delta_c(n) := \mathbb{E}_S\!\left[\, m\big(v_{\text{RFM}}(S)\big) - m\big(v_{\text{DM}}(S)\big) \right],
$$

em que $S$ é uma subamostra balanceada de tamanho $n$ por classe, tirada do conjunto de
treino de $c$.

> **H3a (conjectura: DiffMeans vence em amostra pequena).** Para $n \in \{2, 4\}$,
> $\Delta_c(n) \le 0$ nos dois conceitos.

> **H3b (conjectura: as curvas se cruzam).** Existe $n^\star$ na grade tal que
> $\Delta_c(n) < 0$ para algum $n < n^\star$ e $\Delta_c(n) > 0$ para algum $n \ge n^\star$.

> **H-artigo (a afirmação que está sendo testada).** Existe $n$ na grade em que
> $\Delta_c(n) > 0$, isto é, o RFM extrai uma direção melhor que o DiffMeans no Guppy.

H3a e H-artigo não são complementares: as duas podem valer ao mesmo tempo, e esse é
justamente o cenário de H3b.

---

## 3. Desenho formal

### 3.1 Conceitos e blocos

| Papel | Conceito | Pool de treino (por classe) | Teste (por classe) |
|---|---|---|---|
| principal | `valence` | 60 | 20 |
| principal | `env_social` | 180 | 60 |
| controle de teto | `food` | 180 | 60 |

`food` entra só como controle. No teste de viabilidade (`scratchpad/f3_probe.py`), o
DiffMeans já tem AUROC mediano igual a 1 com $n = 1$, então esse conceito não discrimina
métodos.

**Bloco primário:** $\ell = 3$, fixado agora. Em `valence` e `env_social` ele não está entre
as células indistinguíveis do nulo no Check Ia (que foram valence b0, b2, b5 e env_social b1).
**Análise secundária:** todos os 6 blocos, com os mesmos critérios, reportados sem seleção.

Nenhuma inferência nova: tudo usa `data/cache/{conceito}_{train,test}_activations.npy`.

### 3.2 Amostragem

- **Grade:** $n \in \{1, 2, 4, 8, 16, 32\}$, mais o ponto $n = n_{\max}$ (60 ou 180), que é
  uma única avaliação determinística.
- **Repetições:** $R = 200$ subamostras independentes por $(c, n)$. Cada uma tira, sem
  reposição, $n$ índices positivos e $n$ negativos do pool de treino. A semente é fixa e fica
  registrada.
- **Desenho pareado:** a **mesma** subamostra $S_r$ alimenta os quatro métodos. Os
  contrastes são diferenças pareadas por $r$.

### 3.3 Métodos e hiperparâmetros

> **Correção de premissa sobre $L$.** O artigo **não** usa $L$ fixo. O Apêndice A diz que
> $(L, T, \text{normalize})$ é escolhido por busca em $L \in \{1, 10, 100\}$,
> $T \in \{1, \dots, 10\}$ e normalize $\in \{\text{sim}, \text{não}\}$, maximizando $|\rho|$
> numa validação de 20% do treino, com $\lambda = 10^{-3}$ fixo. Essa busca não roda em
> amostra pequena: com $n = 2$, 20% dos dados é menos de um ponto por classe.
>
> **Escolha adotada (decidida em 2026-10-08):** congelar $(L, T)$ no valor que o procedimento do
> artigo escolhe com o treino completo, com **`normalize=False` fixo para todos os conceitos e
> blocos**. É a versão de "$L$ fixo" mais fiel ao artigo. Os mesmos valores congelados valem
> para todo $n$ e para toda subamostra.
>
> **Isonomia entre conceitos.** O normalize não é escolhido por conceito. Todos os conceitos
> passam pela mesma regra: `extract_concept_vector` com `sweep={"normalize": [False]}`, isto
> é, busca em $L \in \{1, 10, 100\}$ e $T \in \{1, \dots, 10\}$, ajuste no split `train`
> completo, seleção por $|\rho|$ no split `val` do cache, $\lambda = 10^{-3}$. Em caso de
> empate, vale a primeira combinação na ordem de varredura do código ($L$ crescente, depois
> $T$ crescente). Escolheu-se `False` por dois motivos:
>
> 1. O Corolário 2 está provado só para `normalize=False`. Assim, o critério C0 com $n = 1$
>    cobre todos os conceitos.
> 2. A busca original já escolheu `False` em `valence` e `env_social`, em todos os 6 blocos.
>
> **Consequência para o cache.** O código varre `normalize=False` primeiro e só troca de
> melhor com desigualdade estrita. Então, onde o cache tem `hp_normalize = 0`, a busca restrita
> devolve os mesmos $(L, T)$ do cache. Isso cobre `valence` e `env_social` em todos os blocos e
> `food` no bloco 0. Já `food` nos blocos 1–5 tinha `normalize=True` no cache, e lá $(L, T)$ é
> **recalculado**. O passo 0 do notebook recalcula tudo, verifica a igualdade com o cache onde
> ela deve valer e **só depois** abre o split de teste.
>
> **Alternativa descartada:** fixar $L = 10$, o centro da grade do artigo, em todos os
> conceitos, com $T$ e normalize também fixados *a priori*.

| Método | Definição na subamostra $S$ | Hiperparâmetros (bloco 3) |
|---|---|---|
| RFM | Seção 1.1, top autovetor de $M_T$, orientado | normalize = não em todos. `valence`: $L=10$, $T=10$. `env_social`: $L=10$, $T=3$. `food`: recalculado no passo 0 (o cache tinha $L=1$, $T=2$ com normalize = sim). Todos com $\lambda = 10^{-3}$ |
| DiffMeans | $(\hat\mu_+ - \hat\mu_-)/\lVert\cdot\rVert$, orientado | nenhum |
| PCA | **não centrado** (Apêndice H do artigo), com pareamento aleatório dos $n$ positivos com os $n$ negativos de $S$ | nenhum |
| LogReg | `liblinear`, penalidade $\ell_2$, $w/\lVert w\rVert$ orientado | $C$ congelado, recalculado no passo 0 (não está salvo no cache), com a mesma regra para todos os conceitos: ajuste no split `train` completo, seleção por $|\rho|$ no split `val` do cache, grade $\{1000, 100, 1, 0.1\}$, empate resolvido pela primeira na ordem da grade, como em `baselines.logreg_vector`. Essa função não devolve o $C$, então o notebook reproduz o laço e o registra |

**Viés da escolha, declarado:** os hiperparâmetros do RFM e da LogReg foram escolhidos com o
treino completo, olhando a validação. Isso **favorece** esses dois métodos frente ao
DiffMeans e ao PCA, que não têm hiperparâmetros. Se o RFM não vencer mesmo assim, a conclusão
é robusta. Se vencer, a vantagem é limite superior.

### 3.3.1 Sensibilidade a $T$ (análise secundária, decidida em 2026-10-08)

O teste primário usa o $T$ congelado da seção 3.3. Nenhuma escolha de $T$ feita depois de ver
os dados altera o teste primário.

Para cada $(c, n, r)$, o RFM é ajustado uma única vez até $T_{\max} = 10$, guardando
$M_1, \dots, M_{10}$. Como `rfm_fit` devolve só $M_T$, o notebook chama `_agop_step` num laço.
Seja $v_{\text{RFM}}^{(T)}(S_r)$ o top autovetor orientado de $M_T$. Definem-se

$$
\Delta_c(n, T) := \mathbb{E}_S\!\left[ m\big(v^{(T)}_{\text{RFM}}(S)\big) - m\big(v_{\text{DM}}(S)\big) \right], \qquad T = 1, \dots, 10,
$$

e o estimador $\hat\Delta_c(n, T)$, com IC calculado como na seção 3.5.

1. **Curva $\hat\Delta_c(n, T)$.** É descritiva e não envolve seleção. Mostra se a conclusão
   do teste primário depende do $T$ escolhido.
2. **$T$ oráculo.** Para cada célula $(c, n)$, seja
   $T^\star_{c,n} := \arg\max_T \hat\Delta_c(n, T)$. É **um** $T$ por célula, não um por
   subamostra. Define-se $\hat\Delta^{\max}_c(n) := \hat\Delta_c(n, T^\star_{c,n})$, com o IC
   bootstrap das $R$ diferenças pareadas nesse $T$. A seleção de $T$ é feita no teste e não é
   incorporada ao IC. Por isso $\hat\Delta^{\max}_c(n)$ é um **limite superior otimista do RFM**:
   - pode **reforçar** H3a (critério na seção 3.6);
   - **não pode** sustentar H-artigo nem H3b.

Pela Proposição 1, o C0 vale para todo $T$. Pelo Corolário 2, $\Delta_c(1, T) = 0$ para todo
$T$ quando `normalize=False`.

**Fora do desenho:** escolher $T$ por validação cruzada leave-one-out dentro de $S$. Com
$n = 2$, cada ajuste teria 3 pontos, e a AUROC de um único ponto de validação não é definida.

### 3.4 Métricas

Para uma direção unitária $v$, o escore de um exemplo de teste é
$s(x) = \langle a_\ell(x), v \rangle$. A AUROC não muda ao somar uma constante ao escore,
então não é preciso centrar.

- **Primária:** AUROC no split de teste completo, $m_{\text{AUC}}(v)$.
- **Secundária:** $m_\rho(v) = |\operatorname{corr}(s(x), y)|$ no teste.
- **Descritivas, ligadas a P1:**
  - $|\cos(v_{\text{RFM}}, v_{\text{DM}})|$ por subamostra;
  - $\operatorname{rank}(M_T)$ numérico;
  - os resíduos $\lVert P_{\mathcal{D}^\perp} v \rVert$ dos quatro métodos, que verificam a
    Proposição 1 e o Corolário 1 na prática.

### 3.5 Estimador e incerteza

Para cada $(c, n)$ e cada par de métodos (RFM–DM é o principal; RFM–LogReg e RFM–PCA são
secundários):

$$
\hat\Delta_c(n) = \frac{1}{R}\sum_{r=1}^R \Big[ m\big(v_{\text{RFM}}(S_r)\big) - m\big(v_{\text{DM}}(S_r)\big) \Big].
$$

O IC de 95% vem de um bootstrap percentil sobre as $R$ diferenças pareadas, com 10.000
reamostragens.

**O que o IC cobre:** a variabilidade da subamostra de treino, *condicional* ao pool de treino
e ao conjunto de teste fixos. **O que não cobre:** a variabilidade do conjunto de teste (só 40
exemplos em `valence`) e a de um pool de treino novo.

**Comparações múltiplas (só H-artigo).** H-artigo afirma que *existe* uma célula com
$\Delta_c(n) > 0$, então é testada com controle da taxa de erro familiar sobre as $k = 10$
células $(c, n)$, com $c \in \{\texttt{valence}, \texttt{env\_social}\}$ e
$n \in \{2, 4, 8, 16, 32\}$.

1. *p-valor por célula.* Para cada célula, com $\hat\Delta^{*}_{b}$ a média da $b$-ésima
   reamostragem bootstrap ($b = 1, \dots, B$, $B = 10.000$), o p-valor unilateral para
   $H_0: \Delta_c(n) \le 0$ é
   $$
   p_{c,n} := \frac{1 + \#\{\, b : \hat\Delta^{*}_{b} \le 0 \,\}}{1 + B}.
   $$
2. *Holm.* Ordenam-se os $k$ p-valores, $p_{(1)} \le \dots \le p_{(k)}$. Seja $i^\star$ o maior
   $i$ tal que $p_{(j)} \le \alpha/(k - j + 1)$ para todo $j \le i$, com $\alpha = 0{,}025$.
   As células de posições $1, \dots, i^\star$ são declaradas com $\Delta_c(n) > 0$. Se
   $p_{(1)} > \alpha/k$, nenhuma é.
3. O nível unilateral $\alpha = 0{,}025$ equivale ao limite inferior do IC bilateral de 95%
   usado nos demais critérios.

H3a, H3b e o $T$ oráculo (seção 3.3.1) usam o IC bootstrap de 95% **sem correção**, célula a
célula. H3a e H3b são afirmações sobre células específicas, não sobre a existência de uma
célula na grade.

### 3.6 Critérios de decisão (fixados antes de rodar)

| Código | Critério | Consequência |
|---|---|---|
| **C0** (sanidade) | Nos três conceitos, todos com `normalize=False`. Com $n = 1$, para toda subamostra: $1 - |\cos(v_{\text{RFM}}, v_{\text{DM}})| < 10^{-8}$ e $|\Delta| < 10^{-8}$. Para todo $n$: resíduo $\lVert P_{\mathcal{D}^\perp} v\rVert < 10^{-6}$ nos quatro métodos. No passo 0: $(L, T)$ recalculados iguais aos do cache onde `hp_normalize = 0` | Se falhar, o experimento **para** e investiga-se a implementação |
| **H3a** | É **refutada** se, em algum conceito e para $n = 2$ ou $n = 4$, o limite inferior do IC de 95% (sem correção) de $\Delta_c(n)$ for $> 0$ | — |
| **H-artigo** | É **sustentada**, restrita a este regime, se ao menos uma célula for declarada com $\Delta_c(n) > 0$ pelo procedimento de Holm da seção 3.5. Caso contrário, é **refutada** no Guppy | — |
| **H3b** | É **sustentada** se, no mesmo conceito, houver uma célula com limite superior do IC de 95% (sem correção) $< 0$ e outra, com $n$ maior, com limite inferior $> 0$ | — |
| **H3a, reforço** (secundário) | Para $n = 2$ ou $n = 4$, o limite superior do IC de 95% de $\hat\Delta^{\max}_c(n)$ é $< 0$ (seção 3.3.1) | O DiffMeans vence mesmo contra o RFM com o melhor $T$ escolhido no teste |

**Leitura dos resultados possíveis:**

| Padrão | Interpretação |
|---|---|
| $\Delta \le 0$ em toda a grade | O RFM não extrai direção melhor que o DiffMeans no Guppy, nem em amostra pequena |
| $\Delta < 0$ para $n$ pequeno, $\Delta > 0$ para $n$ grande | Cruzamento: o RFM só compensa com dados suficientes, e $n^\star$ é uma grandeza a reportar |
| $\Delta > 0$ já em $n$ pequeno | O RFM é mais eficiente em amostra que o DiffMeans, contra a previsão clássica (H3a refutada) |
| $\Delta \approx 0$ com IC estreito em toda a grade | Equivalência prática entre RFM e DM neste regime |

### 3.7 Limitações conhecidas

1. **Duas unidades de conceito.** Não há inferência *entre* conceitos com $K = 2$. A extensão
   natural é aplicar o mesmo protocolo a conceitos *one-vs-one* entre as 60 categorias.
2. **Estrutura por categoria.** Cada classe de `valence` é a união de 4 categorias. Com $n$
   pequeno, a subamostra pode cobrir só parte delas. Por isso cada subamostra registra
   quantas categorias foram cobertas, para análise posterior. Não há estratificação.
3. **Pool pequeno em `valence`.** Com $n = 32$ de um pool de 60, as subamostras se
   sobrepõem muito, e as $R$ repetições ficam correlacionadas. O IC por bootstrap é
   otimista nesse ponto da grade.
4. **Hiperparâmetros congelados** (seção 3.3). Favorecem RFM e LogReg.
5. **Só um modelo e só conceitos de tópico.** Nada aqui se estende a modelos maiores sem
   hipótese adicional.

### 3.8 Saídas previstas

- `IV-small-n.ipynb`: notebook executado.
- `IV-small-n_draws.csv`: uma linha por $(c, n, r, \text{método})$, com métricas, $\cos$,
  resíduo e categorias cobertas.
- `IV-small-n_summary.csv`: $\hat\Delta$, ICs e p-valores (bruto e posição de Holm) por
  $(c, n, \text{contraste})$.
- `IV-small-n_T_sensitivity.csv`: $\hat\Delta_c(n, T)$ e IC para $T = 1, \dots, 10$, com
  $T^\star_{c,n}$ marcado.
- `IV-small-n_curves.png`: as quatro curvas $m(n)$ e a curva $\Delta(n)$ com faixa de IC.
- `IV-small-n_T_curves.png`: $\hat\Delta_c(n, T)$ por conceito, uma curva por $n$.
- `IV-small-n.md`: resultado, no formato dos Checks I–III.

**Custo estimado:** 4–5 h, sendo cerca de +1 h pela seção 3.3.1.

---

## Emenda 1 — 2026-10-09 (antes de qualquer resultado de H3a/H-artigo/H3b)

**O que o C0 detectou.** Na primeira execução, o notebook parou na primeira célula do laço
principal: `valence`, bloco 3, $n = 1$, $r = 0$, resíduo do RFM igual a $1{,}00$. Nesse caso
$M_{10} \equiv 0$. A função `eigh` aplicada à matriz nula devolve um vetor arbitrário da base
canônica, que fica fora de $\mathcal{D}$. O passo 0 do C0 tinha passado.

**Causa.** Com $n = 1$, a escala de $M_t$ encolhe cerca de $1{,}6\times10^{3}$ vezes por
iteração (com $L = 10$), mas a direção continua sendo $\pm u$. O código de `_agop_step` e
`agop_grad` descartava os termos com $D_M < 10^{-12}$, um corte **absoluto**. Quando
$D_M(a^+, a^-)$ fica abaixo desse limiar (em $t = 9$ com $L = 10$ e em $t = 6$ com $L = 100$),
todos os pesos $W$ zeram, e com eles $M_t$. Isso contraria a convenção da seção 1.1, que só
descarta distâncias **nulas**, e o Corolário 2, que garante $c_t > 0$.

**Células afetadas no $T$ congelado, com $n = 1$ e 200 de 200 subamostras.** `valence` nos
blocos 0, 1, 3 e 4 ($L = 10$, $T = 10$); `env_social` b0 ($L = 100$, $T = 10$); `env_social`
b2 ($L = 10$, $T = 9$); `env_social` b5 ($L = 10$, $T = 10$).

**Mudança.** Nas duas funções de `rfm_guppy/rfm.py`, o corte passou a ser **relativo**: um
termo é descartado quando $D_M \le 10^{-12}\,\max D_M$. Assim, zeros exatos continuam
descartados, inclusive quando $\max D_M = 0$, e nada mais muda. Isso equivale à seção 1.1 a
menos de erro de ponto flutuante. Notebook, sementes e hiperparâmetros ficaram iguais.

**Evidência de que nada mais mudou.** Os Checks I, II e III foram reexecutados com o código
novo:
- todos os CSVs são byte a byte idênticos aos de `runs/2026-10-09/`;
- os NPZ do Check I têm diferença máxima igual a $0$;
- o gate de reprodução do Check I passou (as 18 *shares* e direções coincidem com o cache);
- os gates (a) e (b) do Check II passaram.

O cache não foi alterado, e o passo 0 do C0 confere de novo a igualdade de $(L, T)$ com o
cache. Com o código novo, o mapa de diagnóstico com $n = 1$ não tem nenhuma das 18 células
conceito×bloco com $M_t \equiv 0$ para $t \le 10$, e nenhuma subamostra falha no $T$
congelado.

**Ressalva.** No $T$ congelado, $\lambda_1(M_T)$ com $n = 1$ chega a $\sim 10^{-33}$ com
$L = 10$, $T = 10$, e a $\sim 10^{-52}$ em `env_social` b0 ($L = 100$, $T = 10$). Como o corte
agora é relativo, o resultado não é afetado. A escala de $M_t$, porém, continua encolhendo
geometricamente, e para $T$ muito maior que 10 chegaria ao *underflow* de float64.

**Nenhum resultado de H3a, H-artigo, H3b ou do reforço de H3a** tinha sido calculado antes
desta emenda: a primeira execução parou no gate, antes de qualquer contraste.
