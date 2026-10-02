# Propósito do projeto: um passo atrás

*Explicação em nível de começo de graduação. Escrito em 2026-10-02, durante a rodada de revisão `revision_20261002/`.*

---

## 1. O ponto de partida: um modelo de linguagem é uma composição de funções

Pense no GuppyLM como uma máquina que lê texto e, a cada palavra (token), guarda um **vetor de 384 números**. Esse vetor muda ao passar por 6 blocos. Em fórmula:

$$
a_0 \;\xrightarrow{\text{bloco }0}\; a_1 \;\xrightarrow{\text{bloco }1}\; \cdots \;\xrightarrow{\text{bloco }5}\; a_6 \;\longrightarrow\; \text{próxima palavra},
\qquad a_\ell \in \mathbb{R}^{384}.
$$

Os vetores $a_\ell$ se chamam **ativações**. São o "estado interno" do modelo: tudo o que ele "sabe" sobre o texto até ali precisa estar codificado nesses 384 números.

## 2. A hipótese central da área: conceitos são direções

### Em qual espaço?

O espaço é o **residual stream na saída do bloco $\ell$, na posição do último token real do prompt**. Cada bloco *soma* uma correção ao vetor que recebe:

$$
x_{\ell+1} = x_\ell + \mathrm{Attn}_\ell(\cdot) + \mathrm{MLP}_\ell(\cdot), \qquad x_\ell \in \mathbb{R}^{384}.
$$

Para um prompt $x$, a ativação usada é $a_\ell(x) := x_{\ell+1}^{(t_{\text{last}})} \in E_\ell := \mathbb{R}^{384}$ (`rfm_guppy/activations.py:65–68`). Três observações:

- **Um espaço por bloco.** Os $E_\ell$ têm as mesmas coordenadas, porque os blocos somam ao stream em vez de substituí-lo, mas a distribuição dos pontos muda (norma mediana de $11{,}85$ no bloco 0 a $30{,}86$ no bloco 4). Por isso o projeto estima um $v_\ell$ separado para cada $\ell \in \{0, \dots, 5\}$.
- **Direção, não vetor.** $v$ e $cv$ (com $c > 0$) são a mesma direção, então $v_\ell \in S^{383}$. O autovetor da AGOP só é definido a menos de sinal, e o sentido é fixado depois, pelo sinal da correlação com o rótulo.
- **Não é nenhum outro espaço do modelo.** Não é a camada oculta do MLP ($\mathbb{R}^{768}$), nem uma cabeça de atenção ($\mathbb{R}^{64}$), nem os logits ($\mathbb{R}^{4096}$), nem os pesos.

### A hipótese

A **hipótese da representação linear** é uma afirmação empírica, não um teorema. Para um conceito $c$ (tom positivo ou negativo, falar de comida, etc.) e um bloco $\ell$, ela diz que existem $v_\ell \in S^{383}$ e uma referência $\mu_\ell \in \mathbb{R}^{384}$ (a média de treino) tais que:

- **Ler:** a projeção $\langle a_\ell(x) - \mu_\ell,\, v_\ell \rangle$ indica se $c$ está presente. O $\mu_\ell$ aparece porque as ativações não estão centradas na origem.
- **Escrever:** somar $\varepsilon\, v_\ell$ ao residual stream do bloco $\ell$, **em todas as posições** (`rfm_guppy/steering.py:74`), faz o modelo passar a produzir $c$.

A leitura usa uma posição e a escrita usa todas. Nada garante a priori que a mesma direção sirva para as duas: a hipótese assume isso.

*Sutileza:* com `normalize=True` (caso de `food`), o RFM extrai $v_\ell$ no espaço das ativações centradas e normalizadas, $(a_\ell - \mu_\ell)/\lVert a_\ell - \mu_\ell \rVert \in S^{383}$. O steering, porém, soma $v_\ell$ no espaço bruto $E_\ell$.

Uma analogia geométrica: imagine os pontos $a_\ell(x)$ como uma nuvem em $E_\ell$. Se a hipótese vale, os textos "sobre comida" e "não sobre comida" ficam em lados opostos de um hiperplano afim, e $v_\ell$ é a seta perpendicular a esse hiperplano.

Daí saem as **três operações** que o projeto implementa:

| Operação | Pergunta | O que fazemos |
|---|---|---|
| **Extração** | Qual é a direção $v$? | Um algoritmo recebe exemplos rotulados e devolve $v$. |
| **Steering** | Empurrar ao longo de $v$ muda o comportamento? | Gerar texto com $a_\ell + \varepsilon v$ e ver o que sai. |
| **Monitoring** | Projetar em $v$ detecta o conceito? | Classificar textos usando $\langle a_\ell - \mu_\ell, v_\ell \rangle$. |

No projeto, o tamanho do empurrão é relativo à escala típica das ativações de cada bloco:

$$
\varepsilon_\ell = r \cdot \operatorname{mediana}\, \lVert a_\ell \rVert ,
$$

com $r$ varrendo valores positivos e negativos.

### Como a projeção é feita

Com $\tilde a = a_\ell - \mu_\ell$ e $\lVert v_\ell \rVert_2 = 1$, projetar é calcular o escalar $p = \langle \tilde a, v_\ell \rangle = \sum_j \tilde a_j (v_\ell)_j$, o comprimento com sinal da "sombra" de $\tilde a$ na reta de $v_\ell$. Um limiar em $p$ já é um classificador.

No código (`rfm_guppy/monitoring.py:51–57`), projeta-se nos 3 autovetores principais da AGOP, $V^\top \tilde a \in \mathbb{R}^3$ com $V = [v_1\, v_2\, v_3]$, nos 6 blocos. Os 18 números resultantes entram numa regressão logística. Logo, o AUROC reportado avalia uma direção aprendida dentro do subespaço gerado por esses 18 autovetores, e não apenas o $v_1$ usado no steering.

## 3. O que Beaglehole et al. afirmam

O artigo (arXiv:2502.03708) propõe **um método específico de extração**: o RFM (*Recursive Feature Machine*), com autovetores da AGOP. A ideia, de forma simples:

1. Treinar um preditor $f : \mathbb{R}^{384} \to \mathbb{R}$ que, dada a ativação $a$, diz se o conceito está presente. No RFM, $f$ é uma regressão *kernel ridge*.
2. Ver **em que direções $f$ é sensível**, isto é, para onde apontam os gradientes $\nabla f(a)$. A AGOP (*Average Gradient Outer Product*) é a matriz

$$
M = \frac{1}{n} \sum_{i=1}^{n} \nabla f(a^{(i)})\, \nabla f(a^{(i)})^\top \;\in\; \mathbb{R}^{384 \times 384}.
$$

3. O RFM repete os passos 1 e 2 $T$ vezes, usando $M$ para "deformar" o kernel a cada iteração. O kernel compara as ativações de dois prompts $i$ e $j$ no mesmo bloco $\ell$ pela distância de Mahalanobis induzida por $M$:

$$
\lVert a_\ell^{(i)} - a_\ell^{(j)} \rVert_M = \sqrt{\big(a_\ell^{(i)} - a_\ell^{(j)}\big)^\top M \,\big(a_\ell^{(i)} - a_\ell^{(j)}\big)}.
$$

  Essa distância é calculada para todos os pares de prompts de treino, formando a matriz $n \times n$

$$
K_{ij} = \exp\!\left(-\frac{\lVert a_\ell^{(i)} - a_\ell^{(j)} \rVert_M}{L}\right), \qquad i, j = 1, \dots, n.
$$

  O preditor resolve $(K + \lambda I)\,\alpha = y$ e prevê com $f(a) = \sum_{i=1}^{n} \alpha_i\, K\big(a, a_\ell^{(i)}\big)$.

   No fim, o autovetor principal $v_1$ de $M_T$ é a direção em que mexer em $a$ mais muda a previsão. Esse $v_1$ é o vetor de conceito.

A **concentração espectral da AGOP** é a fração do traço concentrada no primeiro autovalor:

$$
s = \frac{\lambda_1}{\sum_{j} \lambda_j} \in (0, 1].
$$

A afirmação forte do artigo é que esse método é **"universal"**: funciona para muitos conceitos e modelos, e **dá steering e monitoring melhores que métodos mais simples**.

Os métodos mais simples são os **baselines**:

- **DiffMeans:** $v \propto \hat\mu_+ - \hat\mu_-$, a média dos exemplos positivos menos a média dos negativos. É a coisa mais ingênua possível.
- **PCA:** a direção de maior variância das diferenças entre pares positivo–negativo.
- **LogReg:** o vetor de pesos de uma regressão logística.

## 4. O que o projeto faz

O projeto é **uma replicação em miniatura**. Em vez de modelos com bilhões de parâmetros, usa o GuppyLM, com cerca de 8,7 milhões, treinado num dataset com 60 categorias de conversa. Isso traz três vantagens:

- **Barato:** tudo roda num laptop, em minutos.
- **Controlado:** os conceitos (`food`, `valence`, `env_social`) vêm das categorias do próprio dataset de treino, então sabemos exatamente o que o modelo viu.
- **Teste de limite:** se o mecanismo do artigo funciona até num modelo minúsculo, isso diz algo sobre quão básico ele é. Se não funciona, também diz.

## 5. O que o projeto já mostrou

Em linguagem simples, com o grau de certeza de cada item:

1. **Os conceitos são linearmente separáveis nas ativações** (forte). Um classificador linear separa os textos com AUROC $\approx 1{,}0$ no teste, e isso vale até usando a direção do DiffMeans.
   *Ressalva (hipótese, não testada):* como os conceitos são categorias de tópico, a separação pode vir só das palavras do próprio prompt. Seria a hipótese linear valendo de um jeito "fácil".
2. **Empurrar ao longo de $v$ muda o texto gerado** (moderado). Com `valence`, $r > 0$ e $r < 0$ levam a textos opostos. As métricas automáticas de "sucesso", porém, estavam mal construídas, porque $r$ negativo também passava do limiar $0{,}5$. O próprio paper já trata isso como medição quebrada.
3. **O RFM não é claramente melhor que o DiffMeans** (é o que a revisão indica). Ele ganha um pouco na correlação de validação $\lvert\rho\rvert$ em `valence` ($0{,}92$ contra $0{,}80$) e `env_social` ($0{,}95$ contra $0{,}91$), e empata em `food`. Só que os hiperparâmetros do RFM foram escolhidos olhando a validação, e isso o favorece. A vantagem sobre o PCA era, em boa parte, um **bug**: `baselines.py:86` centrava a matriz de diferenças $D$, e com classes balanceadas a média das linhas é $\bar d = \hat\mu_+ - \hat\mu_-$. Daí

$$
\tilde D^\top \tilde D = D^\top D - m\, \bar d\, \bar d^\top ,
$$

   ou seja, a centragem **removia exatamente a direção DiffMeans** antes do PCA. Sem centragem, o PCA chega a AUROC entre $0{,}99$ e $1{,}00$.
4. **"$s = 1$" não quer dizer "conceito perfeitamente linear"** (forte). Com rótulos **embaralhados**, que não carregam conceito nenhum, $s$ já sobe de cerca de $0{,}1$–$0{,}3$ em $T = 1$ para cerca de $0{,}5$–$0{,}7$ em $T = 10$, só por iterar o RFM. Então $s$ só significa algo comparado a um nulo com os mesmos $(\sigma, T, \text{normalize})$.

## 6. Por que parece que há "muita coisa" acontecendo

Cada checagem da revisão responde a uma de duas perguntas:

> **(P1)** O RFM encontra uma direção que os métodos simples não encontram?
>
> **(P2)** Os números que reportamos medem o que dizemos que medem?

| Checagem | Pergunta que responde |
|---|---|
| Check II (bug do PCA) | P1: a comparação com o baseline era justa? |
| Check 01, regimes A e B (rótulos embaralhados) | P2: $s = 1$ é evidência de conceito ou artefato do algoritmo? |
| Teste $s(T)$ | P2: a iteração infla $s$ sozinha? |
| Enigma do `food` embaralhado (direções sem conceito com $\lvert\rho\rvert \approx 0{,}86$) | P2: $\lvert\rho\rvert$ alto prova que a direção é do conceito? |
| $\cos(v_{\text{RFM}}, v_{\text{DM}})$ por bloco (ainda não feito) | P1, diretamente: as direções são diferentes? |

São ferramentas diferentes para as mesmas duas perguntas.

## 7. Onde isso se encaixa em entender steering

A literatura de steering hoje é quase toda do tipo **"funciona?"**: alguém propõe um método, mostra exemplos bons e uma tabela em que ganha. Quase não há resposta para **"por que funciona, quando funciona e o que exatamente a direção encontrada é?"**. Essa segunda pergunta é o tema do mestrado: *propriedades matemáticas do steering*.

O projeto contribui em três níveis:

1. **Como laboratório:** um ambiente pequeno e controlado, onde dá para rodar nulos, permutações e comparações que seriam caras num modelo grande.
2. **Como crítica metodológica:** já mostrou duas coisas concretas. A estatística $s$ não se interpreta sem um nulo, e a vantagem do RFM depende de baselines bem implementados. Isso é contribuição real, mesmo sendo "negativa".
3. **Como ponte para a teoria:** se $v_{\text{RFM}} \approx v_{\text{DM}}$ quando os conceitos são linearmente separáveis, surge uma pergunta matemática precisa.

> **Conjectura (informal, não demonstrada).** Considere duas classes com médias $\mu_\pm \in \mathbb{R}^d$ e covariância comum $\Sigma \succ 0$, e o preditor kernel do RFM com $T$ iterações. Sob hipóteses ainda a determinar sobre $\Sigma$, o kernel e a regularização, o autovetor principal da AGOP $M_T$ converge (quando $n \to \infty$), ou fica próximo, da direção de Fisher
> $$ v_{\text{Fisher}} \propto \Sigma^{-1} (\mu_+ - \mu_-). $$

Se for verdade, o RFM seria uma forma não linear e cara de chegar a uma direção clássica. O experimento $\cos(v_{\text{RFM}}, v_{\text{DM}})$ por bloco é o primeiro teste empírico dessa ideia. Note que $v_{\text{DM}}$ coincide com $v_{\text{Fisher}}$ só quando $\Sigma \propto I$.

## 8. Uma frase para guardar

> **Este projeto testa, num modelo pequeno e controlado, se o método de extração de conceitos proposto por Beaglehole et al. encontra algo além do que métodos lineares simples já encontram, e se as estatísticas usadas para defendê-lo significam o que dizem.**

Critério para tarefas novas: ela responde a P1 ou a P2? Se não responder a nenhuma das duas, vai para `Perguntas de pesquisa`, não para a fila do dia.
