# Edit Pass 2 — LaTeX / compilation fixes for `docs/latex/`

**For the applying agent.** This is pass 2 of 3. Pass 1 (`EDIT_PASS1_EQUATIONS.md`) has
already been applied; every line number below refers to the files **as they stand after
pass 1**.

## Scope

Pass 2 makes the document **compile cleanly to PDF**. It fixes structure, packages,
characters and floats. It does **not** change mathematics (pass 1's job, done) and does
**not** change claims or wording (pass 3's job — see the hand-off at the bottom, and do
not touch those lines).

## How this was verified

I compiled the document with TeX Live 2026 (`pdflatex` + `bibtex`, 4 passes). Every fix
below was applied to a scratch copy and re-compiled. The end state is:

```
Output written on main.pdf (14 pages, 618594 bytes).
0 errors · 0 undefined citations · 0 undefined references · 6 overfull hboxes
```

So this list is not a guess about what LaTeX dislikes — it is the exact set of changes
that took the document from "hangs forever" to "14-page PDF".

## Priority tiers

- **Tier 0 (C1–C4)** — the build is *broken* without these. C1 makes `pdflatex` **hang**,
  not merely error, so nothing downstream can even be diagnosed until it is fixed.
- **Tier 1 (C5–C6)** — fatal `! LaTeX Error` / undefined-character stops.
- **Tier 2 (C7–C9)** — warnings and real typographic defects.
- **Tier 3 (L1–L5)** — consistency and polish. Apply if uncontroversial; skip if unsure.

---

# TIER 0 — build-blocking

## C1 — `\begin{tabular}` with no column specification makes `pdflatex` hang

**File:** `tex_files/03_replication.tex`, line 8.

This is the single worst fault in the document. `tabular` takes its column preamble as a
mandatory argument; with the argument missing, TeX swallows `\hline` and everything after
it as the preamble and never recovers. Under `-interaction=nonstopmode` the run does not
error out — it **spins indefinitely**. I killed it after 120 s with no PDF and no useful
log. Any earlier attempt to compile this document would have looked like a frozen
terminal, which is worth knowing.

The same block also has three unescaped underscores (`env_social`, `temp_hot`,
`temp_cold`). In text mode `_` is the subscript operator and each one raises
`! Missing $ inserted.`

**FIND:**
```latex
\begin{tabular}
\hline
\textbf{Concept}& \textbf{Positive Categories} & \textbf{Negative Categories} & \textbf{N/class} \\
\hline
food &  food, taste & weather, seasons, tv, music, glass, rain, outside & 300 \\
\hline
valence &  happy, excited, love, curious & scared, fear, bored, tired & 100 \\
\hline
env_social &  water, filter, temp_hot, temp_cold, algae & greeting, bye, friends, visitors, lonely & 300 \\
\end{tabular}
```

**REPLACE:**
```latex
\begin{tabular}{@{}l p{4.2cm} p{4.6cm} c@{}}
\hline
\textbf{Concept}& \textbf{Positive Categories} & \textbf{Negative Categories} & \textbf{N/class} \\
\hline
food &  food, taste & weather, seasons, tv, music, glass, rain, outside & 300 \\
\hline
valence &  happy, excited, love, curious & scared, fear, bored, tired & 100 \\
\hline
env\_social &  water, filter, temp\_hot, temp\_cold, algae & greeting, bye, friends, visitors, lonely & 300 \\
\end{tabular}
```

The `p{}` widths matter: with plain `l` columns the category lists run past the right
margin (the text block is 5.5 in under `margin=1.5in`). These two widths were checked and
produce a 1.1 pt overfull box, which is invisible.

## C2 — `table` environment with `\toprule` but no `tabular` at all

**File:** `tex_files/03_replication.tex`, lines 218–229 (the *Monitoring AUROC across
extraction methods* table).

There is a `\begin{table}` and a `\end{tabular}`, but no `\begin{tabular}`. The `&` and
`\\` therefore appear in ordinary paragraph text (`! Misplaced alignment tab character &`)
and the stray `\end{tabular}` raises
`! LaTeX Error: \begin{document} ended by \end{tabular}`.

It also uses `\makecell`, which needs the `makecell` package — see C4 for why the
replacement below deliberately does **not** use it.

**FIND:**
```latex
\begin{table}[htbp]
  \centering
  \caption{Monitoring AUROC (test split, aggregate probe) across extraction methods.}
    \toprule
    Concept & \makecell{RFM\\3 eigvecs (18 feat)} & \makecell{RFM\\top eigvec (6 feat)} & \makecell{DiffMeans\\(6 feat)} & \makecell{LogReg\\(6 feat)} & \makecell{PCA\\(6 feat)} \\
    \midrule
```

**REPLACE:**
```latex
\begin{table}[htbp]
  \centering
  \caption{Monitoring AUROC (test split, aggregate probe) across extraction methods.}
  \label{tab:baselines_auroc}
  \small
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    \textbf{Concept} & \shortstack{\textbf{RFM}\\3 eigvecs (18 feat)} & \shortstack{\textbf{RFM}\\top eigvec (6 feat)} & \shortstack{\textbf{DiffMeans}\\(6 feat)} & \shortstack{\textbf{LogReg}\\(6 feat)} & \shortstack{\textbf{PCA}\\(6 feat)} \\
    \midrule
```

Leave the body rows and the closing `\bottomrule` / `\end{tabular}` / `\end{table}`
exactly as they are — they are already correct, they were simply orphaned. The `\small`
is needed: six columns of that header text overflow the margin at 12 pt.

## C3 — `booktabs` is used in four tables but never loaded

**File:** `main.tex`, preamble.

`\toprule`, `\midrule`, `\bottomrule` and `\addlinespace` appear in
`03_replication.tex` at lines 122–136, 165–171, 204–212 and 221–227, but `booktabs` is
not in the preamble. Each is `! Undefined control sequence`.

**FIND:**
```latex
\usepackage{subcaption}
```

**REPLACE:**
```latex
\usepackage{subcaption}
\usepackage{booktabs}
```

(`booktabs` is present in TeX Live basic — confirmed via `kpsewhich booktabs.sty` — so
this compiles locally as well as on Overleaf.)

## C4 — `\makecell` without `makecell`, and why we avoid the package

C2's replacement uses `\shortstack`, which is part of the LaTeX kernel and needs no
package. This is deliberate: `makecell.sty` is **not** in the local TeX Live 2026 *basic*
installation on this machine — I hit
`! LaTeX Error: File 'makecell.sty' not found.` and had to back it out. It *is* available
on Overleaf, so `\usepackage{makecell}` would also work there, but `\shortstack` works in
both places and costs nothing.

If Maxwell later wants makecell's nicer vertical centring, the local fix is
`tlmgr install makecell`. **Do not run that as part of this pass** — modifying his TeX
installation is his call, not ours.

---

# TIER 1 — fatal errors

## C5 — unbalanced parenthesis

**File:** `tex_files/03_replication.tex`, line 1.

Not a LaTeX error, but the sentence opens a parenthesis it never closes, and the reader
lands in the next subsection still inside it.

**FIND:**
```latex
verified to the order of $7\times 10^{-7}$.
```

**REPLACE:**
```latex
verified to the order of $7\times 10^{-7}$).
```

## C6 — Unicode mathematical characters in text mode

**File:** `tex_files/03_replication.tex`.

With `\usepackage[utf8]{inputenc}` and the default OT1 font encoding, mathematical Unicode
glyphs are **not** mapped to any control sequence. Each produces
`! Package inputenc Error: Unicode character ... not set up for use with LaTeX.`

Apply these substitutions. Each `FIND` string is unique in the file — verify that before
replacing, and replace only the first occurrence if your tooling does otherwise.

| Line | FIND | REPLACE |
|---|---|---|
| 151 | `14\% → 86\%` | `14\% $\to$ 86\%` |
| 152 | `57\% → 29\%` | `57\% $\to$ 29\%` |
| 153 | `r = ±0.5` | `$r = \pm 0.5$` |
| 185 | `top-p=3` | `top-$p=3$` |
| 185 | `R^{18} (6 blocks × 3 components)` | `$\mathbb{R}^{18}$ ($6$ blocks $\times$ $3$ components)` |
| 185 | `raw 6 × 384 activations` | `raw $6 \times 384$ activations` |
| 201 | `validation $|\rho|$` | *(already correct — do not touch)* |
| 231 | `0.90 on env_social` | `0.90 on \verb\|env_social\|` |
| 231 | `at r = −2` | `at $r = -2$` |
| 231 | `"success"` | ` ``success'' ` |
| 231 | `val \|ρ\|` *(both occurrences)* | `val $\|\rho\|$` |
| 237 | `"≫ 0.5" bar` | ` ``$\gg 0.5$'' bar ` |
| 237 | `validation \|ρ\| as high as 0.978` | `validation $\|\rho\|$ as high as 0.978` |
| 239 | `λ₁/Σλ is 0.886` | `$\lambda_1/\sum\lambda$ is 0.886` |
| 241 | `+ε/−ε` | `$+\varepsilon$/$-\varepsilon$` |
| 241 | `(env_social collapses 7 for 7 at r ≥ 1)` | `(\verb\|env_social\| collapses 7 for 7 at $r \geq 1$)` |
| 241 | `at every r ≠ 0 tested` | `at every $r \neq 0$ tested` |
| 241 | `absolute ε values don't transfer` | `absolute $\varepsilon$ values don't transfer` |

Note on line 231: the two `−` characters there are U+2212 MINUS SIGN, **not** ASCII
hyphen-minus. Searching for `r = -2` with a plain hyphen will not match. Copy the string
from the file.

### Characters that are FINE — do not "fix" these

A naive sweep for non-ASCII will mangle the document. The following are all correctly
handled by `inputenc utf8` and were verified to typeset properly in the 14-page build:

- em dash `—` and en dash `–` (lines 151, 153, 181, 193, 201, 215, 231, 237, 239, 241, 243)
- curly quotes `“ ” ’` (lines 118, 126–135, 181)
- `§` (`1_fundamentals.tex`, line 27)
- `ã é í ç à` in `main.tex` lines 56 and 62 (`Valadão`, `Matemática`, `Estatística`, `São Paulo`)

Leave every one of them alone.

---

# TIER 2 — warnings and real defects

## C7 — `2_guppy.tex` declares three columns for a two-column table

**File:** `tex_files/2_guppy.tex`, line 3.

`\begin{tabular}{c c c}` with rows of the form `Property & Value \\`. LaTeX does not error
— it silently reserves a third, empty column and the table renders visibly off-centre.
`l l` also reads better than `c c` for a property/value list.

**FIND:**
```latex
\begin{tabular}{c c c}
```

**REPLACE:**
```latex
\begin{tabular}{l l}
```

## C8 — `todonotes` warning, and five packages loaded but never used

**File:** `main.tex`, preamble.

The build emits:

```
Package todonotes Warning: The length marginparwidth is less than 2cm and will
most likely cause issues with the appearance of inserted todonotes.
```

`todonotes` is loaded but no `\todo` or `\missingfigure` appears anywhere in the document —
I grepped all five `.tex` files. The same is true of `multicol` (no `multicols`),
`wrapfig` (no `wrapfigure`), `algorithm` + `algpseudocode` (no `algorithm` environment),
`subcaption` (no `subfigure`) and `xcolor` (no `\textcolor`; `hyperref` pulls in its own
colour support).

Removing `todonotes` is the correct fix — it removes the warning at its source rather than
padding `\marginparwidth` for a feature that is not in use.

**FIND:**
```latex
\usepackage[colorinlistoftodos]{todonotes}
```

**REPLACE:** *(delete the line)*

The other four are harmless; removing them only shortens the log. If you prefer to keep
`todonotes` because Maxwell may want margin notes while drafting, then instead insert
`\setlength{\marginparwidth}{2cm}` on the line **before** it. Do one or the other, not
both.

## C9 — three overfull display equations and lines

Six overfull `\hbox` warnings survive. Three are ≤ 7 pt and invisible; ignore them
(`03_replication.tex` lines 8–18, 23–27, 223–232). The other three are visible and worth
fixing:

**C9a — `1_fundamentals.tex` line 24, 57.6 pt over.** The KRR display puts the definition
of $\alpha_t$ and the definition of $f_t$ on one line. Break it:

**FIND:**
```latex
    \alpha_t = \left(K_{M_t}(Z,Z) + \lambda I\right)^{-1} y \;\in\; \mathbb{R}^{n},
    \qquad
    f_t(z) = \alpha_t^{\top} K_{M_t}(Z,z) = \sum_{i=1}^{n} \alpha_{t,i}\, K_{M_t}\!\left(a^{(i)}, z\right).
  \]
```

**REPLACE:**
```latex
    \alpha_t = \left(K_{M_t}(Z,Z) + \lambda I\right)^{-1} y \;\in\; \mathbb{R}^{n},
  \]
  \[
    f_t(z) = \alpha_t^{\top} K_{M_t}(Z,z) = \sum_{i=1}^{n} \alpha_{t,i}\, K_{M_t}\!\left(a^{(i)}, z\right).
  \]
```

**C9b — `1_fundamentals.tex` line 38, 14.8 pt over.** The AGOP display carries $M_{t+1}$,
$g^{(i)}$ and $\bar g$ on one line. Move the two definitions to a second line by replacing
the `\qquad` before `g^{(i)}` with `\]` + `\[`, or simply change that `\qquad` to
`\\[4pt]` inside a `gather*`. The minimal edit:

**FIND:**
```latex
      \left(g^{(i)} - \bar{g}\right)\left(g^{(i)} - \bar{g}\right)^{\top},
    \qquad
    g^{(i)} = \nabla_z f_t\!\left(a^{(i)}\right),
```

**REPLACE:**
```latex
      \left(g^{(i)} - \bar{g}\right)\left(g^{(i)} - \bar{g}\right)^{\top},
  \]
  \[
    g^{(i)} = \nabla_z f_t\!\left(a^{(i)}\right),
```

**C9c — `2_guppy.tex` lines 18–19, 9.6 pt over.** Caused by `residual-strem`, which TeX
cannot hyphenate because it is not a word. This is a **typo for `residual-stream`**;
fixing the spelling also fixes the overfull box.

**FIND:** `a residual-strem tensor`
**REPLACE:** `a residual-stream tensor`

---

# TIER 3 — consistency (apply only if uncontroversial)

**L1 — mixed table rule styles.** Three tables use `\hline` (`2_guppy.tex` line 3;
`03_replication.tex` lines 8 and 67) and three use `booktabs` rules
(`03_replication.tex` lines 121, 164, 221). Since `booktabs` is being loaded anyway (C3),
converting the first three is a one-line-per-rule change: leading `\hline` → `\toprule`,
interior → `\midrule`, final → `\bottomrule`, and **delete** the `\hline` between every
data row — booktabs style has no interior rules between data rows, and the current
`\hline`-per-row look is what makes those three tables read as busier than the others.

**L2 — two tables float free of any `table` environment.** `2_guppy.tex` line 3 and
`03_replication.tex` line 8 are bare `tabular`s: no caption, no label, no centring. Every
other table in the document is a captioned float. Wrapping them
(`\begin{table}[htbp]\centering\caption{...}\label{...}` … `\end{table}`) would make the
document uniform. Suggested captions: *"GuppyLM architecture and training data."* and
*"The three binary concepts, defined as category contrasts."*

**L3 — no float is ever cross-referenced.** Four tables carry `\label`s
(`tab:rfm-val-pearson`, `tab:steering_samples`, `tab:steering_success`,
`tab:baselines_val_rho`) and C2 adds a fifth, but there is not one `\ref` in the document;
the prose says "Below are…" and "the table below". Since `htbp` floats can migrate across
a page break, "below" can become false at any time. Replacing those phrases with
`Table~\ref{...}` would make the references robust. **The six figures have no labels at
all** — add `\label{fig:...}` to each and reference them the same way.

**L4 — `imgs/07_baselines_val_rho.png` exists but is never included.** The Baselines
subsection is the only results section with no figure. Either add it after line 213 or
delete the file. **Ask Maxwell — do not decide this.**

**L5 — `\bibliographystyle` is genuinely not needed.** For the record, since it looks like
an omission: `\usepackage[alf]{abntex2cite}` sets the bibliography style itself. I tested
compiling with and without an explicit `\bibliographystyle{abntex2-alf}` and both produce
an identical `main.bbl` with both citations resolved. **Do not add one.**

---

# Verification protocol

Run this from `docs/latex/` after applying the edits. Note the timeout wrapper — without
it, a surviving C1-class fault will hang your shell rather than fail.

```bash
# 1. No structural faults remain
grep -n 'begin{tabular}$' tex_files/*.tex            # must be empty (C1)
grep -c 'begin{tabular}' tex_files/03_replication.tex # must equal 6
grep -c 'end{tabular}'   tex_files/03_replication.tex # must equal 6
grep -n 'makecell' tex_files/*.tex main.tex          # must be empty (C4)
grep -n 'booktabs' main.tex                          # must match (C3)

# 2. No unescaped underscores in text mode
grep -n '[^\\|]_[a-z]' tex_files/*.tex               # only \verb|...| hits allowed

# 3. Only safe non-ASCII survives — dashes, quotes, §, Portuguese accents
grep -oP '[^\x00-\x7F]' tex_files/*.tex main.tex | sort -u

# 4. Full build, four passes, with a hang guard
rm -f main.aux main.bbl main.blg main.log main.out main.pdf
for i in 1 2; do
  ( pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1 & \
    P=$!; for s in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done; \
    kill -9 $P 2>/dev/null && echo "HANG — a tabular preamble is still missing" )
  [ $i = 1 ] && bibtex main
done
pdflatex -interaction=nonstopmode -file-line-error main.tex > final.log 2>&1

# 5. Assert the end state
grep -E '^(\./|!)' final.log            # must be empty
grep -c 'Citation.*undefined' final.log # must be 0
grep -c 'Reference.*undefined' final.log# must be 0
grep -c 'Overfull' final.log            # must be <= 3 after C9
grep -o 'Output written.*' final.log    # must report a PDF of ~14 pages
```

**Expected end state:** 0 errors, 0 undefined citations, 0 undefined references, ≤ 3
overfull boxes (all under 7 pt), `main.pdf` at roughly 14 pages.

If the build still hangs, the cause is almost always another `tabular`, `array` or
`array`-like environment missing its `{...}` preamble argument — check `grep -n
'begin{tabular}$'` first, before reading any log.

---

# Hand-off to pass 3 — DO NOT FIX HERE

These are wording and claim-level problems. They compile perfectly well and are pass 3's
subject. Changing them now would collide with that pass.

1. `1_fundamentals.tex:57` — "the RFM can surface concept structure that a linear
   separator cannot" is contradicted by this document's own Table
   `tab:baselines_val_rho`, where LogReg beats RFM on all three concepts.
2. `03_replication.tex:81` — "does not report uniform success on **any** concept" inverts
   the intended meaning (should be "on **every** concept").
3. `03_replication.tex:140` — "which makes sense for the small context the model have"
   asserts an explanation the evidence does not support.
4. `03_replication.tex:215` — "The separation we expected does in fact appear, except for
   the PCA" states the reverse of the finding: the separation appears **only** against
   PCA.
5. `2_guppy.tex:18` — "the only concepts GuppyLM can represent" is stronger than the
   README's "can *plausibly* represent".
6. `03_replication.tex:243` — the final sentence of the document ends without a full stop.
7. Throughout `03_replication.tex` — `r = +0.25`, `r = 0`, `T = 1` and similar appear as
   bare text in prose while identical quantities appear in math mode elsewhere. Uniform
   math-mode treatment is a legitimate improvement, but it touches sentences pass 3 will
   rewrite, so it belongs there.
