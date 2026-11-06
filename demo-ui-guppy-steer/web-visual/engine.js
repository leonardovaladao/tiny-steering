/* GuppyLM in the browser, with additive steering (Seam B in SPEC.md).

DOM-free ES module, no dependencies; runs in Node ≥ 18 and in the browser. It mirrors, op for op,
guppylm/guppylm/model.py and demo-ui-guppy-steer/steer_ref.py:

    x_t     = E[id_t] + Pos[t]
    x      ← x + Attn(LN1 x)          (causal, 6 heads of 64, softmax(q kᵀ / 8) v)
    x      ← x + Down(ReLU(Up(LN2 x)))
    h_{ℓ,t} = x after block ℓ,  then   h_{ℓ,t} ← h_{ℓ,t} + 1[t ∈ P] · ε_ℓ · v_ℓ ,  ε_ℓ = r · med_ℓ
    z_t     = Eᵀ LN_f(h_{5,t})          (lm_head tied to the token embedding)

    P_all = every t,   P_prompt = {t < T0 − 1},   P_gen = {t ≥ T0 − 1},   T0 = |ChatML prompt|.

The steered state enters the residual stream, so it is in the K/V that later positions read. With
the KV cache each position's states are computed once, with that position's own mask value; since
attention is causal and the mask depends only on absolute t, this equals the Python reference,
which recomputes the whole sequence at every step.

Numerics: tensors and activations are stored as float32 (Float32Array), dot products and norms are
accumulated in float64. The steering term is rounded as torch does it, fp32(h + fp32(ε · v)).

Also here: the byte-level BPE tokenizer of `tokenizers`, a seeded sampler, the clean re-read
p_t = ⟨a_{5,t} − μ_5, v_5⟩ on the unsteered model, the dose ρ and the TF-IDF + logistic judge g_c.
*/

const f32 = Math.fround;

let CFG = null;          // weights.json config
let W = null;            // tensor name → Float32Array
let SHARED = null;       // shared.json
const BUNDLES = {};      // concept → parsed concept_<c>.json
const VEC = {};          // "c|m" → Float32Array[6], the steering vectors as torch holds them (fp32)
const JUDGE = {};        // concept → { index: Map(feature → column), idf, coef, intercept }
let TOK = null;          // tokenizer state

// ─────────────────────────────────────────────────────────────────────────────
// Loading
// ─────────────────────────────────────────────────────────────────────────────

/** Fetches weights, tokenizer, shared.json and every concept bundle. onProgress(f), f ∈ [0, 1],
 *  follows the bytes of the three weight files (the JSON files are small). */
export async function load(base = "./data/", { fetch = globalThis.fetch, onProgress } = {}) {
  const get = async (name) => {
    const res = await fetch(base + name);
    if (!res.ok) throw new Error(`fetch ${base + name}: ${res.status}`);
    return res;
  };
  const json = async (name) => (await get(name)).json();

  const manifest = await json("weights.json");
  CFG = manifest.config;

  // expected size of each weight file, from the manifest, for the progress fraction
  const size = {};
  for (const t of Object.values(manifest.tensors)) {
    const end = t.offset + 4 * t.shape.reduce((a, b) => a * b, 1);
    size[t.file] = Math.max(size[t.file] || 0, end);
  }
  const total = Object.values(size).reduce((a, b) => a + b, 0);
  const got = {};
  const report = () => onProgress && onProgress(Math.min(1, Object.values(got).reduce((a, b) => a + b, 0) / total));

  const readBin = async (file) => {
    const res = await get(file);
    if (!res.body || !res.body.getReader) {
      const buf = await res.arrayBuffer();
      got[file] = buf.byteLength; report();
      return buf;
    }
    const reader = res.body.getReader();
    const parts = [];
    let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); n += value.byteLength;
      got[file] = n; report();
    }
    const out = new Uint8Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.byteLength; }
    return out.buffer;
  };

  const [bufs, tok, shared] = await Promise.all([
    Promise.all(manifest.files.map(readBin)),
    json("tokenizer.json"),
    json("shared.json"),
  ]);
  const byFile = Object.fromEntries(manifest.files.map((f, i) => [f, bufs[i]]));
  W = {};
  for (const [name, t] of Object.entries(manifest.tensors))
    W[name] = new Float32Array(byFile[t.file], t.offset, t.shape.reduce((a, b) => a * b, 1));
  for (const [name, src] of Object.entries(manifest.tied || {})) W[name] = W[src];

  SHARED = shared;
  VEC["*|random"] = shared.u.map((row) => Float32Array.from(row));
  initTokenizer(tok);

  const bundles = await Promise.all(shared.concepts.map((c) => json(`concept_${c}.json`)));
  shared.concepts.forEach((c, i) => {
    const B = bundles[i];
    BUNDLES[c] = B;
    VEC[`${c}|rfm`] = B.blocks.map((b) => Float32Array.from(b.v_rfm));
    VEC[`${c}|dm`] = B.blocks.map((b) => Float32Array.from(b.v_dm));
    const j = B.judge;
    JUDGE[c] = { index: new Map(j.vocab.map((t, k) => [t, k])), idf: j.idf, coef: j.coef, intercept: j.intercept };
  });
  onProgress && onProgress(1);
}

export function shared() { return SHARED; }
export function bundle(concept) { return BUNDLES[concept]; }

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer: byte-level BPE as in HF `tokenizers`
// ─────────────────────────────────────────────────────────────────────────────
//
// encode: split out the added special tokens, GPT-2 regex pre-tokenization, bytes → printable
// chars (GPT-2 bytes_to_unicode), BPE by merge rank. The vocab has no unk token and lacks most
// characters, so a char not in the vocab is dropped before merging, as `tokenizers` does.
// decode: skip special ids and ids not in the vocab, chars → bytes, lossy UTF-8.

// Python/oniguruma \s (Unicode White_Space plus U+0085); JS \s differs on U+0085 and U+FEFF.
const WS = "\\t\\n\\v\\f\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PRETOK = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");

function initTokenizer(tj) {
  // GPT-2 bytes_to_unicode
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  for (let b = 0, n = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const byteChar = new Array(256), charByte = new Map();
  bs.forEach((b, i) => { byteChar[b] = String.fromCodePoint(cs[i]); charByte.set(byteChar[b], b); });

  const vocab = new Map(Object.entries(tj.model.vocab));
  const inv = [];
  for (const [t, id] of vocab) inv[id] = t;
  const ranks = new Map();
  tj.model.merges.forEach((m, i) => {
    const [a, b] = Array.isArray(m) ? m : m.split(" ");
    ranks.set(a + " " + b, i);
  });
  const added = tj.added_tokens.slice().sort((a, b) => b.content.length - a.content.length);
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  TOK = {
    byteChar, charByte, vocab, inv, ranks, cache: new Map(),
    added: new Map(added.map((t) => [t.content, t.id])),
    special: new Set(tj.added_tokens.filter((t) => t.special).map((t) => t.id)),
    addedRe: new RegExp(added.map((t) => esc(t.content)).join("|"), "g"),
    enc: new TextEncoder(), dec: new TextDecoder("utf-8", { ignoreBOM: true }),
  };
}

function bpe(word) {
  const hit = TOK.cache.get(word);
  if (hit) return hit;
  let sym = [...word].filter((ch) => TOK.vocab.has(ch));      // unknown chars vanish (no unk)
  for (;;) {
    let best = Infinity, at = -1;
    for (let i = 0; i + 1 < sym.length; i++) {
      const r = TOK.ranks.get(sym[i] + " " + sym[i + 1]);
      if (r !== undefined && r < best) { best = r; at = i; }
    }
    if (at < 0) break;
    sym = [...sym.slice(0, at), sym[at] + sym[at + 1], ...sym.slice(at + 2)];
  }
  const ids = sym.map((s) => TOK.vocab.get(s));
  TOK.cache.set(word, ids);
  return ids;
}

function encodeOrdinary(text, out) {
  for (const m of text.matchAll(PRETOK)) {
    let w = "";
    for (const b of TOK.enc.encode(m[0])) w += TOK.byteChar[b];
    for (const id of bpe(w)) out.push(id);
  }
}

/** Same ids as `tokenizers` encode(text).ids (no ChatML added). */
export function tokenize(text) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(TOK.addedRe)) {
    encodeOrdinary(text.slice(last, m.index), out);
    out.push(TOK.added.get(m[0]));
    last = m.index + m[0].length;
  }
  encodeOrdinary(text.slice(last), out);
  return out;
}

/** Same string as `tokenizers` decode(ids) (skip_special_tokens = True). */
export function decode(ids) {
  const bytes = [];
  for (const id of ids) {
    if (TOK.special.has(id)) continue;
    const t = TOK.inv[id];
    if (t === undefined) continue;                              // logits rows ≥ |vocab|
    for (const ch of t) {
      const b = TOK.charByte.get(ch);
      if (b !== undefined) bytes.push(b);
      else for (const x of TOK.enc.encode(ch)) bytes.push(x);
    }
  }
  return TOK.dec.decode(new Uint8Array(bytes));
}

export function chatml(prompt) {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}
export function promptIds(prompt) { return tokenize(chatml(prompt)); }
export function tokenStrings(ids) { return ids.map((id) => decode([id])); }

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass with KV cache
// ─────────────────────────────────────────────────────────────────────────────
//
// forward(st, ids, t0, steer) runs the rows t0 … t0+n−1 through all blocks, layer by layer,
// writing their K/V into the cache and attending to cache rows ≤ t. One row at a time is the
// incremental (cached) pass; all rows in one call from t0 = 0 is the full causal forward.

function newState() {
  const T = CFG.max_seq_len, d = CFG.d_model, L = CFG.n_layers, f = CFG.ffn_hidden;
  return {
    K: Array.from({ length: L }, () => new Float32Array(T * d)),
    V: Array.from({ length: L }, () => new Float32Array(T * d)),
    X: new Float32Array(T * d), H: new Float32Array(T * d), O: new Float32Array(T * d),
    Q: new Float32Array(T * d), U: new Float32Array(T * f), S: new Float64Array(T),
    n: 0,
  };
}

/** out[oo + i] = Σ_j W[i, j] x[xo + j] + b[i],  W [nOut, nIn] row-major (nn.Linear). */
function linear(Wt, b, nOut, nIn, x, xo, out, oo) {
  for (let i = 0; i < nOut; i++) {
    const w = i * nIn;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let j = 0; j < nIn; j += 4) {
      s0 += Wt[w + j] * x[xo + j];
      s1 += Wt[w + j + 1] * x[xo + j + 1];
      s2 += Wt[w + j + 2] * x[xo + j + 2];
      s3 += Wt[w + j + 3] * x[xo + j + 3];
    }
    out[oo + i] = (s0 + s1) + (s2 + s3) + (b ? b[i] : 0);
  }
}

function layerNorm(x, xo, g, b, out, oo, d) {
  let m = 0;
  for (let j = 0; j < d; j++) m += x[xo + j];
  m /= d;
  let v = 0;
  for (let j = 0; j < d; j++) { const c = x[xo + j] - m; v += c * c; }
  const inv = 1 / Math.sqrt(v / d + CFG.layer_norm_eps);
  for (let j = 0; j < d; j++) out[oo + j] = (x[xo + j] - m) * inv * g[j] + b[j];
}

function forward(st, ids, t0, steer) {
  const d = CFG.d_model, nh = CFG.n_heads, hd = d / nh, f = CFG.ffn_hidden, n = ids.length;
  const { X, H, O, Q, U, S } = st;
  const scale = 1 / Math.sqrt(hd);
  const qkvRow = new Float32Array(3 * d);
  const tmp = new Float32Array(d);
  const E = W["tok_emb.weight"], P = W["pos_emb.weight"];

  for (let r = 0; r < n; r++)
    for (let j = 0; j < d; j++) X[r * d + j] = E[ids[r] * d + j] + P[(t0 + r) * d + j];

  for (let l = 0; l < CFG.n_layers; l++) {
    const p = `blocks.${l}.`, K = st.K[l], V = st.V[l];
    // ── attention ──
    for (let r = 0; r < n; r++) {
      const t = t0 + r;
      layerNorm(X, r * d, W[p + "norm1.weight"], W[p + "norm1.bias"], H, r * d, d);
      linear(W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"], 3 * d, d, H, r * d, qkvRow, 0);
      Q.set(qkvRow.subarray(0, d), r * d);
      K.set(qkvRow.subarray(d, 2 * d), t * d);
      V.set(qkvRow.subarray(2 * d), t * d);
    }
    for (let r = 0; r < n; r++) {
      const t = t0 + r;
      for (let h = 0; h < nh; h++) {
        const qo = r * d + h * hd;
        let mx = -Infinity;
        for (let s = 0; s <= t; s++) {
          const ko = s * d + h * hd;
          let a = 0;
          for (let j = 0; j < hd; j++) a += Q[qo + j] * K[ko + j];
          a = f32(f32(a) * scale);
          S[s] = a;
          if (a > mx) mx = a;
        }
        let z = 0;
        for (let s = 0; s <= t; s++) { S[s] = Math.exp(S[s] - mx); z += S[s]; }
        for (let j = 0; j < hd; j++) {
          let a = 0;
          for (let s = 0; s <= t; s++) a += S[s] * V[s * d + h * hd + j];
          O[qo + j] = a / z;
        }
      }
    }
    for (let r = 0; r < n; r++) {
      linear(W[p + "attn.out.weight"], W[p + "attn.out.bias"], d, d, O, r * d, tmp, 0);
      for (let j = 0; j < d; j++) X[r * d + j] += tmp[j];
    }
    // ── feed-forward ──
    for (let r = 0; r < n; r++) {
      layerNorm(X, r * d, W[p + "norm2.weight"], W[p + "norm2.bias"], H, r * d, d);
      linear(W[p + "ffn.up.weight"], W[p + "ffn.up.bias"], f, d, H, r * d, U, r * f);
      for (let j = 0; j < f; j++) if (U[r * f + j] < 0) U[r * f + j] = 0;
      linear(W[p + "ffn.down.weight"], W[p + "ffn.down.bias"], d, f, U, r * f, tmp, 0);
      for (let j = 0; j < d; j++) X[r * d + j] += tmp[j];
    }
    // ── steering: h_{ℓ,t} ← fp32(h_{ℓ,t} + fp32(ε_ℓ v_ℓ)) for t ∈ P ──
    if (steer && steer.eps[l] !== 0) {
      const e = steer.eps[l], v = steer.vec[l];
      for (let r = 0; r < n; r++)
        if (steer.inP(t0 + r)) for (let j = 0; j < d; j++) X[r * d + j] += f32(e * v[j]);
    }
    if (st.capture === l) st.onCapture(X, n, t0);
  }
  st.n = t0 + n;
}

/** z = Eᵀ LN_f(x_row), length vocab_size. */
function head(st, r, out) {
  const d = CFG.d_model;
  layerNorm(st.X, r * d, W["norm.weight"], W["norm.bias"], st.H, r * d, d);
  linear(W["lm_head.weight"], null, CFG.vocab_size, d, st.H, r * d, out, 0);
  return out;
}

/** Unsteered logits at every position, Float32Array [T · vocab] row-major. */
export function logits(ids, { useCache = true } = {}) {
  const T = ids.length, Vn = CFG.vocab_size;
  if (T > CFG.max_seq_len) throw new Error(`${T} tokens exceed the ${CFG.max_seq_len}-token context`);
  const st = newState(), out = new Float32Array(T * Vn);
  if (useCache) {
    for (let t = 0; t < T; t++) {
      forward(st, [ids[t]], t, null);
      head(st, 0, out.subarray(t * Vn, (t + 1) * Vn));
    }
  } else {
    forward(st, ids, 0, null);
    for (let t = 0; t < T; t++) head(st, t, out.subarray(t * Vn, (t + 1) * Vn));
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Steering cell: vectors, ε, dose
// ─────────────────────────────────────────────────────────────────────────────

function blockIndices(S) { return S === "all" ? [0, 1, 2, 3, 4, 5] : [Number(S)]; }

function vectors(concept, source) {
  if (source === "random") return VEC["*|random"];
  const v = VEC[`${concept}|${source}`];
  if (!v) throw new Error(`no vectors for concept ${concept}, source ${source}`);
  return v;
}

/** ε_ℓ = r · med_ℓ for ℓ ∈ S, 0 elsewhere. Rounded to fp32 as numpy computes r * med (float32). */
export function epsilon(blocks, r) {
  const eps = new Array(CFG ? CFG.n_layers : 6).fill(0);
  for (const l of blockIndices(blocks)) eps[l] = f32(f32(r) * SHARED.med[l]);
  return eps;
}

/** ρ = ‖Σ_{ℓ∈S} r med_ℓ v_ℓ‖₂ / med_5 (unit v_ℓ in float64, as in steer_ref.dose). */
export function dose(concept, source, blocks, r) {
  const eps = epsilon(blocks, r), med = SHARED.med, d = SHARED.u[0].length;
  const s = new Float64Array(d);
  for (const l of blockIndices(blocks)) {
    const v = source === "random" ? SHARED.u[l] : BUNDLES[concept].blocks[l][`v_${source}`];
    for (let j = 0; j < d; j++) s[j] += eps[l] * v[j];
  }
  let n = 0;
  for (let j = 0; j < d; j++) n += s[j] * s[j];
  return Math.sqrt(n) / med[med.length - 1];
}

function steering(concept, source, blocks, positions, r, T0) {
  const inP = { all: () => true, prompt: (t) => t < T0 - 1, gen: (t) => t >= T0 - 1 }[positions];
  if (!inP) throw new Error(`positions must be all | prompt | gen, got ${positions}`);
  return { eps: epsilon(blocks, r), vec: vectors(concept, source), inP };
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation
// ─────────────────────────────────────────────────────────────────────────────

/** sfc32 seeded through splitmix32; returns uniforms in [0, 1). */
function prng(seed) {
  let s = seed >>> 0;
  const mix = () => {
    s = (s + 0x9e3779b9) | 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = mix(), b = mix(), c = mix(), d = mix();
  return () => {
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

function argmax(z) {
  let k = 0;
  for (let i = 1; i < z.length; i++) if (z[i] > z[k]) k = i;   // first max on ties, like torch
  return k;
}

/** logits / T, keep z ≥ k-th largest (torch: z < kth → −inf), softmax, one draw. */
function sample(z, temperature, topK, rand) {
  const n = z.length, y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = f32(z[i] / temperature);
  let kth = -Infinity;
  if (topK > 0) kth = Float64Array.from(y).sort()[n - Math.min(topK, n)];
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (y[i] >= kth && y[i] > mx) mx = y[i];
  let s = 0;
  for (let i = 0; i < n; i++) { y[i] = y[i] >= kth ? Math.exp(y[i] - mx) : 0; s += y[i]; }
  let u = rand() * s, last = 0;
  for (let i = 0; i < n; i++) {
    if (y[i] === 0) continue;
    last = i;
    if ((u -= y[i]) < 0) return i;
  }
  return last;
}

// Python str.strip() whitespace (str.isspace), which differs from JS trim on U+001C–1F, U+0085, U+FEFF.
const PYWS = "[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const STRIP = new RegExp(`^${PYWS}+|${PYWS}+$`, "gu");

function cleanText(ids) {
  let s = decode(ids);
  for (const ctrl of ["<|im_end|>", "<|im_start|>"]) if (s.includes(ctrl)) s = s.split(ctrl)[0];
  return s.replace(STRIP, "");
}

// Yield to the event loop between tokens so that concurrent calls interleave.
const tick = typeof setImmediate === "function"
  ? () => new Promise((res) => setImmediate(res))
  : (() => {
      const ch = new MessageChannel(), q = [];
      ch.port1.onmessage = () => q.shift()();
      return () => new Promise((res) => { q.push(res); ch.port2.postMessage(0); });
    })();

/** Steered generation, as steer_ref.generate. All state (KV cache, PRNG) is local to the call. */
export async function generate(prompt, {
  concept, source = "rfm", blocks = "all", positions = "all", r = 0, seed = 42, greedy = false,
  maxNewTokens = 64, temperature = 0.7, topK = 50, onToken, signal,
} = {}) {
  const pids = promptIds(prompt), T0 = pids.length;
  if (T0 + maxNewTokens > CFG.max_seq_len)
    throw new Error(`prompt (${T0} tokens) + ${maxNewTokens} new tokens exceeds the ` +
                    `${CFG.max_seq_len}-token context; positions would shift`);
  const steer = r !== 0 ? steering(concept, source, blocks, positions, r, T0) : null;
  const st = newState(), z = new Float32Array(CFG.vocab_size), rand = prng(seed), ids = [];

  forward(st, pids, 0, steer);
  let row = T0 - 1;
  for (let i = 0; i < maxNewTokens; i++) {
    if (signal && signal.aborted) break;
    head(st, row, z);
    const next = greedy ? argmax(z) : sample(z, temperature, topK, rand);
    ids.push(next);
    if (onToken) onToken({ id: next, ids: ids.slice(), text: cleanText(ids) });
    if (next === CFG.eos_id) break;
    await tick();
    if (i + 1 < maxNewTokens) { forward(st, [next], T0 + i, steer); row = 0; }
  }
  return { text: cleanText(ids), ids, T0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Monitoring: clean re-read and judge
// ─────────────────────────────────────────────────────────────────────────────

/** p_t = ⟨a_{5,t} − μ_5, v_5⟩ for the generated positions t = T0 … , on the UNSTEERED model.
 *  v_5 = v_rfm for source "random", so the strip always measures the concept. */
export function cleanReread(concept, source, prompt, ids) {
  const pids = promptIds(prompt), T0 = pids.length;
  const all = pids.concat(ids).slice(0, CFG.max_seq_len);
  const L = CFG.n_layers, d = CFG.d_model, B = BUNDLES[concept].blocks[L - 1];
  const v = source === "dm" ? B.v_dm : B.v_rfm, mu = B.mu;
  const st = newState(), p = [];
  st.capture = L - 1;
  st.onCapture = (X) => {
    for (let t = T0; t < all.length; t++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += (X[t * d + j] - mu[j]) * v[j];
      p.push(s);
    }
  };
  forward(st, all, 0, null);
  return p;
}

/** g_c = 2 σ(intercept + coef · x) − 1, x = the sklearn TfidfVectorizer row of the text. */
export function judge(concept, text) {
  const J = JUDGE[concept];
  // sklearn: lowercase, (?u)\b\w+\b with Python's \w = [\p{L}\p{N}_], 1- and 2-grams
  const toks = text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  const tf = new Map();
  const add = (g) => { const k = J.index.get(g); if (k !== undefined) tf.set(k, (tf.get(k) || 0) + 1); };
  for (let i = 0; i < toks.length; i++) {
    add(toks[i]);
    if (i + 1 < toks.length) add(toks[i] + " " + toks[i + 1]);
  }
  let nrm = 0, dot = 0;
  for (const [k, n] of tf) {
    const x = (1 + Math.log(n)) * J.idf[k];
    nrm += x * x; dot += J.coef[k] * x;
  }
  const z = J.intercept + (nrm > 0 ? dot / Math.sqrt(nrm) : 0);
  return 2 / (1 + Math.exp(-z)) - 1;
}
