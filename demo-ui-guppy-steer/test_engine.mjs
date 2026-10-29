// Seam B gate: the JS engine (web/engine.js) reproduces the Python reference (fixtures.json).
//
// Run from the project root:   node demo-ui-guppy-steer/test_engine.mjs
//
// Checks
//   T1  tokenizer: same ids as `tokenizers` on every fixture string; decode equal
//   T2  logits on the 20 ChatML prompts: max |z_JS − z_PT| < 1e-3 (last position; every position
//       for 2 prompts); cached and uncached logits agree to < 1e-4
//   T3  64-token greedy generations: same ids and text, 20 unsteered + 20 steered (P = all, prompt, gen)
//   T4  judge: |g_JS − g_sklearn| < 1e-6 on every fixture text, every concept
//   T5  clean re-read within 1e-3; dose within 1e-6
//   T6  two concurrent sampled generate() calls give the same text as alone (isolation)
//   T7  timing (reported, not a gate)

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import * as E from "./web/engine.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const results = [];

function check(name, ok, detail = "") {
  results.push(ok);
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}` + (detail ? `  — ${detail}` : ""));
}

const f32 = (b64) => { const b = Buffer.from(b64, "base64"); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const fileFetch = async (url) => new Response(await readFile(path.join(HERE, "web", url)));

// ── load ─────────────────────────────────────────────────────────────────────
let t = performance.now();
await E.load("data/", { fetch: fileFetch });
const tLoad = performance.now() - t;
const fx = JSON.parse(await readFile(path.join(HERE, "fixtures", "fixtures.json"), "utf8"));
const V = 4096;                                   // vocab_size (logits rows)

// ── T1 tokenizer ─────────────────────────────────────────────────────────────
{
  const bad = fx.tokenize.filter((e) => !same(E.tokenize(e.text), e.ids));
  check("T1a tokenize: same ids as `tokenizers`", bad.length === 0,
        `${fx.tokenize.length - bad.length}/${fx.tokenize.length}` + (bad.length ? `, first bad ${JSON.stringify(bad[0].text)}` : ""));
  const badD = fx.decode.filter((e) => E.decode(e.ids) !== e.text);
  check("T1b decode: same string as `tokenizers`", badD.length === 0, `${fx.decode.length - badD.length}/${fx.decode.length}`);
}

// ── T2 logits ────────────────────────────────────────────────────────────────
{
  let errLast = 0, errAll = 0, errCache = 0, idsOk = true;
  for (const e of fx.logits) {
    idsOk &&= same(E.promptIds(e.prompt), e.ids);
    const T = e.ids.length;
    const zc = E.logits(e.ids, { useCache: true }), zn = E.logits(e.ids, { useCache: false });
    errLast = Math.max(errLast, maxAbs(zc.subarray((T - 1) * V), f32(e.last)));
    if (e.all) errAll = Math.max(errAll, maxAbs(zc, f32(e.all)));
    errCache = Math.max(errCache, maxAbs(zc, zn));
  }
  check("T2a promptIds equal the fixture's ChatML ids", idsOk);
  check("T2b last-position logits, 20 prompts: max |Δz| < 1e-3", errLast < 1e-3, `max |Δz| = ${errLast.toExponential(2)}`);
  check("T2c all-position logits, 2 prompts: max |Δz| < 1e-3", errAll < 1e-3, `max |Δz| = ${errAll.toExponential(2)}`);
  check("T2d cached vs uncached logits: max |Δz| < 1e-4", errCache < 1e-4, `max |Δz| = ${errCache.toExponential(2)}`);
}

// ── T3 greedy ────────────────────────────────────────────────────────────────
async function greedyCheck(entries, label) {
  let ok = 0;
  const bad = [];
  for (const g of entries) {
    const out = await E.generate(g.prompt, {
      concept: g.concept ?? undefined, source: g.source ?? "rfm", blocks: g.blocks ?? "all",
      positions: g.positions, r: g.r, greedy: true, maxNewTokens: fx.max_new_tokens,
    });
    if (same(out.ids, g.ids) && out.text === g.text) ok++;
    else bad.push({ g, out });
  }
  let detail = `${ok}/${entries.length}`;
  for (const { g, out } of bad) {
    const k = out.ids.findIndex((x, i) => x !== g.ids[i]);
    // top-2 gap of the JS logits at the first divergent step (prompt + common prefix)
    let gap = "";
    if (k >= 0) {
      const prefix = E.promptIds(g.prompt).concat(g.ids.slice(0, k));
      gap = ` (unsteered top-2 gap at that step ${top2(E.logits(prefix).subarray((prefix.length - 1) * V))})`;
    }
    detail += `\n      ${g.concept}/${g.source}/S=${g.blocks}/P=${g.positions}/r=${g.r} ${JSON.stringify(g.prompt)}: ` +
              `first divergent token ${k}${gap}; text equal: ${out.text === g.text}`;
  }
  check(label, bad.length === 0, detail);
}
function top2(z) {
  let a = -Infinity, b = -Infinity;
  for (const x of z) { if (x > a) { b = a; a = x; } else if (x > b) b = x; }
  return (a - b).toExponential(2);
}
await greedyCheck(fx.greedy.filter((g) => g.r === 0), "T3a greedy, unsteered: same ids and text");
await greedyCheck(fx.greedy.filter((g) => g.r !== 0), "T3b greedy, steered (P = all, prompt, gen): same ids and text");

// ── T4 judge ─────────────────────────────────────────────────────────────────
for (const c of E.shared().concepts) {
  const ref = fx.judge.g[c];
  let err = 0;
  fx.judge.texts.forEach((s, i) => { err = Math.max(err, Math.abs(E.judge(c, s) - ref[i])); });
  check(`T4 ${c}: judge equals sklearn g_c to 1e-6`, err < 1e-6, `${fx.judge.texts.length} texts, max |Δg| = ${err.toExponential(2)}`);
}

// ── T5 re-read and dose ──────────────────────────────────────────────────────
{
  let err = 0, lenOk = true;
  for (const e of fx.reread) {
    const p = E.cleanReread(e.concept, e.source, e.prompt, e.ids);
    lenOk &&= p.length === e.p.length;
    err = Math.max(err, maxAbs(p, e.p));
  }
  check("T5a clean re-read within 1e-3", lenOk && err < 1e-3, `${fx.reread.length} generations, max |Δp| = ${err.toExponential(2)}`);
  let errD = 0;
  for (const e of fx.dose) errD = Math.max(errD, Math.abs(E.dose(e.concept, e.source, e.blocks, e.r) - e.rho));
  check("T5b dose within 1e-6", errD < 1e-6, `${fx.dose.length} cells, max |Δρ| = ${errD.toExponential(2)}`);
}

// ── T6 isolation ─────────────────────────────────────────────────────────────
{
  const p = fx.greedy[0].prompt;
  const steered = { concept: "food", source: "rfm", blocks: "3", positions: "gen", r: 0.15, seed: 7 };
  const aloneL = await E.generate(p, { ...steered, r: 0 });
  const aloneR = await E.generate(p, steered);
  let interleaved = 0, lastSide = null;
  const note = (side) => () => { if (lastSide && lastSide !== side) interleaved++; lastSide = side; };
  const [L, R] = await Promise.all([
    E.generate(p, { ...steered, r: 0, onToken: note("L") }),
    E.generate(p, { ...steered, onToken: note("R") }),
  ]);
  check("T6 concurrent sampled generations equal the same calls run alone",
        L.text === aloneL.text && R.text === aloneR.text && same(L.ids, aloneL.ids) && same(R.ids, aloneR.ids) && interleaved > 0,
        `${interleaved} switches between the two calls; left ${L.ids.length} tokens, right ${R.ids.length} tokens`);
}

// ── T7 timing ────────────────────────────────────────────────────────────────
// Unsteered answers usually stop early at EOS; steering every block at r = 0.5 gives word salad
// that runs to 64 tokens, so it times a full-length sampled generation.
{
  let run = null;
  for (const g of fx.greedy.slice(0, 20)) {
    t = performance.now();
    const out = await E.generate(g.prompt, { concept: "food", source: "rfm", blocks: "all", r: 0.5, seed: 42 });
    const ms = performance.now() - t;
    if (out.ids.length === 64) { run = { ms, T0: out.T0 }; break; }
  }
  console.log(`[INFO] load ${tLoad.toFixed(0)} ms; ` + (run
    ? `64-token sampled generation after a ${run.T0}-token prompt: ${run.ms.toFixed(0)} ms (${(run.ms / 64).toFixed(1)} ms/token)`
    : "no fixture prompt ran to 64 tokens"));
}

console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`);
process.exit(results.every(Boolean) ? 0 : 1);
