import React, { useState, useMemo } from "react";

/* ------------------------------------------------------------------ *
 * PERFVIEWER — prototype
 * Static architecture graph as the coordinate system; runtime traces
 * from vLLM / SGLang / atom pinned onto it; analytical roofline as the
 * reference everything is judged against.
 *
 * All numbers below are synthetic but internally consistent: bytes and
 * FLOPs are derived from GLM-5.2's real config against MI355X's real
 * spec sheet, and every displayed metric is computed from them.
 * ------------------------------------------------------------------ */

/* ---------- hardware ---------- */
const HW = {
  name: "AMD Instinct MI355X",
  arch: "CDNA 4",
  hbmGB: 288,
  bwTBs: 8.0,            // 8 TB/s peak HBM3E
  peakFp8Tflops: 5000,   // 5 PFLOPS MXFP8
  peakFp4Tflops: 10100,  // 10.1 PFLOPS MXFP4
  fabricGBs: 153,
};
const RIDGE_FP8 = (HW.peakFp8Tflops * 1e12) / (HW.bwTBs * 1e12); // FLOP/byte

/* ---------- model ---------- */
const MODEL = {
  repo: "zai-org/GLM-5.2",
  layers: 78,
  hidden: 6144,
  attn: "MLA (multi-head latent)",
  experts: 256,
  active: 8,
  shared: 1,
  vocab: 154880,
  totalB: 392,
  activeB: 25.4,
  ckptFormat: "NVFP4",
};

/* ---------- workload ---------- */
const WORKLOAD = {
  id: "wl-serve-a",
  batch: 32,
  inputLen: 2048,
  outputLen: 256,
  ctx: 4096,
  tp: 4,
};

/* Expected distinct experts touched per step:
   256 * (1 - (1 - 8/256)^32) ≈ 164  → you stream 64% of all expert
   weights to use 3% of them. This is the whole story of MoE decode. */
const EXPERTS_TOUCHED = 164;

/* ---------- nodes ----------
   bytes / flops are per decode step, per GPU, aggregated over 78 layers.
   `t` holds measured microseconds per run id.                        */
const NODES = [
  { id: "embed_tokens", label: "embed_tokens", kind: "embedding", group: null,
    bytes: 0.012e9, flops: 0.4e9,
    t: { r1: 12, r2: 12, r3: 11, r4: 13 } },

  { id: "input_norm", label: "input_layernorm", kind: "norm", group: "layer",
    bytes: 0.020e9, flops: 0.06e9,
    t: { r1: 41, r2: 41, r3: 26, r4: 44 } },

  { id: "q_a_proj", label: "self_attn.q_a_proj", kind: "linear", group: "layer",
    bytes: 0.090e9, flops: 11.8e9,
    t: { r1: 24, r2: 24, r3: 22, r4: 21 } },

  { id: "q_b_proj", label: "self_attn.q_b_proj", kind: "linear", group: "layer",
    bytes: 0.250e9, flops: 31.4e9,
    t: { r1: 48, r2: 48, r3: 45, r4: 41 } },

  { id: "kv_a_proj", label: "self_attn.kv_a_proj", kind: "linear", group: "layer",
    bytes: 0.035e9, flops: 4.4e9,
    t: { r1: 18, r2: 18, r3: 17, r4: 16 } },

  { id: "kv_b_proj", label: "self_attn.kv_b_proj", kind: "linear", group: "layer",
    bytes: 0.160e9, flops: 21.0e9,
    t: { r1: 39, r2: 39, r3: 37, r4: 33 } },

  { id: "attn_core", label: "self_attn.core (KV cache)", kind: "attention", group: "layer",
    bytes: 1.470e9, flops: 167.5e9,
    bytesOverride: { r2: 2.940e9 },   // bf16 KV cache doubles the read
    t: { r1: 402, r2: 780, r3: 430, r4: 355 } },

  { id: "o_proj", label: "self_attn.o_proj", kind: "linear", group: "layer",
    bytes: 0.980e9, flops: 125.7e9,
    t: { r1: 168, r2: 168, r3: 160, r4: 140 } },

  { id: "post_norm", label: "post_attention_layernorm", kind: "norm", group: "layer",
    bytes: 0.020e9, flops: 0.06e9,
    t: { r1: 40, r2: 40, r3: 25, r4: 43 } },

  { id: "router", label: "mlp.gate (router)", kind: "router", group: "layer",
    bytes: 0.015e9, flops: 1.96e9,
    t: { r1: 88, r2: 88, r3: 42, r4: 79 } },

  { id: "shared_expert", label: "mlp.shared_expert", kind: "mlp", group: "layer",
    bytes: 0.180e9, flops: 23.6e9,
    t: { r1: 51, r2: 51, r3: 44, r4: 47 } },

  { id: "experts", label: "mlp.experts  ·  8 of 256 active", kind: "moe", group: "layer",
    bytes: 30.10e9, flops: 188.7e9,
    t: { r1: 5310, r2: 5310, r3: 4620, r4: 4890 } },

  { id: "final_norm", label: "model.norm", kind: "norm", group: null,
    bytes: 0.001e9, flops: 0.01e9,
    t: { r1: 9, r2: 9, r3: 7, r4: 10 } },

  { id: "lm_head", label: "lm_head", kind: "head", group: null,
    bytes: 0.480e9, flops: 15.2e9,
    t: { r1: 86, r2: 86, r3: 80, r4: 88 } },
];

/* ---------- non-node time: gaps and communication ---------- */
const GAPS = {
  r1: [ { k: "kernel launch gap", us: 310 }, { k: "host sync", us: 95 }, { k: "exposed all-reduce", us: 180 } ],
  r2: [ { k: "kernel launch gap", us: 312 }, { k: "host sync", us: 96 }, { k: "exposed all-reduce", us: 181 } ],
  r3: [ { k: "kernel launch gap", us: 240 }, { k: "host sync", us: 70 }, { k: "exposed all-reduce", us: 175 } ],
  r4: [ { k: "kernel launch gap", us: 480 }, { k: "host sync", us: 140 }, { k: "exposed all-reduce", us: 165 } ],
};

/* ---------- runs ---------- */
const RUNS = {
  r1: { id: "r1", label: "vllm · baseline", stack: "vLLM", version: "0.11.2+rocm",
        flags: { "tensor-parallel-size": "4", "kv-cache-dtype": "fp8", "quantization": "mxfp4",
                 "enable-chunked-prefill": "true", "max-num-batched-tokens": "8192",
                 "attention-backend": "TRITON_MLA", "hip-graphs": "on" },
        attribution: { tier: 1, method: "ROCTX module ranges", confidence: 0.94 },
        noiseFloorPct: 2.1, steps: 4180 },

  r2: { id: "r2", label: "vllm · kv bf16", stack: "vLLM", version: "0.11.2+rocm",
        flags: { "tensor-parallel-size": "4", "kv-cache-dtype": "bf16", "quantization": "mxfp4",
                 "enable-chunked-prefill": "true", "max-num-batched-tokens": "8192",
                 "attention-backend": "TRITON_MLA", "hip-graphs": "on" },
        attribution: { tier: 1, method: "ROCTX module ranges", confidence: 0.94 },
        noiseFloorPct: 2.3, steps: 4102 },

  r3: { id: "r3", label: "sglang · baseline", stack: "SGLang", version: "0.5.4+rocm",
        flags: { "tp-size": "4", "kv-cache-dtype": "fp8", "quantization": "mxfp4",
                 "chunked-prefill-size": "8192", "attention-backend": "aiter",
                 "hip-graphs": "on" },
        attribution: { tier: 1, method: "ROCTX module ranges", confidence: 0.91 },
        noiseFloorPct: 2.4, steps: 4260 },

  r4: { id: "r4", label: "atom · baseline", stack: "atom", version: "internal",
        flags: { "tp": "4", "kv_dtype": "fp8", "weights": "mxfp4", "graph_mode": "on" },
        attribution: { tier: 3, method: "structural inference (period detection)", confidence: 0.68 },
        noiseFloorPct: 3.8, steps: 3990 },
};

/* ---------- palette: the heat ramp is the metric scale ---------- */
const RAMP = ["#2B4B7A", "#1E7F92", "#2FA383", "#C9A227", "#E0533D"];
function heat(f) {
  const x = Math.max(0, Math.min(1, f)) * (RAMP.length - 1);
  const i = Math.floor(x), t = x - i;
  if (i >= RAMP.length - 1) return RAMP[RAMP.length - 1];
  const a = RAMP[i].match(/\w\w/g).map(h => parseInt(h, 16));
  const b = RAMP[i + 1].match(/\w\w/g).map(h => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(",")})`;
}

const KIND_MARK = {
  embedding: "EMB", norm: "NRM", linear: "LIN", attention: "ATT",
  router: "RTR", mlp: "MLP", moe: "MOE", head: "HED",
};

/* ---------- derived metrics ---------- */
function bytesOf(n, runId) {
  return (n.bytesOverride && n.bytesOverride[runId]) || n.bytes;
}
function metricsFor(n, runId) {
  const bytes = bytesOf(n, runId);
  const us = n.t[runId];
  const sec = us / 1e6;
  const predUs = (bytes / (HW.bwTBs * 1e12)) * 1e6;      // bandwidth-limited floor
  const ai = bytes > 0 ? n.flops / bytes : 0;
  const achievedBW = bytes / sec / 1e12;                  // TB/s
  const achievedTflops = n.flops / sec / 1e12;
  return {
    bytes, us, predUs, ai,
    achievedBW,
    achievedTflops,
    bwPct: achievedBW / HW.bwTBs,
    roofPct: predUs / us,
    bound: ai < RIDGE_FP8 ? "memory" : "compute",
  };
}
function stepTotal(runId) {
  const nodes = NODES.reduce((s, n) => s + n.t[runId], 0);
  const gaps = GAPS[runId].reduce((s, g) => s + g.us, 0);
  return { nodes, gaps, total: nodes + gaps };
}

const fmtUs = v => v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${Math.round(v)} µs`;
const fmtGB = v => v >= 1e9 ? `${(v / 1e9).toFixed(2)} GB` : `${(v / 1e6).toFixed(0)} MB`;
const pct = v => `${(v * 100).toFixed(1)}%`;

/* ================================================================== */

export default function PerfViewer() {
  const [tab, setTab] = useState("overlay");
  const [runId, setRunId] = useState("r1");
  const [metric, setMetric] = useState("share");
  const [sel, setSel] = useState("experts");
  const [cmpA, setCmpA] = useState("r1");
  const [cmpB, setCmpB] = useState("r2");

  const totals = useMemo(() => stepTotal(runId), [runId]);
  const run = RUNS[runId];

  const rows = useMemo(() => NODES.map(n => {
    const m = metricsFor(n, runId);
    const share = m.us / totals.total;
    const f = metric === "share" ? Math.min(1, share / 0.5)
            : metric === "bw"    ? m.bwPct
            : metric === "roof"  ? 1 - m.roofPct
            :                      Math.min(1, m.us / 2000);
    return { n, m, share, f };
  }), [runId, metric, totals]);

  const selNode = NODES.find(n => n.id === sel);
  const selM = selNode ? metricsFor(selNode, runId) : null;

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      {/* ---------------- masthead ---------------- */}
      <header style={S.mast}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
          <span style={S.wordmark}>perfviewer</span>
          <span style={S.model}>{MODEL.repo}</span>
          <span style={S.arrow}>on</span>
          <span style={S.model}>{HW.name}</span>
        </div>
        <div style={S.specStrip}>
          <Spec k="layers" v={MODEL.layers} />
          <Spec k="experts" v={`${MODEL.active}/${MODEL.experts}`} />
          <Spec k="attention" v="MLA" />
          <Spec k="HBM" v={`${HW.hbmGB} GB`} />
          <Spec k="peak BW" v={`${HW.bwTBs} TB/s`} />
          <Spec k="MXFP4" v={`${(HW.peakFp4Tflops / 1000).toFixed(1)} PF`} />
          <Spec k="ridge (fp8)" v={`${RIDGE_FP8.toFixed(0)} F/B`} />
        </div>
      </header>

      {/* ---------------- standing finding ---------------- */}
      <div style={S.alert}>
        <span style={S.alertTag}>format</span>
        <span>
          Checkpoint is <b style={{ color: "#E0533D" }}>{MODEL.ckptFormat}</b>, target has native{" "}
          <b style={{ color: "#58D5FF" }}>MXFP4 / MXFP6</b>. Different block size and scale encoding —
          not interchangeable. Requantize from a higher-precision checkpoint before tuning anything.
        </span>
      </div>

      {/* ---------------- nav ---------------- */}
      <nav style={S.nav}>
        {[["overlay", "trace overlay"], ["roofline", "roofline"],
          ["compare", "compare runs"], ["runs", "run registry"]].map(([k, l]) => (
          <button key={k} onClick={() => setTab(k)} style={S.tab(tab === k)}>{l}</button>
        ))}
        <div style={{ flex: 1 }} />
        {tab !== "compare" && tab !== "runs" && (
          <select value={runId} onChange={e => setRunId(e.target.value)} style={S.select}>
            {Object.values(RUNS).map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
          </select>
        )}
      </nav>

      {tab === "overlay" && (
        <Overlay {...{ rows, totals, run, metric, setMetric, sel, setSel, selNode, selM, runId }} />
      )}
      {tab === "roofline" && <Roofline {...{ runId, sel, setSel }} />}
      {tab === "compare" && <Compare {...{ cmpA, cmpB, setCmpA, setCmpB }} />}
      {tab === "runs" && <Registry />}

      <footer style={S.foot}>
        prototype · synthetic measurements derived from GLM-5.2 config × MI355X spec ·
        every metric on screen is computed from bytes and FLOPs, none are hardcoded
      </footer>
    </div>
  );
}

/* ================= trace overlay ================= */

function Overlay({ rows, totals, run, metric, setMetric, sel, setSel, selNode, selM, runId }) {
  const groupRows = rows.filter(r => r.n.group === "layer");
  const preRows = rows.filter(r => !r.n.group && ["embed_tokens"].includes(r.n.id));
  const postRows = rows.filter(r => !r.n.group && !["embed_tokens"].includes(r.n.id));
  const gaps = GAPS[runId];
  const gapTotal = gaps.reduce((s, g) => s + g.us, 0);

  return (
    <div style={S.split}>
      {/* ---- left: the graph ---- */}
      <section style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>decode step · per GPU · TP{WORKLOAD.tp} · batch {WORKLOAD.batch}</span>
          <div style={{ display: "flex", gap: 6 }}>
            {[["share", "share of step"], ["bw", "% of 8 TB/s"], ["roof", "gap to floor"]].map(([k, l]) => (
              <button key={k} onClick={() => setMetric(k)} style={S.chip(metric === k)}>{l}</button>
            ))}
          </div>
        </div>

        <div style={S.legend}>
          <span style={S.dim}>cold</span>
          <div style={S.ramp} />
          <span style={S.dim}>hot</span>
          <span style={{ flex: 1 }} />
          <span style={S.dim}>
            step {fmtUs(totals.total)} · nodes {fmtUs(totals.nodes)} · gaps {fmtUs(totals.gaps)}
          </span>
        </div>

        <div style={S.flow}>
          {preRows.map(r => <NodeRow key={r.n.id} r={r} sel={sel} setSel={setSel} />)}

          <div style={S.group}>
            <div style={S.groupTab}>
              decoder layer <span style={S.repeat}>× {MODEL.layers}</span>
              <span style={S.groupNote}>collapsed — structurally identical</span>
            </div>
            {groupRows.map(r => <NodeRow key={r.n.id} r={r} sel={sel} setSel={setSel} indent />)}
          </div>

          {postRows.map(r => <NodeRow key={r.n.id} r={r} sel={sel} setSel={setSel} />)}

          <div style={S.gapBlock}>
            <div style={S.gapHead}>unattributed to any node · {fmtUs(gapTotal)} · {pct(gapTotal / totals.total)} of step</div>
            {gaps.map(g => (
              <div key={g.k} style={S.gapRow}>
                <span style={{ color: "#7A8DA0" }}>{g.k}</span>
                <span style={S.mono}>{fmtUs(g.us)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---- right: detail ---- */}
      <aside style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>node detail</span>
          <span style={S.tier(run.attribution.tier)}>
            tier {run.attribution.tier} · {pct(run.attribution.confidence)} attributed
          </span>
        </div>

        {selNode && (
          <div style={{ padding: "16px 18px" }}>
            <div style={S.detailTitle}>{selNode.label}</div>
            <div style={S.detailKind}>{KIND_MARK[selNode.kind]} · {selNode.kind}
              {selNode.group === "layer" && <> · ×{MODEL.layers} layers aggregated</>}</div>

            <VerdictCard m={selM} node={selNode} />

            <div style={S.kvGrid}>
              <KV k="measured" v={fmtUs(selM.us)} />
              <KV k="bandwidth floor" v={fmtUs(selM.predUs)} />
              <KV k="bytes moved" v={fmtGB(selM.bytes)} />
              <KV k="FLOPs" v={`${(selNode.flops / 1e9).toFixed(1)} G`} />
              <KV k="arithmetic intensity" v={`${selM.ai.toFixed(1)} F/B`} />
              <KV k="ridge point" v={`${RIDGE_FP8.toFixed(0)} F/B`} />
              <KV k="achieved BW" v={`${selM.achievedBW.toFixed(2)} TB/s`} hi={selM.bwPct} />
              <KV k="achieved compute" v={`${selM.achievedTflops.toFixed(1)} TF/s`} />
            </div>

            {selNode.id === "experts" && (
              <div style={S.insight}>
                <div style={S.insightHead}>why this node dominates</div>
                <p style={S.p}>
                  Top-{MODEL.active} routing over {MODEL.experts} experts at batch {WORKLOAD.batch} touches{" "}
                  <b style={{ color: "#E0533D" }}>~{EXPERTS_TOUCHED} distinct experts</b> per step. You stream{" "}
                  {pct(EXPERTS_TOUCHED / MODEL.experts)} of all expert weights to use{" "}
                  {pct(MODEL.active / MODEL.experts)} of them.
                </p>
                <p style={S.p}>
                  Arithmetic intensity {selM.ai.toFixed(1)} against a ridge of {RIDGE_FP8.toFixed(0)} — this is
                  memory-bound by a factor of {(RIDGE_FP8 / selM.ai).toFixed(0)}×. Extra FLOPs are free here;
                  every byte is not.
                </p>
                <div style={S.leverHead}>levers, in order of expected effect</div>
                <ul style={S.ul}>
                  <li>Larger batch — amortizes the same weight read over more tokens</li>
                  <li>Expert parallelism — fewer experts resident per GPU, less streamed per step</li>
                  <li>Kernel: {pct(selM.bwPct)} of peak bandwidth today, so ~{pct(0.9 - selM.bwPct)} left on the table</li>
                </ul>
              </div>
            )}

            {selNode.id === "attn_core" && (
              <div style={S.insight}>
                <div style={S.insightHead}>reading this node</div>
                <p style={S.p}>
                  MLA compresses the KV cache into a latent, which raises arithmetic intensity to{" "}
                  {selM.ai.toFixed(0)} F/B — roughly {(selM.ai / 6.3).toFixed(0)}× the expert layers. Still
                  memory-bound, but far less severely.
                </p>
                <p style={S.p}>
                  KV cache dtype moves this node almost linearly. Compare against{" "}
                  <b style={{ color: "#58D5FF" }}>vllm · kv bf16</b> to see it.
                </p>
              </div>
            )}

            {selNode.id === "router" && (
              <div style={S.insight}>
                <div style={S.insightHead}>disproportionate</div>
                <p style={S.p}>
                  {fmtGB(selM.bytes)} of traffic taking {fmtUs(selM.us)}. The floor is {fmtUs(selM.predUs)}.
                  This is not a bandwidth problem — it is launch overhead and small-kernel inefficiency on a
                  tiny operation repeated {MODEL.layers} times. SGLang runs the same work in{" "}
                  {fmtUs(NODES.find(n => n.id === "router").t.r3)}.
                </p>
              </div>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

function VerdictCard({ m }) {
  const bad = m.roofPct < 0.75;
  return (
    <div style={{ ...S.verdict, borderColor: bad ? "#E0533D44" : "#2FA38344" }}>
      <div style={S.verdictRow}>
        <span style={S.verdictLabel}>{m.bound}-bound</span>
        <span style={{ ...S.verdictBig, color: heat(1 - m.roofPct) }}>{pct(m.roofPct)}</span>
      </div>
      <div style={S.barTrack}>
        <div style={{ ...S.barFill, width: `${Math.min(100, m.roofPct * 100)}%`,
                      background: heat(1 - m.roofPct) }} />
      </div>
      <div style={S.verdictFoot}>
        of the bandwidth-limited floor · {fmtUs(m.us - m.predUs)} above the theoretical minimum
      </div>
    </div>
  );
}

function NodeRow({ r, sel, setSel, indent }) {
  const active = sel === r.n.id;
  return (
    <button onClick={() => setSel(r.n.id)}
      style={{ ...S.node, marginLeft: indent ? 14 : 0,
               borderColor: active ? "#58D5FF" : "#232E3A",
               background: active ? "#132330" : "#141B23" }}>
      <span style={{ ...S.mark, background: heat(r.f) }}>{KIND_MARK[r.n.kind]}</span>
      <span style={S.nodeLabel}>{r.n.label}</span>
      <span style={S.nodeBarWrap}>
        <span style={{ ...S.nodeBar, width: `${Math.min(100, r.share * 200)}%`, background: heat(r.f) }} />
      </span>
      <span style={S.nodeTime}>{fmtUs(r.m.us)}</span>
      <span style={S.nodePct}>{pct(r.share)}</span>
    </button>
  );
}

/* ================= roofline ================= */

function Roofline({ runId, sel, setSel }) {
  const W = 760, H = 420, PAD = { l: 62, r: 24, t: 24, b: 46 };
  const xMin = 1, xMax = 5000, yMin = 1, yMax = 12000;
  const lx = v => PAD.l + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (W - PAD.l - PAD.r);
  const ly = v => H - PAD.b - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - PAD.t - PAD.b);

  const roofY = ai => Math.min(HW.peakFp8Tflops, ai * HW.bwTBs);
  const roofPath = [];
  for (let e = 0; e <= 100; e++) {
    const ai = xMin * Math.pow(xMax / xMin, e / 100);
    roofPath.push(`${e === 0 ? "M" : "L"}${lx(ai)},${ly(Math.max(yMin, roofY(ai)))}`);
  }

  const pts = NODES.map(n => {
    const m = metricsFor(n, runId);
    return { n, m };
  }).filter(p => p.m.ai > 0.5 && p.m.achievedTflops > 0.5);

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>roofline · {RUNS[runId].label} · decode</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          {[1, 10, 100, 1000].map(g => (
            <g key={`x${g}`}>
              <line x1={lx(g)} y1={PAD.t} x2={lx(g)} y2={H - PAD.b} stroke="#1D2733" />
              <text x={lx(g)} y={H - PAD.b + 16} fill="#5C6E80" fontSize="10" textAnchor="middle"
                    fontFamily="ui-monospace, monospace">{g}</text>
            </g>
          ))}
          {[1, 10, 100, 1000, 10000].map(g => (
            <g key={`y${g}`}>
              <line x1={PAD.l} y1={ly(g)} x2={W - PAD.r} y2={ly(g)} stroke="#1D2733" />
              <text x={PAD.l - 8} y={ly(g) + 3} fill="#5C6E80" fontSize="10" textAnchor="end"
                    fontFamily="ui-monospace, monospace">{g}</text>
            </g>
          ))}

          {/* the roof */}
          <path d={roofPath.join(" ")} fill="none" stroke="#58D5FF" strokeWidth="1.5" opacity="0.85" />
          <text x={lx(3)} y={ly(30)} fill="#58D5FF" fontSize="10" opacity="0.8"
                fontFamily="ui-monospace, monospace" transform={`rotate(-31 ${lx(3)} ${ly(30)})`}>
            8 TB/s bandwidth roof
          </text>

          {/* ridge */}
          <line x1={lx(RIDGE_FP8)} y1={PAD.t} x2={lx(RIDGE_FP8)} y2={H - PAD.b}
                stroke="#E0533D" strokeDasharray="3 3" strokeWidth="1.2" />
          <text x={lx(RIDGE_FP8) + 6} y={PAD.t + 12} fill="#E0533D" fontSize="10"
                fontFamily="ui-monospace, monospace">ridge {RIDGE_FP8.toFixed(0)} F/B</text>

          {/* shaded memory-bound region */}
          <rect x={PAD.l} y={PAD.t} width={lx(RIDGE_FP8) - PAD.l} height={H - PAD.t - PAD.b}
                fill="#E0533D" opacity="0.045" />

          {pts.map(p => {
            const on = sel === p.n.id;
            return (
              <g key={p.n.id} onClick={() => setSel(p.n.id)} style={{ cursor: "pointer" }}>
                <circle cx={lx(p.m.ai)} cy={ly(p.m.achievedTflops)} r={on ? 7 : 4.5}
                        fill={heat(1 - p.m.roofPct)} stroke={on ? "#58D5FF" : "#0F141B"} strokeWidth={on ? 2 : 1} />
                {(on || p.n.kind === "moe" || p.n.kind === "attention") && (
                  <text x={lx(p.m.ai) + 10} y={ly(p.m.achievedTflops) + 3} fill={on ? "#58D5FF" : "#8FA3B5"}
                        fontSize="10" fontFamily="ui-monospace, monospace">{p.n.id}</text>
                )}
              </g>
            );
          })}

          <text x={W / 2} y={H - 6} fill="#5C6E80" fontSize="10" textAnchor="middle"
                fontFamily="ui-monospace, monospace">arithmetic intensity — FLOP / byte</text>
          <text x={14} y={H / 2} fill="#5C6E80" fontSize="10" textAnchor="middle"
                fontFamily="ui-monospace, monospace" transform={`rotate(-90 14 ${H / 2})`}>achieved TFLOP/s</text>
        </svg>
      </section>

      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>what this plot says</span></div>
        <div style={{ padding: "16px 18px" }}>
          <p style={S.pBig}>
            Every node sits in the shaded region. On this chip the ridge is at{" "}
            <b style={{ color: "#E0533D" }}>{RIDGE_FP8.toFixed(0)} FLOP/byte</b> — decode never comes close.
          </p>
          <p style={S.p}>
            Decode arithmetic intensity is roughly twice the batch size. To reach the ridge you would need a
            batch in the hundreds. Until then, FLOPs are free and bytes are everything: no kernel that trades
            memory traffic for arithmetic can lose here.
          </p>
          <p style={S.p}>
            Vertical distance to the roof is the recoverable win. Horizontal position is what the model and
            serving config decide — you move a point left or right by changing batching, routing, or cache
            layout, not by tuning a kernel.
          </p>
          <div style={S.tableMini}>
            <div style={S.tmHead}><span>node</span><span>AI</span><span>× below ridge</span></div>
            {pts.slice().sort((a, b) => a.m.ai - b.m.ai).slice(0, 5).map(p => (
              <div key={p.n.id} style={S.tmRow} onClick={() => setSel(p.n.id)}>
                <span style={{ color: sel === p.n.id ? "#58D5FF" : "#C9D6E2" }}>{p.n.id}</span>
                <span style={S.mono}>{p.m.ai.toFixed(1)}</span>
                <span style={{ ...S.mono, color: heat(Math.min(1, (RIDGE_FP8 / p.m.ai) / 120)) }}>
                  {(RIDGE_FP8 / p.m.ai).toFixed(0)}×
                </span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

/* ================= compare ================= */

function Compare({ cmpA, cmpB, setCmpA, setCmpB }) {
  const A = RUNS[cmpA], B = RUNS[cmpB];
  const tA = stepTotal(cmpA), tB = stepTotal(cmpB);
  const noise = Math.max(A.noiseFloorPct, B.noiseFloorPct) / 100;

  const deltas = NODES.map(n => {
    const a = n.t[cmpA], b = n.t[cmpB];
    const d = b - a;
    const rel = d / a;
    return { id: n.label, short: n.id, a, b, d, rel, sig: Math.abs(rel) > noise };
  }).concat(
    GAPS[cmpA].map((g, i) => {
      const a = g.us, b = GAPS[cmpB][i].us, d = b - a;
      return { id: g.k, short: g.k, a, b, d, rel: d / a, sig: Math.abs(d / a) > noise, gap: true };
    })
  ).sort((x, y) => Math.abs(y.d) - Math.abs(x.d));

  const totalD = tB.total - tA.total;
  const maxAbs = Math.max(...deltas.map(d => Math.abs(d.d)), 1);

  const flagKeys = Array.from(new Set([...Object.keys(A.flags), ...Object.keys(B.flags)]));
  const flagDiff = flagKeys.filter(k => A.flags[k] !== B.flags[k]);

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>delta attribution</span>
          <div style={{ display: "flex", gap: 6 }}>
            <select value={cmpA} onChange={e => setCmpA(e.target.value)} style={S.select}>
              {Object.values(RUNS).map(r => <option key={r.id} value={r.id}>A · {r.label}</option>)}
            </select>
            <select value={cmpB} onChange={e => setCmpB(e.target.value)} style={S.select}>
              {Object.values(RUNS).map(r => <option key={r.id} value={r.id}>B · {r.label}</option>)}
            </select>
          </div>
        </div>

        <div style={S.headline}>
          <div>
            <div style={S.dim}>step time</div>
            <div style={S.headNums}>
              <span style={S.mono}>{fmtUs(tA.total)}</span>
              <span style={S.arrow}>→</span>
              <span style={S.mono}>{fmtUs(tB.total)}</span>
            </div>
          </div>
          <div style={{ ...S.headDelta, color: totalD > 0 ? "#E0533D" : "#2FA383" }}>
            {totalD > 0 ? "+" : ""}{((totalD / tA.total) * 100).toFixed(1)}%
          </div>
        </div>

        <div style={{ padding: "4px 18px 18px" }}>
          {deltas.map(d => {
            const w = (Math.abs(d.d) / maxAbs) * 46;
            const col = !d.sig ? "#3B4A5A" : d.d > 0 ? "#E0533D" : "#2FA383";
            return (
              <div key={d.short} style={S.wRow}>
                <span style={{ ...S.wLabel, color: d.sig ? "#C9D6E2" : "#5C6E80" }}>
                  {d.gap && <span style={S.gapTag}>gap</span>}{d.id}
                </span>
                <div style={S.wTrack}>
                  <div style={S.wMid} />
                  <div style={{ position: "absolute", top: 4, height: 10, background: col, borderRadius: 1,
                                left: d.d > 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} />
                </div>
                <span style={{ ...S.mono, width: 74, textAlign: "right", color: col }}>
                  {d.d > 0 ? "+" : ""}{fmtUs(Math.abs(d.d)).replace(/^/, d.d < 0 ? "−" : "")}
                </span>
                <span style={{ ...S.mono, width: 52, textAlign: "right",
                               color: d.sig ? col : "#3B4A5A", fontSize: 11 }}>
                  {d.sig ? `${(d.rel * 100).toFixed(0)}%` : "noise"}
                </span>
              </div>
            );
          })}
          <div style={S.noiseNote}>
            Greyed rows fall inside the noise floor (±{(noise * 100).toFixed(1)}%, from{" "}
            {Math.min(A.steps, B.steps).toLocaleString()} steps). They are not findings.
          </div>
        </div>
      </section>

      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>what actually changed</span></div>
        <div style={{ padding: "16px 18px" }}>
          {A.stack !== B.stack && (
            <div style={S.stackWarn}>
              Cross-stack comparison — nodes are aligned at layer-component granularity only.
              Op-level deltas are not comparable across {A.stack} and {B.stack}.
            </div>
          )}

          <div style={S.cfgHead}>configuration diff</div>
          {flagDiff.length === 0 && <div style={S.dim}>identical flags</div>}
          {flagDiff.map(k => (
            <div key={k} style={S.cfgRow}>
              <span style={S.cfgKey}>{k}</span>
              <span style={{ ...S.mono, color: "#7A8DA0" }}>{A.flags[k] ?? "—"}</span>
              <span style={S.arrow}>→</span>
              <span style={{ ...S.mono, color: "#58D5FF" }}>{B.flags[k] ?? "—"}</span>
            </div>
          ))}

          <div style={S.cfgHead}>attribution quality</div>
          <div style={S.cfgRow}>
            <span style={S.cfgKey}>{A.label}</span>
            <span style={S.tier(A.attribution.tier)}>tier {A.attribution.tier} · {pct(A.attribution.confidence)}</span>
          </div>
          <div style={S.cfgRow}>
            <span style={S.cfgKey}>{B.label}</span>
            <span style={S.tier(B.attribution.tier)}>tier {B.attribution.tier} · {pct(B.attribution.confidence)}</span>
          </div>
          {(A.attribution.confidence < 0.7 || B.attribution.confidence < 0.7) && (
            <div style={S.lowConf}>
              One run is below the 70% attribution gate. Per-node deltas from it are directional only —
              the top-line number is still sound.
            </div>
          )}

          <div style={S.cfgHead}>reading</div>
          <p style={S.p}>{narrative(A, B, deltas, totalD, tA)}</p>
        </div>
      </aside>
    </div>
  );
}

function narrative(A, B, deltas, totalD, tA) {
  const top = deltas.filter(d => d.sig).slice(0, 2);
  if (A.stack !== B.stack) {
    const wins = deltas.filter(d => d.sig && d.d < 0).slice(0, 1);
    const loss = deltas.filter(d => d.sig && d.d > 0).slice(0, 1);
    return `${B.stack} is ${totalD < 0 ? "faster" : "slower"} overall by ${Math.abs((totalD / tA.total) * 100).toFixed(1)}%. `
      + (wins.length ? `It wins on ${wins[0].id} by ${fmtUs(Math.abs(wins[0].d))} — worth reading that kernel. ` : "")
      + (loss.length ? `It loses on ${loss[0].id} by ${fmtUs(loss[0].d)}.` : "");
  }
  if (!top.length) return "No node moved beyond the noise floor. This change did nothing measurable.";
  return `${totalD > 0 ? "Regression" : "Improvement"} of ${Math.abs((totalD / tA.total) * 100).toFixed(1)}%, `
    + `concentrated in ${top[0].id} (${top[0].d > 0 ? "+" : "−"}${fmtUs(Math.abs(top[0].d))}). `
    + `The config diff shows exactly one flag moved — the causal chain is short.`;
}

/* ================= registry ================= */

function Registry() {
  return (
    <div style={{ ...S.panel, margin: "0 20px 20px" }}>
      <div style={S.panelHead}>
        <span style={S.eyebrow}>run registry — the actual substance of the tool</span>
      </div>
      <div style={{ padding: "6px 18px 18px", overflowX: "auto" }}>
        <table style={S.table}>
          <thead>
            <tr>{["run", "stack", "version", "workload", "TP", "attribution", "step", "TPOT"].map(h =>
              <th key={h} style={S.th}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {Object.values(RUNS).map(r => {
              const t = stepTotal(r.id);
              return (
                <tr key={r.id}>
                  <td style={{ ...S.td, color: "#58D5FF" }}>{r.label}</td>
                  <td style={S.td}>{r.stack}</td>
                  <td style={S.td}>{r.version}</td>
                  <td style={S.td}>{WORKLOAD.id}</td>
                  <td style={S.td}>{WORKLOAD.tp}</td>
                  <td style={S.td}><span style={S.tier(r.attribution.tier)}>
                    tier {r.attribution.tier} · {pct(r.attribution.confidence)}</span></td>
                  <td style={{ ...S.td, ...S.mono }}>{fmtUs(t.total)}</td>
                  <td style={{ ...S.td, ...S.mono }}>{(t.total / 1000).toFixed(2)} ms</td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div style={S.regNote}>
          <p style={S.p}>
            A run is <b style={{ color: "#C9D6E2" }}>(model artifact × stack + flags × workload spec × hardware +
            parallelism) → measurements</b>. Every capability in this prototype — heat overlay, roofline,
            regression, cross-stack — is a query over this table.
          </p>
          <p style={S.p}>
            The workload spec is referenced, never re-specified. Two runs that don't share{" "}
            <span style={S.mono}>{WORKLOAD.id}</span> cannot be compared, and the tool should refuse rather
            than produce a plausible-looking wrong number.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ================= small parts ================= */

const Spec = ({ k, v }) => (
  <span style={S.spec}><span style={S.specK}>{k}</span><span style={S.specV}>{v}</span></span>
);
const KV = ({ k, v, hi }) => (
  <div style={S.kv}>
    <div style={S.kvK}>{k}</div>
    <div style={{ ...S.kvV, color: hi !== undefined ? heat(1 - hi) : "#DCE7F1" }}>{v}</div>
  </div>
);

/* ================= styles ================= */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&display=swap');
* { box-sizing: border-box; }
button { font-family: inherit; cursor: pointer; }
button:focus-visible, select:focus-visible { outline: 2px solid #58D5FF; outline-offset: 2px; }
@media (max-width: 900px) { .pv-split { grid-template-columns: 1fr !important; } }
`;

const mono = `'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace`;
const sans = `'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif`;
const cond = `'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif`;

const S = {
  root: { background: "#0F141B", color: "#C9D6E2", fontFamily: sans, minHeight: "100vh", fontSize: 13 },
  mast: { padding: "20px 20px 12px", borderBottom: "1px solid #1D2733" },
  wordmark: { fontFamily: cond, fontSize: 20, letterSpacing: "0.14em", textTransform: "uppercase", color: "#58D5FF" },
  model: { fontFamily: mono, fontSize: 13, color: "#DCE7F1" },
  arrow: { color: "#5C6E80", fontSize: 12 },
  specStrip: { display: "flex", gap: 18, flexWrap: "wrap", marginTop: 12 },
  spec: { display: "flex", flexDirection: "column", gap: 2 },
  specK: { fontFamily: cond, fontSize: 9.5, letterSpacing: "0.13em", textTransform: "uppercase", color: "#5C6E80" },
  specV: { fontFamily: mono, fontSize: 12.5, color: "#A9BECE" },

  alert: { display: "flex", gap: 12, alignItems: "flex-start", margin: "14px 20px 0", padding: "11px 14px",
           background: "#1A1520", border: "1px solid #E0533D33", borderRadius: 3, fontSize: 12.5, lineHeight: 1.55 },
  alertTag: { fontFamily: cond, fontSize: 9.5, letterSpacing: "0.13em", textTransform: "uppercase",
              color: "#E0533D", paddingTop: 3, flexShrink: 0 },

  nav: { display: "flex", gap: 4, alignItems: "center", padding: "14px 20px 0" },
  tab: on => ({ background: "none", border: "none", borderBottom: `2px solid ${on ? "#58D5FF" : "transparent"}`,
                color: on ? "#DCE7F1" : "#6E8095", fontFamily: cond, fontSize: 12,
                letterSpacing: "0.1em", textTransform: "uppercase", padding: "6px 10px" }),
  select: { background: "#141B23", border: "1px solid #232E3A", color: "#A9BECE", fontFamily: mono,
            fontSize: 11.5, padding: "5px 8px", borderRadius: 2 },

  split: { display: "grid", gridTemplateColumns: "1.35fr 1fr", gap: 14, padding: "14px 20px 20px",
           alignItems: "start", className: "pv-split" },
  panel: { background: "#121921", border: "1px solid #1D2733", borderRadius: 3 },
  panelHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10,
               padding: "10px 14px", borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  eyebrow: { fontFamily: cond, fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6E8095" },
  chip: on => ({ background: on ? "#16303D" : "transparent", border: `1px solid ${on ? "#58D5FF66" : "#232E3A"}`,
                 color: on ? "#58D5FF" : "#6E8095", fontFamily: mono, fontSize: 10.5, padding: "3px 8px", borderRadius: 2 }),
  tier: t => ({ fontFamily: mono, fontSize: 10.5, padding: "2px 7px", borderRadius: 2,
                background: t === 1 ? "#123028" : t === 3 ? "#2E1E14" : "#1A2430",
                color: t === 1 ? "#2FA383" : t === 3 ? "#C9A227" : "#7A8DA0" }),

  legend: { display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderBottom: "1px solid #1D2733" },
  ramp: { width: 90, height: 7, borderRadius: 1, background: `linear-gradient(90deg, ${RAMP.join(",")})` },
  dim: { color: "#5C6E80", fontFamily: mono, fontSize: 10.5 },

  flow: { padding: "12px 14px", display: "flex", flexDirection: "column", gap: 3 },
  node: { display: "flex", alignItems: "center", gap: 9, width: "100%", textAlign: "left",
          border: "1px solid #232E3A", borderRadius: 2, padding: "6px 9px", color: "#C9D6E2" },
  mark: { fontFamily: mono, fontSize: 9, fontWeight: 600, color: "#0F141B", padding: "2px 4px",
          borderRadius: 1, letterSpacing: "0.04em", flexShrink: 0 },
  nodeLabel: { fontFamily: mono, fontSize: 11.5, flex: "0 0 210px", overflow: "hidden",
               textOverflow: "ellipsis", whiteSpace: "nowrap" },
  nodeBarWrap: { flex: 1, height: 8, background: "#0F141B", borderRadius: 1, overflow: "hidden", minWidth: 40 },
  nodeBar: { display: "block", height: "100%" },
  nodeTime: { fontFamily: mono, fontSize: 11, width: 62, textAlign: "right", color: "#A9BECE" },
  nodePct: { fontFamily: mono, fontSize: 10.5, width: 46, textAlign: "right", color: "#5C6E80" },

  group: { border: "1px dashed #2A3A48", borderRadius: 3, padding: "20px 8px 8px", margin: "8px 0",
           position: "relative", display: "flex", flexDirection: "column", gap: 3 },
  groupTab: { position: "absolute", top: -1, left: 10, transform: "translateY(-50%)", background: "#121921",
              padding: "0 8px", fontFamily: cond, fontSize: 10, letterSpacing: "0.12em",
              textTransform: "uppercase", color: "#6E8095", display: "flex", gap: 8, alignItems: "center" },
  repeat: { color: "#58D5FF", fontFamily: mono, letterSpacing: 0 },
  groupNote: { color: "#3B4A5A", letterSpacing: "0.04em", textTransform: "none", fontFamily: mono, fontSize: 9.5 },

  gapBlock: { marginTop: 10, border: "1px solid #232E3A", borderRadius: 2, background: "#0F141B", padding: "8px 10px" },
  gapHead: { fontFamily: mono, fontSize: 10.5, color: "#C9A227", marginBottom: 6 },
  gapRow: { display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11, padding: "2px 0" },

  detailTitle: { fontFamily: mono, fontSize: 14, color: "#DCE7F1" },
  detailKind: { fontFamily: mono, fontSize: 10.5, color: "#5C6E80", marginTop: 3 },

  verdict: { marginTop: 14, border: "1px solid", borderRadius: 3, padding: "11px 12px", background: "#0F141B" },
  verdictRow: { display: "flex", justifyContent: "space-between", alignItems: "baseline" },
  verdictLabel: { fontFamily: cond, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "#6E8095" },
  verdictBig: { fontFamily: mono, fontSize: 22, fontWeight: 500 },
  barTrack: { height: 5, background: "#1A2430", borderRadius: 1, margin: "8px 0 6px", overflow: "hidden" },
  barFill: { height: "100%" },
  verdictFoot: { fontFamily: mono, fontSize: 10.5, color: "#5C6E80" },

  kvGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, marginTop: 14,
            background: "#1D2733", border: "1px solid #1D2733", borderRadius: 2 },
  kv: { background: "#0F141B", padding: "8px 10px" },
  kvK: { fontFamily: cond, fontSize: 9.5, letterSpacing: "0.11em", textTransform: "uppercase", color: "#5C6E80" },
  kvV: { fontFamily: mono, fontSize: 13, marginTop: 3 },

  insight: { marginTop: 16, borderLeft: "2px solid #58D5FF", paddingLeft: 12 },
  insightHead: { fontFamily: cond, fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase",
                 color: "#58D5FF", marginBottom: 7 },
  leverHead: { fontFamily: cond, fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase",
               color: "#6E8095", margin: "12px 0 6px" },
  p: { fontSize: 12.5, lineHeight: 1.6, color: "#A9BECE", margin: "0 0 9px" },
  pBig: { fontSize: 14, lineHeight: 1.55, color: "#DCE7F1", margin: "0 0 12px" },
  ul: { margin: 0, paddingLeft: 16, fontSize: 12.5, lineHeight: 1.7, color: "#A9BECE" },

  tableMini: { marginTop: 16, border: "1px solid #1D2733", borderRadius: 2 },
  tmHead: { display: "grid", gridTemplateColumns: "1fr 60px 90px", padding: "6px 10px",
            borderBottom: "1px solid #1D2733", fontFamily: cond, fontSize: 9.5,
            letterSpacing: "0.11em", textTransform: "uppercase", color: "#5C6E80" },
  tmRow: { display: "grid", gridTemplateColumns: "1fr 60px 90px", padding: "5px 10px",
           fontFamily: mono, fontSize: 11.5, cursor: "pointer", textAlign: "left" },

  headline: { display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "14px 18px", borderBottom: "1px solid #1D2733" },
  headNums: { display: "flex", gap: 10, alignItems: "baseline", fontSize: 16, marginTop: 4 },
  headDelta: { fontFamily: mono, fontSize: 30, fontWeight: 500 },

  wRow: { display: "flex", alignItems: "center", gap: 8, padding: "3px 0" },
  wLabel: { fontFamily: mono, fontSize: 11, flex: "0 0 190px", overflow: "hidden",
            textOverflow: "ellipsis", whiteSpace: "nowrap" },
  wTrack: { flex: 1, height: 18, position: "relative", minWidth: 80 },
  wMid: { position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#232E3A" },
  gapTag: { color: "#C9A227", marginRight: 6, fontSize: 9.5 },
  noiseNote: { marginTop: 12, fontFamily: mono, fontSize: 10.5, color: "#5C6E80", lineHeight: 1.5 },

  stackWarn: { border: "1px solid #C9A22733", background: "#1C1810", padding: "9px 11px", borderRadius: 2,
               fontSize: 12, lineHeight: 1.5, color: "#C9A227", marginBottom: 14 },
  lowConf: { border: "1px solid #C9A22733", background: "#1C1810", padding: "9px 11px", borderRadius: 2,
             fontSize: 12, lineHeight: 1.5, color: "#C9A227", marginTop: 10 },
  cfgHead: { fontFamily: cond, fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase",
             color: "#6E8095", margin: "16px 0 8px" },
  cfgRow: { display: "flex", gap: 8, alignItems: "center", padding: "4px 0", flexWrap: "wrap" },
  cfgKey: { fontFamily: mono, fontSize: 11.5, color: "#DCE7F1", flex: "0 0 auto" },

  table: { width: "100%", borderCollapse: "collapse", marginTop: 8 },
  th: { fontFamily: cond, fontSize: 9.5, letterSpacing: "0.12em", textTransform: "uppercase",
        color: "#5C6E80", textAlign: "left", padding: "7px 10px", borderBottom: "1px solid #1D2733" },
  td: { fontFamily: mono, fontSize: 11.5, padding: "8px 10px", borderBottom: "1px solid #161E27", color: "#A9BECE" },
  regNote: { marginTop: 18, maxWidth: 760 },

  mono: { fontFamily: mono },
  foot: { padding: "14px 20px 26px", fontFamily: mono, fontSize: 10.5, color: "#3B4A5A",
          borderTop: "1px solid #1D2733", lineHeight: 1.6 },
};
