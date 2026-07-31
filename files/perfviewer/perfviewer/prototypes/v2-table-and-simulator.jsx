import React, { useState, useMemo } from "react";

/* ==================================================================
 * PERFVIEWER v2 — GLM-5.2 on MI355X
 *
 * v1 had the graph and the diff. v2 adds the parts of Talaria
 * (Hohman et al., CHI '24) that actually made it useful:
 *   · a real Table View — sort, filter, search, find top offenders  (T1, C2)
 *   · cross-linked table ↔ graph selection                          (C1)
 *   · interactive optimization simulation, model-wide and targeted  (T3)
 *   · source-location tracking per node                             (T5)
 *   · an execution timeline as a complementary view
 *
 * The simulator is a real analytical model, not a lookup: node bytes
 * are computed from GLM-5.2's config under the chosen configuration,
 * and time is bytes ÷ (8 TB/s × measured efficiency). Efficiency is
 * taken from the baseline trace, so the prediction inherits reality
 * rather than assuming perfection.
 * ================================================================== */

const HW = { name: "AMD Instinct MI355X", hbmGB: 288, bwTBs: 8.0,
             peakFp8: 5000, peakFp4: 10100 };
const RIDGE = (HW.peakFp8 * 1e12) / (HW.bwTBs * 1e12);

const MODEL = { repo: "zai-org/GLM-5.2", layers: 78, hidden: 6144, experts: 256,
                active: 8, vocab: 154880, ckpt: "NVFP4",
                expertParamsPerLayer: 4.83e9, attnParamsPerLayer: 156e6,
                headParams: 951.6e6, kvPerTokenPerLayer: 576 };

const BPP = { mxfp4: 0.5, mxfp6: 0.75, mxfp8: 1.0, bf16: 2.0 };
const KVB = { fp4: 0.5, fp8: 1.0, bf16: 2.0 };

const BASE_CFG = { batch: 32, ctx: 4096, tp: 4, ep: false,
                   weights: "mxfp4", kv: "fp8", moeKernel: "triton" };

/* ---- nodes: geometry + how each responds to configuration ---- */
const NODES = [
  { id: "embed_tokens", label: "embed_tokens", kind: "embedding", group: null, scaling: "fixed",
    src: "vllm/model_executor/layers/vocab_parallel_embedding.py:389",
    flops: 0.4e9, base: { bytes: 0.012e9, us: 12 } },
  { id: "input_norm", label: "input_layernorm", kind: "norm", group: "layer", scaling: "fixed",
    src: "vllm/model_executor/layers/layernorm.py:112",
    flops: 0.06e9, base: { bytes: 0.020e9, us: 41 } },
  { id: "q_a_proj", label: "self_attn.q_a_proj", kind: "linear", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/linear.py:1043",
    flops: 11.8e9, base: { bytes: 0.090e9, us: 24 } },
  { id: "q_b_proj", label: "self_attn.q_b_proj", kind: "linear", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/linear.py:1043",
    flops: 31.4e9, base: { bytes: 0.250e9, us: 48 } },
  { id: "kv_a_proj", label: "self_attn.kv_a_proj", kind: "linear", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/linear.py:1043",
    flops: 4.4e9, base: { bytes: 0.035e9, us: 18 } },
  { id: "kv_b_proj", label: "self_attn.kv_b_proj", kind: "linear", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/linear.py:1043",
    flops: 21.0e9, base: { bytes: 0.160e9, us: 39 } },
  { id: "attn_core", label: "self_attn.core (KV cache)", kind: "attention", group: "layer", scaling: "kv",
    src: "vllm/attention/backends/triton_mla.py:274",
    flops: 167.5e9, base: { bytes: 1.470e9, us: 402 } },
  { id: "o_proj", label: "self_attn.o_proj", kind: "linear", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/linear.py:1043",
    flops: 125.7e9, base: { bytes: 0.980e9, us: 168 } },
  { id: "post_norm", label: "post_attention_layernorm", kind: "norm", group: "layer", scaling: "fixed",
    src: "vllm/model_executor/layers/layernorm.py:112",
    flops: 0.06e9, base: { bytes: 0.020e9, us: 40 } },
  { id: "router", label: "mlp.gate (router)", kind: "router", group: "layer", scaling: "fixed",
    src: "vllm/model_executor/layers/fused_moe/layer.py:196",
    flops: 1.96e9, base: { bytes: 0.015e9, us: 88 } },
  { id: "shared_expert", label: "mlp.shared_expert", kind: "mlp", group: "layer", scaling: "weight",
    src: "vllm/model_executor/layers/fused_moe/layer.py:731",
    flops: 23.6e9, base: { bytes: 0.180e9, us: 51 } },
  { id: "experts", label: "mlp.experts · 8 of 256", kind: "moe", group: "layer", scaling: "moe",
    src: "vllm/model_executor/layers/fused_moe/fused_moe.py:1204",
    flops: 188.7e9, base: { bytes: 30.10e9, us: 5310 } },
  { id: "final_norm", label: "model.norm", kind: "norm", group: null, scaling: "fixed",
    src: "vllm/model_executor/layers/layernorm.py:112",
    flops: 0.01e9, base: { bytes: 0.001e9, us: 9 } },
  { id: "lm_head", label: "lm_head", kind: "head", group: null, scaling: "weight",
    src: "vllm/model_executor/layers/logits_processor.py:88",
    flops: 15.2e9, base: { bytes: 0.480e9, us: 86 } },
];

const BASE_GAPS = [ { k: "kernel launch gap", us: 310 },
                    { k: "host sync", us: 95 },
                    { k: "exposed collective", us: 180 } ];

/* efficiency implied by the baseline trace: what fraction of 8 TB/s
   each node actually achieved. The simulator holds this constant
   unless an option explicitly changes the kernel. */
const EFF = Object.fromEntries(NODES.map(n => {
  const achieved = n.base.bytes / (n.base.us / 1e6);
  return [n.id, achieved / (HW.bwTBs * 1e12)];
}));

const expertsTouched = b => MODEL.experts * (1 - Math.pow(1 - MODEL.active / MODEL.experts, b));

/* ---------- the analytical model ---------- */
function simulate(cfg) {
  const wb = BPP[cfg.weights], kvb = KVB[cfg.kv];
  const wScale = wb / BPP[BASE_CFG.weights];
  const tpScale = BASE_CFG.tp / cfg.tp;
  const touched = expertsTouched(cfg.batch);
  const touchedBase = expertsTouched(BASE_CFG.batch);

  const nodes = NODES.map(n => {
    let bytes = n.base.bytes, eff = EFF[n.id];

    if (n.scaling === "weight") bytes = n.base.bytes * wScale * tpScale;
    if (n.scaling === "kv")
      bytes = n.base.bytes * (kvb / KVB[BASE_CFG.kv]) * (cfg.batch / BASE_CFG.batch)
              * (cfg.ctx / BASE_CFG.ctx) * tpScale;
    if (n.scaling === "moe") {
      bytes = n.base.bytes * wScale * tpScale * (touched / touchedBase);
      if (cfg.ep) eff *= 1.16;                       // whole experts → longer contiguous reads
      if (cfg.moeKernel === "fused") eff *= 1.15;    // gate+up in one launch
    }
    if (n.scaling === "fixed") bytes = n.base.bytes * (cfg.batch / BASE_CFG.batch) * 0.15 + n.base.bytes * 0.85;

    /* smaller per-GPU tiles at higher TP lose some streaming efficiency */
    if (cfg.tp > BASE_CFG.tp && n.scaling !== "fixed") eff *= 0.92;

    const us = n.scaling === "fixed"
      ? n.base.us * (1 + 0.06 * Math.log2(cfg.batch / BASE_CFG.batch || 1))
      : (bytes / (HW.bwTBs * 1e12 * eff)) * 1e6;

    const predUs = (bytes / (HW.bwTBs * 1e12)) * 1e6;
    const flops = n.flops * (cfg.batch / BASE_CFG.batch) * tpScale;
    return { ...n, bytes, us, predUs, eff, flops,
             ai: bytes > 0 ? flops / bytes : 0,
             achievedBW: bytes / (us / 1e6) / 1e12,
             achievedTf: flops / (us / 1e6) / 1e12,
             roofPct: predUs / us };
  });

  const gaps = BASE_GAPS.map(g => {
    let us = g.us;
    if (g.k === "exposed collective") {
      us = cfg.ep ? us * 1.44 : us;                          // all-to-all costs more than all-reduce
      us *= cfg.tp / BASE_CFG.tp;
    }
    if (g.k === "kernel launch gap") us *= 1 + 0.04 * Math.log2(cfg.batch / BASE_CFG.batch || 1);
    return { ...g, us };
  });

  const nodeUs = nodes.reduce((s, n) => s + n.us, 0);
  const gapUs = gaps.reduce((s, g) => s + g.us, 0);
  const total = nodeUs + gapUs;

  /* memory: weights + KV cache resident per GPU */
  const weightGB = ((MODEL.expertParamsPerLayer + MODEL.attnParamsPerLayer) * MODEL.layers
                    + MODEL.headParams) * wb / cfg.tp / 1e9;
  const kvGB = cfg.batch * cfg.ctx * MODEL.kvPerTokenPerLayer * kvb * MODEL.layers / cfg.tp / 1e9;

  return { nodes, gaps, nodeUs, gapUs, total,
           throughput: cfg.batch / (total / 1e6),
           weightGB, kvGB, fits: weightGB + kvGB < HW.hbmGB, touched };
}

const fmtUs = v => v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${Math.round(v)} µs`;
const fmtB = v => v >= 1e9 ? `${(v / 1e9).toFixed(2)} GB` : `${(v / 1e6).toFixed(0)} MB`;
const pct = v => `${(v * 100).toFixed(1)}%`;
const RAMP = ["#2B4B7A", "#1E7F92", "#2FA383", "#C9A227", "#E0533D"];
function heat(f) {
  const x = Math.max(0, Math.min(1, f)) * (RAMP.length - 1);
  const i = Math.floor(x), t = x - i;
  if (i >= RAMP.length - 1) return RAMP[RAMP.length - 1];
  const a = RAMP[i].match(/\w\w/g).map(h => parseInt(h, 16));
  const b = RAMP[i + 1].match(/\w\w/g).map(h => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(",")})`;
}
const MARK = { embedding: "EMB", norm: "NRM", linear: "LIN", attention: "ATT",
               router: "RTR", mlp: "MLP", moe: "MOE", head: "HED" };

/* ================================================================== */

export default function PerfViewer() {
  const [tab, setTab] = useState("analyze");
  const [sel, setSel] = useState("experts");
  const [cfg, setCfg] = useState(BASE_CFG);

  const base = useMemo(() => simulate(BASE_CFG), []);
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const dirty = JSON.stringify(cfg) !== JSON.stringify(BASE_CFG);

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      <header style={S.mast}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
          <span style={S.wordmark}>perfviewer</span>
          <span style={S.mono2}>{MODEL.repo}</span>
          <span style={S.arrow}>on</span>
          <span style={S.mono2}>{HW.name}</span>
        </div>
        <div style={S.specStrip}>
          <Spec k="layers" v={MODEL.layers} />
          <Spec k="experts" v={`${MODEL.active}/${MODEL.experts}`} />
          <Spec k="peak BW" v={`${HW.bwTBs} TB/s`} />
          <Spec k="HBM" v={`${HW.hbmGB} GB`} />
          <Spec k="ridge" v={`${RIDGE.toFixed(0)} F/B`} />
          <Spec k="step" v={fmtUs(sim.total)} hot={dirty} />
          <Spec k="throughput" v={`${sim.throughput.toFixed(2)} tok/ms`} hot={dirty} />
        </div>
      </header>

      <div style={S.alert}>
        <span style={S.alertTag}>format</span>
        <span>Checkpoint is <b style={{ color: "#E0533D" }}>{MODEL.ckpt}</b>; this chip has native{" "}
          <b style={{ color: "#58D5FF" }}>MXFP4/MXFP6</b>. Different block size and scale encoding —
          requantize before tuning anything below.</span>
      </div>

      <nav style={S.nav}>
        {[["analyze", "analyze"], ["simulate", "simulate"], ["roofline", "roofline"],
          ["timeline", "timeline"]].map(([k, l]) => (
          <button key={k} onClick={() => setTab(k)} style={S.tab(tab === k)}>{l}</button>
        ))}
        <div style={{ flex: 1 }} />
        {dirty && (
          <button onClick={() => setCfg(BASE_CFG)} style={S.reset}>
            reset to measured baseline
          </button>
        )}
      </nav>

      {tab === "analyze"  && <Analyze  {...{ sim, sel, setSel }} />}
      {tab === "simulate" && <Simulate {...{ cfg, setCfg, sim, base, sel, setSel }} />}
      {tab === "roofline" && <Roofline {...{ sim, sel, setSel }} />}
      {tab === "timeline" && <Timeline {...{ sim, sel, setSel }} />}

      <footer style={S.foot}>
        prototype · synthetic measurements, real geometry · source locations are illustrative ·
        simulator holds measured per-node bandwidth efficiency constant unless a kernel option changes it
      </footer>
    </div>
  );
}

/* ================= ANALYZE — Talaria's split view ================= */

function Analyze({ sim, sel, setSel }) {
  const [sortKey, setSortKey] = useState("us");
  const [desc, setDesc] = useState(true);
  const [minUs, setMinUs] = useState(0);
  const [q, setQ] = useState("");
  const [metric, setMetric] = useState("share");

  const filtered = sim.nodes.filter(n =>
    n.us >= minUs && (q === "" || n.label.toLowerCase().includes(q.toLowerCase()) ||
                      n.kind.includes(q.toLowerCase())));
  const filteredIds = new Set(filtered.map(n => n.id));

  const sorted = [...filtered].sort((a, b) => {
    const v = { us: a.us - b.us, bytes: a.bytes - b.bytes, ai: a.ai - b.ai,
                bw: a.achievedBW - b.achievedBW, roof: a.roofPct - b.roofPct,
                label: a.label.localeCompare(b.label) }[sortKey];
    return desc ? -v : v;
  });

  const selNode = sim.nodes.find(n => n.id === sel);
  const COLS = [["label", "node"], ["us", "time"], ["bytes", "bytes"], ["ai", "AI"],
                ["bw", "TB/s"], ["roof", "% floor"]];

  return (
    <div style={S.split}>
      {/* ---- TABLE VIEW ---- */}
      <section style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>table view · {filtered.length} of {sim.nodes.length} operations</span>
        </div>
        <div style={S.toolbar}>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="search node or kind…"
                 style={S.input} />
          <label style={S.filterLabel}>
            hide under
            <input type="range" min="0" max="600" step="10" value={minUs}
                   onChange={e => setMinUs(+e.target.value)} style={{ width: 90 }} />
            <span style={{ ...S.mono, width: 52, color: minUs ? "#58D5FF" : "#5C6E80" }}>
              {minUs} µs
            </span>
          </label>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table style={S.table}>
            <thead>
              <tr>{COLS.map(([k, l]) => (
                <th key={k} style={{ ...S.th, cursor: "pointer",
                      color: sortKey === k ? "#58D5FF" : "#5C6E80" }}
                    onClick={() => { sortKey === k ? setDesc(!desc) : setSortKey(k); }}>
                  {l}{sortKey === k ? (desc ? " ↓" : " ↑") : ""}
                </th>))}
              </tr>
            </thead>
            <tbody>
              {sorted.map(n => {
                const on = sel === n.id;
                const share = n.us / sim.total;
                return (
                  <tr key={n.id} onClick={() => setSel(n.id)}
                      style={{ background: on ? "#132330" : "transparent", cursor: "pointer" }}>
                    <td style={S.td}>
                      <span style={{ ...S.mark, background: heat(Math.min(1, share / 0.5)) }}>
                        {MARK[n.kind]}</span>
                      <span style={{ color: on ? "#58D5FF" : "#C9D6E2", marginLeft: 7 }}>{n.label}</span>
                    </td>
                    <td style={S.tdN}>
                      <span style={S.sparkWrap}>
                        <span style={{ ...S.spark, width: `${Math.min(100, share * 180)}%`,
                                       background: heat(Math.min(1, share / 0.5)) }} />
                      </span>
                      {fmtUs(n.us)}
                    </td>
                    <td style={S.tdN}>{fmtB(n.bytes)}</td>
                    <td style={S.tdN}>{n.ai.toFixed(1)}</td>
                    <td style={{ ...S.tdN, color: heat(1 - n.achievedBW / HW.bwTBs) }}>
                      {n.achievedBW.toFixed(2)}</td>
                    <td style={{ ...S.tdN, color: heat(1 - n.roofPct) }}>{pct(n.roofPct)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {minUs > 0 && (
          <div style={S.crossNote}>
            Filter is cross-applied — the graph dims everything under {minUs} µs.
            This is the "find the top offenders" workflow.
          </div>
        )}
      </section>

      {/* ---- GRAPH VIEW ---- */}
      <aside style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>graph view</span>
          <div style={{ display: "flex", gap: 5 }}>
            {[["share", "share"], ["bw", "% BW"], ["roof", "gap"]].map(([k, l]) => (
              <button key={k} onClick={() => setMetric(k)} style={S.chip(metric === k)}>{l}</button>
            ))}
          </div>
        </div>

        <div style={S.flow}>
          {sim.nodes.filter(n => n.id === "embed_tokens").map(n =>
            <GNode key={n.id} n={n} sim={sim} sel={sel} setSel={setSel} metric={metric} on={filteredIds.has(n.id)} />)}

          <div style={S.group}>
            <div style={S.groupTab}>decoder layer <span style={S.repeat}>× {MODEL.layers}</span></div>
            {sim.nodes.filter(n => n.group === "layer").map(n =>
              <GNode key={n.id} n={n} sim={sim} sel={sel} setSel={setSel} metric={metric} on={filteredIds.has(n.id)} indent />)}
          </div>

          {sim.nodes.filter(n => !n.group && n.id !== "embed_tokens").map(n =>
            <GNode key={n.id} n={n} sim={sim} sel={sel} setSel={setSel} metric={metric} on={filteredIds.has(n.id)} />)}

          <div style={S.gapBlock}>
            <div style={S.gapHead}>unattributed · {fmtUs(sim.gapUs)} · {pct(sim.gapUs / sim.total)}</div>
            {sim.gaps.map(g => (
              <div key={g.k} style={S.gapRow}>
                <span style={{ color: "#7A8DA0" }}>{g.k}</span>
                <span style={S.mono}>{fmtUs(g.us)}</span>
              </div>
            ))}
          </div>
        </div>

        {selNode && <SourcePanel n={selNode} />}
      </aside>
    </div>
  );
}

function GNode({ n, sim, sel, setSel, metric, on, indent }) {
  const share = n.us / sim.total;
  const f = metric === "share" ? Math.min(1, share / 0.5)
          : metric === "bw" ? 1 - n.achievedBW / HW.bwTBs
          : 1 - n.roofPct;
  const active = sel === n.id;
  return (
    <button onClick={() => setSel(n.id)}
      style={{ ...S.node, marginLeft: indent ? 12 : 0, opacity: on ? 1 : 0.22,
               borderColor: active ? "#58D5FF" : "#232E3A",
               background: active ? "#132330" : "#141B23" }}>
      <span style={{ ...S.mark, background: heat(f) }}>{MARK[n.kind]}</span>
      <span style={S.nodeLabel}>{n.label}</span>
      <span style={S.nodeBarWrap}>
        <span style={{ ...S.nodeBar, width: `${Math.min(100, share * 190)}%`, background: heat(f) }} />
      </span>
      <span style={S.nodeTime}>{fmtUs(n.us)}</span>
    </button>
  );
}

/* Talaria T5 — attribute the operation back to the line that spawned it */
function SourcePanel({ n }) {
  const [file, line] = n.src.split(":");
  return (
    <div style={S.srcPanel}>
      <div style={S.srcHead}>source location · {n.label}</div>
      <div style={S.srcPath}>
        <span style={{ color: "#7A8DA0" }}>{file.split("/").slice(0, -1).join("/")}/</span>
        <span style={{ color: "#DCE7F1" }}>{file.split("/").pop()}</span>
        <span style={{ color: "#58D5FF" }}>:{line}</span>
      </div>
      <div style={S.srcNote}>
        Recovered by parsing the call stack at export, the way Talaria maps hardware tasks to code.
        Turns a finding into an edit.
      </div>
    </div>
  );
}

/* ================= SIMULATE — Talaria's novel contribution ================= */

function Simulate({ cfg, setCfg, sim, base, sel, setSel }) {
  const set = (k, v) => setCfg({ ...cfg, [k]: v });
  const dStep = sim.total - base.total;
  const dThr = sim.throughput - base.throughput;

  /* targeted options for the selected node — Talaria's per-op modal */
  const selNode = sim.nodes.find(n => n.id === sel);
  const targeted = useMemo(() => {
    const opts = [];
    if (selNode?.scaling === "moe") {
      opts.push({ label: "expert parallel instead of tensor parallel", apply: { ep: !cfg.ep } });
      opts.push({ label: "fused gate+up MoE kernel (SGLang-style)",
                  apply: { moeKernel: cfg.moeKernel === "fused" ? "triton" : "fused" } });
    }
    if (selNode?.scaling === "kv") {
      opts.push({ label: "KV cache fp8", apply: { kv: "fp8" } });
      opts.push({ label: "KV cache fp4", apply: { kv: "fp4" } });
      opts.push({ label: "KV cache bf16", apply: { kv: "bf16" } });
    }
    if (selNode?.scaling === "weight") {
      opts.push({ label: "weights MXFP4", apply: { weights: "mxfp4" } });
      opts.push({ label: "weights MXFP6", apply: { weights: "mxfp6" } });
      opts.push({ label: "weights MXFP8", apply: { weights: "mxfp8" } });
    }
    return opts.map(o => {
      const next = simulate({ ...cfg, ...o.apply });
      const nn = next.nodes.find(x => x.id === sel);
      return { ...o, dNode: nn.us - selNode.us, dStep: next.total - sim.total,
               dThr: next.throughput - sim.throughput };
    });
  }, [cfg, sel, selNode, sim]);

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>model-wide configuration</span></div>
        <div style={{ padding: "14px 18px" }}>
          <Opt label="batch size" note="decode arithmetic intensity ≈ 2 × batch">
            {[16, 32, 64, 128, 256].map(b =>
              <Pill key={b} on={cfg.batch === b} onClick={() => set("batch", b)}>{b}</Pill>)}
          </Opt>
          <Opt label="tensor parallel" note="fewer GPUs = less collective, larger tiles">
            {[2, 4, 8].map(t => <Pill key={t} on={cfg.tp === t} onClick={() => set("tp", t)}>TP{t}</Pill>)}
          </Opt>
          <Opt label="weight format" note="halving bytes halves the bandwidth floor everywhere">
            {["mxfp4", "mxfp6", "mxfp8"].map(w =>
              <Pill key={w} on={cfg.weights === w} onClick={() => set("weights", w)}>{w}</Pill>)}
          </Opt>
          <Opt label="KV cache dtype" note="scales attention decode almost linearly">
            {["fp4", "fp8", "bf16"].map(k =>
              <Pill key={k} on={cfg.kv === k} onClick={() => set("kv", k)}>{k}</Pill>)}
          </Opt>
          <Opt label="expert placement" note="EP trades all-reduce for all-to-all, gains streaming efficiency">
            <Pill on={!cfg.ep} onClick={() => set("ep", false)}>TP experts</Pill>
            <Pill on={cfg.ep} onClick={() => set("ep", true)}>EP experts</Pill>
          </Opt>
          <Opt label="MoE kernel" note="fused gate+up halves launches per expert">
            <Pill on={cfg.moeKernel === "triton"} onClick={() => set("moeKernel", "triton")}>triton</Pill>
            <Pill on={cfg.moeKernel === "fused"} onClick={() => set("moeKernel", "fused")}>fused</Pill>
          </Opt>

          <div style={S.memBox}>
            <div style={S.memRow}>
              <span style={S.dim}>weights / GPU</span><span style={S.mono}>{sim.weightGB.toFixed(1)} GB</span>
            </div>
            <div style={S.memRow}>
              <span style={S.dim}>KV cache / GPU</span><span style={S.mono}>{sim.kvGB.toFixed(1)} GB</span>
            </div>
            <div style={S.memTrack}>
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.weightGB / HW.hbmGB * 100)}%`,
                            background: "#2B4B7A" }} />
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.kvGB / HW.hbmGB * 100)}%`,
                            background: "#1E7F92" }} />
            </div>
            <div style={{ ...S.dim, marginTop: 6, color: sim.fits ? "#2FA383" : "#E0533D" }}>
              {sim.fits
                ? `${(HW.hbmGB - sim.weightGB - sim.kvGB).toFixed(0)} GB headroom of ${HW.hbmGB} GB`
                : `over capacity by ${(sim.weightGB + sim.kvGB - HW.hbmGB).toFixed(0)} GB — will not load`}
            </div>
          </div>
        </div>
      </section>

      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>predicted impact</span></div>
        <div style={{ padding: "16px 18px" }}>
          <div style={S.simGrid}>
            <SimStat k="step time" a={base.total} b={sim.total} fmt={fmtUs} lowerBetter />
            <SimStat k="throughput" a={base.throughput} b={sim.throughput}
                     fmt={v => `${v.toFixed(2)} tok/ms`} />
            <SimStat k="experts touched" a={base.touched} b={sim.touched}
                     fmt={v => `${v.toFixed(0)} / 256`} lowerBetter />
            <SimStat k="bytes / step" a={base.nodes.reduce((s, n) => s + n.bytes, 0)}
                     b={sim.nodes.reduce((s, n) => s + n.bytes, 0)} fmt={fmtB} lowerBetter />
          </div>

          <div style={S.cfgHead}>per-node effect</div>
          {sim.nodes.map(n => {
            const b = base.nodes.find(x => x.id === n.id);
            const d = n.us - b.us;
            if (Math.abs(d) < 1) return null;
            const w = Math.min(46, Math.abs(d) / Math.max(...sim.nodes.map((x, i) =>
              Math.abs(x.us - base.nodes[i].us))) * 46);
            const col = d > 0 ? "#E0533D" : "#2FA383";
            return (
              <div key={n.id} style={S.wRow} onClick={() => setSel(n.id)}>
                <span style={{ ...S.wLabel, color: sel === n.id ? "#58D5FF" : "#A9BECE" }}>{n.label}</span>
                <div style={S.wTrack}>
                  <div style={S.wMid} />
                  <div style={{ position: "absolute", top: 5, height: 8, background: col, borderRadius: 1,
                                left: d > 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} />
                </div>
                <span style={{ ...S.mono, width: 72, textAlign: "right", color: col }}>
                  {d > 0 ? "+" : "−"}{fmtUs(Math.abs(d))}
                </span>
              </div>
            );
          })}

          {targeted.length > 0 && (
            <>
              <div style={S.cfgHead}>targeted · {selNode.label}</div>
              <div style={S.tOpts}>
                <div style={S.tHead}><span>option</span><span>this node</span><span>step</span></div>
                {targeted.map(o => (
                  <button key={o.label} onClick={() => setCfg({ ...cfg, ...o.apply })} style={S.tRow}>
                    <span style={{ textAlign: "left" }}>{o.label}</span>
                    <span style={{ ...S.mono, color: o.dNode > 0 ? "#E0533D" : o.dNode < 0 ? "#2FA383" : "#5C6E80" }}>
                      {o.dNode === 0 ? "—" : `${o.dNode > 0 ? "+" : "−"}${fmtUs(Math.abs(o.dNode))}`}
                    </span>
                    <span style={{ ...S.mono, color: o.dStep > 0 ? "#E0533D" : o.dStep < 0 ? "#2FA383" : "#5C6E80" }}>
                      {o.dStep === 0 ? "—" : `${o.dStep > 0 ? "+" : "−"}${fmtUs(Math.abs(o.dStep))}`}
                    </span>
                  </button>
                ))}
              </div>
            </>
          )}

          <div style={S.caveat}>
            Predictions hold each node's <b>measured</b> bandwidth efficiency constant. They are directional
            for anything that changes kernel shape, and should be confirmed by a real run — Talaria's
            estimates landed within 1–3% of hardware, which is the bar to hold this to.
          </div>
        </div>
      </aside>
    </div>
  );
}

const Opt = ({ label, note, children }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={S.optLabel}>{label}</div>
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap", margin: "6px 0 4px" }}>{children}</div>
    <div style={S.optNote}>{note}</div>
  </div>
);
const Pill = ({ on, onClick, children }) => (
  <button onClick={onClick} style={S.pill(on)}>{children}</button>
);
function SimStat({ k, a, b, fmt, lowerBetter }) {
  const d = b - a, rel = a ? d / a : 0;
  const good = lowerBetter ? d < 0 : d > 0;
  const col = Math.abs(rel) < 0.005 ? "#5C6E80" : good ? "#2FA383" : "#E0533D";
  return (
    <div style={S.simStat}>
      <div style={S.kvK}>{k}</div>
      <div style={{ ...S.mono, fontSize: 15, color: "#DCE7F1", marginTop: 3 }}>{fmt(b)}</div>
      <div style={{ ...S.mono, fontSize: 11, color: col, marginTop: 2 }}>
        {Math.abs(rel) < 0.005 ? "unchanged" : `${d > 0 ? "+" : ""}${(rel * 100).toFixed(1)}%`}
      </div>
    </div>
  );
}

/* ================= ROOFLINE ================= */

function Roofline({ sim, sel, setSel }) {
  const W = 720, H = 400, P = { l: 58, r: 20, t: 20, b: 44 };
  const xMin = 1, xMax = 5000, yMin = 1, yMax = 12000;
  const lx = v => P.l + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (W - P.l - P.r);
  const ly = v => H - P.b - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - P.t - P.b);
  const roof = ai => Math.min(HW.peakFp8, ai * HW.bwTBs);
  const path = Array.from({ length: 101 }, (_, e) => {
    const ai = xMin * Math.pow(xMax / xMin, e / 100);
    return `${e ? "L" : "M"}${lx(ai)},${ly(Math.max(yMin, roof(ai)))}`;
  }).join(" ");
  const pts = sim.nodes.filter(n => n.ai > 0.5 && n.achievedTf > 0.5);

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>roofline · decode</span></div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          {[1, 10, 100, 1000].map(g => (
            <g key={g}>
              <line x1={lx(g)} y1={P.t} x2={lx(g)} y2={H - P.b} stroke="#1D2733" />
              <text x={lx(g)} y={H - P.b + 15} fill="#5C6E80" fontSize="10" textAnchor="middle" fontFamily="monospace">{g}</text>
            </g>))}
          {[1, 10, 100, 1000, 10000].map(g => (
            <g key={g}>
              <line x1={P.l} y1={ly(g)} x2={W - P.r} y2={ly(g)} stroke="#1D2733" />
              <text x={P.l - 7} y={ly(g) + 3} fill="#5C6E80" fontSize="10" textAnchor="end" fontFamily="monospace">{g}</text>
            </g>))}
          <rect x={P.l} y={P.t} width={lx(RIDGE) - P.l} height={H - P.t - P.b} fill="#E0533D" opacity="0.045" />
          <path d={path} fill="none" stroke="#58D5FF" strokeWidth="1.5" opacity="0.85" />
          <line x1={lx(RIDGE)} y1={P.t} x2={lx(RIDGE)} y2={H - P.b} stroke="#E0533D" strokeDasharray="3 3" />
          <text x={lx(RIDGE) + 6} y={P.t + 12} fill="#E0533D" fontSize="10" fontFamily="monospace">
            ridge {RIDGE.toFixed(0)}</text>
          {pts.map(p => {
            const on = sel === p.id;
            return (
              <g key={p.id} onClick={() => setSel(p.id)} style={{ cursor: "pointer" }}>
                <circle cx={lx(p.ai)} cy={ly(p.achievedTf)} r={on ? 7 : 4.5} fill={heat(1 - p.roofPct)}
                        stroke={on ? "#58D5FF" : "#0F141B"} strokeWidth={on ? 2 : 1} />
                {(on || p.kind === "moe" || p.kind === "attention") && (
                  <text x={lx(p.ai) + 10} y={ly(p.achievedTf) + 3} fill={on ? "#58D5FF" : "#8FA3B5"}
                        fontSize="10" fontFamily="monospace">{p.id}</text>)}
              </g>);
          })}
          <text x={W / 2} y={H - 5} fill="#5C6E80" fontSize="10" textAnchor="middle" fontFamily="monospace">
            arithmetic intensity — FLOP / byte</text>
        </svg>
      </section>
      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>reading</span></div>
        <div style={{ padding: "16px 18px" }}>
          <p style={S.pBig}>
            Every node sits left of the ridge at <b style={{ color: "#E0533D" }}>{RIDGE.toFixed(0)} FLOP/byte</b>.
            Bytes are the currency; FLOPs are free.
          </p>
          <p style={S.p}>
            Change the batch size on the simulate tab and watch the points move right. That horizontal motion
            is the only thing that changes which resource limits you — no kernel rewrite does it.
          </p>
          <div style={S.tableMini}>
            <div style={S.tmHead}><span>node</span><span>AI</span><span>× below ridge</span></div>
            {[...pts].sort((a, b) => a.ai - b.ai).slice(0, 5).map(p => (
              <div key={p.id} style={S.tmRow} onClick={() => setSel(p.id)}>
                <span style={{ color: sel === p.id ? "#58D5FF" : "#C9D6E2" }}>{p.id}</span>
                <span style={S.mono}>{p.ai.toFixed(1)}</span>
                <span style={{ ...S.mono, color: heat(Math.min(1, (RIDGE / p.ai) / 120)) }}>
                  {(RIDGE / p.ai).toFixed(0)}×</span>
              </div>))}
          </div>
        </div>
      </aside>
    </div>
  );
}

/* ================= TIMELINE — Talaria's complementary view ================= */

function Timeline({ sim, sel, setSel }) {
  const seq = [];
  let t = 0;
  const push = (id, label, us, kind) => { seq.push({ id, label, t0: t, us, kind }); t += us; };
  push("embed_tokens", "embed", sim.nodes.find(n => n.id === "embed_tokens").us, "embedding");
  sim.nodes.filter(n => n.group === "layer").forEach(n => push(n.id, n.label, n.us, n.kind));
  seq.push({ id: "gap", label: "launch gaps + sync + collective", t0: t, us: sim.gapUs, kind: "gap" });
  t += sim.gapUs;
  sim.nodes.filter(n => !n.group && n.id !== "embed_tokens").forEach(n => push(n.id, n.label, n.us, n.kind));
  const span = t;

  return (
    <div style={{ ...S.panel, margin: "14px 20px 20px" }}>
      <div style={S.panelHead}>
        <span style={S.eyebrow}>execution timeline · one decode step · {fmtUs(span)}</span>
      </div>
      <div style={{ padding: "16px 18px" }}>
        <div style={S.tlBar}>
          {seq.map((s, i) => (
            <div key={i} onClick={() => s.id !== "gap" && setSel(s.id)}
                 title={`${s.label} · ${fmtUs(s.us)}`}
                 style={{ width: `${(s.us / span) * 100}%`,
                          background: s.kind === "gap" ? "repeating-linear-gradient(45deg,#C9A227,#C9A227 3px,#1A1810 3px,#1A1810 6px)"
                                    : heat(Math.min(1, (s.us / span) / 0.5)),
                          height: "100%",
                          outline: sel === s.id ? "2px solid #58D5FF" : "none",
                          outlineOffset: -2, cursor: "pointer" }} />
          ))}
        </div>
        <div style={S.tlLegend}>
          <span style={S.dim}>0</span><span style={{ flex: 1 }} /><span style={S.dim}>{fmtUs(span)}</span>
        </div>
        <div style={S.tlRows}>
          {seq.filter(s => s.us / span > 0.005).sort((a, b) => b.us - a.us).map((s, i) => (
            <div key={i} style={{ ...S.tlRow, color: sel === s.id ? "#58D5FF" : "#A9BECE" }}
                 onClick={() => s.id !== "gap" && setSel(s.id)}>
              <span style={{ width: 14, height: 10, borderRadius: 1, flexShrink: 0,
                             background: s.kind === "gap" ? "#C9A227" : heat(Math.min(1, (s.us / span) / 0.5)) }} />
              <span style={{ flex: 1 }}>{s.label}</span>
              <span style={S.mono}>{fmtUs(s.us)}</span>
              <span style={{ ...S.mono, width: 54, textAlign: "right", color: "#5C6E80" }}>
                {pct(s.us / span)}</span>
            </div>))}
        </div>
        <p style={{ ...S.p, marginTop: 16, maxWidth: 720 }}>
          The striped block is time that belongs to no node. On a memory-bound decode loop it is routinely
          the second largest entry, and it is invisible on any view that only colors the graph.
        </p>
      </div>
    </div>
  );
}

/* ================= chrome ================= */

const Spec = ({ k, v, hot }) => (
  <span style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={S.specK}>{k}</span>
    <span style={{ ...S.specV, color: hot ? "#58D5FF" : "#A9BECE" }}>{v}</span>
  </span>
);

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&display=swap');
*{box-sizing:border-box}
button{font-family:inherit;cursor:pointer;background:none;border:none;color:inherit}
input[type=range]{accent-color:#58D5FF}
button:focus-visible,input:focus-visible{outline:2px solid #58D5FF;outline-offset:2px}
tbody tr:hover{background:#151E28 !important}
@media(max-width:920px){.pv-split{grid-template-columns:1fr !important}}
`;
const mono = `'IBM Plex Mono',ui-monospace,Menlo,monospace`;
const sans = `'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif`;
const cond = `'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif`;

const S = {
  root: { background: "#0F141B", color: "#C9D6E2", fontFamily: sans, minHeight: "100vh", fontSize: 13 },
  mast: { padding: "20px 20px 12px", borderBottom: "1px solid #1D2733" },
  wordmark: { fontFamily: cond, fontSize: 20, letterSpacing: ".14em", textTransform: "uppercase", color: "#58D5FF" },
  mono2: { fontFamily: mono, fontSize: 13, color: "#DCE7F1" },
  arrow: { color: "#5C6E80", fontSize: 12 },
  specStrip: { display: "flex", gap: 18, flexWrap: "wrap", marginTop: 12 },
  specK: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".13em", textTransform: "uppercase", color: "#5C6E80" },
  specV: { fontFamily: mono, fontSize: 12.5 },
  alert: { display: "flex", gap: 12, margin: "14px 20px 0", padding: "11px 14px", background: "#1A1520",
           border: "1px solid #E0533D33", borderRadius: 3, fontSize: 12.5, lineHeight: 1.55 },
  alertTag: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".13em", textTransform: "uppercase",
              color: "#E0533D", paddingTop: 3, flexShrink: 0 },
  nav: { display: "flex", gap: 4, alignItems: "center", padding: "14px 20px 0" },
  tab: on => ({ borderBottom: `2px solid ${on ? "#58D5FF" : "transparent"}`, color: on ? "#DCE7F1" : "#6E8095",
                fontFamily: cond, fontSize: 12, letterSpacing: ".1em", textTransform: "uppercase", padding: "6px 10px" }),
  reset: { fontFamily: mono, fontSize: 11, color: "#C9A227", border: "1px solid #C9A22733",
           padding: "4px 9px", borderRadius: 2 },
  split: { display: "grid", gridTemplateColumns: "1.25fr 1fr", gap: 14, padding: "14px 20px 20px",
           alignItems: "start", className: "pv-split" },
  panel: { background: "#121921", border: "1px solid #1D2733", borderRadius: 3 },
  panelHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10,
               padding: "10px 14px", borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  eyebrow: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase", color: "#6E8095" },
  chip: on => ({ background: on ? "#16303D" : "transparent", border: `1px solid ${on ? "#58D5FF66" : "#232E3A"}`,
                 color: on ? "#58D5FF" : "#6E8095", fontFamily: mono, fontSize: 10.5, padding: "3px 8px", borderRadius: 2 }),
  toolbar: { display: "flex", gap: 12, alignItems: "center", padding: "9px 14px",
             borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  input: { background: "#0F141B", border: "1px solid #232E3A", color: "#C9D6E2", fontFamily: mono,
           fontSize: 11.5, padding: "5px 8px", borderRadius: 2, flex: 1, minWidth: 130 },
  filterLabel: { display: "flex", alignItems: "center", gap: 7, fontFamily: cond, fontSize: 10,
                 letterSpacing: ".1em", textTransform: "uppercase", color: "#6E8095" },
  table: { width: "100%", borderCollapse: "collapse" },
  th: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".12em", textTransform: "uppercase",
        textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #1D2733", whiteSpace: "nowrap" },
  td: { fontFamily: mono, fontSize: 11.5, padding: "7px 10px", borderBottom: "1px solid #161E27", whiteSpace: "nowrap" },
  tdN: { fontFamily: mono, fontSize: 11.5, padding: "7px 10px", borderBottom: "1px solid #161E27",
         textAlign: "right", color: "#A9BECE", whiteSpace: "nowrap" },
  sparkWrap: { display: "inline-block", width: 40, height: 6, background: "#0F141B", marginRight: 7,
               borderRadius: 1, overflow: "hidden", verticalAlign: "middle" },
  spark: { display: "block", height: "100%" },
  crossNote: { padding: "9px 14px", fontFamily: mono, fontSize: 10.5, color: "#58D5FF",
               borderTop: "1px solid #1D2733", lineHeight: 1.5 },
  mark: { fontFamily: mono, fontSize: 9, fontWeight: 600, color: "#0F141B", padding: "2px 4px",
          borderRadius: 1, flexShrink: 0 },
  flow: { padding: "12px 14px", display: "flex", flexDirection: "column", gap: 3 },
  node: { display: "flex", alignItems: "center", gap: 8, width: "100%", textAlign: "left",
          border: "1px solid #232E3A", borderRadius: 2, padding: "6px 8px", transition: "opacity .15s" },
  nodeLabel: { fontFamily: mono, fontSize: 11, flex: "0 0 150px", overflow: "hidden",
               textOverflow: "ellipsis", whiteSpace: "nowrap" },
  nodeBarWrap: { flex: 1, height: 7, background: "#0F141B", borderRadius: 1, overflow: "hidden", minWidth: 26 },
  nodeBar: { display: "block", height: "100%" },
  nodeTime: { fontFamily: mono, fontSize: 10.5, width: 58, textAlign: "right", color: "#A9BECE" },
  group: { border: "1px dashed #2A3A48", borderRadius: 3, padding: "18px 7px 7px", margin: "8px 0",
           position: "relative", display: "flex", flexDirection: "column", gap: 3 },
  groupTab: { position: "absolute", top: -1, left: 10, transform: "translateY(-50%)", background: "#121921",
              padding: "0 8px", fontFamily: cond, fontSize: 10, letterSpacing: ".12em",
              textTransform: "uppercase", color: "#6E8095" },
  repeat: { color: "#58D5FF", fontFamily: mono, letterSpacing: 0 },
  gapBlock: { marginTop: 10, border: "1px solid #232E3A", borderRadius: 2, background: "#0F141B", padding: "8px 10px" },
  gapHead: { fontFamily: mono, fontSize: 10.5, color: "#C9A227", marginBottom: 6 },
  gapRow: { display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 11, padding: "2px 0" },
  srcPanel: { borderTop: "1px solid #1D2733", padding: "12px 14px", background: "#0F141B" },
  srcHead: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase",
             color: "#58D5FF", marginBottom: 7 },
  srcPath: { fontFamily: mono, fontSize: 11.5, wordBreak: "break-all" },
  srcNote: { fontFamily: sans, fontSize: 11.5, color: "#5C6E80", marginTop: 7, lineHeight: 1.5 },
  optLabel: { fontFamily: cond, fontSize: 10.5, letterSpacing: ".12em", textTransform: "uppercase", color: "#8FA3B5" },
  optNote: { fontFamily: sans, fontSize: 11, color: "#5C6E80", lineHeight: 1.45 },
  pill: on => ({ border: `1px solid ${on ? "#58D5FF" : "#232E3A"}`, background: on ? "#16303D" : "#0F141B",
                 color: on ? "#58D5FF" : "#7A8DA0", fontFamily: mono, fontSize: 11,
                 padding: "4px 10px", borderRadius: 2 }),
  memBox: { marginTop: 18, border: "1px solid #1D2733", borderRadius: 2, padding: "11px 12px", background: "#0F141B" },
  memRow: { display: "flex", justifyContent: "space-between", fontSize: 11.5, padding: "2px 0" },
  memTrack: { display: "flex", height: 8, background: "#1A2430", borderRadius: 1, marginTop: 8, overflow: "hidden" },
  memFill: { height: "100%" },
  simGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, background: "#1D2733",
             border: "1px solid #1D2733", borderRadius: 2 },
  simStat: { background: "#0F141B", padding: "9px 11px" },
  kvK: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".11em", textTransform: "uppercase", color: "#5C6E80" },
  cfgHead: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase",
             color: "#6E8095", margin: "18px 0 8px" },
  wRow: { display: "flex", alignItems: "center", gap: 8, padding: "3px 0", cursor: "pointer" },
  wLabel: { fontFamily: mono, fontSize: 10.5, flex: "0 0 150px", overflow: "hidden",
            textOverflow: "ellipsis", whiteSpace: "nowrap" },
  wTrack: { flex: 1, height: 18, position: "relative", minWidth: 60 },
  wMid: { position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#232E3A" },
  tOpts: { border: "1px solid #1D2733", borderRadius: 2, overflow: "hidden" },
  tHead: { display: "grid", gridTemplateColumns: "1fr 78px 78px", padding: "6px 10px",
           background: "#0F141B", borderBottom: "1px solid #1D2733", fontFamily: cond, fontSize: 9.5,
           letterSpacing: ".11em", textTransform: "uppercase", color: "#5C6E80" },
  tRow: { display: "grid", gridTemplateColumns: "1fr 78px 78px", padding: "7px 10px", width: "100%",
          fontFamily: sans, fontSize: 11.5, borderBottom: "1px solid #161E27", alignItems: "center", gap: 4 },
  caveat: { marginTop: 18, borderLeft: "2px solid #C9A227", paddingLeft: 11, fontSize: 11.5,
            lineHeight: 1.55, color: "#8FA3B5" },
  tlBar: { display: "flex", height: 44, borderRadius: 2, overflow: "hidden", border: "1px solid #232E3A" },
  tlLegend: { display: "flex", marginTop: 5 },
  tlRows: { marginTop: 16, display: "flex", flexDirection: "column", gap: 1 },
  tlRow: { display: "flex", alignItems: "center", gap: 9, padding: "5px 0", fontFamily: mono,
           fontSize: 11.5, cursor: "pointer", borderBottom: "1px solid #161E27" },
  tableMini: { marginTop: 16, border: "1px solid #1D2733", borderRadius: 2 },
  tmHead: { display: "grid", gridTemplateColumns: "1fr 60px 90px", padding: "6px 10px",
            borderBottom: "1px solid #1D2733", fontFamily: cond, fontSize: 9.5,
            letterSpacing: ".11em", textTransform: "uppercase", color: "#5C6E80" },
  tmRow: { display: "grid", gridTemplateColumns: "1fr 60px 90px", padding: "5px 10px",
           fontFamily: mono, fontSize: 11.5, cursor: "pointer" },
  p: { fontSize: 12.5, lineHeight: 1.6, color: "#A9BECE", margin: "0 0 9px" },
  pBig: { fontSize: 14, lineHeight: 1.55, color: "#DCE7F1", margin: "0 0 12px" },
  dim: { color: "#5C6E80", fontFamily: mono, fontSize: 10.5 },
  mono: { fontFamily: mono },
  foot: { padding: "14px 20px 26px", fontFamily: mono, fontSize: 10.5, color: "#3B4A5A",
          borderTop: "1px solid #1D2733", lineHeight: 1.6 },
};
