import React, { useState, useMemo, useRef, useCallback } from "react";

/* ==================================================================
 * PERFVIEWER v3 — GLM-5.2 on MI355X
 * v3 replaces the fake "graph" (a list with bars) with a real
 * node-link canvas: branch/merge topology, tensor shapes per node,
 * residual edges, zoom + pan, minimap, expandable layer stack.
 * ================================================================== */

const HW = { name: "AMD Instinct MI355X", hbmGB: 288, bwTBs: 8.0, peakFp8: 5000, peakFp4: 10100 };
const RIDGE = (HW.peakFp8 * 1e12) / (HW.bwTBs * 1e12);
const MODEL = { repo: "zai-org/GLM-5.2", layers: 78, hidden: 6144, experts: 256, active: 8,
                vocab: 154880, ckpt: "NVFP4", expertParamsPerLayer: 4.83e9,
                attnParamsPerLayer: 156e6, headParams: 951.6e6, kvPerTokenPerLayer: 576 };
const BPP = { mxfp4: .5, mxfp6: .75, mxfp8: 1 };
const KVB = { fp4: .5, fp8: 1, bf16: 2 };
const BASE_CFG = { batch: 32, ctx: 4096, tp: 4, ep: false, weights: "mxfp4", kv: "fp8", moeKernel: "triton" };

/* ---- geometry: x,y,w,h in graph space, plus output shape ---- */
const NODES = [
  { id: "embed_tokens", label: "embed_tokens", short: "embed", kind: "embedding", group: null,
    scaling: "fixed", flops: .4e9, base: { bytes: .012e9, us: 12 },
    src: "vllm/model_executor/layers/vocab_parallel_embedding.py:389",
    xy: [206, 16, 148, 36], shape: "32×1×6144" },

  { id: "input_norm", label: "input_layernorm", short: "RMSNorm", kind: "norm", group: "layer",
    scaling: "fixed", flops: .06e9, base: { bytes: .020e9, us: 41 },
    src: "vllm/model_executor/layers/layernorm.py:112",
    xy: [206, 96, 148, 36], shape: "32×1×6144" },

  { id: "q_a_proj", label: "self_attn.q_a_proj", short: "q_a_proj", kind: "linear", group: "layer",
    scaling: "weight", flops: 11.8e9, base: { bytes: .090e9, us: 24 },
    src: "vllm/model_executor/layers/linear.py:1043",
    xy: [64, 170, 130, 36], shape: "32×1×1536" },

  { id: "kv_a_proj", label: "self_attn.kv_a_proj", short: "kv_a_proj", kind: "linear", group: "layer",
    scaling: "weight", flops: 4.4e9, base: { bytes: .035e9, us: 18 },
    src: "vllm/model_executor/layers/linear.py:1043",
    xy: [252, 170, 130, 36], shape: "32×1×576" },

  { id: "q_b_proj", label: "self_attn.q_b_proj", short: "q_b_proj", kind: "linear", group: "layer",
    scaling: "weight", flops: 31.4e9, base: { bytes: .250e9, us: 48 },
    src: "vllm/model_executor/layers/linear.py:1043",
    xy: [64, 232, 130, 36], shape: "32×1×16384" },

  { id: "kv_b_proj", label: "self_attn.kv_b_proj", short: "kv_b_proj", kind: "linear", group: "layer",
    scaling: "weight", flops: 21e9, base: { bytes: .160e9, us: 39 },
    src: "vllm/model_executor/layers/linear.py:1043",
    xy: [252, 232, 130, 36], shape: "32×1×32768" },

  { id: "attn_core", label: "self_attn.core", short: "MLA attention", kind: "attention", group: "layer",
    scaling: "kv", flops: 167.5e9, base: { bytes: 1.470e9, us: 402 },
    src: "vllm/attention/backends/triton_mla.py:274",
    xy: [150, 300, 200, 42], shape: "32×1×16384" },

  { id: "o_proj", label: "self_attn.o_proj", short: "o_proj", kind: "linear", group: "layer",
    scaling: "weight", flops: 125.7e9, base: { bytes: .980e9, us: 168 },
    src: "vllm/model_executor/layers/linear.py:1043",
    xy: [176, 372, 148, 36], shape: "32×1×6144" },

  { id: "post_norm", label: "post_attention_layernorm", short: "RMSNorm", kind: "norm", group: "layer",
    scaling: "fixed", flops: .06e9, base: { bytes: .020e9, us: 40 },
    src: "vllm/model_executor/layers/layernorm.py:112",
    xy: [176, 486, 148, 36], shape: "32×1×6144" },

  { id: "router", label: "mlp.gate (router)", short: "router", kind: "router", group: "layer",
    scaling: "fixed", flops: 1.96e9, base: { bytes: .015e9, us: 88 },
    src: "vllm/model_executor/layers/fused_moe/layer.py:196",
    xy: [56, 556, 126, 36], shape: "32×1×256" },

  { id: "shared_expert", label: "mlp.shared_expert", short: "shared expert", kind: "mlp", group: "layer",
    scaling: "weight", flops: 23.6e9, base: { bytes: .180e9, us: 51 },
    src: "vllm/model_executor/layers/fused_moe/layer.py:731",
    xy: [300, 556, 132, 36], shape: "32×1×6144" },

  { id: "experts", label: "mlp.experts · 8 of 256", short: "experts  8/256", kind: "moe", group: "layer",
    scaling: "moe", flops: 188.7e9, base: { bytes: 30.10e9, us: 5310 },
    src: "vllm/model_executor/layers/fused_moe/fused_moe.py:1204",
    xy: [40, 620, 158, 44], shape: "32×1×6144" },

  { id: "final_norm", label: "model.norm", short: "RMSNorm", kind: "norm", group: null,
    scaling: "fixed", flops: .01e9, base: { bytes: .001e9, us: 9 },
    src: "vllm/model_executor/layers/layernorm.py:112",
    xy: [206, 806, 148, 36], shape: "32×1×6144" },

  { id: "lm_head", label: "lm_head", short: "lm_head", kind: "head", group: null,
    scaling: "weight", flops: 15.2e9, base: { bytes: .480e9, us: 86 },
    src: "vllm/model_executor/layers/logits_processor.py:88",
    xy: [206, 868, 148, 36], shape: "32×1×154880" },
];

/* merge points + the KV store: structural, not measured */
const OPS = [
  { id: "add1", label: "+", x: 265, y: 440, r: 15, title: "residual add (attention)" },
  { id: "add_moe", label: "+", x: 265, y: 700, r: 15, title: "expert sum" },
  { id: "add2", label: "+", x: 265, y: 754, r: 15, title: "residual add (MLP)" },
];
const KVSTORE = { x: 396, y: 300, w: 108, h: 42 };
const LAYER_BOX = { x: 24, y: 78, w: 500, h: 700 };
const RAIL = 470;   // residual rail x
const GW = 552, GH = 924;

const BASE_GAPS = [ { k: "kernel launch gap", us: 310 }, { k: "host sync", us: 95 },
                    { k: "exposed collective", us: 180 } ];

const EFF = Object.fromEntries(NODES.map(n =>
  [n.id, (n.base.bytes / (n.base.us / 1e6)) / (HW.bwTBs * 1e12)]));
const touchedFor = b => MODEL.experts * (1 - Math.pow(1 - MODEL.active / MODEL.experts, b));

function simulate(cfg) {
  const wScale = BPP[cfg.weights] / BPP[BASE_CFG.weights];
  const tpScale = BASE_CFG.tp / cfg.tp;
  const touched = touchedFor(cfg.batch), touchedBase = touchedFor(BASE_CFG.batch);
  const nodes = NODES.map(n => {
    let bytes = n.base.bytes, eff = EFF[n.id];
    if (n.scaling === "weight") bytes *= wScale * tpScale;
    if (n.scaling === "kv") bytes *= (KVB[cfg.kv] / KVB[BASE_CFG.kv]) * (cfg.batch / BASE_CFG.batch)
                                    * (cfg.ctx / BASE_CFG.ctx) * tpScale;
    if (n.scaling === "moe") {
      bytes *= wScale * tpScale * (touched / touchedBase);
      if (cfg.ep) eff *= 1.16;
      if (cfg.moeKernel === "fused") eff *= 1.15;
    }
    if (cfg.tp > BASE_CFG.tp && n.scaling !== "fixed") eff *= .92;
    const us = n.scaling === "fixed"
      ? n.base.us * (1 + .06 * Math.log2(cfg.batch / BASE_CFG.batch || 1))
      : (bytes / (HW.bwTBs * 1e12 * eff)) * 1e6;
    const predUs = (bytes / (HW.bwTBs * 1e12)) * 1e6;
    const flops = n.flops * (cfg.batch / BASE_CFG.batch) * tpScale;
    return { ...n, bytes, us, predUs, flops, eff,
             ai: bytes ? flops / bytes : 0,
             achievedBW: bytes / (us / 1e6) / 1e12,
             achievedTf: flops / (us / 1e6) / 1e12,
             roofPct: predUs / us,
             shape: n.shape.replace(/^32/, String(cfg.batch)) };
  });
  const gaps = BASE_GAPS.map(g => {
    let us = g.us;
    if (g.k === "exposed collective") us *= (cfg.ep ? 1.44 : 1) * (cfg.tp / BASE_CFG.tp);
    if (g.k === "kernel launch gap") us *= 1 + .04 * Math.log2(cfg.batch / BASE_CFG.batch || 1);
    return { ...g, us };
  });
  const nodeUs = nodes.reduce((s, n) => s + n.us, 0);
  const gapUs = gaps.reduce((s, g) => s + g.us, 0);
  const total = nodeUs + gapUs;
  const weightGB = ((MODEL.expertParamsPerLayer + MODEL.attnParamsPerLayer) * MODEL.layers
                    + MODEL.headParams) * BPP[cfg.weights] / cfg.tp / 1e9;
  const kvGB = cfg.batch * cfg.ctx * MODEL.kvPerTokenPerLayer * KVB[cfg.kv] * MODEL.layers / cfg.tp / 1e9;
  return { nodes, gaps, nodeUs, gapUs, total, touched, weightGB, kvGB,
           throughput: cfg.batch / (total / 1e6), fits: weightGB + kvGB < HW.hbmGB };
}

const fmtUs = v => v >= 1000 ? `${(v / 1000).toFixed(2)} ms` : `${Math.round(v)} µs`;
const fmtB = v => v >= 1e9 ? `${(v / 1e9).toFixed(2)} GB` : `${(v / 1e6).toFixed(0)} MB`;
const pct = v => `${(v * 100).toFixed(1)}%`;
const RAMP = ["#2B4B7A", "#1E7F92", "#2FA383", "#C9A227", "#E0533D"];
function heat(f) {
  const x = Math.max(0, Math.min(1, f)) * (RAMP.length - 1), i = Math.floor(x), t = x - i;
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
          <span style={S.arrow}>on</span><span style={S.mono2}>{HW.name}</span>
        </div>
        <div style={S.specStrip}>
          <Spec k="layers" v={MODEL.layers} /><Spec k="experts" v={`${MODEL.active}/${MODEL.experts}`} />
          <Spec k="peak BW" v={`${HW.bwTBs} TB/s`} /><Spec k="ridge" v={`${RIDGE.toFixed(0)} F/B`} />
          <Spec k="step" v={fmtUs(sim.total)} hot={dirty} />
          <Spec k="throughput" v={`${sim.throughput.toFixed(2)} tok/ms`} hot={dirty} />
        </div>
      </header>

      <div style={S.alert}><span style={S.alertTag}>format</span>
        <span>Checkpoint is <b style={{ color: "#E0533D" }}>{MODEL.ckpt}</b>; this chip has native{" "}
        <b style={{ color: "#58D5FF" }}>MXFP4/MXFP6</b> — different block size and scale encoding.</span></div>

      <nav style={S.nav}>
        {[["analyze", "analyze"], ["simulate", "simulate"], ["roofline", "roofline"]].map(([k, l]) =>
          <button key={k} onClick={() => setTab(k)} style={S.tab(tab === k)}>{l}</button>)}
        <div style={{ flex: 1 }} />
        {dirty && <button onClick={() => setCfg(BASE_CFG)} style={S.reset}>reset to baseline</button>}
      </nav>

      {tab === "analyze"  && <Analyze  {...{ sim, sel, setSel, cfg }} />}
      {tab === "simulate" && <Simulate {...{ cfg, setCfg, sim, base, sel, setSel }} />}
      {tab === "roofline" && <Roofline {...{ sim, sel, setSel }} />}

      <footer style={S.foot}>
        prototype · synthetic measurements, real geometry · edges follow the actual GLM-5.2 forward pass
      </footer>
    </div>
  );
}

/* ================= ANALYZE ================= */

function Analyze({ sim, sel, setSel, cfg }) {
  const [sortKey, setSortKey] = useState("us");
  const [desc, setDesc] = useState(true);
  const [minUs, setMinUs] = useState(0);
  const [q, setQ] = useState("");
  const [metric, setMetric] = useState("share");

  const filtered = sim.nodes.filter(n => n.us >= minUs &&
    (!q || n.label.toLowerCase().includes(q.toLowerCase()) || n.kind.includes(q.toLowerCase())));
  const visible = new Set(filtered.map(n => n.id));
  const sorted = [...filtered].sort((a, b) => {
    const v = { us: a.us - b.us, bytes: a.bytes - b.bytes, ai: a.ai - b.ai,
                bw: a.achievedBW - b.achievedBW, roof: a.roofPct - b.roofPct,
                label: a.label.localeCompare(b.label) }[sortKey];
    return desc ? -v : v;
  });
  const selNode = sim.nodes.find(n => n.id === sel);
  const COLS = [["label", "node"], ["us", "time"], ["bytes", "bytes"], ["ai", "AI"], ["roof", "% floor"]];

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}>
          <span style={S.eyebrow}>table view · {filtered.length} of {sim.nodes.length} ops</span>
        </div>
        <div style={S.toolbar}>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="search…" style={S.input} />
          <label style={S.filterLabel}>hide under
            <input type="range" min="0" max="600" step="10" value={minUs}
                   onChange={e => setMinUs(+e.target.value)} style={{ width: 80 }} />
            <span style={{ ...S.mono, width: 48, color: minUs ? "#58D5FF" : "#5C6E80" }}>{minUs}µs</span>
          </label>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={S.table}>
            <thead><tr>{COLS.map(([k, l]) =>
              <th key={k} style={{ ...S.th, cursor: "pointer", color: sortKey === k ? "#58D5FF" : "#5C6E80" }}
                  onClick={() => sortKey === k ? setDesc(!desc) : setSortKey(k)}>
                {l}{sortKey === k ? (desc ? " ↓" : " ↑") : ""}</th>)}</tr></thead>
            <tbody>{sorted.map(n => {
              const on = sel === n.id, share = n.us / sim.total;
              return (
                <tr key={n.id} onClick={() => setSel(n.id)}
                    style={{ background: on ? "#132330" : "transparent", cursor: "pointer" }}>
                  <td style={S.td}>
                    <span style={{ ...S.mark, background: heat(Math.min(1, share / .5)) }}>{MARK[n.kind]}</span>
                    <span style={{ color: on ? "#58D5FF" : "#C9D6E2", marginLeft: 7 }}>{n.label}</span>
                  </td>
                  <td style={S.tdN}>
                    <span style={S.sparkWrap}><span style={{ ...S.spark,
                      width: `${Math.min(100, share * 180)}%`, background: heat(Math.min(1, share / .5)) }} /></span>
                    {fmtUs(n.us)}</td>
                  <td style={S.tdN}>{fmtB(n.bytes)}</td>
                  <td style={S.tdN}>{n.ai.toFixed(1)}</td>
                  <td style={{ ...S.tdN, color: heat(1 - n.roofPct) }}>{pct(n.roofPct)}</td>
                </tr>);
            })}</tbody>
          </table>
        </div>
        {minUs > 0 && <div style={S.crossNote}>Cross-filtered — the canvas dims everything under {minUs} µs.</div>}
        {selNode && <SourcePanel n={selNode} />}
      </section>

      <GraphCanvas {...{ sim, sel, setSel, metric, setMetric, visible, cfg }} />
    </div>
  );
}

/* ================= THE GRAPH CANVAS ================= */

function GraphCanvas({ sim, sel, setSel, metric, setMetric, visible, cfg }) {
  const [view, setView] = useState({ k: 0.72, x: 40, y: 8 });
  const [expanded, setExpanded] = useState(false);
  const drag = useRef(null);
  const wrapRef = useRef(null);

  const byId = Object.fromEntries(sim.nodes.map(n => [n.id, n]));
  const heatOf = n => {
    const share = n.us / sim.total;
    return metric === "share" ? Math.min(1, share / .5)
         : metric === "bw" ? 1 - n.achievedBW / HW.bwTBs
         : 1 - n.roofPct;
  };

  const onWheel = useCallback(e => {
    e.preventDefault();
    const r = wrapRef.current.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    setView(v => {
      const k = Math.min(2.4, Math.max(.32, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      return { k, x: mx - (mx - v.x) * (k / v.k), y: my - (my - v.y) * (k / v.k) };
    });
  }, []);
  const onDown = e => { drag.current = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y }; };
  const onMove = e => {
    if (!drag.current) return;
    setView(v => ({ ...v, x: drag.current.ox + (e.clientX - drag.current.sx),
                          y: drag.current.oy + (e.clientY - drag.current.sy) }));
  };
  const onUp = () => { drag.current = null; };

  /* edge helpers -------------------------------------------------- */
  const bot = id => { const n = byId[id]; return [n.xy[0] + n.xy[2] / 2, n.xy[1] + n.xy[3]]; };
  const top = id => { const n = byId[id]; return [n.xy[0] + n.xy[2] / 2, n.xy[1]]; };
  const flow = (a, b) => {
    const [x1, y1] = bot(a), [x2, y2] = top(b), d = Math.max(14, (y2 - y1) * .45);
    return `M${x1},${y1} C${x1},${y1 + d} ${x2},${y2 - d} ${x2},${y2}`;
  };
  const toOp = (a, op) => {
    const [x1, y1] = bot(a), o = OPS.find(z => z.id === op);
    const d = Math.max(12, (o.y - o.r - y1) * .5);
    return `M${x1},${y1} C${x1},${y1 + d} ${o.x},${o.y - o.r - d} ${o.x},${o.y - o.r}`;
  };
  const fromOp = (op, b) => {
    const o = OPS.find(z => z.id === op), [x2, y2] = top(b);
    const d = Math.max(12, (y2 - (o.y + o.r)) * .5);
    return `M${o.x},${o.y + o.r} C${o.x},${o.y + o.r + d} ${x2},${y2 - d} ${x2},${y2}`;
  };
  const opToOp = (a, b) => {
    const A = OPS.find(z => z.id === a), B = OPS.find(z => z.id === b);
    return `M${A.x},${A.y + A.r} L${B.x},${B.y - B.r}`;
  };

  const FLOWS = [
    ["embed_tokens", "input_norm"], ["input_norm", "q_a_proj"], ["input_norm", "kv_a_proj"],
    ["q_a_proj", "q_b_proj"], ["kv_a_proj", "kv_b_proj"],
    ["q_b_proj", "attn_core"], ["kv_b_proj", "attn_core"],
    ["attn_core", "o_proj"], ["post_norm", "router"], ["post_norm", "shared_expert"],
    ["router", "experts"], ["final_norm", "lm_head"],
  ];

  return (
    <aside style={S.panel}>
      <div style={S.panelHead}>
        <span style={S.eyebrow}>graph view</span>
        <div style={{ display: "flex", gap: 5, alignItems: "center", flexWrap: "wrap" }}>
          {[["share", "share"], ["bw", "% BW"], ["roof", "gap"]].map(([k, l]) =>
            <button key={k} onClick={() => setMetric(k)} style={S.chip(metric === k)}>{l}</button>)}
          <button onClick={() => setExpanded(!expanded)} style={S.chip(expanded)}>
            {expanded ? "collapse stack" : "show full graph"}</button>
          <button onClick={() => setView({ k: .72, x: 40, y: 8 })} style={S.chip(false)}>fit</button>
        </div>
      </div>

      <div ref={wrapRef} style={S.canvasWrap} onWheel={onWheel} onPointerDown={onDown}
           onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
        <svg width="100%" height="100%" style={{ display: "block", cursor: drag.current ? "grabbing" : "grab" }}>
          <defs>
            <marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 z" fill="#3E5468" />
            </marker>
            <marker id="arR" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 z" fill="#5C7A94" />
            </marker>
            <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="0" y2="6" stroke="#C9A227" strokeWidth="2.5" opacity=".5" />
            </pattern>
          </defs>

          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
            {/* repeated-stack ghosts behind the layer box */}
            {expanded && [3, 2, 1].map(i => (
              <rect key={i} x={LAYER_BOX.x + i * 9} y={LAYER_BOX.y - i * 9}
                    width={LAYER_BOX.w} height={LAYER_BOX.h} rx="4"
                    fill="#101820" stroke="#22303C" strokeDasharray="3 3" opacity={.85 - i * .2} />
            ))}

            {/* layer group */}
            <rect x={LAYER_BOX.x} y={LAYER_BOX.y} width={LAYER_BOX.w} height={LAYER_BOX.h} rx="4"
                  fill="#0E141B" stroke="#2A3A48" strokeDasharray="4 3" />
            <rect x={LAYER_BOX.x + 12} y={LAYER_BOX.y - 9} width={168} height={18} rx="2" fill="#121921" />
            <text x={LAYER_BOX.x + 20} y={LAYER_BOX.y + 4} fill="#6E8095" fontSize="10"
                  fontFamily="monospace" letterSpacing="1">decoder layer
              <tspan fill="#58D5FF"> × {MODEL.layers}</tspan></text>

            {/* residual rail */}
            <path d={`M${340},${132} H${RAIL} V${440} H${280}`} fill="none" stroke="#5C7A94"
                  strokeWidth="1.3" strokeDasharray="5 4" markerEnd="url(#arR)" />
            <path d={`M${290},${440} H${RAIL - 26} V${754} H${280}`} fill="none" stroke="#5C7A94"
                  strokeWidth="1.3" strokeDasharray="5 4" markerEnd="url(#arR)" />
            <text x={RAIL + 6} y={286} fill="#5C7A94" fontSize="9" fontFamily="monospace">residual</text>

            {/* dataflow edges */}
            {FLOWS.map(([a, b]) => (
              <path key={a + b} d={flow(a, b)} fill="none" stroke="#3E5468" strokeWidth="1.3"
                    markerEnd="url(#ar)" opacity={visible.has(a) && visible.has(b) ? 1 : .25} />
            ))}
            <path d={toOp("o_proj", "add1")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />
            <path d={fromOp("add1", "post_norm")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />
            <path d={toOp("experts", "add_moe")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />
            <path d={toOp("shared_expert", "add_moe")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />
            <path d={opToOp("add_moe", "add2")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />
            <path d={fromOp("add2", "final_norm")} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />

            {/* routing fan-out: router decides which experts run */}
            <path d={`M${182},${574} C${210},${574} ${210},${620} ${198},${634}`} fill="none"
                  stroke="#C9A227" strokeWidth="1.2" strokeDasharray="2 3" />
            <text x={206} y={600} fill="#C9A227" fontSize="9" fontFamily="monospace">top-8</text>

            {/* KV cache store */}
            <g onClick={() => setSel("attn_core")} style={{ cursor: "pointer" }}>
              <rect x={KVSTORE.x} y={KVSTORE.y} width={KVSTORE.w} height={KVSTORE.h} rx="3"
                    fill="#12202A" stroke="#2A4656" strokeDasharray="3 2" />
              <text x={KVSTORE.x + 10} y={KVSTORE.y + 17} fill="#7FA8BF" fontSize="10" fontFamily="monospace">
                KV cache</text>
              <text x={KVSTORE.x + 10} y={KVSTORE.y + 31} fill="#4E6B7E" fontSize="9" fontFamily="monospace">
                {fmtB(sim.nodes.find(n => n.id === "attn_core").bytes)}/step</text>
              <path d={`M${KVSTORE.x},${KVSTORE.y + 21} H${350}`} fill="none" stroke="#2A4656"
                    strokeWidth="1.3" markerEnd="url(#ar)" />
            </g>

            {/* merge ops */}
            {OPS.map(o => (
              <g key={o.id}>
                <circle cx={o.x} cy={o.y} r={o.r} fill="#101820" stroke="#3E5468" strokeWidth="1.3" />
                <text x={o.x} y={o.y + 5} textAnchor="middle" fill="#8FA3B5" fontSize="15"
                      fontFamily="monospace">{o.label}</text>
              </g>
            ))}

            {/* nodes */}
            {sim.nodes.map(n => {
              const [x, y, w, h] = n.xy;
              const on = sel === n.id, dim = !visible.has(n.id);
              const c = heat(heatOf(n));
              return (
                <g key={n.id} onClick={e => { e.stopPropagation(); setSel(n.id); }}
                   opacity={dim ? .22 : 1} style={{ cursor: "pointer" }}>
                  <rect x={x} y={y} width={w} height={h} rx="3"
                        fill={on ? "#14262F" : "#141B23"} stroke={on ? "#58D5FF" : "#2A3644"}
                        strokeWidth={on ? 2 : 1} />
                  <rect x={x} y={y} width={4} height={h} rx="1.5" fill={c} />
                  <text x={x + 12} y={y + 15} fill={on ? "#58D5FF" : "#DCE7F1"} fontSize="10.5"
                        fontFamily="monospace">{n.short}</text>
                  <text x={x + 12} y={y + 27} fill="#5C7285" fontSize="8.5" fontFamily="monospace">
                    {n.shape}</text>
                  <text x={x + w - 8} y={y + 15} textAnchor="end" fill="#8FA3B5" fontSize="9.5"
                        fontFamily="monospace">{fmtUs(n.us)}</text>
                  <rect x={x + w - 44} y={y + h - 9} width={36} height={3} rx="1.5" fill="#0B1016" />
                  <rect x={x + w - 44} y={y + h - 9} width={36 * Math.min(1, n.us / sim.total / .5)}
                        height={3} rx="1.5" fill={c} />
                </g>
              );
            })}

            {/* gap block, drawn as part of the step */}
            <g>
              <rect x={40} y={GH - 40} width={472} height={26} rx="3" fill="url(#hatch)" opacity=".5" />
              <rect x={40} y={GH - 40} width={472} height={26} rx="3" fill="none" stroke="#C9A22755" />
              <text x={52} y={GH - 23} fill="#C9A227" fontSize="10" fontFamily="monospace">
                unattributed · {fmtUs(sim.gapUs)} · {pct(sim.gapUs / sim.total)} of step</text>
            </g>
          </g>
        </svg>

        {/* minimap */}
        <div style={S.minimap}>
          <svg viewBox={`0 0 ${GW} ${GH}`} width="100%" height="100%">
            <rect x="0" y="0" width={GW} height={GH} fill="#0B1016" />
            <rect x={LAYER_BOX.x} y={LAYER_BOX.y} width={LAYER_BOX.w} height={LAYER_BOX.h}
                  fill="none" stroke="#2A3A48" />
            {sim.nodes.map(n => (
              <rect key={n.id} x={n.xy[0]} y={n.xy[1]} width={n.xy[2]} height={n.xy[3]}
                    fill={heat(heatOf(n))} opacity={sel === n.id ? 1 : .65} />
            ))}
            <rect x={-view.x / view.k} y={-view.y / view.k}
                  width={(wrapRef.current?.clientWidth || 400) / view.k}
                  height={(wrapRef.current?.clientHeight || 560) / view.k}
                  fill="#58D5FF" opacity=".1" stroke="#58D5FF" strokeWidth="3" />
          </svg>
        </div>

        <div style={S.hint}>scroll to zoom · drag to pan</div>
      </div>

      <div style={S.legendRow}>
        <span style={S.dim}>cold</span>
        <div style={S.ramp} /><span style={S.dim}>hot</span>
        <span style={{ flex: 1 }} />
        <LegendKey color="#3E5468" label="dataflow" />
        <LegendKey color="#5C7A94" label="residual" dash />
        <LegendKey color="#C9A227" label="routing" dash />
      </div>
    </aside>
  );
}

const LegendKey = ({ color, label, dash }) => (
  <span style={{ display: "flex", alignItems: "center", gap: 5, marginLeft: 12 }}>
    <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={color} strokeWidth="1.6"
      strokeDasharray={dash ? "4 3" : "0"} /></svg>
    <span style={S.dim}>{label}</span>
  </span>
);

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
      <div style={S.srcNote}>Recovered by parsing the call stack at export — turns a finding into an edit.</div>
    </div>
  );
}

/* ================= SIMULATE ================= */

function Simulate({ cfg, setCfg, sim, base, sel, setSel }) {
  const set = (k, v) => setCfg({ ...cfg, [k]: v });
  const selNode = sim.nodes.find(n => n.id === sel);
  const targeted = useMemo(() => {
    const o = [];
    if (selNode?.scaling === "moe") {
      o.push({ label: "expert parallel instead of tensor parallel", apply: { ep: !cfg.ep } });
      o.push({ label: "fused gate+up MoE kernel", apply: { moeKernel: cfg.moeKernel === "fused" ? "triton" : "fused" } });
    }
    if (selNode?.scaling === "kv") ["fp4", "fp8", "bf16"].forEach(k =>
      o.push({ label: `KV cache ${k}`, apply: { kv: k } }));
    if (selNode?.scaling === "weight") ["mxfp4", "mxfp6", "mxfp8"].forEach(w =>
      o.push({ label: `weights ${w}`, apply: { weights: w } }));
    return o.map(x => {
      const nx = simulate({ ...cfg, ...x.apply });
      return { ...x, dNode: nx.nodes.find(z => z.id === sel).us - selNode.us, dStep: nx.total - sim.total };
    });
  }, [cfg, sel, selNode, sim]);

  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>model-wide configuration</span></div>
        <div style={{ padding: "14px 18px" }}>
          <Opt label="batch size" note="decode arithmetic intensity ≈ 2 × batch">
            {[16, 32, 64, 128, 256].map(b => <Pill key={b} on={cfg.batch === b} onClick={() => set("batch", b)}>{b}</Pill>)}</Opt>
          <Opt label="tensor parallel" note="fewer GPUs = less collective, larger tiles">
            {[2, 4, 8].map(t => <Pill key={t} on={cfg.tp === t} onClick={() => set("tp", t)}>TP{t}</Pill>)}</Opt>
          <Opt label="weight format" note="halving bytes halves the bandwidth floor everywhere">
            {["mxfp4", "mxfp6", "mxfp8"].map(w => <Pill key={w} on={cfg.weights === w} onClick={() => set("weights", w)}>{w}</Pill>)}</Opt>
          <Opt label="KV cache dtype" note="scales attention decode almost linearly">
            {["fp4", "fp8", "bf16"].map(k => <Pill key={k} on={cfg.kv === k} onClick={() => set("kv", k)}>{k}</Pill>)}</Opt>
          <Opt label="expert placement" note="EP trades all-reduce for all-to-all, gains streaming efficiency">
            <Pill on={!cfg.ep} onClick={() => set("ep", false)}>TP experts</Pill>
            <Pill on={cfg.ep} onClick={() => set("ep", true)}>EP experts</Pill></Opt>
          <Opt label="MoE kernel" note="fused gate+up halves launches per expert">
            <Pill on={cfg.moeKernel === "triton"} onClick={() => set("moeKernel", "triton")}>triton</Pill>
            <Pill on={cfg.moeKernel === "fused"} onClick={() => set("moeKernel", "fused")}>fused</Pill></Opt>

          <div style={S.memBox}>
            <div style={S.memRow}><span style={S.dim}>weights / GPU</span><span style={S.mono}>{sim.weightGB.toFixed(1)} GB</span></div>
            <div style={S.memRow}><span style={S.dim}>KV cache / GPU</span><span style={S.mono}>{sim.kvGB.toFixed(1)} GB</span></div>
            <div style={S.memTrack}>
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.weightGB / HW.hbmGB * 100)}%`, background: "#2B4B7A" }} />
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.kvGB / HW.hbmGB * 100)}%`, background: "#1E7F92" }} />
            </div>
            <div style={{ ...S.dim, marginTop: 6, color: sim.fits ? "#2FA383" : "#E0533D" }}>
              {sim.fits ? `${(HW.hbmGB - sim.weightGB - sim.kvGB).toFixed(0)} GB headroom of ${HW.hbmGB} GB`
                        : `over capacity by ${(sim.weightGB + sim.kvGB - HW.hbmGB).toFixed(0)} GB`}</div>
          </div>
        </div>
      </section>

      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>predicted impact</span></div>
        <div style={{ padding: "16px 18px" }}>
          <div style={S.simGrid}>
            <SimStat k="step time" a={base.total} b={sim.total} fmt={fmtUs} lower />
            <SimStat k="throughput" a={base.throughput} b={sim.throughput} fmt={v => `${v.toFixed(2)} tok/ms`} />
            <SimStat k="experts touched" a={base.touched} b={sim.touched} fmt={v => `${v.toFixed(0)}/256`} lower />
            <SimStat k="bytes / step" a={base.nodes.reduce((s, n) => s + n.bytes, 0)}
                     b={sim.nodes.reduce((s, n) => s + n.bytes, 0)} fmt={fmtB} lower />
          </div>
          <div style={S.cfgHead}>per-node effect</div>
          {sim.nodes.map((n, i) => {
            const d = n.us - base.nodes[i].us;
            if (Math.abs(d) < 1) return null;
            const mx = Math.max(...sim.nodes.map((x, j) => Math.abs(x.us - base.nodes[j].us))) || 1;
            const w = Math.min(46, Math.abs(d) / mx * 46), col = d > 0 ? "#E0533D" : "#2FA383";
            return (
              <div key={n.id} style={S.wRow} onClick={() => setSel(n.id)}>
                <span style={{ ...S.wLabel, color: sel === n.id ? "#58D5FF" : "#A9BECE" }}>{n.label}</span>
                <div style={S.wTrack}><div style={S.wMid} />
                  <div style={{ position: "absolute", top: 5, height: 8, background: col, borderRadius: 1,
                                left: d > 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} /></div>
                <span style={{ ...S.mono, width: 70, textAlign: "right", color: col }}>
                  {d > 0 ? "+" : "−"}{fmtUs(Math.abs(d))}</span>
              </div>);
          })}
          {targeted.length > 0 && (<>
            <div style={S.cfgHead}>targeted · {selNode.label}</div>
            <div style={S.tOpts}>
              <div style={S.tHead}><span>option</span><span>node</span><span>step</span></div>
              {targeted.map(o => (
                <button key={o.label} onClick={() => setCfg({ ...cfg, ...o.apply })} style={S.tRow}>
                  <span style={{ textAlign: "left" }}>{o.label}</span>
                  <span style={{ ...S.mono, color: o.dNode > 0 ? "#E0533D" : o.dNode < 0 ? "#2FA383" : "#5C6E80" }}>
                    {o.dNode ? `${o.dNode > 0 ? "+" : "−"}${fmtUs(Math.abs(o.dNode))}` : "—"}</span>
                  <span style={{ ...S.mono, color: o.dStep > 0 ? "#E0533D" : o.dStep < 0 ? "#2FA383" : "#5C6E80" }}>
                    {o.dStep ? `${o.dStep > 0 ? "+" : "−"}${fmtUs(Math.abs(o.dStep))}` : "—"}</span>
                </button>))}
            </div></>)}
          <div style={S.caveat}>Predictions hold each node's measured bandwidth efficiency constant —
            directional for anything that changes kernel shape, confirm with a real run.</div>
        </div>
      </aside>
    </div>
  );
}

const Opt = ({ label, note, children }) => (
  <div style={{ marginBottom: 15 }}>
    <div style={S.optLabel}>{label}</div>
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap", margin: "6px 0 4px" }}>{children}</div>
    <div style={S.optNote}>{note}</div></div>);
const Pill = ({ on, onClick, children }) => <button onClick={onClick} style={S.pill(on)}>{children}</button>;
function SimStat({ k, a, b, fmt, lower }) {
  const d = b - a, rel = a ? d / a : 0, good = lower ? d < 0 : d > 0;
  const col = Math.abs(rel) < .005 ? "#5C6E80" : good ? "#2FA383" : "#E0533D";
  return (<div style={S.simStat}><div style={S.kvK}>{k}</div>
    <div style={{ ...S.mono, fontSize: 15, color: "#DCE7F1", marginTop: 3 }}>{fmt(b)}</div>
    <div style={{ ...S.mono, fontSize: 11, color: col, marginTop: 2 }}>
      {Math.abs(rel) < .005 ? "unchanged" : `${d > 0 ? "+" : ""}${(rel * 100).toFixed(1)}%`}</div></div>);
}

/* ================= ROOFLINE ================= */

function Roofline({ sim, sel, setSel }) {
  const W = 720, H = 400, P = { l: 58, r: 20, t: 20, b: 44 };
  const xMin = 1, xMax = 5000, yMin = 1, yMax = 12000;
  const lx = v => P.l + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (W - P.l - P.r);
  const ly = v => H - P.b - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - P.t - P.b);
  const path = Array.from({ length: 101 }, (_, e) => {
    const ai = xMin * Math.pow(xMax / xMin, e / 100);
    return `${e ? "L" : "M"}${lx(ai)},${ly(Math.max(yMin, Math.min(HW.peakFp8, ai * HW.bwTBs)))}`;
  }).join(" ");
  const pts = sim.nodes.filter(n => n.ai > .5 && n.achievedTf > .5);
  return (
    <div style={S.split}>
      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>roofline · decode</span></div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
          {[1, 10, 100, 1000].map(g => (<g key={g}>
            <line x1={lx(g)} y1={P.t} x2={lx(g)} y2={H - P.b} stroke="#1D2733" />
            <text x={lx(g)} y={H - P.b + 15} fill="#5C6E80" fontSize="10" textAnchor="middle" fontFamily="monospace">{g}</text></g>))}
          {[1, 10, 100, 1000, 10000].map(g => (<g key={g}>
            <line x1={P.l} y1={ly(g)} x2={W - P.r} y2={ly(g)} stroke="#1D2733" />
            <text x={P.l - 7} y={ly(g) + 3} fill="#5C6E80" fontSize="10" textAnchor="end" fontFamily="monospace">{g}</text></g>))}
          <rect x={P.l} y={P.t} width={lx(RIDGE) - P.l} height={H - P.t - P.b} fill="#E0533D" opacity=".045" />
          <path d={path} fill="none" stroke="#58D5FF" strokeWidth="1.5" opacity=".85" />
          <line x1={lx(RIDGE)} y1={P.t} x2={lx(RIDGE)} y2={H - P.b} stroke="#E0533D" strokeDasharray="3 3" />
          <text x={lx(RIDGE) + 6} y={P.t + 12} fill="#E0533D" fontSize="10" fontFamily="monospace">ridge {RIDGE.toFixed(0)}</text>
          {pts.map(p => { const on = sel === p.id; return (
            <g key={p.id} onClick={() => setSel(p.id)} style={{ cursor: "pointer" }}>
              <circle cx={lx(p.ai)} cy={ly(p.achievedTf)} r={on ? 7 : 4.5} fill={heat(1 - p.roofPct)}
                      stroke={on ? "#58D5FF" : "#0F141B"} strokeWidth={on ? 2 : 1} />
              {(on || p.kind === "moe" || p.kind === "attention") &&
                <text x={lx(p.ai) + 10} y={ly(p.achievedTf) + 3} fill={on ? "#58D5FF" : "#8FA3B5"}
                      fontSize="10" fontFamily="monospace">{p.id}</text>}</g>); })}
          <text x={W / 2} y={H - 5} fill="#5C6E80" fontSize="10" textAnchor="middle" fontFamily="monospace">
            arithmetic intensity — FLOP / byte</text>
        </svg>
      </section>
      <aside style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>reading</span></div>
        <div style={{ padding: "16px 18px" }}>
          <p style={S.pBig}>Every node sits left of the ridge at{" "}
            <b style={{ color: "#E0533D" }}>{RIDGE.toFixed(0)} FLOP/byte</b>. Bytes are the currency.</p>
          <p style={S.p}>Change batch size on the simulate tab and watch the points slide right. That
            horizontal motion is the only thing that changes which resource limits you.</p>
          <div style={S.tableMini}>
            <div style={S.tmHead}><span>node</span><span>AI</span><span>× below</span></div>
            {[...pts].sort((a, b) => a.ai - b.ai).slice(0, 5).map(p => (
              <div key={p.id} style={S.tmRow} onClick={() => setSel(p.id)}>
                <span style={{ color: sel === p.id ? "#58D5FF" : "#C9D6E2" }}>{p.id}</span>
                <span style={S.mono}>{p.ai.toFixed(1)}</span>
                <span style={{ ...S.mono, color: heat(Math.min(1, (RIDGE / p.ai) / 120)) }}>
                  {(RIDGE / p.ai).toFixed(0)}×</span></div>))}
          </div>
        </div>
      </aside>
    </div>
  );
}

const Spec = ({ k, v, hot }) => (
  <span style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={S.specK}>{k}</span>
    <span style={{ ...S.specV, color: hot ? "#58D5FF" : "#A9BECE" }}>{v}</span></span>);

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
  reset: { fontFamily: mono, fontSize: 11, color: "#C9A227", border: "1px solid #C9A22733", padding: "4px 9px", borderRadius: 2 },
  split: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, padding: "14px 20px 20px",
           alignItems: "start", className: "pv-split" },
  panel: { background: "#121921", border: "1px solid #1D2733", borderRadius: 3 },
  panelHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10,
               padding: "10px 14px", borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  eyebrow: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase", color: "#6E8095" },
  chip: on => ({ background: on ? "#16303D" : "transparent", border: `1px solid ${on ? "#58D5FF66" : "#232E3A"}`,
                 color: on ? "#58D5FF" : "#6E8095", fontFamily: mono, fontSize: 10.5, padding: "3px 8px", borderRadius: 2 }),
  canvasWrap: { position: "relative", height: 580, overflow: "hidden", background: "#0B1016",
                borderBottom: "1px solid #1D2733", touchAction: "none" },
  minimap: { position: "absolute", left: 10, bottom: 10, width: 76, height: 128, border: "1px solid #26333F",
             borderRadius: 2, overflow: "hidden", background: "#0B1016", opacity: .92 },
  hint: { position: "absolute", right: 10, bottom: 10, fontFamily: mono, fontSize: 9.5, color: "#3E5468" },
  legendRow: { display: "flex", alignItems: "center", gap: 7, padding: "8px 14px", flexWrap: "wrap" },
  ramp: { width: 70, height: 6, borderRadius: 1, background: `linear-gradient(90deg,${RAMP.join(",")})` },
  toolbar: { display: "flex", gap: 12, alignItems: "center", padding: "9px 14px",
             borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  input: { background: "#0F141B", border: "1px solid #232E3A", color: "#C9D6E2", fontFamily: mono,
           fontSize: 11.5, padding: "5px 8px", borderRadius: 2, flex: 1, minWidth: 110 },
  filterLabel: { display: "flex", alignItems: "center", gap: 6, fontFamily: cond, fontSize: 10,
                 letterSpacing: ".1em", textTransform: "uppercase", color: "#6E8095" },
  table: { width: "100%", borderCollapse: "collapse" },
  th: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".12em", textTransform: "uppercase",
        textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #1D2733", whiteSpace: "nowrap" },
  td: { fontFamily: mono, fontSize: 11.5, padding: "7px 10px", borderBottom: "1px solid #161E27", whiteSpace: "nowrap" },
  tdN: { fontFamily: mono, fontSize: 11.5, padding: "7px 10px", borderBottom: "1px solid #161E27",
         textAlign: "right", color: "#A9BECE", whiteSpace: "nowrap" },
  sparkWrap: { display: "inline-block", width: 34, height: 6, background: "#0F141B", marginRight: 7,
               borderRadius: 1, overflow: "hidden", verticalAlign: "middle" },
  spark: { display: "block", height: "100%" },
  crossNote: { padding: "9px 14px", fontFamily: mono, fontSize: 10.5, color: "#58D5FF",
               borderTop: "1px solid #1D2733", lineHeight: 1.5 },
  mark: { fontFamily: mono, fontSize: 9, fontWeight: 600, color: "#0F141B", padding: "2px 4px", borderRadius: 1 },
  srcPanel: { borderTop: "1px solid #1D2733", padding: "12px 14px", background: "#0F141B" },
  srcHead: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase",
             color: "#58D5FF", marginBottom: 7 },
  srcPath: { fontFamily: mono, fontSize: 11.5, wordBreak: "break-all" },
  srcNote: { fontSize: 11.5, color: "#5C6E80", marginTop: 7, lineHeight: 1.5 },
  optLabel: { fontFamily: cond, fontSize: 10.5, letterSpacing: ".12em", textTransform: "uppercase", color: "#8FA3B5" },
  optNote: { fontSize: 11, color: "#5C6E80", lineHeight: 1.45 },
  pill: on => ({ border: `1px solid ${on ? "#58D5FF" : "#232E3A"}`, background: on ? "#16303D" : "#0F141B",
                 color: on ? "#58D5FF" : "#7A8DA0", fontFamily: mono, fontSize: 11, padding: "4px 10px", borderRadius: 2 }),
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
  wLabel: { fontFamily: mono, fontSize: 10.5, flex: "0 0 140px", overflow: "hidden",
            textOverflow: "ellipsis", whiteSpace: "nowrap" },
  wTrack: { flex: 1, height: 18, position: "relative", minWidth: 50 },
  wMid: { position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#232E3A" },
  tOpts: { border: "1px solid #1D2733", borderRadius: 2, overflow: "hidden" },
  tHead: { display: "grid", gridTemplateColumns: "1fr 72px 72px", padding: "6px 10px", background: "#0F141B",
           borderBottom: "1px solid #1D2733", fontFamily: cond, fontSize: 9.5,
           letterSpacing: ".11em", textTransform: "uppercase", color: "#5C6E80" },
  tRow: { display: "grid", gridTemplateColumns: "1fr 72px 72px", padding: "7px 10px", width: "100%",
          fontSize: 11.5, borderBottom: "1px solid #161E27", alignItems: "center", gap: 4 },
  caveat: { marginTop: 18, borderLeft: "2px solid #C9A227", paddingLeft: 11, fontSize: 11.5,
            lineHeight: 1.55, color: "#8FA3B5" },
  tableMini: { marginTop: 16, border: "1px solid #1D2733", borderRadius: 2 },
  tmHead: { display: "grid", gridTemplateColumns: "1fr 60px 80px", padding: "6px 10px",
            borderBottom: "1px solid #1D2733", fontFamily: cond, fontSize: 9.5,
            letterSpacing: ".11em", textTransform: "uppercase", color: "#5C6E80" },
  tmRow: { display: "grid", gridTemplateColumns: "1fr 60px 80px", padding: "5px 10px",
           fontFamily: mono, fontSize: 11.5, cursor: "pointer" },
  p: { fontSize: 12.5, lineHeight: 1.6, color: "#A9BECE", margin: "0 0 9px" },
  pBig: { fontSize: 14, lineHeight: 1.55, color: "#DCE7F1", margin: "0 0 12px" },
  dim: { color: "#5C6E80", fontFamily: mono, fontSize: 10 },
  mono: { fontFamily: mono },
  foot: { padding: "14px 20px 26px", fontFamily: mono, fontSize: 10.5, color: "#3B4A5A",
          borderTop: "1px solid #1D2733", lineHeight: 1.6 },
};
