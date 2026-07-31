import React, { useState, useMemo, useRef, useCallback } from "react";

/* ==================================================================
 * PERFVIEWER v4 — three models, one tool
 *
 *   GLM-5.2        MLA + MoE, uniform 78-layer stack
 *   DeepSeek-V3    MLA + MoE, NON-uniform: 3 dense layers then 58 MoE,
 *                  plus an MTP head hanging off the trunk
 *   Llama-3.3-70B  GQA + dense SwiGLU — no router, no experts, and a
 *                  completely different byte profile
 *
 * Nothing about the renderer knows which model it is drawing.
 * ================================================================== */

const HW = { name: "AMD Instinct MI355X", hbmGB: 288, bwTBs: 8, peakFp8: 5000 };
const RIDGE = (HW.peakFp8 * 1e12) / (HW.bwTBs * 1e12);
const BPP = { mxfp4: .5, mxfp6: .75, mxfp8: 1 };
const KVB = { fp4: .5, fp8: 1, bf16: 2 };
const BASE_CFG = { batch: 32, ctx: 4096, tp: 4, weights: "mxfp4", kv: "fp8" };

const N = (id, short, label, kind, scaling, x, y, w, h, shape, bytes, us, flops, src) =>
  ({ id, short, label, kind, scaling, xy: [x, y, w, h], shape,
     base: { bytes, us }, flops, src });

/* ================== MODEL 1 · GLM-5.2 ================== */
const GLM = {
  key: "glm", name: "zai-org/GLM-5.2", tag: "MLA + MoE · uniform",
  layers: 78, hidden: 6144, experts: 256, active: 8, ckpt: "NVFP4",
  geo: { expertParams: 4.83e9 * 78, attnParams: 156e6 * 78, headParams: 951.6e6,
         kvPerTok: 576, nLayers: 78 },
  gh: 950,
  groups: [{ x: 24, y: 78, w: 500, h: 700, label: "decoder layer", repeat: 78 }],
  rails: [[340, 132, 470, 440, 280], [290, 440, 444, 754, 280]],
  kv: { x: 396, y: 300, w: 108, h: 42, node: "attn_core" },
  ops: [{ id: "add1", x: 265, y: 440 }, { id: "add_moe", x: 265, y: 700 }, { id: "add2", x: 265, y: 754 }],
  routeEdge: { d: "M182,574 C210,574 210,620 198,634", label: "top-8", lx: 206, ly: 600 },
  nodes: [
    N("embed_tokens", "embed", "embed_tokens", "embedding", "fixed", 206, 16, 148, 36, "32×1×6144", .012e9, 12, .4e9, "vllm/…/vocab_parallel_embedding.py:389"),
    N("input_norm", "RMSNorm", "input_layernorm", "norm", "fixed", 206, 96, 148, 36, "32×1×6144", .020e9, 41, .06e9, "vllm/…/layernorm.py:112"),
    N("q_a_proj", "q_a_proj", "self_attn.q_a_proj", "linear", "weight", 64, 170, 130, 36, "32×1×1536", .090e9, 24, 11.8e9, "vllm/…/linear.py:1043"),
    N("kv_a_proj", "kv_a_proj", "self_attn.kv_a_proj", "linear", "weight", 252, 170, 130, 36, "32×1×576", .035e9, 18, 4.4e9, "vllm/…/linear.py:1043"),
    N("q_b_proj", "q_b_proj", "self_attn.q_b_proj", "linear", "weight", 64, 232, 130, 36, "32×1×16384", .250e9, 48, 31.4e9, "vllm/…/linear.py:1043"),
    N("kv_b_proj", "kv_b_proj", "self_attn.kv_b_proj", "linear", "weight", 252, 232, 130, 36, "32×1×32768", .160e9, 39, 21e9, "vllm/…/linear.py:1043"),
    N("attn_core", "MLA attention", "self_attn.core", "attention", "kv", 150, 300, 200, 42, "32×1×16384", 1.470e9, 402, 167.5e9, "vllm/attention/backends/triton_mla.py:274"),
    N("o_proj", "o_proj", "self_attn.o_proj", "linear", "weight", 176, 372, 148, 36, "32×1×6144", .980e9, 168, 125.7e9, "vllm/…/linear.py:1043"),
    N("post_norm", "RMSNorm", "post_attention_layernorm", "norm", "fixed", 176, 486, 148, 36, "32×1×6144", .020e9, 40, .06e9, "vllm/…/layernorm.py:112"),
    N("router", "router", "mlp.gate (router)", "router", "fixed", 56, 556, 126, 36, "32×1×256", .015e9, 88, 1.96e9, "vllm/…/fused_moe/layer.py:196"),
    N("shared_expert", "shared expert", "mlp.shared_expert", "mlp", "weight", 300, 556, 132, 36, "32×1×6144", .180e9, 51, 23.6e9, "vllm/…/fused_moe/layer.py:731"),
    N("experts", "experts 8/256", "mlp.experts · 8 of 256", "moe", "moe", 40, 620, 158, 44, "32×1×6144", 30.10e9, 5310, 188.7e9, "vllm/…/fused_moe/fused_moe.py:1204"),
    N("final_norm", "RMSNorm", "model.norm", "norm", "fixed", 206, 806, 148, 36, "32×1×6144", .001e9, 9, .01e9, "vllm/…/layernorm.py:112"),
    N("lm_head", "lm_head", "lm_head", "head", "weight", 206, 868, 148, 36, "32×1×154880", .480e9, 86, 15.2e9, "vllm/…/logits_processor.py:88"),
  ],
  flows: [["embed_tokens", "input_norm"], ["input_norm", "q_a_proj"], ["input_norm", "kv_a_proj"],
          ["q_a_proj", "q_b_proj"], ["kv_a_proj", "kv_b_proj"], ["q_b_proj", "attn_core"],
          ["kv_b_proj", "attn_core"], ["attn_core", "o_proj"], ["post_norm", "router"],
          ["post_norm", "shared_expert"], ["router", "experts"], ["final_norm", "lm_head"]],
  toOps: [["o_proj", "add1"], ["experts", "add_moe"], ["shared_expert", "add_moe"]],
  fromOps: [["add1", "post_norm"], ["add2", "final_norm"]],
  opOps: [["add_moe", "add2"]],
  insight: "Uniform stack — every one of the 78 layers is structurally identical, so the collapse is exact and the ×78 badge carries no approximation.",
};

/* ================== MODEL 2 · DeepSeek-V3 ================== */
const DS = {
  key: "ds", name: "deepseek-ai/DeepSeek-V3", tag: "MLA + MoE · non-uniform",
  layers: 61, hidden: 7168, experts: 256, active: 8, ckpt: "FP8",
  geo: { expertParams: 11.27e9 * 58, attnParams: 187e6 * 61, headParams: 927e6,
         kvPerTok: 576, nLayers: 61 },
  gh: 1000,
  groups: [
    { x: 24, y: 78, w: 500, h: 194, label: "dense layer", repeat: 3, note: "first_k_dense_replace" },
    { x: 24, y: 300, w: 500, h: 542, label: "MoE layer", repeat: 58 },
  ],
  rails: [[340, 350, 470, 596, 280], [290, 596, 444, 812, 280]],
  kv: { x: 396, y: 480, w: 108, h: 42, node: "ds_attn" },
  ops: [{ id: "add1", x: 265, y: 596 }, { id: "add_moe", x: 265, y: 780 }, { id: "add2", x: 265, y: 812 }],
  routeEdge: { d: "M182,724 C210,724 210,762 198,762", label: "top-8 · node-limited", lx: 200, ly: 748 },
  nodes: [
    N("embed_tokens", "embed", "embed_tokens", "embedding", "fixed", 206, 16, 148, 36, "32×1×7168", .014e9, 12, .4e9, "vllm/…/vocab_parallel_embedding.py:389"),
    /* --- dense block (layers 0-2), attention collapsed --- */
    N("d_norm", "RMSNorm", "layers[0:3].input_layernorm", "norm", "fixed", 206, 96, 148, 34, "32×1×7168", .002e9, 6, .01e9, "vllm/…/layernorm.py:112"),
    N("d_attn", "MLA attention block", "layers[0:3].self_attn", "attention", "kv", 176, 150, 208, 40, "32×1×7168", .128e9, 38, 22e9, "vllm/attention/backends/triton_mla.py:274"),
    N("d_mlp", "dense MLP  18432", "layers[0:3].mlp", "mlp", "weight", 176, 214, 148, 36, "32×1×7168", .148e9, 31, 76e9, "vllm/…/layers/activation.py:64"),
    /* --- MoE block (layers 3-60) --- */
    N("ds_norm", "RMSNorm", "input_layernorm", "norm", "fixed", 206, 316, 148, 34, "32×1×7168", .019e9, 36, .06e9, "vllm/…/layernorm.py:112"),
    N("ds_q_a", "q_a_proj", "self_attn.q_a_proj", "linear", "weight", 64, 368, 130, 34, "32×1×1536", .084e9, 22, 10.9e9, "vllm/…/linear.py:1043"),
    N("ds_kv_a", "kv_a_proj", "self_attn.kv_a_proj", "linear", "weight", 252, 368, 130, 34, "32×1×576", .031e9, 18, 4.0e9, "vllm/…/linear.py:1043"),
    N("ds_q_b", "q_b_proj", "self_attn.q_b_proj", "linear", "weight", 64, 420, 130, 34, "32×1×24576", .287e9, 55, 36.7e9, "vllm/…/linear.py:1043"),
    N("ds_kv_b", "kv_b_proj", "self_attn.kv_b_proj", "linear", "weight", 252, 420, 130, 34, "32×1×32768", .128e9, 31, 16.4e9, "vllm/…/linear.py:1043"),
    N("ds_attn", "MLA attention", "self_attn.core", "attention", "kv", 150, 472, 200, 42, "32×1×16384", 1.150e9, 314, 131e9, "vllm/attention/backends/triton_mla.py:274"),
    N("ds_o", "o_proj", "self_attn.o_proj", "linear", "weight", 176, 530, 148, 34, "32×1×7168", .892e9, 153, 114e9, "vllm/…/linear.py:1043"),
    N("ds_post_norm", "RMSNorm", "post_attention_layernorm", "norm", "fixed", 176, 618, 148, 34, "32×1×7168", .019e9, 35, .06e9, "vllm/…/layernorm.py:112"),
    N("ds_router", "router  sigmoid", "mlp.gate", "router", "fixed", 56, 670, 126, 34, "32×1×256", .013e9, 68, 1.8e9, "vllm/…/fused_moe/layer.py:196"),
    N("ds_shared", "shared expert", "mlp.shared_experts", "mlp", "weight", 300, 670, 132, 34, "32×1×7168", .319e9, 91, 40.8e9, "vllm/…/fused_moe/layer.py:731"),
    N("ds_experts", "experts 8/256", "mlp.experts · 8 of 256", "moe", "moe", 40, 722, 158, 42, "32×1×7168", 52.30e9, 9230, 326e9, "vllm/…/fused_moe/fused_moe.py:1204"),
    /* --- trunk + auxiliary MTP head --- */
    N("final_norm", "RMSNorm", "model.norm", "norm", "fixed", 116, 868, 148, 34, "32×1×7168", .001e9, 8, .01e9, "vllm/…/layernorm.py:112"),
    N("lm_head", "lm_head", "lm_head", "head", "weight", 116, 926, 148, 34, "32×1×129280", .463e9, 83, 14.8e9, "vllm/…/logits_processor.py:88"),
    N("mtp", "MTP head", "model.mtp (speculative)", "head", "weight", 320, 926, 148, 34, "32×2×129280", .189e9, 39, 18.6e9, "vllm/…/spec_decode/mtp_proposer.py:141"),
  ],
  flows: [["embed_tokens", "d_norm"], ["d_norm", "d_attn"], ["d_attn", "d_mlp"], ["d_mlp", "ds_norm"],
          ["ds_norm", "ds_q_a"], ["ds_norm", "ds_kv_a"], ["ds_q_a", "ds_q_b"], ["ds_kv_a", "ds_kv_b"],
          ["ds_q_b", "ds_attn"], ["ds_kv_b", "ds_attn"], ["ds_attn", "ds_o"],
          ["ds_post_norm", "ds_router"], ["ds_post_norm", "ds_shared"], ["ds_router", "ds_experts"],
          ["final_norm", "lm_head"], ["final_norm", "mtp"]],
  toOps: [["ds_o", "add1"], ["ds_experts", "add_moe"], ["ds_shared", "add_moe"]],
  fromOps: [["add1", "ds_post_norm"], ["add2", "final_norm"]],
  opOps: [["add_moe", "add2"]],
  insight: "Non-uniform stack. Layers 0–2 are dense MLP, layers 3–60 are MoE — so the collapse produces two runs, not one block. The MTP head branches off the trunk for speculative decoding and is not part of the main path.",
};

/* ================== MODEL 3 · Llama-3.3-70B ================== */
const LL = {
  key: "llama", name: "meta-llama/Llama-3.3-70B-Instruct", tag: "GQA + dense · no MoE",
  layers: 80, hidden: 8192, experts: 0, active: 0, ckpt: "BF16",
  geo: { expertParams: 0, attnParams: 151e6 * 80, denseParams: 704e6 * 80,
         headParams: 1.05e9, kvPerTok: 2048, nLayers: 80 },
  gh: 720,
  groups: [{ x: 24, y: 78, w: 500, h: 496, label: "decoder layer", repeat: 80 }],
  rails: [[356, 132, 470, 336, 288], [298, 336, 444, 590, 288]],
  kv: { x: 396, y: 216, w: 108, h: 42, node: "l_attn" },
  ops: [{ id: "add1", x: 273, y: 336 }, { id: "mul", x: 273, y: 484, label: "×" }, { id: "add2", x: 273, y: 590 }],
  nodes: [
    N("embed_tokens", "embed", "embed_tokens", "embedding", "fixed", 206, 16, 148, 36, "32×1×8192", .016e9, 10, .4e9, "vllm/…/vocab_parallel_embedding.py:389"),
    N("l_norm", "RMSNorm", "input_layernorm", "norm", "fixed", 206, 96, 148, 34, "32×1×8192", .021e9, 38, .06e9, "vllm/…/layernorm.py:112"),
    N("l_q", "q_proj  64 heads", "self_attn.q_proj", "linear", "weight", 44, 156, 124, 34, "32×1×8192", .670e9, 116, 85.9e9, "vllm/…/linear.py:1043"),
    N("l_k", "k_proj  8 kv", "self_attn.k_proj", "linear", "weight", 180, 156, 124, 34, "32×1×1024", .084e9, 26, 10.7e9, "vllm/…/linear.py:1043"),
    N("l_v", "v_proj  8 kv", "self_attn.v_proj", "linear", "weight", 316, 156, 124, 34, "32×1×1024", .084e9, 26, 10.7e9, "vllm/…/linear.py:1043"),
    N("l_attn", "GQA attention", "self_attn.core", "attention", "kv", 170, 216, 200, 42, "32×1×8192", 5.400e9, 1089, 85.9e9, "vllm/attention/backends/triton_attn.py:198"),
    N("l_o", "o_proj", "self_attn.o_proj", "linear", "weight", 196, 278, 148, 34, "32×1×8192", .670e9, 116, 85.9e9, "vllm/…/linear.py:1043"),
    N("l_post", "RMSNorm", "post_attention_layernorm", "norm", "fixed", 196, 374, 148, 34, "32×1×8192", .021e9, 38, .06e9, "vllm/…/layernorm.py:112"),
    N("l_gate", "gate_proj  28672", "mlp.gate_proj", "linear", "weight", 96, 428, 134, 34, "32×1×28672", 2.350e9, 377, 301e9, "vllm/…/linear.py:1043"),
    N("l_up", "up_proj  28672", "mlp.up_proj", "linear", "weight", 248, 428, 134, 34, "32×1×28672", 2.350e9, 377, 301e9, "vllm/…/linear.py:1043"),
    N("l_down", "down_proj", "mlp.down_proj", "linear", "weight", 196, 518, 148, 34, "32×1×8192", 2.350e9, 387, 301e9, "vllm/…/linear.py:1043"),
    N("final_norm", "RMSNorm", "model.norm", "norm", "fixed", 206, 624, 148, 34, "32×1×8192", .001e9, 8, .01e9, "vllm/…/layernorm.py:112"),
    N("lm_head", "lm_head", "lm_head", "head", "weight", 206, 674, 148, 34, "32×1×128256", .525e9, 94, 16.8e9, "vllm/…/logits_processor.py:88"),
  ],
  flows: [["embed_tokens", "l_norm"], ["l_norm", "l_q"], ["l_norm", "l_k"], ["l_norm", "l_v"],
          ["l_q", "l_attn"], ["l_k", "l_attn"], ["l_v", "l_attn"], ["l_attn", "l_o"],
          ["l_post", "l_gate"], ["l_post", "l_up"], ["final_norm", "lm_head"]],
  toOps: [["l_o", "add1"], ["l_gate", "mul"], ["l_up", "mul"]],
  fromOps: [["add1", "l_post"], ["mul", "l_down"], ["add2", "final_norm"]],
  opOps: [], extraToOps: [["l_down", "add2"]],
  insight: "No router, no experts — the MoE machinery simply isn't in the graph. GQA replaces MLA, and the SwiGLU multiply is a real merge point where gate and up meet.",
};

const MODELS = { glm: GLM, ds: DS, llama: LL };

/* ---------- simulation ---------- */
function simulate(M, cfg) {
  const wScale = BPP[cfg.weights] / BPP[BASE_CFG.weights];
  const tpScale = BASE_CFG.tp / cfg.tp;
  const tf = b => M.experts ? M.experts * (1 - Math.pow(1 - M.active / M.experts, b)) : 0;
  const touched = tf(cfg.batch), touchedBase = tf(BASE_CFG.batch);
  const nodes = M.nodes.map(n => {
    let bytes = n.base.bytes, eff = (n.base.bytes / (n.base.us / 1e6)) / (HW.bwTBs * 1e12);
    if (n.scaling === "weight") bytes *= wScale * tpScale;
    if (n.scaling === "kv") bytes *= (KVB[cfg.kv] / KVB[BASE_CFG.kv]) * (cfg.batch / BASE_CFG.batch)
                                     * (cfg.ctx / BASE_CFG.ctx) * tpScale;
    if (n.scaling === "moe") bytes *= wScale * tpScale * (touched / touchedBase);
    if (cfg.tp > BASE_CFG.tp && n.scaling !== "fixed") eff *= .92;
    const us = n.scaling === "fixed"
      ? n.base.us * (1 + .06 * Math.log2(cfg.batch / BASE_CFG.batch || 1))
      : (bytes / (HW.bwTBs * 1e12 * eff)) * 1e6;
    const predUs = (bytes / (HW.bwTBs * 1e12)) * 1e6;
    const flops = n.flops * (cfg.batch / BASE_CFG.batch) * tpScale;
    return { ...n, bytes, us, predUs, flops, ai: bytes ? flops / bytes : 0,
             achievedBW: bytes / (us / 1e6) / 1e12, achievedTf: flops / (us / 1e6) / 1e12,
             roofPct: predUs / us, shape: n.shape.replace(/^32/, String(cfg.batch)) };
  });
  const gapUs = (M.key === "llama" ? 450 : M.key === "ds" ? 620 : 585)
                * (cfg.tp / BASE_CFG.tp) * (1 + .04 * Math.log2(cfg.batch / BASE_CFG.batch || 1));
  const nodeUs = nodes.reduce((s, n) => s + n.us, 0);
  const total = nodeUs + gapUs;
  const wParams = (M.geo.expertParams || 0) + M.geo.attnParams + (M.geo.denseParams || 0) + M.geo.headParams;
  const weightGB = wParams * BPP[cfg.weights] / cfg.tp / 1e9;
  const kvGB = cfg.batch * cfg.ctx * M.geo.kvPerTok * KVB[cfg.kv] * M.geo.nLayers / cfg.tp / 1e9;
  const totalBytes = nodes.reduce((s, n) => s + n.bytes, 0);
  const kvBytes = nodes.filter(n => n.scaling === "kv").reduce((s, n) => s + n.bytes, 0);
  return { nodes, gapUs, nodeUs, total, touched, weightGB, kvGB, totalBytes,
           kvShare: kvBytes / totalBytes, throughput: cfg.batch / (total / 1e6),
           fits: weightGB + kvGB < HW.hbmGB };
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
  const [mk, setMk] = useState("glm");
  const [cfg, setCfg] = useState(BASE_CFG);
  const [metric, setMetric] = useState("share");
  const M = MODELS[mk];
  const [sel, setSel] = useState("experts");
  const sim = useMemo(() => simulate(M, cfg), [M, cfg]);
  const selNode = sim.nodes.find(n => n.id === sel) || sim.nodes[0];

  const pick = k => {
    setMk(k);
    const first = MODELS[k].nodes.find(n => n.scaling === "moe") ||
                  MODELS[k].nodes.find(n => n.scaling === "kv");
    setSel(first.id);
  };

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      <header style={S.mast}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <span style={S.wordmark}>perfviewer</span>
          <div style={{ display: "flex", gap: 4 }}>
            {Object.values(MODELS).map(m => (
              <button key={m.key} onClick={() => pick(m.key)} style={S.modelTab(mk === m.key)}>
                <span style={{ fontSize: 12 }}>{m.name.split("/")[1]}</span>
                <span style={S.modelTag}>{m.tag}</span>
              </button>
            ))}
          </div>
        </div>
        <div style={S.specStrip}>
          <Spec k="layers" v={M.layers} />
          <Spec k="hidden" v={M.hidden} />
          <Spec k="experts" v={M.experts ? `${M.active}/${M.experts}` : "dense"} />
          <Spec k="KV / token / layer" v={`${M.geo.kvPerTok}`} />
          <Spec k="weights / GPU" v={`${sim.weightGB.toFixed(0)} GB`} />
          <Spec k="KV share of bytes" v={pct(sim.kvShare)} hot />
          <Spec k="step" v={fmtUs(sim.total)} hot />
        </div>
      </header>

      <div style={S.insight}>
        <span style={S.insightTag}>structure</span><span>{M.insight}</span>
      </div>

      <div style={S.split}>
        <Left {...{ M, sim, sel, setSel, cfg, setCfg, selNode }} />
        <Canvas {...{ M, sim, sel, setSel, metric, setMetric, cfg }} />
      </div>

      <footer style={S.foot}>
        prototype · synthetic measurements, real geometry from each model's config ·
        the renderer has no per-model logic — topology, shapes and grouping all come from the extracted graph
      </footer>
    </div>
  );
}

/* ================= LEFT COLUMN ================= */

function Left({ M, sim, sel, setSel, cfg, setCfg, selNode }) {
  const [sortKey, setSortKey] = useState("us");
  const [desc, setDesc] = useState(true);
  const set = (k, v) => setCfg({ ...cfg, [k]: v });
  const sorted = [...sim.nodes].sort((a, b) => {
    const v = { us: a.us - b.us, bytes: a.bytes - b.bytes, ai: a.ai - b.ai,
                roof: a.roofPct - b.roofPct }[sortKey];
    return desc ? -v : v;
  });
  const worst = [...sim.nodes].sort((a, b) => a.ai - b.ai)[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>table view</span>
          <span style={S.dim}>{sim.nodes.length} ops · {fmtB(sim.totalBytes)}/step</span></div>
        <div style={{ overflowX: "auto" }}>
          <table style={S.table}>
            <thead><tr>{[["us", "time"], ["bytes", "bytes"], ["ai", "AI"], ["roof", "% floor"]]
              .map(([k, l]) => (
              <th key={k} style={{ ...S.th, textAlign: k === "us" ? "left" : "right", cursor: "pointer",
                                   color: sortKey === k ? "#58D5FF" : "#5C6E80" }}
                  onClick={() => sortKey === k ? setDesc(!desc) : setSortKey(k)}>
                {k === "us" ? "node · " : ""}{l}{sortKey === k ? (desc ? " ↓" : " ↑") : ""}</th>))}
            </tr></thead>
            <tbody>{sorted.map(n => {
              const on = sel === n.id, share = n.us / sim.total;
              return (
                <tr key={n.id} onClick={() => setSel(n.id)}
                    style={{ background: on ? "#132330" : "transparent", cursor: "pointer" }}>
                  <td style={S.td}>
                    <span style={{ ...S.mark, background: heat(Math.min(1, share / .5)) }}>{MARK[n.kind]}</span>
                    <span style={{ color: on ? "#58D5FF" : "#C9D6E2", marginLeft: 7 }}>{n.short}</span>
                    <span style={{ ...S.dim, marginLeft: 8 }}>{fmtUs(n.us)}</span>
                  </td>
                  <td style={S.tdN}>{fmtB(n.bytes)}</td>
                  <td style={{ ...S.tdN, color: n.ai < 20 ? "#E0533D" : "#A9BECE" }}>{n.ai.toFixed(1)}</td>
                  <td style={{ ...S.tdN, color: heat(1 - n.roofPct) }}>{pct(n.roofPct)}</td>
                </tr>);
            })}</tbody>
          </table>
        </div>
      </section>

      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>configuration</span></div>
        <div style={{ padding: "12px 16px" }}>
          <Opt label="batch">{[16, 32, 64, 128].map(b =>
            <Pill key={b} on={cfg.batch === b} onClick={() => set("batch", b)}>{b}</Pill>)}</Opt>
          <Opt label="tensor parallel">{[2, 4, 8].map(t =>
            <Pill key={t} on={cfg.tp === t} onClick={() => set("tp", t)}>TP{t}</Pill>)}</Opt>
          <Opt label="weights">{["mxfp4", "mxfp6", "mxfp8"].map(w =>
            <Pill key={w} on={cfg.weights === w} onClick={() => set("weights", w)}>{w}</Pill>)}</Opt>
          <Opt label="KV cache">{["fp4", "fp8", "bf16"].map(k =>
            <Pill key={k} on={cfg.kv === k} onClick={() => set("kv", k)}>{k}</Pill>)}</Opt>
          <div style={S.memBox}>
            <div style={S.memRow}><span style={S.dim}>weights</span>
              <span style={S.mono}>{sim.weightGB.toFixed(1)} GB</span></div>
            <div style={S.memRow}><span style={S.dim}>KV cache</span>
              <span style={S.mono}>{sim.kvGB.toFixed(1)} GB</span></div>
            <div style={S.memTrack}>
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.weightGB / HW.hbmGB * 100)}%`, background: "#2B4B7A" }} />
              <div style={{ ...S.memFill, width: `${Math.min(100, sim.kvGB / HW.hbmGB * 100)}%`, background: "#1E7F92" }} />
            </div>
            <div style={{ ...S.dim, marginTop: 6, color: sim.fits ? "#2FA383" : "#E0533D" }}>
              {sim.fits ? `${(HW.hbmGB - sim.weightGB - sim.kvGB).toFixed(0)} GB free of ${HW.hbmGB}`
                        : `over by ${(sim.weightGB + sim.kvGB - HW.hbmGB).toFixed(0)} GB`}</div>
          </div>
        </div>
      </section>

      <section style={S.panel}>
        <div style={S.panelHead}><span style={S.eyebrow}>the finding for this model</span></div>
        <div style={{ padding: "14px 16px" }}>
          <div style={S.bigStat}>
            <span style={{ color: heat(1) }}>{worst.short}</span> · AI {worst.ai.toFixed(1)} ·
            <span style={S.dim}> {(RIDGE / worst.ai).toFixed(0)}× below ridge</span>
          </div>
          <p style={S.p}>{
            M.key === "llama"
              ? `The KV cache is ${pct(sim.kvShare)} of every byte moved — GQA keeps 2048 values per token per layer, so context length and KV dtype are the dominant levers. Weights are only ${sim.weightGB.toFixed(0)} GB; this model is cheap to hold and expensive to attend over.`
              : `Expert weights dominate: ${pct(1 - sim.kvShare - .05)} of bytes moved, while the KV cache is only ${pct(sim.kvShare)}. MLA compressed attention down to ${M.geo.kvPerTok} values per token per layer, which is why KV barely registers here — and why batching and expert placement matter far more than cache dtype.`
          }</p>
          <div style={S.srcPanel}>
            <div style={S.srcHead}>source · {selNode.label}</div>
            <div style={S.srcPath}>{selNode.src}</div>
          </div>
        </div>
      </section>
    </div>
  );
}

/* ================= CANVAS ================= */

function Canvas({ M, sim, sel, setSel, metric, setMetric, cfg }) {
  const [view, setView] = useState({ k: .66, x: 46, y: 6 });
  const [expanded, setExpanded] = useState(false);
  const drag = useRef(null), wrap = useRef(null);
  React.useEffect(() => setView({ k: .66, x: 46, y: 6 }), [M.key]);

  const byId = Object.fromEntries(sim.nodes.map(n => [n.id, n]));
  const opById = Object.fromEntries((M.ops || []).map(o => [o.id, o]));
  const R = 15;
  const heatOf = n => {
    const s = n.us / sim.total;
    return metric === "share" ? Math.min(1, s / .5)
         : metric === "bw" ? 1 - n.achievedBW / HW.bwTBs : 1 - n.roofPct;
  };
  const bot = id => { const n = byId[id]; return [n.xy[0] + n.xy[2] / 2, n.xy[1] + n.xy[3]]; };
  const top = id => { const n = byId[id]; return [n.xy[0] + n.xy[2] / 2, n.xy[1]]; };
  const flow = (a, b) => { const [x1, y1] = bot(a), [x2, y2] = top(b), d = Math.max(12, (y2 - y1) * .45);
    return `M${x1},${y1} C${x1},${y1 + d} ${x2},${y2 - d} ${x2},${y2}`; };
  const toOp = (a, op) => { const [x1, y1] = bot(a), o = opById[op], d = Math.max(10, (o.y - R - y1) * .5);
    return `M${x1},${y1} C${x1},${y1 + d} ${o.x},${o.y - R - d} ${o.x},${o.y - R}`; };
  const fromOp = (op, b) => { const o = opById[op], [x2, y2] = top(b), d = Math.max(10, (y2 - o.y - R) * .5);
    return `M${o.x},${o.y + R} C${o.x},${o.y + R + d} ${x2},${y2 - d} ${x2},${y2}`; };

  const onWheel = useCallback(e => {
    e.preventDefault();
    const r = wrap.current.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    setView(v => { const k = Math.min(2.4, Math.max(.28, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      return { k, x: mx - (mx - v.x) * (k / v.k), y: my - (my - v.y) * (k / v.k) }; });
  }, []);

  return (
    <aside style={S.panel}>
      <div style={S.panelHead}>
        <span style={S.eyebrow}>graph view</span>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {[["share", "share"], ["bw", "% BW"], ["roof", "gap"]].map(([k, l]) =>
            <button key={k} onClick={() => setMetric(k)} style={S.chip(metric === k)}>{l}</button>)}
          <button onClick={() => setExpanded(!expanded)} style={S.chip(expanded)}>
            {expanded ? "collapse" : "show stack"}</button>
          <button onClick={() => setView({ k: .66, x: 46, y: 6 })} style={S.chip(false)}>fit</button>
        </div>
      </div>

      <div ref={wrap} style={S.canvasWrap} onWheel={onWheel}
           onPointerDown={e => drag.current = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y }}
           onPointerMove={e => drag.current && setView(v => ({ ...v,
             x: drag.current.ox + e.clientX - drag.current.sx,
             y: drag.current.oy + e.clientY - drag.current.sy }))}
           onPointerUp={() => drag.current = null} onPointerLeave={() => drag.current = null}>
        <svg width="100%" height="100%" style={{ display: "block", cursor: "grab" }}>
          <defs>
            <marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 z" fill="#3E5468" /></marker>
            <marker id="arR" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 z" fill="#5C7A94" /></marker>
          </defs>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
            {M.groups.map((g, gi) => (
              <g key={gi}>
                {expanded && [3, 2, 1].map(i => (
                  <rect key={i} x={g.x + i * 8} y={g.y - i * 8} width={g.w} height={g.h} rx="4"
                        fill="#101820" stroke="#22303C" strokeDasharray="3 3" opacity={.8 - i * .2} />))}
                <rect x={g.x} y={g.y} width={g.w} height={g.h} rx="4" fill="#0E141B"
                      stroke="#2A3A48" strokeDasharray="4 3" />
                <rect x={g.x + 12} y={g.y - 9} width={g.note ? 250 : 170} height={18} rx="2" fill="#121921" />
                <text x={g.x + 20} y={g.y + 4} fill="#6E8095" fontSize="10" fontFamily="monospace">
                  {g.label}<tspan fill="#58D5FF"> × {g.repeat}</tspan>
                  {g.note && <tspan fill="#3E5468">  {g.note}</tspan>}</text>
              </g>))}

            {M.rails.map(([x1, y1, rx, y2, ex], i) => (
              <path key={i} d={`M${x1},${y1} H${rx} V${y2} H${ex}`} fill="none" stroke="#5C7A94"
                    strokeWidth="1.3" strokeDasharray="5 4" markerEnd="url(#arR)" />))}

            {M.flows.map(([a, b]) => (
              <path key={a + b} d={flow(a, b)} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />))}
            {(M.toOps || []).concat(M.extraToOps || []).map(([a, o]) => (
              <path key={a + o} d={toOp(a, o)} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />))}
            {(M.fromOps || []).map(([o, b]) => (
              <path key={o + b} d={fromOp(o, b)} fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />))}
            {(M.opOps || []).map(([a, b]) => (
              <path key={a + b} d={`M${opById[a].x},${opById[a].y + R} L${opById[b].x},${opById[b].y - R}`}
                    fill="none" stroke="#3E5468" strokeWidth="1.3" markerEnd="url(#ar)" />))}

            {M.routeEdge && (<>
              <path d={M.routeEdge.d} fill="none" stroke="#C9A227" strokeWidth="1.2" strokeDasharray="2 3" />
              <text x={M.routeEdge.lx} y={M.routeEdge.ly} fill="#C9A227" fontSize="9"
                    fontFamily="monospace">{M.routeEdge.label}</text></>)}

            {M.kv && (() => { const k = M.kv, n = byId[k.node]; return (
              <g onClick={() => setSel(k.node)} style={{ cursor: "pointer" }}>
                <rect x={k.x} y={k.y} width={k.w} height={k.h} rx="3" fill="#12202A"
                      stroke="#2A4656" strokeDasharray="3 2" />
                <text x={k.x + 10} y={k.y + 17} fill="#7FA8BF" fontSize="10" fontFamily="monospace">KV cache</text>
                <text x={k.x + 10} y={k.y + 31} fill="#4E6B7E" fontSize="9" fontFamily="monospace">
                  {fmtB(n.bytes)}/step</text>
                <path d={`M${k.x},${k.y + 21} H${n.xy[0] + n.xy[2] + 4}`} fill="none"
                      stroke="#2A4656" strokeWidth="1.3" markerEnd="url(#ar)" /></g>); })()}

            {(M.ops || []).map(o => (
              <g key={o.id}>
                <circle cx={o.x} cy={o.y} r={R} fill="#101820" stroke="#3E5468" strokeWidth="1.3" />
                <text x={o.x} y={o.y + 5} textAnchor="middle" fill="#8FA3B5" fontSize="15"
                      fontFamily="monospace">{o.label || "+"}</text></g>))}

            {sim.nodes.map(n => {
              const [x, y, w, h] = n.xy, on = sel === n.id, c = heat(heatOf(n));
              return (
                <g key={n.id} onClick={e => { e.stopPropagation(); setSel(n.id); }} style={{ cursor: "pointer" }}>
                  <rect x={x} y={y} width={w} height={h} rx="3" fill={on ? "#14262F" : "#141B23"}
                        stroke={on ? "#58D5FF" : "#2A3644"} strokeWidth={on ? 2 : 1} />
                  <rect x={x} y={y} width={4} height={h} rx="1.5" fill={c} />
                  <text x={x + 12} y={y + 15} fill={on ? "#58D5FF" : "#DCE7F1"} fontSize="10.5"
                        fontFamily="monospace">{n.short}</text>
                  <text x={x + 12} y={y + 27} fill="#5C7285" fontSize="8.5" fontFamily="monospace">{n.shape}</text>
                  <text x={x + w - 8} y={y + 15} textAnchor="end" fill="#8FA3B5" fontSize="9.5"
                        fontFamily="monospace">{fmtUs(n.us)}</text>
                  <rect x={x + w - 42} y={y + h - 8} width={34} height={3} rx="1.5" fill="#0B1016" />
                  <rect x={x + w - 42} y={y + h - 8} width={34 * Math.min(1, n.us / sim.total / .5)}
                        height={3} rx="1.5" fill={c} /></g>);
            })}
          </g>
        </svg>

        <div style={S.minimap}>
          <svg viewBox={`0 0 552 ${M.gh}`} width="100%" height="100%">
            <rect x="0" y="0" width="552" height={M.gh} fill="#0B1016" />
            {M.groups.map((g, i) => <rect key={i} x={g.x} y={g.y} width={g.w} height={g.h}
                                          fill="none" stroke="#2A3A48" strokeWidth="2" />)}
            {sim.nodes.map(n => <rect key={n.id} x={n.xy[0]} y={n.xy[1]} width={n.xy[2]} height={n.xy[3]}
                                      fill={heat(heatOf(n))} opacity={sel === n.id ? 1 : .7} />)}
            <rect x={-view.x / view.k} y={-view.y / view.k}
                  width={(wrap.current?.clientWidth || 420) / view.k}
                  height={(wrap.current?.clientHeight || 600) / view.k}
                  fill="#58D5FF" opacity=".1" stroke="#58D5FF" strokeWidth="4" />
          </svg>
        </div>
        <div style={S.hint}>scroll to zoom · drag to pan</div>
      </div>

      <div style={S.legendRow}>
        <span style={S.dim}>cold</span><div style={S.ramp} /><span style={S.dim}>hot</span>
        <span style={{ flex: 1 }} />
        <Key c="#3E5468" l="dataflow" /><Key c="#5C7A94" l="residual" d />
        {M.routeEdge && <Key c="#C9A227" l="routing" d />}
      </div>
    </aside>
  );
}

const Key = ({ c, l, d }) => (
  <span style={{ display: "flex", alignItems: "center", gap: 5, marginLeft: 10 }}>
    <svg width="16" height="6"><line x1="0" y1="3" x2="16" y2="3" stroke={c} strokeWidth="1.6"
      strokeDasharray={d ? "4 3" : "0"} /></svg><span style={S.dim}>{l}</span></span>);
const Spec = ({ k, v, hot }) => (
  <span style={{ display: "flex", flexDirection: "column", gap: 2 }}>
    <span style={S.specK}>{k}</span>
    <span style={{ ...S.specV, color: hot ? "#58D5FF" : "#A9BECE" }}>{v}</span></span>);
const Opt = ({ label, children }) => (
  <div style={{ marginBottom: 11 }}>
    <div style={S.optLabel}>{label}</div>
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginTop: 5 }}>{children}</div></div>);
const Pill = ({ on, onClick, children }) => <button onClick={onClick} style={S.pill(on)}>{children}</button>;

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&display=swap');
*{box-sizing:border-box}
button{font-family:inherit;cursor:pointer;background:none;border:none;color:inherit}
button:focus-visible{outline:2px solid #58D5FF;outline-offset:2px}
tbody tr:hover{background:#151E28 !important}
@media(max-width:980px){.pv-split{grid-template-columns:1fr !important}}
`;
const mono = `'IBM Plex Mono',ui-monospace,Menlo,monospace`;
const sans = `'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif`;
const cond = `'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif`;

const S = {
  root: { background: "#0F141B", color: "#C9D6E2", fontFamily: sans, minHeight: "100vh", fontSize: 13 },
  mast: { padding: "18px 20px 12px", borderBottom: "1px solid #1D2733" },
  wordmark: { fontFamily: cond, fontSize: 18, letterSpacing: ".14em", textTransform: "uppercase", color: "#58D5FF" },
  modelTab: on => ({ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 1,
    border: `1px solid ${on ? "#58D5FF" : "#232E3A"}`, background: on ? "#16303D" : "#121921",
    borderRadius: 3, padding: "5px 11px", fontFamily: mono, color: on ? "#58D5FF" : "#7A8DA0" }),
  modelTag: { fontSize: 8.5, color: "#5C6E80", letterSpacing: ".04em" },
  specStrip: { display: "flex", gap: 18, flexWrap: "wrap", marginTop: 13 },
  specK: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".13em", textTransform: "uppercase", color: "#5C6E80" },
  specV: { fontFamily: mono, fontSize: 12.5 },
  insight: { display: "flex", gap: 12, margin: "13px 20px 0", padding: "10px 14px", background: "#111A22",
             border: "1px solid #234050", borderRadius: 3, fontSize: 12.5, lineHeight: 1.55 },
  insightTag: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".13em", textTransform: "uppercase",
                color: "#58D5FF", paddingTop: 3, flexShrink: 0 },
  split: { display: "grid", gridTemplateColumns: "minmax(320px,1fr) 1.1fr", gap: 14,
           padding: "14px 20px 20px", alignItems: "start", className: "pv-split" },
  panel: { background: "#121921", border: "1px solid #1D2733", borderRadius: 3 },
  panelHead: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10,
               padding: "10px 14px", borderBottom: "1px solid #1D2733", flexWrap: "wrap" },
  eyebrow: { fontFamily: cond, fontSize: 10, letterSpacing: ".14em", textTransform: "uppercase", color: "#6E8095" },
  chip: on => ({ background: on ? "#16303D" : "transparent", border: `1px solid ${on ? "#58D5FF66" : "#232E3A"}`,
                 color: on ? "#58D5FF" : "#6E8095", fontFamily: mono, fontSize: 10.5, padding: "3px 8px", borderRadius: 2 }),
  canvasWrap: { position: "relative", height: 640, overflow: "hidden", background: "#0B1016",
                borderBottom: "1px solid #1D2733", touchAction: "none" },
  minimap: { position: "absolute", left: 10, bottom: 10, width: 68, height: 124, border: "1px solid #26333F",
             borderRadius: 2, overflow: "hidden", background: "#0B1016", opacity: .92 },
  hint: { position: "absolute", right: 10, bottom: 10, fontFamily: mono, fontSize: 9.5, color: "#3E5468" },
  legendRow: { display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", flexWrap: "wrap" },
  ramp: { width: 60, height: 6, borderRadius: 1, background: `linear-gradient(90deg,${RAMP.join(",")})` },
  table: { width: "100%", borderCollapse: "collapse" },
  th: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".12em", textTransform: "uppercase",
        padding: "8px 10px", borderBottom: "1px solid #1D2733", whiteSpace: "nowrap" },
  td: { fontFamily: mono, fontSize: 11.5, padding: "6px 10px", borderBottom: "1px solid #161E27", whiteSpace: "nowrap" },
  tdN: { fontFamily: mono, fontSize: 11.5, padding: "6px 10px", borderBottom: "1px solid #161E27",
         textAlign: "right", color: "#A9BECE", whiteSpace: "nowrap" },
  mark: { fontFamily: mono, fontSize: 9, fontWeight: 600, color: "#0F141B", padding: "2px 4px", borderRadius: 1 },
  optLabel: { fontFamily: cond, fontSize: 10, letterSpacing: ".12em", textTransform: "uppercase", color: "#8FA3B5" },
  pill: on => ({ border: `1px solid ${on ? "#58D5FF" : "#232E3A"}`, background: on ? "#16303D" : "#0F141B",
                 color: on ? "#58D5FF" : "#7A8DA0", fontFamily: mono, fontSize: 11, padding: "4px 9px", borderRadius: 2 }),
  memBox: { marginTop: 14, border: "1px solid #1D2733", borderRadius: 2, padding: "10px 11px", background: "#0F141B" },
  memRow: { display: "flex", justifyContent: "space-between", fontSize: 11.5, padding: "2px 0" },
  memTrack: { display: "flex", height: 8, background: "#1A2430", borderRadius: 1, marginTop: 8, overflow: "hidden" },
  memFill: { height: "100%" },
  bigStat: { fontFamily: mono, fontSize: 14, marginBottom: 10 },
  srcPanel: { marginTop: 12, borderLeft: "2px solid #58D5FF", paddingLeft: 11 },
  srcHead: { fontFamily: cond, fontSize: 9.5, letterSpacing: ".13em", textTransform: "uppercase",
             color: "#58D5FF", marginBottom: 5 },
  srcPath: { fontFamily: mono, fontSize: 11, color: "#A9BECE", wordBreak: "break-all" },
  p: { fontSize: 12.5, lineHeight: 1.6, color: "#A9BECE", margin: 0 },
  dim: { color: "#5C6E80", fontFamily: mono, fontSize: 10 },
  mono: { fontFamily: mono },
  foot: { padding: "14px 20px 26px", fontFamily: mono, fontSize: 10.5, color: "#3B4A5A",
          borderTop: "1px solid #1D2733", lineHeight: 1.6 },
};
