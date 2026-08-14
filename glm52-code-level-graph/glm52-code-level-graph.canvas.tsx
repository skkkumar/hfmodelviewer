import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Swatch,
  Table,
  Text,
  UsageBar,
  computeDAGLayout,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";
import type { Color } from "cursor/canvas";

type Depth = "architecture" | "modules" | "functions" | "kernels";
type Heat = "none" | "latency" | "bound" | "ai";
type Engine = "sglang" | "vllm";
type Bound = "memory" | "compute" | "mixed" | "launch";

type Alt = {
  id: string;
  label: string;
  selected: boolean;
  when: string;
};

type GraphNode = {
  id: string;
  title: string;
  file: string;
  latencyUs: number;
  ai: number;
  bound: Bound;
  bwPct: number;
  smPct: number;
  kernel?: string;
  alts?: Alt[];
  stack: string[];
};

const NODES: Record<string, GraphNode> = {
  embed: {
    id: "embed",
    title: "VocabParallelEmbedding",
    file: "sglang/srt/models/glm_moe_dsa.py",
    latencyUs: 18,
    ai: 1.2,
    bound: "memory",
    bwPct: 62,
    smPct: 11,
    kernel: "at::native::index_select / embedding_fwd",
    stack: [
      "GlmMoeDsaForCausalLM.forward",
      "GlmMoeDsaModel.forward",
      "VocabParallelEmbedding.forward",
    ],
  },
  layer: {
    id: "layer",
    title: "GlmMoeDsaDecoderLayer ×78",
    file: "modeling_glm_moe_dsa.py :: GlmMoeDsaDecoderLayer.forward",
    latencyUs: 612,
    ai: 48,
    bound: "mixed",
    bwPct: 54,
    smPct: 41,
    stack: [
      "ModelRunner.forward_decode",
      "GlmMoeDsaForCausalLM.forward",
      "GlmMoeDsaModel.forward",
      "for i in range(78): layers[i].forward",
    ],
  },
  in_norm: {
    id: "in_norm",
    title: "input_layernorm",
    file: "GlmMoeDsaDecoderLayer.forward",
    latencyUs: 9,
    ai: 0.6,
    bound: "memory",
    bwPct: 71,
    smPct: 8,
    kernel: "fused_add_rmsnorm_kernel",
    alts: [
      {
        id: "fused",
        label: "fused_add_rmsnorm (selected)",
        selected: true,
        when: "default SGLang / vLLM fused epilogue",
      },
      {
        id: "triton",
        label: "triton_rmsnorm",
        selected: false,
        when: "fallback; easier to compile, slower on H200",
      },
      {
        id: "torch",
        label: "torch.nn.RMSNorm",
        selected: false,
        when: "eager path; never what you want in serving",
      },
    ],
    stack: [
      "GlmMoeDsaDecoderLayer.forward",
      "self.input_layernorm(hidden)",
      "sglang.srt.layers.layernorm.RMSNorm.forward",
    ],
  },
  qa: {
    id: "qa",
    title: "q_a_proj",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 14,
    ai: 62,
    bound: "compute",
    bwPct: 38,
    smPct: 72,
    kernel: "cutlass::gemm::device::GemmUniversal<fp8>",
    alts: [
      {
        id: "cutlass",
        label: "CUTLASS FP8 GEMM (selected)",
        selected: true,
        when: "Hopper/Blackwell, FP8 weights",
      },
      {
        id: "deepgemm",
        label: "DeepGEMM",
        selected: false,
        when: "often faster on H200 for K-aligned shapes",
      },
      {
        id: "cublas",
        label: "cuBLASLt",
        selected: false,
        when: "PyTorch default; loses fused epilogue",
      },
    ],
    stack: [
      "GlmMoeDsaAttention.forward",
      "self.q_a_proj(hidden)",
      "ColumnParallelLinear.forward",
      "quant_fp8_linear → cutlass gemm",
    ],
  },
  qaln: {
    id: "qaln",
    title: "q_a_layernorm",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 4,
    ai: 0.5,
    bound: "memory",
    bwPct: 68,
    smPct: 7,
    kernel: "rmsnorm_kernel",
    stack: ["GlmMoeDsaAttention.forward", "self.q_a_layernorm(q_lora)"],
  },
  qb: {
    id: "qb",
    title: "q_b_proj",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 16,
    ai: 71,
    bound: "compute",
    bwPct: 34,
    smPct: 78,
    kernel: "cutlass::gemm::device::GemmUniversal<fp8>",
    stack: ["GlmMoeDsaAttention.forward", "self.q_b_proj(q_lora_norm)"],
  },
  kva: {
    id: "kva",
    title: "kv_a_proj_with_mqa",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 11,
    ai: 55,
    bound: "mixed",
    bwPct: 49,
    smPct: 58,
    kernel: "cutlass::gemm::device::GemmUniversal<fp8>",
    stack: [
      "GlmMoeDsaAttention.forward",
      "self.kv_a_proj_with_mqa(hidden)",
      "latent KV compression (MLA)",
    ],
  },
  kvaln: {
    id: "kvaln",
    title: "kv_a_layernorm",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 3,
    ai: 0.5,
    bound: "memory",
    bwPct: 64,
    smPct: 6,
    kernel: "rmsnorm_kernel",
    stack: ["GlmMoeDsaAttention.forward", "self.kv_a_layernorm(kv_lora)"],
  },
  kvb: {
    id: "kvb",
    title: "kv_b_proj",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 7,
    ai: 44,
    bound: "mixed",
    bwPct: 51,
    smPct: 49,
    kernel: "cutlass::gemm::device::GemmUniversal<fp8>",
    stack: ["GlmMoeDsaAttention.forward", "self.kv_b_proj(kv_lora_norm)"],
  },
  rope: {
    id: "rope",
    title: "apply_rotary_pos_emb",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 5,
    ai: 2.1,
    bound: "memory",
    bwPct: 59,
    smPct: 14,
    kernel: "fused_rope_kernel",
    alts: [
      {
        id: "fused",
        label: "fused RoPE (selected)",
        selected: true,
        when: "default in SGLang MLA path",
      },
      {
        id: "torch",
        label: "PyTorch complex mul",
        selected: false,
        when: "eager / debug only",
      },
    ],
    stack: [
      "GlmMoeDsaAttention.forward",
      "apply_rotary_pos_emb(q, k, cos, sin)",
    ],
  },
  indexer: {
    id: "indexer",
    title: "DSA indexer top-k",
    file: "sglang/srt/layers/attention/nsa_backend.py",
    latencyUs: 22,
    ai: 8,
    bound: "memory",
    bwPct: 74,
    smPct: 22,
    kernel: "tilelang_dsa_indexer_topk",
    alts: [
      {
        id: "tilelang",
        label: "tilelang (--dsa-*-backend tilelang)",
        selected: true,
        when: "GLM-5.2 cookbook default on H200/B200 and ROCm",
      },
      {
        id: "flashinfer",
        label: "FlashInfer sparse indexer",
        selected: false,
        when: "NVIDIA SM90+ when FlashInfer sparse MLA is selected",
      },
      {
        id: "triton",
        label: "Triton top-k JIT",
        selected: false,
        when: "CUDA fallback; will not build on gfx950",
      },
      {
        id: "share",
        label: "IndexShare skip (no launch)",
        selected: false,
        when: "layers 1,2,3 of every 4 reuse previous top-k",
      },
    ],
    stack: [
      "GlmMoeDsaAttention.forward",
      "DSABackend.forward(q, kv, page_table)",
      "indexer_kernel → topk_indices",
    ],
  },
  sparse: {
    id: "sparse",
    title: "sparse MLA decode",
    file: "sglang/srt/layers/attention/nsa_backend.py",
    latencyUs: 95,
    ai: 6.4,
    bound: "memory",
    bwPct: 78,
    smPct: 19,
    kernel: "flashinfer::mla_sparse_decode / tilelang_sparse_mla",
    alts: [
      {
        id: "fi_sparse",
        label: "FLASHINFER_MLA_SPARSE",
        selected: true,
        when: "vLLM default for FP8 KV sparse MLA on NVIDIA",
      },
      {
        id: "tilelang",
        label: "tilelang sparse MLA (SGLang)",
        selected: false,
        when: "--dsa-decode-backend tilelang",
      },
      {
        id: "flashmla",
        label: "FLASHMLA_SPARSE",
        selected: false,
        when: "vLLM fallback when FlashInfer sparse is unavailable",
      },
      {
        id: "cutlass_mla",
        label: "CUTLASS_MLA (dense)",
        selected: false,
        when: "dense MLA only — not the DSA path",
      },
      {
        id: "triton_mla",
        label: "TRITON_MLA",
        selected: false,
        when: "portable fallback; usually the slowest decode path",
      },
      {
        id: "aiter",
        label: "ROCM_AITER_MLA_SPARSE",
        selected: false,
        when: "AMD MI3xx with AITER",
      },
    ],
    stack: [
      "GlmMoeDsaAttention.forward",
      "DSABackend.forward",
      "sparse_mla_decode(q, paged_kv, topk_idx)",
      "kernel launch on current stream",
    ],
  },
  oproj: {
    id: "oproj",
    title: "o_proj",
    file: "GlmMoeDsaAttention.forward",
    latencyUs: 13,
    ai: 58,
    bound: "compute",
    bwPct: 36,
    smPct: 69,
    kernel: "cutlass::gemm::device::GemmUniversal<fp8>",
    stack: ["GlmMoeDsaAttention.forward", "self.o_proj(attn_out)"],
  },
  res1: {
    id: "res1",
    title: "residual add",
    file: "GlmMoeDsaDecoderLayer.forward",
    latencyUs: 4,
    ai: 0.3,
    bound: "memory",
    bwPct: 55,
    smPct: 5,
    kernel: "vectorized_elementwise_kernel (often fused into rmsnorm)",
    stack: ["hidden = residual + attn_out"],
  },
  post_norm: {
    id: "post_norm",
    title: "post_attention_layernorm",
    file: "GlmMoeDsaDecoderLayer.forward",
    latencyUs: 8,
    ai: 0.6,
    bound: "memory",
    bwPct: 70,
    smPct: 8,
    kernel: "fused_add_rmsnorm_kernel",
    stack: [
      "GlmMoeDsaDecoderLayer.forward",
      "self.post_attention_layernorm(hidden)",
    ],
  },
  gate: {
    id: "gate",
    title: "MoE router / gate",
    file: "sglang/srt/layers/moe/fused_moe.py",
    latencyUs: 12,
    ai: 18,
    bound: "mixed",
    bwPct: 44,
    smPct: 31,
    kernel: "fused_moe_gate_softmax_topk",
    alts: [
      {
        id: "fused",
        label: "fused gate+softmax+topk (selected)",
        selected: true,
        when: "always-on for serving",
      },
      {
        id: "torch",
        label: "torch.softmax + topk",
        selected: false,
        when: "eager debug",
      },
    ],
    stack: [
      "GlmMoeDsaMoE.forward",
      "self.gate(hidden)",
      "fused_topk(scores, k=8)",
    ],
  },
  shared: {
    id: "shared",
    title: "shared expert (always-on)",
    file: "GlmMoeDsaMoE.forward",
    latencyUs: 38,
    ai: 92,
    bound: "compute",
    bwPct: 41,
    smPct: 81,
    kernel: "cutlass / DeepGEMM SwiGLU fused",
    stack: [
      "GlmMoeDsaMoE.forward",
      "self.shared_experts(hidden)",
      "w1 GEMM → silu_and_mul → w2 GEMM",
    ],
  },
  align: {
    id: "align",
    title: "moe_align_block_size",
    file: "fused_moe.py",
    latencyUs: 6,
    ai: 0.4,
    bound: "launch",
    bwPct: 12,
    smPct: 4,
    kernel: "moe_align_block_size_kernel",
    stack: [
      "fused_moe.forward",
      "permute tokens by expert",
      "moe_align_block_size",
    ],
  },
  w1: {
    id: "w1",
    title: "grouped GEMM w1 (8/256)",
    file: "fused_moe experts",
    latencyUs: 188,
    ai: 140,
    bound: "compute",
    bwPct: 47,
    smPct: 86,
    kernel: "deep_gemm::fp8_grouped_gemm / flashinfer_cutlass_moe",
    alts: [
      {
        id: "deepgemm",
        label: "DeepGEMM grouped (selected)",
        selected: true,
        when: "FP8 MoE on Hopper/Blackwell, good occupancy",
      },
      {
        id: "fi_cutlass",
        label: "FLASHINFER_CUTLASS",
        selected: false,
        when: "vLLM Fp8MoeBackend; strong on SM100",
      },
      {
        id: "fi_trtllm",
        label: "FLASHINFER_TRTLLM",
        selected: false,
        when: "monolithic TRT-LLM experts on Blackwell",
      },
      {
        id: "triton",
        label: "TRITON / BATCHED_TRITON",
        selected: false,
        when: "portable; usually loses to DeepGEMM on H200",
      },
      {
        id: "marlin",
        label: "MARLIN",
        selected: false,
        when: "weight-only INT4/INT8 MoE, not this FP8 checkpoint",
      },
      {
        id: "aiter",
        label: "AITER",
        selected: false,
        when: "ROCm --moe-backend aiter",
      },
    ],
    stack: [
      "GlmMoeDsaMoE.forward",
      "FusedMoE.forward",
      "TritonOrDeepGemmExperts",
      "grouped_gemm(w1)  # 8 active of 256",
    ],
  },
  act: {
    id: "act",
    title: "silu_and_mul",
    file: "fused_moe epilogue",
    latencyUs: 11,
    ai: 1.1,
    bound: "memory",
    bwPct: 66,
    smPct: 12,
    kernel: "silu_and_mul_kernel (often fused into GEMM epilogue)",
    alts: [
      {
        id: "fused",
        label: "GEMM epilogue fused (selected)",
        selected: true,
        when: "CUTLASS/DeepGEMM epilogue",
      },
      {
        id: "standalone",
        label: "standalone silu_and_mul",
        selected: false,
        when: "Triton MoE that does not fuse activation",
      },
    ],
    stack: ["FusedMoE.forward", "silu_and_mul(w1_out)"],
  },
  w2: {
    id: "w2",
    title: "grouped GEMM w2 (8/256)",
    file: "fused_moe experts",
    latencyUs: 164,
    ai: 128,
    bound: "compute",
    bwPct: 45,
    smPct: 84,
    kernel: "deep_gemm::fp8_grouped_gemm",
    stack: ["FusedMoE.forward", "grouped_gemm(w2)"],
  },
  combine: {
    id: "combine",
    title: "moe_combine + shared add",
    file: "GlmMoeDsaMoE.forward",
    latencyUs: 9,
    ai: 1.4,
    bound: "memory",
    bwPct: 61,
    smPct: 11,
    kernel: "moe_sum_kernel",
    stack: [
      "unpermute expert outputs",
      "weighted sum by router scores",
      "add shared expert",
    ],
  },
  mtp: {
    id: "mtp",
    title: "MTP speculative head",
    file: "GlmMoeDsaForCausalLM.forward",
    latencyUs: 86,
    ai: 40,
    bound: "mixed",
    bwPct: 52,
    smPct: 38,
    kernel: "same kernels as a shallow decoder layer",
    stack: [
      "speculative_algorithm=EAGLE/MTP",
      "mtp_layer.forward(hidden)",
      "draft token logits",
    ],
  },
  lm: {
    id: "lm",
    title: "lm_head",
    file: "GlmMoeDsaForCausalLM.forward",
    latencyUs: 24,
    ai: 36,
    bound: "mixed",
    bwPct: 48,
    smPct: 44,
    kernel: "cutlass gemm vocab projection",
    stack: ["ParallelLMHead.forward", "hidden @ vocab_weight.T"],
  },
};

const MAX_LAT = 188;

const DEPTH_GRAPHS: Record<
  Depth,
  { nodes: string[]; edges: Array<{ from: string; to: string }> }
> = {
  architecture: {
    nodes: ["embed", "layer", "mtp", "lm"],
    edges: [
      { from: "embed", to: "layer" },
      { from: "layer", to: "mtp" },
      { from: "layer", to: "lm" },
    ],
  },
  modules: {
    nodes: ["in_norm", "sparse", "res1", "post_norm", "w1", "combine"],
    edges: [
      { from: "in_norm", to: "sparse" },
      { from: "sparse", to: "res1" },
      { from: "res1", to: "post_norm" },
      { from: "post_norm", to: "w1" },
      { from: "w1", to: "combine" },
    ],
  },
  functions: {
    nodes: [
      "in_norm",
      "qa",
      "qb",
      "kva",
      "kvb",
      "rope",
      "indexer",
      "sparse",
      "oproj",
      "res1",
      "post_norm",
      "gate",
      "shared",
      "w1",
      "combine",
    ],
    edges: [
      { from: "in_norm", to: "qa" },
      { from: "in_norm", to: "kva" },
      { from: "qa", to: "qb" },
      { from: "kva", to: "kvb" },
      { from: "qb", to: "rope" },
      { from: "kvb", to: "rope" },
      { from: "rope", to: "indexer" },
      { from: "indexer", to: "sparse" },
      { from: "sparse", to: "oproj" },
      { from: "oproj", to: "res1" },
      { from: "res1", to: "post_norm" },
      { from: "post_norm", to: "gate" },
      { from: "post_norm", to: "shared" },
      { from: "gate", to: "w1" },
      { from: "w1", to: "combine" },
      { from: "shared", to: "combine" },
    ],
  },
  kernels: {
    nodes: [
      "in_norm",
      "qa",
      "qaln",
      "qb",
      "kva",
      "kvaln",
      "kvb",
      "rope",
      "indexer",
      "sparse",
      "oproj",
      "res1",
      "post_norm",
      "gate",
      "align",
      "shared",
      "w1",
      "act",
      "w2",
      "combine",
    ],
    edges: [
      { from: "in_norm", to: "qa" },
      { from: "in_norm", to: "kva" },
      { from: "qa", to: "qaln" },
      { from: "qaln", to: "qb" },
      { from: "kva", to: "kvaln" },
      { from: "kvaln", to: "kvb" },
      { from: "qb", to: "rope" },
      { from: "kvb", to: "rope" },
      { from: "rope", to: "indexer" },
      { from: "indexer", to: "sparse" },
      { from: "sparse", to: "oproj" },
      { from: "oproj", to: "res1" },
      { from: "res1", to: "post_norm" },
      { from: "post_norm", to: "gate" },
      { from: "post_norm", to: "shared" },
      { from: "gate", to: "align" },
      { from: "align", to: "w1" },
      { from: "w1", to: "act" },
      { from: "act", to: "w2" },
      { from: "w2", to: "combine" },
      { from: "shared", to: "combine" },
    ],
  },
};

const BOUND_COLOR: Record<Bound, Color> = {
  memory: "yellow",
  compute: "blue",
  mixed: "orange",
  launch: "gray",
};

function boundLabel(b: Bound): string {
  if (b === "memory") return "memory-bound";
  if (b === "compute") return "compute-bound";
  if (b === "launch") return "launch-bound";
  return "mixed / ridge";
}

function heatFill(
  node: GraphNode,
  heat: Heat,
  theme: ReturnType<typeof useHostTheme>,
): string {
  if (heat === "none") return theme.fill.tertiary;
  if (heat === "bound") return theme.category[BOUND_COLOR[node.bound]];
  if (heat === "ai") {
    if (node.ai < 8) return theme.category.yellow;
    if (node.ai < 80) return theme.category.orange;
    return theme.category.blue;
  }
  const t = node.latencyUs / MAX_LAT;
  if (t > 0.55) return theme.category.red;
  if (t > 0.25) return theme.category.orange;
  if (t > 0.1) return theme.category.yellow;
  return theme.fill.tertiary;
}

function heatText(
  node: GraphNode,
  heat: Heat,
  theme: ReturnType<typeof useHostTheme>,
): string {
  if (heat === "none") return theme.text.primary;
  const hot =
    heat === "latency"
      ? node.latencyUs / MAX_LAT > 0.55
      : heat === "bound"
        ? node.bound === "compute"
        : node.ai >= 80;
  return hot ? theme.bg.editor : theme.text.primary;
}

function FlowGraph({
  depth,
  heat,
  selected,
  onSelect,
}: {
  depth: Depth;
  heat: Heat;
  selected: string;
  onSelect: (id: string) => void;
}) {
  const theme = useHostTheme();
  const g = DEPTH_GRAPHS[depth];
  const layout = computeDAGLayout({
    nodes: g.nodes.map((id) => ({ id })),
    edges: g.edges,
    direction: "vertical",
    nodeWidth: 168,
    nodeHeight: 44,
    rankGap: 36,
    nodeGap: 16,
    padding: 12,
  });
  const pos = Object.fromEntries(layout.nodes.map((n) => [n.id, n]));

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: layout.height,
        overflow: "auto",
        background: theme.bg.editor,
        border: `1px solid ${theme.stroke.tertiary}`,
        borderRadius: 8,
      }}
    >
      <svg
        width={layout.width}
        height={layout.height}
        style={{ position: "absolute", inset: 0 }}
      >
        {layout.edges.map((e, i) => (
          <line
            key={`${e.from}-${e.to}-${i}`}
            x1={e.sourceX}
            y1={e.sourceY}
            x2={e.targetX}
            y2={e.targetY}
            stroke={theme.stroke.secondary}
            strokeWidth={1}
            strokeDasharray={e.isBackEdge ? "4 3" : undefined}
          />
        ))}
      </svg>
      {g.nodes.map((id) => {
        const n = NODES[id];
        const p = pos[id];
        const isSel = selected === id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onSelect(id)}
            style={{
              position: "absolute",
              left: p.x,
              top: p.y,
              width: 168,
              height: 44,
              margin: 0,
              padding: "6px 8px",
              border: `1px solid ${isSel ? theme.accent.primary : theme.stroke.secondary}`,
              borderRadius: 6,
              background: heatFill(n, heat, theme),
              color: heatText(n, heat, theme),
              cursor: "pointer",
              textAlign: "left",
              font: "inherit",
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                lineHeight: "14px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {n.title}
            </div>
            <div
              style={{
                fontSize: 10,
                opacity: 0.85,
                lineHeight: "14px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {n.kernel ? n.kernel.split(" ")[0] : n.file.split("::").pop()}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function Inspector({ id, engine }: { id: string; engine: Engine }) {
  const n = NODES[id];
  const theme = useHostTheme();
  return (
    <Stack gap={12}>
      <Row gap={8} align="center" wrap>
        <Text weight="semibold">{n.title}</Text>
        <Swatch color={BOUND_COLOR[n.bound]} />
        <Text size="small" tone="secondary">
          {boundLabel(n.bound)}
        </Text>
      </Row>
      <Text size="small" tone="secondary">
        {n.file}
      </Text>
      <Grid columns={4} gap={12}>
        <Stat value={`${n.latencyUs} µs`} label="GPU time this decode step" />
        <Stat value={`${n.ai.toFixed(1)}`} label="FLOP / byte (AI)" />
        <Stat value={`${n.bwPct}%`} label="HBM bandwidth util" />
        <Stat value={`${n.smPct}%`} label="SM compute util" />
      </Grid>
      {n.kernel ? (
        <Callout tone="info" title="Kernel that actually launched">
          <Text>
            {engine === "sglang" ? "SGLang" : "vLLM"} dispatched `{n.kernel}`.
            This name is what you would see in nsys / Perfetto, mapped back onto
            the model node instead of a timeline row.
          </Text>
        </Callout>
      ) : null}
      <H3>Call stack (code-level, nothing dropped)</H3>
      <Table
        headers={["Frame", "Symbol"]}
        rows={n.stack.map((frame, i) => [String(i), frame])}
        columnAlign={["right", "left"]}
        striped
      />
      {n.alts && n.alts.length > 0 ? (
        <>
          <H3>Alternative kernels / backends</H3>
          <Table
            headers={["Backend", "Status", "When you would pick it"]}
            rows={n.alts.map((a) => [
              a.label,
              a.selected ? "running now" : "available",
              a.when,
            ])}
            rowTone={n.alts.map((a) => (a.selected ? "success" : "neutral"))}
            striped
          />
        </>
      ) : (
        <Text size="small" tone="secondary">
          No registered alternatives for this op — usually a fused epilogue or
          a tiny elementwise that rides along with the previous GEMM.
        </Text>
      )}
      <Text size="small" tone="tertiary">
        Source of alternatives: vLLM AttentionBackend registry + Fp8MoeBackend
        oracle, SGLang `--dsa-*-backend` / `--moe-runner-backend`, not the
        trace. A trace only tells you what ran, never what else could have.
      </Text>
      <div
        style={{
          height: 8,
          borderRadius: 4,
          background: theme.fill.quaternary,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.min(100, (n.latencyUs / MAX_LAT) * 100)}%`,
            height: "100%",
            background: theme.accent.primary,
          }}
        />
      </div>
      <Text size="small" tone="tertiary">
        Bar length is GPU time vs the hottest kernel in this layer (grouped GEMM
        w1 = {MAX_LAT} µs).
      </Text>
    </Stack>
  );
}

export default function Glm52CodeLevelGraph() {
  const [depth, setDepth] = useCanvasState<Depth>("depth", "kernels");
  const [heat, setHeat] = useCanvasState<Heat>("heat", "latency");
  const [engine, setEngine] = useCanvasState<Engine>("engine", "sglang");
  const [selected, setSelected] = useCanvasState("selected", "sparse");

  const theme = useHostTheme();
  const node = NODES[selected] ?? NODES.sparse;

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Code-level model view, not HF Viewer</H1>
        <Text>
          HF Viewer is not very detailed — it is only a high-level outline.
          The view you want is code-level: module → function → backend →
          kernel, with latency and bound heatmaps on the graph itself.
        </Text>
        <Text>
          [HF Viewer](https://hfviewer.com/zai-org/GLM-5.2-FP8) is a
          config-derived outline. It names blocks (MLA, MoE, RMSNorm) from
          `config.json` and modeling code. It never runs the model, never walks
          vLLM/SGLang, and never names a kernel. The graph below is a
          code-level execution graph: the same GLM-5.2 layer as the functions
          that ran, the backend that dispatched them, the kernel that launched,
          the other kernels that could have, and latency / memory-vs-compute
          painted on those nodes — not on a Perfetto timeline.
        </Text>
        <Text size="small" tone="secondary">
          Open this canvas beside the chat. Use the depth control to go from
          hfviewer-like architecture down to CUDA/Triton launches, then click a
          node. Numbers are an illustrative decode step for one GLM-5.2 layer
          (batch 8, ~8k cached context, H200-class, FP8, SGLang DSA tilelang).
          Architecture, file names, backend registries, and kernel families are
          real; timings are typical-order walkthrough data, not a live nsys
          capture.
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="78" label="Decoder layers (IndexShare every 4)" />
        <Stat value="256 / 8+1" label="Routed experts / active + shared" />
        <Stat value="6" label="Correlation layers to join" />
        <Stat
          value={`${node.latencyUs} µs`}
          label={`Selected: ${node.title}`}
          tone={node.bound === "memory" ? "warning" : "info"}
        />
      </Grid>

      <Callout tone="warning" title="HF Viewer stops one layer too high">
        Replacing huggingface.co with hfviewer.com gives you module names and
        tensor ranks. It does not execute the model, does not record vLLM or
        SGLang, and has no kernel. The missing product is a join: architecture
        IR × serving-engine IR × profiler IR, with the architecture graph as
        the canvas and Perfetto as a drill-through, not the home screen.
      </Callout>

      <H2>What each zoom level actually contains</H2>
      <Table
        headers={["Zoom", "What you see", "Where it comes from", "HF Viewer?"]}
        rows={[
          [
            "Architecture",
            "embed → decoder×78 → MTP → lm_head",
            "config.json + modeling_glm_moe_dsa.py",
            "Yes, this is hfviewer",
          ],
          [
            "Modules",
            "RMSNorm, MLA, residual, MoE",
            "nn.Module tree / torch.fx",
            "Partially, named blocks only",
          ],
          [
            "Functions",
            "q_a_proj, kv_a_proj_with_mqa, DSA indexer, fused_topk…",
            "Python + C++ stack (Kineto, NVTX, torch.profiler)",
            "No",
          ],
          [
            "Kernels",
            "cutlass gemm, tilelang indexer, grouped GEMM w1/w2…",
            "nsys CUDA trace, CUPTI kernel names",
            "No",
          ],
          [
            "Counters",
            "AI, HBM%, SM%, bound class",
            "Nsight Compute / ncu roofline, not Perfetto",
            "No",
          ],
          [
            "Alternatives",
            "FLASHINFER_MLA_SPARSE vs TILELANG vs TRITON_MLA",
            "Engine registries (never in a trace)",
            "No",
          ],
        ]}
        rowTone={[
          "neutral",
          "info",
          "warning",
          "warning",
          "danger",
          "danger",
        ]}
        striped
      />

      <H2>One GLM-5.2 decoder layer — click a node</H2>
      <Text>
        Switch depth from architecture (hfviewer-like) to kernels (every launch
        in this layer). Heatmaps color the same graph: red/orange = hot, yellow
        = memory-bound or low AI, blue = compute-bound or high AI. Perfetto is
        still useful as a time axis; it is the wrong shape for “which module is
        the problem.”
      </Text>
      <Row gap={16} align="end" wrap>
        <Stack gap={4} style={{ minWidth: 180 }}>
          <Text size="small" tone="secondary">
            Depth
          </Text>
          <Select
            value={depth}
            onChange={(v) => setDepth(v as Depth)}
            options={[
              { value: "architecture", label: "Architecture (hfviewer)" },
              { value: "modules", label: "Modules (nn.Module)" },
              { value: "functions", label: "Functions (Python/C++)" },
              { value: "kernels", label: "Kernels (CUDA/Triton)" },
            ]}
          />
        </Stack>
        <Stack gap={4} style={{ minWidth: 180 }}>
          <Text size="small" tone="secondary">
            Heatmap on the graph
          </Text>
          <Select
            value={heat}
            onChange={(v) => setHeat(v as Heat)}
            options={[
              { value: "none", label: "Structure only" },
              { value: "latency", label: "GPU time" },
              { value: "bound", label: "Memory vs compute" },
              { value: "ai", label: "Arithmetic intensity" },
            ]}
          />
        </Stack>
        <Stack gap={4} style={{ minWidth: 180 }}>
          <Text size="small" tone="secondary">
            Serving engine
          </Text>
          <Select
            value={engine}
            onChange={(v) => setEngine(v as Engine)}
            options={[
              { value: "sglang", label: "SGLang" },
              { value: "vllm", label: "vLLM" },
            ]}
          />
        </Stack>
        <Row gap={8} wrap>
          <Pill
            active={heat === "latency"}
            onClick={() => setHeat("latency")}
          >
            Latency
          </Pill>
          <Pill active={heat === "bound"} onClick={() => setHeat("bound")}>
            Bound
          </Pill>
          <Pill active={heat === "ai"} onClick={() => setHeat("ai")}>
            AI
          </Pill>
        </Row>
      </Row>

      <Row gap={12} align="center" wrap>
        <Swatch color="red" />
        <Text size="small">Hot (high µs)</Text>
        <Swatch color="yellow" />
        <Text size="small">Memory-bound / low AI</Text>
        <Swatch color="orange" />
        <Text size="small">Mixed / ridge</Text>
        <Swatch color="blue" />
        <Text size="small">Compute-bound / high AI</Text>
        <Swatch color="gray" />
        <Text size="small">Launch-bound</Text>
      </Row>

      <Grid columns="minmax(0, 1.15fr) minmax(320px, 0.85fr)" gap={20}>
        <FlowGraph
          depth={depth}
          heat={heat}
          selected={selected}
          onSelect={setSelected}
        />
        <Card>
          <CardHeader trailing={<Text size="small">{engine}</Text>}>
            Selected node
          </CardHeader>
          <CardBody>
            <Inspector id={node.id} engine={engine} />
          </CardBody>
        </Card>
      </Grid>

      <H2>GPU time in this layer (illustrative decode step)</H2>
      <BarChart
        height={220}
        horizontal
        categories={[
          "grouped GEMM w1",
          "grouped GEMM w2",
          "sparse MLA",
          "shared expert",
          "DSA indexer",
          "Q/KV/O projections",
          "router + align",
          "norms / RoPE / residual",
        ]}
        series={[
          {
            name: "GPU time (µs)",
            data: [188, 164, 95, 38, 22, 61, 18, 29],
            tone: "info",
          },
        ]}
        valueSuffix=" µs"
      />
      <Text size="small" tone="tertiary">
        Source: walkthrough decode step for one of 78 layers · H200-class · FP8
        · batch 8 · ~8k KV. MoE grouped GEMMs dominate compute; sparse MLA and
        the DSA indexer stay memory-bound even when GEMMs are on the compute
        roof.
      </Text>

      <UsageBar
        total={612}
        topLeftLabel="Layer GPU time share"
        topRightLabel="612 µs / layer (illustrative)"
        segments={[
          { id: "moe", value: 352, color: "blue" },
          { id: "attn", value: 95, color: "yellow" },
          { id: "proj", value: 61, color: "cyan" },
          { id: "shared", value: 38, color: "purple" },
          { id: "dsa", value: 22, color: "orange" },
          { id: "other", value: 44, color: "gray" },
        ]}
      />

      <H2>Call tree that HF Viewer never emits</H2>
      <Text>
        This is the same layer as a stack, including scheduler frames that are
        not “model architecture” but do steal GPU idle time. Expanding a row is
        the code-level equivalent of hfviewer’s granularity slider.
      </Text>
      <CollapsibleSection
        title="SGLang Scheduler.run_batch / vLLM Worker.execute_model"
        count={1}
        trailing={<Text size="small">engine, not in hfviewer</Text>}
        defaultOpen
      >
        <CollapsibleSection
          title="ModelRunner.forward_decode  (CUDA graph replay if captured)"
          count={1}
          defaultOpen
        >
          <CollapsibleSection
            title="GlmMoeDsaForCausalLM.forward"
            count={3}
            defaultOpen
          >
            <CollapsibleSection title="VocabParallelEmbedding.forward" count={1}>
              <Text size="small">
                Kernel: embedding gather. Bound: memory. Alternative: fused
                embedding+scale on some ROCm paths.
              </Text>
            </CollapsibleSection>
            <CollapsibleSection
              title="GlmMoeDsaDecoderLayer.forward  ×78  (IndexShare on 3/4 sparse layers)"
              count={6}
              defaultOpen
            >
              <CollapsibleSection title="input_layernorm → fused_add_rmsnorm_kernel">
                <Text size="small">
                  Function: RMSNorm.forward. Often fused with the previous
                  residual add so Perfetto shows one kernel for two Python
                  lines.
                </Text>
              </CollapsibleSection>
              <CollapsibleSection
                title="GlmMoeDsaAttention.forward  (MLA + DSA)"
                count={8}
                defaultOpen
              >
                <Text size="small">
                  q_a_proj → q_a_layernorm → q_b_proj; kv_a_proj_with_mqa →
                  kv_a_layernorm → kv_b_proj; apply_rotary_pos_emb;
                  DSABackend.indexer; sparse_mla_decode; o_proj.
                </Text>
                <Text size="small" tone="secondary">
                  Click “sparse MLA decode” or “DSA indexer top-k” on the graph
                  to see the exact kernel and the vLLM/SGLang alternatives
                  (FlashInfer sparse, tilelang, FlashMLA sparse, Triton MLA,
                  AITER).
                </Text>
              </CollapsibleSection>
              <CollapsibleSection title="residual add (may be fused)">
                <Text size="small">
                  Python still executes `hidden = residual + attn_out`. The
                  CUDA launch may have already happened inside the next
                  rmsnorm. Code-level view must show both the Python line and
                  the fused kernel, or you will “lose” the add.
                </Text>
              </CollapsibleSection>
              <CollapsibleSection title="post_attention_layernorm" />
              <CollapsibleSection
                title="GlmMoeDsaMoE.forward"
                count={6}
                defaultOpen
              >
                <Text size="small">
                  gate → fused_topk → moe_align_block_size → shared_experts
                  (always-on) → grouped GEMM w1 (8 of 256) → silu_and_mul →
                  grouped GEMM w2 → moe_combine.
                </Text>
              </CollapsibleSection>
              <CollapsibleSection title="residual add" />
            </CollapsibleSection>
            <CollapsibleSection title="MTP head + ParallelLMHead.forward">
              <Text size="small">
                Speculative decoding is a sibling of the target model, not a
                block inside layer 78. HF Viewer often omits it because it is a
                serving feature, not a Hugging Face module.
              </Text>
            </CollapsibleSection>
          </CollapsibleSection>
        </CollapsibleSection>
      </CollapsibleSection>

      <H2>The join you have to build — traces are not a model graph</H2>
      <Text>
        Perfetto is a timeline of events. A model view is a graph of
        *operations*. They only line up if every kernel launch carries a stable
        op id that also exists on the architecture IR. That is the whole
        product.
      </Text>
      <Grid columns={3} gap={16}>
        <Card>
          <CardHeader>1. Architecture IR</CardHeader>
          <CardBody>
            <Text>
              Parse `config.json` + `modeling_glm_moe_dsa.py` (or torch.fx /
              dynamo) into nodes with stable ids:
              `layers.17.self_attn.indexer`. This is all hfviewer does.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>2. Engine IR</CardHeader>
          <CardBody>
            <Text>
              Walk vLLM/SGLang registries at init: which AttentionBackend,
              which Fp8MoeBackend, whether IndexShare skips the indexer this
              layer, whether the CUDA graph captured the decode path. This is
              where alternatives live.
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>3. Profiler IR</CardHeader>
          <CardBody>
            <Text>
              NVTX ranges around each `nn.Module.forward`, Kineto
              `record_function`, and CUPTI kernel names. nsys gives duration;
              ncu gives AI / HBM% / SM% for the heatmap. Without NVTX, you
              cannot map `fused_moe_kernel_16` back to layer 17.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H3>Correlation key (what must be on every event)</H3>
      <Table
        headers={["Field", "Example", "Used for"]}
        rows={[
          [
            "op_id",
            "layers.17.self_attn.sparse_mla",
            "Which node on the model graph to color",
          ],
          [
            "module_fqn",
            "model.layers.17.self_attn",
            "Python module tree / fx node",
          ],
          [
            "function",
            "DSABackend.forward",
            "Engine frame hfviewer never has",
          ],
          [
            "kernel",
            "flashinfer::mla_sparse_decode",
            "Exact launch; Perfetto row name",
          ],
          [
            "backend",
            "tilelang | FLASHINFER_MLA_SPARSE",
            "Selected alternative",
          ],
          [
            "phase",
            "decode / prefill / mtp_draft",
            "Same op, different kernels",
          ],
          [
            "layer_idx + indexshare",
            "17, reuse_from=16",
            "Indexer present vs skipped",
          ],
          [
            "ai, bytes, flops, dur",
            "6.4 FLOP/B, 95 µs",
            "Heatmaps: latency, bound, AI",
          ],
        ]}
        striped
      />

      <H2>Tools today vs this view</H2>
      <Table
        headers={["Tool", "Shape", "Misses"]}
        rows={[
          [
            "HF Viewer / model-unfolder",
            "Static architecture graph",
            "No runtime, no vLLM/SGLang, no kernels, no heatmaps",
          ],
          [
            "Netron / torch.fx draw",
            "Static ops graph",
            "Serving fusions and paged KV are invisible",
          ],
          [
            "Perfetto / torch.profiler",
            "Timeline",
            "You cannot see the model; fusions hide Python ops",
          ],
          [
            "Nsight Systems",
            "Timeline + CUDA",
            "Kernel names, still not a model graph",
          ],
          [
            "Nsight Compute",
            "Per-kernel roofline",
            "One kernel at a time; no layer context",
          ],
          [
            "vLLM/SGLang gputrc2graph.py",
            "Stacked bars by kernel category",
            "Categories, not the GLM layer tree",
          ],
          [
            "This view (to build)",
            "Architecture graph + attributes",
            "Requires the join keys above; not a website you can open",
          ],
        ]}
        striped
      />

      <H2>Practical way to get here on GLM-5.2</H2>
      <Text>
        You do not start from hfviewer. You start from a traced serving run,
        then *project* the trace onto the module tree.
      </Text>
      <Table
        headers={["Step", "Do this", "You get"]}
        rows={[
          [
            "1",
            "Serve with SGLang (`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`) or vLLM 0.23+ GLM-5.2 recipe",
            "The real kernels, not Hugging Face eager",
          ],
          [
            "2",
            "Wrap `nn.Module.forward` and backend `forward` with NVTX / `torch.profiler.record_function(op_id)`",
            "Every kernel nested under a stable op_id",
          ],
          [
            "3",
            "nsys profile -t cuda,nvtx,osrt  (and dump chrome/Perfetto JSON)",
            "Durations + kernel names",
          ],
          [
            "4",
            "ncu on the top kernels (sparse MLA, grouped GEMM, indexer)",
            "AI, HBM%, SM% → bound class",
          ],
          [
            "5",
            "Dump engine registries at init (attention backend, MoE oracle, IndexShare map)",
            "Alternative options per node",
          ],
          [
            "6",
            "Join on op_id; render the module tree; color by the heatmap you picked",
            "The graph on this canvas, with live numbers",
          ],
        ]}
        columnAlign={["right", "left", "left"]}
        striped
      />

      <Callout tone="info" title="Fusions are why a naive stack dump lies">
        `residual + x` and `RMSNorm` often become one `fused_add_rmsnorm`
        kernel. `silu_and_mul` often dies as a CUTLASS epilogue. CUDA graphs
        collapse hundreds of launches into `cudaGraphLaunch`. The code-level
        view must keep the Python functions *and* show they fused, otherwise a
        perf engineer will hunt a kernel that no longer exists.
      </Callout>

      <Divider />
      <Text size="small" tone="tertiary">
        GLM-5.2 architecture from zai-org docs (MLA + DSA + IndexShare, 256
        experts / 8 active + 1 shared, MTP). Backend names from vLLM
        AttentionBackend / Fp8MoeBackend registries and the SGLang GLM-5.2
        cookbook. Heatmap values are a walkthrough, not a captured profile.
      </Text>
      <Text size="small" tone="tertiary">
        Related: [HF Viewer GLM-5.2-FP8](https://hfviewer.com/zai-org/GLM-5.2-FP8)
        · [SGLang GLM-5.2 cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2)
        · [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/)
      </Text>
    </Stack>
  );
}
