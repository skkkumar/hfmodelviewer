# HF Perf Viewer — build plan v2

Supersedes v1. The static architecture graph is no longer the product; it's the
**coordinate system**. Everything else pins to it.

Target user: a performance engineer optimizing an LLM for a specific serving
stack. The job to be done is *"why is this slow, where exactly, and did my change
help or hurt."*

---

## 1. Reframe

Four data planes, joined on the same node ids:

| Plane | Source | Gives you |
|---|---|---|
| **Static** | HF config + safetensors index + meta-device module tree | The graph. Node ids, shapes, param counts, dtypes, MoE topology. |
| **Analytical** | Derived from static + a run config | Theoretical FLOPs, bytes moved, arithmetic intensity, roofline bound per node. **Free** — no execution needed. |
| **Measured** | torch profiler / nsys / OTLP from vLLM/SGLant/atom | Wall time, kernel names, occupancy, achieved bandwidth, comm time. |
| **Config** | Serving flags, env, model revision, hardware | What actually differed between two runs. |

The product is the **join**. Netron has plane 1. Perfetto/HTA have plane 3.
Nothing joins them, and nobody at all does plane 2 × plane 3, which is where the
insight lives: *the gap between what the op should cost and what it did cost.*

---

## 2. The hard problem: kernel → node attribution

Be clear-eyed. This is the make-or-break, and it's harder than it looks, because
**vLLM and SGLang do not run the eager `nn.Module` graph on the hot path.**
They run CUDA graph replays, custom fused kernels (FlashAttention, FlashInfer,
paged attention, fused MoE), and `torch.compile` regions. Module boundaries are
largely erased by the time you're looking at
`sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128`.

Three attribution tiers. Support all three; degrade gracefully.

### Tier 1 — NVTX instrumentation (best, needs deployment access)

The serving stacks still define Python `nn.Module`s at layer granularity
(`LlamaAttention`, `FusedMoE`, ...). Wrap their `forward` with NVTX ranges keyed
to your static node ids:

```python
import torch.cuda.nvtx as nvtx

def instrument(model, id_of):
    for name, mod in model.named_modules():
        fwd = mod.forward
        def wrapped(*a, _f=fwd, _n=id_of(name), **kw):
            nvtx.range_push(_n)
            try: return _f(*a, **kw)
            finally: nvtx.range_pop()
        mod.forward = wrapped
```

Ship this as a small importable shim (`perfviewer.instrument_vllm()`), not a fork.

Then `nsys profile --cuda-graph-trace=node --trace=cuda,nvtx,osrt,cudnn,cublas`.
`--cuda-graph-trace=node` is essential: without it, a whole graph replay collapses
into one opaque event and per-layer attribution is gone.

Caveat to surface in the UI: CUDA-graph-captured runs may need `--enforce-eager`
for clean attribution, and eager mode changes the performance you're measuring.
So the tool needs an explicit **"attribution run"** vs **"measurement run"**
distinction, and should map attribution learned in eager mode onto graph-mode
kernel sequences by launch-order alignment.

### Tier 2 — torch profiler module hierarchy

`torch.profiler.profile(with_modules=True, record_shapes=True, with_stack=True)`
records module qualnames in the op stack, and `correlation_id` links CPU op →
CUDA kernel. Works well for eager, partially for compiled regions, poorly inside
fused custom kernels. vLLM/SGLang both expose `/start_profile` and
`/stop_profile` endpoints (set `VLLM_TORCH_PROFILER_DIR`) which emit exactly
this. Lowest-friction path — start here.

### Tier 3 — structural inference (no instrumentation)

Given an opaque kernel timeline, exploit the fact that a decoder stack is
*N identical repetitions*:

1. Extract the kernel-name + launch-config sequence for one step
2. Find the dominant period via autocorrelation on the sequence → that's one
   decoder layer, and the period count should equal `num_hidden_layers`
3. Align one period against the static graph's per-layer node sequence, using
   shape signatures from launch dims and the analytical FLOP ordering
4. Everything before the first period is embedding; after the last is norm + head

Crude, needs zero cooperation from the serving stack, and works on a trace
someone emailed you. Confidence-scored, and the UI must show attribution
confidence per node rather than pretending certainty.

---

## 3. The analytical model (do this early — it's the differentiator)

Every static node already carries shapes. Given a run config
(batch size `B`, sequence length `S`, phase, dtype, TP degree), compute expected
cost with no execution at all:

| Node kind | FLOPs | Bytes moved |
|---|---|---|
| Linear `[M,K]→[M,N]` | `2·M·K·N` | `(M·K + K·N + M·N)·s` |
| Attention prefill | `4·B·H·S²·D` | activations + `2·B·H_kv·S·D·s` |
| Attention decode (step) | `4·B·H·S_ctx·D` | KV read `2·B·H_kv·S_ctx·D·s` dominates |
| MoE FFN | `2·B·S·k_active·(3·d·d_ff)` | weights of *touched* experts + activations |
| RMSNorm / residual | `~5·B·S·d` | `2·B·S·d·s` |

Then:
- **Arithmetic intensity** = FLOPs / bytes
- Compare against hardware ridge point (peak FLOPs / peak HBM BW) → memory-bound
  or compute-bound, analytically, before you ever run anything
- With a measured time, derive **achieved** TFLOP/s and GB/s → % of roofline

Two facts this immediately surfaces, which are exactly what a perf engineer wants:

- **Decode is bandwidth-bound at low batch.** A `[B,K]×[K,N]` GEMV has
  AI ≈ `2B`. At `B=1` you are burning ~0.4% of an H100's compute. The tool should
  say so on the node, in red.
- **GQA ratio sets attention decode intensity.** AI ≈ `2·H/H_kv`. A model with
  `H=64, H_kv=8` has 8× the attention arithmetic intensity of MHA. This is why
  MLA/GQA exist, and the tool can show the number directly from `config.json`.

This is buildable in phase 1 with zero traces and is already useful on its own —
"what will this model cost on this GPU at this batch size" is a question people
pay to answer.

---

## 4. Prefill vs decode must be first-class

Never aggregate them. Different regimes entirely: prefill is compute-bound
batched GEMM; decode is memory-bound GEMV + KV-cache streaming. With continuous
batching and chunked prefill, a single step contains both — so the IR needs:

```jsonc
"phases": ["prefill", "decode", "mixed"],
"measurements": {
  "node_id": {
    "prefill": { "p50_us": 0, "p95_us": 0, "n": 0, "kernels": [] },
    "decode":  { "p50_us": 0, "p95_us": 0, "n": 0, "kernels": [] }
  }
}
```

Every view is scoped to a phase. A tool that reports one blended number per node
is actively misleading and a perf engineer will drop it in five minutes.

---

## 5. Measurement IR

Extends the static IR from v1. Static IR is keyed by `(repo, revision)`; a run is
keyed by `(static_ir_id, run_config_hash)`.

```jsonc
{
  "run_id": "...",
  "static_ir": "nvidia/GLM-5.2-NVFP4@<sha>",
  "stack": { "name": "vllm", "version": "...", "flags": {...}, "env": {...} },
  "hardware": { "gpu": "H100-SXM", "count": 8, "peak_tflops": 989, "peak_bw_gbs": 3350 },
  "parallelism": { "tp": 8, "pp": 1, "ep": 1, "dp": 1 },
  "workload": { "batch": 32, "input_len": 2048, "output_len": 256, "arrival": "..." },
  "totals": { "ttft_ms": {...}, "tpot_ms": {...}, "throughput_tok_s": 0 },
  "attribution": { "tier": 1, "confidence": 0.94, "unattributed_pct": 3.2 },
  "measurements": { /* per node, per phase, distributions not point values */ },
  "comm": [ { "op": "all_reduce", "after_node": "...", "bytes": 0, "p50_us": 0 } ],
  "gaps": [ { "kind": "launch_gap|sync|host_bound", "p50_us": 0, "after_node": "..." } ]
}
```

Three things people forget and you shouldn't:

- **Distributions, not point values.** Traces are noisy. Store p50/p95/p99 + n
  across steps. A single-step number cannot support a regression claim.
- **`unattributed_pct`.** Always show what fraction of GPU time you *couldn't*
  pin to a node. If it's 40%, the user needs to know the picture is a lie.
- **Gaps are the finding.** Bubbles between kernels — launch overhead, host
  sync, scheduler stalls — are frequently the actual problem in serving, and
  they belong to no node. Model them as first-class edge-level entities.

For multi-GPU: one trace per rank, merged on a common timebase, with NCCL
kernels forming the comm plane. Rank skew and exposed (non-overlapped) comm are
usually the top finding in TP deployments — surface "exposed comm %" prominently.

---

## 6. Regression diff

The hard part is not the visualization, it's making the comparison *valid*.

**Alignment.** If both runs share a static IR, align by node id. If not (different
quantization, TP degree, kernel backend), align by structural hash from the v1
collapse logic and explicitly render nodes that exist in only one run.

**Normalization.** Never compare raw wall time across runs with different batch
or sequence config. Normalize to per-token, per-request, or per-step and make the
choice visible and switchable.

**Significance.** Given per-node distributions and step counts, compute whether a
delta clears the noise floor. Bootstrap CI on the median is enough. Anything
inside the CI renders grey, not red or green. This single decision is what makes
the tool trustworthy.

**Attribution of the top-line delta.** The headline output is a waterfall:

```
TPOT   18.4ms → 21.1ms   (+14.7%)
  ├ +1.9ms  layers.*.mlp.experts        fused MoE kernel changed (v0.8.2→v0.9.0)
  ├ +0.6ms  layers.*.self_attn          KV cache dtype fp8→bf16, +2× bytes read
  ├ +0.3ms  exposed all_reduce          TP comm no longer overlapped
  └ -0.1ms  everything else             within noise
```

**Config diff panel.** Auto-diff the two `stack.flags` / `env` / `hardware` /
`parallelism` blocks and rank differences by likely blast radius. Half the time
the regression is one flag and the tool should just say so.

---

## 7. Cross-stack comparison ("port items and get ideas")

Same model, same workload, vLLM vs SGLang vs atom. Align the static graphs (they
share the HF architecture), then diff per node.

Output: *"SGLang wins 1.4× on `layers.*.mlp.experts` — its fused MoE kernel does
gate+up in one launch; vLLM issues two. vLLM wins 1.2× on attention decode at
this seqlen via FlashInfer."*

That's a genuinely useful artifact and I don't believe anything produces it
today. It's also the feature most likely to get the project attention from the
vLLM/SGLang communities, which matters more than any marketing you'd do
otherwise.

---

## 8. Phases

### Phase 0a — static extraction spike (from v1, unchanged)
Meta-device instantiation + safetensors name-tree. Gate: Tier A works on ≥7/10
test models. See v1 plan.

### Phase 0b — attribution feasibility spike ← **the real risk, do it first**

Do not build anything else until this resolves. One model (`Qwen3-0.6B`, small
enough to iterate), one GPU, vLLM.

**Claude Code prompt:**

> Stand up vLLM with `VLLM_TORCH_PROFILER_DIR` set, hit `/start_profile`, run a
> fixed workload (batch 8, 512 in / 64 out), `/stop_profile`. Parse the resulting
> chrome trace JSON: build the CPU-op → CUDA-kernel mapping via `correlation_id`,
> and extract module qualnames from the `Module Hierarchy` / stack fields where
> `with_modules` recorded them.
>
> Report: what fraction of total GPU kernel time can be attributed to a named
> `nn.Module`, broken down by prefill vs decode steps. Repeat with
> `--enforce-eager` and compare attribution coverage.
>
> Then write the NVTX shim: monkeypatch `forward` on every module in
> `llm_engine.model_executor.driver_worker.model_runner.model` to push/pop NVTX
> ranges. Profile with `nsys profile --cuda-graph-trace=node -t cuda,nvtx`,
> export to SQLite, and compute the same attribution coverage number.
>
> Deliverable: a table of attribution coverage % for
> {torch-profiler, nvtx} × {cudagraph, eager}.

**Gate:** ≥70% of GPU time attributable to a static node in *at least one*
configuration. Below that, the honest pivot is a trace-diff tool with a coarse
architectural grouping rather than a per-node graph overlay — still viable, but
a different product, and you want to know now.

### Phase 1 — static IR + analytical cost model
v1 phase 1, plus `analytical.py`: FLOPs/bytes/AI per node given a run config, and
a roofline classification against a hardware profile table. Ships as a useful
standalone: "cost this model on this GPU at this batch size" with no traces.

### Phase 2 — trace ingestion + attribution engine
Parsers for chrome-trace (torch profiler), nsys SQLite export, and vLLM/SGLang
Prometheus + OTLP. All three attribution tiers. Gap and comm detection. Emit
measurement IR. Golden tests on committed trace fixtures.

### Phase 3 — overlay UI
Graph from v1 phase 2, plus: heat colouring by any metric (time, % step,
% roofline, achieved BW), a linked timeline strip below the canvas (brush the
timeline → graph filters to that window; click a node → timeline highlights its
kernels), phase toggle, and per-node roofline chart in the info panel.

### Phase 4 — diff engine
Alignment, normalization, significance, waterfall, config diff. Two-run URL
(`/compare?a=<run>&b=<run>`).

### Phase 5 — capture harness + cross-stack
`perfviewer capture --stack vllm --model X --workload Y` that owns the whole
loop: launch, instrument, profile, collect, upload, open. Then the same harness
targeting SGLang and atom, producing directly comparable runs. This is what makes
the tool usable by someone who isn't you.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Attribution fails under CUDA graphs / fused kernels** | Phase 0b gate before anything else. Fallback: coarse grouping, not per-node. |
| Observer effect — eager-mode attribution ≠ graph-mode performance | Explicit attribution-run vs measurement-run split; map by launch-order alignment; never present eager timings as production numbers. |
| Serving stacks move fast; internals shift every release | Depend on public profiler endpoints and `nn.Module` names, not private internals. Pin + test per stack version. Expect this to be ongoing maintenance, not a one-off. |
| Trace file sizes (nsys reps are GB) | Server-side reduction to measurement IR immediately; never ship raw traces to the browser. |
| Multi-GPU trace merging + clock skew | Start single-GPU. Add TP only after phase 4. |
| Users can't run your harness in their env | Trace-upload path must work standalone from day one — Tier 3 attribution exists for exactly this. |

---

## 10. Open design questions

1. **"atom"** — I've assumed a serving/quantization runtime; if it's something
   specific its trace format needs a parser in phase 2 and its instrumentation
   surface may differ from vLLM/SGLang. Pin this down before phase 2.
2. **Deployment access.** If you control the serving deployment, Tier 1 NVTX is
   available and everything gets easier. If the tool must consume traces thrown
   over the wall by someone else, Tier 3 becomes load-bearing and phase 0b should
   spike *that* rather than NVTX.
3. **Single vs multi-GPU.** TP/PP/EP roughly doubles phase 2 and 4 scope (rank
   merging, comm plane, skew). Single-GPU first is the right call unless the
   models you care about can't fit.
