# Session log

A record of the design session this repo came out of.

**This is a reconstruction, not a transcript.** It's written from the work itself
— what was asked, what was decided, what turned out wrong and how — rather than
copied from the conversation. For the verbatim exchange see
`99-transcript.md`, which has to be exported from the chat client.

---

## How the requirement moved

The project changed shape three times, and each change invalidated part of the
previous plan. Worth recording because the final design only makes sense as the
endpoint of that drift.

**1 — "an app that takes any HF model and makes an interactive viewer, like
hfviewer.com."** Produced `docs/01-static-viewer-plan.md`: extraction ladder,
graph IR, sandboxing, SEO, growth loops. Framed as a public website.

**2 — "attach vLLM/SGLang/atom, attach traces, show regression between two
runs, optimize for a specific stack and port ideas."** This inverted the project.
The architecture graph stopped being the product and became the coordinate
system. Produced `docs/02-perf-tool-plan.md`. The technical risk moved from
"can I extract the graph" (solved) to "can I attribute kernels to modules"
(open).

**3 — "GLM-5.2 on AMD MI355X."** Not NVIDIA. Every profiling assumption changed:
ROCTX instead of NVTX, rocprofv3 instead of nsys, HIP graphs instead of CUDA
graphs, MXFP4 instead of NVFP4. And it made the checkpoint-format mismatch a
day-one finding rather than a footnote.

**4 — "I have weights, vLLM, and the ability to run experiments. atom is an
internal AMD framework."** This resolved the biggest open question: because the
deployment is controllable, ROCTX instrumentation is available and attribution is
an engineering problem rather than a research one — for vLLM and SGLang. For atom
it may not be, since a compiled internal framework may have no Python `forward`
to wrap. Attribution needs spiking per stack, not once.

It also reframed the value: vLLM vs SGLang is a nice-to-have you could squint at
in two Perfetto tabs. vLLM vs atom is something no off-the-shelf tool can do,
because atom is in nobody's parser.

---

## Findings, with the arithmetic

Everything below is derived from published specs and model configs. No
measurements — these are all computable before booking a GPU, which is the point.

### Ridge point

MI355X: 8 TB/s HBM3E, 10.1 PFLOPS MXFP4, 5 PFLOPS MXFP8.

```
ridge(MXFP4) = 10.1e15 / 8e12 = 1262 FLOP/byte
ridge(MXFP8) =  5.0e15 / 8e12 =  625 FLOP/byte
```

Decode arithmetic intensity for a weight-bound linear layer at `bytes_per_param`
b and batch B is `2B/b` — at MXFP4 that's `4B`. Reaching the MXFP8 ridge needs
B ≈ 156; the MXFP4 ridge needs B ≈ 315. No realistic decode batch gets there.

**Consequence:** on this chip, decode is memory-bound by a wide margin. Bytes are
the currency. Any kernel trading arithmetic for memory traffic wins automatically,
and evaluating anything by TFLOPs is meaningless.

### MoE expert streaming

Top-k routing over E experts at batch B touches, in expectation:

```
E_touched = E · (1 − (1 − k/E)^B)
```

For GLM-5.2 (E=256, k=8) at B=32: **~164 distinct experts.** You stream 64% of
all expert weights to use 3% of them.

Arithmetic intensity of the expert layer:

```
AI = 2·(k·p_expert)·B / (E_touched·p_expert·b) = (2B/b)·(k/E_touched)
   = 128 · (8/164) ≈ 6.2 FLOP/byte
```

Against a ridge of 625, that is ~100× memory-bound. Levers in order of effect:
batch size (amortizes the same read over more tokens), expert parallelism (fewer
experts resident per GPU), then kernel efficiency.

Counterintuitive result the simulator surfaces: raising batch 32→128 *increases*
step time and *increases* experts touched (164→252), but throughput rises
substantially, because you're touching nearly all experts either way.

### MLA vs GQA

KV bytes per token per layer:

| Model | Attention | KV/token/layer | KV share of bytes moved |
|---|---|---|---|
| GLM-5.2 | MLA | 576 | ~4% |
| DeepSeek-V3 | MLA | 576 | ~2% |
| Llama-3.3-70B | GQA (64q/8kv) | 2048 | ~37% |

Attention decode arithmetic intensity is ≈ `2·H/H_kv`, so the GQA ratio sets it
directly — around 114 for MLA against about 16 for GQA.

**Consequence:** same tool, opposite advice. On the MoE+MLA models, KV dtype is
nearly irrelevant and expert streaming dominates. On the dense+GQA model, the KV
cache is over a third of all traffic and context length plus KV quantization are
the levers. This is the clearest evidence the tool must be model-agnostic rather
than tuned to one architecture.

### Checkpoint format mismatch

The referenced checkpoint was `nvidia/GLM-5.2-NVFP4`. CDNA 4 has native
MXFP4/MXFP6 (OCP microscaling). NVFP4 and MXFP4 differ in block size and scale
encoding — not interchangeable. The first task is requantization from a
higher-precision checkpoint, not tuning.

Catchable from `config.json` plus a hardware profile, before any GPU time.

### Capacity

288 GB per GPU is enough that a model needing 8 NVIDIA GPUs may fit on 4 here.
Fewer GPUs means less tensor-parallel collective traffic, which on a
memory-bound decode loop can beat any kernel change. That's a serving-topology
decision available on day one from parameter counts alone.

---

## The attribution problem

The one genuinely open risk. vLLM and SGLang don't execute the eager `nn.Module`
graph on the hot path — HIP/CUDA graph replay, fused MoE kernels, FlashInfer or
AITER attention, `torch.compile` regions. Module boundaries are largely erased by
the time you're reading kernel names.

**Tier 1 — ROCTX/NVTX.** Monkeypatch `forward` on every module to push/pop a
range keyed to your static node id. Ship as an importable shim, not a fork.
Profile with `rocprofv3` (or `nsys --cuda-graph-trace=node`; without node-level
graph tracing a whole replay collapses to one opaque event and per-layer
attribution is gone). Requires deployment access.

**Tier 2 — torch profiler.** `with_modules=True, record_shapes=True`, correlation
ids linking CPU op to GPU kernel. Both vLLM and SGLang expose `/start_profile`
and `/stop_profile`; works on ROCm via Kineto. Lowest friction, start here.

**Tier 3 — structural inference.** Autocorrelate the kernel-name and
launch-config sequence to find the dominant period; the period count should equal
`num_hidden_layers`. Align one period against the static graph using shape
signatures. Needs zero cooperation — works on a trace someone emailed you. Likely
the only route for a compiled internal framework.

**Observer effect:** clean attribution may require `--enforce-eager`, which
changes the performance you're measuring. So the design needs an explicit
*attribution run* vs *measurement run* distinction, mapping eager-learned
attribution onto graph-mode kernel sequences by launch-order alignment.

**Gate: ≥70% of GPU time attributable in at least one configuration.** Below
that, the honest product is a trace differ with coarse architectural grouping.
Different tool, and worth knowing before building a frontend.

---

## Design decisions and why

**The run registry is the substance, not the graph.** A run is
`(model + revision) × (stack + flags) × (workload spec) × (hardware + parallelism)
→ measurements`. Heat overlay, roofline, regression, cross-stack comparison are
all queries over that table. Build it first; it delivers value before any graph
exists.

**Workload specs are referenced, never re-specified.** Two runs with different
input/output length distributions or arrival rates are not comparable, and the
numbers look perfectly plausible anyway. Fix the spec as a versioned file; the
tool should refuse to compare runs that don't share one.

**Prefill and decode are never aggregated.** Compute-bound batched GEMM vs
memory-bound KV streaming. A blended per-node number is actively misleading and a
perf engineer will notice in minutes.

**Distributions, not point values.** p50/p95/p99 across steps. A regression claim
built on a single step isn't a claim.

**Always report `unattributed_pct`.** If 40% of GPU time couldn't be pinned, the
user needs to know the picture is partial.

**Gaps are first-class.** Launch overhead, host sync, exposed collectives belong
to no node and are frequently the actual finding. In the prototypes they render
as a hatched block inside the step, not exiled to a side panel.

**Cross-stack comparison granularity is set by the worst-resolving stack.**
Different stacks fuse differently. Compare at layer-component level (attention /
MLP-or-MoE / norm / residual), never op level. The IR should carry a
`comparison_level` and the diff engine should refuse to go below it.

**Adapters are plugins, atom's separate from day one.** Traces from an internal
framework carry internal kernel names. Self-hosted, no public SaaS, clean
boundary so the core plus open-stack adapters could be published later while the
atom adapter stays private. Retrofitting that boundary is painful; declaring it
now is free.

---

## Prototype progression, including the wrong turns

Four versions. Three of them were corrected after the fact, and the corrections
are more instructive than the final state.

**v1 — `v1-overlay-and-diff.jsx`.** Trace overlay, roofline, diff engine with
significance greying, run registry. Real analytical derivation throughout.

*Wrong:* what I called a "graph view" was a vertical list of rows with bars.

**v2 — `v2-table-and-simulator.jsx`.** Added, after auditing the prototype
against the Talaria paper and finding two major gaps: a proper Table View with
sort/filter/search and cross-filtering into the graph, an optimization simulator
(model-wide and targeted), source-location tracking, and an execution timeline.

The simulator is a real analytical model: node bytes recomputed from model
geometry under the chosen config, time = bytes ÷ (8 TB/s × measured efficiency).
Holding *measured* efficiency constant means the prediction inherits reality
instead of assuming a perfect machine — the same trick that got Talaria within
1–3% of hardware.

*Still wrong:* the graph was still a list.

**v3 — `v3-graph-canvas.jsx`.** Replaced it with an actual node-link canvas after
being shown Talaria's Figure 1B. This mattered beyond cosmetics: a transformer
decoder layer isn't linear. MLA runs q and kv paths in parallel before merging;
MoE fans out from the router and sums with the shared expert; two residuals skip
the whole thing. A list hides every one of those, and the parallel branches are
exactly what could overlap while the merge points are where fusion happens or
doesn't.

Added: branch/merge topology, three merge operators, two residual rails, tensor
shapes per node, KV cache as a distinct store, conditional top-8 routing edge in
amber, zoom/pan, minimap, expandable stack ghosts.

**v4 — `v4-multi-model.jsx`.** Added DeepSeek-V3 and Llama-3.3-70B to test
whether the renderer had per-model logic baked in. It didn't.

DeepSeek-V3 breaks the uniform collapse: `first_k_dense_replace: 3` means layers
0–2 are dense MLP and 3–60 are MoE, so the graph renders two group boxes rather
than one. Plus an MTP head branching off the trunk. This is exactly the
run-detection case flagged in the original plan.

Llama-3.3-70B is structurally opposite: no router, no experts, three parallel
projections instead of MLA's two-stage path, and a SwiGLU multiply as a real
merge point.

**The pattern in the mistakes:** each time I built the *summary* of a thing
rather than the thing. A list summarizes a graph; a heat bar summarizes a
topology. For a tool whose entire value is showing structure you'd otherwise hold
in your head, that shortcut isn't available.

---

## Sources consulted

Full notes in `research/prior-art.md`.

- **hfviewer.com** — fetched directly. Config-first extraction, trace-backed vs
  config-derived provenance labelling, article-linked-to-graph format, embed
  card growth loop. No public API; can't ingest runtime data.
- **Model Explorer** (`google-ai-edge/model-explorer`) — README, User Guide,
  Limitations pages fetched. Adapter framework, custom node data with color
  mapping and aggregated stats, edge overlays, split-pane with sync and diff
  highlighting, identical-layer detection, WebGL at tens of thousands of nodes.
  Limits: `torch.export` only for PyTorch, one scalar per op node, structural
  diff only, no preloaded models (HF Space is upload-only; Hub integration open
  as a low-priority issue since 2024).
- **Talaria** (Hohman et al., CHI 2024, arXiv 2404.03085) — full paper read. ACM
  DL blocked bot access; arXiv version used. Formative research, split
  table+graph design, precomputed optimization simulation, source code tracking,
  three-method evaluation over a two-year deployment. Their §8.1 (one model at a
  time) and §8.5 (transformer graphs unintelligible at op scale, automatic
  supernode mining as future work) are this project's starting premises.
- **Interactive explainers** — bbycroft.net/llm, poloclub/transformer-explainer
  (MIT, forkable), Alammar, 3Blue1Brown.
- **MI355X specs** — AMD product page.

---

## Open questions

1. **Does atom emit any module-, layer-, or region-level markers in its traces,
   and can they be added?** A five-minute conversation with its owners that could
   save a month. If no, Tier 3 structural inference is the only route and should
   be spiked first, since it's the most likely to fail and the one you can't work
   around by patching Python.

2. **Single-GPU or multi-GPU first?** TP/PP/EP roughly doubles the ingest and
   diff scope — rank merging, comm plane, clock skew. Single-GPU first unless the
   target models can't fit.

3. **What is this for?** An internal AMD framework points at work rather than the
   independent-income goal that framed the first plan. It changes what phases 4
   and 5 should be, and what the open-core split is protecting.

---

## Immediate next step

Not the graph. Not the frontend.

Pick one model, one workload spec, one GPU count. Capture three runs: default
flags, one flag changed, same thing on SGLang. Write the ingest that reduces each
trace to per-module aggregates keyed by phase. Then write the diff.

If that diff report tells you something you didn't already know from reading the
raw traces, there's a product here. If it doesn't, you've learned that for the
cost of a week.
