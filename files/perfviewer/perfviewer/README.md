# perfviewer

Design work and prototypes for a performance-engineering tool for LLM serving.

**The idea in one line:** a model's architecture graph is a stable coordinate
system; runtime traces get pinned to it; an analytical model says what each node
*should* cost; the product is the gap between the two.

Target case driving the design: optimizing GLM-5.2 on AMD MI355X under vLLM,
compared against SGLang and other serving stacks.

---

## Why

Three things exist today and none of them join up:

| Tool | Has | Missing |
|---|---|---|
| Netron, hfviewer | architecture graph | any notion of runtime |
| Perfetto, HTA, rocprof | kernel timeline | any notion of model semantics |
| Nsight, rocprof-compute | hardware counters | model semantics, cross-run diffing |

Nothing joins *architecture* × *serving-stack runtime* × *cross-run regression*.
And nothing at all does analytical-vs-measured, which is where the actual
insight lives.

## The four data planes

| Plane | Source | Cost |
|---|---|---|
| **Static** | HF config + safetensors index + meta-device module tree | free, no weights downloaded |
| **Analytical** | derived from static geometry + a run config | free, no GPU |
| **Measured** | torch profiler / rocprofv3 / nsys from vLLM, SGLang, … | one profiling run |
| **Config** | serving flags, env, hardware, parallelism | free |

Planes 1 and 2 alone answer "what will this model cost on this GPU at this
batch size", which is already worth shipping.

---

## Repo layout

```
docs/
  00-session-log.md           how the requirement moved, findings with the math,
                              design decisions, the wrong turns, open questions
  01-static-viewer-plan.md    extraction ladder, IR schema, sandboxing, phases
  02-perf-tool-plan.md        attribution tiers, roofline, regression diffing
  99-transcript.md            placeholder for the raw conversation
prototypes/
  v1-overlay-and-diff.jsx     trace overlay, roofline, diff engine, run registry
  v2-table-and-simulator.jsx  + sortable table, optimization simulator, timeline
  v3-graph-canvas.jsx         + real node-link canvas: topology, zoom, minimap
  v4-multi-model.jsx          + GLM-5.2 / DeepSeek-V3 / Llama-3.3-70B
research/
  prior-art.md                what exists, what it does, where it stops
```

The prototypes are React single-file components with synthetic measurements over
**real geometry** — bytes and FLOPs computed from each model's actual config
against MI355X's actual spec sheet. Every displayed metric is derived; nothing is
a hardcoded label.

Read them in order. Each one fixes something the previous got wrong, and the
progression is the useful part:

- **v1** built the graph as a list of bars. Wrong: a list can't show topology.
- **v2** added the table and simulator missing from v1.
- **v3** replaced the list with an actual canvas — branch/merge, residual rails,
  tensor shapes, zoom/pan/minimap.
- **v4** proved the renderer has no per-model logic by adding two more models
  with genuinely different structure.

## Running a prototype

Single-file React components, no build config. Drop one into any React app with
Tailwind-free inline styles (all styling is inline or in a `<style>` tag):

```bash
npm create vite@latest perfviewer-ui -- --template react
cd perfviewer-ui && npm install
cp ../prototypes/v4-multi-model.jsx src/App.jsx
npm run dev
```

---

## Key findings baked into the design

**Decode is memory-bound and it isn't close.** MI355X: 8 TB/s HBM, 10.1 PFLOPS
MXFP4 → ridge point ~1260 FLOP/byte (625 at FP8). Decode arithmetic intensity is
roughly 2× batch size. Reaching the ridge needs a batch in the hundreds. Bytes
are the currency; FLOPs are free.

**MoE decode streams weights it doesn't use.** Top-8 of 256 experts at batch 32
touches ~164 distinct experts — 64% of expert weights read to use 3% of them.
Arithmetic intensity lands around 6, roughly 100× below the ridge. The levers, in
order: batch size, expert parallelism, then kernel.

**MLA vs GQA is a bandwidth story.** MLA keeps 576 values per token per layer;
GQA (Llama-3.3-70B) keeps 2048. Result: KV cache is ~2% of bytes moved on
DeepSeek-V3 and ~37% on Llama-3.3-70B. Same tool, opposite advice — cache dtype
barely matters on one and dominates the other.

**Checkpoint format is a day-one finding.** An NVFP4 checkpoint is not usable on
CDNA 4's native MXFP4 path — different block size and scale encoding. The tool
catches this from `config.json` + a hardware profile before any GPU is booked.

**Prefill and decode must never be aggregated.** Compute-bound batched GEMM vs
memory-bound KV streaming. A blended per-node number is actively misleading.

---

## The open risk

Kernel→module attribution. vLLM and SGLang don't run the eager `nn.Module` graph
on the hot path — CUDA/HIP graph replay, fused MoE, FlashInfer/AITER attention.
Module boundaries are largely erased by the time you're reading kernel names.

Three tiers, and this needs spiking **per stack** before anything else is built:

1. **ROCTX/NVTX instrumentation** — wrap module `forward`s. Best, needs
   deployment access.
2. **torch profiler module hierarchy** — `with_modules=True`, correlation ids.
   Lowest friction; vLLM and SGLang both expose `/start_profile`.
3. **Structural inference** — period-detect the repeating decoder signature in an
   opaque kernel timeline. Works on a trace someone emailed you.

**Gate: ≥70% of GPU time attributable to a static node in at least one
configuration.** Below that, the honest product is a trace differ with coarse
architectural grouping — a different tool, and better to know before building a
frontend.

## Sequencing

The registry comes first, not the graph. A run is
`(model + revision) × (stack + flags) × (workload spec) × (hardware + parallelism) → measurements`.
Every feature — heat overlay, roofline, regression, cross-stack — is a query over
that table. Fix the workload spec as a versioned file and make every run
reference it; two runs that don't share one aren't comparable and the tool should
refuse rather than produce a plausible wrong number.

## Status

Design and prototypes only. No extraction, no trace ingestion, no attribution
engine yet. The prototypes exist to pin down what the tool should do before
committing to how.

## License

MIT
