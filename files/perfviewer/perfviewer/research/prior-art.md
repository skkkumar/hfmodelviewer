# Prior art

What exists, what it actually does, and where it stops. Assembled from primary
sources — docs, papers, and repos rather than summaries.

---

## hfviewer.com (Embedl AB)

Paste a Hugging Face model URL, get an interactive architecture graph. Built as a
free contribution to the HF community by Embedl, whose actual products are edge
deployment and quantization tooling.

**What it does well**

- Works from config and metadata, never downloads weights — the right approach
  for large models, and the reason it can render a 400B MoE
- Marks graphs as trace-backed where the model could be executed, config-derived
  otherwise. Honest provenance labelling
- Repeated blocks grouped with true repeat counts
- **Article format**: prose linked bidirectionally to graph nodes, split-view.
  Read a section about an architectural decision, jump into that part of the
  graph, come back with context intact. The most interesting thing they built
- Growth loop worth copying: URL substitution (`huggingface.co` → `hfviewer.com`),
  embeddable SVG card for model READMEs, Chrome extension

**Where it stops**

- No public API. Endpoints are `/api/card.svg?source={repo}` and an OG image
  route — embeddable, not programmable
- No way to push runtime data in. Dead end for perf work
- Closed source

---

## Model Explorer (google-ai-edge)

`pip install ai-edge-model-explorer`. ~1.5k stars, actively developed. The
strongest candidate to build on.

**Features**

- Hierarchical nested layers, expand/collapse, flatten-all
- **Identical-layer detection** — selecting a layer auto-highlights structurally
  identical siblings. This is the repeat-collapse problem, already solved
- Artificial layer nodes: auto-partitions large graphs into sections capped at
  1000 children, purple-bordered
- Op nodes degrade to color blocks when zoomed out, keeping structure readable
- Regex search across four match types: label, attributes, inputs, outputs —
  including tensor metadata, so `shape=.*x3` works
- Trace inputs/outputs: highlight all ancestors and descendants, dim the rest
- **Custom node data**: per-node values + threshold or gradient color mapping.
  Side panel gains aggregated stats, child-node Sum% breakdown, sortable
  color-coded tables. Layer nodes show a distribution color bar
- **Edge overlays**: extra edges beyond model structure, JSON-defined with color,
  width, labels
- Node styler: query-based rules (op node AND label regex AND attribute match) →
  styles, exportable, persisted in local storage
- Split-pane with two arbitrary graphs, navigation sync by node id or uploaded
  mapping JSON (1-to-1, 1-to-many, many-to-many)
- Diff highlighting: deleted nodes red-bordered, new nodes green
- Save up to 9 graph states; permalinks; PNG export; processed-graph JSON export
- WebGL rendering, 60fps at tens of thousands of nodes
- Python adapter extension framework; npm package `ai-edge-model-explorer-visualizer`
  for embedding the renderer
- Community adapters: ONNX (justinchuby), VGF and TOSA (arm)

**Where it stops**

- PyTorch only via `torch.export` → ExportedProgram → `.pt2`. Format has no API
  or schema stability; exporting torch version must match local. Fragile at any
  scale, impractical at 400B
- Custom node data is **op nodes only**, one scalar per node. No distributions,
  no prefill/decode dimension, no predicted-vs-achieved pair
- Diff is structural (which nodes exist), not statistical (which deltas clear
  noise)
- No concept of hardware, bytes, or a performance ceiling — it colors numbers you
  hand it
- Windows via WSL only
- MLIR: only tf, tfl, stablehlo dialects
- **No preloaded models.** HF Space supports uploads only; feature request for
  HF Hub integration open since July 2024 at priority 2

**Implication:** write an adapter emitting `graph_builder.GraphNode`s directly
from HF config + safetensors weight map, bypassing torch entirely. That sidesteps
the biggest limitation and would produce something that doesn't currently exist —
a large MoE rendered in Model Explorer.

**Links**

- Repo: https://github.com/google-ai-edge/model-explorer
- Wiki: https://github.com/google-ai-edge/model-explorer/wiki
- HF Space (uploads only): https://huggingface.co/spaces/google/model-explorer
- Blog: https://research.google/blog/model-explorer/

---

## Talaria (Apple, CHI 2024)

*Hohman et al., "Talaria: Interactively Optimizing Machine Learning Models for
Efficient Inference"* — arXiv 2404.03085, DOI 10.1145/3613904.3642628

Closest prior art to the perf half. An internal Apple tool for on-device model
optimization, evaluated over a two-year deployment.

**Problem framing** (from formative research: 12-expert needfinding survey plus a
month of participatory design)

- Practitioners want to know *where* in the compiled graph aggregate cost lives,
  not just the totals. 9 of 12 asked for tools to sort, filter, and locate the
  biggest "offenders"
- Analysis requires large tables and large network diagrams simultaneously, and
  toggling between them is cumbersome but critical
- Governing philosophy is **minimal edit**: accuracy degrades with optimization,
  so optimize only the few operations that matter

**Design**

- Split interface: rich sortable/filterable table + graph of the **compiled**
  hardware graph (explicitly not the model-definition graph), cross-linked
- Color graph nodes by any metric to find bottlenecks geometrically
- **Optimization simulation**: precompute every possible optimization per task at
  compile time, so selecting one updates instantly. Estimates landed within 1–3%
  of hardware benchmarking
- Model-wide vs targeted (per-operation) optimization
- **Source code tracking**: parse the call stack during graph construction, ship
  a JSON mapping hardware tasks → source lines. Turns a finding into an edit
- Complementary views: metric histograms, scatterplot, execution timeline
- Collaboration: named saved analyses, shareable URLs, forking
- Vue + D3 + Monaco, Flask backend, most logic client-side

**Evaluation** — three methods, no controlled study

| | Method | Scale |
|---|---|---|
| E1 | log analysis, 2021→2023 | 800 users, 161 submitters, 3,600+ models |
| E2 | usability survey | 26 respondents rating 20 features |
| E3 | semi-structured interviews | 7 power users, thematic analysis |

**Findings worth carrying over**

- Table-first vs graph-first preference was nearly split (3 vs 4), but nearly all
  used both together; cross-view selection was one participant's favourite
  feature of the entire system
- Unanticipated use: verifying architecture changes compiled as expected — a
  "quick check", not bottleneck hunting
- Surprising bottlenecks were routine ("happens all the time"), e.g. redundant
  dtype conversions between operation inputs and outputs
- **Collaboration and source-code mapping were the least used features** — both
  of which sound essential when planning. Only 1 of 26 called them not useful;
  the rest said not applicable
- Applying optimizations to code is only one of several next iterations; others
  are trying a different architecture or updating the compiler

**Stated limitations, both directly relevant**

- One model at a time. A Diff View was prototyped after the study concluded —
  four panes, new operations green, removed red. Structural only
- At tens of thousands of operations the graph stayed usable but stopped being
  intuitive. **They name transformers explicitly**, propose representing
  thousands of ops as a handful of sequential modules, and list automatic
  supernode construction by mining repeated operation patterns as future work

That last paragraph is this project's starting premise, published as their open
problem.

---

## Interactive explainers

Reference points for the "understand a section of the model" half.

- **bbycroft.net/llm** — 3D walkthrough of a GPT-style transformer. TypeScript,
  renderer handles arbitrary network sizes, animated dataflow through attention
  and feedforward with walkthrough text. Best-in-class for stepping through
  model sections
- **poloclub/transformer-explainer** — MIT, runs live GPT-2 in-browser, Sankey
  design, smooth transitions between abstraction levels. 125k users. Forkable
- Jay Alammar's Illustrated Transformer; 3Blue1Brown; Distill.pub for format
- Anthropic's Transformer Circuits for depth

**The differentiated version:** generate the article per-model with real numbers
substituted. Not "MoE routes tokens to experts" but "this model routes to 8 of
256, so at batch 1 you load 256 experts' weights to use 8 — here is the measured
bandwidth waste on your trace." Static explainers can't do that.

---

## Profiling stack (AMD / ROCm)

Because the target is MI355X, not NVIDIA:

| NVIDIA | ROCm equivalent |
|---|---|
| NVTX | ROCTX (roctracer) |
| nsys | rocprofv3 / rocprof-sys |
| ncu | rocprof-compute (roofline built in) |
| CUDA graphs | HIP graphs (same attribution collapse) |

PyTorch profiler works on ROCm via Kineto, so vLLM's `/start_profile` and
`VLLM_TORCH_PROFILER_DIR` produce usable chrome traces on AMD. Kernel libraries
in play: hipBLASLt, Composable Kernel, Triton, AITER.

---

## Hardware reference — MI355X (CDNA 4)

| | |
|---|---|
| HBM3E | 288 GB |
| Peak bandwidth | 8 TB/s |
| MXFP4 / MXFP6 matrix | 10.1 PFLOPS |
| MXFP8 matrix | 5 PFLOPS |
| Compute units | 256 |
| Last level cache | 256 MB |
| Infinity Fabric peak link | 153 GB/s |
| TBP | 1400 W, liquid cooled |

Derived ridge points: **1262 FLOP/byte** at MXFP4, **625** at MXFP8.

Source: AMD product page for Instinct MI355X.
