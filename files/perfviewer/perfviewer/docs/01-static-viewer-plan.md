# HF Model Viewer — build plan

Paste a Hugging Face repo id → get an interactive architecture graph.
Reference implementation to match: `hfviewer.com` (Embedl).

This doc is written to be handed to Claude Code phase by phase. Each phase has a
goal, a file layout, the actual prompt to give Claude Code, and an acceptance
gate. Don't start phase N+1 until phase N's gate passes.

---

## 0. The one insight the whole product rests on

You never download weights.

For any HF repo you can get the complete module tree, real tensor shapes, and
parameter counts from a few kilobytes of JSON:

| Artifact | URL | What it gives you |
|---|---|---|
| Model info | `https://huggingface.co/api/models/{repo}` | pipeline_tag, tags, per-dtype param counts, file list |
| Config | `.../resolve/main/config.json` | model_type, architectures[], layer counts, hidden sizes, MoE fields, rope config, quantization_config |
| Weight index | `.../resolve/main/model.safetensors.index.json` | `weight_map`: **every parameter name** → shard. This reconstructs the module tree exactly. |
| Safetensors header | HTTP `Range: bytes=0-7` for header length, then `Range: bytes=8-{8+N}` | tensor names + **shapes** + dtypes for single-shard repos, without pulling the body |
| Remote code | `.../resolve/main/modeling_*.py` | needed for `trust_remote_code` architectures |

And for anything `transformers` supports natively, you can instantiate the real
`nn.Module` tree on the **meta device** — zero bytes allocated, works for a 1T
parameter MoE on a laptop:

```python
from transformers import AutoConfig, AutoModel
from accelerate import init_empty_weights

cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trusted)
with init_empty_weights():
    model = AutoModel.from_config(cfg, trust_remote_code=trusted)
# model.named_modules() is now the ground-truth hierarchy
```

Everything else in this plan is plumbing around that.

---

## 1. Extraction ladder

Implement all four tiers. Each model falls through until one succeeds. Tag every
node with which tier produced it so the UI can honestly say "trace-backed" vs
"config-derived" — hfviewer does exactly this and it's the right call.

**Tier A — traced (best).** Meta-device instantiate, then recover dataflow:
- `torch.fx.symbolic_trace` for well-behaved models
- fallback: `FakeTensorMode` + a `TorchDispatchMode` that records ops while you
  run one forward with fake inputs. Gives op-level edges *and* real intermediate
  shapes, and tolerates control flow that FX chokes on.
- fallback: `torch.export` with dynamic shapes

**Tier B — config template.** A declarative per-`model_type` spec (YAML) that
builds the graph from config fields. Covers models newer than your pinned
transformers, or where remote code fails. ~40 templates covers ~90% of traffic.

**Tier C — name tree (universal floor).** Parse `weight_map` keys:
`model.layers.17.self_attn.q_proj.weight` → split on `.`, integer segments mark
repeat dimensions, leaves carry shapes. Works for literally any safetensors repo
including custom architectures you've never seen. No dataflow edges, but a real
hierarchy with real shapes — still a useful page.

**Tier D — card only.** Metadata + "we couldn't build a graph" state. Never 404.

## 2. Security — do not skip this

`trust_remote_code=True` executes arbitrary Python from a stranger's repo. On a
public site this is the whole attack surface.

- Extraction runs in a **separate container**, never in the web process
- Non-root, read-only rootfs, tmpfs scratch only
- Network **disabled after the fetch step** (fetch artifacts first, then drop net)
- seccomp profile, no `CAP_*`, memory + CPU + 120s wall clock caps
- One container per job, destroyed after
- Consider gVisor/Firecracker once you have real traffic
- Allowlist: run Tier A with `trust_remote_code=False` by default; only enable it
  for repos above a download threshold or manually approved, and cache the result

## 3. Graph IR

One versioned JSON schema between extractor and frontend. Everything downstream
(renderer, SVG cards, OG images, embeds, API) consumes only this.

```jsonc
{
  "schema": 1,
  "source": { "repo": "nvidia/GLM-5.2-NVFP4", "revision": "<sha>", "extracted_at": "..." },
  "model": {
    "model_type": "glm_moe_dsa",
    "task": "text-generation",
    "layers": 78, "hidden_size": 6144,
    "attention": "mla", "ffn": "moe",
    "moe": { "experts": 256, "active": 8, "shared": 1 },
    "vocab_size": 154880, "context_length": 1048576,
    "params": { "total": 0, "by_dtype": {} },
    "quantization": { "method": "nvfp4" }
  },
  "levels": [                      // precomputed granularities, not client-collapsed
    { "id": "L0", "label": "Overview", "nodes": [...], "edges": [...] },
    { "id": "L1", "label": "Blocks",   "nodes": [...], "edges": [...] },
    { "id": "L2", "label": "Modules",  "nodes": [...], "edges": [...] },
    { "id": "L3", "label": "Ops",      "nodes": [...], "edges": [...] }
  ],
  "provenance": { "tier": "A", "traced": true, "notes": [] }
}
```

Node shape:

```jsonc
{
  "id": "model.layers.*.self_attn",
  "label": "MLA self-attention",
  "class": "GlmMoeDsaAttention",     // source-faithful class name
  "kind": "attention",                // taxonomy key → glossary + colour
  "parent": "model.layers.*",
  "repeat": 78,                       // collapsed stack
  "params": 1234567,
  "in_shape": [["B","S",6144]],
  "out_shape": [["B","S",6144]],
  "glossary": "multi-head-latent-attention"
}
```

Edge kinds: `data`, `residual`, `route` (MoE), `aux` (MTP head, side outputs).
Residuals rendered distinctly matter a lot for readability.

**Repeat detection.** Hash each subtree: (class name, ordered child names,
shapes with the layer index normalised out). Identical hashes inside a
`ModuleList` collapse to one representative + count. Handle non-uniform stacks
(Gemma's 5-sliding-then-1-full pattern) by collapsing into *runs*, not one blob —
`5× sliding, 1× full, ×10` is the correct and more interesting rendering.

Precompute levels server-side. A 78-layer MoE at op granularity is hundreds of
thousands of nodes; you cannot collapse that in the browser.

## 4. Stack

- **Extractor**: Python 3.12, FastAPI, torch CPU-only, transformers pinned,
  packaged as its own image. Exposes `POST /extract {repo, revision}` → IR.
- **Web**: Next.js App Router, TypeScript. Route `app/[org]/[model]/page.tsx`
  gives you the URL trick for free (swap `huggingface.co` → your domain).
- **Layout**: `elkjs` (`layered`, `hierarchyHandling: INCLUDE_CHILDREN`) in a web
  worker; cache computed layouts server-side per (repo, level).
- **Render**: React Flow. Keep visible nodes under ~1500 by collapsing groups; go
  canvas/WebGL only if you actually hit the wall.
- **Storage**: Postgres for job/model rows, S3 or R2 for IR + layout blobs, Redis
  for the job queue.
- **Caching**: IR is immutable per (repo, revision sha). Cache forever, key on sha.

You already use React Flow in cvnode — the node/edge component work is largely
portable in both directions.

---

## Phase 0 — spike (target: one weekend)

**Goal:** prove extraction works before writing a single line of UI.

```
spike/
  extract.py          # CLI: python extract.py Qwen/Qwen3-0.6B --tier auto
  tiers/meta.py       # Tier A
  tiers/nametree.py   # Tier C
  fetch.py            # HF API + range-request safetensors header
  out/                # one JSON per model
```

**Claude Code prompt:**

> Build a Python CLI `extract.py` that takes a Hugging Face repo id and emits a
> JSON module tree with no weight download.
>
> Tier A: fetch config.json, use `AutoConfig.from_pretrained` +
> `accelerate.init_empty_weights()` + `AutoModel.from_config` to build the model
> on the meta device. Walk `named_modules()` into a tree with class names and
> per-module parameter counts (from `named_parameters(recurse=False)` shapes —
> meta tensors have shapes, just no storage).
>
> Tier C: fetch `model.safetensors.index.json` (falling back to a range request
> on `model.safetensors` for single-shard repos: first 8 bytes little-endian u64
> = header length, then that many bytes of JSON header). Parse the parameter
> names into the same tree structure, with real shapes.
>
> Emit both to `out/{org}__{model}.json` with a `tier` field. Print a summary
> table. No web server, no UI, no caching.

**Run against these 10, they cover the interesting failure modes:**
`gpt2`, `google-bert/bert-base-uncased`, `google-t5/t5-small`,
`openai/clip-vit-base-patch32`, `google/vit-base-patch16-224`,
`openai/whisper-base`, `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3.2-1B-Instruct`,
a large MoE, and one `trust_remote_code` model.

**Gate:** Tier A succeeds on ≥7/10, Tier C on 10/10, whole run under 60s and
under 2GB RSS. If Tier A fails on more than 3, stop and reconsider — the product
degrades to "pretty config viewer" and you should know that before phase 1.

---

## Phase 1 — extractor service + IR

**Goal:** frozen IR schema, all four tiers, sandboxed service.

```
extractor/
  app/main.py           # FastAPI: POST /extract
  app/tiers/{meta,traced,template,nametree,card}.py
  app/ir/{schema.py,builder.py,collapse.py,levels.py}
  app/templates/*.yaml  # Tier B specs
  app/taxonomy.py       # class name → kind → glossary slug
  tests/golden/         # committed IR snapshots for the 10 spike models
  Dockerfile
```

**Claude Code prompt:**

> Wrap the spike into a FastAPI service. Define the IR as pydantic models in
> `ir/schema.py` matching the schema in the plan doc — version it, and write a
> JSON Schema out to `schema/ir.v1.json`.
>
> Add Tier A-traced: after meta instantiation, attempt `torch.fx.symbolic_trace`;
> on failure, run a forward under `FakeTensorMode` with a `TorchDispatchMode`
> that records (op, inputs, output shape) to build op-level edges. Classify edges
> as data/residual/route — residual edges are adds whose two inputs trace back to
> a common ancestor.
>
> Add `ir/collapse.py`: subtree structural hashing (class name + ordered child
> names + shapes with layer indices normalised), collapse identical siblings in a
> ModuleList into representative + repeat count, and collapse *runs* of a
> repeating pattern rather than requiring uniformity.
>
> Add `ir/levels.py`: derive L0–L3 from the full tree by depth and node kind.
>
> `taxonomy.py`: map source class names to a kind enum (embedding, attention,
> mlp, moe_router, moe_expert, norm, head, residual, ...) with a glossary slug.
> Use pattern matching on class names, not a hardcoded model list.
>
> Golden tests: snapshot IR for the 10 spike models, assert node counts and key
> structural facts (layer count, expert count, attention kind) match config.

**Gate:** all 10 golden tests pass; Docker image runs non-root, read-only rootfs,
network dropped after fetch; a deliberately malicious `modeling_x.py` (try
writing to disk, opening a socket, forking) is contained.

---

## Phase 2 — renderer

**Goal:** IR JSON in, interactive graph out. Still no backend integration.

```
web/
  app/dev/[fixture]/page.tsx     # renders from tests/golden fixtures
  components/graph/{Canvas,Node,Edge,GroupNode,Minimap,InfoPanel}.tsx
  lib/layout/{elk.ts,worker.ts}
  lib/ir.ts                      # generated types from schema/ir.v1.json
```

**Claude Code prompt:**

> Build a React Flow canvas that renders an IR JSON fixture. Layout via elkjs
> `layered` with `hierarchyHandling: INCLUDE_CHILDREN`, run in a web worker,
> top-to-bottom flow.
>
> Custom node types per `kind` with distinct visual treatment. Group nodes show
> `×78` repeat badges and expand/collapse. Residual edges render as curved dashed
> paths behind the main flow; MoE routing edges get a distinct style.
>
> A granularity slider switches between IR levels with animated transitions that
> preserve viewport focus on the node the user was looking at — that transition is
> the single most impressive interaction on hfviewer, budget real time for it.
>
> Clicking a node opens an info panel: source-faithful class name, shapes,
> parameter count, provenance tier, glossary link.
>
> URL state: selected node, level, and viewport in the query string so links are
> shareable.

**Gate:** 78-layer MoE fixture renders in under 2s, pan/zoom holds 60fps, level
switching doesn't lose the user's place.

---

## Phase 3 — the app

**Goal:** public site, cold-start pipeline, SEO.

```
web/
  app/[org]/[model]/page.tsx     # the URL trick
  app/[org]/page.tsx             # publisher index
  app/api/extract/route.ts       # enqueue + poll
  app/api/card.svg/route.ts      # README embed
  app/api/og/route.tsx           # social card (satori)
  app/glossary/[slug]/page.tsx
worker/
  consumer.py                    # pulls jobs, calls extractor, writes IR+layout
```

**Claude Code prompt:**

> Wire the renderer to a real pipeline. `/[org]/[model]`: resolve the repo's
> current revision sha from the HF API, look up cached IR by (repo, sha). Hit →
> render statically. Miss → enqueue an extraction job and stream progress with a
> skeleton graph built from config.json alone (that lands in ~200ms and makes the
> cold path feel instant).
>
> Precompute ELK layouts server-side for cached models and ship them with the IR
> so the client skips layout entirely on warm loads.
>
> ISR with on-demand revalidation. Generate `sitemap.xml` from cached models.
> Per-page metadata built from the IR: title, a description synthesised from the
> architecture facts, OG image.
>
> `/api/card.svg?source={repo}` — a self-contained SVG summary card with no
> external refs, cacheable, safe to embed in a HF README (GitHub/HF markdown
> strips scripts, so it must be pure static SVG).

**Gate:** cold model end-to-end under 30s; warm page TTFB under 300ms; Lighthouse
SEO ≥95; the SVG card actually renders inside a Hugging Face model card.

---

## Phase 4 — distribution

This is where hfviewer's real cleverness is, and it's cheap to copy:

1. **URL substitution** — `huggingface.co/x` → `yourdomain.com/x`. Zero-friction, spreads verbally.
2. **README embed card** — model authors paste your SVG into their model card, which backlinks from every model page. Self-propagating.
3. **Chrome extension** — inject a "view architecture" button on HF model pages.
4. **Family compare pages** — synchronised pan/zoom across related models.
5. **Glossary** — one page per architectural concept, linked from every node. Large surface of long-tail search traffic.

Build #1 and #2 in phase 3 already; they cost almost nothing and they're the loop.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `trust_remote_code` RCE | Phase 1 sandbox gate. Non-negotiable. |
| transformers version drift breaks Tier A on new models | Tier B templates + a "new model type" alert; pin and bump deliberately |
| HF rate limits / bandwidth | Cache by revision sha forever; authenticated token; never fetch weights |
| Layout time on huge graphs | Server-side precompute; cap L3 node count and paginate by block |
| Cost of extracting every model on demand | Only extract on request; keep the config-only skeleton as a permanent cheap fallback |
| Legal/ToS | You're reading public metadata via the public API — fine. Attribute HF, link back, respect robots and rate limits. |

---

## Sequencing note

Phases 0→1 are the entire technical risk. Phases 2→4 are execution you already
know how to do. If phase 0's gate fails, the honest pivot is a config-driven
viewer (Tier B/C only), which is a smaller but still real product — decide that
at the gate, not after building a frontend.
