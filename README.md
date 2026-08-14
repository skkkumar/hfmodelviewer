help in understanding. given a model like glm5.2 i want to visually see in a tree or graph form something like https://hfviewer.com/zai-org/GLM-5.2-FP8 where I should be able to see the whole model flow. but i also dont want to miss any information out of the traces like the functions that are executed , it will also include vllm or sglang framework and also i will want to see the different function calls, module calls, till the kernel calls. i want to know the exact kernel that is called and also want to know what are the alternative options that are available. this is to be used by a performance engineer who wants to also later understand different heatmaps on the graph or tree like which had low latency, which had memory bound and which had compute bound. i want to see them visually in the model view rather than in the traces in the perfetto. 
btw hfviewer is not very detailed it is only highlevel outlined.  but i would want the code level

HF Viewer is a **config outline**. It names blocks (MLA, MoE, RMSNorm) from `config.json` and modeling code. It never runs the model, never walks vLLM/SGLang, and never names a kernel. The view you want is a **code-level execution graph**: the same GLM-5.2 layer as the functions that ran, the backend that dispatched them, the kernel that launched, the other kernels that could have, and latency / memory-vs-compute painted on those nodes — not on a Perfetto timeline.

I put that view in [GLM-5.2 code-level graph](C:\Users\devan\.cursor\projects\c-Users-devan-Desktop\canvases\glm52-code-level-graph.canvas.tsx). A canvas is a live panel you can open beside the chat; use the depth control to go from hfviewer-like architecture down to CUDA/Triton launches, then click a node.

**What “code-level” means here**

| Zoom | Example | In HF Viewer? |
|---|---|---|
| Architecture | embed → decoder×78 → MTP → lm_head | Yes |
| Modules | `input_layernorm`, `self_attn`, `mlp` | Named blocks only |
| Functions | `q_a_proj`, `kv_a_proj_with_mqa`, `DSABackend.forward`, `fused_topk` | No |
| Kernels | `tilelang_dsa_indexer_topk`, `flashinfer::mla_sparse_decode`, DeepGEMM grouped w1/w2 | No |
| Counters | FLOP/byte, HBM%, SM% → memory vs compute | No |
| Alternatives | FLASHINFER_MLA_SPARSE vs tilelang vs TRITON_MLA vs AITER | No — registries, not traces |

A typical SGLang decode path for one layer is not “an attention block.” It is:

`Scheduler.run_batch` → `ModelRunner.forward_decode` → `GlmMoeDsaDecoderLayer.forward` → `GlmMoeDsaAttention.forward` → `DSABackend.forward` → `tilelang_dsa_indexer_topk` + `sparse_mla_decode` → then the MoE path `gate` → `moe_align_block_size` → grouped GEMM w1/w2 for 8 of 256 experts.

HF Viewer stops at the second name in that list.

**Why Perfetto is the wrong home screen**

Perfetto is a timeline of events. You want a **graph of ops**. They only line up if every kernel carries a stable `op_id` (`layers.17.self_attn.sparse_mla`) that also exists on the module tree. nsys gives duration and kernel names; Nsight Compute gives the heatmap (arithmetic intensity, HBM%, SM%). Alternative backends never appear in a trace — they come from vLLM’s AttentionBackend / Fp8MoeBackend registries and SGLang flags like `--dsa-decode-backend tilelang`.

Fusions make a naive stack dump lie: `residual + x` and `RMSNorm` often become one `fused_add_rmsnorm`; `silu_and_mul` often dies as a GEMM epilogue; CUDA graphs collapse hundreds of launches into `cudaGraphLaunch`. The code-level view has to keep the Python functions **and** show they fused.

There is no website that already does this for GLM-5.2. The practical path is: serve with SGLang/vLLM → NVTX/`record_function` around every `forward` → nsys + ncu → dump engine registries → join on `op_id` → color the module tree. The canvas is that target UI, with walkthrough timings so you can see the shape before you wire a live profile.
