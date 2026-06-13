---
title: >-
  [Paper Note] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference
description: >-
  [ICML 2026][Information Retrieval & RAG][Long Context] This paper unifies optimizations in modern LLM long-context inference—such as sparse attention, RAG…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Long Context"
  - "Sparse Attention"
  - "RAG"
  - "Memory Processing Pipeline"
  - "GPU-FPGA"
date: 2026-05-08
content_hash: 1611c7ad4d92e190
---

# Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference

**Conference**: ICML 2026  
**arXiv**: [2603.29002](https://arxiv.org/abs/2603.29002)  
**Code**: https://github.com/OswaldHe/HeteroLLM (available)  
**Area**: Information Retrieval / LLM Inference Acceleration / Heterogeneous Systems  
**Keywords**: Long Context, Sparse Attention, RAG, Memory Processing Pipeline, GPU-FPGA

## TL;DR
This paper unifies optimizations in modern LLM long-context inference—such as sparse attention, RAG, and compressed context memory—into a four-stage "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" memory processing pipeline. It quantitatively demonstrates that this pipeline accounts for 22%-97% of total latency and that each stage exhibits highly heterogeneous computational characteristics. Based on this, a GPU-FPGA heterogeneous system is proposed: regular/compute-intensive operations remain on the GPU, while sparse/irregular/memory-intensive operations are offloaded to the FPGA. On MI210 + Alveo U55C, up to 2.2× end-to-end speedup and 4.7× energy reduction are achieved.

## Background & Motivation

**Background**: Modern LLMs routinely process 128k-1M tokens (e.g., GPT-OSS-120B requires 69GB GPU memory for a 1M-token KV cache). Numerous long-context optimization schemes have emerged: sparse attention (DeepSeek Attention, SeerAttention-R, LServe), RAG (DRAGIN, FLARE, two-stage RAG), compressed context memory (Titans, HMT, MemAgent), and TTT (LaCT), among others.

**Limitations of Prior Work**: (1) Literature treats these methods as isolated technical studies, lacking a unified abstraction framework, which prevents system-level optimization reuse across methods; (2) The computational overhead of these "memory optimizations" and their contribution to end-to-end latency, as well as bottleneck identification, lack systematic profiling; (3) The memory access patterns, compute density, and data dependencies vary greatly among methods, yet all are typically run on GPUs, resulting in overuse of GPUs for regular dense computation and severe underutilization for sparse/irregular operations.

**Key Challenge**: LLM computation is evolving from "matrix multiplication-centric" to a hybrid of "matrix multiplication + sparse/irregular/memory-intensive operations," but underlying hardware (pure GPU) remains optimized for the former. This mismatch between "algorithmic computational characteristics ↔ hardware architecture" is becoming increasingly pronounced.

**Goal**: (i) Use a unified abstraction to "see through" all long-context optimizations as the same pipeline; (ii) Systematically quantify the latency proportion and compute density heterogeneity of each stage; (iii) Design a GPU-FPGA heterogeneous system to validate the acceleration potential of targeted mapping.

**Key Insight**: Decompose LLM inference into "memory generation $g(\cdot)$ + memory processing $f(\cdot,\cdot)$" (Def 3.1), then focus on $f$; force all $f$ implementations into a four-stage pipeline with fully consistent input/output schemas for each stage, enabling independent optimization and hardware mapping.

**Core Idea**: Replace "independent end-to-end GPU implementations for each optimization" with "four-stage unified pipeline + heterogeneous hardware mapping by stage compute density," compressing "algorithmic diversity" into "four-stage configuration combinations," and leveraging GPU/FPGA for their respective strengths.

## Method

### Overall Architecture
The authors first define the LLM generative model $L(g, f, \{x_i\}_{i<t}, x_t)=y_t$, where $M_{<t}=g(\{x_i\}_{i<t})$ is the generated memory, and $O_{<t}=f(M_{<t}, x_t)$ is the intermediate output from the memory processor. All long-context optimizations are reframed as implementations of $f$. $f$ is uniformly decomposed into four steps: (1) $\text{prep}(M_{<t})=I_{<t}$—compress raw memory into a retrievable index; (2) $\text{comp}(I_{<t}, x_t)=S$—compute relevancy scores between each memory and the current query; (3) $\text{ret}(M_{<t}, S)=M'_{<t}$—select top-$k$ or thresholded memories based on scores; (4) $\text{apply}(M'_{<t}, x_t)=O_{<t}$—inject selected memories into subsequent inference. System-wise, the authors build a heterogeneous platform with AMD MI210 GPU + Alveo U55C FPGA + PCIe, mapping the four stages according to "compute-intensive + regular access → GPU" and "memory-intensive + irregular + data-dependent → FPGA," and design a streaming dataflow FPGA kernel library.

### Key Designs

1. **Unified Four-Stage Memory Processing Pipeline Abstraction**:

    - **Function**: Provides a unified description for diverse algorithms such as sparse attention, RAG, Titans/HMT, MemAgent, and TTT.
    - **Mechanism**: As detailed in Table 1—DeepSeek Attention's prep is linear projection + RoPE, comp is multi-head dot product, ret is top-$k$, apply is fine-grain sparse attention; MemAgent skips comp, prep is model decoding to generate text summaries, apply is model prefill; TTT lacks ret, prep is backpropagation, comp is loss computation. This unification enables kernel reuse (e.g., top-$k$, BM25 shared across methods) and system-level scheduling (bypassing unnecessary stages incurs no overhead).
    - **Design Motivation**: Previous literature lacked this abstraction, requiring hardware acceleration to be rewritten for each new method; the unified pipeline allows the hardware layer to implement "a stage library + cross-stage scheduling" to cover all current and future methods.

2. **Analysis of Computational Heterogeneity (Compute Density + Memory Access Pattern + Data Dependency)**:

    - **Function**: Quantitatively demonstrates that the four stages differ greatly in compute density (FLOPs/byte, determining compute-bound vs memory-bound) and memory access pattern (regular vs irregular), necessitating divide-and-conquer hardware mapping.
    - **Mechanism**: See Table 2—In sparse attention, prep (linear proj, 10-100 FLOPs/byte, regular access) and apply (fine-grain sparse attention, 10-100 FLOPs/byte, regular) are compute-intensive; comp (skinny matrix-vector, 1-10 FLOPs/byte) and ret (top-$k$, ~1 FLOPs/byte, irregular + cross-memory data dependency) are memory-intensive. RAG's comp (BM25) is especially irregular—BM25 queries token frequency in a non-deterministic order; top-$k$ maintains the running maximum with irregular eviction due to data dependency.
    - **Design Motivation**: GPUs excel at regular dense computation but are severely underutilized for irregular + data-dependent memory-intensive operations (peak HBM bandwidth utilization often < 30%); FPGAs, with customizable microarchitecture, naturally offer a 5× SRAM bandwidth advantage for such operations.

3. **GPU-FPGA Heterogeneous System + Streaming Dataflow Kernel**:

    - **Function**: Stage partitioning by compute density, maximizing PCIe communication and computation overlap.
    - **Mechanism**: (a) General Setup (sparse attention + RAG): prep + apply on GPU, fused comp + ret kernel on FPGA, only top-$k$ indices transmitted via PCIe (minimizing communication); (b) Synthesized Memory (MemAgent): prefill-decode disaggregation, decoding (memory-intensive) on FPGA, prefilling (compute-intensive) on GPU; (c) Memory as Context (Titans/HMT): memory embedding fully on FPGA, retrieved memory sent back to GPU with minimal communication as in General Setup. FPGA kernels use a three-level SRAM hierarchy (BRAM 21.8 TB/s + URAM 10.4 TB/s + HBM 460 GB/s), prioritizing token IDs to faster memory, with inner product engines pipelined to top-$k$ retrievers forming a streaming dataflow, avoiding unnecessary off-chip accesses.
    - **Design Motivation**: U55C's on-chip BRAM+URAM 40MB provides about 5× the effective bandwidth of GPU SRAM; streaming dataflow fuses multiple stages into a single kernel, avoiding GPU multi-kernel launch synchronization overhead; PCIe is chosen over tighter interconnects to demonstrate the idea's generalizability on commodity hardware.

### Loss & Training
This is a pure inference systems paper with no training loss. Benchmark evaluation methods include: DeepSeek Attention (vLLM + DeepSeek V3.2 Exp), SeerAttention-R (Qwen3-8B + TileLang), LServe (Llama 3.1-8B, HIPIFY ported to AMD), RAG (Llama-2-7B or Llama 3.1-8B + Wikipedia dump), Titans/HMT (HMT open-source implementation replaced with linear projection), MemAgent (Qwen2.5-7B), LaCT/TTT. FPGA is implemented with Vitis HLS 2024.2 + Vivado 2024.2, using P2P DMA mode.

## Key Experimental Results

### Main Results

End-to-end and memory processing stage acceleration (baseline is single GPU node):

| Method | Memory Processing Speedup | End-to-End Speedup | Energy Savings |
|--------|--------------------------|--------------------|---------------|
| DeepSeek Attention | 1.3-2.2× | 1.1-1.2× | 1.61× geomean |
| SeerAttention-R (Top-$k$) | 1.8-2.2× | up to 1.49× | 1.11× |
| SeerAttention-R (Threshold) | 2.6-4.9× | - | 1.14× |
| LServe | 1.2-5.6× | up to 1.49× | 1.43× |
| DRAGIN (RAG) | 5.2-7.7× | up to 2.2× | 1.10× |
| FLARE (RAG) | 5.1-6.6× | - | 1.14× |
| FS-RAG | 5.1-6.6× | - | 1.21× |
| Two-stage RAG | 1.1-2.1× | 1.47-1.84× | 1.07× |
| MemAgent | - | 1.8× | **4.66×** |
| Titans/HMT (Memory-as-Context) | 3.1-4.0× | 1.3-1.6× | 1.65× |

Latency breakdown profiling (core motivation data): sparse attention accounts for 22%-81% at 1M tokens, RAG for 40%-61% at 20M documents, and MemAgent prep stage can account for **97%** of latency—demonstrating that memory processing is a primary bottleneck, not a minor overhead.

### Ablation Study

| Configuration Dimension | Key Observations |
|------------------------|-----------------|
| Batch size sweep (BS=1 → 32) | Sparse attention speedup increases from 1.07× to 1.32-1.83× (KV cache not shared, batch amplifies sparsity ratio) |
| RAG batch sweep | DRAGIN increases from 1.14× to 1.92× (BM25 not shareable, FPGA advantage amplified) |
| Memory-as-Context batch sweep | Speedup decreases (1.48→1.15×) as linear proj achieves higher GPU utilization at large batch |
| MemAgent BS=8/32 | Speedup drops to 0.49/0.13×, system dynamically falls back to GPU-only |
| FPGA on-chip SRAM hit rate | About 5× effective bandwidth over GPU SRAM |
| PCIe communication overhead | About 3 orders of magnitude lower than end-to-end, nearly negligible |

### Key Findings
- **MemAgent achieves the most dramatic 4.66× energy savings**: Prefill-decode disaggregation lets FPGA handle memory-intensive decode and GPU handle compute-intensive prefill, perfectly matching each hardware's sweet spot.
- **Latency proportion does not monotonically increase with sequence length**: Two-stage RAG's reranker saturates at 500K documents, so latency proportion grows slowly; sparse attention jumps from 1-11% at 4K to 22-81% at 1M tokens, a classic "long-context bottleneck emerges only at scale."
- **MemAgent slows down at BS>4**: Reveals FPGA's fundamental disadvantage in dense decode (weight reuse inferior to GPU); system-level BS-aware dynamic scheduling provides fallback.
- **U55C, despite being an older and half-priced GPU, still achieves significant acceleration**, indicating efficiency gains stem mainly from hardware-feature matching rather than process advantage—upgrading to newer FPGA (Versal V80) holds even greater potential.

## Highlights & Insights
- **Methodological value of the "four-stage pipeline" unified abstraction**: Compresses 8+ long-context optimizations into a single schema; future algorithms only need to specify "which stage uses what computation" to plug into the acceleration framework—textbook system-algorithm co-design paradigm.
- **Compute density heterogeneity framing is highly actionable**: Provides clear decision criteria for "which operation should go on which hardware," moving beyond vague "FPGA is always better/worse than GPU" debates.
- **PCIe communication overhead is 3 orders of magnitude lower**: This counterintuitive but crucial result means most GPU-FPGA collaboration does not require tightly coupled interconnects like NVLink/CXL; PCIe suffices, greatly lowering deployment barriers.
- **BS-aware dynamic scheduling**: MemAgent falls back to GPU-only at large batch sizes, illustrating that hardware acceleration is not all-or-nothing; such fallback mechanisms are vital for production systems.
- **DSA streaming dataflow + three-level SRAM hierarchy**: Prioritizing key vectors by token ID into faster BRAM, spilling to URAM, then HBM, elegantly matches hardware hierarchy to algorithmic access patterns (recent tokens accessed more frequently).

## Limitations & Future Work
- Evaluation is limited to single FPGA + single GPU node; scalability to multi-node setups is unverified. The memory processing pipeline may differ entirely in MoE/tensor parallel scenarios.
- Currently, each new algorithm requires hand-written FPGA kernels (though a kernel library is reusable); the paper acknowledges that "arbitrary new methods may require custom kernels," and design automation is future work.
- TTT/LaCT's compute-intensive nature prevents FPGA benefit, meaning the approach fails for TTT-style methods; applicability boundaries must be clarified.
- Lacks comparison with dedicated LLM ASICs like Groq and SambaNova, so the cost-effectiveness of GPU-FPGA is unclear.
- Experiments use AMD MI210 rather than H100/H200; absolute performance data may have limited industrial relevance. The appendix provides A100 estimates but not real hardware tests.

## Related Work & Insights
- **vs DeepSeek Attention and other single-algorithm works**: This is a meta-level work, providing a unified acceleration framework for all such algorithms, rather than proposing yet another new algorithm.
- **vs Yang 2024a prefill-decode disaggregation**: This paper's MemAgent deployment directly borrows this paradigm, further generalizing it to the memory processing perspective.
- **vs FlashAttention and other GPU kernel optimizations**: FlashAttention optimizes the attention operator itself; this work optimizes the surrounding memory pipeline—orthogonal and composable.
- **vs vLLM / SGLang and other inference frameworks**: vLLM optimizes scheduling, this work optimizes hardware mapping; can be integrated into vLLM as a backend.
- **Insights**: (a) Elevating "memory processing" to a first-class citizen can be extended to multi-modal LLMs (vision token processing also involves prep/comp/ret/apply) and agentic LLMs (tool memory is a homologous pipeline); (b) The compute density-based hardware partitioning approach can guide future ASIC design—dedicated dataflow accelerators for the four memory processing stages.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified abstraction + heterogeneous system mapping is a new framing, though individual techniques (FPGA RAG, sparse attention acceleration) have precedents
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 algorithms × multiple batch configs × end-to-end + memory processing perspectives + energy + PCIe communication overhead, extremely detailed profiling
- Writing Quality: ⭐⭐⭐⭐ Three Claims structure is clear, Table 1/2 make heterogeneity and method classification immediately apparent
- Value: ⭐⭐⭐⭐⭐ Provides the community with a GitHub repo + a conceptual framework, with lasting impact—future long-context LLM acceleration will be inseparable from this abstraction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)

</div>

<!-- RELATED:END -->
