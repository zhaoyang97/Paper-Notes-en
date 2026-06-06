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
content_hash: 58e9b3b87daac526
---

# Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference

**Conference**: ICML 2026  
**arXiv**: [2603.29002](https://arxiv.org/abs/2603.29002)  
**Code**: https://github.com/OswaldHe/HeteroLLM (Available)  
**Area**: Information Retrieval / LLM Inference Acceleration / Heterogeneous Systems  
**Keywords**: Long Context, Sparse Attention, RAG, Memory Processing Pipeline, GPU-FPGA

## TL;DR
This paper unifies optimizations in modern LLM long-context inference—such as sparse attention, RAG, and compressed context memory—into a four-stage "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" memory processing pipeline. It quantitatively demonstrates that this pipeline accounts for 22%-97% of the total latency and that each stage possesses highly heterogeneous computational characteristics. Based on these findings, it proposes a GPU-FPGA heterogeneous system: regular/compute-intensive operations remain on the GPU, while sparse/irregular/memory-intensive operations are offloaded to the FPGA. This achieves up to 2.2× end-to-end speedup and a 4.7× reduction in energy consumption on an MI210 + Alveo U55C platform.

## Background & Motivation

**Background**: Modern LLMs commonly process 128k-1M tokens (storing the KV cache for 1M tokens in GPT-OSS-120B requires 69GB of GPU memory). Various long-context optimization schemes have emerged: sparse attention (DeepSeek Attention, SeerAttention-R, LServe), RAG (DRAGIN, FLARE, two-stage RAG), compressed context memory (Titans, HMT, MemAgent), and TTT (LaCT).

**Limitations of Prior Work**: (1) Literature treats these methods as isolated techniques, lacking a unified abstract framework, which prevents system optimizations from being reused across methods. (2) There is no systematic profiling of the actual computational overhead these "memory optimizations" contribute to end-to-end latency or where the bottlenecks lie. (3) The memory access patterns, compute density, and data dependencies of various methods differ significantly, yet they are all executed on GPUs by default. Consequently, GPUs are overused for regular dense computation but severely underutilized for sparse/irregular operations.

**Key Challenge**: LLM computation is evolving from "matrix multiplication-centric" to a "hybrid of matrix multiplication and sparse/irregular/memory-intensive operations," yet the underlying hardware (pure GPU) remains optimized for the former. This mismatch between "algorithmic computational characteristics ↔ hardware architecture" is becoming increasingly severe.

**Goal**: (i) Use a unified abstraction to frame all long-context optimizations as the same pipeline; (ii) systematically quantify the latency contribution and computational heterogeneity of each stage; (iii) design a GPU-FPGA heterogeneous system to verify the potential of "matching the medicine to the disease."

**Key Insight**: LLM inference is decomposed into two parts: "Memory Generation $g(\cdot)$ + Memory Processing $f(\cdot,\cdot)$" (Def 3.1), focusing specifically on $f$. By forcing all $f$ into a four-stage pipeline where input/output schemas are identical across stages, each can be independently optimized and mapped to hardware.

**Core Idea**: Replace "independent end-to-end GPU implementations for each optimization" with a "four-stage unified pipeline + heterogeneous hardware mapping according to stage compute density." This compresses "algorithmic diversity" into "four-stage configuration combinations," allowing GPUs and FPGAs to handle what they each do best.

## Method

### Overall Architecture
The authors first define the LLM generative model as $L(g, f, \{x_i\}_{i<t}, x_t)=y_t$, where $M_{<t}=g(\{x_i\}_{i<t})$ is the generated memory and $O_{<t}=f(M_{<t}, x_t)$ is the intermediate output from the memory processor. All long-context optimizations are reframed as implementations of $f$. $f$ is unified into four steps: (1) $\text{prep}(M_{<t})=I_{<t}$—compressing raw memory into a searchable index; (2) $\text{comp}(I_{<t}, x_t)=S$—calculing relevancy scores between memory and the current query; (3) $\text{ret}(M_{<t}, S)=M'_{<t}$—selecting top-$k$ or applying a threshold; (4) $\text{apply}(M'_{<t}, x_t)=O_{<t}$—injecting selected memory into subsequent inference. On the system level, an AMD MI210 GPU + Alveo U55C FPGA + PCIe heterogeneous platform is built. Stages are mapped according to the principle: "compute-intensive + regular access → GPU" and "memory-intensive + irregular + data-dependent → FPGA," supported by a library of streaming dataflow FPGA kernels.

### Key Designs

1.  **Unified Abstraction of the Four-Stage Memory Processing Pipeline**:
    - **Function**: Unifies diverse algorithms like sparse attention, RAG, Titans/HMT, MemAgent, and TTT.
    - **Mechanism**: Based on the per-method decomposition in Table 1—DeepSeek Attention's `prep` involves linear projection + RoPE, `comp` is multi-head dot product, `ret` is top-$k$, and `apply` is fine-grained sparse attention; MemAgent skips `comp`, `prep` is model decoding for summaries, and `apply` is model prefill; TTT lacks `ret`, its `prep` is backpropagation, and `comp` is loss calculation. This unification enables kernel reuse (e.g., top-$k$ and BM25 shared across methods) and system scheduling (bypassing unnecessary steps without overhead).
    - **Design Motivation**: Previous work lacked this abstraction, requiring new hardware acceleration for each method; the unified pipeline allows the hardware layer to implement only a "stage library + cross-stage scheduler" to cover all current and future methods.

2.  **Computational Heterogeneity Analysis (Compute Density + Access Patterns + Data Dependency)**:
    - **Function**: Quantitatively proves that the four stages vary greatly in compute density (FLOPs/byte, determining compute-bound vs. memory-bound) and access patterns (regular vs. irregular), necessitating a divide-and-conquer hardware mapping.
    - **Mechanism**: See Table 2—in sparse attention, `prep` (linear projection, 10-100 FLOPs/byte, regular access) and `apply` (fine-grained sparse attention, 10-100 FLOPs/byte, regular) are compute-intensive. `comp` (skinny matrix-vector, 1-10 FLOPs/byte) and `ret` (top-$k$, ~1 FLOPs/byte, irregular + data dependency across memory) are memory-intensive. RAG's `comp` (BM25) is particularly irregular—BM25 queries token frequency in non-deterministic orders; top-$k$ involves maintaining running maximums with irregular evictions.
    - **Design Motivation**: While GPUs outperform FPGAs in regular dense computation, they are severely underutilized in irregular + data-dependent memory-intensive operations (peak HBM bandwidth utilization is often < 30%). FPGAs, with customizable microarchitectures, offer a natural 5× SRAM bandwidth advantage for such operations.

3.  **GPU-FPGA Heterogeneous System + Streaming Dataflow Kernels**:
    - **Function**: Partitions stages by compute density to maximize the overlap of PCIe communication and computation.
    - **Mechanism**: (a) General Setup (Sparse Attention + RAG): `prep` + `apply` on GPU, fused `comp` + `ret` kernel on FPGA, with only top-$k$ indices transmitted via PCIe (minimizing communication); (b) Synthesized Memory (MemAgent): prefill-decode disaggregation, with decoding (memory-intensive) on FPGA and prefilling (compute-intensive) on GPU; (c) Memory as Context (Titans/HMT): memory embeddings stored entirely on FPGA, retrieved memory returned to GPU with low communication volume. The FPGA kernel utilizes a three-level SRAM hierarchy (BRAM 21.8 TB/s + URAM 10.4 TB/s + HBM 460 GB/s) to prioritize faster memory by token ID. A dot-product engine feeds into a top-$k$ retriever to form a streaming dataflow, avoiding off-chip access.
    - **Design Motivation**: The U55C's 40MB on-chip BRAM+URAM provides ~5× the effective bandwidth of GPU SRAM. Streaming dataflow fuses stages into a single kernel, avoiding GPU multi-kernel synchronization overhead. PCIe was chosen over tighter interconnects to prove the generalizability of the idea using commodity hardware.

### Loss & Training
This work focuses purely on the inference system and does not involve training loss. Baseline evaluation methods include: DeepSeek Attention (vLLM + DeepSeek V3.2 Exp), SeerAttention-R (Qwen3-8B + TileLang), LServe (Llama 3.1-8B, HIPIFY ported to AMD), RAG (Llama-2-7B or Llama 3.1-8B + Wikipedia dump), Titans/HMT (open-source HMT with linear projection replacement), MemAgent (Qwen2.5-7B), and LaCT/TTT. FPGA implementation uses Vitis HLS 2024.2 + Vivado 2024.2 with P2P DMA mode.

## Key Experimental Results

### Main Results

End-to-end + memory processing stage acceleration (baseline is equivalent GPU-only setup):

| Method | Memory Processing Gain | End-to-end Gain | Energy Saving |
|------|--------------|------------|----------|
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

Latency contribution profiling: Sparse attention at 1M tokens accounts for 22%-81% of latency; RAG at 20M documents accounts for 40%-61%; MemAgent's `prep` stage alone can account for **97%** of single-step latency—proving memory processing is the primary bottleneck, not a marginal overhead.

### Ablation Study

| Configuration Dimension | Key Findings |
|----------|---------|
| Batch size scaling (BS=1 → 32) | Sparse attention gain increases from 1.07× to 1.32-1.83× (KV cache non-shared, BS scales sparse ratio). |
| RAG batch scaling | DRAGIN gain increases from 1.14× to 1.92× (BM25 non-sharable, magnifying FPGA advantage). |
| Memory-as-Context BS scaling | Gain decreases (1.48→1.15×) as GPU utilization for linear projection improves at larger BS. |
| MemAgent BS=8/32 | Gain drops to 0.49/0.13×, prompting fallback to GPU-only mode. |
| FPGA on-chip SRAM hit rate | Provides ~5× higher effective bandwidth than GPU SRAM. |
| PCIe communication overhead | Approximately 3 orders of magnitude lower than end-to-end latency, making it negligible. |

### Key Findings
- **MemAgent achieves a staggering 4.66× energy saving**: Prefill-decode disaggregation places memory-intensive decoding on the FPGA and compute-intensive prefilling on the GPU, hitting the "sweet spot" of both hardware architectures.
- **Latency contribution is not monotonic with sequence length**: For two-stage RAG at 500K documents, the reranker saturates, slowing latency growth; sparse attention jumps from 1-11% at 4K to 22-81% at 1M, representing a bottleneck that only emerges in long contexts.
- **MemAgent slows down when BS > 4**: This reveals the FPGA's inherent weakness in dense decoding (weight reuse is inferior to GPU), addressed by BS-aware dynamic scheduling.
- **U55C remains significantly faster despite being older and half the price of the GPU**: This indicates that efficiency gains stem from hardware characteristic matching rather than process technology advantages—newer FPGAs (e.g., Versal V80) hold even greater potential.

## Highlights & Insights
- **Methodological value of the "Four-Stage Pipeline"**: Compressing 8+ algorithms into a single schema allows future algorithms to be integrated into the acceleration framework simply by defining their stage computations—a textbook case of system-algorithm co-design.
- **Heterogeneity framing is highly actionable**: It provides clear criteria for deciding which operations belong on which hardware, moving beyond vague discussions of whether FPGAs are "generally" better or worse than GPUs.
- **PCIe communication is 3 orders of magnitude lower than latency**: This counter-intuitive data suggests that most GPU-FPGA collaborations do not require expensive NVLink/CXL interconnects; PCIe is sufficient, lowering deployment barriers.
- **BS-aware dynamic scheduling**: The fallback mechanism for MemAgent at large batch sizes shows that hardware acceleration is not "all-or-nothing," a critical feature for production systems.
- **DSA streaming dataflow + three-level SRAM**: Mapping the hardware hierarchy to the algorithm's access pattern (prioritizing recent tokens in faster BRAM) is an elegant architectural choice.

## Limitations & Future Work
- Evaluation is limited to single FPGA + single GPU nodes; scalability in MoE or tensor-parallel scenarios remains unverified.
- Currently, new FPGA kernels must be handwritten for new algorithms (though a library exists); the paper acknowledges that design automation is required for future work.
- TTT/LaCT are compute-dominated and do not benefit from FPGAs, indicating a clear boundary for the proposed solution's applicability.
- Comparison with specialized LLM ASICs (e.g., Groq, SambaNova) is missing, leaving the relative cost-performance of GPU-FPGA unclear.
- The use of AMD MI210 instead of H100/H200 may limit the reference value for industrial deployment, despite A100 estimations in the appendix.

## Related Work & Insights
- **vs. DeepSeek Attention and single-algorithm works**: This is a meta-level work providing a framework for all such algorithms rather than proposing a new one.
- **vs. Yang 2024a (prefill-decode disaggregation)**: MemAgent deployment adopts this paradigm and extends it to the memory processing perspective.
- **vs. FlashAttention and GPU kernel optimizations**: FlashAttention optimizes the attention operator itself; this work optimizes the surrounding memory pipeline. The two are orthogonal and complementary.
- **vs. vLLM / SGLang**: vLLM optimizes scheduling; this work optimizes hardware mapping. It can be integrated into vLLM as a backend.
- **Insights**: (a) The "memory processing" perspective can be extended to multi-modal LLMs (vision token processing) and agentic LLMs (tool memory); (b) partitioning by compute density can guide future ASIC designs for dedicated memory processing dataflow accelerators.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified abstraction + heterogeneous mapping is a fresh framing, though single-point technologies (FPGA RAG/Sparse Attention) have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 algorithms × multiple batch configs × dual perspective (end-to-end + memory processing) + energy + PCIe profiling.
- Writing Quality: ⭐⭐⭐⭐ Clear "Three Claims" structure; Tables 1/2 provide an immediate understanding of heterogeneity and categorization.
- Value: ⭐⭐⭐⭐⭐ Providing both a GitHub repository and a conceptual framework ensures lasting impact—long-context LLM acceleration cannot ignore this abstraction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[ICML 2026\] Vector Linking based on Cross-Model Local Isometry Consistency](vector_linking_via_cross-model_local_isometric_consistency.md)

</div>

<!-- RELATED:END -->
