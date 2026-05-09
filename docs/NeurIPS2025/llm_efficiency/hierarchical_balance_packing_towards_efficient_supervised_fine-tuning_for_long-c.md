---
title: >-
  [Paper Note] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM
description: >-
  [NeurIPS 2025][LLM Efficiency][Data packing] This paper proposes Hierarchical Balance Packing (HBP), which addresses attention computation imbalance and communication waste in mixed long/short-context SFT through multi-level packing groups, balanced batching, adaptive sequence parallelism, and stable loss normalization. HBP achieves a 2.4× training speedup on DeepSeek-V2 (236B) without performance degradation.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Data packing
  - long-context fine-tuning
  - sequence parallelism
  - attention balancing
  - curriculum learning
date: 2026-05-08
content_hash: b9a0fd58678eec92
---

# Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM

**Conference**: NeurIPS 2025
**arXiv**: [2503.07680](https://arxiv.org/abs/2503.07680)
**Code**: [https://github.com/ModelTC/HBP](https://github.com/ModelTC/HBP)
**Area**: LLM Efficiency
**Keywords**: Data packing, long-context fine-tuning, sequence parallelism, attention balancing, curriculum learning

## TL;DR
This paper proposes Hierarchical Balance Packing (HBP), which addresses attention computation imbalance and communication waste in mixed long/short-context SFT through multi-level packing groups, balanced batching, adaptive sequence parallelism, and stable loss normalization. HBP achieves a 2.4× training speedup on DeepSeek-V2 (236B) without performance degradation.

## Background & Motivation
**Background**: Long-context LLMs require joint training on both long-context (e.g., 128K) and short-context data during SFT to preserve general capabilities. The dominant approach is data packing, which concatenates variable-length samples into fixed-length mini-batches.

**Limitations of Prior Work**: Naïve packing suffers from three problems: (a) mixing long and short samples causes severe attention computation imbalance (ABR as high as 0.506), leading to significant GPU idle time; (b) short sequences are unnecessarily subjected to sequence parallel (SP) communication, wasting bandwidth (CR = 1.0); (c) uneven compute allocation across DP groups causes synchronization stalls.

**Key Challenge**: A single packing length cannot simultaneously optimize training efficiency for both short sequences (no SP needed, low attention complexity) and long sequences (SP required, high attention complexity), as the two regimes demand fundamentally different training strategies (SP degree, gradient checkpointing configuration).

**Goal**: (a) How to automatically determine the optimal multi-level packing groups? (b) How to assign samples to groups while balancing attention computation? (c) How to design a dynamic training pipeline that accommodates multi-level inputs?

**Key Insight**: The authors introduce two new metrics—ABR (Attention Balance Ratio) and CR (Communication Ratio)—to quantify the inefficiency of naïve packing. Profiling reveals that optimal SP/GC strategies differ substantially across sequence lengths (e.g., optimal SP=8 for 32K vs. 128K, but with different GC layer counts), motivating a hierarchical treatment.

**Core Idea**: Replace single-level packing with multi-level hierarchical packing, where each level is assigned its optimal SP/GC configuration, and long and short data are physically isolated to eliminate communication waste and attention imbalance.

## Method

### Overall Architecture
HBP consists of three stages: (1) **Hierarchical Group Auto-Selection**: profiling-based identification of the optimal packing length set and corresponding SP/GC configurations; (2) **Balance Packing**: data assignment to groups via greedy filling and attention-complexity-aware sorting to construct balanced batches; (3) **Dynamic Training Pipeline**: adaptive SP switching, curriculum learning, and stable loss normalization.

### Key Designs

1. **Hierarchical Group Auto-Selection (Algorithm 1)**:

    - Function: Automatically determines the optimal packing length set $L_p = \{l_1, l_{best}, l_2, l_{max}\}$.
    - Mechanism: Stage 1 searches over candidate lengths (e.g., 8K/16K/32K/64K/128K) to find the optimal SP degree and GC configuration for each, selecting the $l_{best}$ with the shortest iteration time. Stage 2 further reduces communication overhead by computing $l_1 = \lfloor l_{best}/sp \rfloor$ as the minimum group that requires no SP communication.
    - Design Motivation: Manual group selection is suboptimal (ablations show automatic selection is 12% faster), and optimal strategies vary across hardware and model configurations, making automation necessary.

2. **Balance Packing (Algorithm 2)**:

    - Function: Distributes samples across hierarchical groups to jointly optimize ABR, CR, DBR, and PR.
    - Mechanism: (a) The dataset is partitioned into subsets $D_1, ..., D_n$ by sequence length; (b) each subset is independently packed to its corresponding length $l_i$; (c) GreedyFill uses residual data from smaller groups to fill gaps in larger groups, reducing PR; (d) Balance Batching sorts sequences by attention complexity to construct balanced batches and reduce ABR.
    - Design Motivation: The hierarchical structure naturally isolates long and short data so that short sequences bypass SP communication (CR drops from 1.0 to 0.173), while attention-complexity-based sorting equalizes compute across DP groups (ABR drops from 0.506 to 0.002).

3. **Stable Loss Normalizer**:

    - Function: Normalizes the loss using the global average token count $T_{ave}$ as a replacement for Token-Mean or Sample-Mean normalization.
    - Mechanism: $\mathcal{L}_{stable} = \frac{\sum_i^{B_l} loss_i}{B_l \cdot T_{ave}}$, where $T_{ave} = \frac{\sum_i^{B_g} T_i}{B_g}$ is the average token count over the global batch.
    - Design Motivation: Token-Mean introduces bias when token counts differ across DP groups; Sample-Mean over-weights short sequences; Sum Loss causes gradient explosion (up to 1e+5). Stable Loss ensures each token contributes equally to the total loss.

### Loss & Training
- **Curriculum Learning**: Training begins with short-context data only for the first 500 steps, followed by interleaved long/short mixed training, avoiding severe loss fluctuations caused by insufficient instruction-following ability in early training.
- **Adaptive SP**: Multiple SP configurations are pre-initialized (zero switching overhead), allowing each packing group to use its own optimal SP degree.

## Key Experimental Results

### Main Results

| Model | Method | General AVE | Ruler-32K | Ruler-128K | LongBench | GPU Days | Speedup |
|-------|--------|-------------|-----------|------------|-----------|----------|---------|
| LLaMA3.1-8B | ISF (baseline) | 56.0 | 85.0 | 67.4 | 44.0 | 5.22 | 1.0× |
| LLaMA3.1-8B | **HBP** | **58.2** | 85.6 | **70.8** | 43.1 | **3.73** | **1.4×** |
| Qwen2.5-32B | ISF | 73.5 | 88.2 | 59.3 | 51.0 | 21.3 | 1.0× |
| Qwen2.5-32B | **HBP** | **76.2** | 88.3 | 59.0 | **51.9** | **16.0** | **1.33×** |
| LLaMA3.1-70B | ISF | 72.1 | 91.8 | 57.1 | 50.4 | 44.4 | 1.0× |
| LLaMA3.1-70B | **HBP** | **74.2** | **93.4** | 57.5 | **52.2** | **31.1** | **1.42×** |
| DeepSeek-V2 (236B) | ISF | 71.8 | 86.6 | - | 47.1 | 57.1 | 1.0× |
| DeepSeek-V2 (236B) | **HBP** | **72.0** | **87.3** | - | **50.3** | **23.8** | **2.4×** |

### Ablation Study

| Configuration | ABR | CR | AVE | GPU Days (Speedup) |
|---------------|-----|----|-----|--------------------|
| ISF baseline | 0.506 | 1.0 | 56.0 | 5.22 (1.0×) |
| + Hierarchical packing | 0.288 | 0.173 | 56.4 | 4.51 (1.2×) |
| + Balance batching | **0.002** | **0.173** | 56.6 | **3.73 (1.4×)** |
| + Curriculum learning | 0.002 | 0.173 | **58.2** | 3.73 (1.4×) |

### Key Findings
- **Hierarchical packing** reduces CR from 1.0 to 0.173 (short sequences no longer participate in unnecessary SP communication), contributing a 1.2× speedup.
- **Balance batching** reduces ABR from 0.288 to 0.002, contributing an additional 0.2× speedup.
- **Stable Loss** outperforms all four normalization variants: AVE 57.6, Ruler-128K 70.8 vs. Token-Mean's 56.6 and 67.5, respectively.
- **MoE models benefit most**: The high communication overhead of DeepSeek-V2 amplifies HBP's communication savings to 2.4×.
- Curriculum learning requires only 100–500 warmup steps to yield substantial gains (AVE +1.6, LongBench +1.5).

## Highlights & Insights
- **Metric-driven methodology**: ABR and CR precisely quantify the root causes of naïve packing inefficiency, directly informing the design of hierarchical packing. This paradigm of "quantify first, then solve" carries significant methodological value.
- **Hierarchical structure as a unified solution**: Multi-level grouping naturally isolates long and short data (reducing CR), naturally accommodates curriculum learning (short-to-long progression), and naturally matches different SP/GC configurations—one structure addressing three distinct problems.
- **Theoretical grounding of Stable Loss**: The derivation of $T_{ave}$ normalization from the principle of equal per-token loss contribution is both elegant and highly effective, and can be directly applied to any mixed-length training setting.
- **Transferability**: The hierarchical strategy is transferable to multimodal training (where image-text length disparities are large) and to MoE pre-training.

## Limitations & Future Work
- **Profiling overhead**: Automatic group selection requires upfront profiling of iteration times across configurations, incurring a one-time cost for new hardware or model architectures.
- **Static grouping**: The grouping scheme is fixed prior to training and cannot adapt to dynamic shifts in data distribution during the training process.
- **SFT-only validation**: The method is not evaluated in pre-training settings, where different data distributions and scales may introduce new challenges.
- **Simple curriculum design**: Only a two-stage strategy (short-only → mixed) is employed; more fine-grained curricula (e.g., progressively increasing sequence length) may yield further improvements.

## Rating
- Novelty: ⭐⭐⭐⭐ — The multi-level packing concept is clear and well-motivated, and the ABR/CR metrics offer genuine insight; however, the individual technical components (packing, curriculum learning, SP) are all combinations of known techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers four model scales from 8B to 236B, with comprehensive ablations and metric analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear, algorithmic descriptions are complete, and metric definitions are rigorous.
- Value: ⭐⭐⭐⭐⭐ — Addresses a genuine bottleneck in long-context SFT; the 2.4× speedup is practically deployable and directly benefits industrial applications.

## Related Work & Insights

| Method | Core Idea | HBP Advantage |
|--------|-----------|---------------|
| LongAlign | Single-level packing + loss reweighting | HBP's multi-level packing eliminates attention imbalance (ABR 0.506→0.002) without sacrificing long-context performance (Ruler-128K 70.8 vs. 57.5). |
| FlexSP | Online dynamic grouping + flexible SP | HBP's offline grouping incurs zero runtime overhead (vs. FlexSP's 5–15s per iteration), achieves lower ABR (0.002 vs. 0.36), and is faster overall (3.73 vs. 4.35 GPU Days). |
| ISF Packing | Single-level iterative sampling fill | ISF ignores attention complexity differences and SP communication waste; HBP achieves 1.33–2.4× speedup across all model scales. |
| Sorted Batching | Length-sorted batch construction | Sorting cannot control attention balance across DP groups, and long sequences still share SP communication with short ones. |

Core distinction: HBP is the first to extend the packing length from a single fixed value to a multi-level hierarchical structure, with each level independently configured for SP and GC, fundamentally decoupling the training pipelines for long and short data.

## Inspiration & Connections
- **Metric-driven design as a research paradigm**: The introduction of ABR and CR serves as the foundation of the entire work—quantifying problems (attention imbalance, communication waste) before designing targeted solutions. This "define metrics → analyze bottlenecks → design method" paradigm is broadly applicable.
- **Connection to multimodal training**: Multimodal SFT exhibits even greater length disparities between image and text (image tokens: 576–2048; text: 100–8K), making HBP's hierarchical strategy directly transferable.
- **Broad applicability of Stable Loss**: The theoretical derivation of $T_{ave}$ normalization (equal per-token loss contribution) is independent of HBP's hierarchical structure and can be applied standalone in any mixed-length training scenario, including pre-training.
- **Synergy with MoE training**: The 2.4× speedup on DeepSeek-V2 demonstrates that MoE models, due to their large expert communication overhead, are more sensitive to SP communication optimization. HBP's communication reduction strategy is especially important for MoE architectures.
- **Natural embedding of curriculum learning**: The hierarchical structure inherently supports a short-to-long curriculum without requiring a custom sampler, and this engineering simplicity is critical for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ICLR 2026\] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](../../ICLR2026/llm_efficiency/tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)
- [\[NeurIPS 2025\] Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[NeurIPS 2025\] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)

</div>

<!-- RELATED:END -->
