---
title: >-
  [Paper Note] LightMem: Lightweight and Efficient Memory-Augmented Generation
description: >-
  [ICLR 2026][Model Compression][LLM memory system] This paper proposes LightMem, a three-stage lightweight memory system inspired by the human Atkinson–Shiffrin memory model. Through three modules — cognitive sensory memo…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "LLM memory system"
  - "sensory memory"
  - "short-term memory"
  - "long-term memory"
  - "sleep-time updating"
date: 2026-05-08
content_hash: c5d00edb1ff4dfdc
---

# LightMem: Lightweight and Efficient Memory-Augmented Generation

**Conference**: ICLR 2026
**arXiv**: [2510.18866](https://arxiv.org/abs/2510.18866)  
**Code**: [GitHub](https://github.com/zjunlp/LightMem)  
**Area**: Model Compression
**Keywords**: LLM memory system, sensory memory, short-term memory, long-term memory, sleep-time updating

## TL;DR
This paper proposes LightMem, a three-stage lightweight memory system inspired by the human Atkinson–Shiffrin memory model. Through three modules — cognitive sensory memory pre-compression, topic-aware short-term memory consolidation, and offline sleep-time updating — LightMem achieves up to 7.7% accuracy improvement on LongMemEval while reducing token consumption by up to 38×.

## Background & Motivation
LLMs struggle to effectively leverage historical information in dynamic and complex interactive environments, and memory systems have emerged as a promising solution. However, existing memory systems suffer from three major efficiency bottlenecks:

**Redundant sensory input**: Raw dialogues contain large amounts of irrelevant information; processing them directly wastes resources and may even impair in-context learning.

**Coarse-grained organization**: Fixed-window segmentation causes semantic confusion, while single-turn segmentation leads to frequent API calls — neither approach is satisfactory.

**Real-time update bottleneck**: Memory updates executed during inference introduce latency, and read-write dependencies require serial processing.

The root cause is the performance–efficiency trade-off: existing systems are either accurate but costly, or efficient but coarse. The core idea of LightMem is to emulate the three-layer structure of human memory: rapid filtering (sensory memory) → organization (short-term memory) → deep consolidation (long-term memory, executed offline).

## Method

### Overall Architecture
LightMem passes dialogue information sequentially through three "Light" modules: Light1 pre-compression + topic segmentation → Light2 topic-aware short-term memory → Light3 long-term memory with sleep-time updating. During online inference, only lightweight operations are performed (direct insertion); expensive recomputation is deferred to the offline "sleep" phase.

### Key Designs
1. **Light1: Cognitive Sensory Memory**:

    - Pre-compression sub-module: Uses LLMLingua-2 as the compression model, computing a retention probability $P(\text{retain } x_i | \bm{x}; \theta)$ for each token. A threshold $\tau$ is set at the $r$-th percentile to retain high-information tokens. Cross-entropy-based filtering is also supported (high entropy = high information = retain).
    - Topic segmentation sub-module: Maintains a sensory buffer and triggers hybrid segmentation when the buffer is full. Attention boundaries $\mathcal{B}_1$ are identified as local maxima in attention scores between adjacent turns; semantic boundaries $\mathcal{B}_2$ are positions where embedding similarity between adjacent turns falls below a threshold. The final boundary set is $\mathcal{B} = \mathcal{B}_1 \cap \mathcal{B}_2$.
    - Design motivation: Compression removes redundancy; topic segmentation prevents semantic confusion.

2. **Light2: Topic-Aware Short-Term Memory**:

    - Function: Topic segments are temporarily stored in the STM buffer; once the token threshold is reached, an LLM is called to generate concise summaries.
    - Mechanism: The storage structure is {topic, {$\text{sum}_i$, $\text{user}_i$, $\text{model}_i$}}, with summaries indexed into long-term memory via embeddings.
    - Design motivation: Topic-constrained granularity achieves the optimal balance between minimizing API calls and maintaining summarization accuracy.

3. **Light3: Sleep-Time Long-Term Memory Updating**:

    - Online phase: New memory entries are directly inserted (soft update) without merging or deduplication.
    - Offline phase: An update queue $\mathcal{Q}(e_i) = \text{Top}_k\{(e_j, \text{sim}(v_i, v_j)) | t_j \geq t_i\}$ is computed for each memory entry, allowing only entries with later timestamps to update earlier ones. Queues are independent and can be processed in parallel.
    - Design motivation: Decoupling updates from inference eliminates online latency; parallel updating significantly reduces offline latency.

### Loss & Training
LightMem is a training-free pipeline system. Its core hyperparameters are the compression ratio $r$ and the STM buffer capacity $th$.

## Key Experimental Results

### Main Results (LongMemEval-S, GPT-4o-mini)

| Method | ACC (%) | Total Tokens (k) | API Calls | Runtime (s) |
|--------|---------|-----------------|-----------|-------------|
| FullText | 56.80 | 105.07 | - | - |
| A-MEM | 62.60 | 1605.81 | 986 | 5132 |
| MemoryOS | 44.80 | 2991.75 | 2938 | 8030 |
| Mem0 | 53.61 | 1152.62 | 812 | 4248 |
| **LightMem** (r=0.7, th=512) | **68.64** | **28.25** | **18** | **284** |
| + Offline Update | 67.07 | 111.69 | 144 | 496 |

### Ablation Study

| Configuration | ACC (%) | Token Efficiency | Notes |
|---------------|---------|-----------------|-------|
| r=0.5, th=256 | 64.29 | 30.81k | Lower compression, more information retained |
| r=0.6, th=256 | 67.78 | 35.11k | Moderate compression |
| r=0.7, th=512 | 68.64 | 28.25k | Best accuracy–efficiency combination |
| No compression | Lower | Higher | Redundant information disrupts in-context learning |
| No topic segmentation | Lower | Slightly lower | Semantic confusion degrades summarization quality |

### Key Findings
- LightMem achieves the highest accuracy while incurring the lowest token cost: +6% accuracy over A-MEM with 57× fewer tokens.
- Online testing overhead is minimal: token consumption reduced by up to 106×, API calls reduced by up to 159×.
- Moderate compression (r=0.7) outperforms low compression (r=0.5), confirming that redundancy removal genuinely improves in-context learning.
- Sleep-time parallel updating is several times faster than serial updating.

## Highlights & Insights
- The three-stage mapping to the human memory model is natural and effective: sensory filtering → working memory → long-term consolidation.
- The "sleep-time updating" concept is elegant, fully decoupling expensive operations from user interaction.
- The training-free, plug-and-play design enables seamless integration with any LLM backend.
- The finding that compression can improve performance offers meaningful guidance for memory system design.

## Limitations & Future Work
- The quality of the compression model (LLMLingua-2) directly affects downstream performance.
- Topic segmentation thresholds require tuning, and cross-domain generalization remains uncertain.
- There is no adaptive mechanism for determining the timing and frequency of offline updates.
- Evaluation is primarily conducted on conversational scenarios; other long-term interaction paradigms (e.g., code development) have not been tested.

## Related Work & Insights
- **vs. A-MEM**: Higher accuracy with over 60× fewer tokens.
- **vs. MemoryOS**: Avoids the high latency introduced by real-time updates.
- **vs. NaiveRAG**: Structured memory organization is more effective than simple retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage memory architecture is cleverly designed, and the sleep-time updating concept is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers LongMemEval + LoCoMo, multiple backbone models, and comprehensive efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed complexity analysis.
- Value: ⭐⭐⭐⭐⭐ Highly practical, with substantial efficiency gains and strong potential for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QKV Projections Require a Fraction of Their Memory](qkv_projections_require_a_fraction_of_their_memory.md)
- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [\[NeurIPS 2025\] Memory-Efficient Training with In-Place FFT Implementation](../../NeurIPS2025/model_compression/memory-efficient_training_with_in-place_fft_implementation.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)

</div>

<!-- RELATED:END -->
