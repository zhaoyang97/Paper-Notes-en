---
title: >-
  [Paper Note] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers
description: >-
  [NeurIPS 2025][Reinforcement Learning][hybrid memory] This paper proposes Hybrid Quadratic-Linear Transformers (HQLT), which integrate KV-memory (softmax attention: precise retrieval but quadratic complexity) and FW-memo…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "hybrid memory"
  - "softmax attention"
  - "fast weight programming"
  - "DeltaNet"
  - "linear transformer"
  - "complementary learning systems"
date: 2026-05-08
content_hash: 2794aeac66d65250
---

# Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2506.00744](https://arxiv.org/abs/2506.00744)  
**Code**: [https://github.com/kazuki-irie/hybrid-memory](https://github.com/kazuki-irie/hybrid-memory)  
**Area**: Transformer Architecture / Hybrid Models / Sequence Modeling
**Keywords**: hybrid memory, softmax attention, fast weight programming, DeltaNet, linear transformer, complementary learning systems

## TL;DR
This paper proposes Hybrid Quadratic-Linear Transformers (HQLT), which integrate KV-memory (softmax attention: precise retrieval but quadratic complexity) and FW-memory (DeltaNet/linear attention: linear complexity but coarse retrieval) as complementary memory systems. Three hybrid strategies are compared (Delayed-Streaming, Delayed-Chunk, and Synchronous), and the Synchronous variant is shown to be optimal across language modeling, retrieval, algorithmic reasoning, and RL tasks at 340M and 1.3B parameter scales.

## Background & Motivation
**Background**: Modern Transformers fall into two categories — Quadratic Transformers (QT, standard softmax attention) and Linear Transformers (LT, e.g., DeltaNet). Their computational properties are complementary yet each has fundamental limitations:
   - QT achieves precise retrieval via softmax, but is constrained by quadratic complexity and requires a fixed maximum context window.
   - LT achieves linear complexity and supports arbitrarily long sequences via fast weight matrices; the DeltaNet variant additionally supports state tracking computations that QT cannot perform, but at the cost of retrieval precision.

**Limitations of Prior Work**: No single system can simultaneously satisfy precise retrieval, long-context processing, and high expressivity. Existing hybrid attempts (Arora et al., Munkhdalai et al.) rely on outdated LT formulations (vanilla linear attention) and fail to leverage DeltaNet's expressive advantages.

**Biological Inspiration**: The brain integrates multiple memory mechanisms through Complementary Learning Systems (CLS) — the hippocampus handles episodic memory (precise but capacity-limited), while the cortex handles semantic memory (abstract but persistent). Analogously, KV-memory corresponds to precise short-term memory, and FW-memory corresponds to compressed long-term memory.

**Key Challenge**: Precise retrieval requires explicitly storing all key-value pairs (quadratic growth), while long-context processing requires a fixed-size compressed state (linear complexity); the two are fundamentally incompatible within a single system.

**Key Insight**: Rather than choosing between the two Transformer classes, the paper designs a hybrid architecture in which two memory systems serve complementary roles — the central question being when and how information is allocated to each system.

**Core Idea**: Construct a complementary memory system using DeltaNet's FW-memory and softmax KV-memory, and maximize their joint benefits through a synchronous input strategy.

## Method

### Overall Architecture
At each time step, HQLT receives input $\mathbf{x}_t$ and simultaneously maintains two memory types: KV-memory (key-value pairs within a bounded sliding window) and FW-memory (a fixed-size fast weight matrix). The output is a weighted combination of the outputs from both memory systems. The three hybrid strategies differ in the timing and manner in which information flows to each system.

### Key Designs
1. **Delayed-Streaming HQLT**:

    - Newly generated key-value pairs enter KV-memory; pairs evicted by the sliding window subsequently "stream into" FW-memory.
    - A clear division of labor: FW-memory covers history from $\leq t-S$ steps ago, and KV-memory handles precise retrieval over the most recent $S$ steps.
    - Conceptually elegant — memory systems are allocated by information "age."
    - Drawback: FW-memory only processes stale information and cannot exploit its expressivity on current inputs.

2. **Delayed-Chunk HQLT**:

    - Derived from chunk-wise parallel training: softmax attention (QT) is applied within chunks, and the recurrent form of FW-memory (LT) is applied across chunks.
    - Directly related to the model of Munkhdalai et al.
    - Shares the expressivity limitations of the delayed architecture.

3. **Synchronous HQLT**:

    - At each time step, key-value pairs are fed **simultaneously** into both KV-memory and FW-memory.
    - Motivation: DeltaNet's state-tracking capability (e.g., parity computation, modular arithmetic) is inaccessible to softmax attention; FW-memory must process current inputs to exercise this advantage.
    - Output blending: $\mathbf{y}_t = \gamma_t \odot \text{FW-output} + (1-\gamma_t) \odot \text{KV-output}$, where $\gamma_t$ is a dynamic vector gate.

4. **Memory Blending / Gating Mechanisms**:

    - Additive blending: directly sums the outputs of both systems.
    - Dynamic scalar blending: generates two sigmoid scalars $\alpha_t^{FW}, \alpha_t^{KV}$ to scale each output independently.
    - Dynamic vector blending: generates a vector $\gamma_t \in \mathbb{R}^{d_{out}}$ for element-wise interpolation — experiments show this strategy is optimal.

### DeltaNet as the FW-memory Component
- DeltaNet updates its fast weight matrix using the delta learning rule: $\mathbf{W}_t = \mathbf{W}_{t-1} + \sigma(\beta_t)(\mathbf{v}_t - \mathbf{W}_{t-1}\phi(\mathbf{k}_t)) \otimes \phi(\mathbf{k}_t)$
- Here $\mathbf{v}_t$ is the target, $\phi(\mathbf{k}_t)$ is the input, $\sigma(\beta_t)$ is the learning rate, and $\phi$ denotes SiLU followed by L2 normalization.
- Key advantage: the rank-one update of the delta rule endows DeltaNet with state-tracking capability (via negative eigenvalues), which vanilla linear attention and softmax attention do not possess.

## Key Experimental Results

### Language Modeling (15B tokens, FineWeb-Edu)

| Model (340M) | Wiki PPL↓ | LAMBADA PPL↓ | PiQA | HellaSwag | ARC-e | Mean |
|-------------|----------|-------------|------|----------|-------|------|
| Transformer++ | 26.5 | 34.9 | 67.6 | 41.0 | 60.2 | 47.6% |
| DeltaNet | 27.6 | 35.0 | 67.1 | 40.8 | 58.5 | 46.8% |
| HQLT Delayed-Streaming | 26.5 | 30.5 | 66.7 | 42.1 | 60.8 | 47.5% |
| **HQLT Synchronous** | **26.3** | **29.4** | 66.2 | **42.7** | **61.5** | **47.8%** |

| Model (1.3B) | Wiki PPL↓ | LAMBADA PPL↓ | PiQA | HellaSwag | ARC-e | Mean |
|-------------|----------|-------------|------|----------|-------|------|
| Transformer++ | 19.8 | 17.9 | 71.0 | 50.3 | 65.2 | 53.0% |
| **HQLT Synchronous** | **19.8** | **15.9** | **72.0** | **51.5** | **68.1** | **53.9%** |

### Synthetic Algorithmic Tasks (Critical Discriminative Experiments)

| Task | Transformer | DeltaNet | Delayed-Streaming | Delayed-Chunk | Synchronous |
|------|-----------|---------|--------|--------|------|
| Parity (mod 2) | Fails | ✓ Succeeds | Partial | Partial | ✓ Succeeds |
| Modular arithmetic (mod 7) | Fails | ✓ Succeeds | Fails | Fails | ✓ Succeeds |

### Ablation Study — Blending Strategy Comparison

| Gating Mechanism | Wiki PPL | LAMBADA PPL | Mean |
|---------|---------|-------------|------|
| Additive blending | 26.5 | 30.9 | 47.4% |
| Dynamic scalar | 26.4 | 30.0 | 47.6% |
| **Dynamic vector** | **26.3** | **29.4** | **47.8%** |

### RL Experiments (Partially Observable Environments)
- HQLT outperforms pure QT and pure LT on partially observable Atari tasks, demonstrating the value of hybrid memory in RL settings.

### Key Findings
- **Synchronous >> Delayed**: The gap is most pronounced on synthetic tasks — delayed architectures entirely fail to preserve DeltaNet's state-tracking capability (failing on mod 7) because FW-memory only receives stale information.
- **Dynamic vector gating is optimal**: Per-dimension control over the contributions of both memory systems is more flexible than fixed or scalar weighting.
- **Benefits amplified at 1.3B scale**: Synchronous HQLT achieves a larger gain over Transformer++ at 1.3B (mean 53.9% vs. 53.0%), suggesting that the advantage of hybrid memory grows with scale.
- **Most significant improvement on LAMBADA**: HQLT reduces LAMBADA PPL from 17.9 to 15.9 — the largest relative gain, reflecting FW-memory's long-context compression capability.

## Highlights & Insights
- **Precise complementarity analysis**: Table 1 clearly contrasts the two memory types across four dimensions (complexity / context length / retrieval precision / expressivity); the conceptual contribution exceeds the engineering implementation.
- **Synthetic tasks as a killer experiment**: The mod 7 task precisely exposes the fatal flaw of delayed architectures — only Synchronous HQLT simultaneously inherits DeltaNet's state tracking and softmax's precise retrieval. This finding is entirely absent from prior hybrid Transformer literature.
- **Biologically grounded analogy**: Rather than a superficial parallel, the paper derives a genuinely functionally complementary dual-system architecture from Complementary Learning Systems theory.
- **Engineering compatibility**: All models are compatible with efficient implementations via flash-attention and flash-linear-attention, making them practically deployable.

## Limitations & Future Work
- **Limited training scale**: Only 15B tokens, far below current LLM standards (hundreds of billions to trillions); whether hybrid advantages persist at larger scales remains unknown.
- **Marginal gains on downstream tasks**: Improvements on standard NLP benchmarks are consistent but modest (0.1–0.9%).
- **DeltaNet variants unexplored**: Only the base DeltaNet is evaluated; stronger variants such as Gated Delta Rule or Delta Product Rule are not tested.
- **Window size $S$ not systematically analyzed**: The effect of KV-memory window size on hybrid performance lacks thorough ablation.
- **MoE integration**: The hybrid memory paradigm could be combined with Mixture-of-Experts (with different experts employing different memory ratios), but remains unexplored.

## Related Work & Insights
- **vs. Jamba/Zamba**: Large-scale Mamba-Transformer hybrids, but use Mamba (not DeltaNet) and do not analyze expressivity complementarity.
- **vs. Munkhdalai et al.**: The first QT-LT hybrid, but employs a delayed-chunk strategy with vanilla linear attention; this paper demonstrates that such design misses DeltaNet's core advantages.
- **vs. Arora et al.**: Uses a synchronous strategy but pairs it with vanilla linear attention, resulting in substantially weaker performance than DeltaNet-based variants.
- **Complementary Learning Systems theory**: The hippocampus-cortex complementarity hypothesis of McClelland et al. receives a computational realization here — KV-memory corresponds to the hippocampus (precise episodic memory), and FW-memory corresponds to the cortex (compressed semantic memory).

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic comparison of hybrid strategies fills a gap in the literature; the finding that synchronous > delayed is genuinely valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-pronged validation across language modeling, synthetic tasks, and RL, though training scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from theoretical background through method design to experimental validation is complete; concepts are explained with clarity.
- Value: ⭐⭐⭐⭐ Significant contributions to both Transformer architecture design and memory systems theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
