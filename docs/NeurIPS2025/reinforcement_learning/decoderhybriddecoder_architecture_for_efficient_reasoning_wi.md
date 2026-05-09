---
title: >-
  [Paper Note] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation
description: >-
  [NeurIPS 2025][Reinforcement Learning][Hybrid Architecture] SambaY proposes the Gated Memory Unit (GMU) for sharing SSM token-mixing representations across layers, replacing half of the cross-attention layers in YOCO's cross-decoder with lightweight GMUs. This maintains linear prefill complexity and long-context retrieval capability while substantially improving decoding efficiency. The resulting product, Phi4-mini-Flash-Reasoning (3.8B), outperforms Phi4-mini-Reasoning on reasoning benchmarks and achieves up to 10× decoding throughput improvement in the 2K prompt + 32K generation setting.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Hybrid Architecture
  - SSM
  - Decoding Efficiency
  - Long Sequence Generation
  - KV Cache
date: 2026-05-08
content_hash: 2609485741cff204
---

# Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation

**Conference**: NeurIPS 2025
**arXiv**: [2507.06607](https://arxiv.org/abs/2507.06607)
**Code**: [github.com/microsoft/ArchScale](https://github.com/microsoft/ArchScale)
**Area**: Reinforcement Learning
**Keywords**: Hybrid Architecture, SSM, Decoding Efficiency, Long Sequence Generation, KV Cache

## TL;DR
SambaY proposes the Gated Memory Unit (GMU) for sharing SSM token-mixing representations across layers, replacing half of the cross-attention layers in YOCO's cross-decoder with lightweight GMUs. This maintains linear prefill complexity and long-context retrieval capability while substantially improving decoding efficiency. The resulting product, Phi4-mini-Flash-Reasoning (3.8B), outperforms Phi4-mini-Reasoning on reasoning benchmarks and achieves up to 10× decoding throughput improvement in the 2K prompt + 32K generation setting.

## Background & Motivation

**Background**: Hybrid architectures combining SSMs/RNNs (e.g., Mamba) with Transformers (e.g., Samba, YOCO) have demonstrated the ability to significantly improve inference efficiency without sacrificing performance. YOCO achieves linear prefill complexity through a decoder-decoder structure that caches KV only once.

**Limitations of Prior Work**: YOCO's cross-decoder still employs full cross-attention layers, incurring $O(d_{kv} \cdot N)$ attention memory I/O cost during the **generation phase** (rather than the prefill phase). For the extremely long chain-of-thought sequences produced by modern reasoning LLMs (e.g., 32K tokens), this overhead becomes a new bottleneck.

**Key Challenge**: SSM layers naturally produce intermediate token-mixing representations that could be shared across layers, yet no prior work has explored leveraging such cross-layer representation sharing to reduce memory I/O during decoding.

**Goal**: How can the memory I/O overhead of attention layers in the cross-decoder be reduced without sacrificing long-context retrieval capability?

**Key Insight**: Design the GMU (Gated Memory Unit)—a simple gating mechanism that allows certain layers of the cross-decoder to directly reuse the token-mixing output $M^{(l')}$ from the last SSM layer of the self-decoder, applying channel-wise recalibration via a gating signal derived from the current layer's input.

**Core Idea**: Replace half of the cross-attention layers with GMUs, reducing decoding memory I/O from linear $O(d_{kv} \cdot N)$ to constant $O(d_h)$, while enabling fine-grained reweighting of prior token-mixing representations through gating.

## Method

### Overall Architecture
SambaY adopts a decoder-hybrid-decoder architecture: (1) the **self-decoder** (first half of layers) uses Samba (alternating Mamba + SWA), with a single full attention layer at the end to produce the KV cache; (2) the **cross-decoder** (second half of layers) alternates between cross-attention (reusing the KV cache) and GMU (reusing the SSM token-mixing output). During prefill, only the self-decoder needs to be executed (linear complexity); during decoding, GMU layers require only $O(d_h)$ constant memory I/O.

### Key Designs

1. **Gated Memory Unit (GMU)**:

    - **Function**: Replaces cross-attention with a lightweight gating mechanism to enable cross-layer representation sharing.
    - **Mechanism**: $Y^l = (M^{(l')} \odot \sigma(X^l W_1^T)) W_2$, where $M^{(l')}$ is the token-mixing output from a prior SSM layer, $X^l$ is the current layer input, and $\sigma$ denotes SiLU activation. The gating signal lifts the 2D token-mixing matrix $A^{(l')}$ into a 3D tensor $\tilde{A}_{ijk} = G_{ik}^{(l)} A_{ij}^{(l')}$, enabling channel-wise reweighting of token mixing.
    - **Design Motivation**: During decoding, SSMs need only maintain a constant-size state $\mathbf{m} \in \mathbb{R}^{d_h \times d_h}$, avoiding the linear I/O over the entire KV cache required by cross-attention. Both parameter count and computational cost are far lower than standard attention layers.

2. **μP++ Hyperparameter Scaling Rules**:

    - **Function**: Provides a unified hyperparameter transfer scheme for simultaneous depth and width scaling.
    - **Mechanism**: Integrates μP (width scaling) + Depth-μP (depth scaling: learning rate $\eta \propto 1/d$, residual branch output divided by $\sqrt{2d}$) + zero weight decay for vector-like parameters.
    - **Design Motivation**: Standard μP exhibits training instability (NaN loss) during large-scale training (600B tokens); zero weight decay resolves this issue.

3. **Iso-Parametric Equation**:

    - **Function**: Enables fair comparison of scaling behavior across different architectures.
    - **Mechanism**: By formulating parameter count equations for different architectures and solving for the aspect ratio $\alpha$ of each, architectures of equal parameter count can be made to share the same depth, ensuring consistent KV cache sizes for fair inference-time comparisons.
    - **Design Motivation**: Naively adjusting depth to match parameter counts alters KV cache size, leading to unfair runtime comparisons.

### Loss & Training
Standard language model cross-entropy loss. Phi4-mini-Flash is pretrained on 5T tokens using standard parameterization (μP++ was not applied due to resource constraints). Severe loss divergence was encountered during training and mitigated via FP32 upscaling and attention dropout. The reasoning model was produced through SFT + DPO distillation (no RL).

## Key Experimental Results

### Main Results

**Reasoning Performance (Phi4-mini-Flash-Reasoning vs. Phi4-mini-Reasoning, both 3.8B)**

| Benchmark | Phi4-mini-Reasoning | Phi4-mini-Flash-Reasoning |
|------|----|----|
| AIME24 (Pass@1, avg 64) | 48.13 | **52.29** |
| AIME25 (Pass@1, avg 64) | 31.77 | **33.59** |
| Math500 (Pass@1, avg 8) | 91.20 | **92.45** |
| GPQA Diamond (Pass@1, avg 8) | 44.51 | **45.08** |
| Decoding Throughput (2K prompt + 32K gen) | 1× | **~10×** |

**Scaling Experiments (FLOPs scaling, 1B–3.4B models)**

| Architecture | Irreducible Loss $C$ | Notes |
|------|------------|------|
| Transformer++ (μP++) | 0.64 | Highest irreducible loss |
| Samba+YOCO (μP++) | 0.60 | Second best |
| **SambaY (μP++)** | **0.58** | Lowest irreducible loss; greatest scaling potential |

### Ablation Study

| Configuration | PB-32K | Short-Task Avg. | Notes |
|------|--------|----------|------|
| SambaY (SWA=256) | 78.13 | 52.16 | Best overall balance |
| MambaY (no SWA) | 12.50 | 51.87 | Retrieval collapses without local attention |
| SambaY-2 (Mamba→Mamba-2) | 40.63 | 51.00 | Scalar forget gate loses positional information |
| SambaY-MLP (GMU gates MLP repr.) | 64.84 | **52.65** | Best on short tasks but weaker long-context |
| SambaY-AA (all GMU, no cross-attn) | 46.88 | 52.06 | Cross-attention remains indispensable |

### Key Findings
- Phi4-mini-Flash-Reasoning **without RL** surpasses Phi4-mini-Reasoning **with RL**, demonstrating that architectural improvements can compensate for differences in training pipelines.
- SambaY's irreducible loss (0.58) is substantially lower than Transformer++ (0.64), indicating that SambaY achieves lower loss in the infinite-compute limit.
- A small SWA window (256) suffices for SambaY to attain strong long-context retrieval performance; large windows are unnecessary.
- Hybrid architectures without positional encodings (NoPE) can zero-shot extrapolate to 2× the training length.
- GMU gating over SSM representations significantly outperforms gating over attention or MLP representations for long-context retrieval.

## Highlights & Insights
- **The mathematical interpretation of GMU is elegant**: The gating operation is equivalent to lifting the 2D token-mixing matrix into a 3D tensor $\tilde{A}_{ijk}$, providing a clean theoretical framework for understanding cross-layer representation sharing.
- **The finding that "RL-free training surpasses RL-trained models" is highly compelling**: It suggests that the reasoning quality gains from efficient architectures may be underestimated—faster inference enables longer/more CoT sequences within the same wall-clock time, indirectly improving reasoning quality.
- **The engineering contribution of μP++ is significant**: It addresses training instability encountered in practical large-scale training, offering direct practical value to the community.

## Limitations & Future Work
- A single full attention layer is retained, meaning decoding complexity remains linear rather than constant.
- Severe loss divergence was encountered during large-scale training of Phi4-mini-Flash; the current mitigation (label smoothing + attention dropout) is heuristic.
- No fair comparison against RL-trained reasoning models is provided (Flash-Reasoning uses only SFT + DPO).
- No targeted search over optimization hyperparameters was conducted, leaving potential performance gains unexplored.
- The vLLM implementation of Differential Attention relies on four naive FlashAttention calls, leaving efficiency under-optimized.

## Related Work & Insights
- **vs. YOCO**: SambaY replaces half of the cross-attention layers in YOCO's cross-decoder with GMUs, reducing decoding memory I/O from linear to constant. Scaling experiments show SambaY achieves lower irreducible loss.
- **vs. Samba**: Samba is simply an alternating arrangement of Mamba and SWA. SambaY extends this with YOCO's decoder-decoder structure and the GMU cross-layer sharing mechanism.
- **vs. CLA (Cross-Layer Attention)**: CLA shares KV caches across layers; SambaY shares SSM token-mixing outputs, avoiding the materialization of recurrent states.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — GMU design is elegant; decoder-hybrid-decoder is a meaningful extension of YOCO.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Scaling experiments from 1B to 3.8B, long-context retrieval, ablations, reasoning benchmarks, and throughput measurements are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture descriptions are clear, though some notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ — Serves as the architectural backbone of Microsoft's Phi4 series with real product deployment; highly significant for efficient serving of long-CoT reasoning models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[NeurIPS 2025\] Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics](solving_neural_min-max_games_the_role_of_architecture_initialization_dynamics.md)
- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[NeurIPS 2025\] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)

<!-- RELATED:END -->
