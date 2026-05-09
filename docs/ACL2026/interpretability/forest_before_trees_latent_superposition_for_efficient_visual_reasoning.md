---
title: >-
  [Paper Note] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning
description: >-
  [ACL 2026][latent reasoning] This paper proposes Laser, a framework that conducts visual reasoning in latent space via Dynamic Window Alignment Learning (DWAL), enabling the model to maintain a probabilistic "superposition state" over future semantics rather than performing precise per-token prediction. This realizes a "global-before-local" cognitive hierarchy, achieving state-of-the-art performance among latent reasoning methods on 6 benchmarks with only 6 reasoning tokens (a reduction of 97%+), surpassing Monet by an average of 5.03%.
tags:
  - ACL 2026
  - latent reasoning
  - dynamic window alignment
  - semantic superposition
  - visual reasoning
  - token efficiency
date: 2026-05-08
content_hash: 3456c4936efd86f7
---

# Forest Before Trees: Latent Superposition for Efficient Visual Reasoning

**Conference**: ACL 2026
**arXiv**: [2601.06803](https://arxiv.org/abs/2601.06803)
**Code**: [GitHub](https://github.com/Laser-VLM/Laser)
**Area**: Interpretability
**Keywords**: latent reasoning, dynamic window alignment, semantic superposition, visual reasoning, token efficiency

## TL;DR

This paper proposes Laser, a framework that conducts visual reasoning in latent space via Dynamic Window Alignment Learning (DWAL), enabling the model to maintain a probabilistic "superposition state" over future semantics rather than performing precise per-token prediction. This realizes a "global-before-local" cognitive hierarchy, achieving state-of-the-art performance among latent reasoning methods on 6 benchmarks with only 6 reasoning tokens (a reduction of 97%+), surpassing Monet by an average of 5.03%.

## Background & Motivation

**Background**: Vision-language models (VLMs) have achieved strong visual understanding by integrating LLMs with visual encoders, and Chain-of-Thought has been introduced to enable multi-step reasoning. Concurrently, latent-space reasoning methods (Coconut, SoftCoT, Monet, etc.) attempt to reason in high-dimensional hidden states to avoid the information loss inherent in explicit tokenization.

**Limitations of Prior Work**: (1) Explicit textual reasoning suffers from an information bandwidth bottleneck, as continuous visual details are lost during discrete tokenization. (2) Existing latent reasoning methods still adopt the standard autoregressive objective, forcing each hidden state to strictly minimize prediction error for the next token, leading to "premature semantic collapse"—the model is compelled to focus on a single concrete token before grasping the global context. (3) This pointwise mapping is inconsistent with the hierarchical nature of visual perception, which proceeds from global semantics to local features.

**Key Challenge**: The strict per-token prediction objective is fundamentally misaligned with the hierarchical nature of visual reasoning—early reasoning steps should maintain openness to global semantics, only gradually narrowing to a specific answer.

**Goal**: To design a latent reasoning paradigm that allows reasoning states to encode a "superposition" of global semantics in early steps, progressively narrowing toward locally precise information as reasoning advances.

**Key Insight**: Inspired by the Global Precedence Hypothesis—human visual perception processes overall structure before local details—the paper redefines the reasoning objective from pointwise prediction to dynamic window alignment.

**Core Idea**: Replace the per-token prediction objective with a dynamic semantic window: at each step, the hidden state is not required to predict the next token but is instead aligned with a dynamic window containing all remaining reasoning steps. The window naturally shrinks as reasoning progresses, enabling a gradual transition from global exploration to local precision.

## Method

### Overall Architecture

Laser operates in two stages: (1) a latent visual reasoning stage, in which the model generates a sequence of high-dimensional hidden states as intermediate reasoning paths, aligned with dynamic semantic windows via DWAL; and (2) an explicit answer generation stage, in which the final answer is generated using standard cross-entropy loss based on the evolved visual understanding. Training data consists of cognitive scan paths (ScanPaths) synthesized by GPT-4o in a global-to-local order (270K samples).

### Key Designs

1. **Dynamic Window Alignment Learning (DWAL)**:

    - Function: Replaces the standard per-token prediction objective, allowing hidden states to encode global semantic superpositions.
    - Mechanism: For reasoning step $t$, a dynamic semantic window $W_t = \{c_k | t \leq k \leq T\}$ is defined, containing all remaining reasoning tokens from the current step to the last. The hidden state $h_t$ is not required to predict $c_{t+1}$, but is aligned with the entire $W_t$. As $t$ increases, the window naturally shrinks ($|W_t| \to 1$), realizing a progressive transition from global superposition to local precision.
    - Design Motivation: The standard autoregressive objective forces early hidden states to prematurely collapse into a single semantic point, discarding global contextual information. The dynamic window allows early states to remain open.

2. **Self-Refined Superposition**:

    - Function: Constructs a stable supervisory target for the dynamic window without relying on external soft labels.
    - Mechanism: The logits corresponding to tokens within window $W_t$ are extracted, and a reference superposition distribution $Q_t$ is constructed via stop-gradient and temperature-scaled Softmax. The model's own estimates of future semantics serve as soft targets, avoiding unstable self-reinforcing loops.
    - Design Motivation: Pure soft targets may cause optimization to diverge toward a high-entropy uniform distribution, necessitating a stable self-supervised mechanism.

3. **Entropy-Regularized Intervention**:

    - Function: Injects hard-label guidance when model uncertainty is high, preventing semantic drift.
    - Mechanism: The normalized entropy $H(Q_t)$ of the reference distribution is computed. When $H(Q_t) > \eta$ (high uncertainty), hard labels and the soft distribution are mixed: $P^{target}_t = \alpha \cdot \mathbf{y}_{hard} + (1-\alpha) \cdot Q_t$; otherwise, $Q_t$ is used directly. This forms an implicit curriculum—enforcing precise alignment under high uncertainty while permitting superposition exploration under low uncertainty.
    - Design Motivation: An entirely unconstrained latent space may diverge into a meaningless high-entropy distribution, requiring hard corrections at critical moments.

### Loss & Training

The total loss is $\mathcal{L}_{Total} = \mathcal{L}_{DWAL} + \mathcal{L}_{CE}$, where the DWAL loss aligns hidden states with the mixed target over the reasoning chain, and the CE loss is applied during the answer generation stage. The backbone model is Qwen2.5-VL-7B-Instruct; the visual tower is frozen and only LLM parameters are optimized. $\eta=0.6$, $\alpha=0.8$.

## Key Experimental Results

### Main Results

| Method | Type | MMVP | BLINK | SEED2+ | MMStar | Hallusion | HRBench | Overall |
|--------|------|------|-------|--------|--------|-----------|---------|---------|
| Qwen2.5-VL-7B | Zero-shot | 65.67 | 53.60 | 65.31 | 59.70 | 56.57 | 68.25 | 61.52 |
| Vision-R1 | RL | 72.67 | 52.71 | 68.95 | 62.67 | 63.83 | 75.12 | 65.99 |
| VL-Rethinker | RL | 72.67 | 55.55 | 70.27 | 63.20 | 71.08 | 63.50 | 66.05 |
| Monet | Latent | 68.00 | 50.71 | 65.88 | 60.33 | 56.36 | 68.00 | 61.55 |
| LVR | Latent | 64.00 | 53.60 | 47.39 | 57.93 | 65.19 | 53.62 | 56.96 |
| **Laser** | Latent | **72.00** | **56.92** | **70.05** | **60.27** | **67.72** | **72.50** | **66.58** |

### Ablation Study

**Efficiency Comparison (Average Reasoning Token Count)**

| Method | BLINK Avg. Tokens | HRBench Avg. Tokens | Reduction |
|--------|------------------|---------------------|-----------|
| Qwen2.5-VL-7B | 223.5 | 55.9 | — |
| VL-Rethinker | 207.0 | 143.8 | +157.2% (HRBench) |
| Monet | 118.3 | 86.8 | — |
| LVR | 8.0 | 8.0 | -96.4% |
| **Laser** | **6.0** | **5.7** | **-97.3%** |

### Key Findings

- Laser surpasses all latent reasoning baselines by an average of 5.03%, even outperforming the computationally intensive RL methods Vision-R1 and VL-Rethinker.
- Only 6 reasoning tokens are required (a 97.3% reduction), while performance improves rather than degrades—demonstrating that latent superposition states can encode rich semantics in an extremely compact space.
- Ablation studies show that removing DWAL (reverting to per-token prediction) primarily harms fine-grained perception, while removing the dynamic window (using a fixed window) primarily harms complex reasoning.
- Significant improvements are also observed on out-of-domain tasks (Web +8.03%, Chart +5.18%) with no catastrophic forgetting.
- Latent trajectories can be decoded into interpretable top-k tokens via the LM head, revealing a multi-hop reasoning process of "entity localization → spatial analysis → semantic inference."

## Highlights & Insights

- The concept of "semantic superposition" is elegant—it imports the quantum mechanical intuition of superposition into visual reasoning, allowing reasoning states to maintain multiple possibilities before collapsing into an answer.
- A 97%+ reduction in token count paired with performance improvement fundamentally challenges the prevailing assumption that reasoning requires lengthy chains of thought.
- The implicit curriculum design is sophisticated—the entropy threshold automatically governs when to enforce alignment and when to permit exploration.

## Limitations & Future Work

- Performance is slightly lacking on absolute pixel-level localization tasks (e.g., Object Localization, Jigsaw)—the "global-before-local" strategy inherently favors semantic understanding over precise measurement.
- Synthetic data relies on GPT-4o and may inherit its biases.
- Validation is performed only on a 7B model; behavior on larger models remains unknown.
- The window shrinkage strategy (linear shrinkage) may not be optimal; adaptive shrinkage could be superior.

## Related Work & Insights

- **vs. Monet**: Monet reasons in latent space but still generates dense sequences (118 tokens); Laser compresses to 6 tokens via superposition states.
- **vs. LVR**: LVR enforces strict autoregressive reconstruction, leading to semantic degradation (−9.62%); Laser avoids collapse through flexible window alignment.
- **vs. Vision-R1/VL-Rethinker**: These methods improve performance via RL and long-form reasoning but incur high computational overhead; Laser's purely latent-space reasoning is substantially more efficient.
- **vs. CoT**: Explicit CoT is constrained by the information bottleneck of discrete tokenization; Laser bypasses this by reasoning in continuous space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Dynamic window alignment combined with semantic superposition is highly novel, redefining the optimization objective for latent reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks + efficiency analysis + fine-grained task analysis + out-of-domain transfer + interpretability + detailed ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are articulated elegantly, with the "Forest Before Trees" metaphor sustained throughout.
- Value: ⭐⭐⭐⭐⭐ A 97% token reduction with performance gains carries significant implications for real-time VLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](../../ICLR2026/interpretability/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[ICLR 2026\] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](../../ICLR2026/interpretability/mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)
- [\[NeurIPS 2025\] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](../../NeurIPS2025/interpretability/knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)
- [\[CVPR 2026\] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](../../CVPR2026/interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](../../NeurIPS2025/interpretability/efficient_vision-language_reasoning_via_adaptive_token_pruning.md)

</div>

<!-- RELATED:END -->
