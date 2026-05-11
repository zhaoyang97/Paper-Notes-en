---
title: >-
  [Paper Note] Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought
description: >-
  [ICLR 2026][Video Understanding][Continuous CoT] This paper theoretically analyzes the training dynamics of a two-layer Transformer trained with continuous Chain-of-Thought (Coconut) on the directed graph reachability pr…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Continuous CoT"
  - "Superposition"
  - "Training Dynamics"
  - "Transformer Theory"
  - "Graph Reachability"
date: 2026-05-08
content_hash: 3399682b680352d1
---

# Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought

**Conference**: ICLR 2026
**arXiv**: [2509.23365](https://arxiv.org/abs/2509.23365)
**Code**: None
**Area**: Video Understanding / LLM Reasoning Theory
**Keywords**: Continuous CoT, Superposition, Training Dynamics, Transformer Theory, Graph Reachability

## TL;DR

This paper theoretically analyzes the training dynamics of a two-layer Transformer trained with continuous Chain-of-Thought (Coconut) on the directed graph reachability problem, revealing how a "superposition" mechanism naturally emerges: the index-matching logit first grows and then remains bounded, thereby achieving a balance between exploration and exploitation.

## Background & Motivation

**Empirical advantages of continuous CoT**: Coconut (Hao et al., 2024) demonstrates theoretical and empirical advantages on multiple tasks by maintaining reasoning trajectories in a continuous latent space rather than a discrete token space.

**Constructive proof of the superposition mechanism**: Prior work (Zhu et al., 2025) proved that a two-layer Transformer with continuous CoT can efficiently solve graph reachability via "superposition"—i.e., the model simultaneously maintains multiple reasoning trajectories under uncertainty.

**Core gap**: The constructive proof only establishes the existence of such parameters, without explaining whether gradient-based training can naturally learn the superposition mechanism.

**Comparison with discrete CoT**: Discrete CoT can only select one path per step (requiring global planning or backtracking), whereas continuous CoT can maintain multiple paths in parallel (requiring only local search capability).

**Theoretical contribution**: This paper answers the open question of whether gradient descent naturally leads to a superposition construction.

## Method

### Overall Architecture

The theoretical analysis is divided into two training phases: (1) the thought generation phase, in which the model autoregressively extends continuous thoughts, and training teaches the model to expand the set of reachable nodes by one step; and (2) the prediction phase, in which the model uses the generated continuous thoughts to produce a final answer. The analysis targets the gradient flow dynamics of a simplified two-layer Transformer on the directed graph reachability problem.

### Key Designs

**1. Definition and Analysis of the Index-Matching Logit**

- **Function**: Defines the index-matching logit $\mu$ to quantify the strength of the model's local search capability.
- **Mechanism**: $\mu$ controls the matching strength between "currently explored nodes" and "edge source nodes" in the attention mechanism. By analyzing the gradient flow $\dot{\mu}(t) = \frac{\alpha}{n\sqrt{K}}(d_{p_{c+1}} - F(\mu(t)))$, it is proved that $\mu$ converges to a finite value under the Coconut loss.
- **Design Motivation**: If $\mu$ is too small, the model lacks local search capability (random guessing); if $\mu$ is too large, the model becomes overconfident, relying solely on local features (e.g., node in-degree) and discarding correct paths.

**2. Bounded Logit Induces Superposition Emergence (Theorem 1)**

- **Function**: Proves that attention logits are bounded under the Coconut loss, whereas logits under the Coconut-BFS loss diverge at least at a logarithmic rate.
- **Mechanism**: Under Coconut training, as long as the target node in-degree $d_\star < d_{max}$, $\mu(t) \to \mu^* < \infty$; under Coconut-BFS, $\mu(t) \to \infty$.
- **Design Motivation**: Bounded logits produce smooth probability distributions, enabling the model to assign similar weights to multiple paths under uncertainty (superposition); unbounded logits produce near-one-hot distributions, over-committing to a single path.

**3. One-Step Frontier Expansion (Theorem 2)**

- **Function**: Proves that when $\mu > 0$, the continuous thought achieves one-step expansion from $\mathcal{N}_c$ to $\mathcal{N}_{c+1}$.
- **Mechanism**: The token projection $\mathbf{U}^\top [t_{c+1}]$ of the next-step thought has positive mass only on the one-step expansion set $\mathcal{N}_{c+1}$, with coefficients $\beta_v$ composed of two terms: carryover (nodes already in the set) and one-hop expansion (newly expanded nodes).
- **Design Motivation**: Validates that the bounded positive $\mu$ obtained through training indeed enables BFS-style parallel search.

**4. Prediction Phase Analysis (Theorem 3)**

- **Function**: Proves that the model can correctly predict reachable nodes using the generated superposition continuous thoughts.
- **Mechanism**: Only the reachable candidate node $c_\star$ simultaneously has positive residual carryover and candidate lift; gradient flow drives the ratio $(\mu_A(t), \mu_R(t))$ to converge in a direction that ensures $c_\star$ obtains the highest logit.
- **Design Motivation**: Completes the full end-to-end theoretical chain—training naturally produces superposition, and superposition supports correct prediction.

### Loss & Training

- **Coconut loss** (used in practice): $\ell^{coco} = -\log \frac{\exp(\xi_{p_{c+1}})}{\sum_v \exp(\xi_v)}$, cross-entropy over the next node on a single demonstration path.
- **Coconut-BFS loss** (for comparison): $\ell^{BFS} = -\log \frac{\sum_{v \in \mathcal{N}_{c+1}} \exp(\xi_v)}{\sum_v \exp(\xi_v)}$, multi-label cross-entropy over all reachable nodes.
- Permutation-averaged dataset loss is used to ensure vertex symmetry.
- Experiments use curriculum learning: at stage $c+1$, the model first unsupervisedly generates $c$-step continuous thoughts, then trains on step $c+1$.

## Key Experimental Results

### Main Results

| Setting | Model | Test Accuracy |
|---------|-------|---------------|
| GPT-2 style, 2-layer, d=768 | Coconut training | 96.2% |
| Training strategy | Stage 1: 150 epochs, subsequent stages: 25 epochs each | 350 epochs total |
| Stage mixing probability | 0.1 (to prevent forgetting prior stages) | — |

The graph reachability dataset is a subset of ProsQA (Hao et al., 2024), augmented with random vertex permutations.

### Ablation Study

| Training Stage | Observation | Theoretical Prediction |
|----------------|-------------|------------------------|
| Stage 1 (c=1) | Logit difference steadily increases, saturating at ~60 around epoch 125 | Theorem 1: $\mu$ bounded ✓ |
| Stage 2 (c=2) | Positive $\mu$ established within very few epochs | Superposition mechanism reused ✓ |
| Stage 3–4 (c=3,4) | Generalization achieved without explicit training | Length generalization ✓ |

### Key Findings

1. **Coconut loss naturally produces bounded logits**: Superposition emerges even when training data provides only a single demonstration path—answering the open question posed by Zhu et al. (2025).
2. **Bounded logits are the key mechanism for superposition emergence**: They balance exploration (maintaining multiple possible paths) and exploitation (using local graph structure to identify relevant paths).
3. **Length generalization**: Once superposition emerges in early stages, subsequent stages can rapidly reuse it, even without training on longer sequences.
4. **Contrast with discrete CoT theory**: In discrete settings, logits typically grow logarithmically and diverge (Tian et al., 2023a; Nichani et al., 2024a); the bounded behavior in the continuous setting represents a fundamental difference.

## Highlights & Insights

- **Bridges the gap between constructive proofs and training dynamics**: Prior work only established that superposition "can exist"; this paper shows it "will naturally emerge."
- **Counterintuitive finding**: Even when training data demonstrates only a single path, the model learns to track multiple paths simultaneously—a unique advantage of the continuous latent space.
- **New perspective on exploration–exploitation**: Directly connects the boundedness of attention logits to the exploration–exploitation trade-off in reasoning, providing a new tool for understanding LLM internal reasoning mechanisms.
- **Strong alignment between theory and experiment**: The experimentally observed logit growth followed by saturation perfectly validates the theoretical predictions.

## Limitations & Future Work

1. The analysis is restricted to a simplified setting of two-layer Transformers with linear attention, leaving a gap with practical deep Transformers using softmax attention.
2. Only the directed graph reachability problem is considered; generalization to broader reasoning tasks requires additional work.
3. The copy mechanism in the first layer is assumed to be already established (citing prior work); its learning process is not analyzed.
4. The permutation symmetry assumption may not strictly hold in practical LLM training.
5. Experiments are limited in scale (2-layer Transformer, simple graph structures); validation on larger models and more complex tasks is needed.

## Related Work & Insights

- **Zhu et al. (2025)**: The direct predecessor of this paper, providing a constructive proof that continuous CoT solves graph reachability—this paper contributes the complementary training dynamics analysis.
- **Hao et al. (2024) Coconut**: Introduced the concept of continuous CoT and the curriculum learning approach—this paper explains the theoretical foundation of its success.
- **Nichani et al. (2024a)**: Analyzed the training dynamics of induction heads, but logits diverge in the discrete setting—forming a contrast with the bounded results of this paper.
- The findings have theoretical implications for latent-space reasoning approaches (pause tokens, filler tokens, planning tokens): the "exploration–exploitation balance" in continuous space may be a common mechanism underlying the success of these methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to explain the emergence of superposition in continuous CoT from a training dynamics perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments are limited in scale, serving primarily as theoretical validation; large-scale models and real-world reasoning tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and figures are intuitive, though the work demands substantial background knowledge.
- **Value**: ⭐⭐⭐⭐ Provides a solid theoretical foundation for understanding how continuous CoT works, with broad implications for the latent reasoning research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](../../AAAI2026/video_understanding/distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](../../ICCV2025/video_understanding/an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)

</div>

<!-- RELATED:END -->
