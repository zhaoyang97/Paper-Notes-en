---
title: >-
  [Paper Note] DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Few-Shot Learning] This paper proposes DVLA-RL, a framework that employs Dual-level Semantic Construction (DSC) to generate complementary low-level attributes and high-level descriptio…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Few-Shot Learning"
  - "Vision-Language Alignment"
  - "Reinforcement Learning Gating"
  - "Dual-Level Semantics"
  - "Cross-Modal Fusion"
date: 2026-05-08
content_hash: eaf6a69137733c89
---

# DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning

**Conference**: ICLR 2026
**arXiv**: [2602.00795](https://arxiv.org/abs/2602.00795)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Few-Shot Learning, Vision-Language Alignment, Reinforcement Learning Gating, Dual-Level Semantics, Cross-Modal Fusion

## TL;DR

This paper proposes DVLA-RL, a framework that employs Dual-level Semantic Construction (DSC) to generate complementary low-level attributes and high-level descriptions, and uses RL-gated Attention (RLA) to dynamically balance the contributions of self-attention and cross-attention across different network layers. This achieves hierarchical vision-language alignment from low to high levels, attaining state-of-the-art performance on 9 few-shot learning benchmarks.

## Background & Motivation

Few-shot learning (FSL) aims to generalize to novel categories from only a handful of examples. Current semantic-based FSL methods leverage LLM-generated textual semantics to enhance visual representations, but suffer from two key limitations:

**Single-level semantic bottleneck**: Existing methods rely either solely on high-level descriptions (e.g., SemFew generates class-level descriptions) or solely on low-level attributes (e.g., ECER generates specific entity attributes), and thus fail to simultaneously provide fine-grained discrimination and holistic class understanding.

**Static fusion modules**: Existing methods employ fixed MLPs to fuse cross-modal information, unable to adaptively adjust the vision-language alignment strategy across different network depths—shallower layers should focus on local details, while deeper layers should emphasize global semantics.

Core innovations: (1) constructing complementary dual-level semantics (attributes + descriptions); (2) being the first to introduce RL into vision-language alignment for FSL, enabling dynamic gating of cross-modal fusion.

## Method

### Overall Architecture

DVLA-RL consists of two core modules:

1. **Dual-level Semantic Construction (DSC)**: Generates dual-level semantics comprising low-level attributes and high-level descriptions.
2. **RL-gated Attention (RLA)**: Dynamically balances cross-modal attention via an RL policy.

### Key Design 1: Dual-level Semantic Construction (DSC)

**Step 1: Visual Attribute Extraction**

Conditioned on the class name and support samples, an LLM (Qwen2.5-VL-32B) is queried: `"What are the key distinguishing attributes of the CLASS in the given image? List concise attributes"`, yielding a candidate attribute set $A^{C^i_{sup}} = \{a_1, \dots, a_s\}$.

**Step 2: Progressive Top-k Selection**

Each attribute is encoded via the CLIP text encoder, and cosine similarity $s_j = \cos(T^{(i)}, a_j)$ is computed against the current template embedding, with the initial template being `"A photo of a {CLASS}"`. At each step, the most relevant attribute is selected and the template is updated; after $k$ iterations, the most discriminative attributes are retained, suppressing hallucinations and redundant attributes from the LLM. Each selected attribute is embedded into the template `"A photo of a {CLASS}, which has {attribute}"` for low-level alignment.

**Step 3: Attribute Description Synthesis**

The selected attributes are synthesized into a fluent scientific description $D_i$ by the LLM, providing holistic semantics complementary to the local attributes. For example: "The Komondor is a … dog with massive size and uniquely corded white coat."

### Key Design 2: RL-gated Attention (RLA)

Given visual tokens $H_{\mathrm{img}}$ and textual semantics $H_{\mathrm{text}}$, RLA executes two dual attention pathways:

- **Image-guided path** (cross-attention): $\hat{H} = \mathrm{Attn}(W^q_\text{text}\bar{H}_\text{text}, W^k_\text{img}\bar{H}_\text{img}, W^v_\text{img}\bar{H}_\text{img})$
- **Text-guided path** (self-attention): $\tilde{H} = \mathrm{Attn}(W^q_\text{text}\bar{H}_\text{text}, W^k_\text{text}\bar{H}_\text{text}, W^v_\text{text}\bar{H}_\text{text})$

These are fused via stochastic gating: $H = \alpha \hat{H} + (1-\alpha) \tilde{H}$, where $\alpha \sim \pi_\theta(\cdot|s)$.

**State representation**: $s = \phi([\mathrm{GAP}(\bar{H}_\text{img}) \| \mathrm{GAP}(\bar{H}_\text{text}) \| \cos(\mathrm{GAP}(\bar{H}_\text{img}), \mathrm{GAP}(\bar{H}_\text{text}))])$

**Policy distribution**: $\pi_\theta(\alpha|s) = \mathrm{Beta}(\kappa p_\theta(s), \kappa(1 - p_\theta(s)))$, where $\kappa$ controls the trade-off between exploration and determinism.

### Loss & Training

**RL Reward**: $R_t = \lambda_\text{sim} \cdot \cos(U \cdot \mathrm{GAP}(H), \mathbf{t}^\star) + \lambda_\text{imp} \cdot (\mathrm{Acc}_t - \mathrm{Acc}_{t-1})$

- The first term encourages vision-text alignment (cosine similarity with CLIP ground-truth text embeddings).
- The second term measures intra-episode accuracy improvement.

**Policy gradient**: $\nabla_\theta \mathcal{J} = \mathbb{E}[(R_t - b_t) \nabla_\theta \log \pi_\theta(\alpha|s)] + \tau \nabla_\theta \mathsf{H}(\pi_\theta(\cdot|s))$

Entropy regularization is included to prevent premature policy collapse; an exponential moving average baseline is used to reduce variance.

**Total loss**: $\mathcal{L}_\text{total} = \mathcal{L}_\text{sup} + \lambda \mathcal{L}_\text{RL}$, where $\mathcal{L}_\text{sup}$ is cross-entropy loss based on a prototypical classifier.

**Training procedure**: Two stages—(1) large-scale pre-training for 300–800 epochs; (2) episodic meta-tuning for 100 epochs. RL hyperparameters: $\kappa=10$, $\lambda_\text{sim}=0.5$, $\lambda_\text{imp}=1.0$, $\lambda=0.1$, $\tau=0.2$.

## Key Experimental Results

### Main Results: General Few-Shot Classification

| Method | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | CIFAR-FS 1-shot |
|--------|-------------------|-------------------|---------------------|----------------|
| SemFew (CVPR'24) | 78.94 | 86.49 | 82.37 | 84.34 |
| ECER (AAAI'25) | 81.14 | - | 81.81 | 86.01 |
| CPL (TPAMI'25) | 72.82 | 87.93 | 78.05 | 78.82 |
| **DVLA-RL** | **81.69** | **88.25** | **83.02** | **87.18** |

### Main Results: Fine-Grained Few-Shot Classification

| Method | CUB 1-shot | CUB 5-shot | Dogs 1-shot | Cars 1-shot |
|--------|-----------|-----------|------------|------------|
| SUITED (AAAI'25) | 86.02 | 94.13 | 76.55 | 89.97 |
| BSFA (TCSVT'23) | 86.00 | 92.53 | 69.58 | 88.93 |
| **DVLA-RL** | **91.93** | **95.06** | **89.64** | **92.95** |

On fine-grained tasks, DVLA-RL surpasses the second-best method by 5.4%–15.3% (1-shot), demonstrating that dual-level semantics are particularly effective at capturing subtle inter-class distinctions.

### Cross-Domain Few-Shot Classification

| Method | CUB 1-shot | Places 1-shot | ChestX 1-shot |
|--------|-----------|--------------|--------------|
| MEFP (NeurIPS'24) | 51.55 | 52.06 | 22.82 |
| SVasP (AAAI'25) | 49.49 | 59.07 | 23.23 |
| **DVLA-RL** | **67.46** | **69.26** | **23.47** |

In cross-domain settings, DVLA-RL outperforms the second-best by 15.9% on CUB and 10.2% on Places, demonstrating strong domain transfer capability.

### Ablation Study

Ablation experiments validate the necessity of each component:

- Removing DSC (using only class-name templates): ~3–5% drop under 1-shot.
- Fixing $\alpha$ (removing RL gating): significant performance degradation, confirming adaptive fusion is superior to static fusion.
- Removing low-level attributes or high-level descriptions: both cause performance drops, confirming the complementarity of dual-level semantics.
- Removing Progressive Top-k: degraded attribute quality leads to lower performance.

### Key Findings

- In shallower RLA layers, the policy tends toward larger $\alpha$ (more cross-attention → focus on attribute details); in deeper layers, toward smaller $\alpha$ (more self-attention → integration of global semantics).
- The Beta distribution policy exhibits clear adaptive behavior across different episodic tasks.

## Highlights & Insights

1. **First application of RL to vision-language alignment in FSL**: The Beta distribution policy combined with the REINFORCE algorithm elegantly achieves layer-adaptive fusion.
2. **Complementary dual-level semantics**: Low-level attributes provide fine-grained discriminative cues, while high-level descriptions offer holistic class understanding; Progressive Top-k effectively suppresses LLM hallucinations.
3. Large margins on fine-grained and cross-domain tasks (5–16%) indicate the method is particularly effective at capturing subtle differences and enabling domain transfer.
4. Lightweight design: the RLA module introduces minimal additional parameters, and RL training is stable.

## Limitations & Future Work

1. Reliance on an LLM (Qwen2.5-VL-32B) for attribute generation introduces inference latency.
2. Attributes and descriptions can be precomputed, but novel classes still require LLM inference.
3. Gains are limited in extreme cross-domain scenarios such as ChestX (<1%), indicating that vision-language alignment under severe domain shift remains challenging.
4. Hyperparameters such as $\kappa$ in RL gating require validation-set tuning.

## Related Work & Insights

- Compared to SemFew, which uses only high-level descriptions, and ECER, which uses only low-level entities, the dual-level design of DSC represents a natural unification of both approaches.
- The RL gating concept is generalizable to any scenario requiring adaptive cross-modal fusion (e.g., VQA, image-text retrieval).
- The Progressive Top-k selection mechanism can be applied to other tasks that require filtering high-quality information from LLM outputs.
- A Beta distribution policy is more suitable than Bernoulli or Gaussian for continuous gating over the $[0,1]$ interval.

## Rating

- Novelty: ⭐⭐⭐⭐ (RL gating + dual-level semantics is a meaningful and novel combination)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 benchmarks, 3 scenarios, 20+ baselines)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, complete formulations)
- Value: ⭐⭐⭐⭐ (significant SOTA results, good methodological generalizability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](../../ACL2026/reinforcement_learning/healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICLR 2026\] Dual Goal Representations](dual_goal_representations.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
