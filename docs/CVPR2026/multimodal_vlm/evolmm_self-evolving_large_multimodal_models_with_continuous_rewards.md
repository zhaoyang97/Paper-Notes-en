---
title: >-
  [Paper Note] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
description: >-
  [CVPR2026][Multimodal VLM][Self-evolving learning] This paper proposes EvoLMM, a fully unsupervised self-evolving framework that instantiates two roles from a single backbone LMM: a Proposer (generating visual questions) and a Solver (producing multiple answers). By replacing discrete majority voting with continuous self-consistency rewards, the model improves multimodal mathematical reasoning using only raw images (ChartQA +2.7%, MathVista +2.1%).
tags:
  - CVPR2026
  - Multimodal VLM
  - Self-evolving learning
  - large multimodal models
  - continuous rewards
  - Proposer-Solver
  - unsupervised reinforcement learning
  - self-consistency
date: 2026-05-08
content_hash: f4dbf4eca90d3720
---

# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**Conference**: CVPR2026
**arXiv**: [2511.16672](https://arxiv.org/abs/2511.16672)
**Code**: [mbzuai-oryx/EvoLMM](https://github.com/mbzuai-oryx/EvoLMM)
**Area**: Multimodal VLM
**Keywords**: Self-evolving learning, large multimodal models, continuous rewards, Proposer-Solver, unsupervised reinforcement learning, self-consistency

## TL;DR

This paper proposes EvoLMM, a fully unsupervised self-evolving framework that instantiates two roles from a single backbone LMM: a Proposer (generating visual questions) and a Solver (producing multiple answers). By replacing discrete majority voting with continuous self-consistency rewards, the model improves multimodal mathematical reasoning using only raw images (ChartQA +2.7%, MathVista +2.1%).

## Background & Motivation

1. **Annotation dependency bottleneck**: Existing LMM training relies heavily on human-annotated data (QA pairs, metadata), limiting scalability and domain generalization.
2. **Dependence on external reward models**: Many methods require additional reward models or human evaluation to provide learning signals, increasing system complexity.
3. **Instability of discrete rewards**: Prior self-evolving LLM work (e.g., SQLM) uses majority-voting discrete rewards, which frequently produce zero reward in multimodal settings due to early high inconsistency among Solver outputs, leading to unstable optimization.
4. **Poor transferability of language-domain methods**: Existing self-evolving research (e.g., Multi-Agent Evolve) is confined to purely textual domains; multimodal scenarios require visual grounding, making self-evaluation more challenging.
5. **Residual semi-supervision**: Existing multimodal self-improvement methods (Vision-SR1, ViPER, Vision-Zero) still rely on reconstruction objectives, SFT warm-starts, or implicit supervision from external generative models (GPT-4o/Gemini).
6. **Core problem**: Can an LMM autonomously improve its reasoning ability without any human annotations or external rewards?

## Method

### Overall Architecture

EvoLMM instantiates two collaborative roles from the same pretrained backbone:

- **Proposer $\pi_\phi(q|x)$**: Given a raw image $x$, generates visually grounded mathematical questions $q$.
- **Solver $\pi_\theta(y|x,q)$**: Samples $N=5$ independent answers $y_{1:N}$ for question $q$.

Both roles form a closed-loop training pipeline via internal consistency signals, without using any ground-truth labels.

### Continuous Self-Consistency Solver Reward

$$r_i^{\text{sol}} = \big(p(y_i|x,q)\big)^\gamma \cdot \Big(1 - \lambda_{\text{len}} \cdot \max\{0, (w_i - \tau)/\tau\}\Big)$$

- $p(y_i|x,q)$: consistency score of $y_i$ under the empirical answer distribution.
- $\gamma=0.7$: reward softening exponent that amplifies differences at intermediate confidence levels.
- Length penalty: encourages concise answers (target length $\tau=6$ words).
- **Key advantage**: Unlike discrete majority voting, even partial agreement (e.g., 2 out of 5) yields non-zero gradients.

### Entropy-Guided Continuous Proposer Reward

$$r^{\text{prop}} = \exp\!\left(-\frac{(H(x,q) - \mu_H)^2}{2\sigma_H^2}\right)$$

- $H(x,q)$: entropy of the Solver's answer distribution, measuring question difficulty.
- Bandpass filter design: $\mu_H=0.90$, $\sigma_H=0.35$.
- When $H \approx 0$ (trivially easy questions) or $H$ is excessively large (unsolvable questions), the reward is low.
- Adaptive curriculum: as the Solver improves, the Proposer is incentivized to generate slightly harder but still solvable questions.

### Training Optimization

- **Policy gradient**: REINFORCE with EMA baseline $b_A$ for variance reduction.
- **KL regularization**: Token-level KL divergence constraint to prevent excessive deviation from the pretrained distribution.
- **Dynamic KL controller**: $\beta_A \leftarrow \text{clip}(\beta_A \cdot \exp(\eta \cdot (\overline{KL}_A - \tau_A)/\tau_A), \beta_{\min}, \beta_{\max})$
- **Update frequency**: Solver updates every step; Proposer updates every 5 steps.
- **Parameter efficiency**: Dual LoRA adapters with a frozen backbone.
- **Training configuration**: 6,000 steps, batch size 1, AdamW, lr=1e-6, 8× AMD MI250X GPUs.

## Key Experimental Results

### Main Results: Continuous vs. Discrete Rewards (Qwen2.5-VL-7B)

| Method | ChartQA | MathVista | MathVision | MathVerse | ScienceQA | AI2D | MMMU |
|--------|---------|-----------|------------|-----------|-----------|------|------|
| Baseline | 84.00 | 68.46 | 23.91 | 43.78 | 88.30 | 82.61 | 51.11 |
| + Discrete Reward | 84.62 | 68.88 | 22.52 | 42.10 | 87.98 | 82.18 | 50.84 |
| + Continuous Reward (Ours) | **86.70** | **70.52** | **24.81** | **44.88** | **89.50** | **83.41** | **52.01** |
| Δ | +2.7% | +2.06% | +0.9% | +1.1% | +1.2% | +0.8% | +0.9% |

Discrete rewards even cause performance degradation on MathVision/MathVerse, demonstrating the fragility of majority voting in multimodal reasoning.

### Cross-Backbone Generalization (Table 3)

| Model | ChartQA Δ | MathVista Δ | ScienceQA Δ |
|-------|-----------|-------------|-------------|
| Qwen2.5-VL-7B | +2.70 | +2.06 | +1.20 |
| InternVL3-8B | +2.57 | +2.00 | +0.36 |
| Gemma3-12B-It | +2.97 | +2.00 | +1.08 |
| Llama-3.2-11B | +3.00 | +2.00 | +1.20 |

Consistent gains of +1–3% across all backbones demonstrate architecture-agnostic applicability.

### Ablation Study: Fine-tuning Strategies (Table 2)

| Strategy | ChartQA | MathVista | MathVision |
|----------|---------|-----------|------------|
| LoRA (default) | **86.70** | **70.52** | **24.81** |
| QLoRA | 85.32 | 68.92 | 23.97 |
| Full Fine-tune | 84.20 | 68.41 | 23.37 |

- Full fine-tuning leads to degradation, as full-parameter updates are prone to overfitting and conflict with KL regularization in the unsupervised setting.
- QLoRA suffers from quantization noise that reduces Solver consistency.

### Model Scale Scaling (Table 4)

On Qwen2.5-VL-72B, ChartQA improves from 88.20 to 91.04 (+2.84) and MathVista from 73.93 to 76.44 (+2.51), with larger models yielding greater absolute gains.

## Highlights & Insights

1. **Fully unsupervised**: Only ~6k raw images (no QA pairs or metadata) are needed to stably improve reasoning ability.
2. **Elegant continuous reward design**: Soft consistency-probability-based rewards on the Solver side combined with entropy-based bandpass rewards on the Proposer side effectively avoid gradient vanishing from discrete rewards.
3. **Emergent curriculum learning**: The Proposer automatically transitions from simple to moderately difficult questions without manual curriculum design.
4. **Lightweight and reproducible**: Dual LoRA with a frozen backbone converges in 6,000 training steps.
5. **Strong generalization**: Effective across 4 different architectures (Qwen, InternVL, Gemma, Llama) and 2 model scales (7B/72B).

## Limitations & Future Work

1. **Limited improvement magnitude**: Gains are at most ~+3%, with diminishing returns on near-saturated benchmarks (e.g., ScienceQA 88→89%).
2. **Restricted to mathematical reasoning**: The method has not been extended to broader tasks (OCR, VQA, visual dialogue), leaving generality unclear.
3. **Small training data scale**: Only ~6k images are used; it remains unexplored whether larger-scale data would yield greater gains.
4. **Single-turn question answering**: The Proposer generates a single question per image; multi-turn interaction or chained questioning is not considered.
5. **Fixed Solver sampling count**: $N=5$ is used as a fixed hyperparameter; the impact of different values of $N$ is not thoroughly explored.
6. **No direct comparison with RLHF/RLVR**: A fair comparison against supervised reinforcement learning methods is absent.

## Related Work & Insights

| Method | Supervision Type | Reward Form | External Model | Multimodal |
|--------|-----------------|-------------|----------------|------------|
| SQLM [5] | No annotation | Discrete majority voting | None | ❌ Text only |
| Multi-Agent Evolve [6] | No annotation | Judge role | Implicit Judge | ❌ Text only |
| Vision-SR1 [18] | SFT warm-start | Perception+reasoning decomposition | None | ✅ |
| ViPER [47] | Reconstruction objective | Image/instance reconstruction | OmniGen2/Qwen-Image | ✅ |
| Vision-Zero [36] | Synthetic image pairs | Social reasoning game | GPT-4o/Gemini | ✅ |
| **EvoLMM (Ours)** | **Fully unsupervised** | **Continuous self-consistency** | **None** | ✅ |

EvoLMM is the only multimodal self-evolving approach in this comparison that relies on no form of external supervision or auxiliary models.

## Rating

- Novelty: ⭐⭐⭐⭐ (Replacing discrete rewards with continuous rewards and the entropy-guided bandpass Proposer reward are cleverly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Cross-backbone, cross-scale, and multiple ablations, though comparisons with supervised RL are missing)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, complete mathematical derivations)
- Value: ⭐⭐⭐⭐ (Provides a clean and effective baseline for unsupervised multimodal self-evolution, though absolute gains are modest)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)

<!-- RELATED:END -->
