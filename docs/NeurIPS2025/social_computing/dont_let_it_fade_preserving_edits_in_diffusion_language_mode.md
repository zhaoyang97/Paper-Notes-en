---
title: >-
  [Paper Note] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation
description: >-
  [Social Computing] This paper proposes Token Timestep Allocation (TTA-Diffusion), which assigns independent denoising timesteps to each token to address the *update-forgetting* problem caused by classifier guidance in diffusion language models, achieving substantial improvements in both stability and efficiency for controllable text generation.
tags:
  - Social Computing
date: 2026-05-08
content_hash: d46099bc753f3c03
---

# Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation

## Basic Information
- **arXiv**: 2510.26200
- **Conference**: NeurIPS 2025
- **Authors**: Woojin Kim, Jaeyoung Do (Seoul National University)
- **Institution**: AIDAS Laboratory, Seoul National University
- **Code**: Planned open-source (Apache 2.0)

## TL;DR
This paper proposes Token Timestep Allocation (TTA-Diffusion), which assigns independent denoising timesteps to each token to address the *update-forgetting* problem caused by classifier guidance in diffusion language models, achieving substantial improvements in both stability and efficiency for controllable text generation.

## Background & Motivation
Diffusion language models (DLMs) generate text through iterative denoising, and classifier guidance can inject external gradients to steer the generation direction (e.g., sentiment control, detoxification). However, the authors identify a core failure mode — **update-forgetting**: because all tokens share a uniform, context-agnostic noise update, semantic edits introduced by classifier guidance in earlier timesteps (e.g., changing "hate" to "love") are overwritten or undone in subsequent denoising steps. This leads to three problems:
1. **Degraded fluency**: excessive fluctuation disrupts coherence across tokens.
2. **Weakened controllability**: critical semantic edits are forgotten, preventing persistent guidance effects.
3. **Inefficiency**: more than 200 steps are required to gradually enforce control, incurring substantial computational overhead.

## Core Problem
How can classifier-guided semantic edits be preserved during inference in diffusion language models, preventing subsequent denoising steps from overwriting them?

## Method

### 1. Problem Formalization
Two key concepts are defined:
- **Diffusion Fluctuation** $R_t = \text{dist}(x_{t+1}, x_{t+1}^{in})$: the input-output deviation of a single denoising step.
- **Update-Forgetting** $F_t = \text{dist}(x_t^{guided}, x_{t+1})$: the semantic drift of a guidance effect at the next step.

Experimental validation shows that fluctuation is strongly positively correlated with perplexity ($r = 0.86$), and classifier confidence drops by more than 10% when key tokens are modified.

### 2. Token Timestep Allocation (TTA)
The core idea is to replace the uniform timestep applied to all tokens with an independent local timestep $t_i = f(i, t)$ per token. A larger timestep implies stronger noise and larger denoising updates; a smaller timestep effectively "freezes" the token.

**Fixed strategy**: a linear schedule $f_{linear}(i,t) = \lfloor \frac{i}{N-1} t \rfloor$, allowing tokens earlier in the sequence to stabilize sooner.

**Adaptive strategy**: classifier gradients are used as token importance indicators:
$$t_i^{adaptive} = \alpha_{smooth} \cdot t + (1 - \alpha_{smooth}) \cdot (1 - \hat{g}_i) \cdot t$$
where $\hat{g}_i$ is the normalized gradient magnitude. A large gradient for a token indicates it has been sufficiently guided by the classifier; a smaller timestep is then assigned to reduce subsequent noise perturbations and preserve the edit.

### 3. Progressive Step Reduction
Starting from a $T=5000$-step model, the approach progressively fine-tunes to $T \in \{1000, 200, 50\}$ using cross-entropy loss directly (no distillation required), enabling inference acceleration.

### 4. Theoretical Justification
- Excessive fluctuation lower-bounds the per-step denoising KL divergence, which in turn upper-bounds perplexity.
- The adaptive allocation of TTA is equivalent to the KKT solution that jointly minimizes the upper bound of cross-entropy and the margin-drop bound under a fixed noise budget.
- $\sigma_i^2 \propto (1 - \hat{g}_i)$ is a Pareto-optimal solution.

## Key Experimental Results

### Detoxification (RealToxicityPrompts)

| Method | Avg. Tox↓ | Max. Tox↓ | PPL↓ |
|---|---|---|---|
| DExperts | 15.1 | 32.0 | 48.0 |
| SSD-LM (T=1000) | 24.6 | 50.3 | 58.3 |
| **TTA (T=200)** | **12.2** | **26.0** | **—** |
| **TTA (T=50)** | **12.5** | **—** | **59.5** |

### Sentiment Control (PPLM prompts)

| Method | Acc↑ | PPL↓ |
|---|---|---|
| LM-Steer | 85.4 | 78.8 |
| TESS (T=1000) | 82.6 | 42.8 |
| **TTA (T=200)** | **92.1** | **23.2** |
| **TTA (T=50)** | **85.9** | **40.2** |

Accuracy improves by more than 20% over the strongest baseline, perplexity is nearly halved, and only 1/5 of the steps are required.

### Lexically Constrained Generation

| Method | Syntax Tree Acc | Mean PPL |
|---|---|---|
| Diffusion-LM | 86.0 | 248.6 |
| **TTA** | **93.1** | **111.4** |

### Generalization Across Diffusion Frameworks
- Continuous diffusion: accuracy on Diffusion-LM improves from 72.8% to 75.6%; PPL decreases from 89.3 to 77.9.
- Discrete diffusion: D-CBG validity improves from 98% to 99%; mean property score increases from 0.474 to 0.494.

## Highlights & Insights
1. **Precise problem formulation**: the formalization and experimental validation of update-forgetting are rigorous and well-grounded.
2. **Training-free inference-time method**: TTA operates purely at inference time and can be directly applied to existing DLMs.
3. **Theory–practice alignment**: the adaptive allocation rule is derived as a KKT optimal solution, ensuring theoretical rigor.
4. **Dramatic efficiency gains**: 50 steps suffice to surpass 200+-step baselines, yielding a 5–10× speedup.
5. **Strong generality**: applicable to simplex, continuous, and discrete diffusion frameworks alike.

## Limitations & Future Work
1. Validation is limited to 330M-scale models; scaling to large DLMs (e.g., LLaDA, MDLM) has not been explored.
2. The approach relies on gradient signals from an external classifier, making performance directly dependent on classifier quality.
3. Evaluation focuses primarily on single-attribute control; multi-attribute joint control is not thoroughly investigated.
4. The generative capacity of RoBERTa-large as the backbone is inherently limited.

## Related Work & Insights
- **vs. Diffusion-LM / SSD-LM**: both are diffusion-based text generation models, but TTA addresses their update-forgetting problem.
- **vs. AR-Diffusion**: AR-Diffusion assigns timesteps at training time based on position, while TTA assigns them at inference time based on semantic importance.
- **vs. MDLM / Simple Diffusion**: discrete diffusion achieves "hard" ordering via unmasking schedules; TTA provides a more flexible "soft" ordering.
- **vs. PPLM / DExperts**: autoregressive methods cannot modify already-generated tokens due to sequential dependencies; DLMs natively support modification but suffer from the forgetting problem.
- **vs. Token ordering (Kim et al., ICML 2025)**: a complementary work that theoretically analyzes the importance of token ordering; TTA provides the corresponding practical inference-time solution.

The connection to SANA-Sprint/DiCo is notable: both focus on inference efficiency, though in text versus image domains respectively. TTA's progressive step reduction parallels step compression in image diffusion. As DLMs scale to LLM size (e.g., LLaDA), update-forgetting is likely to become more severe, making TTA directly applicable. More broadly, the work represents a paradigm shift in controllable generation — from "enforcing control with more steps" to "allocating steps more intelligently with fewer" — an efficiency-first approach to control.

## Rating
- **Novelty**: ★★★★☆ — the discovery of update-forgetting and the design of TTA are distinctive.
- **Technical Depth**: ★★★★★ — complete theoretical derivation from problem formulation to KKT solution.
- **Experimental Thoroughness**: ★★★★☆ — validated across multiple tasks and frameworks; large-scale DLM validation is missing.
- **Writing Quality**: ★★★★★ — logic is clear and well-structured, progressing coherently from phenomenon to cause to solution.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?](position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[CVPR 2026\] As Language Models Scale, Low-order Linear Depth Dynamics Emerge](../../CVPR2026/social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)

<!-- RELATED:END -->
