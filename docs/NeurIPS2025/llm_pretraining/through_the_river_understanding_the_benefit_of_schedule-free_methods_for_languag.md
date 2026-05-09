---
title: >-
  [Paper Note] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training
description: >-
  [NeurIPS 2025][LLM Pretraining][Schedule-Free optimizer] From the geometric perspective of the river-valley loss landscape, this paper analyzes why the Schedule-Free (SF) optimizer can continuously track the optimal solution during language model pre-training without requiring learning rate decay or weight averaging. It further reveals that SF implicitly performs weight averaging, and proposes an improved SF-AdamW that decouples the momentum and averaging window parameters.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Schedule-Free optimizer
  - learning rate schedule
  - river-valley loss landscape
  - weight averaging
  - large-scale pre-training
date: 2026-05-08
content_hash: 879da4ae945e45b5
---

# Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training

**Conference**: NeurIPS 2025
**arXiv**: [2507.09846](https://arxiv.org/abs/2507.09846)
**Code**: None (based on the public llm-baselines codebase)
**Area**: LLM Pre-training / Optimization
**Keywords**: Schedule-Free optimizer, learning rate schedule, river-valley loss landscape, weight averaging, large-scale pre-training

## TL;DR

From the geometric perspective of the river-valley loss landscape, this paper analyzes why the Schedule-Free (SF) optimizer can continuously track the optimal solution during language model pre-training without requiring learning rate decay or weight averaging. It further reveals that SF implicitly performs weight averaging, and proposes an improved SF-AdamW that decouples the momentum and averaging window parameters.

## Background & Motivation

**Core Problem**: As model and data scales continue to grow, traditional fixed-training-budget strategies (e.g., cosine learning rate schedules) become increasingly unsuitable. The mainstream alternatives each have their own limitations:

1. **WSD (Warmup-Stable-Decay) Schedule**: Flexible but requires manually triggering the decay phase to evaluate model quality; training progress cannot be assessed during the stable phase.
2. **Weight Averaging**: Eliminates the need for explicit decay, but requires additional memory to store a copy of model parameters (e.g., an extra 16 GB for LLaMA-8B).

**Key Question**: Does there exist a method that tracks the optimal solution continuously without requiring explicit learning rate decay or additional memory overhead?

**River-Valley Loss Landscape**: Recent research has found that the loss landscape of neural networks exhibits a "river-valley" structure—steep "hill" directions (high curvature) and flat "river" directions (low curvature). A good optimizer should advance along the river direction while remaining near the valley floor.

## Method

### Overall Architecture

The paper centers on Schedule-Free AdamW (SF-AdamW) and analyzes its optimization dynamics from both theoretical and empirical perspectives:

1. **Empirical Observation**: Demonstrates that SF-AdamW naturally tracks the river without requiring decay or averaging.
2. **Theoretical Analysis**: Reveals that SF implicitly performs weight averaging and operates at the Edge of Stability (EoS).
3. **Proposed Improvement**: Introduces a decoupled parameter $C$ to improve momentum sensitivity and large-batch performance.

### Key Designs

**1. Core Update Rule of the SF Method**

The SF method maintains three iterative sequences $(x_t, y_t, z_t)$:

$$x_t = (1-c_t) x_{t-1} + c_t z_t$$
$$y_t = (1-\beta) z_t + \beta x_t$$
$$z_{t+1} = z_t - \gamma \Delta_t$$

where $c_t = 1/t$, $\gamma$ is the learning rate, and $\beta$ is the momentum parameter. Gradients are computed at $y_t$, and $x_t$ is the output iterate.

**2. Key Finding: SF Tracks the River**

- Applying a brief AdamW decay to SF-AdamW checkpoints yields almost no additional loss reduction, indicating that SF already resides near the valley floor.
- Linear interpolation experiments show: AdamW (constant LR) exhibits a convex profile (spanning the valley walls); AdamW (decay) shows monotonic descent (from valley wall to valley floor); SF-AdamW shows a flat, gradual descent (already at the valley floor).

**3. The Central Role of the $y_t$ Iterate**

By re-deriving an equivalent form of SF, it is shown that $x_t$ is in fact a weighted average of all past $y_t$ iterates:

$$x_T = \sum_{t=1}^T \alpha_t y_t$$

This means SF implicitly performs weight averaging without requiring additional memory.

### Loss & Training

**Edge of Stability Analysis**:

The stability threshold of SF-GD on a quadratic objective is $\lambda_1(H) > \frac{2}{(1-\beta)\gamma}$, which is $(1-\beta)^{-1}$ times larger than the standard GD threshold of $2/\gamma$, explaining why SF can use larger learning rates.

**Central Flow Analysis**:

Under the EoS regime, the time-averaged trajectory of $y_t$ follows a central flow differential equation. Theoretical derivation shows that as $\beta_1$ increases, the oscillation variance $\sigma^2$ decreases—i.e., a larger momentum parameter suppresses oscillations in the hill direction, enabling $y_t$ to track the river more closely.

**Refined SF-AdamW**:

A decoupled parameter $C$ is introduced by redefining $c_t = \frac{(1-\beta)C}{t}$, such that the averaging weights $\alpha_t$ depend only on $C$, while $\beta$ independently controls momentum. This resolves the coupling issue in the original SF where $\beta$ simultaneously controls both momentum and the averaging window.

## Key Experimental Results

### Main Results (with table)

Experimental setup: 124M-parameter LLaMA-style model, SlimPajama-6B dataset, 0.5M token batch size.

| Optimizer | Requires LR Decay | Requires Weight Averaging | Extra Memory | Final Val. PPL |
|---|:---:|:---:|:---:|:---:|
| AdamW + constant LR | ✗ (suboptimal) | ✗ (suboptimal) | None | Higher |
| AdamW + cosine LR | ✓ | ✗ | None | Baseline |
| AdamW + WSD | ✓ | ✗ | None | ≈Baseline |
| AdamW + EWA | ✗ | ✓ | +16GB/8B | ≈Baseline |
| SF-AdamW | ✗ | ✗ | None | ≈Baseline |
| Refined SF-AdamW (C=200) | ✗ | ✗ | None | **Better than baseline** |

### Ablation Study (with table)

**Momentum Sensitivity Ablation**:

| $\beta_1$ | Original SF $x_t$ | Original SF $y_t$ | EWA of $y_t$ | Refined SF $x_t$ (C=50) |
|:---:|:---:|:---:|:---:|:---:|
| 0.1 | Poor (deviates from river) | Good | Better | **Recovered** |
| 0.5 | Poor (deviates from river) | Good | Better | **Recovered** |
| 0.95 | Best | Best | No further gain | Further improved (C=200) |

**Large-Batch Experiment** (2M token batch):

| Method | $\beta_1$ | Final Val. Loss |
|---|:---:|:---:|
| AdamW + cosine | 0.9 | Baseline |
| Original SF-AdamW | 0.98 | Below baseline |
| Refined SF-AdamW (C=200) | 0.98 | **Matches baseline** |

### Key Findings

1. **SF-AdamW requires neither decay nor averaging**: Unlike AdamW, SF-AdamW continuously tracks the river throughout training; decay and EWA bring no additional benefit.
2. **$y_t$ is more robust than $x_t$**: Even with suboptimal momentum parameters, $y_t$ still tracks the river, while $x_t$ deviates.
3. **Implicit weight averaging**: $x_t$ is a weighted average of $y_t$, achieving weight averaging effects without additional memory overhead.
4. **EoS behavior**: The $y_t$ iterate of SF naturally stabilizes near the Edge of Stability threshold during training.
5. **Decoupling improves performance**: Introducing parameter $C$ yields significant improvements in both momentum robustness and large-batch performance.

## Highlights & Insights

1. **Clear geometric intuition**: The river-valley landscape provides an intuitive explanation for why SF works—it naturally advances along the river without oscillating to the valley walls.
2. **Strong theory–experiment consistency**: Observations are highly consistent from 2D toy models to 124M-parameter language models.
3. **High practical value**: The SF method requires no predetermined training budget, no decay schedule, and no extra memory, making it an ideal choice for large-scale continual pre-training.
4. **Elegant and concise improvement**: Introducing a single parameter $C$ simultaneously addresses both momentum sensitivity and large-batch performance issues.

## Limitations & Future Work

1. **Limited experimental scale**: Validation is conducted only on 124M-parameter models; experiments at larger scales (e.g., 7B+) are absent.
2. **Simplified theoretical assumptions**: The central flow analysis relies on simplifying assumptions (e.g., $c_t = 1/t$ is negligible), and its verification in deep learning remains to be further established.
3. **Selection of parameter $C$**: Although the paper claims insensitivity to $C$, the optimal value still requires a sweep.
4. **Integration with other optimizers**: The potential combination of SF with newer optimizers such as AdEMAMix has not been explored.
5. **Single-architecture validation**: Evaluation is primarily conducted on LLaMA and GPT-2 style models, with no coverage of other architectures (e.g., MoE).

## Related Work & Insights

- **Defazio et al., 2024 (Schedule-Free)**: The original SF paper; this work builds upon it with an in-depth analysis of optimization dynamics.
- **Wen et al., 2025 (River-valley interpretation of WSD)**: Provides an understanding of WSD from the river-valley perspective.
- **Cohen et al., 2021/2025 (Edge of Stability / Central Flow)**: The EoS theoretical framework, which this paper extends to the SF method.
- **Zhang et al., 2025 (Exponential Weight Averaging)**: Analysis of EWA + constant LR; this paper demonstrates that SF's implicit averaging is superior.
- Implications for optimizer design: In large-scale training, **decoupling hyperparameters that serve distinct functions** is a general strategy for improving robustness.

## Rating

⭐⭐⭐⭐⭐ (5/5)

Rationale: The paper is highly rigorous in both theoretical depth and experimental validation. From intuitive geometric explanations to formal mathematical analysis (EoS stability thresholds, central flow derivation), to clear experimental comparisons, the logical chain is complete and compelling. The design of the refined SF-AdamW is elegant and well-motivated. Although the experimental scale is constrained by computational resources, the core findings have strong generalization potential and provide important theoretical guidance for optimizer selection in large-scale LLM pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)

</div>

<!-- RELATED:END -->
