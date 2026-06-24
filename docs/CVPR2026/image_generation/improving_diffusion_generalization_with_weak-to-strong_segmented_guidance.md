---
title: >-
  [Paper Note] Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance
description: >-
  [CVPR 2026][Image Generation][Diffusion Guidance] This paper unifies guidance methods in diffusion sampling under a "weak-to-strong (W2S)" perspective, categorizing them into "Condition-Dependent Guidance (CDG, e.g., CFG)" and "Condition-Agnostic Guidance (CAG, e.g., AG/SLG)". By characterizing their respective effective intervals through synthetic experiments, the authors propose **SGG (Segmented Guidance)**, which switches between the two guidance types based on noise level…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Guidance"
  - "Weak-to-Strong"
  - "CFG"
  - "AutoGuidance"
  - "Segmented Guidance"
date: 2026-05-08
content_hash: e9f95fe4fc1b51cd
---

# Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Improving_Diffusion_Generalization_with_Weak-to-Strong_Segmented_Guidance_CVPR_2026_paper.html)  
**Code**: https://github.com/Westlake-AGI-Lab/SGG  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Diffusion Guidance, Weak-to-Strong, CFG, AutoGuidance, Segmented Guidance

## TL;DR
This paper unifies guidance methods in diffusion sampling under a "weak-to-strong (W2S)" perspective, categorizing them into "Condition-Dependent Guidance (CDG, e.g., CFG)" and "Condition-Agnostic Guidance (CAG, e.g., AG/SLG)". By characterizing their respective effective intervals through synthetic experiments, the authors propose **SGG (Segmented Guidance)**, which switches between the two guidance types based on noise levels. This principle is further migrated into the training objective to enhance the inherent generalization capabilities of guidance-free models.

## Background & Motivation

**Background**: Diffusion and flow-matching models generate images through multi-step iterative denoising. During inference, "guidance" is almost always required to enhance generation quality and controllability. The most common method is Classifier-Free Guidance (CFG), which randomly drops conditions during training and extrapolates between unconditional and conditional predictions during inference. Recently, AutoGuidance (AG) proposed an alternative by using a "condition-aligned but weaker" inferior model to guide the primary model.

**Limitations of Prior Work**: The "application boundaries" of these guidance methods have remained vague. While CAG methods like AG can outperform CFG in ImageNet class-conditional generation, they are often less robust than CFG in large-scale text-to-image (T2I) tasks and typically serve only as a supplement to CFG. Practitioners lack a systematic way to determine the optimal guidance for new tasks.

**Key Challenge**: The essence of guidance is using a "weak signal" to extrapolate a "strong signal": $\mathbf{v}_w = \mathbf{v}_{\text{weak}} + w(\mathbf{v}_{\text{strong}} - \mathbf{v}_{\text{weak}})$. The difference between methods lies solely in how the "weak signal" is constructed—CFG constructs it by dropping conditions, while AG does so by weakening the model. The effectiveness of these two constructions varies depending on condition granularity and model convergence, with no single method being absolutely superior.

**Goal**: (1) Clarify the scenarios in which CDG and CAG are effective or ineffective; (2) Design a hybrid guidance strategy that captures the benefits of both; (3) Migrate these principles from inference-time "plugins" to "training objectives" to reduce the additional forward pass overhead of inference guidance.

**Key Insight**: The authors use a controllable recursive Gaussian mixture toy dataset to isolated failure modes by precisely adjusting class counts (condition granularity) and recursive depth (intra-class complexity). They then quantify the correction capability of both guidance types for the "optimal velocity field" across different timesteps on ImageNet.

**Core Idea**: CDG excels at "inter-class separation and discovering the correct manifold" during high-noise stages, while CAG excels at "intra-class detail refinement" during low-noise stages. Since the two govern different segments of the sampling timeline, the authors propose **SGG (Segmented Guidance)** to switch guidance types based on sampling time $\tau$: using CDG for high noise and CAG for low noise, later solidifying this strategy into the training objective.

## Method

### Overall Architecture
Rather than proposing a new network, this work establishes a unified W2S framework for applying guidance and implements it across both inference and training. The unified extrapolation formula is $\mathbf{v}_w(\mathbf{x}_t,t,\mathbf{c}) = \mathbf{v}_{\text{strong}} + (w-1)(\mathbf{v}_{\text{strong}} - \mathbf{v}_{\text{weak}})$, where the strong signal is the conditional model output $\mathbf{v}(\mathbf{x}_t,t,\mathbf{c})$. The construction of the weak signal $\tilde{\mathbf{v}}(\mathbf{x}_t,t,\tilde{\mathbf{c}})$ distinguishes the two categories: **CDG** (Condition-Dependent; modifies condition, keeps model constant: $\tilde{\mathbf{v}}=\mathbf{v}, \tilde{\mathbf{c}}=\varnothing$, representing CFG) and **CAG** (Condition-Agnostic; preserves condition, weakens model: $\tilde{\mathbf{v}}=\mathbf{v}_{\text{inferior}}, \tilde{\mathbf{c}}=\mathbf{c}$, representing AG/SLG). After quantifying their complementary effective intervals on the time axis using ImageNet, the authors propose SGG for inference and migrate the same principle into regression objectives for training. This represents an improvement to the guidance mechanism and training objective and does not involve multi-module serial pipelines.

### Key Designs

**1. CDG/CAG Dichotomy and Effective Interval Analysis**: 
The authors unify various guidance methods along the axis of weak signal construction. CDG creates weak signals by "manipulating conditions" (e.g., dropping $\mathbf{c} \to \varnothing$ in CFG), while CAG creates them by "manipulating the model" (e.g., using smaller/under-trained networks in AG or perturbing the primary model in SLG). Toy experiments with recursive Gaussian mixtures show that when classes are few and intra-class complexity is high, CDG suffers from mode-seeking (collapsing diversity), whereas CAG maintains intra-class coverage. Conversely, when classes are numerous and the model is under-fitted, CAG produces off-manifold outliers, while CDG pulls samples back to the correct manifold. On ImageNet using SiT-B/2, measuring the distance $\Delta_e = \mathbb{E}_{\mathbf{x}_t}[d(\dot{\mathbf{v}}, \mathbf{v}_w)]$ between guided and optimal velocity revealed that CDG correction is concentrated in high-noise steps, while CAG concentrates in low-noise steps.

**2. SGG Segmented Guidance**: 
SGG segments the guidance direction $\mathbf{g}$ by a time threshold $\tau$:

$$\mathbf{g}(\mathbf{x}_t, t, \mathbf{c}) = \begin{cases} \mathbf{v}(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{v}(\mathbf{x}_t, t, \varnothing) & t > \tau \;(\text{CDG}) \\ \mathbf{v}(\mathbf{x}_t, t, \mathbf{c}) - \tilde{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{c}) & t \le \tau \;(\text{CAG}) \end{cases}$$

The final velocity is $\mathbf{v}_w(\mathbf{x}_t,t,\mathbf{c}) = \mathbf{v}(\mathbf{x}_t,t,\mathbf{c}) + (w-1)\cdot\mathbf{g}(\mathbf{x}_t,t,\mathbf{c})$. This ensures high prompt alignment (via CFG at high noise) and high aesthetic quality (via SLG/CAG at low noise).

**3. W2S Training Objective Migration**: 
To reduce inference overhead, the W2S principle is incorporated into the training objective. An extrapolation term is added to the standard velocity matching target $\mathbf{u}=\epsilon-\mathbf{x}_0$: $\mathcal{L}_s = \mathbb{E}\big[\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}) - (\mathbf{u} + w\cdot\text{sg}[\mathbf{g}])\|_2^2\big]$, where $\text{sg}[\cdot]$ denotes stop-gradient for stability. For weak signals during training, the authors propose **BR (Branch)**—extracting an auxiliary output branch from intermediate layers as the weak signal. This is condition-agnostic and requires no extra guidance forward passes during training. The training version of SGG also segments by $\tau$: using CFG signals for high noise and BR signals for low noise.

## Key Experimental Results

### Main Results: Inference-time Guidance Comparison (SD3 / SD3.5)
Evaluated on SD3-Medium and SD3.5-Medium using MS-COCO-1K and LAION-1K prompts. Metrics include HPSv2.1 (prompt alignment) and Aesthetic scores.

| Method | NFE/s | HPSv2.1 ↑ | Aesthetic ↑ |
|------|-------|-----------|-------------|
| Conditional (No Guidance) | 1 | 21.204 | 4.978 |
| CFG | 2 | 29.199 | 5.279 |
| SLG | 2 | 27.295 | 5.714 |
| S2-Guidance | 3 | 29.614 | 5.342 |
| **SGG (Ours)** | 2 | **29.736** | **5.717** |

SGG achieves the highest prompt alignment and aesthetic scores simultaneously, validating the "best of both worlds" design.

### Ablation Study: Training Integration (ImageNet 256×256, SiT-B/2)
Migrating W2S into the training objective allows single-forward inference (NFE/s=1).

| Configuration | NFE/s | FID ↓ | sFID ↓ | IS ↑ |
|------|-------|-------|--------|------|
| SiT-B/2 (Baseline) | 1 | 31.22 | 6.41 | 49.59 |
| + CFG (Inference) | 2 | 6.02 | 5.47 | 183.83 |
| MG | 1 | 5.88 | 6.19 | 253.74 |
| BR | 1 | 16.02 | 5.13 | 76.21 |
| **SGG** | 1 | **4.58** | **4.95** | 264.06 |
| SGG + REPA | 1 | **3.07** | 4.88 | 242.15 |

### Key Findings
- SGG at NFE/s=1 (FID 4.58) outperforms inference-time CFG at NFE/s=2 (6.02), demonstrating that internalizing guidance into training saves cost while improving quality.
- SGG is orthogonal to representation alignment methods like REPA, further reducing FID to 3.07.
- While CDG is inapplicable in unconditional settings, CAG (AG/BR) remains effective, reducing FID from 61.27 to ~44.
- BR (Branch) is more practical than AG as it avoids maintaining an additional weak network during training.

## Highlights & Insights
- **Rationalizing Guidance Selection**: Transforms guidance choice from empirical guesswork into a localized interval problem. Isolating failure modes (mode-seeking vs. off-manifold) via toy experiments provides the most explanatory insight in the paper.
- **Zero-cost Execution**: SGG is a simple trick that switches guidance terms based on a threshold $\tau$. It requires no network changes or retraining to improve both alignment and aesthetics.
- **Guidance Internalization**: The W2S training objective with stop-gradients internalizes extrapolation capabilities into the model, which is highly significant for low-NFE deployment scenarios.

## Limitations & Future Work
- Systematic verification of training-time integration was primarily performed on ImageNet due to compute constraints; its effectiveness in large-scale T2I training remains to be fully explored.
- The segmentation threshold $\tau$ is a hyperparameter; its sensitivity across different tasks/models was not exhaustively discussed.
- Layer perturbation methods like SLG were found to cause performance drops when integrated into training, suggesting that training-time weak signal construction is still sensitive.
- Metrics focused on perceived quality (FID, HPS, Aesthetic) rather than directly quantifying diversity, which is a known weakness of CDG.

## Related Work & Insights
- **vs. CFG**: CFG is purely CDG, relying on condition-dropping throughout. It has robust alignment but weaker aesthetics/diversity; SGG compensates by using CAG at low noise.
- **vs. AutoGuidance (AG)**: AG is purely CAG. While strong on class-conditional tasks, it is unstable in large-scale T2I; SGG restricts CAG to the low-noise stages where it excels.
- **vs. MG / GFT**: These methods integrate unconditional terms into training; SGG improves upon them by introducing time-segmentation and the lightweight BR construction.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unified perspective + time segmentation + training migration; clear insights but components use existing methods.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Dual verification on SD3/SD3.5 and ImageNet, though training scale was restricted.)
- Writing Quality: ⭐⭐⭐⭐ (Logical progression from toy analysis to implementation; clear conclusions.)
- Value: ⭐⭐⭐⭐ (SGG is a practical guidance trick; W2S training is valuable for low-NFE deployment.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Weak-to-Strong Diffusion with Reflection](../../ICLR2026/image_generation/weak-to-strong_diffusion_with_reflection.md)
- [\[CVPR 2026\] Generative Modeling of Weights: Generalization or Memorization?](generative_modeling_of_weights_generalization_or_memorization.md)
- [\[ICML 2026\] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](../../ICML2026/image_generation/weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)
- [\[CVPR 2026\] Smoothing the Score Function to Enhance Generalization in Diffusion Models](smoothing_the_score_function_to_enhance_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Meta-CoT: Enhancing Granularity and Generalization in Image Editing](meta-cot_enhancing_granularity_and_generalization_in_image_editing.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](../../ICML2026/image_generation/weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)
- [\[CVPR 2026\] Smoothing the Score Function to Enhance Generalization in Diffusion Models](smoothing_the_score_function_to_enhance_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_the_score_function_for_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](understanding_accelerating_and_improving_meanflow_training.md)
- [\[CVPR 2026\] Meta-CoT: Enhancing Granularity and Generalization in Image Editing](meta-cot_enhancing_granularity_and_generalization_in_image_editing.md)

</div>

<!-- RELATED:END -->
