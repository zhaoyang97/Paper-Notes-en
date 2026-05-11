---
title: >-
  [Paper Note] Enhancing Spatial Understanding in Image Generation via Reward Modeling
description: >-
  [CVPR 2026][Image Generation][Spatial Understanding] The authors construct the SpatialReward-Dataset, an 80K adversarial preference dataset, to train SpatialScore—a reward model specifically for evaluating spatial relationship accuracy (outperforming GPT-5). By integrating a top-k filtering strategy with GRPO online RL, they significantly enhance the spatial generation capabilities of FLUX.1-dev.
tags:
  - CVPR 2026
  - Image Generation
  - Spatial Understanding
  - Reward Model
  - GRPO
  - Diffusion Models
  - FLUX
date: 2026-05-08
content_hash: f6dd507e0bd2111f
---

# Enhancing Spatial Understanding in Image Generation via Reward Modeling

**Conference**: CVPR 2026  
**arXiv**: [2602.24233](https://arxiv.org/abs/2602.24233)  
**Code**: None  
**Area**: Text-to-Image Generation / Reinforcement Learning  
**Keywords**: Spatial Understanding, Reward Model, GRPO, Diffusion Models, FLUX

## TL;DR

The authors construct the SpatialReward-Dataset, an 80K adversarial preference dataset, to train SpatialScore—a reward model specifically for evaluating spatial relationship accuracy (outperforming GPT-5). By integrating a top-k filtering strategy with GRPO online RL, they significantly enhance the spatial generation capabilities of FLUX.1-dev.

## Background & Motivation

Despite significant progress in visual quality for text-to-image generation, accurately depicting complex spatial relationships remains difficult, particularly in long-prompt scenarios involving multiple objects. Enhancing spatial understanding via Reinforcement Learning (RL) is a natural direction, but the core bottleneck is the **lack of reliable reward models**:

**Human Preference Reward Models** (HPSv2, PickScore, etc.): Focus on overall aesthetics and text-image alignment; unable to accurately evaluate complex spatial relations.

**VQA Alignment Models** (VQAScore, etc.): Similarly perform poorly on multi-object spatial reasoning.

**Proprietary Large VLMs** (GPT-5, Gemini): High cost; unsuitable for frequent RL queries.

**Open-source VLMs** (Qwen2.5-VL 72B): Exhibit severe hallucinations; spatial reasoning is unreliable.

**Rule-based GenEval**: Only covers simple two-object template prompts; fails to generalize to long-prompt scenarios, and object detectors are sensitive to occlusions.

## Method

### Overall Architecture

A three-stage pipeline: (1) Construction of the SpatialReward-Dataset preference pair dataset → (2) Training the SpatialScore reward model → (3) Optimizing FLUX.1-dev via GRPO online RL using SpatialScore as the reward signal.

### Key Designs

1. **SpatialReward-Dataset (80K Adversarial Preference Pairs)**:

    - Use GPT-5 to generate initial prompts containing complex multi-object spatial relationships.
    - GPT-5 performs spatial relationship perturbations on the original prompts (e.g., left → right, swapping relative positions) while keeping other relations unchanged.
    - A "perfect image" is generated for the original prompt, and a "perturbed image" for the perturbed prompt.
    - Images are generated using strong text-to-image models such as Qwen-Image, HunyuanImage-2.1, and Seedream-4.0.
    - Samples that do not satisfy spatial constraints are filtered out via manual review to ensure high data quality.

2. **SpatialScore Reward Model**:

    - Backbone: Qwen2.5-VL-7B + LoRA fine-tuning.
    - Models the reward score as a Gaussian distribution $s \sim \mathcal{N}(\mu, \sigma^2)$ rather than a deterministic value for better robustness.
    - Inserts a special `<reward>` token at the end of the prompt; the final layer embedding is mapped to $\mu, \sigma$ via an MLP.
    - Optimization of preference loss using the Bradley-Terry model:

    $\mathcal{L}_{\text{Reward}}(\theta) = \mathbb{E}_{c, y_w, y_l}[-\log \sigma(R_\phi(H_\phi(y_w, c)) - R_\phi(H_\phi(y_l, c)))]$

3. **Top-k Filtered GRPO**: Addresses the advantage bias issue caused by prompts of varying difficulty:

    - Simple prompts generate many high-reward samples → some high-quality samples receive negative advantage.
    - Difficult prompts generally yield low rewards → also leading to advantage bias.
    - For each group of $G$ samples, rewards are ranked, and only the top-$k$ and bottom-$k$ are selected to compute advantage values for training.
    - A choice of $k=6$ (group size $G=24$) achieves the best trade-off between diversity and balance.
    - Significantly reduces NFE: from $24 \times 6$ to $12 \times 6$.

### Loss & Training

**Reward Model Training**:
- Qwen2.5-VL-7B + LoRA, learning rate $2 \times 10^{-6}$, batch size 32.
- Completed in 1 day on 8×H20 GPUs.

**RL Training**:
- Base Model: FLUX.1-dev + LoRA (rank=32).
- GRPO: Learning rate $3 \times 10^{-4}$, clip range $1 \times 10^{-4}$, KL penalty 0.01.
- Policy exploration achieved by converting deterministic ODEs to stochastic SDEs (Euler-Maruyama discretization).
- 32×H20 GPUs.

## Key Experimental Results

### Main Results

| Method | SpatialScore | DPG-Bench Overall | TIIF-short BR | TIIF-long BR | UniBench-short Lay-2D | UniBench-long Lay-2D |
|------|-------------|-------------------|---------------|--------------|----------------------|---------------------|
| FLUX.1-dev | 2.18 | 82.91 | 0.769 | 0.758 | 0.766 | 0.819 |
| Flow-GRPO* | 3.01 | 57.02 | 0.851 | 0.577 | 0.726 | 0.445 |
| **Ours** | **7.81** | **85.03** | **0.875** | **0.845** | **0.875** | **0.891** |

Internal evaluation of SpatialScore increased from 2.18 to 7.81 (+258%), and the overall DPG-Bench score approached GPT-Image-1 (85.03 vs 85.15).

### Reward Model Evaluation

| Model | Overall Accuracy |
|------|-----------------|
| PickScore | 0.509 |
| HPSv3 | 0.605 |
| Qwen2.5-VL-72B | 0.764 |
| GPT-5 | 0.890 |
| Gemini-2.5 Pro | 0.951 |
| **SpatialScore (7B)** | **0.958** |

SpatialScore with 7B parameters outperforms GPT-5 and Gemini-2.5 Pro in spatial understanding evaluation.

### Ablation Study

| Configuration | SpatialScore | DPG-bench Rel | UniBench Lay-3D(long) | NFE/step |
|------|-------------|---------------|----------------------|--------|
| w/o top-k | 7.73 | 0.919 | 0.793 | 24×6 |
| top-k (k=4) | 7.71 | 0.916 | 0.796 | 8×6 |
| **top-k (k=6)** | **7.81** | **0.932** | **0.801** | **12×6** |

### Key Findings

- Flow-GRPO trained on GenEval improves short prompts but **severely degrades on long prompts**, even losing the base model's long-text following capability.
- Scaling SpatialScore from 3B to 7B increased accuracy from 89.1% to 95.8%, demonstrating a significant scaling effect.
- Improvement in spatial understanding shows positive transfer; all five dimensions of DPG-Bench improved.

## Highlights & Insights

- **Adversarial Data Construction**: Generation of preference pairs via spatial relation perturbation precisely eliminates interference from non-spatial factors.
- **7B Model Outperforms Proprietary Models**: Small models trained for specific tasks can surpass general large-scale models in specialized domains.
- **Simple and Effective Top-k Filtering**: Resolves advantage bias in GRPO caused by uneven prompt difficulty while reducing computation by 2x.
- The technical route of transitioning from SDE/ODE to policy exploration has become relatively mature.

## Limitations & Future Work

- Focused solely on spatial relationships; does not cover other compositional generation dimensions (e.g., attribute binding, numerical accuracy).
- SpatialReward-Dataset depends on strong generative models (Qwen-Image, etc.); evaluation of weaker models might be biased.
- High computational cost for RL training (32×H20 GPUs).
- Impact of improved spatial understanding on aesthetic quality was not discussed.

## Related Work & Insights

- Key difference from Flow-GRPO: Specialized reward model vs. rule-based GenEval reward; the latter is unreliable in complex scenarios.
- The success pattern of RLHF in LLMs is being systematically migrated to image generation; this work is a representative for the spatial dimension.
- Insight: Similar specialized reward models and RL frameworks could be built for other dimensions (e.g., attribute binding, action consistency).

## Rating

- Novelty: ⭐⭐⭐⭐ First reward model specifically for spatial understanding; top-k filtering strategy is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks, detailed ablations, and comparisons with various baselines and proprietary models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation analysis and experimental presentation with rich visualizations.
- Value: ⭐⭐⭐⭐ Successful validation of the reward model + RL paradigm for improving generation quality in the spatial dimension; holds methodological significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)
- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dualconditioned_di.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)

</div>

<!-- RELATED:END -->
