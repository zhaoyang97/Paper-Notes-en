---
title: >-
  [Paper Note] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning
description: >-
  [ICCV 2025][Video Understanding][Continual Learning] This paper proposes RainbowPrompt, a prompt-evolving mechanism that integrates multiple task-specific prompts into a diversity-enhanced unified prompt via attention-based transformation and task-guided alignment, achieving an average improvement of 8.23% over existing methods on image classification and video action recognition tasks.
tags:
  - ICCV 2025
  - Video Understanding
  - Continual Learning
  - Prompt Learning
  - Knowledge Integration
  - Class-Incremental Learning
  - Video Action Recognition
date: 2026-05-08
content_hash: c8190f5c20d4a3f8
---

# RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning

**Conference**: ICCV 2025
**arXiv**: [2507.22553](https://arxiv.org/abs/2507.22553)
**Code**: None
**Area**: Video Understanding
**Keywords**: Continual Learning, Prompt Learning, Knowledge Integration, Class-Incremental Learning, Video Action Recognition

## TL;DR

This paper proposes RainbowPrompt, a prompt-evolving mechanism that integrates multiple task-specific prompts into a diversity-enhanced unified prompt via attention-based transformation and task-guided alignment, achieving an average improvement of 8.23% over existing methods on image classification and video action recognition tasks.

## Background & Motivation

Prompt-based continual learning (PCL) keeps the pre-trained model frozen while fine-tuning a small number of prompt parameters, serving as an effective replay-free continual learning paradigm. The core challenge lies in how to effectively integrate task-specific knowledge within prompts.

Limitations of existing methods:

**Fixed prompt methods** (e.g., CODA-Prompt): learned prompt representations remain unchanged during new task training, failing to adapt to new task requirements, resulting in low representational diversity.

**Generative prompt methods** (e.g., ConvPrompt): prompts are generated from an entangled task-shared space, suffering from task interference and dominance effects, leading to limited diversity.

**Empirical validation** (Fig. 2): Nuclear norm analysis demonstrates insufficient prompt representational diversity in existing methods; higher diversity correlates with higher accuracy and lower forgetting rates.

Core motivation: to design a prompt-evolving mechanism that continuously evolves knowledge by transforming and aligning all accumulated base prompts (both previously learned and newly introduced), thereby enhancing representational diversity.

## Method

### Overall Architecture

When a new task $t$ arrives, the method maintains an accumulated prompt set $\boldsymbol{\mathcal{P}} = \{\boldsymbol{p}^1, ..., \boldsymbol{p}^t\}$. Through attention-based transformation and task-guided alignment, all base prompts are evolved and integrated into a unified RainbowPrompt $\boldsymbol{p}_l^{\text{rainbow}(t)}$. A learnable probabilistic gate is also introduced to adaptively determine which layers to insert prompts into.

### Key Designs

1. **Attention-based Transformation**:

    - **Task Conditioning**: Learnable task embeddings $\boldsymbol{e}^t$ are used to compute attention weights, highlighting task-relevant prompt components.
    - **Task-Level Transformation (TLT)**: The new task prompt serves as Query while the accumulated prompt set serves as Key/Value, computing a cross-task affinity matrix $\mathcal{G}$ to reweight each task's contribution to the new task.
    - **Feature-Level Transformation (FLT)**: Transposed Query/Key are employed to capture cross-feature interactions (inspired by bilinear pooling), $\hat{V} = \mathcal{F} \cdot \tilde{V}^T$, integrating cross-feature contributions at a finer granularity.

2. **Task-Guided Alignment (TGA)**: A nonlinear transformation $\text{LT}(x) = \max(0, xW_l^1)W_l^2$ refines the transformed representations to adapt to new task characteristics while preserving the intrinsic properties of each prompt. The final RainbowPrompt is obtained via mean aggregation: $\boldsymbol{p}_l^{\text{rainbow}(t)} = \frac{1}{t}\sum_{i=1}^t \tilde{\boldsymbol{\mathcal{P}}}_l[i]$

3. **Adaptive Prompting**: Task-specific learnable probabilistic gates $\boldsymbol{g}_l^t$ (Bernoulli variables) are introduced, optimized differentiably via Gumbel-Softmax relaxation, adaptively determining which ViT layers to insert the RainbowPrompt into, avoiding suboptimal manual layer selection.

### Loss & Training

Total loss: $\min_{\Theta^t} \sum_i \text{CE}(\boldsymbol{z}_i, y_i) + \lambda_s \mathcal{L}_{\text{sparse}} + \lambda_m \mathcal{L}_{\text{match}}$

- $\text{CE}$: Cross-entropy classification loss
- $\mathcal{L}_{\text{sparse}} = \sum_l \log \alpha_l^t$: Sparsity regularization encouraging compact prompt insertion patterns
- $\mathcal{L}_{\text{match}} = \gamma(q(\boldsymbol{x}), \boldsymbol{e}^t)$: Task embedding matching loss
- $\lambda_s = \lambda_m = 0.01$
- Learnable parameters $\Theta^t = \{\boldsymbol{p}^t, \boldsymbol{e}^t, \boldsymbol{G}^t, W^{\text{evolution}}, \phi\}$
- Prompt evolution is performed only during training; the stored RainbowPrompt is used directly at inference

## Key Experimental Results

### Main Results

Image Classification (ImageNet-R / CIFAR-100):

| Method | ImgNet-R A₁₀↑ | ImgNet-R F₁₀↓ | ImgNet-R A₂₀↑ | CIFAR-100 A₁₀↑ | CIFAR-100 A₂₀↑ |
|------|---------------|---------------|---------------|----------------|----------------|
| L2P | 63.49 | 6.85 | 59.38 | 82.76 | 77.95 |
| DualPrompt | 68.50 | 5.14 | 63.21 | 85.07 | 80.49 |
| CODA-Prompt | 74.24 | 4.92 | 70.86 | 87.00 | 82.15 |
| ConvPrompt | 77.86 | 4.33 | 75.10 | 88.87 | 87.37 |
| **RainbowPrompt** | **79.09** | **3.90** | **78.36** | **89.86** | **90.15** |

Video Action Recognition (UCF-101 / ActivityNet):

| Method | UCF A₁₀↑ | UCF A₂₀↑ | ActivityNet A₁₀↑ | ActivityNet A₂₀↑ |
|------|----------|----------|------------------|------------------|
| CODA-Prompt | 84.77 | 75.35 | 66.13 | 58.62 |
| CPrompt | 87.16 | 81.78 | 66.81 | 62.17 |
| ConvPrompt | 85.58 | 78.83 | 67.32 | 60.01 |
| **RainbowPrompt** | **89.03** | **84.05** | **69.87** | **70.55** |

### Ablation Study

Component ablation (ImageNet-R 10-task):

| Configuration | A₁₀↑ | F₁₀↓ | Note |
|------|------|------|------|
| Full model | 79.09 | 3.90 | - |
| w/o Task Conditioning (TC) | 78.92 | 4.19 | Provides complementary information |
| w/o Task-Level Transformation (TLT) | 78.70 | 4.14 | Captures inter-task dependencies |
| w/o Feature-Level Transformation (FLT) | 78.57 | 4.29 | Fine-grained feature interaction is more critical |
| w/o Task-Guided Alignment (TGA) | 66.31 | 4.84 | **Most critical component**, -12.78% |
| w/o Adaptive Prompting (AP) | 78.13 | 4.07 | Adaptive selection outperforms manual layer choice |

### Key Findings

- Task-guided alignment is the most critical component; its removal causes a 12.78% accuracy drop.
- In the CIFAR-100 20-task setting, competing methods suffer accuracy drops of 1.5–4.85%, whereas RainbowPrompt gains 0.29%.
- On ActivityNet 20-task, ConvPrompt degrades from 67.32% to 60.01% (−7.31%), while RainbowPrompt improves from 69.87% to 70.55% (+0.68%).
- At inference, 76.5% of the evolution parameters (6.2M) are discarded, requiring only 18.5B MACs.

## Highlights & Insights

- The concept of prompt "evolution" is novel: rather than simple prompt selection or fusion, prompt representations are transformed and aligned to adapt to new tasks.
- The theoretical analysis using nuclear norm as a diversity metric is convincing, and experiments confirm a positive correlation between diversity and performance.
- The performance advantage grows with the number of tasks, indicating that the evolution mechanism effectively leverages accumulated knowledge.
- The adaptive gating mechanism eliminates the need for manual prompt layer selection, with optimal layer assignments varying across tasks and datasets.

## Limitations & Future Work

- As the number of tasks increases, the accumulated prompt set grows, and the computational cost of the evolution process scales linearly.
- Task identity is unknown at test time (class-incremental setting), making the approach reliant on the accuracy of task embedding matching.
- The method is currently evaluated on ViT-B/16 only; its effectiveness on larger pre-trained models remains unverified.
- Video action recognition samples only 3 frames, limiting the model's capacity to capture complex temporal dynamics.

## Related Work & Insights

- The prompt-evolving paradigm is potentially generalizable to other parameter-efficient fine-tuning scenarios (e.g., continual learning with LoRA).
- The dual-level attention transformation (task-level + feature-level) can be transferred to other settings requiring multi-source knowledge integration.
- The adaptive layer selection mechanism offers general reference value for the layer selection problem in prompt tuning.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](../../NeurIPS2025/video_understanding/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[NeurIPS 2025\] QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code](../../NeurIPS2025/video_understanding/qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)
- [\[NeurIPS 2025\] INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning](../../NeurIPS2025/video_understanding/inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)

</div>

<!-- RELATED:END -->
