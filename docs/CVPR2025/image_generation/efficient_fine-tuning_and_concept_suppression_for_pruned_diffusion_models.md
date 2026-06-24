---
title: >-
  [Paper Note] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Diffusion model pruning] A bilevel optimization framework is proposed to unify the fine-tuning recovery (lower-level: distillation + diffusion loss minimization) and undesirable concept suppression (upper-level: guiding the model away from target concepts) of pruned diffusion models into a single-stage optimization. This addresses the cyclic dependency issue in two-stage "fine-tune then erase" methods, where the optimal fine-tuning point does not…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion model pruning"
  - "knowledge distillation"
  - "concept erasing"
  - "bilevel optimization"
  - "safe deployment"
date: 2026-05-08
content_hash: 7ca95055ccb7dc32
---

# Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.15341](https://arxiv.org/abs/2412.15341)  
**Code**: [GitHub](https://github.com/rezashkv/bilevel-pruning)  
**Area**: Image Generation / Model Compression  
**Keywords**: Diffusion model pruning, knowledge distillation, concept erasing, bilevel optimization, safe deployment

## TL;DR

A bilevel optimization framework is proposed to unify the fine-tuning recovery (lower-level: distillation + diffusion loss minimization) and undesirable concept suppression (upper-level: guiding the model away from target concepts) of pruned diffusion models into a single-stage optimization. This addresses the cyclic dependency issue in two-stage "fine-tune then erase" methods, where the optimal fine-tuning point does not equate to the optimal initialization for erasing, achieving a 27% reduction in the CSD metric for style removal.

## Background & Motivation

**Background**: Although the generation quality of diffusion models (e.g., Stable Diffusion) continues to improve, their parameters remain massive, making deployment in resource-constrained environments like mobile devices difficult. Model pruning combined with knowledge distillation is a mainstream compression scheme: parameters are first removed based on importance, and then distillation is used to recover the generative capability of the pruned model from a base model.

**Limitations of Prior Work**: While distillation accelerates convergence, it also propagates undesirable attributes of the base model (e.g., copyrighted styles, NSFW content) to the pruned model. Even if the fine-tuning dataset is completely free of such content, distillation still "teaches" these capabilities from the teacher model to the student model. A naive fix is a two-stage approach of "first distill and fine-tune, then erase concepts," but this suffers from a fundamental cyclic dependency.

**Key Challenge**: The optimal parameters $\hat{\theta}$ obtained from fine-tuning are not necessarily the optimal initialization point for concept erasing. Erasing starting from $\hat{\theta}$ yields $\theta'$, which may deviate from the optimal fine-tuning manifold, leading to degraded generation quality. Conversely, parameters optimized for erasing may also affect fine-tuning quality. The two objectives are coupled, and sequential optimization cannot achieve a joint optimum.

**Goal**: How to simultaneously recover generative capability and remove undesirable concepts during the fine-tuning process of pruned models?

**Key Insight**: Formulate the problem as a bilevel optimization: the lower-level performs distillation fine-tuning to recover generative capability, while the upper-level conducts concept erasing to remove undesirable content. The two levels achieve joint optimization through parameter sharing and gradient interaction.

**Core Idea**: Use bilevel optimization to transform pruned fine-tuning and concept erasing from "two sequential stages" into "alternating inner and outer loops." The lower-level distillation recovery provides a constraint on generation quality, while the upper-level erasing maximizes the concept removal effect under this constraint.

## Method

### Overall Architecture

Given a pruned diffusion model $\epsilon_{\theta_{pruned}}$, this method unifies fine-tuning and concept erasing into a bilevel optimization problem: $\min_{\theta} \mathcal{L}^{CU}(\theta)$, s.t. $\theta \in \arg\min \mathcal{L}^{ft}(\vartheta)$. This is transformed into a minimax problem $\min_\theta \max_\vartheta G_\lambda(\theta, \vartheta)$ via the penalty method, which is solved using a double-loop algorithm: the inner loop fixes $\theta$ to perform K-step distillation fine-tuning (equivalent to standard fine-tuning with zero extra overhead), and the outer loop fixes $\vartheta$ to update $\theta$ using the gradient of $G_\lambda$. This framework can serve as a plug-and-play module to be paired with any pruning and concept erasing methods.

### Key Designs

1. **Bilevel Optimization Formulation**:

    - Function: Resolves the cyclic dependency between fine-tuning and erasing to find a joint optimum for both objectives.
    - Mechanism: Converts the constrained optimization problem $\min \mathcal{L}^{CU}$ s.t. $\mathcal{L}^{ft}(\theta) - \inf_\vartheta \mathcal{L}^{ft}(\vartheta) \leq 0$ into a minimax problem using the penalty method and variable splitting. The core formulation is $G_\lambda(\theta, \vartheta) = \mathcal{L}^{CU}(\theta) + \lambda(\mathcal{L}^{ft}(\theta) - \mathcal{L}^{ft}(\vartheta))$, where $\lambda$ controls the strength of the fine-tuning constraint. A larger $\lambda$ keeps $\theta$ closer to the optimal fine-tuning manifold.
    - Design Motivation: Classical bilevel optimization requires second-order derivative information (Hessian), which incurs massive computational and memory overhead. Recent first-order bilevel optimization frameworks (such as the penalty method) only require gradient information, making them feasible for large-scale models like diffusion models. The inner loop $\max_\vartheta$ is equivalent to $\min_\vartheta \mathcal{L}^{ft}$, which represents standard fine-tuning—introducing zero computational overhead.

2. **Lower-level: Distillation Fine-tuning for Generative Capacity Recovery**:

    - Function: Recovers the generation quality of the pruned model at the constraint level.
    - Mechanism: The lower-level loss is formulated as $\mathcal{L}^{ft} = \mathcal{L}^{Diff} + \lambda^{OutKD}\mathcal{L}^{OutKD} + \lambda^{FeatKD}\mathcal{L}^{FeatKD}$, which includes the standard denoising loss, output distillation (matching predicted noise between student and teacher), and feature distillation (matching intermediate feature maps). K steps of gradient descent are performed on $\vartheta$ in the inner loop.
    - Design Motivation: The authors first quantified the impact of distillation on convergence speed through experiments—adding distillation leads to significantly faster FID convergence than training with pure diffusion loss, and pruned initialization outperforms random initialization (even when both use distillation). This confirms the double-edged nature of distillation: it is indispensable but also propagates undesirable attributes.

3. **Upper-level: Concept Erasing Guidance**:

    - Function: Guides the pruned model away from target concepts without damaging general generation quality.
    - Mechanism: The upper-level employs an ESD-like negative guidance strategy. Given a target concept $c$ (e.g., "Van Gogh style") and an anchor concept $c'$ (e.g., "painting"), it minimizes $\|\epsilon_\theta(x_t, t, c') - \epsilon_{\theta_{pruned}}(x_t, t, c)\|^2$, prompting the model to generate the anchor concept when cued with the target concept. The outer loop updates $\theta$ using the gradient of $G_\lambda$, which simultaneously contains information from the erasing loss and the fine-tuning constraints.
    - Design Motivation: Unlike two-stage methods, bilevel optimization ensures that the gradients in the upper-level erasing step contain information about the lower-level fine-tuning constraints (via the term $\lambda \cdot \nabla_\theta \mathcal{L}^{ft}(\theta)$). This prevents the parameters from moving too far from the optimal fine-tuning manifold, thereby avoiding the formulation-induced degradation in generation quality observed in two-stage methods after erasing.

### Loss & Training

- **Lower-level Fine-tuning Loss**: $\mathcal{L}^{ft} = \mathcal{L}^{Diff} + \lambda^{OutKD}\mathcal{L}^{OutKD} + \lambda^{FeatKD}\mathcal{L}^{FeatKD}$
- **Upper-level Erasing Loss**: $\mathcal{L}^{CU} = \mathbb{E}\|\epsilon_\theta(x_t,t,c') - \epsilon_{\theta_{pruned}}(x_t,t,c)\|^2$
- **Joint Objective**: $G_\lambda(\theta, \vartheta) = \mathcal{L}^{CU}(\theta) + \lambda(\mathcal{L}^{ft}(\theta) - \mathcal{L}^{ft}(\vartheta))$, with $\lambda = 100$

## Key Experimental Results

### Style Removal Experiments (Removing Monet, Picasso, Van Gogh)

| Method | CLIP↓ | CP Score↑ | CSD↓ | COCO FID↓ | COCO CLIP↑ |
|:--|:--|:--|:--|:--|:--|
| Stable Diffusion 2.1 | 34.44 | 44.0 | 87.91 | 15.11 | 31.60 |
| Distilled Model (No Erasing) | 34.34 | 0.0 | 100.0 | 22.19 | 29.44 |
| Distillation + ESD | 30.78 | 84.0 | 61.45 | 30.38 | 29.02 |
| Distillation + UCE | 30.48 | 82.66 | 65.09 | 26.63 | 29.28 |
| Distillation + ConceptPrune | 29.96 | 91.3 | 53.19 | 27.86 | 28.94 |
| **Bilevel (Ours)** | **26.28** | **97.6** | **39.04** | **22.24** | **29.19** |

### NSFW Content Removal (MMA + Ring-a-Bell Adversarial Prompts)

| Method | MMA Removal Rate↑ | Ring-a-Bell Removal Rate↑ | COCO FID↓ | COCO CLIP↑ |
|:--|:--|:--|:--|:--|
| Distillation + ESD | 93.70 | 77.27 | 32.47 | 28.57 |
| Distillation + ConceptPrune | 94.12 | 97.72 | 29.56 | 29.45 |
| **Bilevel (Ours, ESD)** | **91.60** | **94.32** | **26.80** | **29.94** |

### Key Findings

- **Significant Lead in Style Removal**: On CSD (a metric specifically designed to measure style similarity), the bilevel method is 27% lower than the best two-stage baseline (39.04 vs 53.19), while achieving a superior COCO FID (22.24 vs 27.86).
- **Distillation Indeed Propagates Undesirable Attributes**: The CSD of the distilled model reaches as high as 100 (higher than original SD), indicating that distillation reinforces style learning.
- **Pruning Beats Random Initialization**: Even with distillation, the convergence speed of pruned initialization remains far superior to random initialization.
- **Zero Computational Overhead Increase for the Bilevel Method**: A total of 20,000 iterations (inner + outer loops) are used, which is identical to standard fine-tuning.
- **Comparable or Slightly Better NSFW Removal**: On NSFW tasks, the bilevel method performs comparably to two-stage baselines but maintains superior generation quality (FID/CLIP).

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to systematically reveal and quantify the issue of distillation propagating undesirable attributes, offering intuitive evidence such as "CSD=100 for the distilled model."
2. **Elegant Theory, Simple Practice**: Extending from constrained optimization $\rightarrow$ penalty method $\rightarrow$ minimax $\rightarrow$ double-loop SGD, the mathematical derivation is complete but the resulting algorithm is remarkably simple (with alternating fine-tuning and erasing steps) and incurs no extra computational overhead.
3. **Plug-and-Play**: The framework is agnostic to the pruning method (compatible with SPDM, BK-SDM, APTP, etc.) and the erasing method (compatible with ESD, UCE, etc.), offering flexible combinations.
4. **Deep Analysis of Cyclic Dependency**: The analysis in Figure 3 clearly illustrates why two-stage methods fall into suboptimality—$\hat{\theta}$ is optimal in terms of fine-tuning loss but is not a good starting point in the erasing direction.

## Limitations & Future Work

1. **Selection of Hyperparameters $\lambda$ and K**: $\lambda=100$ and K=20 are experimentally selected values and may require tuning under different concepts or pruning ratios.
2. **Incomplete Erasing**: Qualitative results show that some "residual leakage" still exists (e.g., generating female portraits in Van Gogh style when prompted).
3. **Experiments Limited to SD 2.1**: The method has not been validated on larger diffusion models (e.g., SDXL, SD3).
4. **Continuous Erasing Scenarios Not Deeply Explored**: The paper acknowledges that sequential erasing scenarios (erasing $c_1$ first, then $c_2$) require further investigation.
5. **Erasing Robustness**: The removal rate on adversarial prompts (MMA) (91.60%) is slightly lower than ConceptPrune (94.12%).

## Related Work & Insights

- **APTP (arXiv 2024)**: A dynamic prompt-aware pruning method, which serves as the primary pruning baseline for this work.
- **ESD (ICCV 2023)**: A negative-guidance-based concept erasing method, adopted in the upper-level optimization of this work.
- **BK-SDM (WACV 2022)**: A structural pruning method that removes redundant blocks in U-Net.
- **UCE (WACV 2024)**: A concept editing method achieved by modifying token embeddings in attention layers.
- **Insight**: Although distillation is a powerful tool for model compression, its "faithful transmission" characteristic implies the propagation of undesirable attributes. Any compression pipeline involving distillation should consider the associated safety risks—an insight that can be generalized to other fields such as LLM distillation.

## Rating

⭐⭐⭐⭐ — The problem definition is novel and practical, with an elegant theoretical derivation coupled with a simple implementation and a highly flexible, composable framework. However, the advantage in NSFW removal is less pronounced compared to style removal, and the experiments are limited to SD 2.1.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](../../ECCV2024/image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)

</div>

<!-- RELATED:END -->
