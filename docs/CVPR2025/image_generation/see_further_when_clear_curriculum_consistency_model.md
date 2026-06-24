---
title: >-
  [Paper Note] See Further When Clear: Curriculum Consistency Model
description: >-
  [CVPR 2025][Image Generation][Consistency Models] This paper proposes the Curriculum Consistency Model (CCM). It reveals that the training difficulty (knowledge discrepancy) across different timesteps during consistency distillation is highly imbalanced. By dynamically adjusting the iteration steps of the teacher model using a PSNR-based KDC metric to maintain a consistent curriculum difficulty, CCM achieves a single-step FID of 1.64 on CIFAR-10 and successfully scales to SDX…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Consistency Models"
  - "Curriculum Learning"
  - "Diffusion Distillation"
  - "Few-step Generation"
  - "Flow Matching"
date: 2026-05-08
content_hash: d6f2d8f06a2f0a2d
---

# See Further When Clear: Curriculum Consistency Model

**Conference**: CVPR 2025  
**arXiv**: [2412.06295](https://arxiv.org/abs/2412.06295)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Consistency Models, Curriculum Learning, Diffusion Distillation, Few-step Generation, Flow Matching

## TL;DR

This paper proposes the Curriculum Consistency Model (CCM). It reveals that the training difficulty (knowledge discrepancy) across different timesteps during consistency distillation is highly imbalanced. By dynamically adjusting the iteration steps of the teacher model using a PSNR-based KDC metric to maintain a consistent curriculum difficulty, CCM achieves a single-step FID of 1.64 on CIFAR-10 and successfully scales to SDXL and SD3.

## Background & Motivation

- **Need for few-step generation**: Diffusion models and Flow Matching suffer from suboptimal sampling efficiency. Consistency Models (CM) greatly reduce sampling steps by enforcing self-consistency along ODE trajectories, but their training efficiency still has room for improvement.
- **Imbalanced knowledge discrepancy**: In consistency distillation, the student-teacher output discrepancy (knowledge discrepancy) faced by the student model is highly uneven across different timesteps—large discrepancy at high noise ($t \to 0$) and small discrepancy at low noise ($t \to 1$). This leads to insufficient learning in low-noise regions.
- **Limitations of prior work**: iCT and ECM address the cumulative error by gradually reducing the distillation step size. However, smaller step sizes further aggravate the diminishing knowledge discrepancy, rendering learning even more inefficient.
- **"See further when clear"**: When the input noise is low (large $t$), the model's perception is clearer but the learning signal is weaker. At this point, the teacher model should be allowed to "see further"—reaching an even further target timestep through multi-step iteration to increase the knowledge discrepancy.
- **Unified framework**: The proposed method is simultaneously applicable to diffusion models (e.g., Stable Diffusion) and Flow Matching models (e.g., SD3), filling the gap in consistency distillation research for Flow Matching.

## Method

### Overall Architecture

CCM introduces three key components to standard consistency distillation:
1. **KDC Metric**: Measures the knowledge discrepancy of the current curriculum based on PSNR.
2. **Dynamic Target Adjustment**: Adaptively determines the target timestep $u$ of the teacher model according to KDC.
3. **Multi-step Iterative Generation**: The teacher model reaches $u$ through multiple small step iterations to guarantee prediction quality.

### Key Designs

**1. Knowledge Discrepancy of the Curriculum (KDC)**
- **Function**: Quantifies the learning difficulty between student and teacher in each distillation iteration in a stable, comparable manner.
- **Mechanism**: $\text{KDC}_t^u = 100 - \text{PSNR}(\boldsymbol{x}_{\text{est}}, \boldsymbol{x}_{\text{target}})$, where $\boldsymbol{x}_{\text{est}} = f_\theta(\boldsymbol{x}_t, t, 1)$ is the student output and $\boldsymbol{x}_{\text{target}} = f_{\theta^-}(\text{Solver}(\boldsymbol{x}_t, t, u; \phi), u, 1)$ is the teacher output. Experiments on different datasets and models (CIFAR-10, ImageNet, CC3M) demonstrate a highly consistent trend for KDC curves.
- **Design Motivation**: PSNR directly measures image difference with a stable scale, and subtracting from 100 aligns larger KDC with larger discrepancies. Empirical validation confirms cross-model consistency of KDC across DM and FM models.

**2. Dynamic Curriculum Target Adjustment**
- **Function**: Ensures roughly equal learning difficulty across all timesteps, preventing the curriculum from being "too easy" or "too difficult".
- **Mechanism**: A KDC threshold $T_{\text{KDC}}$ is defined. At each training step, starting from timestep $t$, the teacher model iteratively performs a small step $s$ until the estimated KDC exceeds $T_{\text{KDC}}$. The final $u$ is the timestep that satisfies the target difficulty.
- **Design Motivation**: In the early training stages, the model is weak, and KDC at large timesteps ($t$) is naturally large and requires no adjustment. As training progresses and the model becomes stronger, a larger distillation step size $l = u - t$ is required to sustain sufficient knowledge discrepancy.

**3. Multi-step Iterative Teacher Generation**
- **Function**: Guarantees teacher prediction accuracy when the distillation step size becomes large.
- **Mechanism**: When $u$ is much larger than $t$, a direct single-step ODE solve from $t$ to $u$ is inaccurate. Thus, the teacher model starts from $t$ and takes small steps $s$ iteratively until reaching $u$, ensuring the quality of $\boldsymbol{x}_u = \text{Solver}(\boldsymbol{x}_t, t, u; \phi)$.
- **Design Motivation**: The benefit of a larger distillation step size is the reduction of accumulation errors and the enhancement of global consistency, but only under the premise that the teacher's trajectory prediction is accurate.

### Loss & Training

The loss function is the same as standard consistency distillation, but with the dynamically adjusted target:
$$\mathcal{L}_{\text{CCM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\lambda(\sigma_t) \cdot d\left(f_\theta(\boldsymbol{x}_t, t, 1), f_{\theta^-}(\boldsymbol{x}_u^{\text{KDC}}, u, 1)\right)\right]$$

where $u$ is dynamically determined by KDC, and $d(\cdot,\cdot)$ is a distance metric.

## Key Experimental Results

### Main Results: Single-step Sampling FID

| Method | CIFAR-10 FID ↓ | ImageNet 64×64 FID ↓ |
|------|:---:|:---:|
| CM (Song et al.) | 2.93 | 6.20 |
| iCT (Song & Dhariwal) | 2.83 | 3.25 |
| ECM | 1.68 | 2.58 |
| **CCM (Ours)** | **1.64** | **2.18** |

### Large-Scale T2I Model Scaling

| Base Model | Method | FID ↓ | CLIP Score ↑ |
|------|------|:---:|:---:|
| SDXL (DM) | LCM | Higher | Lower |
| SDXL (DM) | **CCM** | **Lower** | **Higher** |
| SD3 (FM) | LCM-adapt | Higher | Lower |
| SD3 (FM) | **CCM** | **Lower** | **Higher** |

### Key Findings

- KDC gradually decreases from $t=0$ (approx. 60) to $t=1$ (approx. 35), confirming the existence of imbalanced knowledge discrepancy.
- Decreasing the distillation step size $l$ further lowers the KDC, making the issues of iCT/ECM more severe.
- The adaptive $l$ of CCM gradually increases as training progresses (contrary to iCT), engaging larger steps in later stages.
- On large-scale T2I models, CCM significantly improves text-to-image alignment and the quality of semantic structures.
- CCM is the first to systematically investigate consistency distillation for Flow Matching (SD3).

## Highlights & Insights

1. **"See further when clear"**: An insightful intuition—in regions with low noise, the model's capability is strong but the learning signal is weak, so the distillation range should be expanded to maintain learning efficiency.
2. **Universality of the KDC metric**: The PSNR-based measure of knowledge discrepancy exhibits high consistency across different models (DM/FM), datasets, and resolutions.
3. **Insights compared to iCT**: iCT progressively reduces step size to minimize errors, while CCM progressively increases step size to maintain difficulty—two mutually complementary strategies.
4. **Unified DM + FM Framework**: Introduces consistency distillation simultaneously to both diffusion and Flow Matching models for the first time.

## Limitations & Future Work

- Multi-step iterative teacher generation adds to the training time complexity.
- The KDC threshold $T_{\text{KDC}}$ needs to be manually set, and different scenarios might require different values.
- Current experiments primarily validate single-step sampling; the performance of multi-step sampling (2-4 steps) warrants more comprehensive research.
- Future work could integrate adaptive step-size scheduling with more advanced ODE solvers to further improve quality.

## Related Work & Insights

- **Consistency Models (CM)**: Pioneered few-step generation through self-consistency.
- **iCT/ECM**: Solves cumulative errors by reducing the distillation step size, but exaggerates training imbalance.
- **LCM**: Extends consistency distillation to latent spaces and text-conditioned image synthesis.
- **PCM and SCott**: Improved schemes introducing segmented trajectories and noise control.
- **Insight**: Issues in training efficiency often lie not in the model architecture itself, but in the quality and balance of the training signal—curriculum learning concepts are broadly applicable to speed up learning.

## Rating

⭐⭐⭐⭐ — Deep analysis of the problem; the discovery of "imbalanced knowledge discrepancy" is insightful and backed by experimental evidence. The KDC metric is simple but effective, enabling CCM to achieve SOTA single-step FID on CIFAR-10/ImageNet. Successful scaling to SDXL/SD3 demonstrates the generality of the method. The main downside is the training overhead of multi-step teacher iteration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[NeurIPS 2025\] Riemannian Consistency Model](../../NeurIPS2025/image_generation/riemannian_consistency_model.md)
- [\[CVPR 2025\] PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models](pcm_picard_consistency_model_for_fast_parallel_sampling_of_diffusion_models.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](../../ICCV2025/image_generation/learning_to_see_in_the_extremely_dark.md)
- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](../../ICML2025/image_generation/when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)

</div>

<!-- RELATED:END -->
