---
title: >-
  [Paper Note] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution
description: >-
  [CVPR 2026][Image Generation][Video Super-Resolution] The paper proposes DUO-VSR, a three-stage distillation framework that compresses multi-step video super-resolution models into a one-step generator through progressive guided distillation initialization, dual-stream distillation (joint optimization of DMD and RFS-GAN), and preference-guided refinement. It achieves approximately 50× acceleration while surpassing the visual quality of previous one-step VSR methods.
tags:
  - CVPR 2026
  - Image Generation
  - Video Super-Resolution
  - Diffusion Distillation
  - One-Step Generation
  - GAN
  - Distribution Matching Distillation
date: 2026-05-08
content_hash: 5148d3f12dd4bcd8
---

# DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution

**Conference**: CVPR 2026  
**arXiv**: [2603.22271](https://arxiv.org/abs/2603.22271)  
**Code**: [https://cszy98.github.io/DUO-VSR/](https://cszy98.github.io/DUO-VSR/)  
**Area**: Image Generation / Video Super-Resolution  
**Keywords**: Video Super-Resolution, Diffusion Distillation, One-Step Generation, GAN, Distribution Matching Distillation

## TL;DR
The paper proposes DUO-VSR, a three-stage distillation framework that compresses multi-step video super-resolution models into a one-step generator through progressive guided distillation initialization, dual-stream distillation (joint optimization of DMD and RFS-GAN), and preference-guided refinement. It achieves approximately 50× acceleration while surpassing the visual quality of previous one-step VSR methods.

## Background & Motivation

1.  **Background**: Diffusion-based video super-resolution (VSR) has achieved significant progress in visual quality. Methods like SeedVR and STAR utilize large-scale pre-trained priors to achieve impressive detail restoration. However, these methods typically require 15-50 iterations for denoising, resulting in inference times of hundreds of seconds, which severely hinders practical deployment.

2.  **Limitations of Prior Work**: Existing one-step VSR methods face triple challenges: (1) DOVE uses regression loss to ensure stability but sacrifices detail fidelity; (2) SeedVR2 employs adversarial post-training, but large discriminators tend to dominate optimization and introduce unnatural artifacts; (3) Direct application of Distribution Matching Distillation (DMD) to VSR faces three major issues: **training instability** (one-step student output distribution deviates from the teacher), **degraded supervision** (the frozen real score model has not seen the student's noisy output, leading to spatial shifts and artifacts), and **insufficient supervision** (the real score model itself is inferior to real HR videos, limiting the student's upper bound).

3.  **Key Challenge**: The fundamental difficulty of one-step VSR distillation lies in the "stability-quality" trade-off—trajectory-preserving distillation (e.g., progressive distillation) is stable but produces blurry outputs; distribution matching distillation (e.g., DMD) offers high quality but is unstable and limited by the teacher's upper bound; GAN methods can introduce real video supervision but suffer from unstable discriminator training.

4.  **Goal**: Design a unified framework to simultaneously address initialization instability, degraded supervision, and insufficient supervision in DMD distillation, enabling a one-step VSR generator to match or even surpass the quality of multi-step models.

5.  **Key Insight**: The authors propose jointly optimizing DMD and GAN as complementary dual-stream supervisory signals—DMD ensures stability by aligning with the teacher's distribution, while GAN breaks the teacher's quality upper bound by introducing features from real HR videos.

6.  **Core Idea**: A three-stage progressive distillation pipeline + dual-stream joint optimization of DMD and RFS-GAN + DPO preference refinement to achieve stable, high-quality one-step video super-resolution.

## Method

### Overall Architecture
DUO-VSR is a three-stage pipeline: **Stage I** (Progressive Guided Distillation Initialization): Performs CFG distillation to remove unconditional branches, then progressively halves steps from 64 to 1 to obtain a stable one-step initial model. **Stage II** (Dual-Stream Distillation): The DMD stream ensures distribution matching, while the RFS-GAN stream extracts features from real/fake score models for adversarial training, with both optimized alternately. **Stage III** (Preference-Guided Refinement): Uses the student model to generate multiple HR candidates, ranks them via a video quality assessment model to build a preference dataset, and fine-tunes the student using DPO.

The input is a low-resolution video $x^{LR}$, which is upsampled to the target resolution and encoded into the latent space $z^{LR}$. A DiT-based denoiser, conditioned on $z^{LR}$ and text embedding $c$, predicts the clean HR latent representation. The base model has approximately 1.3B parameters and uses 50-step sampling by default.

### Key Designs

1.  **Progressive Guided Distillation Initialization**:
    *   **Function**: Provides a stable one-step initialization for subsequent dual-stream distillation.
    *   **Mechanism**: Conducted in two steps. First is CFG distillation: the student matches the combined CFG output $v_{\text{cfg}} = (1+w)v_\theta(z_t, t, z^{LR}, c) - v_\theta(z_t, t, z^{LR}, \emptyset)$ to eliminate dual forward passes during inference. Second is progressive distillation: using the CFG-distilled model as the teacher, steps are compressed via $64 \to 32 \to 16 \to ... \to 1$, where the student matches the teacher's two-step prediction with one step. The teacher is updated with the latest student every 500 steps.
    *   **Design Motivation**: Initializing a one-step student directly from a multi-step teacher causes training instability (severe gradient oscillation). Progressive distillation smoothly transitions to the one-step setting by gradually shortening the denoising path.

2.  **Dual-Stream Distillation Strategy**:
    *   **Function**: Provides reliable and sufficient supervision signals to break the teacher model's quality upper bound.
    *   **Mechanism**: Two streams alternate optimization. **DMD Stream**: A frozen real score model captures the high-quality distribution, while a continuously updated fake score model tracks student distribution changes. The student is updated via KL divergence gradients from the difference between the two. **RFS-GAN Stream**: Uses real and fake score models as discriminator backbones to extract intermediate transformer features, which are concatenated and fed into an additional convolutional discriminator head to compare student output (fake) with real HR videos (real). It employs a hinge GAN objective + feature matching loss. Both streams share the diffused student output $\hat{z}_t^S$ to save computation. A stop-gradient is applied between backbone features and the discriminator head to prevent GAN gradients from interfering with the score model's distribution tracking.
    *   **Design Motivation**: DMD alone is limited by the teacher's bound and faces degraded supervision; RFS-GAN introduces adversarial signals from real HR videos, suppressing biased gradients from real score model shifts and breaking the "student cannot exceed teacher" ceiling. Utilizing features from both real and fake score models makes adversarial supervision more comprehensive and balanced.

3.  **Preference-Guided Refinement**:
    *   **Function**: Further enhances perceptual quality.
    *   **Mechanism**: The Stage II student generates multiple HR candidates for each LR video, which are ranked by video quality assessment models (e.g., DOVER) to build a $(z^{LR}, z_0^{S_w}, z_0^{S_l})$ preference pair dataset. The student is then fine-tuned with a DPO loss to bias its predicted velocity field toward high-quality samples.
    *   **Design Motivation**: While the model is already strong after dual-stream distillation, there is still room for perceptual refinement. DPO achieves implicit preference alignment using existing quality assessment signals without requiring an additional discriminator.

### Loss & Training
**Stage I**: Uses MSE loss $\mathcal{L}_{CFG}$ for CFG distillation and trajectory matching loss $\mathcal{L}_{PD}$ for progressive distillation.  
**Stage II**: Student update = $\mathcal{L}_{DMD} + 0.1 \cdot \mathcal{L}_G + 0.05 \cdot \mathcal{L}_{FM}$; auxiliary updates use $\mathcal{L}_{Diff}$ for the fake score model and $\mathcal{L}_D$ for the discriminator head. One student update is performed every 3 auxiliary updates.  
**Stage III**: DPO loss $\mathcal{L}_{DPO}$, fine-tuned for 1000 steps on 2000 preference pairs.

## Key Experimental Results

### Main Results (Multiple Datasets, No-Reference Perceptual Metrics)

| Method | Steps | Time(s) | NIQE↓ | MUSIQ↑ | CLIP-IQA↑ | DOVER↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| STAR | 15 | 200.4 | 5.17 | 59.08 | 0.4068 | 69.29 |
| SeedVR2-7B | 1 | 89.7 | 4.63 | 55.45 | 0.3387 | 59.56 |
| DOVE | 1 | 66.7 | 4.43 | 51.25 | 0.3209 | 69.36 |
| DLoRAL | 1 | 76.6 | 4.91 | 58.44 | 0.4346 | 73.60 |
| **Ours** | **1** | **11.3** | **4.08** | **59.24** | **0.3925** | **69.71** |

*(Taking the YouHQ40 dataset as an example, DUO-VSR reaches a DOVER of 87.28 on UDM10, leading across the board.)*

### Ablation Study (AIGC60 Dataset)

| Configuration | NIQE↓ | MUSIQ↑ | CLIPIQA↑ | DOVER↑ |
| :--- | :--- | :--- | :--- | :--- |
| Base (50 steps) | 4.31 | 63.46 | 0.4712 | 87.98 |
| Stage I only | 5.45 | 58.97 | 0.408 | 86.49 |
| Stage I + II | 4.64 | 63.36 | 0.487 | 88.01 |
| Stage I + III | 5.11 | 60.22 | 0.423 | 87.63 |
| **Stage I + II + III** | **4.42** | **63.68** | **0.489** | **88.15** |

### Dual-Stream Strategy Ablation

| Setting | NIQE↓ | MUSIQ↑ | CLIPIQA↑ | DOVER↑ |
| :--- | :--- | :--- | :--- | :--- |
| DMD only | 4.99 | 61.46 | 0.432 | 87.38 |
| RFS-GAN only | 5.32 | 62.64 | 0.427 | 87.53 |
| Sequential DMD→GAN | 5.17 | 62.76 | 0.419 | 87.67 |
| **Dual-Stream (Joint)** | **4.42** | **63.68** | **0.489** | **88.15** |

### Key Findings
*   **Stage II (Dual-Stream Distillation) is key**: Moving from Stage I to Stage I+II, CLIPIQA improved from 0.408 to 0.487 and DOVER from 86.49 to 88.01, even surpassing the 50-step baseline (87.98), proving that real video adversarial supervision can break the teacher's upper bound.
*   **Joint optimization significantly outperforms sequential optimization**: Compared to Sequential DMD→GAN, Joint optimization improved CLIPIQA by 0.070 and DOVER by 0.48. The two objectives interact and enhance each other dynamically during training.
*   **Impressive Efficiency**: With only 1.3B parameters, DUO-VSR processes 21 frames of 1920×1080 video in 11.3s in one step, roughly 8× faster than SeedVR2-7B (89.7s) and 85× faster than the multi-step MGLD (956.7s).
*   **Complementary Role of RFS-GAN**: While RFS-GAN alone is less effective for texture enhancement than DMD (e.g., in plant regions), it effectively suppresses artifacts and temporal inconsistencies caused by DMD's degraded supervision (e.g., in tile regions and temporal profiles).

## Highlights & Insights
*   The **Dual-Stream Joint Optimization** design is highly clever—DMD ensures a stable baseline for distribution alignment, while GAN introduces high-quality real-world signals to break the ceiling. The shared diffusion samples ensure efficient synergy. The careful use of stop-gradients ensures the two objectives do not interfere. This "stable stream + aggressive stream" paradigm could be transferred to other distillation tasks.
*   The **diagnostic analysis of DMD's three problems in VSR** (instability, degraded supervision, insufficient supervision) is robust. The visualization of spatial shifts and artifacts in the real score model in Fig. 2 intuitively demonstrates why VSR is more prone to degraded supervision than unconditional generation (due to the strong spatial anchors provided by LR inputs).
*   **DPO Preference Refinement** serves as the "icing on the cake" for the third stage. It requires no extra discriminator and achieves quality gains at low cost through candidate generation and ranking, representing an efficient means of preference alignment.

## Limitations & Future Work
*   The training pipeline is complex (three stages, multiple score models), potentially leading to high total training costs and requiring careful tuning of hyperparameters (e.g., loss weights and update frequency ratios).
*   Current training and evaluation rely heavily on synthetic degradation (RealBasicVSR pipeline); although validated, generalization to complex real-world degradations remains limited.
*   While 1.3B parameters is much smaller than SeedVR2-7B, it is still large for edge device deployment. Model compression could be integrated for further reduction.
*   The quality ranking in the refinement stage depends on specific video assessment models; different standards may lead to different optimization directions.

## Related Work & Insights
*   **vs DOVE**: DOVE uses regression loss + two-stage training, resulting in blurry one-step outputs; DUO-VSR ensures both fidelity and perception via dual-stream distillation + DPO.
*   **vs SeedVR2**: SeedVR2 uses a large discriminator for adversarial post-training (APT), which can be unstable; DUO-VSR's RFS-GAN uses existing score model features for lightweight discrimination, with stop-gradients ensuring stability.
*   **vs DMD2**: DMD2 places GAN in a late refinement stage and uses only fake score model features; DUO-VSR optimizes jointly from the start and uses features from both real and fake score models for more comprehensive supervision.

## Rating
*   Novelty: ⭐⭐⭐⭐ The joint DMD+GAN dual-stream approach is innovative, with deep analysis of DMD's failure in VSR.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across five datasets (synthetic+real+AIGC) with full three-stage and strategy ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear logic, thorough problem analysis, and intuitive chart design.
*   Value: ⭐⭐⭐⭐ The efficiency of processing 1080p video in 11.3s for 1 step is attractive, though training complexity is a hurdle for practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)

</div>

<!-- RELATED:END -->
