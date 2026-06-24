---
title: >-
  [Paper Note] Random Conditioning for Diffusion Model Compression with Distillation
description: >-
  [CVPR 2025][Image Generation][Knowledge Distillation] This paper proposes Random Conditioning, a technique that pairs noisy images with randomly selected, unrelated text conditions during the knowledge distillation of conditional diffusion models. This allows the student model to explore the full condition space without needing to generate corresponding images for each text, achieving highly efficient image-free or image-scarce diffusion model compression while enabling the s…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Knowledge Distillation"
  - "Diffusion Model Compression"
  - "Random Conditioning"
  - "Image-Free Distillation"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 8e49a8ddb48eec98
---

# Random Conditioning for Diffusion Model Compression with Distillation

**Conference**: CVPR 2025  
**arXiv**: [2504.02011](https://arxiv.org/abs/2504.02011)  
**Code**: [https://dohyun-as.github.io/Random-Conditioning](https://dohyun-as.github.io/Random-Conditioning)  
**Area**: Diffusion Models / Model Compression  
**Keywords**: Knowledge Distillation, Diffusion Model Compression, Random Conditioning, Image-Free Distillation, Stable Diffusion

## TL;DR

This paper proposes Random Conditioning, a technique that pairs noisy images with randomly selected, unrelated text conditions during the knowledge distillation of conditional diffusion models. This allows the student model to explore the full condition space without needing to generate corresponding images for each text, achieving highly efficient image-free or image-scarce diffusion model compression while enabling the student to generate concepts unseen during training.

## Background & Motivation

**Background**: Text-to-image diffusion models such as Stable Diffusion perform exceptionally well but incur high computational costs (a massive number of sampling steps + enormous parameter size). Knowledge distillation is a mainstream approach for model compression—it has been proven in classification tasks that even "unseen concepts" of the teacher can be transferred to the student (e.g., on MNIST, even if the student is not shown images of '3', the student can still learn to recognize '3').

**Limitations of Prior Work**: (1) In conditional diffusion models, knowledge distillation cannot automatically transfer unseen concepts as classification models do—if the training data lacks images of a certain class, the student cannot generate content for that class; (2) To cover the complete text condition space, paired images must be generated for every text prompt, which incurs prohibitive computational and storage costs given the vast text space; (3) Issues such as privacy and copyright make obtaining large-scale image-text pairs increasingly difficult.

**Key Challenge**: The generation function of conditional diffusion models maps from a semantic condition space to a much larger image space, and each denoising step depends simultaneously on both the condition and the intermediate noisy image. This makes it impossible for the student to learn full generative capabilities solely through limited condition-image pairs, while generating paired images for the entire condition space is too costly.

**Goal**: To achieve data-efficient (or even image-free) knowledge distillation for conditional diffusion models, allowing the student model to generate concepts unseen during training.

**Key Insight**: The authors observe a key phenomenon: during the denoising process of diffusion models, the influence of conditional information varies across timesteps. When $t$ is large (high noise), the model primarily relies on the condition $c$ for generation, almost ignoring the original semantics of the input image; when $t$ is small, the model primarily relies on the input image for denoising, largely ignoring the condition. This implies that at large $t$, the input image and the condition do not need to match strictly.

**Core Idea**: Replace the text conditions of training samples with an unrelated text randomly selected from a larger text pool with a certain probability $p(t)$. This allows the student to explore a broader condition space during distillation without generating paired images for the extra text.

## Method

### Overall Architecture

Given a teacher model $\mathcal{T}$ (e.g., SD v1.4) and a student model $\mathcal{S}$ (a UNet with fewer parameters), the distillation pipeline is as follows: (1) Generate paired images from a small number of text prompts to construct a dataset $\mathcal{D} = \{(\mathbf{x}^n, c^n)\}_{n=1}^{N}$; (2) During training, sample a pair $(\mathbf{x}^n, c^n)$ and add noise to the image to obtain $\mathbf{x}_t$; (3) Replace the condition $c^n$ with a text $\tilde{c}$ randomly sampled from a larger text pool $\mathcal{C}$ ($M \gg N$) with probability $p(t)$; (4) Train the student using output distillation loss + feature distillation loss.

### Key Designs

1. **Random Conditioning**:

    - Function: Enables students to explore unseen text conditions without generating paired images.
    - Mechanism: During training, replace the matching condition $c^n$ with a condition $\tilde{c}$ randomly sampled from the text pool $\mathcal{C}$ with probability $p(t)$. The condition selection formula is $\hat{c} = c^n$ (with probability $1-p(t)$) or $\hat{c} = \tilde{c} \in \mathcal{C}$ (with probability $p(t)$). $p(t)$ uses an exponential function to reduce the replacement probability in middle timesteps (since middle steps are sensitive to condition-image alignment) and increase it at large $t$ and small $t$. The replaced $\hat{c}$ is paired with $\mathbf{x}_t$ to compute output distillation and feature distillation losses.
    - Design Motivation: The core observation is that conditional information dominates generation at large $t$ (where the input noise masks the original image and condition alignment is irrelevant), while the model primarily performs denoising at small $t$ (almost ignoring the condition). Therefore, mismatched condition-image pairs do not seriously degrade distillation quality at most timesteps, but instead allow the student to learn the teacher's behavior under new conditions.

2. **Timestep-Adaptive Replacement Probability $p(t)$**:

    - Function: Controls the frequency of random conditioning across different noise levels.
    - Mechanism: Designed using an exponential function to decrease the replacement probability at middle timesteps (the phase where condition-image pairing is most crucial) and increase it at both ends. Experiments show that a constant $p(t)=1$ (i.e., continuous replacement) yields suboptimal results.
    - Design Motivation: The intermediate timesteps act as a "transition zone"—the image still contains partial semantic information and the condition is active, so mismatched conditions here would produce clear artifacts. Thus, adaptive adjustment is required.

3. **Image-Free Distillation and LLM-Generated Prompts**:

    - Function: Completes distillation in extreme scenarios where neither images nor original text data are accessible.
    - Mechanism: When even text data is unavailable, an LLM (such as GPT) is used to automatically generate text prompts. The teacher model then generates images for only a small fraction of them, while the large remaining set of texts is utilized solely via random conditioning.
    - Design Motivation: Addresses model deployment needs under privacy/copyright restrictions; prompts generated by the LLM can also be customized for target domains.

### Loss & Training

Two loss functions are used with equal weight: (1) Output distillation loss $\mathcal{L}_{out}$: the L2 distance between the noise predictions of the teacher and the student; (2) Feature distillation loss $\mathcal{L}_{feat}$: the L2 distance between the output features of each UNet block (the student can match dimensions using temporary projection modules). Training is conducted using the AdamW optimizer with a learning rate of 5e-5 on 4×A100 GPUs, with a batch size of 256, and a null condition ratio of 10%.

## Key Experimental Results

### Main Results

Impact of Random Conditioning (B-Base architecture, MS-COCO 30K evaluation):

| Config | Rand Cond | Teacher Init | Real Image | FID↓ | IS↑ | CLIP↑ |
|------|-----------|-------------|------------|------|-----|-------|
| #1 | ✗ | ✗ | ✗ | 18.13 | 31.84 | 0.2728 |
| #3 (BK-SDM) | ✗ | ✓ | ✓ | 15.76 | 33.79 | 0.2878 |
| #4 | ✓ | ✗ | ✗ | 15.46 | 34.48 | 0.2834 |
| #5 | ✓ | ✓ | ✗ | 15.76 | 36.03 | 0.2895 |
| #6 | ✓ | ✓ | ✓ | **15.00** | **36.14** | **0.2933** |

Comparison with other models (MS-COCO 30K):

| Model | Parameters | Training Images | FID↓ | IS↑ | CLIP↑ |
|------|--------|-----------|------|-----|-------|
| SD-v1.4 (Teacher) | 1.04B | >2000M | 13.05 | 36.76 | 0.2958 |
| BK-SDM-Base | 0.76B | 0.22M | 15.76 | 33.79 | 0.2878 |
| B-Base (Ours) | 0.76B | 0.22M | **15.76** | **36.03** | **0.2895** |

### Ablation Study

Unseen concept transfer experiments (training excludes animal images, using only non-animal 188K prompts):

| Config | Unseen (Animal) FID↓ | Unseen CLIP↑ | Seen+Unseen FID↓ |
|------|----------------|-----------|---------------|
| w/o Random Cond | 37.86 | 0.2478 | 15.66 |
| + 24K Animal Text | 23.26 | 0.2833 | 13.50 |
| + 24K + 20M Text | 24.71 | 0.2913 | 14.47 |

Completely data-free distillation (GPT-generated prompts):

| Data Source | FID↓ | IS↑ | CLIP↑ |
|---------|------|-----|-------|
| LAION (w/o RC) | 18.15 | 33.81 | 0.2864 |
| LAION (w/ RC) | 15.76 | 36.03 | 0.2896 |
| GPT-generated (w/ RC) | **14.98** | **36.70** | **0.2952** |

### Key Findings

- Random Conditioning's improvement (14.72% reduction in FID, 8.29% increase in IS) even exceeds the gains achieved by using real images.
- Unseen concept transfer is highly effective—providing animal-related texts only (no images) reduces the animal FID from 37.86 to 23.26.
- Surprisingly, completely data-free distillation using GPT-generated prompts (FID 14.98) outperforms using real LAION text (FID 15.76), possibly due to the higher quality and diversity of LLM prompts.
- Even without teacher initialization, models with Random Conditioning can perform on par with those using teacher initialization but without RC.

## Highlights & Insights

- **Counter-intuitive yet effective**: Training a diffusion model with mismatched text-image pairs intuitively introduces noise. However, the authors elegantly explain why this is feasible by analyzing how condition influence changes across timesteps. This insight reveals the essence of the conditioning mechanism in diffusion models.
- **High versatility**: Random Conditioning does not rely on a specific architecture and can be combined with any distillation method (e.g., block pruning / channel compression), making it a plug-and-play technique.
- **Completely data-free distillation**: Combining LLM-generated prompts with RC achieves diffusion model compression without requiring any real data, which holds high application value in privacy-sensitive fields such as healthcare and legal domains.

## Limitations & Future Work

- Currently only validated on Stable Diffusion v1.4, without testing newer models like SDXL or SD3.
- The form of $p(t)$ (exponential function) is manually designed, and better adaptive strategies may exist.
- Channel compression models perform slightly worse than block compression because they cannot reuse teacher weight initialization.
- The optimal relationship between replacement probability and timesteps might vary with model architectures and data distributions—requiring more systematic theoretical analysis.
- Integrating RC with step-acceleration methods (such as Consistency Distillation) could be considered to achieve "smaller + faster" compression.

## Related Work & Insights

- **vs BK-SDM**: BK-SDM is a SOTA SD compression method (block pruning + feature distillation) but requires real images. This work adds RC on top of the BK-SDM architecture, outperforming BK-SDM without requiring real images.
- **vs Consistency Distillation (e.g., LCD)**: Methods like LCD aim to reduce sampling steps (e.g., 50 steps $\to$ 1 step), while this work aims to reduce model size. The two are orthogonal and complementary—one can first compress the model using this method and then accelerate sampling using LCD.
- **vs Classification Model Distillation**: In Hinton's classic distillation, unseen classes can naturally transfer (via inter-class relationships in soft labels). However, in diffusion models—due to the larger generation space, per-input noise prediction outputs, and dependence on intermediate states—unseen classes cannot automatically transfer. RC essentially compensates for this missing piece using text prior.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The core idea is simple, elegant, counter-intuitive yet effective, providing profound insights into the conditional mechanism of diffusion models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The experimental design is highly comprehensive, covering ablations, unseen concept transfer, data-free settings, and comparisons with SOTA.
- Writing Quality: ⭐⭐⭐⭐ The motivation and observations are clearly explained, and the MNIST example is intuitive and effective.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play and highly versatile, offering direct value for deploying diffusion models in data-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](../../NeurIPS2025/image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[CVPR 2025\] PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2025\] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders](randar_decoder-only_autoregressive_visual_generation_in_random_orders.md)

</div>

<!-- RELATED:END -->
