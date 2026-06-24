---
title: >-
  [Paper Note] AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution
description: >-
  [ECCV 2024][Image Generation][Diffusion Models] Observing that the required denoising steps for different image regions in diffusion-based super-resolution vary significantly (background regions converge early while foreground textures still need iterations), this work proposes a dynamic step-skipping strategy based on Multi-Metric Latent Entropy (MMLE) to perceive information gain. Sub-regions are categorized into stable, growth, and saturated types…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Models"
  - "Image Super-Resolution"
  - "Adaptive Inference"
  - "Dynamic Timestep Sampling"
  - "Region-Aware Acceleration"
date: 2026-05-08
content_hash: 97655cb1b501847b
---

# AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution

**Conference**: ECCV 2024  
**arXiv**: [2410.17752](https://arxiv.org/abs/2410.17752)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Image Super-Resolution, Adaptive Inference, Dynamic Timestep Sampling, Region-Aware Acceleration  

## TL;DR
Observing that the required denoising steps for different image regions in diffusion-based super-resolution vary significantly (background regions converge early while foreground textures still need iterations), this work proposes a dynamic step-skipping strategy based on Multi-Metric Latent Entropy (MMLE) to perceive information gain. Sub-regions are categorized into stable, growth, and saturated types, each assigned different step sizes. Concurrently, a Progressive Feature Injection (PFJ) module is developed to balance fidelity and realism. On datasets such as DRealSR, this approach achieves reconstruction quality comparable to StableSR while reducing inference time and FLOPs by 1.5$\times$ and 2.7$\times$, respectively.

## Background & Motivation
Diffusion-model-based super-resolution methods (e.g., StableSR, ResShift) yield excellent performance in real-world scenarios but suffer from high computational overhead, requiring a massive number of denoising iterations for reconstruction. Existing approaches apply a uniform number of timesteps to the entire image, ignoring the variance in recovery difficulty across different regions. This paper observes a key phenomenon: as the denoising steps increase from 50 to 200, the visual appearance of background regions like skies and water remains virtually unchanged, whereas foreground structures like building surfaces and textures continuously improve. This implies that substantial computational resources are wasted on recovering imperceptible details. Previous adaptive inference works (e.g., ClassSR, APE) focused on adaptive early-exits in the network layer dimension, leaving information dynamics along the timestep dimension of diffusion models largely unexplored.

## Core Problem
How to achieve region-level dynamic computational resource allocation in diffusion-based super-resolution—allocating fewer timesteps to simple regions while maintaining sufficient iterations for complex regions—thereby significantly reducing inference overhead without sacrificing reconstruction quality?

## Method

### Overall Architecture
The input low-resolution (LR) image is first cropped into overlapping sub-regions consistent with the pre-trained diffusion model's resolution (typically 512$\times$512), and each sub-region undergoes the denoising process independently. During denoising, the MMLE regressor estimates the multidimensional information gain of each sub-region in real-time, and the DTSS strategy dynamically adjusts the step-skipping interval accordingly (large skips for simple regions, small steps for complex regions, and early exit for saturated regions). Meanwhile, the PFJ module dynamically injects original image features based on the information gain to ensure fidelity. Finally, the sub-regions are stitched in the latent space using a Gaussian weight map to eliminate boundary discontinuities.

### Key Designs
1. **Multi-Metric Latent Entropy Module (MMLE)**: Four Full-Reference (FR) metrics (PSNR, LPIPS, AHIQ, NLPD) and two No-Reference (NR) metrics (BRISQUE, MUSIQ) are selected to measure denoising information gain from multiple dimensions. The representation quality at the current timestep is calculated as $R_i = \sum_{c \in C} \omega_c \times M_c(f_i, o)$, and the gain is scaled to $[-1, 1]$ via $I_i = \tanh(R_i - R_{i-1})$ to capture potential degradation. To avoid the massive overhead of computing continuous IQA metrics in real-time, a lightweight convolutional regressor is trained to approximate the multidimensional information gain. Key discovery: FR metrics robustly reflect information evolution throughout the entire denoising process, whereas NR metrics are easily corrupted by noise and are only effective in the later stages; hence, NR metrics only participate in computation during the latter half of the denoising process.

2. **Dynamic Timestep Sampling Strategy (DTSS)**: A step-skipping codebook is established with an information gain threshold of $\tau=5\times10^{-3}$ and maximum timesteps $T_{max}=1000$. Sub-regions are dynamically classified into three categories based on the gain trends of FR/NR metrics: **Stable regions** (FR increases while NR remains flat $\rightarrow$ large-interval skips), **Growth regions** (both FR and NR increase $\rightarrow$ small-interval steps to guarantee quality), and **Saturated regions** (NR drops significantly $\rightarrow$ save the best results and exit early). Four tiers of step-skipping intervals (5, 10, 15, 20) are utilized, and region categories can switch dynamically during the denoising process.

3. **Progressive Feature Injection Module (PFJ)**: The original image features $o$ are dynamically modulated via $\hat{o} = \alpha \times o + \beta$, where $\alpha, \beta = \phi(o, I_i)$ are predicted by a small CNN based on the current information gain and original features. When the information gain from the NR perspective is prominent (realism is improving), the modulation coefficients are increased to strengthen fidelity constraints; otherwise, they are relaxed to unleash the generation capability of the diffusion model. Compared to simple concatenation (concat) or cross-attention, this dynamic modulation guided by information gain achieves the best performance.

### Loss & Training
- Fine-tuned based on Stable Diffusion 2.1-base with a learning rate of $5\times10^{-5}$ using the Adam optimizer.
- Independent training of the MMLE regressor: The fine-tuned diffusion model parameters are frozen, and 512$\times$512 synthetic LR-HR pairs generated via the Real-ESRGAN degradation pipeline are utilized for L2 loss optimization.
- Sub-region stitching uses a Gaussian kernel weight map for fusion in the latent space and is executed only in the final timestep to avoid neighborhood noise corruption.

## Key Experimental Results

| Dataset | Metric | AdaDiffSR | StableSR | ResShift | Real-ESRGAN+ |
|--------|------|-----------|----------|----------|--------------|
| DIV2K Valid | LPIPS↓ | **0.2153** | 0.2328 | 0.4406 | 0.2284 |
| DIV2K Valid | MUSIQ↑ | **68.81** | 66.73 | 67.84 | 64.65 |
| RealSR | LPIPS↓ | 0.2595 | **0.2543** | 0.2524 | 0.2869 |
| DRealSR | LPIPS↓ | **0.2627** | 0.2853 | 0.5408 | 0.2818 |
| DRealSR | SSIM↑ | **0.8415** | 0.8326 | 0.8056 | 0.7987 |
| DPED-iPhone | NIQE↓ | **3.09** | 3.80 | 5.58 | 3.17 |

Efficiency comparison (512$\times$512, 50-step DDIM): AdaDiffSR reduces inference time by approximately 1.5$\times$ and FLOPs by about 2.7$\times$ compared to StableSR.

### Ablation Study
- **Step-skipping interval**: (5, 10, 15, 20) yields the optimal balance point. Setting all intervals to 5 gives the best quality but is the slowest (13.4s); larger intervals accelerate speed but degrade quality. The final configuration (9.1s) achieves LPIPS = 0.2627 on DRealSR.
- **Combination of IQA metrics**: Using only FR metrics leads to high PSNR but low MUSIQ (excessive fidelity with insufficient realism); using only NR metrics yields high MUSIQ but low PSNR (over-generation). Combining both achieves the best balance.
- **PFJ vs. alternatives**: The Concat mechanism yields a MUSIQ of 32.19, Cross-attention reaches 42.37, while PFJ reaches 51.84. Dynamic modulation guided by information gain is significantly superior to static fusion.
- **MMLE regressor vs. ground-truth IQA calculation**: The performance gap between the two is minimal, but the regressor dramatically reduces inference time.
- **Cropping strategy**: Fixed grid cropping and superpixel segmentation (e.g., SAM/FastSAM/MobileSAM) achieve similar quality, but segmentation methods introduce additional parameters and zero-padding overhead.

## Highlights & Insights
- **The core insight linking region with timestep** is extremely intuitive yet powerful: different regions demand different numbers of denoising steps. Although conceptually straightforward, this observation has not been systematically exploited in diffusion-based SR.
- **The collaborative fusion of multidimensional IQA metrics** is highly novel: utilizing FR and NR metrics complementarily and revealing their temporal difference characteristics during denoising (FR is effective throughout, whereas NR is only effective in the latter half) allows for the design of a phased strategy.
- **Substituting ground-truth IQA computation with a lightweight regressor** is a highly practical engineering trick, rendering the entire scheme nearly zero-overhead during inference.
- The dynamically switchable design of region classification (stable/growth/saturated) enhances robustness.

## Limitations & Future Work
- The authors acknowledge that DM-based SR is still far slower than CNN/GAN-based methods; dynamic step-skipping merely reduces redundant computations within the diffusion model.
- Static grid cropping does not fully align with the original design intent of distinguishing "foreground/background"—future work may explore finer-grained semantic region partitioning.
- The information gain threshold $\tau$ and step-skipping intervals are fixed hyperparameters; different datasets might require different configurations.
- The method is validated only on 512$\times$512 resolution with SD 2.1-base and has not yet been scaled to larger models (e.g., SDXL) or higher resolutions.

## Related Work & Insights
- **vs. StableSR**: Both being SD-based SR models, StableSR applies uniform denoising to all regions, while AdaDiffSR introduces region-adaptive step-skipping to achieve acceleration. They reach comparable reconstruction quality, but AdaDiffSR's computational cost is approximately $1/2.7$ of StableSR.
- **vs. ResShift**: ResShift reduces total timesteps via residual shifting but still processes the entire image uniformly. The region-level dynamic strategy of AdaDiffSR is orthogonal to ResShift and can be combined.
- **vs. ClassSR/APE**: ClassSR performs adaptation in the network layer dimension (allocating models of different capacities to different patches), while APE utilizes layer-by-layer early exits. AdaDiffSR performs adaptation in the timestep dimension; though operating on different dimensions, their core paradigms of adaptive resource allocation are highly similar.
- **Inspiration for adaptive inference research**: The core concept that "different input regions mandate unequal computation budgets" can be extended to other tasks such as text-to-image generation and video generation in diffusion models, where certain background tokens might only require a minimal number of denoising steps.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The adaptive strategy coupling regions with timesteps is systematically proposed in DM-SR for the first time, although adaptive inference itself is not a new concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on both synthetic and real-world datasets, detailed ablation studies coverage for all modules, and solid efficiency comparison.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation diagram (Fig. 1) is intuitive and compelling, and the overall logic is highly clear.
- **Value**: ⭐⭐⭐⭐ It provides an effective and general paradigm for efficiency optimization in diffusion-based SR; the step-skipping codebook + regressor scheme can be easily utilized as a plug-and-play module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)

</div>

<!-- RELATED:END -->
