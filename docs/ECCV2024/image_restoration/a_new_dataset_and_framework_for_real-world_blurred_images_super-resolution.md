---
title: >-
  [Paper Note] A New Dataset and Framework for Real-World Blurred Images Super-Resolution
description: >-
  [ECCV 2024][Image Restoration][Blind Super-Resolution] Addressing the issue where existing blind super-resolution methods over-texturize and destroy the perceptual quality of blurred regions when processing images with blur (defocus/motion blur), this work constructs the ReBlurSR dataset containing nearly 3,000 blurred images. It proposes the PBaSR framework, which employs Cross-Disentanglement training (CDM) and weight-interpolation-based Cross-Fusion (CFM) to simultaneously…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Blind Super-Resolution"
  - "Blurred Images"
  - "Dataset"
  - "Dual-Branch Decoupling"
  - "Weight Interpolation Fusion"
date: 2026-05-08
content_hash: 961a339bb6b6f2ea
---

# A New Dataset and Framework for Real-World Blurred Images Super-Resolution

**Conference**: ECCV 2024  
**arXiv**: [2407.14880](https://arxiv.org/abs/2407.14880)  
**Code**: [https://github.com/Imalne/PBaSR](https://github.com/Imalne/PBaSR)  
**Area**: Image Restoration / Super-Resolution  
**Keywords**: Blind Super-Resolution, Blurred Images, Dataset, Dual-Branch Decoupling, Weight Interpolation Fusion  

## TL;DR

Addressing the issue where existing blind super-resolution methods over-texturize and destroy the perceptual quality of blurred regions when processing images with blur (defocus/motion blur), this work constructs the ReBlurSR dataset containing nearly 3,000 blurred images. It proposes the PBaSR framework, which employs Cross-Disentanglement training (CDM) and weight-interpolation-based Cross-Fusion (CFM) to simultaneously improve the super-resolution quality of both blurred and general images without introducing any additional inference overhead, improving LPIPS by 0.02 to 0.10.

## Background & Motivation

Blind image super-resolution (BSR) has achieved significant progress on general images. However, an overlooked issue is that real-world images often contain **intentional blur** (e.g., defocus blur from a shallow depth-of-field to highlight the foreground, or motion blur from long exposure to suggest motion). These blurred areas should be preserved rather than eliminated during super-resolution. Existing methods, however, enhance textures uniformly across all regions, causing artifacts, over-sharpening, and unnatural texture generation in blurred areas. The authors' statistics on DIV2K-Val show that over 20% of samples contain obvious blur, and multiple SOTA methods suffer a drop of 0.01 to 0.06 in LPIPS on these samples. Neither existing datasets nor methods are tailored to meet this "blur-preservative" requirement.

## Core Problem

How can a blind super-resolution model adaptively preserve the visual perception properties of blurred regions while enhancing the texture quality of sharp regions? The key challenge lies in the training conflict between blurred data and general data: increasing the ratio of one type of data degrades the performance of the other (experimental results show a negative correlation and oscillation in LPIPS between the two). Furthermore, traditional GAN discriminators exhibit lower confidence and insufficient discriminative power in blurred regions.

## Method

### Overall Architecture

PBaSR is a plug-and-play training framework that can be applied to any existing BSR model (such as Real-ESRGAN, FeMaSR, or SRFormer). It takes a low-resolution image as input and outputs a $4\times$ super-resolved image. During training, two branches with identical architectures are maintained (a general branch for general images and a blur branch for blurred images). The two branches are optimized independently via CDM and periodically exchange information via CFM. During inference, the weights of the two branches are averaged equally into a single model, resulting in **zero additional inference overhead**.

### Key Designs

1. **ReBlurSR Dataset**: Consists of 2,931 high-quality blurred images (2,811 for training + 120 for testing) sourced from: (1) 849 images with manually annotated blur maps selected from blur detection datasets like CUHK/EBD; (2) 1,362 images obtained by performing blur detection on DIV2K/Flickr2K/DIV8K with D-DFFNet, followed by manual correction; (3) 601 images synthesized using Stable Diffusion and GPT-3.5. The dataset is finely categorized by blur type (defocus/motion), blur area (small/medium/large), and blur intensity (light/medium/heavy). Each image is accompanied by a blurred region mask.

2. **Cross Disentanglement Module (CDM)**: A parallel dual-branch architecture where the general branch uses standard training strategies for general data, while the blur branch is dedicated to training on blurred data. Crucially, a **conditional adversarial loss** is introduced to the discriminator of the blur branch: the blur map is concatenated with the image before being fed into the discriminator for conditional discrimination, allowing the discriminator to distinguish the optimization goals of blurred and sharp regions. Experiments show that incorporating the blur condition reduces the discriminator loss of blurred regions by about 0.17 and improves the LPIPS by 0.022.

3. **Cross Fusion Module (CFM)**: Resolves the issue where simply averaging weights after training branches independently yields suboptimal results. During training, weight interpolation and exchange are performed between the two branches every $k$ iterations. The interpolation coefficient $\lambda$ is dynamically adjusted by the base value $\lambda_0$ and the cosine similarity between the weights of the two branches: stronger interpolation is applied when weights are more similar, while a more conservative interpolation is used when the weight discrepancy is large. This ensures that the weight distance between the two branches remains within a mergeable range, allowing a simple equal-weight average to achieve a win-win outcome during final inference.

### Loss & Training

- **General Branch**: Follows the standard losses of the original BSR method (perceptual loss, GAN loss, etc.).
- **Blur Branch**: Introduces the blur map as a conditional input in the GAN loss: $L_{D_B} = \frac{1}{WHC}\sum(1-D_B(I_{HR}^B|M)) + (1+D_B(I_{SR}^B|M))$
- **CFM Parameters**: $\lambda_0=0.99$, communication frequency $k=20$, and a total of 200k training iterations.
- **Inference Fusion**: $W_{PBaSR} = W_G/2 + W_B/2$
- Training is completed on two NVIDIA V100 GPUs with a batch size of 16, patch size of 256, Adam optimizer, and learning rate of $1\times10^{-4}$.

## Key Experimental Results

| Dataset | Metric | PBaSR_FeMaSR | Prev. SOTA (HAT) | Gain |
|--------|------|------|----------|------|
| ReBlurSR-Test (Defocus) | LPIPS↓ | **0.3564** | 0.3924 | -0.036 |
| ReBlurSR-Test (Motion) | LPIPS↓ | **0.3624** | 0.3856 | -0.023 |
| 6 General Benchmarks (Avg) | LPIPS↓ | 0.3912 | 0.4850 | -0.094 |
| ReBlurSR-Test (Defocus) | DISTS↓ | **0.1733** | 0.2502 | -0.077 |
| ReBlurSR-Test (Motion) | DISTS↓ | **0.1771** | 0.2250 | -0.048 |

PBaSR is effective across three different architectures (CNN: Real-ESRGAN, VQVAE: FeMaSR, Transformer: SRFormer):

| Method | Architecture | Defocus LPIPS | Motion LPIPS | General LPIPS |
|------|------|------|------|------|
| Real-ESRGAN+PBaSR | CNN | 0.3986 (↓0.021) | 0.3791 (↓0.031) | 0.4390 (↓0.035) |
| FeMaSR+PBaSR | VQVAE | 0.3564 (↓0.047) | 0.3663 (↓0.092) | 0.3826 (↓0.002) |
| SRFormer+PBaSR | Transformer | 0.3740 (↓0.023) | 0.3887 (↓0.026) | 0.3892 (↓0.028) |

### Ablation Study

- **Scalability of Blurred Data**: Defocus and motion blur training data were added Clean and predictably, performance on various blur test types steadily improved without mutual interference, highlighting the scalability of the framework.
- **CFM Communication Frequency $k$**: $k=20$ is optimal. Setting $k=1$ (too frequent) or $k=100$ (too sparse) degrades LPIPS by 0.005 to 0.01.
- **CFM Interpolation Ratio $\lambda_0$**: Performance is stable when $\lambda_0 \in [0.9, 0.99]$. Setting $\lambda_0=1$ (no communication) or $\lambda_0=0$ (complete replacement) harms performance.
- **Direct Weight Averaging Without CFM**: Simply performing a weight average on two independently trained branches yields performance on par with the combined training baseline, justifying the necessity of CFM in maintaining a mergeable weight distance during training.
- **vs. GAN Artifact Correction Methods (LDL/DeSRA)**: PBaSR outperforms DeSRA by 0.007 to 0.02 in LPIPS on blurred data and also achieves superior results on general data.
- **vs. Feature Distillation/Teacher-Student Fusion**: CFM outperforms these two strategies by 0.006 to 0.008 in LPIPS and does not require additional backpropagation.

## Highlights & Insights

- **Unique Problem Insight**: The work identifies that existing BSR methods neglect the preservation of intentional blur. It clearly demonstrates this gap with quantitative experiments (using the 20%+ blurred samples in DIV2K-Val and showing negative correlation oscillations in LPIPS), making the motivation highly convincing.
- **Zero-Inference-Overhead Design Philosophy**: The combination of CDM and CFM enables dual-branch learning during training and equal-weight fusion into a single model for inference, adding no deployment cost. This design philosophy can be readily extended to other domains facing "multi-domain data conflicts."
- **Cross-Domain Communication via Weight Interpolation**: CFM replaces complex methods like feature distillation or teacher-student learning with adaptive weight interpolation, which is simple, efficient, and yields better results. This essentially applies the concept of **model merging (model souping)** directly to the training loop.
- **Strong Framework Generality**: The framework was successfully applied to three different architectures (CNN, VQVAE, and Transformer), and the code has been open-sourced.

## Limitations & Future Work

- **Dependence on Blur Maps**: Training requires blurred region masks (obtained manually or via automatic detection). The accuracy of automatic detection directly affects the training performance, and because no mask is required during inference, the model can only implicitly learn blur awareness.
- **Limited to $4\times$ Super-Resolution**: All experiments in the paper were conducted at $4\times$ magnification, leaving other scales unverified.
- **Limited Blur Types**: The dataset only covers defocus and motion blur, excluding other categories such as Gaussian blur and lens aberrations.
- **Primarily Perceptual Metrics**: Although the paper explains why PSNR/SSIM are insensitive to smooth areas, users in certain scenarios may still prioritize distortion metrics.
- **Promising Research Directions**: (1) Extending the CDM/CFM concepts to SR tasks with more degradation types (e.g., rain/fog + SR); (2) Using blur-aware prompts instead of blur maps for conditioning; (3) Integrating with diffusion models.

## Related Work & Insights

- **vs. CAL-GAN**: While CAL-GAN preserves blurred regions reasonably well, it over-smoothes sharp regions and loses details. PBaSR performs better in both regions.
- **vs. General BSR Methods (Real-ESRGAN/SwinIR/HAT)**: These methods commonly over-enhance blurred regions, generating fake textures and sharpening artifacts. PBaSR avoids this issue through decoupled learning.
- **vs. GAN Artifact Correction Methods (LDL/DeSRA)**: These focus on correcting the naturalness of synthesized textures but do not understand the semantics that blurred regions themselves do not require texture enhancement. PBaSR addresses the problem more fundamentally from the perspective of data distribution decoupling.

## Inspirations & Connections

- The concept of **model merging/weight interpolation** can be generalized: any training task with domain conflicts (e.g., multi-task learning, multi-domain adaptation) can attempt an independent training + periodic weight interpolation + final fusion scheme.
- The practice of introducing prior masks into **conditional discriminators** is similar to attention guidance in segmentation/detection tasks and can be transferred to other generation tasks requiring region-aware capabilities.
- This work complements literature on deblurring (e.g., the RGS method from ICCV 2025): while RGS focuses on "removing" unwanted blur, this paper focuses on "preserving" intentional blur—both indicating a need for more granular treatment of blur.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem perspective is novel (preserving blur instead of removing it); the combined CDM+CFM design is rational and effective, though the individual technical components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated across three architectures with multi-metric evaluations and thorough ablation studies (data scalability, communication frequency, fusion ratio, comparison with distillation, etc.) as well as comparisons against GAC methods.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is solidly supported by data and the pipeline is presented clearly, although some formulas contain complex notations.
- **Value**: ⭐⭐⭐⭐ Contributions in both datasets and frameworks; the framework is plug-and-play and open-source, though the application scenario is somewhat niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[ICLR 2026\] PlantRSR: A New Plant Dataset and Method for Reference-based Super-Resolution](../../ICLR2026/image_restoration/plantrsr_a_new_plant_dataset_and_method_for_reference-based_super-resolution.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[ECCV 2024\] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)

</div>

<!-- RELATED:END -->
