---
title: >-
  [Paper Note] Accelerating Image Super-Resolution Networks with Pixel-Level Classification
description: >-
  [ECCV 2024][Image Restoration][Super-Resolution Acceleration] This and paper introduces PCSR, the first super-resolution method with pixel-level computational resource allocation. By leveraging a lightweight MLP classifier, it determines the restoration difficulty on a pixel-by-pixel basis and assigns them to upsamplers of varying capacities. PCSR reduces FLOPs to $18\% \sim 57\%$ of the original models with almost no drop in PSNR, significantly outperforming existing patch-l…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Super-Resolution Acceleration"
  - "Pixel-Level Classification"
  - "Adaptive Computation Allocation"
  - "LIIF Upsampling"
  - "Large Image Super-Resolution"
date: 2026-05-08
content_hash: 9237eeb2d56bca8d
---

# Accelerating Image Super-Resolution Networks with Pixel-Level Classification

**Conference**: ECCV 2024  
**arXiv**: [2407.21448](https://arxiv.org/abs/2407.21448)  
**Code**: [https://github.com/3587jjh/PCSR](https://github.com/3587jjh/PCSR)  
**Area**: Image Restoration / Super-Resolution / Efficient Inference  
**Keywords**: Super-Resolution Acceleration, Pixel-Level Classification, Adaptive Computation Allocation, LIIF Upsampling, Large Image Super-Resolution  

## TL;DR
This and paper introduces PCSR, the first super-resolution method with pixel-level computational resource allocation. By leveraging a lightweight MLP classifier, it determines the restoration difficulty on a pixel-by-pixel basis and assigns them to upsamplers of varying capacities. PCSR reduces FLOPs to $18\% \sim 57\%$ of the original models with almost no drop in PSNR, significantly outperforming existing patch-level methods like ClassSR and ARM.

## Background & Motivation
With the skyrocketing demand for high-resolution images in $2\text{K} \sim 8\text{K}$, super-resolution models need to process increasingly larger images. In practice, large images cannot be processed in one go and are usually cropped into overlapping patches for individual super-resolution before being stitched back together. Existing methods such as ClassSR and ARM observe that different patches exhibit varying restoration difficulties. They assign smaller models to simple patches and larger models to difficult patches, saving computation. However, patch-level allocation suffers from two fundamental limitations: (1) Even if the vast majority of pixels in a patch are simple (e.g., flat regions), the presence of a few high-frequency details forces the entire patch to be processed by the larger model, leading to computational waste; conversely, if the patch is classified as simple overall, the high-frequency pixels will be poorly restored. (2) Larger patches are more likely to contain a mix of simple and complex pixels, resulting in lower efficiency. Yet, larger patches are desirable to reduce overlapping redundancy and utilize more context, creating a core conflict.

## Core Problem
How to refine the adaptive computation allocation granularity of super-resolution from the patch level to the pixel level, allowing each pixel to be assigned to an upsampler of appropriate capacity, thereby eliminating redundant computation at a finer grain?

## Method

### Overall Architecture
The PCSR model consists of three components: **Backbone** + **Pixel-Level Classifier** + **M pixel-level upsamplers of different capacities (Upsamplers)**. The complete workflow is: (1) The backbone network processes the LR input to generate LR feature maps; (2) The lightweight MLP classifier predicts the probability of belonging to each upsampler for each query pixel in the HR space; (3) Each pixel is routed to its corresponding upsampler to predict the RGB values; (4) The RGB values of all pixels are aggregated to produce the final SR output. The backbone network can be any existing SR model (e.g., FSRCNN/CARN/SRResNet), making PCSR a plug-and-play acceleration framework.

### Key Designs
1. **Pixel-Level MLP Classifier**: Based on the LIIF paradigm, for each HR pixel coordinate, nearest-neighbor features are extracted from the LR feature map and concatenated with relative coordinate offsets. This is fed into a lightweight MLP to output an M-dimensional classification probability. The classifier introduces minimal computational overhead while accurately distinguishing flat areas from high-frequency detail areas. Specifically, the input to the classifier uses the difference between the HR image and the bilinearly upsampled LR image as the target to highlight high-frequency features, helping the classifier focus more on the judgment of restoration difficulty.

2. **Multi-Capacity LIIF Upsamplers**: LIIF (Local Implicit Image Function) is adopted as the upsampler, which naturally supports pixel-level processing. M MLP upsamplers with different hidden layer sizes are defined, ranging from heavy ($U_0$) to light ($U_{M-1}$), with monotonically decreasing capacities. In experiments, setting $M=2$ is highly effective—a heavy upsampler processes difficult pixels (textures/edges), and a lightweight upsampler processes simple pixels (flat regions). Using LIIF brings an additional benefit: a single model can support multi-scale and even arbitrary-scale super-resolution without needing separate training for each scale factor.

3. **Execution-time Adjustable Computation-Performance Trade-off**: A hyperparameter $k$ is introduced to control the allocation preference. The allocation formula is $\operatorname{argmax}_j p_{i,j} / [\operatorname{cost}(U_j)]^k$, where $\operatorname{cost}$ is the softmax-normalized FLOPs of each upsampler. A smaller $k$ routes more pixels to heavy upsamplers (higher quality), whereas a larger $k$ routes more pixels to lightweight upsamplers (lower computation). Users can dynamically adjust the efficiency-quality trade-off without retraining.

4. **Adaptive Decision Making (ADM)**: Besides manually tuning $k$, an automated allocation scheme based on K-means clustering is provided. The pixel difficulty values (sum of probabilities of the heavy upsampler) across the entire image are clustered into M groups using K-means. Pixels are routed to heavy-to-light upsamplers according to the ordered cluster centers. This clustering converges in only 2 to 7 iterations, yielding practically zero extra overhead.

5. **Pixel-wise Refinement**: Because adjacent pixels may be assigned to different upsamplers, artifacts may appear along the boundary. The proposed solution is simple yet effective: if a pixel processed by a lightweight upscampler is adjacent to any neighbor processed by the heavy upsampling branch, its value is replaced with the average RGB of all neighboring pixels (including itself). This refinement requires no additional forward pass, and its computational overhead is negligible.

### Loss & Training
- **Two-stage Training**: The backbone and $U_0$ (the heaviest upsampler) are first trained using reconstruction loss to guarantee the upper bound of performance. Subsequently, the trained modules are frozen, and the lightweight upsamplers and classifier are added and trained sequentially.
- **Reconstruction Loss $L_{\text{recon}}$**: The L1 distance between the predicted value and the target (GT minus bilinear upsampling), emphasizing high-frequency residuals.
- **Balancing Loss $L_{\text{avg}}$**: Encourages roughly equal pixel allocation across categories to prevent the classifier from collapsing into routing all pixels to a single branch.
- The initial stage is trained for 2000K iterations, followed by 500K iterations for each subsequent stage, using the Adam optimizer with cosine annealing.

## Key Experimental Results

### Comparison with Patch-level Methods ($\times 4$ SR)

| Model | Params | Test2K (dB/GFLOPs) | Test4K (dB/GFLOPs) | Test8K (dB/GFLOPs) |
|------|--------|-------|-------|-------|
| FSRCNN | 25K | 25.69 / 45.3 (100%) | 26.99 / 185.3 (100%) | 32.82 / 1067.8 (100%) |
| FSRCNN-ClassSR | 113K | 25.61 / 38.4 (85%) | 26.91 / 146.4 (79%) | 32.73 / 709.2 (66%) |
| FSRCNN-ARM | 25K | 25.61 / 35.6 (79%) | 26.91 / 152.9 (83%) | 32.73 / 746.7 (70%) |
| **FSRCNN-PCSR** | **25K** | **25.61 / 8.5 (19%)** | **26.91 / 32.6 (18%)** | **32.73 / 196.6 (18%)** |
| SRResNet | 1.5M | 26.24 / 502.9 (100%) | 27.71 / 2056.2 (100%) | 33.55 / 11850.7 (100%) |
| SRResNet-ClassSR | 3.1M | 26.20 / 446.7 (89%) | 27.66 / 1686.2 (82%) | 33.50 / 7996.0 (67%) |
| SRResNet-ARM | 1.5M | 26.20 / 429.1 (85%) | 27.66 / 1742.2 (85%) | 33.50 / 7865.3 (66%) |
| **SRResNet-PCSR** | **1.1M** | **26.20 / 245.6 (49%)** | **27.66 / 981.0 (48%)** | **33.52 / 5093.7 (43%)** |

### Comparison with the Per-image Method MGA

| Model | Test2K (dB/GFLOPs) | Test4K (dB/GFLOPs) |
|------|-------|-------|
| FSRCNN-MGA (43K) | 25.66 / 29.2 (64%) | 26.94 / 101.7 (55%) |
| **FSRCNN-PCSR (25K)** | **25.66 / 12.8 (28%)** | **26.94 / 37.8 (20%)** |
| SRResNet-MGA (2.0M) | 26.20 / 249.2 (50%) | 27.66 / 871.9 (42%) |
| **SRResNet-PCSR (0.9M)** | **26.20 / 191.0 (38%)** | **27.66 / 755.3 (37%)** |

### Runtime Comparison (CARN backbone, $\times 4$)

| Method | Test2K | Test4K | Test8K |
|------|--------|--------|--------|
| ClassSR | 1994ms | 4595ms | 19072ms |
| ARM | 518ms | 1069ms | 4608ms |
| **PCSR** | **45ms** | **62ms** | **203ms** |

### Ablation Study

- **Effect of Patch Size**: The efficiency of ClassSR decreases as the patch size increases ($32 \to 256$: $68\% \to 77\%$ FLOPs). In contrast, PCSR becomes more efficient ($62\% \to 55\%$ FLOPs) because larger patches contain a higher percentage of simple pixels.
- **Number of Categories $M$**: There is negligible efficiency difference between $M=2$ and $M=3$ ($57\%$ vs $56\%$). Since $M=2$ requires fewer parameters and is simpler, it is selected as the default.
- **Pixel-wise Refinement**: Even if all lightweight pixels are replaced ($\#h=0$), PSNR drops by only $0.027\text{dB}$. Setting $\#h=1$ successfully eliminates artifacts with almost no impact on PSNR.
- **LIIF vs. PixelShuffle Upsampler**: LIIF does not guarantee a higher PSNR upper bound than the original PixelShuffle. Selecting LIIF is purely to enable pixel-level processing and arbitrary-scale SR.

## Highlights & Insights
- **Granularity Breakthrough**: Transitioning from patch-level to pixel-level computation allocation is a natural yet non-trivial progress. Pixel-level classification precisely distinguishes "hard pixels at texture edges" from "easy pixels in flat regions" within the same patch.
- **Stunning Efficiency Gain**: FSRCNN-PCSR runs on only $18\%$ FLOPs on Test8K. The execution runtime shows overwhelming superiority (PCSR $203\text{ms}$ vs. ClassSR $19072\text{ms}$), which is nearly a 100-fold difference.
- **Plug-and-Play + Runtime Adjustable**: PCSR is compatible with any existing SR backbone. The trade-off between quality and efficiency can be adjustably balanced at runtime using the parameter $k$ without retraining.
- **Ingenious Reuse of LIIF**: Extending the local implicit image function framework from arbitrary-scale SR to serve as a carrier for computation allocation elegantly achieves multi-scale SR capability at the same time.
- **Ultra-simple Refinement**: Utilizing neighborhood mean substitution alone effectively eliminates artifacts without requiring any additional neural network inference.

## Limitations & Future Work
- **Backbone Computation is the FLOPs Lower Bound**: The classifier depends on backbone features. Even if all pixels are assigned to the lightest upsampler, the computation of the backbone network cannot be bypassed. This part of the computation is still redundant for large flat regions.
- **Directions Proposed by Authors**: Allowing the classifier to operate in the early layers of the backbone, or directly applying bilinear interpolation with lookup tables for extremely simple pixels to completely bypass neural network processing.
- **Performance Ceiling of LIIF Upsamplers**: For larger models (e.g., SRResNet), LIIF does not necessarily outperform the original PixelShuffle upsampler, indicating a representation ceiling for pixel-level MLP upsamplers.
- **Generative SR Unexplored**: The method is evaluated only on PSNR-oriented approaches, without integration with GAN-based or Diffusion-based SR.
- **Trade-off between Classifier Accuracy and Overhead**: The current classifier is designed to be extremely lightweight, which may draft less accurate classifications in highly complex texture scenes.

## Related Work & Insights
- **vs. ClassSR (CVPR 2021)**: ClassSR utilizes independent sub-networks to process patches of varying difficulties, doubling the parameter count (e.g., FSRCNN: $25\text{K} \to 113\text{K}$), and the patch-level allocation granularity is too coarse. PCSR routes at a pixel level, uses even fewer parameters, and achieves $3\times$ to $4\times$ greater efficiency gains compared to ClassSR.
- **vs. ARM (ECCV 2022)**: ARM resolves the parameter explosion of ClassSR via parameter-sharing sub-networks but remains a patch-level method, which is limited by the mixing of easy and hard pixels within a single patch. PCSR naturally supports pixel-level execution using LIIF upsamplers, further improving efficiency by approximately $2\times$.
- **vs. MGA (ECCV 2022)**: MGA is a per-image method that performs coarse global restoration followed by fine local restoration, requiring mask prediction and two forward passes. PCSR features a simpler architecture with only a single forward pass, lower FLOPs, and fewer parameters.

## Inspiration & Connection
- **Generality of Adaptive Computation Allocation**: The concept of pixel-level difficulty classification paired with multi-path routing can be extended to other dense prediction tasks (e.g., denoising, enhancement, segmentation), especially when processing high-resolution inputs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The transition from patch to pixel-level allocation is a natural and clear progression; while it is not revolutionary, it is highly practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across multiple backbones (3 types) $\times$ multiple datasets (4+3) $\times$ extensive ablation studies (patch size, $M$, refinement, LIIF); extremely solid.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clearly articulated, the method is comprehensively described, and the visual quality of tables/figures is high.
- **Value**: ⭐⭐⭐⭐ Highly valuable for large-scale image SR scenarios, showing dramatic improvements in both FLOPs and runtime.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](../../CVPR2025/image_restoration/pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)

</div>

<!-- RELATED:END -->
