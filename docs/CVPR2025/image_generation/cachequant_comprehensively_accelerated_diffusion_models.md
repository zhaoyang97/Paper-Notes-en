---
title: >-
  [Paper Note] CacheQuant: Comprehensively Accelerated Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Diffusion Model Acceleration] CacheQuant is proposed, a training-free paradigm that comprehensively accelerates diffusion models by jointly optimizing model caching (temporal level) and quantization (structural level). It achieves 5.18× acceleration and 4× compression on Stable Diffusion, with a CLIP score loss of only 0.02.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Model Acceleration"
  - "Model Caching"
  - "Quantization"
  - "Dynamic Programming"
  - "Training-free"
date: 2026-05-08
content_hash: b0e4201157a0db7b
---

# CacheQuant: Comprehensively Accelerated Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2503.01323](https://arxiv.org/abs/2503.01323)  
**Code**: Open-sourced (mentioned in the paper)  
**Area**: Diffusion Models / Model Compression  
**Keywords**: Diffusion Model Acceleration, Model Caching, Quantization, Dynamic Programming, Training-free

## TL;DR
CacheQuant is proposed, a training-free paradigm that comprehensively accelerates diffusion models by jointly optimizing model caching (temporal level) and quantization (structural level). It achieves 5.18× acceleration and 4× compression on Stable Diffusion, with a CLIP score loss of only 0.02.

## Background & Motivation

**Background**: Diffusion models have achieved remarkable results in the field of image generation, but their slow inference speeds (requiring thousands of denoising iterations) and complex network structures (billions of parameters) seriously hinder practical deployment. Even on a high-performance A6000 GPU, a single inference of Stable Diffusion takes over one minute.

**Limitations of Prior Work**: Existing acceleration methods optimize independently at two levels: the temporal level (e.g., fast solvers, model caching) shortens the denoising trajectory but cannot simplify the network structure; the structural level (e.g., quantization, pruning) simplifies the network but requires expensive retraining. Crucially, as each level is individually pushed to its limit (e.g., shorter denoising paths or fewer parameters), performance degrades severely.

**Key Challenge**: Caching and quantization optimizations are not completely orthogonal. Experiments reveal that when simply combining them after independent optimization (LDM on ImageNet), quantization and caching individually lose only 0.76 and 4.71 in FID, respectively, but their simple combination leads to an FID loss of up to 11.99. The reason is that the errors introduced by each method couple and progressively accumulate—quantization errors cause the cached denoising path to deviate significantly, while caching errors heavily accumulate quantization errors.

**Goal**: How to jointly optimize diffusion model acceleration at both the temporal and structural levels, while controlling coupling errors to maintain generation quality.

**Key Insight**: The authors observe a collaborative relationship between caching and quantization—quantization can reduce the memory overhead increased by caching, while caching can alleviate the quantization difficulty caused by temporal redundancy. The key is to jointly optimize them to handle coupling errors.

**Core Idea**: Select the optimal cache schedule via dynamic programming to minimize the joint cache-quantization error, and sequentially eliminate the coupled cumulative errors using decoupled error correction, achieving training-free comprehensive acceleration.

## Method

### Overall Architecture
The input to CacheQuant is a standard diffusion model (supporting both UNet and DiT frameworks), and the output is an accelerated model featuring both model caching and low-precision quantization. The entire pipeline consists of two phases: (1) determining the optimal caching schedule via Dynamic Programming Schedule (DPS), minimizing errors by simultaneously considering the characteristics of caching and quantization; (2) sequentially eliminating the coupled cumulative errors of caching and quantization at each timestep via Decoupled Error Correction (DEC). For the UNet framework, output features of the upsampling blocks are cached; for the DiT framework, deviations between two blocks are cached.

### Key Designs

1. **Dynamic Programming Schedule (DPS)**

    - **Function**: Find the optimal cache refresh schedule for the caching mechanism to minimize the joint error.
    - **Mechanism**: Model the cache scheduling problem as a dynamic programming problem of ordered sample grouping. For a model with $T$ denoising steps and a cache frequency $N$, all feature maps are divided into $K=T/N$ groups, with each group sharing the same cached feature. Define the within-group error $D_k(i,j) = \sum_{t=i+1}^{j} \|X_g^i - X_g^t\|_1$ (measured by the L1 norm, as quantization errors originate from absolute numerical differences). Then, solve $M(T,K) = \min_{K \leq s \leq T}\{M(s-1,K-1) + D(s,T)\}$ recursively via DP to find the grouping plan that minimizes the total error. To reduce computational complexity, the group length is constrained to the range of $[N/2, 2N]$, reducing the solving time for LDM with 250 steps from 4 hours to 8 minutes.
    - **Design Motivation**: Traditional methods use uniform cache scheduling or manual hyperparameter tuning, which fail to account for the additional error introduced by quantization. By incorporating both caching and quantization characteristics into the error calculation, DPS finds the truly optimal scheduling plan.

2. **Decoupled Error Correction (DEC)**

    - **Function**: Sequentially eliminate the coupled cumulative errors of caching and quantization during inference.
    - **Mechanism**: Decouple the total error $E_o = O_g - O_{cq}$ into caching error $E_c = O_g - O_c$ and quantization error $E_q = O_c - O_{cq}$, and then correct them separately. For the caching error, correct it along the input channel dimension as $X_g = a_1 \cdot X_c + b_1$; for the quantization error, correct it along the output channel dimension as $O_c = a_2 \cdot O_{cq} + b_2$. The correction parameters are solved via the least squares method (leveraging the strong cross-channel correlation). The core advantage is theoretically provable: compared to direct correction, which only adjusts the mean and variance of the output channel, DEC adjusts along both the input and output channel dimensions, which not only eliminates the mean error but also effectively reduces variance.
    - **Design Motivation**: Performing channel correction directly on $O_{cq}$ can eliminate the mean error, but the variance remains large (because the caching error originates from the divergence between $X_g$ and $X_c$ and cannot be effectively handled at the output end). Decoupling and correcting along different dimensions additionally improves FID by 0.91. Moreover, the correction parameters for quantization error can be absorbed into weight quantization, introducing only one additional matrix fused multiply-add during inference.

3. **Synergistic Acceleration Mechanism**

    - **Function**: Achieve complementary advantages of caching and quantization.
    - **Mechanism**: Caching skips redundant computation (accelerating at the temporal level), while quantization compresses the model using low-precision representations (accelerating at the structural level). The combination of both yields results far surpassing their individual optimization. Specifically, caching reduces the computational workload that needs to be quantized (cached blocks do not execute quantization inference), while quantization minimizes the storage overhead of cached features.
    - **Design Motivation**: Vertically pushing either direction to its limit severely degrades performance, but joint optimization sustains quality at a higher acceleration ratio.

### Loss & Training
CacheQuant is a completely training-free paradigm that requires no fine-tuning or retraining. The optimization goal of DPS is to minimize the sum of within-group L1 errors across all groups, and the correction parameters of DEC are computed on a calibration set using the least squares method. Optionally, it can be combined with quantization reconstruction methods to further improve performance (which requires a small amount of training).

## Key Experimental Results

### Main Results

| Model/Dataset | Method | Bops↓ | Speedup↑ | Compression↑ | Retraining Required | FID↓ |
|-------------|------|-------|-------|-------|--------|------|
| SD / MS-COCO | Deepcache-N=10 | 133.58T | 3.52× | 1.00× | No | 23.45 |
| SD / MS-COCO | **CacheQuant-N=5 W8A8** | **8.44T** | **5.18×** | **4.00×** | **No**| **23.74** |
| SD / MS-COCO | BK-SDM-Base | 57.30T | 2.79× | 3.54× | Yes | 28.47 |
| LDM-4 / ImageNet | Deepcache-N=5 | 24.06T | 4.12× | 1.00× | No | 3.79 |
| LDM-4 / ImageNet | **CacheQuant-N=5 W8A8** | **1.50T** | **7.87×** | **4.00×** | **No**| **4.03** |
| LDM-4 / ImageNet | EDA-DM W8A8 | 6.39T | 1.91× | 4.00× | Yes | 4.13 |
| DiT-XL/2 / ImageNet | Δ-DiT-N=2 | 87.88T | 1.31× | 1.00× | No | 9.06 |
| DiT-XL/2 / ImageNet | **CacheQuant-N=2 W8A8** | **5.49T** | **2.72×** | **4.00×** | **No**| **7.86** |

### Ablation Study

| Configuration | FID↓ | IS↑ | Description |
|------|------|-----|------|
| Deepcache-N=20 (FP32) | 8.08 | 159.27 | Caching-only baseline |
| + Direct W8A8 Quantization | 15.36 | 121.78 | Simple combination, performance degrades severely |
| + DPS | 8.47 | 154.07 | Optimal scheduling significantly restores performance |
| + DPS + DEC | 7.21 | 160.68 | Decoupled correction further improves performance |
| + DPS + DEC + Recon | 6.34 | 180.42 | With quantization reconstruction (requires a small amount of training) is optimal |

### Key Findings
- DPS contributes the most: reducing the FID from 15.36 of the simple combination directly to 8.47, indicating that the optimal cache scheduling is key to joint optimization.
- DEC reduces the FID from 8.47 to 7.21 without training, with minimal inference overhead (adding only one matrix fused multiply-add).
- At small cache frequencies (N=2,3), CacheQuant even outperforms the full-precision model in FID (e.g., 3.52 vs 3.99 on LSUN-Church), which is consistent with the regularization effect of quantization.
- In multi-platform deployment tests on GPU/CPU/ARM, GPU acceleration is the most significant (achieving 5× actual speedup on SD).

## Highlights & Insights
- **Synergy of Caching & Quantization**: The two are not orthogonal but rather complementary—quantization reduces caching storage overhead, while caching lowers the difficulty of quantization. This insight opens up a new combination pathway for diffusion model acceleration.
- **Elegant Dynamic Programming Modeling**: Formulating cache scheduling as an ordered grouping problem solved via DP for global optimality is much more systematic than manual tuning or heuristics. The optimization trick of constraining group lengths compresses the solving time from 4 hours to 8 minutes.
- **Clear Theoretical Foundation of DEC**: Through equivalent transformations, it is proven that direct correction is a special case of DEC under the assumption of $a_1=1, b_1=0$. Since this assumption does not hold in practice, DEC is strictly superior to direct correction.

## Limitations & Future Work
- Currently, only W8A8 and W4A8 precision configurations have been validated; the performance under extremely low-bit settings (e.g., W2A4) remains unexplored.
- DPS requires collecting feature maps from all timesteps in advance to calculate grouping errors, which increases the initialization overhead (though reduced to 8 minutes after optimization).
- Validation has only been performed on image generation tasks; generalization to more complex scenarios like video generation remains unexplored.
- Correction parameters in DEC are calculated on a calibration set, and the robustness against different prompt distributions remains to be verified.

## Related Work & Insights
- **vs DeepCache**: DeepCache only performs caching without quantization and utilizes a uniform schedule. At the same cache frequency, CacheQuant introduces quantization and optimal scheduling, reducing Bops by 16× while maintaining comparable FID.
- **vs EDA-DM**: EDA-DM only performs quantization and requires retraining, yielding limited acceleration (1.91×). CacheQuant achieves 7.87× speedup and better FID without any training.
- **vs Δ-DiT**: $\Delta$-DiT applies caching to the DiT framework. CacheQuant incorporates quantization on top of it, improving the speedup ratio from 1.31× to 2.72× and reducing the FID from 9.06 to 7.86.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to jointly optimize caching and quantization. The designs of DPS and DEC are rational, though neither technology is entirely novel on its own.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models including DDPM/LDM/SD/DiT, multiple datasets, and multi-platform deployments, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis, complete theoretical derivations, and rich vector illustrations/tables.
- Value: ⭐⭐⭐⭐ Highly practical; training-free 5× speedup and 4× compression hold direct value for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers](../../CVPR2026/image_generation/training-free_mixed-resolution_latent_upsampling_for_spatially_accelerated_diffu.md)
- [\[ICLR 2026\] Decoupled MeanFlow: Turning Flow Models into Flow Maps for Accelerated Sampling](../../ICLR2026/image_generation/decoupled_meanflow_turning_flow_models_into_flow_maps_for_accelerated_sampling.md)
- [\[ICML 2025\] ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering](../../ICML2025/image_generation/reframe_layer_caching_for_accelerated_inference_in_real-time_rendering.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Decentralized Diffusion Models](decentralized_diffusion_models.md)

</div>

<!-- RELATED:END -->
