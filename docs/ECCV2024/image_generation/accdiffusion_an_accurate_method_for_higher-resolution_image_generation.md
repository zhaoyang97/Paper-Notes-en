---
title: >-
  [Paper Note] AccDiffusion: An Accurate Method for Higher-Resolution Image Generation
description: >-
  [ECCV 2024][Image Generation][High-Resolution Image Generation] This paper proposes AccDiffusion, which decouples global text prompts into patch-level content-aware prompts (utilizing cross-attention maps to determine whether each word belongs to a specific patch) and introduces dilated sampling with window interaction to improve global consistency. Without requiring extra training, this approach effectively solves the object duplication issue in patch-wise high-resolution im…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "High-Resolution Image Generation"
  - "Patch-wise Denoising"
  - "Cross-Attention Guidance"
  - "Object Duplication Elimination"
  - "Training-Free"
date: 2026-05-08
content_hash: b12baef5f3ad639c
---

# AccDiffusion: An Accurate Method for Higher-Resolution Image Generation

**Conference**: ECCV 2024  
**arXiv**: [2407.10738](https://arxiv.org/abs/2407.10738)  
**Code**: [https://github.com/lzhxmu/AccDiffusion](https://github.com/lzhxmu/AccDiffusion)  
**Area**: Image Generation / Diffusion Models / High-Resolution Generation  
**Keywords**: High-Resolution Image Generation, Patch-wise Denoising, Cross-Attention Guidance, Object Duplication Elimination, Training-Free  

## TL;DR
This paper proposes AccDiffusion, which decouples global text prompts into patch-level content-aware prompts (utilizing cross-attention maps to determine whether each word belongs to a specific patch) and introduces dilated sampling with window interaction to improve global consistency. Without requiring extra training, this approach effectively solves the object duplication issue in patch-wise high-resolution image generation, achieving high-quality, duplication-free image extrapolation from 2K to 4K resolutions on SDXL.

## Background & Motivation
Diffusion models, such as Stable Diffusion, are extremely expensive to train (e.g., training SD 1.5 on 256 A100 GPUs takes over 20 days), which usually restricts their training resolution to $512^2$ (SD 1.5) or $1024^2$ (SDXL). However, real-world applications (such as advertising) have a strong demand for high-resolution images. Direct inference at high resolutions leads to severe object duplication and structural inconsistency.

Existing methods can be categorized into two groups: (1) **Direct generation methods** such as Attn-SF and ScaleCrafter, which modify attention scaling factors or convolutional receptive fields. However, their GPU memory usage grows drastically with resolution, and the generated image quality is often poor. (2) **Indirect/Patch-wise generation methods** such as MultiDiffusion and DemoFusion, which segment the high-resolution image into multiple patches, denoise them independently, and then fuse them. Although this approach keeps memory consumption manageable, it leads to severe object duplication. While DemoFusion partially mitigates this duplication by introducing global semantic information through residual connections and dilated sampling, it still generates small-scale duplicated objects at ultra-high resolutions.

## Core Problem
**Why does small object duplication occur in patch-wise high-resolution generation? How can this issue be thoroughly eliminated?**

Through in-depth ablation analysis, the paper identifies the root cause: small object duplication is an **adversarial result** between **applying the same text prompt to all patches** (which tends to repeatedly generate objects in each patch) and the **global semantics provided by residual connections and dilated sampling** (which suppresses duplicate generation). Removing the prompt eliminates duplication but degrades details; removing the global information leads to massive object duplication. Therefore, the key to the problem lies in the need to provide each patch with more precise prompts that match its content.

## Method

### Overall Architecture
Based on the progressive upsampling pipeline of DemoFusion, AccDiffusion is divided into two phases:
1. **Phase 1**: Generate a low-resolution image at the pre-trained resolution (e.g., $1024^2$), while extracting cross-attention maps.
2. **Phase 2**: Progressively upsample the image to higher resolutions. In each upsampling stage, patch-wise denoising is first applied to generate local details, followed by dilated sampling with window interaction to enhance global consistency. Finally, residual connections are used to inject low-resolution structural information.

The core innovations reside in Phase 2: replacing the uniform prompt with patch-content-aware prompts, and improving the quality of dilated sampling with window interaction.

### Key Designs

1. **Patch-Content-Aware Prompts (Core Contribution)**: This technique utilizes the cross-attention maps of the U-Net from Phase 1 to automatically determine which region of the image each word belongs to. Specifically, the column-wise mean of the cross-attention map $\mathcal{M}$ is used as a threshold to binarize it into a mask $\mathcal{B}$. A morphological opening operation (erosion followed by dilation) is performed on the binary mask to eliminate small connected-component noise. After upsampling the mask to the target high resolution, a sliding window is used to crop the regional mask corresponding to each patch. Based on whether the proportion of high-response regions of each word within the patch area exceeds a threshold $c$, the word is assigned to the prompt of that patch. In this way, each patch obtains a content-matched sub-prompt, preventing the forced generation of an object in patches where it does not belong.

2. **Dilated Sampling with Window Interaction**: The dilated sampling in DemoFusion denoises each sampled subset independently, which results in un-smooth and noisy global semantic information due to the lack of interaction between different subsets. In AccDiffusion, before each denoising step, a position-dependent bijection function is used to swap noises between different dilated samples within the same window, allowing them to influence each other during the denoising process. After denoising, an inverse mapping is used to restore the original positions. This window interaction makes the global semantic information produced by dilated sampling smoother and more coherent.

3. **Adaptive Threshold Design**: The range of values of cross-attention maps varies significantly for different words (e.g., the mean for "Astronaut" is around 0.13, while "mars" is around 0.20). Using a fixed threshold would cause some words to be entirely included or excluded. Therefore, the average attention values of each word itself are used as an adaptive threshold to ensure that each word has a reasonable high-response area.

### Loss & Training
AccDiffusion is a completely **training-free**, plug-and-play method that requires no fine-tuning or extra training. It directly reuses pre-trained Stable Diffusion models (such as SDXL, SD 1.5, or SD 2.1) and only modifies the prompt assignment and dilated sampling strategies during inference. The hyperparameter $c=0.3$ controls the threshold for incorporating words into the patch prompt, and the dilated sampling weight $\eta$ is decreased from $1$ to $0$ following a cosine schedule.

## Key Experimental Results

| Resolution | Method | FID_r↓ | IS_r↑ | FID_c↓ | IS_c↑ | CLIP↑ | Time |
|--------|------|--------|-------|--------|-------|-------|------|
| $2048^2$ ($4\times$) | DemoFusion | 60.46 | 16.45 | 38.55 | 24.17 | 32.21 | 3min |
| $2048^2$ ($4\times$) | **AccDiffusion** | **59.63** | **16.48** | **38.36** | **24.62** | **32.79** | 3min |
| $3072^2$ ($9\times$) | DemoFusion | 62.43 | 16.41 | 47.45 | 20.42 | 32.25 | 11min |
| $3072^2$ ($9\times$) | **AccDiffusion** | **61.40** | **17.02** | **46.46** | **20.77** | **32.82** | 11min |
| $4096^2$ ($16\times$) | DemoFusion | 65.97 | 15.67 | 59.94 | 16.60 | 33.21 | 25min |
| $4096^2$ ($16\times$) | **AccDiffusion** | **63.89** | **16.05** | **58.51** | **16.72** | **33.79** | 26min |

Across all resolutions, AccDiffusion outperforms DemoFusion and other methods (SDXL-DI, Attn-SF, MultiDiffusion, ScaleCrafter) on all metrics, with almost the same inference time as DemoFusion.

### Ablation Study
- **Complementarity of the two core modules**: Removing the patch-content-aware prompt leads to a large number of duplicated small objects. Removing the window interaction results in generating small objects that are semantically irrelevant to the image. Removing both causes the most severe duplication, whereas using both completely eliminates duplication.
- **Sensitivity of threshold $c$**: If $c$ is too small (e.g., 0.1), too many words are included in the patch prompt, still causing duplication. If $c$ is too large (e.g., 0.9), it oversimplifies the prompts, leading to detail degradation. $c=0.3$ is a suitable balance point, though it can be adjusted for specific scenarios.
- **Adaptive threshold outperforms fixed threshold**: The value ranges of attention maps of different words vary significantly. Using a mean-based adaptive threshold is more robust than a fixed threshold.

## Highlights & Insights
- **Precise Root-Cause Analysis**: Through ablation experiments, the paper deeply reveals the root cause of object duplication in patch-wise generation—the guiding effect of the uniform prompt on all patches. This finding itself is highly valuable.
- **No External Models Required**: It directly utilizes the cross-attention maps of the diffusion model itself to determine patch content, without introducing extra segmentation models like SAM, which is elegant and efficient.
- **Plug-and-play**: Completely training-free, applicable to multiple Stable Diffusion variants such as SD1.5, SD2.1, and SDXL.
- **Denoising via Morphological Operations**: It cleverly utilizes the morphological opening operation to eliminate small connected-component noise in the attention map, which is a highly reusable trick.
- **Bijection Design for Window Interaction**: It introduces interactions between dilated samples through time-and-position-dependent bijection functions, which maintains theoretical invertibility while improving global consistency.

## Limitations & Future Work
- **Inference Latency Not Improved**: Inheriting the progressive upsampling and overlapping patch denoising strategy of DemoFusion, the inference time increases rapidly with resolution (taking ~25 minutes for 4K).
- **Dependence on Pre-trained Model Quality**: The fidelity of high-resolution images is limited by the capabilities of the underlying diffusion model.
- **Degradation at Extreme High Resolutions**: Detail degradation occurs when exceeding 6K ($36\times$), indicating an upper limit on the accuracy of the cross-attention map guidance.
- **Unreasonable Local Content**: Relying on the LDM's prior knowledge of cropped images, it may generate unreasonable local content during extreme close-up generations.
- **Unexplored Non-overlapping Patch Denoising**: As the paper points out, non-overlapping patch-wise denoising is a potential direction to improve efficiency.

## Related Work & Insights
- **vs DemoFusion (CVPR 2024)**: Both are patch-wise methods. DemoFusion introduces residual connections and dilated sampling but still suffers from small object duplication. AccDiffusion addresses the root cause by using patch-content-aware prompts to completely resolve the duplication issue without increasing inference time.
- **vs MultiDiffusion (ICML 2023)**: MultiDiffusion only performs basic overlapping patch fusion without global semantic information, leading to severe duplication and distortion. AccDiffusion builds upon it by introducing finer prompt control and global information enhancement.
- **vs ScaleCrafter (ICLR 2024)**: A direct generation method that modifies the convolution receptive field, but its GPU memory increases rapidly with resolution, and it still suffers from structural distortion. AccDiffusion avoids memory bottlenecks through a patch-wise approach.
- **The utilization of Cross-Attention Maps is worth learning from**: The paper cleverly transfers cross-attention maps from the domain of Prompt-to-Prompt image editing to high-resolution generation to automatically determine patch-prompt correspondences. This idea of using the diffusion model's internal representations to guide generation control has broad applicability.
- **Indirectly related to process-aware alignment**: AccDiffusion guides subsequent generation by analyzing the intermediate states of the diffusion process (attention maps). This "process-aware" concept shares commonalities with process-aware alignment.
- **Morphological post-processing trick can be transferred**: In binarization/thresholding scenarios of attention maps or feature maps, using mathematical morphological opening to remove small connected components is a general and effective post-processing tool.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach to solving the problem is clear and unique (fundamentally solving duplication from the prompt level), but the overall framework is still based on DemoFusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation experiments sufficiently reveal the contribution of each module, combining both quantitative and qualitative analyses, though FID/IS metrics struggle to fully reflect the degree of duplication.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, tightly linking problem analysis to solution design with beautiful and easy-to-understand diagrams.
- Value: ⭐⭐⭐⭐ Training-free and plug-and-play, with high practical value, though inference speed remains a bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](../../NeurIPS2025/image_generation/scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)

</div>

<!-- RELATED:END -->
