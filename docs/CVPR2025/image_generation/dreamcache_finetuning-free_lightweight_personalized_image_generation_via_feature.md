---
title: >-
  [Paper Note] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching
description: >-
  [CVPR 2025][Image Generation][Personalized Generation] DreamCache is proposed to achieve tuning-free, encoder-free, and plug-and-play personalized image generation by caching the intermediate U-Net features of the reference image at a single denoising step ($t=1$) and injecting these cached features during generation using a lightweight 25M-parameter conditional adapter.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Personalized Generation"
  - "Feature Caching"
  - "Tuning-Free"
  - "Plug-and-Play"
  - "Lightweight Adaptation"
date: 2026-05-08
content_hash: 56ce0a10e3bb7fc4
---

# DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching

**Conference**: CVPR 2025  
**arXiv**: [2411.17786](https://arxiv.org/abs/2411.17786)  
**Code**: Yes (Project Page)  
**Area**: Image Generation  
**Keywords**: Personalized Generation, Feature Caching, Tuning-Free, Plug-and-Play, Lightweight Adaptation

## TL;DR
DreamCache is proposed to achieve tuning-free, encoder-free, and plug-and-play personalized image generation by caching the intermediate U-Net features of the reference image at a single denoising step ($t=1$) and injecting these cached features during generation using a lightweight 25M-parameter conditional adapter.

## Background & Motivation

**Background**: Personalized image generation synthesizes new images that maintain a specific identity or style from reference images and text prompts. Current methods are divided into tuning-based (e.g., DreamBooth, which requires per-user training) and tuning-free methods (e.g., IP-Adapter, which requires CLIP/BLIP encoders).

**Limitations of Prior Work**: (1) Tuning-based methods require several minutes of retraining for every new subject, making them unsuitable for real-time applications. (2) Tuning-free methods, though eliminating fine-tuning, require large external encoders (IP-Adapter 402M, BLIP-Diffusion 380M), which have high parameter counts and are not plug-and-play. (3) Global features extracted by encoders may lose fine-grained visual details.

**Key Challenge**: Extracting rich visual information from reference images for personalization without introducing massive external encoders or requiring fine-tuning.

**Goal**: Extract personalized information from reference images in an extremely lightweight manner—requiring no encoders, no fine-tuning, and offering plug-and-play capability.

**Key Insight**: The pre-trained diffusion model's U-Net is itself a powerful feature extractor. By performing a single forward pass on the reference image at the least noisy timestep ($t=1$) and caching the intermediate features, a rich multi-level visual representation can be obtained. Training a small adapter to inject these cached features into the generation process then suffices.

**Core Idea**: Utilizing the pre-trained U-Net's own features at $t=1$ as the reference representation (caching) and injecting them into the generation process via a lightweight 25M adapter, thereby achieving tuning-free, plug-and-play personalization.

## Method

### Overall Architecture
Reference image $\rightarrow$ single U-Net forward pass at $t=1$ with a null prompt $\rightarrow$ caching the intermediate bottleneck layer and every other decoder layer features $\rightarrow$ conditional adapter (attention-based) injecting cached features into the corresponding layers during generation $\rightarrow$ personalized image output.

### Key Designs

1. **Single-Step Feature Caching**:

    - **Function**: Extracting multi-level features from reference images at zero extra cost.
    - **Mechanism**: Performing a single U-Net forward pass on the reference image at $t=1$ (the cleanest timestep) with a null text prompt. The activations of the intermediate bottleneck layer and every other decoder layer are cached. These features contain multi-level information ranging from low-level textures to high-level semantics.
    - **Design Motivation**: (1) $t=1$ is the least noisy timestep, providing the cleanest features. (2) The null prompt decouples visual content from text, leading to better generalization. (3) Only a single forward pass is required, incurring almost zero overhead.

2. **Conditional Adapter (25M)**:

    - **Function**: Injecting cached features into the denoising process.
    - **Mechanism**: An attention-based lightweight module, with one adapter assigned to each corresponding layer. Taking the cached features as input, it injects them into the corresponding layers of the denoising U-Net via cross-attention.
    - **Design Motivation**: (1) With only 25M parameters, it is $1/16$ the size of IP-Adapter (402M). (2) Plug-and-play capability—the model reverts to the original non-personalized mode when the adapter is removed. (3) Only the adapter is trained while the U-Net remains frozen.

3. **Synthetic Training Data**:

    - **Function**: Generating (prompt, target image, reference image) triplets without manual annotation.
    - **Mechanism**: Automatically generating training pairs containing the same subject with different viewpoints/backgrounds, solely for training the adapter.
    - **Design Motivation**: To avoid the difficulty of collecting real personalized data.

### Loss & Training
Standard diffusion denoising loss, training only the adapter parameters. The U-Net is frozen completely. Training takes 40 hours (vs. 28 days for IP-Adapter).

## Key Experimental Results

### Main Results

| Method | Tuning-Free | Encoder-Free | Plug-and-Play | Extra Params | Training Time |
|------|--------|---------|---------|---------|---------|
| IP-Adapter | ✓ | ✗ | ✓ | 402M | 28 days |
| BLIP-Diffusion | ✓ | ✗ | ✓ | 380M | 96 days |
| BootPig | ✓ | ✓ | ✗ | 0.95B | 18 hours |
| **DreamCache** | **✓** | **✓** | **✓** | **25M** | **40 hours** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Caching Bottleneck + Alternate Decoders | Optimal |
| $t=1$ vs. Other Timesteps | $t=1$ is the cleanest and best |
| Null Prompt vs. With Prompt | Null is better (decoupling vision/text) |

### Key Findings
- **SOTA Achieved with 25M Parameters**: Achieving optimal image alignment and text alignment with only $1/16$ the parameters of IP-Adapter.
- **True Trinity of "Free" (Tuning-Free, Encoder-Free, and Plug-and-Play)**: The only method that simultaneously satisfies all three conditions.
- **Single-Step Caching is Sufficient**: The reference image does not need to be processed at every step—cache once, use for the entire generation.

## Highlights & Insights
- **"The U-Net itself is the best feature extractor"**—Without requiring an external CLIP/BLIP encoder, using features extracted by the denoising U-Net at a clean timestep is both sufficient and naturally aligned with the generation process.
- The **plug-and-play** property is highly valuable for practical deployment—the same base model can freely toggle between personalized and non-personalized modes.
- **Training Efficiency**: 40 hours vs. 28 days for IP-Adapter, a $16\times$ reduction.

## Limitations & Future Work
- The cached features originate from a fixed pre-trained U-Net—switching to a new base model requires retraining the adapter.
- Identity preservation with a single reference image might not perform as well as tuning-based methods with multiple reference images.
- Validated only on SD1.5; performance on SDXL/FLUX is unexplored.

## Related Work & Insights
- **vs. IP-Adapter**: Requires a CLIP encoder (402M). DreamCache is more lightweight by using U-Net's own features (25M adapter).
- **vs. BootPig**: Not plug-and-play. DreamCache reverts to the original model when the adapter is removed.
- **vs. DreamBooth**: Requires fine-tuning for each subject. DreamCache is trained once and applies to all subjects.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The concept of "U-Net self-caching" is elegant and profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-method comparisons and attribute analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear analysis of the three independent "free" properties.
- **Value**: ⭐⭐⭐⭐⭐ Paradigm-shifting value for personalized generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation](patchdpo_patch-level_dpo_for_finetuning-free_personalized_image_generation.md)
- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2025\] ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)
- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)

</div>

<!-- RELATED:END -->
