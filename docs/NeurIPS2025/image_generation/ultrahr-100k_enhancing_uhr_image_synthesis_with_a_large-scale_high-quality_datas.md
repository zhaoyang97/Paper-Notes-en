---
title: >-
  [Paper Note] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset
description: >-
  [NeurIPS 2025][Image Generation][Ultra-High-Resolution] This work constructs UltraHR-100K, a large-scale dataset comprising 100K ultra-high-resolution images with rich annotations, and proposes a Frequency-Aware Post-Training (FAPT) method combining Detail-Oriented Timestep Sampling (DOTS) and Soft-Weighted Frequency Regularization (SWFR) based on DFT, enabling pretrained T2I models to generate fine-grained details at ultra-high resolutions.
tags:
  - NeurIPS 2025
  - Image Generation
  - Ultra-High-Resolution
  - Dataset
  - Frequency-Aware
  - Detail Generation
  - Post-Training
date: 2026-05-08
content_hash: e497081b18560654
---

# UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset

**Conference**: NeurIPS 2025
**arXiv**: [2510.20661](https://arxiv.org/abs/2510.20661)
**Code**: Available (marked in paper)
**Area**: Ultra-High-Resolution Image Generation / Diffusion Models
**Keywords**: Ultra-High-Resolution, Dataset, Frequency-Aware, Detail Generation, Post-Training

## TL;DR

This work constructs UltraHR-100K, a large-scale dataset comprising 100K ultra-high-resolution images with rich annotations, and proposes a Frequency-Aware Post-Training (FAPT) method combining Detail-Oriented Timestep Sampling (DOTS) and Soft-Weighted Frequency Regularization (SWFR) based on DFT, enabling pretrained T2I models to generate fine-grained details at ultra-high resolutions.

## Background & Motivation

Text-to-image (T2I) diffusion models perform well at 1024×1024 resolution but suffer from significant quality degradation and structural artifacts when directly scaled to ultra-high resolutions (UHR, e.g., 4K). Existing approaches fall into two categories:

**Training-free methods** (DemoFusion, HiFlow, etc.): Achieve UHR generation by modifying inference strategies, but tend to produce over-smoothed results with unrealistic details and long inference times.

**Training-based methods** (PixArt-σ, SANA, etc.): Focus primarily on training efficiency while neglecting detail generation quality.

Two core challenges remain unresolved: (1) the lack of large-scale, high-quality open-source UHR T2I datasets — the existing Aesthetic-4K contains only ~10K images without rigorous filtering; and (2) the absence of training strategies targeting fine-grained UHR detail synthesis — current models exhibit strong semantic planning capacity but inadequate high-frequency detail generation at UHR scales.

## Method

### Overall Architecture

This work comprises two components:
1. **UltraHR-100K Dataset**: 100K carefully curated UHR images with rich textual annotations.
2. **Frequency-Aware Post-Training (FAPT)**: A two-stage training strategy — Stage 1 fine-tunes on UltraHR-100K to enhance semantic planning; Stage 2 applies DOTS + SWFR to focus on high-frequency detail learning.

### Key Designs

1. **UltraHR-100K Dataset Construction**:

    - **Data Collection**: A Scrapy-based crawler collects approximately 400K high-resolution images (minimum 3840×2160).
    - **Preliminary Filtering**: Laplacian variance (sharpness) + Sobel operator (edge density) to remove blurry and texture-free images.
    - **Three-Dimensional Fine Filtering**:
        - **Detail Richness**: GLCM (Gray-Level Co-occurrence Matrix)-based contrast, entropy, and correlation; top 50% retained.
        - **Content Complexity**: Shannon entropy measuring pixel intensity diversity; top 50% retained.
        - **Aesthetic Quality**: LAION Aesthetic Predictor scoring; top 50% retained.
    - **Final Dataset** = **intersection** of the three subsets: UltraHR-100K = $S_G \cap S_E \cap S_A$, ensuring every image simultaneously meets all three high standards.
    - **Annotation**: Gemini 2.0 generates detailed long captions covering global summaries and fine-grained descriptions, with annotation length substantially exceeding that of Aesthetic-4K.
    - Final scale: 104,117 images, average height 3648, average width 5119.

2. **Detail-Oriented Timestep Sampling (DOTS)**:

    - Observation: Early denoising steps primarily reconstruct low-frequency structure, while later steps progressively synthesize high-frequency details.
    - Denoising timesteps are sampled from a Beta($\alpha, \beta$) distribution with $\alpha=2, \beta=4$, biasing sampling toward later steps (near $t=0$).
    - Effect: Guides the model to focus on high-frequency detail-related denoising steps during post-training.

3. **Soft-Weighted Frequency Regularization (SWFR)**:

    - 2D DFT is applied to both predictions and targets, with weighted constraints imposed in the frequency domain.
    - Frequency soft-weighting function: $w(r) = 1 + \lambda \cdot \frac{\exp(\gamma r)-1}{\exp(\gamma)-1}$, where $r \in [0,1]$ is the normalized frequency distance.
    - $\lambda$ and $\gamma$ control the intensity and steepness of high-frequency emphasis.
    - $\mathcal{L}_{\text{freq}} = \mathbb{E}\left[\|w(r)\cdot\hat{x} - w(r)\cdot\hat{y}\|^2\right]$
    - Total loss: $\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{freq}} \cdot \mathcal{L}_{\text{freq}}$
    - Provides finer, continuous frequency separation compared to DWT-based approaches (as used in Diffusion4K).

### Loss & Training

- Two-stage training: Stage 1 — 4K steps with Logit-Normal sampling fine-tuning; Stage 2 — 8K steps with DOTS + SWFR.
- CAMEWrapper optimizer, constant learning rate $1\times10^{-4}$, mixed-precision training, batch size 24.
- Built on the SANA model; trained on 4× H20 GPUs.
- Evaluation benchmark: self-constructed UltraHR-eval4K (2,000 images at 4096×4096).

## Key Experimental Results

### Main Results (UltraHR-eval4K, 4096×4096)

| Method | FID↓ | FID_patch↓ | IS↑ | IS_patch↑ | CLIP↑ | FG-CLIP↑ |
|---|---|---|---|---|---|---|
| FLUX + BSRGAN | 37.65 | 43.14 | 11.77 | 5.39 | 31.45 | 28.02 |
| I-Max(FLUX) | 37.67 | 37.84 | 11.99 | 4.39 | 31.49 | 27.78 |
| HiFlow(FLUX) | 35.89 | 38.33 | 11.77 | 4.62 | 31.52 | 27.75 |
| PixArt-σ | 33.17 | 32.20 | 12.21 | 5.39 | 31.78 | 28.65 |
| SANA | 37.07 | 38.80 | 11.78 | 5.65 | 31.70 | 28.60 |
| Diffusion4K | 39.86 | 38.52 | 10.83 | 3.24 | 31.41 | 26.48 |
| **Ours (UltraHR-100K)** | 34.00 | 20.93 | 12.50 | 5.02 | 31.85 | 28.65 |
| **Ours (+FAPT)** | **31.75** | **15.80** | **13.00** | 5.10 | 31.82 | **28.68** |

### Ablation Study

| Model | DOTS | SWFR | Data | FID↓ | FID_patch↓ | CLIP↑ |
|---|---|---|---|---|---|---|
| LoRA | × | × | Full | 35.07 | 35.02 | 31.80 |
| A (full fine-tuning baseline) | × | × | Full | 33.99 | 20.93 | 31.85 |
| B | ✓ | × | Full | 32.57 | 19.95 | 31.79 |
| C | ✓ | ✓ | 15K | 32.75 | 18.42 | 31.81 |
| **D (full method)** | ✓ | ✓ | Full | **31.74** | **15.79** | **31.82** |

### Key Findings

- FID_patch improvement is the most pronounced (38.80→15.80), demonstrating the method's advantage in fine-grained detail generation.
- In user studies, the proposed method achieves a 70% overall preference rate and a 78% detail quality preference rate, substantially outperforming competing methods.
- Data scale is critical: the full 100K dataset yields significantly better results than the 15K subset.
- Beta($\alpha=2, \beta=4$) is optimal for DOTS; excessively large $\alpha$ weakens detail learning, while excessively small $\alpha$ harms semantic consistency.
- SWFR contributes most significantly to FID_patch improvement (19.95→15.79), validating the effectiveness of high-frequency regularization.
- The proposed method also achieves state-of-the-art results on the public Aesthetic-Eval@4096 benchmark, demonstrating generalizability.

## Highlights & Insights

- **Three-dimensional intersection filtering is concise yet effective**: The intersection design across detail richness, content complexity, and aesthetic quality enforces a high quality floor for the dataset.
- **DOTS exploits the frequency characteristics of the denoising process**: It precisely leverages the physical property that early steps generate structure while later steps generate details.
- **The choice of DFT over DWT is well-motivated**: DFT provides a continuous spectrum that enables finer-grained control compared to the coarse discrete decomposition of DWT.
- **The post-training paradigm is broadly applicable**: It demonstrates that significant UHR detail capability can be unlocked through lightweight post-training without retraining from scratch.
- The contribution of a 100K-scale UHR dataset is itself of considerable value and is expected to advance the broader UHR generation community.

## Limitations & Future Work

- Frequency-aware post-training slightly reduces text-image alignment (marginal CLIP score decrease).
- The dataset contains relatively few portrait images, leaving room for improvement in UHR portrait generation.
- Training and evaluation are conducted solely on SANA; generalization to other mainstream models such as FLUX and SD3 remains unverified.
- Computational constraints: 4× H20 GPUs.

## Related Work & Insights

- Compared to Diffusion4K (DWT-based frequency decomposition), the proposed DFT + soft-weighting approach substantially outperforms across all metrics.
- Compared to PixArt-σ (efficient token compression) and SANA (efficient 4K pipeline), the proposed method prioritizes detail quality over computational efficiency.
- Insight: UHR generation requires not only efficient architectures but also high-quality data and detail-oriented training strategies.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](../../ICCV2025/image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[NeurIPS 2025\] RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation](repldm_reprogramming_pretrained_latent_diffusion_models_for_high-quality_high-ef.md)

<!-- RELATED:END -->
