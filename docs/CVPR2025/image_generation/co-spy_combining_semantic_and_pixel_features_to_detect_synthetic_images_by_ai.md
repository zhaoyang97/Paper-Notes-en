---
title: >-
  [Paper Note] Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI
description: >-
  [CVPR 2025][Image Generation][AI-generated image detection] Co-Spy is proposed to fuse two complementary detection pathways: VAE reconstruction artifact features and CLIP semantic features. VAE artifacts generalize well across models but are vulnerable to JPEG compression, while CLIP semantics are robust to JPEG compression but generalize poorly. An adaptive regulator dynamically allocates weights to the two pathways based on the input, establishing a new SOTA across 22 gener…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "AI-generated image detection"
  - "semantic-artifact fusion"
  - "VAE artifacts"
  - "CLIP classification"
  - "JPEG robustness"
date: 2026-05-08
content_hash: 61c7c05ee5e14fc8
---

# Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI

**Conference**: CVPR 2025  
**arXiv**: [2503.18286](https://arxiv.org/abs/2503.18286)  
**Code**: [https://github.com/Megum1/Co-Spy](https://github.com/Megum1/Co-Spy)  
**Area**: Image Generation  
**Keywords**: AI-generated image detection, semantic-artifact fusion, VAE artifacts, CLIP classification, JPEG robustness

## TL;DR
Co-Spy is proposed to fuse two complementary detection pathways: VAE reconstruction artifact features and CLIP semantic features. VAE artifacts generalize well across models but are vulnerable to JPEG compression, while CLIP semantics are robust to JPEG compression but generalize poorly. An adaptive regulator dynamically allocates weights to the two pathways based on the input, establishing a new SOTA across 22 generative models.

## Background & Motivation

**Background**: AI-generated image detection is categorized into artifact detection (capturing generator fingerprints) and semantic detection (analyzing high-level semantic unnaturalness). The former generalizes well but is sensitive to compression, while the latter is robust but generalizes poorly to unseen models.

**Limitations of Prior Work**: (1) Detectors based on upsampling artifacts fail under JPEG compression. (2) CLIP-based semantic detectors generalize poorly to generative models unseen during training. (3) Simple concatenation of the two types of features (e.g., DRCT) yields sub-optimal performance, as the two feature pathways need to be individually enhanced before fusion.

**Key Challenge**: The two types of detection methods possess complementary advantages but also fatal weaknesses. Direct fusion does not work; they must be individually enhanced first and then adaptively fused.

**Goal**: Individually enhance the artifact and semantic pathways, and then design an adaptive fusion mechanism to leverage their complementarity.

**Key Insight**: VAE reconstruction artifacts are higher-level and more robust than upsampling artifacts; soft-label interpolation in the CLIP feature space enhances generalization; regulator networks dynamically allocate weights to both pathways with random dropout to prevent overfitting.

**Core Idea**: Utilizing VAE reconstruction differences to extract high-level artifacts + CLIP soft-label interpolation to enhance semantics + adaptive fusion via a regulator. This three-step combination achieves dual robustness against both model generalization and JPEG compression.

## Method

### Overall Architecture
The input image is processed through two streams: (1) After VAE encoding-decoding, the absolute difference $|x'-x|$ with the original image is calculated, and an artifact encoder extracts artifact features; (2) A frozen OpenCLIP extracts semantic features, enhanced by soft-label interpolation. The features from both pathways are then dynamically weighted by a regulator and fused for final classification.

### Key Designs

1. **VAE Artifact Extraction**:

    - **Function**: More robust generator fingerprint detection compared to upsampling artifacts.
    - **Mechanism**: The image is passed through a pre-trained VAE (encode $\rightarrow$ decode) to obtain the reconstructed image $x'$, and the absolute difference $|x'-x|$ is calculated as the artifact map. Real and AI-generated images exhibit different difference patterns after reconstruction by the same VAE—for generated images, the VAE tends to "repair" generation artifacts, whereas real images are reconstructed uniformly.
    - **Design Motivation**: Upsampling artifacts (NPR, LNP) are lost after JPEG compression, whereas VAE artifacts are higher-level and much more robust—outperforming NPR/LNP by 17-37%.

2. **CLIP Soft-Label Interpolation**:

    - **Function**: Enhance the generalization capability of the semantic detector to novel models.
    - **Mechanism**: In the CLIP feature space, linear interpolation is performed between real image embeddings and generated image embeddings to create virtual training samples, with the soft labels interpolated accordingly. This expands the coverage of the semantic feature space.
    - **Design Motivation**: Improves generalization performance by more than 10% compared to directly using the CLIP detector without enhancement.

3. **Adaptive Regulator Fusion**:

    - **Function**: Dynamically allocate weights to the artifact/semantic pathways based on the input.
    - **Mechanism**: Two MLP regulators generate scaling coefficients $\alpha$ and $\beta$ for the artifact and semantic features, respectively. During training, one of the pathways is randomly dropped out to prevent the model from over-relying on a single stream. The final feature is calculated as $\alpha \cdot \text{artifact feature} + \beta \cdot \text{semantic feature}$.
    - **Design Motivation**: Automatically increases the semantic weight when the artifact pathway fails after JPEG compression, and increases the artifact weight when facing unseen models.

### Loss & Training
Binary cross-entropy loss. Co-SpyBench: 1M+ images, 22 generative models (including FLUX), 5 real-world datasets, and 50K in-the-wild data.

## Key Experimental Results

### Main Results

| Method | Average AP on 22 Models with JPEG | Average AP without JPEG |
|------|---------------------|-------------|
| NPR | ~65% | ~85% |
| DRCT | ~78% | ~92% |
| **Co-Spy** | **~89%** | **~93%** |

### Ablation Study

| Configuration | Effect |
|------|------|
| VAE artifact alone | Good cross-model generalization, but drops after JPEG |
| CLIP semantics alone | Robust to JPEG, but poor on new models |
| Simple concatenation of two pathways | Inferior to fusion after separate enhancement |
| **Enhancement before adaptive fusion** | **Optimal** |

### Key Findings
- **VAE Artifacts $\gg$ Upsampling Artifacts**: Under JPEG compression, VAE artifact detection outperforms NPR/LNP by 17-37%.
- **Enhancement + Fusion $>$ Simple Fusion**: Fusing after independent enhancement yields a performance boost that exceeds the sum of individual increments.
- **Co-SpyBench is the most comprehensive evaluation**: Incorporates 22 generative models (including the latest FLUX) + 50K in-the-wild images.

## Highlights & Insights
- **VAE reconstruction difference as artifacts** is an elegant concept, leveraging the commonality that all latent diffusion models pass through the VAE latent space.
- **The "enhance before fusion" three-step strategy** is generalizable to other scenarios requiring the fusion of complementary detectors.
- **The Co-SpyBench dataset** provides a much-needed comprehensive benchmark for the synthetic image detection community.

## Limitations & Future Work
- VAE artifacts assume that generated images undergo a VAE latent space—which may not apply to GAN-generated images that do not bypass a VAE.
- Training the regulator requires data for both artifact and semantic pathways simultaneously—which might be inconvenient in cold-start scenarios.
- Not yet extended to video synthesis detection.

## Related Work & Insights
- **vs DRCT**: DRCT also fuses multiple pathways but without individual enhancement. Co-Spy's "enhance before fusion" strategy achieves superior results.
- **vs NPR / LNP**: Conventional upsampling artifact methods. VAE artifacts are higher-level and more robust.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovations in both VAE artifact extraction and soft-label interpolation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 generative models, JPEG robustness, in-the-wild data, and Co-SpyBench.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis of complementarity.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for AI-generated content (AIGC) detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/image_generation/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](../../CVPR2026/image_generation/simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[CVPR 2025\] CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)

</div>

<!-- RELATED:END -->
