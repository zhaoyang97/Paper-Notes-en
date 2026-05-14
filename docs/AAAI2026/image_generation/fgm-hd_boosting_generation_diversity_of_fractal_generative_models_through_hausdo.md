---
title: >-
  [Paper Note] FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction
description: >-
  [AAAI 2026][Image Generation][Fractal Generative Model] This paper is the first to introduce Hausdorff Dimension (HD) into Fractal Generative Models (FGM), proposing a learnable HD estimation module…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Fractal Generative Model"
  - "Hausdorff Dimension"
  - "Generation Diversity"
  - "Momentum-driven Scheduling Strategy"
  - "Rejection Sampling"
date: 2026-05-08
content_hash: 70f419437664fcfd
---

# FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction

**Conference**: AAAI 2026
**arXiv**: [2511.08945](https://arxiv.org/abs/2511.08945)
**Code**: None
**Area**: Image Generation
**Keywords**: Fractal Generative Model, Hausdorff Dimension, Generation Diversity, Momentum-driven Scheduling Strategy, Rejection Sampling

## TL;DR
This paper is the first to introduce Hausdorff Dimension (HD) into Fractal Generative Models (FGM), proposing a learnable HD estimation module, a Monotonic Momentum-Driven Scheduling strategy (MMDS), and HD-guided rejection sampling. The method achieves a 39% improvement in generation diversity (Recall) on ImageNet while maintaining image quality.

## Background & Motivation
Generative models (GANs, VAEs, diffusion models, etc.) can produce high-fidelity images, yet balancing image quality and diversity remains a fundamental challenge. Fractal Generative Models (FGMs) leverage recursive self-similarity to efficiently generate high-quality images by repeatedly applying compact generative modules at multiple scales.

**Limitations of Prior Work**: The recursive self-similar structure of FGMs, while ensuring global consistency, also leads to repetitive patterns in generated outputs and insufficient diversity. This inherent self-similarity limits FGMs' ability to capture complex data distributions.

**Key Challenge**: The core advantage of FGMs—recursive self-similarity—is precisely the source of their diversity deficiency; diversity must be enhanced without disrupting structural consistency.

**Key Insight**: Hausdorff Dimension (HD) is borrowed from fractal geometry as a geometric indicator of structural complexity. HD quantifies the variation of spatial detail across scales, with higher HD values generally reflecting greater structural richness.

**Core Idea**: HD is used as a training signal and sampling criterion to guide FGMs toward generating structurally more complex and diverse outputs.

## Method

### Overall Architecture
The FGM-HD framework contains three key innovations: (1) a learnable HD estimation network that directly predicts HD from image embeddings; (2) the MMDS strategy to dynamically adjust HD loss weight during training; and (3) HD-guided rejection sampling at inference to retain high-HD outputs.

### Key Designs

1. **Learnable HD Estimation Module**:

    - **Function**: Directly predicts the Hausdorff Dimension from image embeddings, replacing the traditional box counting method.
    - **Mechanism**: Built on a ResNet152 backbone, with the last two layers replaced by a multi-scale convolutional module (parallel $3\times3$, $5\times5$, $7\times7$ convolutions) to capture spatial information at different scales, followed by a regression layer to output the HD value.
    - **Design Motivation**: Traditional box counting is computationally expensive ($4.70$ s/image) and noise-sensitive, whereas the proposed method requires only $0.32$ s/image with an error of only $0.005$ (vs. $0.002$ for box counting), achieving an ideal accuracy–efficiency trade-off.

2. **Monotonic Momentum-Driven Scheduling Strategy (MMDS)**:

    - **Function**: Dynamically adjusts the weight $\lambda(t)$ of the HD loss in the composite loss.
    - **Mechanism**: In early training, image quality is poor and HD estimation is unreliable, so $\lambda$ should be near zero; it increases gradually as training progresses. This is realized via a momentum-style accumulation scheme: $m \leftarrow \mu \cdot m + (1-\mu) \cdot \gamma \cdot \Delta L$, $\lambda \leftarrow \lambda + m$, where $\Delta L = \max(0, L_{\text{prev}} - L_{\text{val}})$.
    - **Design Motivation**: Using a fixed HD weight in the composite loss leads to (1) image quality degradation and (2) limited diversity improvement. MMDS ensures the model first focuses on quality before progressively incorporating the diversity objective.
    - **Total Loss**: $L_{\text{total}} = L_{\text{gen}} + \lambda(t) \cdot L_{\text{HD}}$, where $L_{\text{HD}} = |HD_{\text{gen}} - HD_{\text{target}}|$.

3. **HD-Guided Rejection Sampling**:

    - **Function**: Filters out low-HD (structurally simple) generated images at inference time.
    - **Mechanism**: After generating a batch of candidate images, the HD of each is estimated; only outputs with HD exceeding a threshold $\tau$ are retained, while low-HD samples are regenerated from scratch.
    - **Design Motivation**: FGMs' recursive structure naturally supports parallel generation of multiple candidates, and this post-processing step modifies neither the model architecture nor training cost.

### Loss & Training
- HD loss: $L_{\text{HD}} = |HD_{\text{gen}} - HD_{\text{target}}|$, where $HD_{\text{target}}$ is the class-specific median HD computed from the training set.
- MMDS parameters: $\mu = 0.9$, $\gamma = 1.0$ provide the best balance.
- Training is stopped after 1000 epochs; further training does not significantly improve quality or diversity.
- HD sampling threshold: $1.55$–$1.60$ provides the best quality–diversity trade-off.

## Key Experimental Results

### Main Results (ImageNet $256\times256$, pixel-level generation)

| Model | Type | FID↓ | IS↑ | Recall↑ |
|-------|------|------|-----|---------|
| StyleGAN-XL | GAN | 2.30 | 265.1 | 0.53 |
| DiffiT | Diffusion | 1.73 | 276.5 | 0.62 |
| RCG | MAGE | 2.15 | 253.4 | 0.53 |
| FGM (baseline) | Fractal | 6.15 | 348.9 | 0.46 |
| **FGM-HD (Ours)** | **Fractal** | **6.21** | **367.1** | **0.64** |

### Ablation Study

| Configuration | FID↓ | IS↑ | Recall↑ | LPIPS↑ |
|---------------|------|-----|---------|--------|
| FGM (baseline) | 6.15 | 348.9 | 0.46 | 0.64 |
| + Fixed HD Loss only | 6.22 | 333.7 | 0.47 | 0.65 |
| + MMDS only | 6.04 | 361.7 | 0.51 | 0.69 |
| + HD Sampling only | 6.78 | 357.9 | 0.58 | 0.73 |
| **+ MMDS & Sampling** | **6.21** | **367.4** | **0.64** | **0.76** |

### HD Estimation Method Comparison

| Method | Type | Error↓ | Time (s)↓ |
|--------|------|--------|-----------|
| Box Counting | Non-learning | 0.002 | 4.70 |
| Power Spectrum | Non-learning | 0.079 | 3.41 |
| ResNet152 | Learning | 0.012 | 0.40 |
| **Ours** | **Learning** | **0.005** | **0.32** |

### Key Findings
- **Fixed HD loss is ineffective**: Directly adding the HD loss yields only marginal Recall improvement ($0.46 \to 0.47$), while IS drops from $348.9$ to $333.7$ and FID increases.
- **MMDS and sampling are complementary**: MMDS provides progressive optimization during training, while sampling provides post-processing filtering at inference; their combination achieves the best results.
- **Kernel selection for multi-scale convolution**: The $3\times3 + 5\times5 + 7\times7$ combination is optimal (loss $0.005$); single-kernel configurations are significantly inferior.
- **Layer replacement in ResNet152**: Replacing Stages 4–5 is optimal (error $0.005$); replacing more layers yields faster inference but substantially lower accuracy.
- **MMDS $\lambda$ begins to increase around epoch 300 and stabilizes after epoch 800**, corresponding to the transition of image quality from chaotic to stable.

## Highlights & Insights
- **Theoretical Contribution**: This is the first work to introduce Hausdorff Dimension from fractal geometry into generative model diversity enhancement, providing a mathematically grounded diversity metric.
- **Generalizability of MMDS**: The momentum-driven loss weight scheduling strategy is transferable to any model employing composite loss functions.
- **39% Recall Improvement**: Coverage of the generated distribution is substantially increased while image quality is maintained.
- **No additional training cost at inference**: Rejection sampling only increases inference time without modifying the model.

## Limitations & Future Work
- The FGM baseline FID ($6.15$) remains far above that of diffusion models ($1.73$), indicating a clear gap in overall generation quality.
- The HD estimator is trained on data annotated via box counting, which may introduce an annotation accuracy bottleneck.
- Validation is limited to $256\times256$ ImageNet; higher resolutions and conditional generation settings remain untested.
- Rejection sampling increases inference time, and high thresholds may cause frequent regeneration.
- The relationship between HD and other diversity metrics (e.g., mode coverage indicators in FID) has not been explored.
- The method has not been validated on conditional or multimodal generative models.

## Related Work & Insights
- **Hausdorff GAN**: Li et al. proposed using HD to align the intrinsic dimensionality of real and generated data; this paper instead focuses on diversity enhancement in FGMs.
- **Unique value of FGMs**: The recursive self-similar structure makes FGMs particularly well-suited for tasks requiring both structural consistency and visual richness.
- **Quality–diversity trade-off**: The methodological principle of progressively introducing diversity signals can inspire other generative tasks facing similar trade-offs.
- **Inspiration from MMDS**: For any training system requiring the balancing of multiple loss terms, momentum-driven scheduling provides a more robust alternative to fixed exponential or linear schedules.

## Rating
- Novelty: ⭐⭐⭐⭐ (HD + FGM combination is entirely novel; MMDS has broad applicability)
- Experimental Thoroughness: ⭐⭐⭐⭐ (thorough ablations with detailed analysis of HD estimation and scheduling strategies)
- Writing Quality: ⭐⭐⭐⭐ (clear structure with well-motivated method descriptions)
- Value: ⭐⭐⭐ (the limited baseline performance of FGMs constrains the practical impact of the method)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Boosting Generative Image Modeling via Joint Image-Feature Synthesis](../../NeurIPS2025/image_generation/boosting_generative_image_modeling_via_joint_imagefeature_sy.md)
- [\[ICLR 2026\] GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models](../../ICLR2026/image_generation/geodiv_framework_for_measuring_geographical_diversity_in_text-to-image_models.md)
- [\[ICLR 2026\] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](../../ICLR2026/image_generation/the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](../../CVPR2026/image_generation/diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)

</div>

<!-- RELATED:END -->
