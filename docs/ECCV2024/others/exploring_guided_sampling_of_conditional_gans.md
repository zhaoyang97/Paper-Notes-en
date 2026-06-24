---
title: >-
  [Paper Note] Exploring Guided Sampling of Conditional GANs
description: >-
  [ECCV 2024][Conditional GAN] This paper proposes introducing a diffusion-like guided sampling strategy into conditional GANs. By estimating the joint data-condition distribution through latent space vector operations, without the need for pre-trained classifiers or learning unconditional models, it significantly improves GAN generation quality, reducing the FID on ImageNet 64×64 from 8.87 to 4.37.
tags:
  - "ECCV 2024"
  - "Conditional GAN"
  - "Guided Sampling"
  - "Latent Space Manipulation"
  - "Image Generation Quality"
  - "Diffusion Model Comparison"
date: 2026-05-08
content_hash: 5478ee4bc09dc316
---

# Exploring Guided Sampling of Conditional GANs

**Conference**: ECCV 2024  
**Code**: [GitHub](https://github.com/zyf0619sjtu/GANdance)  
**Area**: Others  
**Keywords**: Conditional GAN, Guided Sampling, Latent Space Manipulation, Image Generation Quality, Diffusion Model Comparison

## TL;DR

This paper proposes introducing a diffusion-like guided sampling strategy into conditional GANs. By estimating the joint data-condition distribution through latent space vector operations, without the need for pre-trained classifiers or learning unconditional models, it significantly improves GAN generation quality, reducing the FID on ImageNet 64×64 from 8.87 to 4.37.

## Background & Motivation

1. **Background**: Guided sampling is a widely used inference technique in diffusion models that allows for trading off generation fidelity and diversity. Classifier guidance requires a pre-trained classifier, while classifier-free guidance requires training both conditional and unconditional generative models. While highly successful in diffusion models, these strategies have not been systematically applied to GANs.

2. **Limitations of Prior Work**: Although existing GANs offer fast generation (single-step inference), their FID scores on large-scale conditional generation tasks like ImageNet remain higher than those of one-step diffusion models. No prior research has systematically explored the feasibility of translating guided sampling strategies into the GAN framework. Guided sampling in diffusion models relies on gradient signals during the denoising process, making it difficult to directly apply the same approach to the single-step generation paradigm of GANs.

3. **Key Challenge**: Guided sampling in diffusion models requires modifying the iterative denoising process, whereas GANs are single-step generation models. How to achieve a similar guiding effect without altering the GAN inference paradigm is the core challenge. Additionally, both classifier guidance and classifier-free guidance incur extra model training overhead; thus, how to achieve guided sampling at a lower cost is also a key issue.

4. **Goal**: How to implement guided sampling in conditional GANs to improve generation quality—especially closing the performance gap between GANs and one-step diffusion models—without significantly increasing inference costs.

5. **Key Insight**: Leverage the highly structured latent space of GANs to estimate the joint data-condition distribution using simple vector operations. GAN latent spaces possess a natural semantic structure where generation results under different conditions present organized distributions, making it possible to achieve conditional enhancement or attenuation through vector addition and subtraction.

6. **Core Idea**: Achieve guided sampling solely through vector operations in the GAN latent space without requiring classifiers or unconditional models, dramatically improving the generation quality of conditional GANs.

## Method

### Overall Architecture

The GANdance framework is built upon pre-trained conditional GANs. The core mechanism is to enhance the influence of conditional signals by manipulating latent space vectors within the already trained conditional generator. Specifically, given a class condition $c$ and random noise $z$, a traditional GAN directly generates $G(z, c)$. GANdance constructs a guidance signal by estimating the ratio of the conditional probability $p(x|c)$ to the marginal distribution $p(x)$, thereby adjusting the generation process to bias toward samples that better match the condition.

The overall method comprises two variants: (1) a training-free, plug-and-play version based on vector operations that directly exploits the structured properties of the latent space; (2) a learning-based improved version that trains an additional lightweight module to model the dataset distribution more precisely.

### Key Designs

1. **Joint Distribution Estimation via Latent Vector Operations**: This is the core of the method. GANdance leverages the regularity of the conditional GAN latent space to estimate the direction of conditional signals by calculating the difference in generation results of the same noise under different conditions. Specifically, given noise $z$ and a target condition $c$, extrapolation is performed along the condition direction in the latent space to enhance the condition's impact on the generation results. This operation resembles the effect of classifier-free guidance in diffusion models: increasing the weight of the conditional signal. Since the GAN latent space already possesses a well-defined semantic structure, this simple vector operation can effectively improve generation fidelity. This strategy reduces the FID from 8.87 to 6.06 with almost zero increase in inference time.

2. **Learning-based Distribution Approximation Module**: Although the vector operation version is simple and effective, its modeling of the entire dataset distribution is coarse. To approximate the conditional distribution of the dataset more accurately, the authors propose training a lightweight network to learn the mapping from the latent space to conditional probabilities. This module can better capture the statistical relationships among different categories in the dataset, thereby generating samples closer to the real data distribution. This learning-based version further reduces the FID from 6.06 to 4.37.

3. **Guidance Scale Control**: Similar to the guidance scale parameter in diffusion models, GANdance introduces a hyperparameter to control the guidance intensity. A lower guidance scale maintains diversity with limited fidelity, while a higher guidance scale increases fidelity but may sacrifice diversity. By adjusting this parameter, users can flexibly trade off between fidelity and diversity.

### Loss & Training

- **For the training-free version**: No extra training is required. Vector operations are directly applied during the inference stage of the pre-trained GAN.
- **For the learning-based version**: A large volume of samples and their corresponding conditions generated by the pre-trained GAN are used as training data to train the lightweight approximation network. The loss function is a standard distribution matching loss, aiming to minimize the difference between the approximated distribution and the true conditional distribution.
- The overall framework does not modify the parameters of the original GAN, only adding post-processing steps during the inference stage.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (training-free) | Ours (learning-based) | Prev. SOTA GAN | One-step Diffusion |
|--------|------|---------------------|----------------------|-------------|-------------------|
| ImageNet 64×64 | FID↓ | 6.06 | **4.37** | 8.87 | 4.02 |

### Ablation Study

| Configuration | FID | Description |
|------|-----|------|
| Original Conditional GAN | 8.87 | Baseline performance |
| + Vector operation guidance | 6.06 | training-free, FID decreased by 31.7% |
| + Learning-based guidance | 4.37 | Further decreased by 27.9% |
| Different guidance scales | Varying | Flexible control over fidelity-diversity trade-off |

### Key Findings

- GANs can indeed benefit from guided sampling without the need for pre-training classifiers or unconditional models as in diffusion models.
- The vector operation version introduces almost no additional inference time while reducing the FID by approximately 32%.
- The learning-based version of GANdance lowers the FID to 4.37, successfully narrowing the quality gap between GANs and one-step diffusion models (FID 4.02) under comparable model sizes.
- The structured latent space of GANs is a natural advantage for implementing guided sampling, a characteristic that diffusion models lack.

## Highlights & Insights

1. **Ingenious Concept Transfer**: Migrating the guided sampling strategy from diffusion models to GANs leverages the unique structured properties of GAN latent spaces, achieving a "different paths, same destination" effect across different generative paradigms.
2. **Extremely Low Implementation Cost**: The training-free version requires only simple vector operations without altering the model architecture or retraining, making it plug-and-play for existing pre-trained GANs.
3. **Bridging the GAN-Diffusion Gap**: The FID gap of 4.37 vs. 4.02 is remarkably narrow, indicating that the potential of GANs in conditional image generation quality is far from being fully exploited.
4. **New Research Direction**: This work implies that there may be more mutually beneficial techniques to share between GANs and diffusion models, opening up a new perspective on cross-model method transfer.

## Limitations & Future Work

1. Experiments are primarily conducted on ImageNet 64×64; the efficacy on higher resolutions and more complex datasets remains to be validated.
2. The learning-based version still requires an additional training process, which increases deployment complexity despite being lightweight.
3. The choice of guidance scale currently relies on manual tuning; adaptive tuning strategies are worth exploring.
4. The quality gap between this method and the latest large-scale diffusion models (such as SDXL, DALL-E 3, etc.) remains substantial.
5. The effectiveness of the method depends on the GAN possessing a well-structured latent space; the performance may be limited if the GAN is insufficiently trained.

## Related Work & Insights

- **Classifier Guidance (Dhariwal & Nichol, 2021)**: Uses gradients of a pre-trained classifier to guide generation in diffusion models, which requires an auxiliary classifier.
- **Classifier-Free Guidance (Ho & Salimans, 2022)**: Achieves guidance by jointly training conditional and unconditional diffusion models, requiring support during training.
- **StyleGAN Latent Space Manipulation**: Prior works have demonstrated the semantic structure of GAN latent spaces; applying this to guided sampling is a natural extension.
- **Insight**: The structured latent space of GANs may hold more untapped applications, such as fine-grained control in style transfer, attribute editing, etc.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of transferring guided sampling from diffusion models to GANs is novel, with an ingenious angle utilizing the structured properties of latent space.
- **Experimental Thoroughness**: ⭐⭐⭐ Main experiments are concentrated on a single dataset, ImageNet 64×64, yielding limited scale and diversity.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clear, the method explanation is accessible and intuitive, and the logic is rigorous.
- **Value**: ⭐⭐⭐⭐ Opens up a new direction for GAN inference optimization and holds great significance for narrowing the quality gap between GANs and diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)
- [\[ECCV 2024\] A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](a_framework_for_efficient_model_evaluation_through_stratific.md)
- [\[ECCV 2024\] Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing](wavelength-embedding-guided_filter-array_transformer_for_spectral_demosaicing.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](../../ICLR2026/others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](../../NeurIPS2025/others/robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
