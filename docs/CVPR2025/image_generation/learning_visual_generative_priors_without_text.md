---
title: >-
  [Paper Note] Learning Visual Generative Priors without Text
description: >-
  [CVPR 2025][Image Generation][Image-to-Image Generation] Proposes the Lumos framework, which learns visual generative priors through purely visual image-to-image (I2I) self-supervised pre-training, then matches or even surpasses existing T2I models with only 1/10 of the text-image pairs for fine-tuning. It also demonstrates superior performance over T2I priors on text-free visual tasks (I2V, NVS).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image-to-Image Generation"
  - "Visual Generative Priors"
  - "Self-Supervised Learning"
  - "Pre-training"
  - "Diffusion Model"
date: 2026-05-08
content_hash: b66a8be4aa6adf1d
---

# Learning Visual Generative Priors without Text

**Conference**: CVPR 2025  
**arXiv**: [2412.07767](https://arxiv.org/abs/2412.07767)  
**Code**: [https://ant-research.github.io/lumos](https://ant-research.github.io/lumos) (Project Page)  
**Area**: Image Generation  
**Keywords**: Image-to-Image Generation, Visual Generative Priors, Self-Supervised Learning, Pre-training, Diffusion Model

## TL;DR
Proposes the Lumos framework, which learns visual generative priors through purely visual image-to-image (I2I) self-supervised pre-training, then matches or even surpasses existing T2I models with only 1/10 of the text-image pairs for fine-tuning. It also demonstrates superior performance over T2I priors on text-free visual tasks (I2V, NVS).

## Background & Motivation

1. **Background**: Current text-to-image (T2I) models serve as the dominant priors for visual generation and are widely used as initialization weights for downstream tasks (e.g., video generation, 3D synthesis).
2. **Limitations of Prior Work**: T2I models heavily rely on high-quality text-image pairs. Experiments show that when the ratio of text noise increases from 10% to 90%, the CLIP score drops by about 1.0. Scaling up high-quality paired data is extremely expensive, which limits model scaling.
3. **Key Challenge**: T2I models need to learn two difficult tasks simultaneously—texture modeling and text-image alignment. Noisy text not only affects alignment but also interferes with the learning of texture modeling.
4. **Goal**: Can texture modeling and cross-modal alignment be decoupled—first learning a purely visual generative prior using massive unlabeled images, and then fine-tuning for alignment using a small amount of paired data?
5. **Key Insight**: Cross-modal alignment is not a necessary condition for a "good visual generative prior"; the core of a visual prior lies in texture modeling. I2I generation can be learned self-supervised from unlabeled images.
6. **Core Idea**: Using a pre-trained visual encoder (DINO) to extract image features as conditions, an I2I diffusion model is trained on 190 million unlabeled images to serve as a more fundamental visual prior.

## Method

### Overall Architecture
Consists of two stages: (1) I2I pre-training—given an image, features are extracted using a frozen visual encoder (DINO-B), which serves as the condition to train a DiT-XL/2 diffusion model for image reconstruction; (2) Downstream transfer—loading the I2I pre-training weights, switching the conditioning input from image features to the output of a text encoder (T5-XXL), and fine-tuning on a small number of text-image pairs for T2I, or directly transferring to text-free tasks such as NVS and I2V.

### Key Designs

1. **Purely Visual I2I Pre-training Framework**:
    - **Function**: Learns visual generative priors on unlabeled images in a self-supervised manner.
    - **Mechanism**: Given an image $x$, it is first encoded into the latent space $z = \mathcal{E}(x)$ using a pre-trained VAE, while visual semantic features $\tau^{\text{img}}(x) \in \mathbb{R}^{M \times d}$ are extracted using a frozen DINO-B. Conditioned on these features, the information is injected into the DiT backbone through cross-attention, and trained with the standard diffusion denoising objective. Key point: Both the encoder and VAE remain frozen throughout the pre-training process.
    - **Design Motivation**: Self-supervised visual encoders like DINO have been proven to extract richer features than supervised models. Utilizing them as conditions allows the I2I model to fully exploit massive unlabeled images on the internet.

2. **Condition Feature Selection Strategy**:
    - **Function**: Determines whether the I2I model uses global or local visual features as conditions.
    - **Mechanism**: Three types of features are compared: global CLS token, local patch tokens, and all tokens. Experiments reveal that local features significantly accelerate I2I convergence but are detrimental to downstream T2I fine-tuning (due to overly strong dependency on the condition). Although global features lead to slower I2I convergence, they yield better downstream transfer. Thus, global features are selected.
    - **Design Motivation**: Global features provide semantic-level constraints instead of pixel-level constraints, leaving more flexibility for downstream tasks. This finding reveals the "upstream-downstream discrepancy."

3. **Visual Encoder Selection: DINO vs CLIP**:
    - **Function**: Verifies the impact of purely visual encoders versus multimodal encoders on I2I priors.
    - **Mechanism**: Compares DINO and MoCoV3 (purely visual) with CLIP (multimodal). DINO/MoCoV3 converge faster and achieve better FIDs during the I2I stage. For downstream T2I fine-tuning, CLIP has an early advantage (due to its text-alignment capability), but DINO eventually surpasses it—referred to by the authors as a "late bloomer." T5 outperforms the CLIP encoder as the text encoder in the T2I stage.
    - **Design Motivation**: Demonstrates that purely visual priors are not only feasible but also yield better final results. The I2I prior enhances texture modeling, simplifying the T2I learning process.

### Loss & Training
The I2I stage uses the standard diffusion denoising loss and is trained on 190M images. In the T2I stage, competitive results are achieved by fine-tuning on 30M text-image pairs for 65K steps. Conditional dropout is supported to enable classifier-free guidance.

## Key Experimental Results

### Main Results

| Model | T&I Pairs | Steps | FID-30K↓ |
|------|-----------|-------|----------|
| SDv1.5 | 2000M | 1026k | 9.62 |
| PixArt-α | 24M | 240k | 7.32 |
| Imagen | 860M | 5000k | 7.27 |
| **Lumos-T2I** | **30M** | **65k** | **12.20** |
| **Lumos-T2I (Long Captions)** | **30M** | **65k** | **6.44** |

With long captions, Lumos surpasses all existing methods using only 30M data and 65K steps. It achieves Overall=0.57 on GenEval and Average=79.9 on DPG-Bench, which is comparable to or better than models of the same scale.

### Ablation Study

| I2I Data Scale | I2I FID↓ | T2I FID↓ | Description |
|------------|----------|----------|------|
| 10M | Higher | Higher | Small data |
| 50M | Medium | Medium | Continuous improvement |
| 200M | Lowest | Lowest | Effective scaling |

| Prior Type | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ |
|---------|-----------|-----------|------------|
| No Prior | Lower | Lower | Higher |
| T2I Prior | Medium | Medium | Medium |
| **I2I Prior** | **19.63** | **0.8439** | **0.1526** |

### Key Findings
- **Upstream-Downstream Discrepancy**: A better FID of the I2I model itself does not guarantee better downstream T2I transfer. Local features perform well on I2I but poorly on T2I, whereas global features show the opposite trend—revealing that the objectives of pre-training and downstream tasks are not fully aligned.
- **Scalability of I2I Priors**: From 10M to 200M images, the FIDs of both I2I and downstream T2I continuously decrease, proving that learning purely visual priors can benefit effectively from data scaling.
- **I2I Priors Outperform T2I Priors on Text-Free Tasks**: On NVS and I2V tasks, the I2I prior consistently outperforms the T2I prior because it bypasses the need for manual text prompt engineering.
- **DINO is a "Late Bloomer"**: Although CLIP converges faster in the early stages of I2I, DINO eventually surpasses it in the final steps and achieves better downstream T2I performance.

## Highlights & Insights
- **Decoupling Texture Modeling and Alignment**: Splitting the two difficult challenges of T2I—first learning textures using massive unlabeled data (I2I), and then learning alignment with a small amount of paired data (T2I fine-tuning). This represents a fundamental approach to efficiency improvement.
- **Discovery of Upstream-Downstream Discrepancy**: A "good" pre-trained model does not equate to a "good" performance on downstream tasks—this provides crucial insights for choosing pre-training strategies.
- **Data Efficiency**: Competitive performance is achieved using only 1/10 of the text-image pairs, significantly reducing the demand for high-quality paired data. This is highly beneficial for researchers with limited resources.

## Limitations & Future Work
- Currently only validated on DiT-XL/2 (~0.8B parameters); the performance on larger models remains unknown.
- The effects of image filtering criteria and data sources in I2I pre-training on the final results are not fully explored.
- T2I fine-tuning still requires 30M paired data; can it be further reduced to the million level?
- Transferring the I2I prior to more downstream tasks, such as image editing and image inpainting, has not yet been explored.

## Related Work & Insights
- **vs PixArt-α**: PixArt-α uses class-to-image pre-training on ImageNet, which still relies on manually annotated class labels; Lumos is entirely self-supervised and requires no annotation.
- **vs RCG**: RCG also performs I2I generation but is trained only on ImageNet and injects conditioning via adaLN-Zero; Lumos is trained on 190M images using cross-attention, reducing FID from 12.70 to 4.82.
- **vs DALL·E2**: DALL·E2 also leverages the concept of an intermediate I2I bridge, but its unCLIP design requires CLIP alignment; Lumos demonstrates that a purely visual encoder is superior.

## Rating
- Novelty: ⭐⭐⭐⭐ Clearly proposes the concept of purely visual generative priors and systematically validates it.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three downstream tasks (T2I, NVS, I2V) with extensive ablation studies and highly detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with rich figures and tables.
- Value: ⭐⭐⭐⭐ Reduces dependency on data annotation, offering significant reference value for large-scale visual generative pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Visual Generation Without Guidance](../../ICML2025/image_generation/visual_generation_without_guidance.md)
- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)
- [\[CVPR 2026\] PhyCo: Learning Controllable Physical Priors for Generative Motion](../../CVPR2026/image_generation/phyco_learning_controllable_physical_priors_for_generative_motion.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)

</div>

<!-- RELATED:END -->
