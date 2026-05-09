---
title: >-
  [Paper Note] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images
description: >-
  [CVPR 2026][Image Generation][Fake image detection] This paper proposes SimLBR, which regularizes a detector by blending a small amount of fake image information into real image embeddings within the DINOv3 latent space, compelling the model to learn a compact decision boundary around the real image distribution. This design achieves strong generalization to unseen generators, attaining 94.54% average accuracy on GenImage and outperforming AIDE on the challenging Chameleon benchmark by 25% in accuracy and 70% in recall.
tags:
  - CVPR 2026
  - Image Generation
  - Fake image detection
  - AI-generated images
  - latent blending regularization
  - cross-generator generalization
  - DINOv3
date: 2026-05-08
content_hash: 0068469777a2d039
---

# SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images

**Conference**: CVPR 2026
**arXiv**: [2602.20412](https://arxiv.org/abs/2602.20412)
**Code**: Available (to be released on HuggingFace and GitHub)
**Area**: Image Generation
**Keywords**: Fake image detection, AI-generated images, latent blending regularization, cross-generator generalization, DINOv3

## TL;DR
This paper proposes SimLBR, which regularizes a detector by blending a small amount of fake image information into real image embeddings within the DINOv3 latent space, compelling the model to learn a compact decision boundary around the real image distribution. This design achieves strong generalization to unseen generators, attaining 94.54% average accuracy on GenImage and outperforming AIDE on the challenging Chameleon benchmark by 25% in accuracy and 70% in recall.

## Background & Motivation

**State of the Field**: AI-generated image detection is a critical media forensics task. Existing methods typically train on fake images from a single generator and aim to generalize to others. UnivFD uses CLIP features with nearest-neighbor classification, while AIDE combines DCT and CLIP features.

**Limitations of Prior Work**: Nearly all state-of-the-art detectors overfit to generator-specific fingerprints present in the training data, suffering catastrophic performance degradation when encountering unseen generators. On the Chameleon hard test set proposed in the AIDE paper, all existing methods fail severely—fake images are misclassified as real (recall as low as 5%), revealing that these methods effectively learn compact boundaries around fake images and treat "real" as a sink class.

**Root Cause**: The distribution of fake images continuously evolves as new generators emerge; learning boundaries around fake images inevitably misses out-of-distribution fakes. In contrast, the real image distribution is relatively stable—real images are real, and anything that is not real should be classified as fake.

**Paper Goals**: To train a detector that learns a compact decision boundary around the real image distribution, treating "fake" as the sink class, thereby generalizing to arbitrary unseen generators.

**Starting Point**: Latent Blending Regularization (LBR) makes the classification of real images harder during training—by interpolating a small amount of fake image information into real image embeddings and labeling the result as fake, the model is forced to classify only completely uncontaminated real images as real.

**Core Idea**: Perform simple linear interpolation between real and fake image embeddings in the DINOv3 semantic latent space, training the model to draw a compact boundary around the pure real image distribution.

## Method

### Overall Architecture

The SimLBR pipeline is extremely simple, consisting of three steps:

1. **Embedding Precomputation**: DINOv3 embeddings are precomputed for all training images—real image set $\mathbb{R}=\{R_1,...,R_n\}$ and fake image set $\mathbb{F}=\{F_1,...,F_n\}$ from a single training generator $G_s$.
2. **LBR Training**: For each real image $R_i$, a label $y_i \in \{0,1\}$ is sampled randomly. If $y_i=0$, the real image embedding $L_i = I(R_i)$ is used directly; if $y_i=1$, LBR blends in fake image information and the sample is labeled as fake.
3. **Lightweight MLP Classification**: A 2-layer ReLU MLP trained with BCE loss, batch size 500, Adam (lr=1e-4, wd=1e-2), up to 10 epochs.

The entire training process on a single H100 GPU takes only **3 minutes** (excluding embedding precomputation, which takes approximately 22 minutes for GenImage and 50 minutes for AIGC). In comparison, AIDE requires approximately 2 hours of training on 8 A100 GPUs.

### Core Module: Latent Blending Regularization (LBR)

**Function**: Linearly interpolates real image embeddings with fake image embeddings in the pretrained DINOv3 latent space to produce "contaminated real" representations, which are then labeled as fake.

**Procedure**:

1. For real image $R_i$ (with label $y_i=1$), randomly sample a fake image $F_i$.
2. Extract embeddings: $L_i^R = I(R_i)$, $L_i^F = I(F_i)$.
3. Linear interpolation: $L_i = \alpha \cdot L_i^R + (1-\alpha) \cdot L_i^F$.
4. Label $L_i$ as fake and feed it into MLP training.

**Sampling of the blending coefficient $\alpha$**:

$$\alpha \sim \text{Uniform}(0.5, B), \quad B=0.8$$

- $\alpha > 0.5$ ensures the blended result retains **most** real image information while containing sufficient fake information to be labeled as fake.
- Low $\alpha$ (e.g., 0.1–0.3): fake information dominates, making classification too easy and failing to regularize effectively.
- High $\alpha$ (e.g., 0.95–0.99): the blended result is nearly identical to a pure real image but is labeled fake, causing training instability.
- Ablation studies confirm stable performance for $0.7 < B < 0.9$.

**Design Motivation**: The model must classify any sample containing **any degree** of fake information as fake, and is therefore forced to precisely identify "what constitutes a completely pure real image"—forming a compact decision boundary around the real image distribution. At inference time, any sample falling outside this compact boundary is classified as fake, regardless of the generator it originates from.

### Why Blend in Latent Space Rather Than Pixel Space?

Prior face forgery detection methods (e.g., Face X-ray) perform blending in pixel space, which suffers from three limitations:

1. **Low-level artifact leakage**: Pixel-space blending directly copies low-level artifacts from fake images (e.g., frequency-domain features), allowing the model to "take shortcuts" by detecting these artifacts rather than learning the real image distribution.
2. **Domain-specific preprocessing**: Pixel-space blending typically requires task-specific operations such as facial landmark detection, limiting the generality of the approach.
3. **Coarse control granularity**: Pixel-space blending provides limited precision in controlling the amount of injected fake information.

Advantages of blending in DINOv3's high-level semantic space:
- Low-level cues are abstracted away in semantic embeddings, compelling the model to learn **semantic-level** real/fake distinctions.
- Simple vector linear operations suffice, making the approach efficient and robust.
- Fine-grained control over the blending ratio enables virtually unlimited regularization samples.

### Critical Role of the DINOv3 Latent Space

The effectiveness of LBR is **strongly dependent** on the geometric properties of the embedding space. Linear interpolation presupposes that the embedding manifold is sufficiently smooth—the linear path between two embeddings should lie on the valid manifold.

Key experimental findings:
- **DINOv3 + LBR**: Average accuracy improves from 78.05% to 88.26% (+10.21%).
- **DINOv2 + LBR**: Average accuracy changes marginally from 68.92% to 68.51% (essentially no effect).

This demonstrates that DINOv3's embedding manifold satisfies the smoothness requirement, whereas DINOv2's manifold structure may be geometrically deficient. The authors hypothesize this is related to structural limitations of weaker embedding models.

### Training Objective

Standard binary cross-entropy loss:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(F_\theta(L_i)) + (1-y_i)\log(1-F_\theta(L_i))]$$

### Evaluation Metric Innovations

**1. Reliability Score**:

$$\text{Reliability} = \frac{\mu_{\text{acc}} - A_{\text{base}}}{\sigma_{\text{acc}}}$$

Inspired by the Sharpe ratio from finance, where $\mu_{\text{acc}}$ is the mean accuracy across generators, $\sigma_{\text{acc}}$ is the standard deviation, and $A_{\text{base}}=50\%$. High reliability indicates high accuracy combined with low variance.

**2. Worst-Case Estimate (WCE)**: The lowest accuracy achieved by the detector across all evaluated generators, serving as a performance lower bound when facing the most challenging future generators.

## Key Experimental Results

### Main Results 1: GenImage Benchmark (Trained on SD v1.4)

| Method | MidJ | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | Mean↑ | Std↓ | Rel.↑ |
|--------|------|-------|-----|-------|--------|------|--------|-------|------|-------|
| UnivFD | 73.2 | 84.0 | 55.2 | 76.9 | 75.6 | 56.9 | 80.3 | 73.29 | 11.32 | 2.05 |
| PatchCraft | 79.0 | 89.3 | 77.3 | 78.4 | 89.3 | 83.7 | 72.4 | 82.30 | 6.56 | 4.29 |
| AIDE | 79.4 | 99.8 | 78.5 | 91.8 | 98.7 | 80.3 | 66.9 | 86.88 | 12.33 | 2.99 |
| **SimLBR** | **91.7** | **98.1** | **97.0** | **92.4** | **97.4** | **93.5** | **88.0** | **94.54** | **3.74** | **11.91** |

SimLBR achieves the highest accuracy on 5 of 7 test generators and surpasses AIDE by **21%** on the out-of-distribution GAN model BigGAN. Its Reliability Score of 11.91 is 4× that of AIDE.

### Main Results 2: Chameleon Hard Test Set

| Method | ProGAN-trained Total Acc | ProGAN-trained Fake/Real Acc | SD-trained Total Acc | SD-trained Fake/Real Acc |
|--------|--------------------------|------------------------------|----------------------|--------------------------|
| AIDE | 58.37 | 5.04 / 98.46 | 62.60 | 20.33 / 94.38 |
| **SimLBR** | **84.33** | **75.80 / 90.74** | **85.57** | **92.96 / 80.06** |

On Chameleon, SimLBR's fake image recall surges from AIDE's 5.04% to 75.80% (ProGAN-trained), with overall accuracy improving by +25%.

### Ablation Study

| Backbone | LBR | Chameleon | AIGC | GenImage | RSFake | Mean↑ | Std↓ | Rel.↑ | WCE↑ |
|----------|-----|-----------|------|----------|--------|-------|------|-------|------|
| DINOv2 | ✗ | 56.12 | 70.77 | 86.74 | - | 68.92 | 13.31 | 1.42 | 56.12 |
| DINOv2 | ✓ | 55.77 | 74.75 | 81.49 | - | 68.51 | 11.71 | 1.57 | 55.77 |
| DINOv3 | ✗ | 59.65 | 76.70 | 92.19 | 73.90 | 78.05 | 12.77 | 2.19 | 59.65 |
| DINOv3 | ✓ | **84.33** | **88.40** | **94.54** | **88.88** | **88.26** | **3.94** | **9.70** | **84.33** |

### Key Findings
- LBR yields a +10.21% average improvement on DINOv3 but is nearly ineffective on DINOv2, confirming that the geometric properties of the embedding space are a prerequisite for LBR's success.
- Fake recall improvements are remarkable: on Chameleon-ProGAN from 7.24% → 75.80%, on AIGC from 54.07% → 82.07%, and on RSFake from 48.28% → 78.87%.
- MLP depth: 0–4 layers perform well; more than 4 layers leads to severe overfitting.
- The upper bound $B$ of $\alpha$ is stable across the range 0.7–0.9.
- Worst-case performance: SimLBR maintains 88% / 75.54% on GenImage / AIGC respectively, far exceeding other methods' ~50%.

## Highlights & Insights
- The **paradigm shift from "detecting fake images" to "detecting real images"** is highly insightful: it reframes the problem from perpetually chasing new generators to modeling the relatively stable real image distribution, representing a significant methodological contribution.
- The **minimalist design of LBR** is impressive—strong regularization is achieved through simple linear interpolation in the embedding space, with training completed in just 3 minutes.
- The **Reliability Score**, adapted from the financial Sharpe ratio, provides a more comprehensive perspective for detector evaluation and is broadly applicable to robustness assessment in classification tasks.
- t-SNE visualizations clearly illustrate the qualitative change in decision boundaries before and after LBR: without LBR, fake images from unseen generators intermingle with real image clusters; with LBR, real images form a compact cluster, and any deviation is classified as fake.

## Limitations & Future Work
- Strong dependence on DINOv3 embedding quality; the method is entirely ineffective with DINOv2, and validation on future backbones will be necessary.
- LBR relies solely on linear interpolation; more sophisticated blending strategies on the manifold (e.g., spherical interpolation, slerp) could be explored.
- The assumption that the real image distribution is "relatively invariant" may be violated over the long term due to evolving camera sensors and post-processing pipelines.
- The limited capacity of the MLP may be insufficient against carefully crafted adversarial attacks.

## Related Work & Insights
- **vs. UnivFD**: Both leverage pretrained feature spaces, but UnivFD uses nearest-neighbor classification while SimLBR applies LBR-regularized MLP training. The explicit learning of real image boundaries via LBR is the key distinction.
- **vs. AIDE**: AIDE uses DCT + CLIP features and requires 2 hours of training on 8 A100 GPUs; SimLBR uses only DINOv3 embeddings and trains on a single GPU in 3 minutes, yet substantially outperforms AIDE on hard test sets—demonstrating that feature space selection and training paradigm matter more than model complexity.
- **vs. pixel-space blending methods** (e.g., Face X-ray): Pixel-space blending depends on domain-specific preprocessing and leaks low-level artifacts; latent-space blending is domain-agnostic and enforces semantic-level discrimination.
- **vs. anomaly detection methods**: Pure anomaly detection (modeling only real images without fake image guidance) relies on unrealistic assumptions in high-dimensional natural image space. SimLBR uses fake images as guiding signals to construct decision boundaries, proving more effective than purely unsupervised approaches.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift to "detecting real images" and the LBR design are both simple and deeply insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four major benchmarks (GenImage, AIGC, Chameleon, RSFake) with worst-case estimation and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the introduction of reliability metrics adds depth to the evaluation.
- Value: ⭐⭐⭐⭐⭐ Extremely low training cost combined with strong generalization makes SimLBR well-suited for real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Group Editing: Edit Multiple Images in One Go](group_editing_edit_multiple_images_in_one_go.md)
- [\[CVPR 2026\] TokenLight: Precise Lighting Control in Images using Attribute Tokens](tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)
- [\[CVPR 2026\] HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching](hazematching_dehazing_light_microscopy_images_with_guided_conditional_flow_match.md)
- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)

<!-- RELATED:END -->
