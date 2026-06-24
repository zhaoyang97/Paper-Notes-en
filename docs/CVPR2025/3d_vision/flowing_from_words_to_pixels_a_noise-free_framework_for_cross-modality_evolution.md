---
title: >-
  [Paper Note] Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution
description: >-
  [CVPR 2025][3D Vision][Cross-modality generation] CrossFlow is proposed, a general cross-modality Flow Matching framework that directly evolves from the data distribution of one modality to that of another (instead of starting from noise) without cross-attention conditioning mechanisms. It slightly outperforms standard Flow Matching baselines in text-to-image generation and demonstrates superior scaling properties regarding model size and training steps.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Cross-modality generation"
  - "Flow Matching"
  - "text-to-image"
  - "noise-free source distribution"
  - "variational encoder"
date: 2026-05-08
content_hash: 4264140af3ecf89a
---

# Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution

**Conference**: CVPR 2025  
**arXiv**: [2412.15213](https://arxiv.org/abs/2412.15213)  
**Code**: [https://cross-flow.github.io/](https://cross-flow.github.io/)  
**Area**: Diffusion Models  
**Keywords**: Cross-modality generation, Flow Matching, text-to-image, noise-free source distribution, variational encoder

## TL;DR
CrossFlow is proposed, a general cross-modality Flow Matching framework that directly evolves from the data distribution of one modality to that of another (instead of starting from noise) without cross-attention conditioning mechanisms. It slightly outperforms standard Flow Matching baselines in text-to-image generation and demonstrates superior scaling properties regarding model size and training steps.

## Background & Motivation

1. **Background**: Diffusion models and Flow Matching have become the mainstream pipelines for media generation. The standard practice learns a mapping from Gaussian noise to the target data distribution, where cross-modality tasks (such as text-to-image) require extra conditioning mechanisms (such as cross-attention) to integrate conditional information.
2. **Limitations of Prior Work**: Starting from noise implies that models must learn long probability paths and rely on extra parameters (e.g., cross-attention) to inject conditional information, which increases model complexity. Furthermore, different cross-modality tasks typically require task-specific architectural designs.
3. **Key Challenge**: Theoretically, Flow Matching does not require the source distribution to be noise and can accept any distribution—yet this characteristic has rarely been applied to true cross-modality generation (previously restricted to simple settings like same-domain face-to-face translation). This is due to two practical challenges: the source and target must share the same shape, and Classifier-Free Guidance (CFG) requires a conditioning mechanism.
4. **Goal**: (1) How to enable Flow Matching to evolve directly from one modality to another? (2) How to address the shape mismatch between source and target modalities? (3) How to enable CFG in the absence of explicit conditioning mechanisms?
5. **Key Insight**: Due to the information redundancy between different modalities of the same data point, the data distribution of the conditioning modality is naturally correlated with the target distribution. Directly evolving from a correlated distribution should be easier to learn than starting from noise, resulting in shorter and more efficient paths.
6. **Core Idea**: A variational encoder is used to compress the source modality data into a regularized latent space of the same shape as the target modality. Subsequently, Flow Matching directly evolves from the source modality latent space to the target modality space, eliminating the need for noise and conditioning mechanisms.

## Method

### Overall Architecture
Taking text-to-image as an example: the input text is processed by a language model to yield a text embedding $x \in \mathbb{R}^{n \times d}$. This is encoded by a Text Variational Encoder into a text latent representation $z_0 \in \mathbb{R}^{h \times w \times c}$ (sharing the same shape as the image latent space). Then, a standard Flow Matching model (a vanilla transformer using only self-attention without cross-attention) directly evolves $z_0$ into the image latent representation $z_1$. Finally, the image is obtained via a pre-trained VAE decoder.

### Key Designs

1. **Variational Encoder (VE)**:

    - **Function**: Compresses source modality data into a regularized latent distribution with the same shape as the target modality.
    - **Mechanism**: Given input $x$, VE predicts the mean $\bar{\mu}_{z_0}$ and variance $\bar{\sigma}_{z_0}$, sampling $z_0 \sim \mathcal{N}(\bar{\mu}_{z_0}, \bar{\sigma}_{z_0}^2)$. A key finding is the necessity of a variational (regularized) encoder instead of a standard deterministic encoder—directly performing Flow Matching on deterministic encoder outputs $z_0$ yields poor performance because the source distribution is insufficiently smooth. Training VE with an image-text contrastive loss (instead of a reconstruction loss) achieves the best performance by learning better semantic structures.
    - **Design Motivation**: Resolves the dimension mismatch between source and target, while regularizing the source distribution to enable effective Flow Matching. Contrastive loss ensures the latent space retains semantic information.

2. **CFG with Indicator**:

    - **Function**: Enables Classifier-Free Guidance (CFG) without explicit conditional inputs.
    - **Mechanism**: A binary indicator $1_c \in \{0, 1\}$ is introduced, forming the model $v_\theta(z_t, 1_c)$. When $1_c=1$, the model learns the mapping from $z_0$ (paired source data) to $z_1$ (paired target data); when $1_c=0$, it learns the mapping from $z_0$ to $z_1^{uc}$ (randomly unpaired target data). The indicator is concatenated into the transformer input sequence via learned parameters $g^c$ and $g^{uc}$, and the unconditional mode is used with a 10% probability during training.
    - **Design Motivation**: Standard CFG relies on dropping conditional inputs, but CrossFlow lacks independent conditional inputs. The indicator approach allows the model to distinguish between "matched source-target pairs" and "random source-target pairs," achieving equivalent guidance without training additional "poor/negative" models.

3. **Joint Training Strategy**:

    - **Function**: Jointly optimizes the VE and Flow Matching models.
    - **Mechanism**: The total loss is formulated as $L = L_{FM} + L_{Enc} + \lambda L_{KL}$, where $L_{FM}$ is the Flow Matching MSE loss for velocity prediction, $L_{Enc}$ is the encoding loss (contrastive loss), and $L_{KL}$ is the KL divergence regularization. Joint training significantly outperforms two-stage decoupled training (reducing FID from 32.55 to 24.33).
    - **Design Motivation**: The VE and Flow Matching models interact—the structure of VE's latent space directly dictates the path complexity that Flow Matching needs to learn, and joint training encourages mutual adaptation.

### Loss & Training
- Flow Matching MSE Loss: $L_{FM} = \text{MSE}(v_\theta(z_t, t), \hat{v})$
- Encoding Loss: Image-text contrastive loss $L_{Enc} = \text{CLIP}(z_0, \hat{z})$
- KL Divergence: $L_{KL} = \text{KL}(\mathcal{N}(\bar{\mu}_{z_0}, \bar{\sigma}_{z_0}^2) || \mathcal{N}(0,1))$, weight $\lambda=10^{-4}$
- Training Setup: 350M text-image pairs, 256×256 resolution, batch size 1024, learning rate of $10^{-4}$, AdamW optimizer, and the largest model (0.95B parameters) trained for 600K steps.

## Key Experimental Results

### Main Results

| Method | Parameters | FID-30K↓ | CLIP Score↑ |
|------|--------|----------|-------------|
| Standard FM (baseline) | 1.04B | 10.79 | 0.29 |
| **CrossFlow** | 0.95B | **10.13** | 0.29 |
| LDMv1.5 | 0.9B | 9.62 | 0.43* |
| CrossFlow (Sin-Cos) | 0.95B | **8.95** | - |

### Ablation Study

| Configuration | FID↓ | CLIP↑ | Description |
|------|------|-------|------|
| Encoder (Deterministic) | 66.65 | 0.20 | No regularization, extremely poor performance |
| Encoder + noise | 59.91 | 0.21 | Adding noise helps but is insufficient |
| **Variational Encoder** | **40.78** | **0.23** | Regularization is necessary |
| T-T Reconstruction loss | 40.78 | 0.23 | Reconstruction loss |
| T-T Contrastive loss | 34.67 | 0.24 | Text contrastive loss is better |
| **I-T Contrastive loss** | **33.41** | **0.24** | Image-text contrastive loss is optimal |
| No guidance | 33.41 | 0.24 | No guidance |
| AutoGuidance | 26.36 | 0.25 | AG helps |
| **CFG indicator** | **24.33** | **0.26** | Indicator method is optimal |

### Key Findings
- **Variational Encoder is crucial**: The deterministic encoder yields an FID of 66.65, whereas VE reduces it to 40.78. This indicates that regularization of the source distribution is a prerequisite for cross-modality Flow Matching.
- **Better scaling property than standard FM**: CrossFlow demonstrates a steeper slope in performance improvement for larger models and longer training. While smaller models underperform relative to the baseline, CrossFlow progressively outperforms standard FM as model size increases, suggesting more pronounced advantages at larger scales.
- **Latent space arithmetic**: CrossFlow supports addition and subtraction operations in the text latent space, such as $\mathcal{L}$("dog with a hat") + $\mathcal{L}$("sunglasses") - $\mathcal{L}$("hat") $\rightarrow$ generating a dog wearing sunglasses without a hat, which standard FM cannot achieve.
- **Strong versatility**: The same framework can be applied without modifications to image captioning (SoTA), depth estimation, and super-resolution, and supports bidirectional mapping (reversing a T2I model into I2T).

## Highlights & Insights
- **Paradigm Shift — Noise-Free**: Shifting cross-modality generation from "noise $\rightarrow$ target + conditions" to "source modality $\rightarrow$ target modality" eliminates noise and conditioning mechanisms, fundamentally simplifying the architecture. This provides a fresh perspective on generative models.
- **Emergence of Latent Space Arithmetic**: Because CrossFlow maps the source modality into a regularized continuous space, it naturally obtains a semantically structured latent space that supports meaningful vector arithmetic. This does not exist in standard conditional generative models, offering a new tool for controllable editing.
- **Simplicity of Indicator CFG**: Achieving an equivalent guidance effect to CFG with a single binary scalar avoids training additional models or modifying the architecture, presenting an extremely clean concept.

## Limitations & Future Work
- **Underperformance at small model scales**: CrossFlow is inferior to standard FM at 70M-300M parameter scales, indicating that the framework requires sufficient model capacity to effectively learn cross-modality mappings.
- **Information loss in source distribution**: The compression ratio of the VE is highly aggressive ($77 \times 768 \rightarrow 4 \times 32 \times 32$), which inevitably loses some textual details.
- **Comprehensive evaluation limited to 256px**: Evaluations at 512px are limited, and performance at higher resolutions remains unexplored.
- **Potential improvements**: Exploring more efficient VE architectures to minimize information loss; extending CrossFlow to video generation; and studying scaling behaviors at larger scales (10B+ parameters).

## Related Work & Insights
- **vs. Standard Flow Matching + Cross-Attention**: Standard approaches start from noise and use cross-attention to inject text conditions, which requires more parameters. CrossFlow eliminates cross-attention layers, uses fewer parameters while yielding better performance, and has superior scaling properties.
- **vs. InterFlow / $\alpha$-blending**: These methods also explore non-noise source distributions but are limited to same-domain translation (e.g., face-to-face). CrossFlow is the first to achieve true cross-modality evolution.
- **vs. Bit Diffusion**: Bit Diffusion encodes text as binary bits for caption generation, requiring task-specific designs. CrossFlow achieves SoTA-level image captioning with the same framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A true paradigm shift—noise-free and condition-free cross-modality generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task verification, detailed ablations, and scaling analysis, though high-resolution evaluation is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent storytelling, with a clear logical transition from theoretical motivation to experimental validation.
- Value: ⭐⭐⭐⭐⭐ Possesses the potential to shift the paradigm of cross-modality generation, with scaling characteristics suggesting massive potential at larger scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](crossover_3d_scene_cross-modal_alignment.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] ReCap: Better Gaussian Relighting with Cross-Environment Captures](recap_better_gaussian_relighting_with_cross-environment_captures.md)
- [\[CVPR 2025\] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)

</div>

<!-- RELATED:END -->
