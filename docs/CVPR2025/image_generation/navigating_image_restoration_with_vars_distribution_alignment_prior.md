---
title: >-
  [Paper Note] Navigating Image Restoration with VAR's Distribution Alignment Prior
description: >-
  [CVPR 2025][Image Generation][Image Restoration] This paper discovers that the next-scale prediction of the Visual AutoRegressive (VAR) model possesses an inherent multi-scale distribution alignment capability—low-scale restores global degradations (e.g., low-light, haze), while high-scale restores local degradations (e.g., noise, rain streaks). Based on this, the VarFormer framework is constructed, which adaptively selects scale priors via Degradation-Aware Enhancement (DAE)…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Restoration"
  - "VAR"
  - "Generative Prior"
  - "Multi-Scale Distribution Alignment"
  - "Universal Degradation Restoration"
  - "VarFormer"
date: 2026-05-08
content_hash: 2d1bedafafa24e27
---

# Navigating Image Restoration with VAR's Distribution Alignment Prior

**Conference**: CVPR 2025  
**arXiv**: [2412.21063](https://arxiv.org/abs/2412.21063)  
**Code**: [https://github.com/siywang541/Varformer](https://github.com/siywang541/Varformer)  
**Area**: Image Generation/Image Restoration  
**Keywords**: Image Restoration, VAR, Generative Prior, Multi-Scale Distribution Alignment, Universal Degradation Restoration, VarFormer

## TL;DR

This paper discovers that the next-scale prediction of the Visual AutoRegressive (VAR) model possesses an inherent multi-scale distribution alignment capability—low-scale restores global degradations (e.g., low-light, haze), while high-scale restores local degradations (e.g., noise, rain streaks). Based on this, the VarFormer framework is constructed, which adaptively selects scale priors via Degradation-Aware Enhancement (DAE) and fuses prior and degraded features via Adaptive Feature Transformation (AFT), outperforming existing multi-task methods across 6 restoration tasks.

## Background & Motivation

**Background**: Image restoration aims to reconstruct high-quality images from degraded low-quality inputs. Task-specific methods (such as Restormer for deblurring, SwinIR for denoising) perform excellently on a single degradation but lack universality. Universal methods (e.g., AirNet, IDR, Prompt-IR) attempt to handle multiple degradations in a single model, but only use degraded images as feature sources, ignoring the prior distribution of clean images. Generative prior methods based on diffusion models (such as DiffUIR), though effective, suffer from slow inference.

**Limitations of Prior Work**: (1) Purely discriminative universal restoration models lack structural priors of clean images, leading to limited texture restoration and structure reconstruction capabilities; (2) Generative prior methods based on GAN/Diffusion are computationally expensive during inference and unstable to train; (3) Different degradation types affect different frequency scales (haze affects global tone, while noise affects local texture), and existing methods lack multi-scale adaptive processing capabilities.

**Key Challenge**: Different degradation types require different levels of restoration across scales—global degradations (low-light, haze) require correcting the overall tone and contrast, while local degradations (noise, rain streaks) require fine texture reconstruction. It is challenging for a single restoration network to handle both optimally at the same time.

**Goal**: How to utilize the inherent multi-scale prior of VAR, a new generative paradigm, to endow universal restoration models with adaptive, multi-scale degradation-aware restoration capabilities.

**Key Insight**: The authors discover that the next-scale prediction process of VAR naturally aligns the representations of degraded and clean images into a common space—by replacing the VAR features of degraded images at different scales with autoregressive prediction features, different types of degradations can be selectively eliminated.

**Core Idea**: Utilize the multi-scale distribution alignment prior of VAR as restoration guidance—low-scale priors correct global degradations, while high-scale priors repair local degradations, adaptively weighting different scale priors through the DAE module to handle arbitrary degradation types.

## Method

### Overall Architecture

VarFormer is trained in two stages. **Stage 1**: Train the Adapter module on top of the frozen VAR to obtain enhanced multi-scale distribution alignment embeddings $S_v$, utilizing a Feature Matching Loss to bring the VAR representations of degraded and clean images closer. **Stage 2**: Using a U-Net-style encoder-decoder as the restoration backbone, DAE modules (to adaptively select weights for VAR scale priors) and AFT modules (to fuse weighted priors into restoration features) are inserted at each layer, training the restoration network end-to-end via a Reconstruction Loss.

### Key Designs

1. **Degradation-Aware Enhancement (DAE)**:
    - **Function**: Adaptively selects the most effective combination of VAR scale priors for each layer of the restoration network.
    - **Mechanism**: The outputs $F_{e_v}^i$ from different layers of the VAR encoder naturally know what level of information each layer should focus on. Passing $F_{e_v}^i$ through Swin-Transformer blocks (to filter out image content interference) and projection convolutions yields weights $W=[w_i]_{i=1}^K$ for $K$ scale priors, performing a weighted sum to obtain the VAR prior $\hat{S}_w^i = \mathcal{M}(\sum_{j=1}^K w_j \cdot S_v^j)$. Then, RSTB+Softmax generate region-adaptive fusion weights $w_1^g, w_2^g$, resulting in $F_{g_{e/d}}^i = F_{e/d}^i \times w_1^g + \hat{S}_w^i \times w_2^g$.
    - **Design Motivation**: Dehazing requires low-scale global priors while denoising requires high-scale local priors. A one-size-fits-all fixed fusion strategy cannot be optimal—the network must adaptively select "which scale of clean image knowledge is most needed to restore the features of this layer."

2. **Adaptive Feature Transformation (AFT)**:
    - **Function**: Alleviates structural distortions and texture artifacts caused by fusing high-quality priors with low-quality degraded features.
    - **Mechanism**: Introduces a low-dimensional intermediate feature $M$ (bridge feature) as a "mediator" between the prior and degraded features. $Q$ is derived from the degraded feature $F_{e/d}^i$, keys and values $K,V$ are derived from the enhanced feature $F_{g_{e/d}}^i$, and $M$ is obtained from the concatenation and projection of both. The attention mechanism operates in two steps: first computing $A_{q,m} = \text{Softmax}(QM^T/\sqrt{d})$, then $A_{m,k} = \text{Softmax}(MK^T/\sqrt{d})$, and finally $F_{in}^{i+1} = A_{q,m} \cdot (A_{m,k} \cdot V) + F_{e/d}^i$.
    - **Design Motivation**: Direct cross-attention fusion of features with large quality gaps introduces artifacts. Aligning two highly divergent feature spaces through an intermediate mediator is more stable than direct query-key matching.

3. **Adapter for Domain Shift (Stage 1 Training)**:
    - **Function**: Bridges the gap between the pre-trained distribution and the degraded image distribution while keeping VAR parameters frozen.
    - **Mechanism**: Inserts an Adapter module containing self-attention blocks after the VAR encoder, training it with a Feature Matching Loss—combining cross-entropy loss (to align multi-scale prediction tokens) and L2 loss (to pull the Adapter output closer to the ground-truth VQVAE quantized feature $F_{e_{gt}}^q$).
    - **Design Motivation**: Directly fine-tuning the VAR model destroys pre-trained knowledge, and there is a distribution gap between the degraded images observed by the VAR encoder and the clean images used in pre-training. A lightweight Adapter is the best balance between the two.

### Loss & Training

- **Stage 1**: Feature Matching Loss $\mathcal{L}_{fema} = \sum_{i=1}^K -s_i \log(\hat{s}_i) + \|F_a - sg(F_{e_{gt}}^q)\|_2^2$
- **Stage 2**: Reconstruction Loss $\mathcal{L}_{rec} = -\text{PSNR}(I_{gt}, I_{rec}) + \|\psi(I_{gt}) - \psi(I_{rec})\|_2^2$, where $\psi$ is a pre-trained VGG19.

## Key Experimental Results

### Main Results

**Comparison of Four Restoration Tasks (PSNR↑ / SSIM↑)**:

| Method | Deraining | Deblurring | Low-Light Enhancement | Dehazing |
|------|------|--------|----------|------|
| Restormer (task-specific) | 33.96/0.935 | 32.92/0.961 | 20.41/0.806 | 30.87/0.969 |
| AirNet (universal) | 25.44/0.743 | 27.14/0.832 | 18.49/0.767 | 25.48/0.944 |
| DiffUIR (universal) | 31.14/0.907 | 29.88/0.874 | 25.02/0.901 | 32.74/0.944 |
| **VarFormer** | **31.33/0.913** | **30.99/0.956** | **25.13/0.917** | **32.96/0.956** |

VarFormer achieves the best performance among all universal methods, particularly outstanding in deblurring and low-light enhancement.

### Ablation Study

**Ablation of Key Components (PSNR on Deraining Task)**:

| Configuration | PSNR↑ |
|------|-------|
| Baseline (w/o VAR prior) | ~29.5 |
| + VAR prior (fixed weight) | ~30.2 |
| + DAE (adaptive weight) | ~30.8 |
| + AFT (feature transformation) | ~31.1 |
| + Adaptive mix-up skip | **31.33** |

### Key Findings

- t-SNE visualization confirms that the next-scale prediction of VAR indeed maps various degraded images and clean images to closer distributions—this is an inherent property of VAR rather than human design.
- Replacing low-scale VAR features eliminates global degradations (haze, low-light), while replacing high-scale features eliminates local degradations (noise, rain streaks)—aligning perfectly with the intuition of multi-resolution analysis.
- VarFormer also demonstrates good generalization on **unseen tasks** (e.g., real-world denoising), indicating that the VAR prior provides universal "clean image knowledge" beyond the training tasks.
- Compared to DiffUIR, VarFormer reduces training computational costs—Stage 1 only trains a lightweight cross-attention and Adapter.

## Highlights & Insights

1. **Discovering the distribution alignment capability of VAR** is the core contribution of this paper—it is not an explicitly designed function but an inherent property of the next-scale prediction paradigm, providing profound insights.
2. Converting the generative prior from "directly generating clean images" to "extracting multi-scale aligned embeddings to guide restoration" **avoids cumulative errors and the speed bottleneck of pixel-by-pixel generation**.
3. The "degradation type -> scale weight" mechanism of DAE is a simple yet effective design, allowing a single model to automatically adapt to different degradations without explicit degradation classification.
4. Utilizing VAR for image restoration tasks for the first time, **opening up a new application direction for VAR priors in low-level vision**.

## Limitations & Future Work

- The quantization error of the VQVAE in VAR itself limits the upper bound of restoration accuracy.
- It requires a forward pass of the pre-trained VAR to extract the prior, which increases memory and computational over head during inference.
- It is only validated on $256 \times 256$ resolution, and high-resolution scenarios (such as 4K restoration) remain unexplored.
- The reconstruction pre-training in Stage 1 and the restoration training in Stage 2 must be performed in separate steps, resulting in a somewhat complex pipeline.
- Reconstruction capabilities under severe degradation (e.g., extreme overexposure, large-area occlusions) remain to be verified.

## Related Work & Insights

- **VAR** [Tian et al.]: Proposes the next-scale prediction paradigm to replace next-token prediction; this paper identifies and exploits its inherent distribution alignment properties.
- **Restormer** [Zamir et al.]: A Transformer-based state-of-the-art task-specific restoration method; VarFormer achieves performance close to its specialized capacity under the universal setting.
- **DiffUIR** [Zheng et al.]: Applies diffusion priors for universal restoration; VarFormer is faster and yields better results.
- **IDR** [Zhang et al.]: A component-oriented universal restoration paradigm; the proposed method is complementary to it—IDR analyzes task relationships, while VarFormer provides generative priors.
- **Insight**: The intermediate representations of autoregressive generative models often contain rich structural priors, which are worth exploring in more discriminative tasks for their "free" knowledge.

## Rating

⭐⭐⭐⭐ — The discovery of VAR's distribution alignment property is highly insightful, and the VarFormer framework design is reasonable with solid experiments. Introducing VAR to image restoration for the first time is a meaningful pioneering effort. The DAE and AFT modules are designed simply and effectively. Code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reversing Flow for Image Restoration](reversing_flow_for_image_restoration.md)
- [\[ICML 2025\] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](../../ICML2025/image_generation/restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)

</div>

<!-- RELATED:END -->
