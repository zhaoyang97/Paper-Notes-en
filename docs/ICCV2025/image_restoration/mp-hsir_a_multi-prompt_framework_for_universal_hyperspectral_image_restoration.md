---
title: >-
  [Paper Note] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration
description: >-
  [ICCV 2025][Image Restoration][Hyperspectral image restoration] This paper proposes MP-HSIR, a unified hyperspectral image restoration framework that integrates three modalities of guidance—spectral prompts (universal lo…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Hyperspectral image restoration"
  - "multi-prompt learning"
  - "all-in-one restoration"
  - "spectral prompt"
  - "text-visual synergy"
date: 2026-05-08
content_hash: cfbe4515d6d88c57
---

# MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration

**Conference**: ICCV 2025
**arXiv**: [2503.09131](https://arxiv.org/abs/2503.09131)
**Code**: [GitHub](https://github.com/ZhehuiWu/MP-HSIR)
**Area**: Image Restoration
**Keywords**: Hyperspectral image restoration, multi-prompt learning, all-in-one restoration, spectral prompt, text-visual synergy

## TL;DR

This paper proposes MP-HSIR, a unified hyperspectral image restoration framework that integrates three modalities of guidance—spectral prompts (universal low-rank spectral patterns), text prompts, and visual prompts—to comprehensively outperform existing all-in-one methods and numerous task-specific methods across 9 HSI restoration tasks, including denoising, deblurring, super-resolution, inpainting, dehazing, and band completion.

## Background & Motivation

Hyperspectral images (HSIs) provide higher spectral resolution than RGB images and are critical for urban planning, agriculture, and environmental monitoring. However, HSIs are susceptible to various degradations during acquisition, each of which affects spectral characteristics differently:

- **Noise** increases spectral fluctuations
- **Compression** reduces spectral reflectance in certain bands
- **Haze** globally shifts spectral curves

Three major limitations of existing methods:

**Task-specific methods lack generalizability**: Methods designed for specific degradations (e.g., denoising, super-resolution, dehazing) fail to generalize to other degradation scenarios, requiring multiple specialized models in practice.

**Existing all-in-one methods are ill-suited for HSI**:
   - **Visual prompt** methods (e.g., PromptIR) offer poor interpretability
   - **Text prompt** methods (e.g., InstructIR) suffer from text-image semantic gaps
   - **Bimodal prompt** methods (e.g., DACLIP-IR) rely on pretrained VLMs with limited capacity for HSI degradation modeling
   - All of the above **neglect HSI spectral characteristics**, frequently causing spectral distortion

**HSI-specific all-in-one methods are insufficient**: Diffusion-based methods such as DDS2M and HIR-Diff suffer from slow inference and complex hyperparameter tuning; PromptHSI relies solely on text prompts and lacks spectral guidance.

Core motivation: To design a unified framework that simultaneously leverages **spectral priors, degradation semantics, and fine-grained visual features** to effectively restore HSIs across diverse degradation types and intensities.

## Method

### Overall Architecture

MP-HSIR adopts a three-level encoder-decoder architecture:
- A degraded HSI $\mathcal{Y} \in \mathbb{R}^{H \times W \times B}$ is first processed by a single convolutional layer to extract shallow features
- A three-level hierarchical encoder progressively downsamples the features, with each level containing multiple Prompt-Guided Spatial-Spectral Transformer Blocks (PGSSTB)
- Text-Visual Synergistic Prompt (TVSP) modules are embedded at encoder-decoder skip connections
- The final output is produced via a $3 \times 3$ convolution with an image-level residual connection

### Key Designs

1. **Prompt-Guided Spatial-Spectral Transformer Block (PGSSTB)**:

    - **Function**: Serves as the core building block, integrating spatial self-attention (SSA) and prompt-guided dual-branch spectral self-attention (PGSSA).
    - **Mechanism**: SSA computes spatial non-local similarities within $P \times P$ sliding windows. PGSSA comprises two complementary branches:
        - **Global spectral self-attention**: Standard channel-wise attention that captures global inter-band dependencies:
       $$A^g = \text{Softmax}\left(\frac{Q^g \cdot K^g}{\epsilon}\right), \quad \text{Attention}(Q^g, K^g, V^g) = W^P(A^g V^g)$$
        - **Prompt-guided local spectral self-attention**: Divides the input into $\frac{HW}{P^2}$ non-overlapping patches, extracting per-patch spectral features $M_j^s \in \mathbb{R}^{1 \times C}$ via global average pooling (GAP). Outputs of both branches are fused via GMLP, enabling complementary global and local spectral modeling.
    - **Design Motivation**: The spectral dimension of HSIs contains rich task-relevant information. The global branch models overall inter-band relationships, while the local branch focuses on spectral reconstruction within spatial regions. Selective fusion via GMLP allows the network to flexibly balance the two.

2. **Spectral Prompt**:

    - **Function**: Provides universal low-rank spectral patterns as prior knowledge for local spectral self-attention.
    - **Mechanism**: Introduces learnable spectral prompts $P_S \in \mathbb{R}^{L \times D}$, where $L$ denotes the number of universal low-rank spectral patterns and $D$ denotes the dimensionality. For local spectral features $M_j^s$, the following operations are applied:
    $Q^l = \text{Softmax}(M_j^s W_1^l) P_S W_3^l$
    $[K^l, V^l] = M_j^s W_2^l W_3^l$
      The query is generated as a weighted combination of spectral prompts, while keys and values are directly projected from local features. This allows any local spectral feature to be expressed as a linear combination of universal low-rank patterns.
    - **Design Motivation**: While different degradations affect spectra differently, **the fundamental low-rank structure of spectra is shared**. Spectral prompts learn these universal patterns during training to guide spectral reconstruction across degradation types. Experiments show that spectral prompt activation patterns remain highly consistent for the same region across different degradations.

3. **Text-Visual Synergistic Prompt (TVSP)**:

    - **Function**: Fuses text and visual prompt information to provide degradation-specific guidance for the restoration process.
    - **Mechanism**:
        - A degradation predictor $\Phi$ classifies the degradation type of the input
        - A frozen CLIP model encodes predefined text descriptions into text prompts $P_T$:
       $$P_T = \Phi(\mathcal{X}) \cdot \text{Clip}(T_{text})$$
        - Learnable visual prompts $P_V$ are introduced and fused with text prompts via cross-attention
        - The fused result is concatenated with encoder features and passed to the decoder via skip connections:
       $$F_l^{out} = \text{Concat}(F_l^e, \text{Attention}(P_T, P_V))$$
    - **Design Motivation**: Text prompts convey global degradation-type information but lack pixel-level precision; visual prompts provide fine-grained local feature information. Cross-attention fusion enables degradation information to operate at both global semantic and local detail levels. Compared to directly using VLMs for feature extraction, TVSP is more flexible and cross-domain adaptable.

### Loss & Training

- L1 loss
- AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.999$)
- Natural scene HSI: batch size 32, 100 epochs, initial lr $2 \times 10^{-4}$, cosine annealing to $1 \times 10^{-6}$
- Remote sensing HSI: batch size 32, 300 epochs, initial lr $1 \times 10^{-4}$, width factor 1.5
- Training patch size: $64 \times 64$; channels: 31 (natural scene), 100 (remote sensing)
- Natural scene and remote sensing data are trained separately

## Key Experimental Results

### Main Results (All-in-one setting, averaged over 9 tasks)

| Task | MP-HSIR | PromptIR (strongest all-in-one) | Strongest Task-Specific | Note |
|------|---------|--------------------------------|------------------------|------|
| Gaussian denoising (ICVL) | **41.62/0.964** | 40.25/0.953 | LDERT: 41.92/0.969 | Approaches task-specific |
| Complex denoising (ICVL) | **42.29/0.971** | 41.29/0.965 | LDERT: 43.42/0.977 | Approaches task-specific |
| Gaussian deblurring (ICVL) | **48.07/0.990** | 47.67/0.990 | MLWNet: 47.66/0.990 | Surpasses task-specific |
| Super-resolution (ARAD) | 38.25/0.924 | 37.37/0.918 | **PIP: 38.36/0.926** | Close to task-specific |
| Inpainting (ICVL) | **51.53/0.996** | 46.38/0.990 | Restormer: 45.79/0.990 | Large margin over all |
| Dehazing (PaviaU) | **39.59/0.986** | 37.41/0.982 | SCANet: 36.59/0.978 | Large margin over all |
| Band completion (ARAD) | **56.48/0.999** | 46.60/0.994 | InstructIR: 51.31/0.997 | +5.17 dB |

### Ablation Study

| Configuration | PSNR | SSIM | Params (M) | Note |
|--------------|------|------|-----------|------|
| Baseline (spatial SA only) | 39.24 | 0.963 | 20.93 | Baseline |
| + Text prompt $P_T$ | 39.62 | 0.964 | 21.51 | +0.38 dB |
| + Visual prompt $P_V$ | 39.57 | 0.964 | 23.68 | +0.33 dB |
| + $P_T$ + $P_V$ | 39.90 | 0.964 | 24.26 | Synergistic effect |
| + Global spectral SA + $P_T$ + $P_V$ | 40.63 | 0.969 | 30.07 | Spectral SA is critical |
| + Local spectral SA + $P_T$ + $P_V$ + $P_S$ | 41.05 | 0.971 | 25.10 | Spectral prompt largest gain |
| **Full Model** | **41.98** | **0.974** | 30.91 | Complete model |

### Key Findings

- **Cross-degradation robustness of spectral prompts**: Visualizations show that spectral prompt activation patterns remain highly consistent for the same region across different degradation types (noise vs. blur), confirming that spectral prompts capture universal spectral structures.
- **Strong generalization**: On two unseen tasks—motion deblurring (few-shot) and Poisson denoising (zero-shot)—fine-tuning with only 5% of data surpasses all all-in-one methods; on Poisson denoising, the model even outperforms several task-specific methods.
- **Dominant performance** on inpainting, dehazing, and band completion, with a maximum margin exceeding 5 dB.
- Spectral error analysis confirms that MP-HSIR achieves the **most accurate spectral reconstruction** across all tasks.

## Highlights & Insights

- **Three-level prompt design**: Spectral prompts (low-level physical priors) + text prompts (high-level semantics) + visual prompts (mid-level details) cover the full spectrum of information needed for restoration tasks.
- **Low-rank spectral patterns**: Representing spectral structure as learnable low-rank pattern combinations is conceptually elegant and effective—analogous to an end-to-end differentiable dictionary learning approach.
- **Broad task coverage**: A single model handles 9 degradation types (denoising ×2, deblurring ×2, super-resolution, inpainting, dehazing, band completion, Poisson denoising), far exceeding existing all-in-one methods.
- The visualization of **spectral prompt activation consistency** provides convincing empirical validation.
- The TVSP module is more lightweight and flexible than directly employing VLMs.

## Limitations & Future Work

- Natural scene and remote sensing HSIs require separate training; true cross-domain unification has not been achieved.
- The degradation predictor $\Phi$ requires pretraining and depends on classification accuracy.
- The frozen CLIP model may not fully capture HSI-specific semantic information.
- The number of low-rank patterns $L$ in the spectral prompts must be set manually.
- The training patch size of $64 \times 64$ is relatively small, potentially limiting the modeling of large-scale degradations.
- Inference speed has not been analyzed in detail.

## Related Work & Insights

- The spectral prompt design is generalizable to other restoration problems with known prior structure (e.g., frequency priors in MRI, scattering priors in remote sensing SAR).
- The text-visual fusion strategy of TVSP can inspire multimodal-guided universal image restoration.
- The challenges of extending all-in-one restoration from RGB to HSI underscore the importance of explicit spectral dimension modeling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The tri-modal prompt framework is novel; introducing spectral prompts as low-rank priors is a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 tasks, 13 datasets, and 3 evaluation settings (all-in-one / generalization / real-world); extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed method descriptions; some equations require cross-referencing with figures to fully understand.
- Value: ⭐⭐⭐⭐⭐ Establishes a multi-prompt all-in-one paradigm for HSI restoration with significant performance improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](exploiting_diffusion_prior_for_task-driven_image_restoration.md)

</div>

<!-- RELATED:END -->
