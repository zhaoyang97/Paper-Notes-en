---
title: >-
  [Paper Note] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model
description: >-
  [ICCV 2025][Image Generation][Unsupervised deblurring] TP-Diff is the first work to introduce diffusion models into unpaired image deblurring. It learns spatially varying texture priors via a memory-augmented Texture Pri…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Unsupervised deblurring"
  - "diffusion model"
  - "texture prior"
  - "unpaired training"
  - "adaptive filtering"
date: 2026-05-08
content_hash: 9fbb2c1cede6d5d3
---

# Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model

**Conference**: ICCV 2025
**arXiv**: [2507.13599](https://arxiv.org/abs/2507.13599)
**Code**: Unavailable
**Area**: Diffusion Models / Image Restoration
**Keywords**: Unsupervised deblurring, diffusion model, texture prior, unpaired training, adaptive filtering

## TL;DR

TP-Diff is the first work to introduce diffusion models into unpaired image deblurring. It learns spatially varying texture priors via a memory-augmented Texture Prior Encoder (TPE), and designs a Filter-Modulated Multi-head Self-Attention (FM-MSA) to leverage these priors for precise deblurring, achieving a new unsupervised state-of-the-art on multiple benchmarks with only 11.89M parameters.

## Background & Motivation

**Background**: Image deblurring is a classic low-level vision task. Supervised methods rely on synthetic blurry–sharp paired data and achieve strong performance but generalize poorly to real-world scenarios. Real paired data (e.g., acquired via dual-camera systems) is expensive to collect and subject to camera-specific bias. Learning deblurring directly from unpaired data is therefore of greater practical value.

**Limitations of Prior Work**: Existing unsupervised deblurring methods fall into three main paradigms: (1) prior-based methods struggle to handle diverse blur types with a single prior; (2) reblurring-augmentation methods depend on pretrained supervised models; (3) CycleGAN-based methods learn mappings by constructing blurry–sharp cycles but ignore the spatial diversity of blur, making them prone to overfitting a single blur template. A common weakness across all paradigms is the inability to effectively model spatially varying complex blur patterns found in real-world images.

**Key Challenge**: Under the unpaired setting, pixel-level supervision is absent, making it difficult for models to learn spatially discriminative deblurring strategies—the model must "know" that different regions exhibit blur of different degrees and types, and restore texture accordingly.

**Goal**: To enable a diffusion model to learn spatially varying texture priors from unpaired data, and to use those priors to guide the deblurring network toward region-adaptive texture restoration.

**Key Insight**: Diffusion models excel at modeling complex data distributions. Rather than directly generating sharp images with a diffusion model (which tends to introduce artifacts), the authors train the diffusion model to generate an intermediate representation—a texture prior—that encodes the texture information each region should possess, which a dedicated deblurring network then uses to complete the final deblurring.

**Core Idea**: A memory-augmented encoder extracts texture priors from unpaired sharp images; a diffusion model is trained to generate these priors from noise; and an adaptive-filtering Transformer exploits the priors to remove spatially varying blur.

## Method

### Overall Architecture

TP-Diff comprises four core components: the Texture Prior Encoder (TPE), a Diffusion Model (DM), a deblurring network, and a reblurring network. The deblurring and reblurring networks form a cycle structure for unsupervised training. Training proceeds in two stages. In Stage 1, the TPE extracts texture priors $z$ from unpaired data while the full cycle structure is trained jointly. In Stage 2, the TPE parameters are frozen and the DM is trained to generate texture priors $\hat{z}$ from noise, while the cycle structure continues to be optimized. At inference, only the blurry image is required as input: the DM generates the texture prior from pure Gaussian noise, and the deblurring network uses this prior to produce the sharp output.

### Key Designs

1. **Texture Prior Encoder (TPE)**:

    - **Function**: Extracts spatially varying texture priors from unpaired blurry and sharp images.
    - **Mechanism**: TPE consists of a texture-memory enhancement module and a texture-memory transfer module. The enhancement module initializes a set of learnable memory vectors $\mathcal{M} \in \mathbb{R}^{N \times L}$ ($N=256$) and enriches them using texture-rich tokens from sharp images—writing texture templates from sharp images into memory via an attention mechanism. The transfer module encodes the blurry image into tokens $z_b$, then retrieves from the enhanced memory the texture representation most similar to each blurry token (selecting the entry with the highest attention score rather than computing a weighted sum), and assembles the output prior $z \in \mathbb{R}^{H \times W \times C}$. Each spatial position in the prior thus encodes the texture information appropriate for that region.
    - **Design Motivation**: The memory mechanism allows the model to accumulate texture knowledge across all training sharp images rather than relying on a single paired sample. Selecting the most relevant entry rather than a weighted sum ensures spatially precise matching. Experiments confirm that the TPE is robust to different sampling strategies for unpaired sharp images.

2. **Filter-Modulated Multi-head Self-Attention (FM-MSA) in the Texture Transfer Transformer Layer (TTformer)**:

    - **Function**: Adaptively removes spatially varying blur by leveraging the texture prior.
    - **Mechanism**: FM-MSA uses the texture prior $z$ to predict the offset $\Delta p$ and weight $\Delta m$ of adaptive filters ($\Delta p, \Delta m = \text{Convs}(z)$). These filter parameters encode the blur type at each spatial location, yielding spatially adaptive filtered features $\tilde{\mathcal{F}}$. The original features then serve as Queries while the filtered features serve as Keys/Values in a transposed attention operation, enabling adaptive blur removal. TTformer also includes a TM-FFN module that generates dynamic modulation parameters $\gamma, \varphi$ from the prior $z$ to regulate local feature aggregation. The entire TTformer block is reused across multiple scales within the deblurring network.
    - **Design Motivation**: Standard self-attention is spatially invariant and cannot treat different regions with different blur differently. Prior-driven adaptive filtering endows the model with the ability to apply tailored deblurring strategies at different spatial locations.

3. **Wavelet-based Adversarial Loss**:

    - **Function**: Preserves high-frequency texture details during training.
    - **Mechanism**: Wavelet transforms are applied to the deblurred and sharp images to extract high-frequency components $\Phi(\cdot)$, and an adversarial loss is applied in the high-frequency domain: $\mathcal{L}_{Wave} = \mathbb{E}[\log D_S(\Phi(s))] + \mathbb{E}[\log(1 - D_S(\Phi(DN(b))))]$. The discriminator operates in the wavelet high-frequency domain rather than the pixel domain, compelling the deblurring network to preserve authentic texture structure instead of producing blurry but "safe" low-frequency outputs.
    - **Design Motivation**: Standard cycle-GAN losses tend to cause high-frequency texture loss. Wavelet transforms naturally decompose signals into frequency bands, making adversarial training on high-frequency subbands an efficient strategy for texture preservation.

### Loss & Training

The Stage 1 loss is $\mathcal{L}_{s1} = \lambda_{GAN}\mathcal{L}_{GAN} + \lambda_{CYC}\mathcal{L}_{CYC} + \lambda_{Wave}\mathcal{L}_{Wave}$, with $\lambda_{GAN}=1$, $\lambda_{CYC}=0.1$, $\lambda_{Wave}=0.2$. Stage 2 adds the diffusion loss: $\mathcal{L}_{s2} = \mathcal{L}_{s1} + \lambda_{diff}\|\hat{z} - z\|_1$. The diffusion step count is $T=8$; each stage is trained for 200 epochs using the Adam optimizer with learning rate $10^{-4}$, batch size 8, and patch size 256×256.

## Key Experimental Results

### Main Results

Unpaired training comparison on GoPro, HIDE, RealBlur-R, and RealBlur-J (PSNR / SSIM):

| Method | GoPro PSNR | GoPro SSIM | HIDE PSNR | RealBlur-R PSNR | Params |
|--------|-----------|-----------|----------|----------------|--------|
| CycleGAN | 22.54 | 0.757 | 21.81 | 12.38 | 11.38M |
| FCL-GAN | 24.59 | 0.831 | 23.43 | 28.37 | 24.56M |
| UCL | 25.06 | 0.839 | 23.85 | 30.53 | 19.45M |
| **TP-Diff** | **28.13** | **0.903** | **26.70** | **34.95** | **11.89M** |
| SEMGUD | 29.06 | 0.927 | 27.64 | 35.51 | 67.9M |
| **TP-Diff-se** | **30.16** | **0.934** | **28.21** | **35.32** | **11.89M** |

TP-Diff achieves the best performance among unpaired methods with the fewest parameters (11.89M). With the same self-enhancement strategy as SEMGUD (TP-Diff-se), it surpasses SEMGUD by 1.1 dB on GoPro while using only 1/6 of the parameters.

### Ablation Study

| Configuration | PSNR | SSIM | Note |
|---------------|------|------|------|
| Full model | 28.13 | 0.903 | Complete TP-Diff |
| w/o DM | 26.46 | 0.867 | Remove diffusion model; −1.67 dB (largest drop) |
| w/o TPE | 27.36 | 0.886 | Remove texture prior encoder; −0.77 dB |
| w/o TTformer | 27.19 | 0.884 | Remove texture transfer Transformer; −0.94 dB |
| w/o Multi-Scale | 27.89 | 0.896 | Remove multi-scale learning |
| w/o Joint-Train | 27.65 | 0.896 | DM not jointly trained |
| w/o WaveLoss | 28.01 | 0.899 | Remove wavelet adversarial loss |

### Key Findings

- The diffusion-model-generated texture prior is the most critical component (−1.67 dB when removed), substantially outperforming alternative generative approaches such as Memory Bank, Sparse Coding, and Vanilla VQ.
- The memory mechanism in TPE is robust to different sampling strategies for unpaired sharp images (random, clustered, single repeated), owing to the global information accumulated in the learnable memory.
- A diffusion step count of $T=8$ is sufficient to reconstruct high-quality texture priors; increasing steps yields diminishing returns.
- Adaptive filtering via FM-MSA outperforms variants without filtering, with plain filtering, with deformable filtering, and with separable filtering.
- TP-Diff surpasses all unpaired methods on the real-blur datasets RB2V_Street and RSBlur, demonstrating strong generalization.

## Highlights & Insights

- **First application of diffusion models to unpaired image restoration**: This challenges the conventional assumption that diffusion models require paired supervision. The key insight is to have the DM generate an intermediate texture prior rather than the final sharp image, elegantly circumventing the absence of paired labels.
- **Elegantly designed memory-augmented texture encoder**: The learnable memory serves as a texture "knowledge base" that accumulates information across training samples; retrieval then matches each blurry region to the most suitable texture template. This write-then-retrieve mechanism is transferable to any task requiring domain knowledge to be learned from unpaired data.
- **Prior-driven spatially adaptive filtering strategy**: Using the texture prior to predict filter parameters enables the model to apply different deblurring strategies at different spatial locations. This paradigm generalizes naturally to other spatially varying degradation tasks such as dehazing and deraining.

## Limitations & Future Work

- The spatial resolution of the texture prior matches the input image, so the computational cost of the diffusion model scales with high-resolution inputs.
- The reblurring network uses a standard UNet whose blur simulation capacity is limited; improving it may further boost performance.
- The framework is primarily validated on motion blur; its applicability to other types such as defocus blur is not fully explored.
- The two-stage training pipeline is relatively complex; future work could explore end-to-end single-stage optimization.
- Extending the texture prior to a compact fixed-size representation could avoid resolution-dependent computational growth.

## Related Work & Insights

- **vs. HiDiff**: HiDiff also uses a DM to generate auxiliary priors for deblurring, but those priors are spatially unordered, of fixed count, and require paired training data. TP-Diff's priors are spatially aligned, share the same resolution as the input, and are learned from unpaired data, yielding stronger representational capacity (TPE: 0.12M parameters vs. HiDiff's 0.44M denoising network).
- **vs. SEMGUD**: SEMGUD introduces a pretrained supervised model (NAFNet at 33.69 dB) via a self-enhancement strategy, raising fairness concerns. TP-Diff trained from scratch reaches 28.13 dB; with the same self-enhancement strategy it reaches 30.16 dB, surpassing SEMGUD's 29.06 dB.
- **vs. CycleGAN-based methods**: Traditional CycleGAN approaches perform global domain translation between blurry and sharp domains, ignoring spatial diversity. TP-Diff achieves spatially adaptive processing through texture priors, improving PSNR by 3–5 dB.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce DMs into unpaired restoration; both the TPE memory-retrieval design and FM-MSA adaptive filtering are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, 16 SOTA comparisons, 7 ablation groups, and hyperparameter sensitivity analysis—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and figures are informative, though the dense equations require careful cross-referencing with the diagrams.
- **Value**: ⭐⭐⭐⭐ A significant advance in unpaired deblurring; the texture prior paradigm is extensible to a broader range of low-level vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DIIP: Diffusion Image Prior](diffusion_image_prior.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[NeurIPS 2025\] BlurDM: A Blur Diffusion Model for Image Deblurring](../../NeurIPS2025/image_generation/blurdm_a_blur_diffusion_model_for_image_deblurring.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](../../NeurIPS2025/image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)

</div>

<!-- RELATED:END -->
