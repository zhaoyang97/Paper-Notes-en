---
title: >-
  [Paper Note] Disentangled Textual Priors for Diffusion-based Image Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][Diffusion-based SR] This paper proposes DTPSR, which disentangles textual priors along two orthogonal dimensions — spatial hierarchy (global/local) and frequency semantics (low-frequency/hi…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Diffusion-based SR"
  - "text guidance"
  - "disentangled priors"
  - "frequency-awareness"
  - "semantic control"
date: 2026-05-08
content_hash: b6229cb5454da576
---

# Disentangled Textual Priors for Diffusion-based Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.07430](https://arxiv.org/abs/2603.07430)
**Code**: [GitHub](https://github.com/JL6666JL/DTPSR)
**Area**: Image Super-Resolution
**Keywords**: Diffusion-based SR, text guidance, disentangled priors, frequency-awareness, semantic control

## TL;DR

This paper proposes DTPSR, which disentangles textual priors along two orthogonal dimensions — spatial hierarchy (global/local) and frequency semantics (low-frequency/high-frequency) — and constructs a disentangled cross-attention injection pipeline along with a multi-branch CFG strategy, achieving superior perceptual quality in diffusion-based image super-resolution.

## Background & Motivation

Diffusion models (e.g., Stable Diffusion) have demonstrated strong generative capabilities in image super-resolution, yet their performance critically depends on how semantic priors are constructed and injected. Existing methods exhibit two categories of limitations:

**Insufficient semantic granularity**: Local-tag methods (SeeSR) focus on fine details but lack global consistency; global-description methods (SUPIR, PASD) capture global structure but neglect fine-grained details.

**Entangled frequency information**: Existing methods embed structural information (low-frequency: shape, layout) and texture information (high-frequency: edges, material) within the same representation, leading to insufficient semantic controllability and interpretability.

**Hallucination under severe degradation**: Without disentangled semantic guidance, diffusion models are prone to hallucinations, such as misinterpreting a wall as an ocean texture.

Core insight: Disentangling textual priors along two orthogonal dimensions — **spatial hierarchy** and **frequency semantics** — enables the model to simultaneously capture scene-level structure and object-level detail.

## Method

### Overall Architecture

Given a low-resolution image $x_{lr}$, the DTPSR pipeline proceeds as follows:
1. A VAE encoder maps $x_{lr}$ to the latent space $z_0$.
2. Forward diffusion adds noise to obtain $z_t$.
3. During reverse denoising, semantic injection is performed sequentially through four dedicated cross-attention modules:

$$z_t \xrightarrow{\text{GTCA}} z_t^g \xrightarrow{\text{LFCA}} z_t^{lf} \xrightarrow{\text{HFCA}} z_t^{hf} \xrightarrow{\text{LRCA}} z_{t-1}$$

### Key Designs

1. **Global Text Cross-Attention (GTCA)**: The global description $c_g$ is encoded by CLIP into $e_g$ and injected into the latent variable via cross-attention to establish scene-level structural foundations. Design motivation: establishing global layout first (e.g., "an indoor scene with 3 objects") before progressive refinement.

2. **Low-Frequency Cross-Attention (LFCA)**: A set of low-frequency local descriptions $\{c_{lf}^{(i)}\}$ (shape, size, spatial arrangement) is encoded and injected into $z_t^g$ for object-level structural enhancement:
$$E_{lf} = [\text{CLIP\_TextEnc}(c_{lf}^{(1)}), \dots, \text{CLIP\_TextEnc}(c_{lf}^{(n)})]$$
Design motivation: low-frequency information governs structural fidelity; separating it from high-frequency texture prevents mutual interference.

3. **High-Frequency Cross-Attention (HFCA)**: High-frequency descriptions $\{c_{hf}^{(j)}\}$ (texture, edges, surface details) are encoded and injected on top of the LFCA output:
$$z_t^{hf} = \text{HFCA}(z_t^{lf}, E_{hf})$$
Design motivation: high-frequency information governs visual realism; independent injection enables precise control over texture generation.

4. **Low-Resolution Feature Cross-Attention (LRCA)**: A frozen DAPE encoder extracts visual features $f_{lr}$ from $x_{lr}$, which are injected via cross-attention to anchor identity consistency with the input image and prevent semantic drift.

5. **Multi-branch Classifier-Free Guidance (Multi-branch CFG)**: Independent negative prompts $c_g^{\text{neg}}, c_{lf}^{\text{neg}}, c_{hf}^{\text{neg}}$ are designed for the global, low-frequency, and high-frequency branches respectively, enabling disentangled semantic suppression:
$$\tilde{\epsilon} = \hat{\epsilon} + \lambda_s(\hat{\epsilon} - \hat{\epsilon}_{\text{neg}})$$
Design motivation: a single negative prompt cannot simultaneously address hallucinations from multiple semantic sources.

### Loss & Training

- **Training loss**: Standard noise prediction MSE loss
$$\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, z_{lr}, t, c_g, c_{lf}, c_{hf})\|_2^2]$$
- **Dataset DisText-SR**: ~95K image-text pairs built on LSDIR + the first 10K images from FFHQ, with disentangled descriptions generated via Mask2Former segmentation + LLaVA captioning.
- **Backbone**: SD-2-base; DAPE encoder for LR embedding extraction.
- **Training configuration**: AdamW optimizer, lr $5 \times 10^{-5}$, batch size 32, 110K iterations, 4× A800 GPUs.
- **Inference**: DDPM 50 steps, guidance scale $\lambda_s = 7.0$.

## Key Experimental Results

### Main Results

| Dataset | Metric | DTPSR | FaithDiff | SUPIR | Gain |
|--------|------|-------|-----------|-------|------|
| DIV2K-Val | MUSIQ↑ | **71.24** | 69.18 | 62.59 | +2.06 |
| DIV2K-Val | MANIQA↑ | **0.5866** | 0.4309 | 0.5224 | +0.0642 |
| DIV2K-Val | CLIPIQA↑ | **0.7549** | 0.6463 | 0.7040 | +0.0509 |
| RealSR | MUSIQ↑ | **71.84** | 68.86 | 58.51 | +2.98 |
| RealSR | MANIQA↑ | **0.6021** | 0.4644 | 0.4429 | +0.0432 |
| DRealSR | CLIPIQA↑ | **0.7640** | 0.6335 | 0.6307 | +0.0729 |

Note: DTPSR achieves state-of-the-art performance on all no-reference perceptual metrics, though PSNR/SSIM are lower than GAN-based methods (perception–distortion tradeoff).

### Ablation Study

| Configuration | MANIQA↑ | CLIPIQA↑ | MUSIQ↑ | Note |
|------|---------|----------|--------|------|
| No prior | 0.5271 | 0.7064 | 67.48 | Baseline |
| Local only | 0.5851 | 0.7471 | 68.86 | Local prior contributes more |
| Global only | 0.5394 | 0.7211 | 67.80 | Global provides moderate gain |
| Global + Local | **0.6011** | **0.7640** | **69.24** | Complementary effect is optimal |
| Frequency entangled | 0.5947 | 0.7527 | 69.05 | Disentangled outperforms entangled |
| Frequency disentangled | **0.6011** | **0.7640** | **69.24** | Separate injection is more effective |

### Key Findings

- Local priors contribute substantially more than global priors (MANIQA gain: 0.0580 vs. 0.0123).
- Frequency disentanglement consistently outperforms frequency mixing across all metrics.
- Multi-branch CFG significantly improves perceptual quality over single or no CFG (MUSIQ: 66.73→69.24).
- Even when text descriptions are randomly corrupted (replaced with "None"), DTPSR still outperforms competing methods, demonstrating robustness.
- With 10.5B parameters and 14.94s/image inference time, DTPSR is more efficient than SUPIR (17.8B) and FaithDiff (15.6B).

## Highlights & Insights

1. **Elegant disentanglement design**: Decomposing textual priors along spatial hierarchy × frequency semantics as two orthogonal dimensions is conceptually clear and empirically effective.
2. **DisText-SR dataset**: The first large-scale SR dataset with combined global–local and low-frequency–high-frequency textual annotations, providing a foundation for controllable super-resolution research.
3. **Multi-branch CFG strategy**: Achieves fine-grained hallucination suppression via frequency-aware negative prompts without additional training.
4. **Robustness experiments**: The system remains functional even when upstream modules (segmentation, captioning) produce imperfect outputs.

## Limitations & Future Work

1. Full-reference metrics such as PSNR/SSIM fall short of GAN-based methods, reflecting the inherent perception–distortion tradeoff.
2. The framework depends on the quality of upstream segmentation (Mask2Former) and captioning (LLaVA) models.
3. Running segmentation and caption generation at inference time introduces additional end-to-end latency.
4. Only the top-3 largest segmentation regions are processed, potentially missing small but semantically important regions.
5. Future directions include adaptive prompt correction, tighter integration with upstream modules, and more efficient diffusion backbones.

## Related Work & Insights

- **SeeSR**: Employs local semantic tags but focuses solely on details, lacking global consistency.
- **SUPIR/PASD/FaithDiff**: Utilize global descriptions but neglect frequency separation.
- **StableSR/DiffBIR**: Do not leverage textual semantics, thus failing to fully exploit diffusion priors.
- Insight: The disentangled textual prior paradigm is potentially generalizable to other conditional generation tasks such as image editing and inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ The spatial–frequency dual disentanglement design for textual priors is novel; the multi-branch CFG strategy is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-metric evaluation with comprehensive ablations (global/local, frequency, CFG, robustness).
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, architectural diagrams are informative, and experiments are well-organized.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for text-guided diffusion super-resolution; the DisText-SR dataset offers practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)

</div>

<!-- RELATED:END -->
