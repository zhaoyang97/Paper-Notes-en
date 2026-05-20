---
title: >-
  [Paper Note] DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation
description: >-
  [ICLR 2026][Image Generation][online handwriting generation] This paper proposes DiffInk, the first latent diffusion Transformer framework for full-line handwriting generation…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "online handwriting generation"
  - "latent diffusion transformer"
  - "VAE regularization"
  - "glyph-style disentanglement"
  - "Chinese handwriting"
date: 2026-05-08
content_hash: b679cc9a2d73ca50
---

# DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation

**Conference**: ICLR 2026
**arXiv**: [2509.23624](https://arxiv.org/abs/2509.23624)  
**Code**: [https://github.com/awei669/DiffInk](https://github.com/awei669/DiffInk)  
**Area**: Diffusion Models / Handwriting Generation
**Keywords**: online handwriting generation, latent diffusion transformer, VAE regularization, glyph-style disentanglement, Chinese handwriting

## TL;DR
This paper proposes DiffInk, the first latent diffusion Transformer framework for full-line handwriting generation, comprising InkVAE (which learns a structured latent space via dual regularization from OCR and style classification) and InkDiT (which performs conditional denoising generation in the latent space). DiffInk substantially outperforms the state of the art on Chinese handwriting generation (AR 94.38% vs. 91.48%) while achieving an 800× speedup.

## Background & Motivation

**Background**: Text-to-online-handwriting generation (TOHG) has primarily focused on character- or word-level generation. OLHWG (ICLR'25) represents the latest state of the art, generating individual characters and assembling them into lines via an external layout module.

**Limitations of Prior Work**: (a) Character-level methods lack holistic structural modeling when assembling characters into lines, resulting in unnatural boundaries; (b) layout modules introduce additional errors, and decoupling layout from characters ignores structural dependencies between adjacent characters; (c) OLHWG generates only 0.07 characters per second, making it extremely inefficient.

**Key Challenge**: There exists a fundamental gap between character-level modeling and line-level coherence. In real handwriting, adjacent characters are tightly coupled in terms of morphology, spacing, and stroke connectivity—properties that character-by-character generation followed by concatenation cannot capture.

**Goal**: To directly model full-line handwriting generation in an end-to-end manner, ensuring both glyph accuracy and style consistency.

**Key Insight**: The paper first compresses full-line handwriting sequences into a compact latent space via a VAE, then performs conditional generation in that latent space using a DiT. The key innovation lies in dual regularization of the VAE latent space—ensuring the latent space is not merely reconstructive but semantically structured.

**Core Idea**: Good VAE reconstruction does not imply a well-structured latent space. Dual regularization via OCR and style classification imposes glyph/style semantic structure on the latent space, which is critical for robust diffusion-based generation.

## Method

### Overall Architecture

DiffInk = InkVAE (pretrained) + InkDiT (conditional generation)
- **InkVAE**: Encodes a handwriting sequence $X \in \mathbb{R}^{N \times 5}$ (coordinates + pen states) into a latent representation $x \in \mathbb{R}^{l \times d}$
- **InkDiT**: Denoises in the latent space conditioned on text content $Z$ and a reference style $x_{\text{ref}}$, producing a clean latent $\hat{x}_0$
- **Inference**: Starting from Gaussian noise, DDIM sampling yields the latent representation, which InkVAE decodes into a handwriting trajectory

### Key Designs

1. **InkVAE Dual Regularization**:

    - Function: Imposes OCR and style classification constraints on the VAE latent space
    - Mechanism: $\mathcal{L}_{\text{VAE}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}} + \lambda_{\text{ocr}} \mathcal{L}_{\text{ocr}} + \lambda_{\text{sty}} \mathcal{L}_{\text{sty}}$. The OCR module consists of a Transformer recognition head with CTC loss; the style module uses an LSTM with attention pooling and a classification loss. Both operate directly on the latent space.
    - Design Motivation: A vanilla VAE can achieve near-perfect reconstruction (AR 97.59%), yet its latent space lacks semantic structure—small perturbations introduced by the diffusion process lead to content errors or style drift. Dual regularization encourages clustering of identical characters and consistent styles, making the latent space more robust for diffusion-based generation.

2. **InkDiT Conditioning Design**:

    - Function: Injects text content and reference style as two conditioning signals into the diffusion model
    - Mechanism: Text is embedded via a learnable codebook and a ConvNeXt-V2 content encoder to produce $Z$; style is encoded from a reference trajectory via the InkVAE encoder to produce $x_{\text{ref}}$. The three components (noisy latent / content / style) are concatenated along the channel dimension and fed into the DiT.
    - Design Motivation: Large-kernel depthwise convolutions (ConvNeXt-V2) provide a wide receptive field to capture long-range dependencies and handle variable-length text alignment. Using the VAE encoder for style extraction avoids the need for a separate style encoder.

3. **Line-Level Modeling**:

    - Function: Directly models the full-line handwriting trajectory without decomposing it into individual characters
    - Mechanism: A 1D convolutional encoder compresses the full-line sequence; the decoder predicts coordinates via a Gaussian Mixture Model (GMM) combined with pen-state classification.
    - Design Motivation: Line-level modeling naturally captures inter-character spacing, ligatures, and layout dependencies, eliminating concatenation artifacts.

### Loss & Training

- InkVAE: 100 epochs, batch size 128, lr $5 \times 10^{-4}$
- InkDiT: 200k steps, batch size 256, lr $7.5 \times 10^{-5}$
- Diffusion objective: predicting the clean latent $x_0$ (rather than noise), with a masked MSE loss

## Key Experimental Results

### Main Results (CASIA-OLHWDB 2.0–2.2, Chinese Handwriting)

| Method | Output Level | AR% ↑ | CR% ↑ | Style ↑ | DTW ↓ | Speed (char/s) ↑ |
|--------|-------------|-------|-------|---------|-------|-----------------|
| OLHWG (ICLR'25) | Char + Layout | 91.48 | 91.71 | 44.74 | 1.326 | 0.07 |
| SDT (CVPR'23) | Character | 82.53 | 83.00 | 50.51 | 1.270 | 3.35 |
| **DiffInk (Ours)** | **Line** | **94.38** | **94.58** | **77.38** | **1.049** | **58.47** |

### Ablation Study (Effect of InkVAE Regularization on InkDiT)

| Configuration | VAE AR% | DiT AR% | DiT Style ↑ |
|--------------|---------|---------|-------------|
| $\mathcal{L}_{\text{rec+kl}}$ only | 97.59 | 74.77 | 60.68 |
| + $\mathcal{L}_{\text{ocr}}$ | 97.60 | 82.09 | 66.07 |
| + $\mathcal{L}_{\text{sty}}$ | 97.59 | 79.64 | 68.99 |
| + both (InkVAE) | **97.65** | **94.38** | **77.38** |

### Key Findings
- **VAE reconstruction performance is nearly unaffected by regularization** (AR 97.59→97.65), yet diffusion generation performance surges from 74.77% to 94.38%—demonstrating that latent space structure, not reconstruction quality, is the key to generation performance.
- The combined effect of dual regularization far exceeds the sum of individual contributions (82.09 and 79.64 individually vs. 94.38 jointly), indicating a synergistic effect from the joint disentanglement of glyph and style.
- Line-level modeling is 800× faster than OLHWG (58.47 vs. 0.07 char/s), as it directly generates entire lines without per-character diffusion and layout prediction.
- Style consistency score improves from 44.74 to 77.38 (+32.64), demonstrating that line-level modeling preserves stylistic coherence far better than concatenation-based approaches.

## Highlights & Insights
- **"Good VAE reconstruction ≠ good latent space"** is a profound insight. It explains why many VAE-based latent diffusion models underperform—the latent space may reconstruct accurately yet lack semantic structure, causing small perturbations during diffusion to produce uncontrollable semantic changes. This insight is transferable to latent diffusion models in image, video, 3D, and other modalities.
- **Lightweight latent space regularization** (requiring only two small classifiers) substantially improves downstream generation quality at minimal cost—a generalizable design principle.
- The contrast between **line-level modeling and character-level concatenation** highlights the inherent advantage of end-to-end approaches in capturing global dependencies.

## Limitations & Future Work
- Validation is limited to Chinese handwriting; experiments on other writing systems (e.g., English, Arabic) are absent.
- The method requires a predefined character codebook, and generalization to open-vocabulary or rare characters has not been tested.
- Reference style still requires handwriting samples from the target writer; zero-shot style generation from descriptions is not supported.
- The absolute DTW value (1.049) leaves room for further improvement.
- Although 800× faster than OLHWG, the generation speed is only approximately 2× faster than autoregressive methods (WLU, 25 char/s).

## Related Work & Insights
- **vs. OLHWG**: DiffInk performs direct line-level generation, eliminating the additional errors introduced by the layout module. It outperforms OLHWG across all metrics, with a particularly notable 800× speedup.
- **vs. SDT**: SDT employs dual-branch contrastive learning for content/style disentanglement, whereas DiffInk applies regularization in the latent space—a simpler approach that yields superior results.
- **vs. Image-Domain LDM**: DiffInk demonstrates that latent diffusion is equally effective for sequential data (handwriting trajectories), and highlights the potentially overlooked importance of latent space regularization.

## Rating
- Novelty: ⭐⭐⭐⭐ First line-level handwriting latent diffusion framework; the latent space regularization insight is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations, but evaluated on only one dataset (Chinese).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear figures and tables; motivation–method–experiment narrative is coherent.
- Value: ⭐⭐⭐⭐ Represents a significant advance for handwriting generation; the latent space regularization principle has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [\[AAAI 2026\] AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer](../../AAAI2026/image_generation/anostyler_text-driven_localized_anomaly_generation_via_light.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](../../ICCV2025/image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
