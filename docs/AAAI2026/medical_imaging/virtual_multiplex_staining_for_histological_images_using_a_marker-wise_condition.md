---
title: >-
  [Paper Note] Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model
description: >-
  [AAAI 2026][Medical Imaging][Virtual multiplex staining] This paper proposes a virtual multiplex staining framework based on a marker-wise conditioned diffusion model. Through a two-stage training procedure (marker-wise conditional diffusion learning followed by pixel-level fine-tuning), it is the first method to generate multiplex immunofluorescence images of up to 18 distinct markers from a single H&E image, achieving state-of-the-art performance on two public datasets, HEMIT and Orion-CRC.
tags:
  - AAAI 2026
  - Medical Imaging
  - Virtual multiplex staining
  - conditional diffusion model
  - "H&E-to-immunofluorescence"
  - marker-wise conditional generation
  - latent diffusion model
date: 2026-05-08
content_hash: 8cc4519df5026810
---

# Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model

**Conference**: AAAI 2026
**arXiv**: [2508.14681](https://arxiv.org/abs/2508.14681)
**Code**: N/A
**Area**: Medical Imaging / Pathology
**Keywords**: Virtual multiplex staining, conditional diffusion model, H&E-to-immunofluorescence, marker-wise conditional generation, latent diffusion model

## TL;DR

This paper proposes a virtual multiplex staining framework based on a marker-wise conditioned diffusion model. Through a two-stage training procedure (marker-wise conditional diffusion learning followed by pixel-level fine-tuning), it is the first method to generate multiplex immunofluorescence images of up to 18 distinct markers from a single H&E image, achieving state-of-the-art performance on two public datasets, HEMIT and Orion-CRC.

## Background & Motivation

**State of the Field**: Histopathological analysis relies on H&E staining as the gold standard, supplemented by immunohistochemistry (IHC) for molecular information. Multiplex immunofluorescence (mIF) / multiplex immunohistochemistry (mIHC) imaging has recently emerged as a technique capable of visualizing multiple biomarkers within a single tissue section, providing a more comprehensive view of the tumor microenvironment. However, multiplex imaging protocols are complex and costly, limiting large-scale adoption.

**Limitations of Prior Work**: (1) Most existing H&E image repositories lack corresponding multiplex staining images, precluding retrospective multimodal analysis. (2) Existing virtual staining methods either require training a separate model per marker (e.g., pix2pix), which scales poorly, or support only 2–3 markers (e.g., HEMIT supports 3, VIMs supports 2), far below the demands of practical multiplex imaging. (3) Independent training across methods precludes cross-channel knowledge sharing, wasting complementary information.

**Root Cause**: Multiplex imaging involves a large number of marker types (practical applications often require 18 or more), yet existing methods either necessitate training separate models per marker (computationally infeasible) or rely on text conditioning (whose discriminative capacity degrades as the number of markers increases), making them fundamentally non-scalable.

**Paper Goals**: (1) How to generate a large variety of marker images using a single model? (2) How to address color distortion caused by distributional differences in pixel values across markers? (3) How to achieve high generation quality while enabling efficient inference?

**Starting Point**: The paper leverages the strong generative prior of pretrained Stable Diffusion, replacing text conditioning with marker one-hot embeddings to enable scalable multi-marker discrimination, and decouples multi-target generation from color fidelity via two-stage training.

**Core Idea**: Condition a pretrained LDM with marker one-hot embeddings to achieve scalable multi-marker generation, then apply pixel-level fine-tuning to optimize color fidelity and enable single-step inference.

## Method

### Overall Architecture

Training proceeds in two stages: (1) Stage 1 trains a marker-wise conditional diffusion model in the latent space to learn multi-target generation from H&E to each marker image; (2) Stage 2 applies pixel-level loss fine-tuning to optimize color contrast fidelity and enable single-step inference. At inference time, a single denoising step suffices to generate the target marker image. The backbone is SD v2; the VAE encoder/decoder is frozen and only the U-Net is trained.

### Key Designs

1. **Marker-wise Conditional Diffusion**:

    - **Function**: Enables conditional generation of diverse marker images within a single U-Net architecture.
    - **Mechanism**: The H&E image is encoded into a latent representation $\mathbf{x}$; the target marker image is encoded and noised to obtain $\mathbf{z}_{m,t}$; the concatenation $[\mathbf{x}, \mathbf{z}_{m,t}]$ is used as U-Net input (doubling the input channels). The training objective uses v-prediction: $\mathcal{L}_m = \|\mathbf{v}^*_{m,t} - \hat{\mathbf{v}}_{m,t}\|^2_2$, where $\mathbf{v}^*_{m,t} = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}\mathbf{z}_{m,0}$. Each H&E image is replicated $M$ times and paired with different markers during training; the loss is averaged over all markers: $\mathcal{L}_M = \frac{1}{M}\sum_{m=1}^M \mathcal{L}_m$.
    - **Design Motivation**: Replicating H&E latent representations for all markers ensures balanced parameter updates across markers during training. The weight-copying initialization strategy from Marigold (input channel weights are duplicated and halved) is adopted to preserve the pretrained prior.

2. **Marker One-hot Embedding Conditioning**:

    - **Function**: Enables a single model to distinguish different marker types, achieving scalable multi-marker generation.
    - **Mechanism**: Each marker type is represented by a one-hot vector $c_m$, which is positionally encoded and element-wise added to the timestep embedding before being injected into the U-Net conditioning pathway. This differs from text conditioning—textual descriptions lose discriminative power as the number of markers grows, whereas one-hot vectors are inherently orthogonal.
    - **Design Motivation**: Ablation experiments confirm the criticality of this design. On HEMIT (3 markers), text and one-hot conditioning perform comparably; however, on Orion-CRC (18 markers), text conditioning yields an SSIM of only 0.288 compared to 0.662 for one-hot—text conditioning completely fails at scale. One-hot embeddings provide an unambiguous, linearly scalable marker discrimination signal.

3. **Pixel-level Fine-tuning for Color Fidelity and Single-step Inference**:

    - **Function**: Addresses inherent color distortion in diffusion models while enabling fast single-step inference.
    - **Mechanism**: The timestep is fixed at $t=T$ and stochastic noise is replaced by zero noise ($\epsilon=0$), converting the model from iterative denoising to a single-step mapping. A combined loss is applied in pixel space (via the frozen VAE decoder): $\mathcal{L}_{FT} = \frac{1}{M}\sum_{m=1}^M [(1-\lambda)\|\mathbf{I}^*_m - \hat{\mathbf{I}}_m\|_1 + \lambda\|\mathbf{I}^*_m - \hat{\mathbf{I}}_m\|^2_2]$. Only U-Net parameters are updated.
    - **Design Motivation**: The training data distribution is skewed toward dark background regions (multiplex staining images contain large signal-free areas), causing the model to inaccurately reproduce colors of bright marker signals. Prior to fine-tuning, the model exhibits noticeable color distortion and false positives (e.g., spurious panCK signals). Single-step inference not only accelerates generation but also provides a direct gradient path for pixel-level supervision.

### Loss & Training

Stage 1 trains the U-Net using the standard v-prediction diffusion loss averaged over all markers. Stage 2 applies pixel-level L1+L2 loss fine-tuning, with the hyperparameter $\lambda$ varying by dataset (0.5 for HEMIT, 1.0 for Orion-CRC). Training is conducted on 4× H100 GPUs with SD v2 as the backbone and 512×512 patch inputs. Single-channel marker images are replicated into three channels to be compatible with the pretrained VAE.

## Key Experimental Results

### Main Results

| Method | HEMIT SSIM | HEMIT R | HEMIT PSNR | Orion-CRC SSIM (avg 18) | Orion-CRC R (avg 18) | Orion-CRC PSNR (avg 18) |
|--------|-----------|---------|-----------|------------------------|---------------------|------------------------|
| pix2pix | 0.734 | 0.623 | 27.55 | 0.724 | 0.277 | 33.70 |
| pix2pixHD | 0.709 | 0.755 | 29.19 | — | — | — |
| HEMIT | 0.770 | 0.746 | 28.78 | 0.690 | 0.170 | 33.55 |
| Marigold | 0.686 | 0.750 | 29.36 | — | — | — |
| **Ours** | **0.836** | **0.795** | **30.60** | **0.763** | **0.394** | **35.06** |

### Ablation Study

| Configuration | SSIM | R | PSNR | Notes |
|---------------|------|---|------|-------|
| Orion-CRC text conditioning | 0.288 | −0.003 | 18.01 | Text conditioning fails at 18 markers |
| Orion-CRC one-hot conditioning | 0.662 | 0.371 | 30.66 | One-hot substantially outperforms |
| HEMIT no fine-tuning (50 steps + 10× ensemble) | 0.673 | 0.770 | 30.21 | Multi-step denoising is slow and suboptimal |
| HEMIT single-step inference, no fine-tuning | 0.757 | 0.760 | 29.38 | Single-step already surpasses multi-step |
| **HEMIT single-step + fine-tuning** | **0.836** | **0.795** | **30.60** | Best overall |

### Key Findings

- **Marker scalability is the central contribution**: For the first time, 18 different markers are generated from a single H&E image; prior methods handled at most 3 (HEMIT). On Orion-CRC, the proposed method achieves best SSIM on 13/18 markers, best R on 15/18, and best PSNR on 18/18.
- **One-hot vs. text conditioning**: The two are comparable with 3 markers, but text conditioning collapses entirely with 18 markers (SSIM 0.288 vs. 0.662), confirming that text-based conditioning is not scalable.
- **Pixel-level fine-tuning yields substantial gains**: Fine-tuning significantly improves color fidelity and eliminates false positives (e.g., spurious panCK detections), while reducing inference time from 117 seconds/sample to 0.13 seconds/sample (approximately 900× speedup).
- **Single-step inference outperforms 50-step + 10× ensemble**: SSIM improves from 0.673 to 0.757 (before fine-tuning), primarily because iterative denoising introduces additional artifacts on data distributions skewed toward dark backgrounds.
- Rare markers such as CD31 and FOXP3 (fewer than 1,000 training patches) exhibit lower R values, indicating that data scarcity adversely affects generation quality.

## Highlights & Insights

- **Qualitative leap from 3 to 18 markers**: Prior methods were constrained to 2–3 markers; this work scales directly to 18 using a single model. This is not merely a quantitative increase but represents a genuinely scalable multi-marker generation framework enabled by one-hot embeddings.
- **Two-stage training decouples two conflicting objectives**: Stage 1 acquires multi-target generation capability (diffusion prior + marker discrimination); Stage 2 focuses on color fidelity (pixel-level supervision). This decoupling avoids mode collapse that would result from directly optimizing pixel losses during diffusion training.
- **Unexpected finding regarding single-step inference**: In the diffusion modeling literature, multi-step denoising is generally assumed to outperform fewer steps. However, this work finds that for pathology data with skewed distributions (predominantly dark backgrounds), single-step inference performs better—a finding that may carry implications for diffusion-based applications in other domains such as remote sensing or astronomy.

## Limitations & Future Work

- Inference cost scales linearly with the number of markers—generating 18 markers requires 18 separate single-step inference passes, leaving room for further optimization.
- There is no explicit modeling of inter-marker correlations; spatial co-expression relationships across different markers are not exploited.
- Generated images exhibit some blurriness, particularly under the LDM architecture lacking skip connections (consistent with observations in Parmar et al.).
- Generation quality for rare markers such as CD31 and FOXP3 is limited by training data scarcity, necessitating more effective few-shot learning strategies.

## Related Work & Insights

- **vs. pix2pix (isola2017image)**: pix2pix serves as a baseline but requires an independent model per marker and does not scale. The proposed method handles 18 markers with a single model.
- **vs. HEMIT (bian2024hemit)**: HEMIT uses a residual CNN + Swin Transformer to jointly generate 3 mIHC markers, but performs worse than pix2pix on most markers in Orion-CRC, indicating poor architectural adaptability to large numbers of markers.
- **vs. VIMs (dubey2024vims)**: VIMs uses text prompts to condition diffusion-based generation of 2 IHC markers, but relies on expert-designed text prompts. The proposed one-hot scheme requires no prompt engineering and scales to an arbitrary number of markers.
- **vs. Marigold (ke2024repurposing)**: This work borrows the weight-copying initialization and fine-tuning strategy from Marigold, while introducing one-hot conditioning and pixel-level color fidelity fine-tuning tailored to the multi-marker setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of one-hot embedding conditioning and the two-stage training framework is novel and practical, achieving multi-marker generation at the scale of 18 markers for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on two public datasets (3-marker and 18-marker), with comprehensive ablation studies (conditioning strategy, fine-tuning, loss weighting, inference cost) and comparison against 5 baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-articulated motivation and analysis of the two-stage design.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to computational pathology, unlocking multi-marker analysis capabilities for existing H&E image repositories.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](../../CVPR2026/medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)
- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)

<!-- RELATED:END -->
