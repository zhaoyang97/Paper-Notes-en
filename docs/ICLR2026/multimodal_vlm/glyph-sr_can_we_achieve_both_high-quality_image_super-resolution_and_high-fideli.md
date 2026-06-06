---
title: >-
  [Paper Note] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?
description: >-
  [ICLR 2026][Multimodal VLM][Image Super-Resolution] This paper proposes GLYPH-SR, a vision-language-guided diffusion framework that simultaneously optimizes image quality and text readability via a dual-branch Text-SR fu…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Image Super-Resolution"
  - "Scene Text Recovery"
  - "ControlNet"
  - "Diffusion Models"
  - "OCR"
date: 2026-05-08
content_hash: 0b33b286dba03583
---

# GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?

**Conference**: ICLR 2026
**arXiv**: [2510.26339](https://arxiv.org/abs/2510.26339)  
**Code**: Available (noted for release in the paper)  
**Area**: Diffusion Models
**Keywords**: Image Super-Resolution, Scene Text Recovery, ControlNet, Diffusion Models, OCR

## TL;DR
This paper proposes GLYPH-SR, a vision-language-guided diffusion framework that simultaneously optimizes image quality and text readability via a dual-branch Text-SR fusion ControlNet and a ping-pong scheduler, achieving a 15.18-point improvement in OCR F1 on SVT ×8.

## Background & Motivation
Image super-resolution (SR) serves as a foundational technique for many vision systems; however, existing SR methods suffer from two systematic biases: (1) **Metric bias** — global metrics such as PSNR/SSIM assign negligible weight to small text regions (typically less than 1% of the image), so character corruption incurs almost no penalty; (2) **Objective bias** — commonly used training losses treat text as ordinary high-frequency texture rather than the discrete semantic units required by OCR. These biases give rise to two failure modes: hallucination (generating sharp but incorrect characters) and conservative recovery (retaining blur without improvement). The core problem is achieving visual realism and text readability simultaneously — two objectives that exhibit significant tension.

## Method

### Overall Architecture
GLYPH-SR builds upon a pretrained LDM (Juggernaut-XL) and augments it with a Text-SR fusion ControlNet (TS-ControlNet). OCR is used to extract text–position pairs that provide word-level semantic guidance, while a ping-pong scheduler alternates between text-centric and image-centric guidance throughout the denoising process.

### Key Designs
1. **Condition Decomposition**

    - **Function**: Explicitly separates guidance signals into image-oriented and text-oriented components.
    - **Mechanism**: A scene-level caption $\mathcal{S}_{\text{IMG}}$ summarizes global attributes (lighting, composition, etc.); an OCR module detects $K$ text instances and returns position–text pairs $\{(\mathcal{S}_{\text{text}}^k, \mathcal{S}_{\text{pos}}^k)\}_{k=1}^K$, which are converted into structured natural-language prompts (e.g., "HSBC appears at the center of the image").
    - **Design Motivation**: When guidance is provided only in aggregated form, small text regions are still treated as generic high-frequency texture.

2. **Text-SR Fusion ControlNet (TS-ControlNet)**

    - **Function**: Balances image quality and text readability while preserving the generative prior.
    - **Mechanism**: A dual-branch architecture — the SR branch is frozen to maintain overall image quality, while the text branch is trainable and focuses on glyph recovery. Residual mixed injection is formulated as: $c = \frac{1}{2} s_{\text{ctrl}} [\mathcal{C}_{\text{SR}}(z_t; \phi_{\text{img}}(\mathcal{S}_{\text{IMG}}+P)) + \mathcal{C}_{\text{TXT}}(z_t; \phi_{\text{txt}}(\mathcal{S}_{\text{TXT}}+P))]$
    - **Design Motivation**: Directly separating the two guidance signals improves text but degrades non-text regions.

3. **Ping-Pong Scheduler**

    - **Function**: Dynamically reweights text and image guidance along the denoising trajectory.
    - **Mechanism**: A time-dependent coefficient $\lambda_t$ modulates both embedding fusion and residual injection. A binary square-wave strategy alternates between $\lambda_t=0$ (text-centric) and $\lambda_t=1$ (image-centric) with switching period $\tau=1$: $\lambda_t = 0$ if $\lfloor \frac{t-t_0}{\tau} \rfloor \bmod 2 = 0$, otherwise $\lambda_t = 1$.
    - **Design Motivation**: Continuous gradual transitions are less effective than a square wave; text-centric steps inject precise glyph cues, while image-centric steps stabilize global structure.

### Loss & Training
- Standard $\varepsilon$-prediction objective: $\mathcal{L}_{\text{text}} = \mathbb{E}_{z_0, t, \varepsilon} \| \varepsilon - \mathcal{D}_\theta(z_t, t, c) \|_2^2$
- A four-partition synthetic corpus is constructed by independently perturbing glyph quality and global image quality, enabling targeted text recovery learning.
- The LDM backbone and SR branch are frozen; only the text branch is fine-tuned.

## Key Experimental Results

### Main Results (SVT ×4 OCR F1)

| Method | OpenOCR | GOT-OCR | LLaVA-NeXT | MANIQA | CLIP-IQA |
|--------|---------|---------|------------|--------|----------|
| DiffBIR | 38.73 | 42.33 | 45.19 | 47.82 | 58.66 |
| InvSR | 57.79 | 60.96 | 65.00 | 46.78 | 57.30 |
| PiSA-SR | 63.30 | 65.23 | 67.75 | 37.41 | 44.30 |
| **GLYPH-SR** | **67.54** | **71.72** | **73.22** | **47.75** | **59.40** |

### Ablation Study (Contribution of Core Components)

| Configuration | OCR F1 | MANIQA | Notes |
|---------------|--------|--------|-------|
| Condition decomposition only | Improved | Degraded | Non-text regions deteriorate |
| + TS-ControlNet | Further improved | Maintained | Dual-branch balancing |
| + Ping-Pong | Best | Competitive | Square wave outperforms continuous gradients |

### Key Findings
- OCR F1 on SVT ×8 improves by up to 15.18 points over diffusion/GAN baselines.
- Validated across three datasets (SVT / SCUT-CTW1500 / CUTE80) at two scales (4× / 8×).
- OCR metrics improve substantially while MANIQA / CLIP-IQA / MUSIQ scores remain competitive.

## Highlights & Insights
- Scene text SR is explicitly formulated as a dual-objective optimization problem, establishing for the first time a standardized dual-axis evaluation protocol.
- The four-partition synthetic data design is elegant: orthogonal perturbation of glyph and image quality enables decoupled learning.
- The ping-pong scheduler is simple yet effective, outperforming more complex continuous noise-level scheduling strategies.

## Limitations & Future Work
- The pipeline depends on an OCR module to extract text locations, which may itself fail at low resolutions.
- Synthetic training data may not fully represent real-world degradation distributions.
- Evaluation is limited to 4× and 8× upscaling; performance at higher magnification factors remains unknown.

## Related Work & Insights
- **vs. StableSR / DiffBIR**: These methods optimize perceptual quality but are insensitive to character integrity.
- **vs. TATT and other text SR methods**: Text-specific SR methods perform poorly in full-scene settings due to oversimplified scene assumptions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual-objective SR framework and ping-pong scheduler constitute novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons across three datasets and two scales.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and motivation is well-justified.
- **Value**: ⭐⭐⭐⭐ Practical applicability to scene text SR tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](../../CVPR2026/multimodal_vlm/hificl_highfidelity_incontext_learning_for_multimo.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)

</div>

<!-- RELATED:END -->
