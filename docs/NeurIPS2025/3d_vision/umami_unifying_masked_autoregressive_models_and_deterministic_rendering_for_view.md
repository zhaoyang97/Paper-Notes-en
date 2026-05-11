---
title: >-
  [Paper Note] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis
description: >-
  [NeurIPS 2025][3D Vision][Novel View Synthesis] This paper proposes UMAMI, a hybrid framework that unifies Masked Autoregressive Models (MAR) and deterministic rendering for sparse-view novel view synthesis. A bidirectio…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Masked Autoregressive"
  - "Diffusion Models"
  - "Deterministic Rendering"
  - "Hybrid Framework"
date: 2026-05-08
content_hash: d7317f52be9d38a5
---

# UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2512.20107](https://arxiv.org/abs/2512.20107)
**Code**: None
**Area**: 3D Vision
**Keywords**: Novel View Synthesis, Masked Autoregressive, Diffusion Models, Deterministic Rendering, Hybrid Framework

## TL;DR
This paper proposes UMAMI, a hybrid framework that unifies Masked Autoregressive Models (MAR) and deterministic rendering for sparse-view novel view synthesis. A bidirectional Transformer encodes multi-view image tokens and Plücker ray embeddings; two lightweight MLP heads handle visible regions (deterministic regression) and occluded regions (MAR diffusion generation) respectively. The rendering speed is an order of magnitude faster than fully generative baselines.

## Background & Motivation

**Background**: Two dominant paradigms exist for novel view synthesis (NVS)—deterministic methods (PixelSplat, MVSplat, LVSM) are fast but produce blurry results in occluded regions; generative methods (CAT3D, Stable Video 3D) can hallucinate occluded content but incur high training and inference costs.

**Limitations of Prior Work**: Fully diffusion-based methods apply large UNet/Transformer architectures to iteratively denoise **entire images**, even when most of the target view is already covered by context views—representing substantial computational waste.

**Key Challenge**: Deterministic methods cannot synthesize unseen regions, while generative methods are highly inefficient for regions that are already visible.

**Key Insight**: Decompose the target image into "deterministically renderable" regions (well-constrained by geometry) and "regions requiring generation" (occluded/unseen areas), and process each with a dedicated head.

**Core Idea**: A bidirectional Transformer produces a shared representation → a feed-forward regression head directly renders visible pixels → a MAR diffusion head generates occluded pixels. The model is trained end-to-end with no hand-crafted 3D inductive biases.

## Method

### Overall Architecture
**Input**: Sparse context images with camera poses (encoded as Plücker ray embeddings). Context image tokens and masked target image tokens are concatenated and fed into a bidirectional Transformer. The resulting latent representation $\mathbf{z}$ is passed to two lightweight MLP heads: (1) a deterministic head that outputs RGB values and a confidence map; (2) a MAR diffusion head that models the token distribution over occluded regions. During training, target image tokens are randomly masked; during inference, tokens are iteratively unmasked—high-confidence regions are filled by the deterministic head, while low-confidence regions are decoded by the diffusion head.

### Key Designs

1. **Data Representation — Plücker Ray Embeddings**:

    - Function: Encodes camera pose information as per-pixel Plücker rays, concatenated with image tokens along the channel dimension.
    - Mechanism: $(\mathbf{d}, \mathbf{d} \times \mathbf{o}) \in \mathbb{R}^6$, where $\mathbf{d}$ is the ray direction and $\mathbf{o}$ is the ray origin.
    - Design Motivation: Eliminates the need for explicit 3D representations (e.g., point clouds or depth maps), as the rays themselves encode sufficient geometric information.

2. **Dual-Head Architecture**:

    - **Deterministic Head $\varphi$**: A lightweight MLP that directly regresses RGB pixel values and per-pixel confidence scores $\mathbf{s}_p$ from the latent $\mathbf{z}$. Trained with MSE and perceptual loss.
    - **MAR Diffusion Head $\phi$**: A lightweight MLP with timestep embedding that performs conditional denoising on the latent $\mathbf{z}$ for each token. Trained with the DDPM objective.
    - Division of Labor: The confidence score $\mathbf{s}_p$ partitions tokens into $\mathbf{x}_D$ (deterministic rendering) and $\mathbf{x}_S$ (diffusion generation).
    - Formulation: $p(\mathbf{x}|\mathbf{c}) = \delta(\mathbf{x}_D - F(\mathbf{c})) \cdot p(\mathbf{x}_S | \mathbf{x}_D, \mathbf{c})$

3. **Hybrid Sampler**:

    - Function: Efficiently merges the outputs of both heads during inference.
    - Mechanism: Iterative unmasking—at each step, the deterministic head fills high-confidence tokens (confidence $> \tau$), while the remaining tokens are decoded by the diffusion head. Filled tokens serve as conditioning in subsequent steps.
    - Design Motivation: Avoids the high cost of full-image diffusion—generation is applied only to occluded regions, while visible regions are rendered in real time.

4. **Confidence Loss**:

    - $\mathcal{L}_{conf} = \mathbf{m} \odot (\mathbf{s}_p \odot \|\hat{\mathbf{I}} - \mathbf{I}\|_2^2 - \lambda_s \cdot \log \mathbf{s}_p)$
    - Encourages accurate predictions in high-confidence regions while penalizing overconfidence.

### Loss & Training
- Total loss = $\mathcal{L}_{render}$ (MSE + perceptual) + $\mathcal{L}_{conf}$ (confidence) + $\mathcal{L}_{diff}$ (DDPM denoising)
- All losses are jointly optimized end-to-end.
- Experiments use RealEstate10K and DL3DV for training.

## Key Experimental Results

### Main Results — RealEstate10K (1→1 View Synthesis)

| Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ | Render Time |
|--------|------|-------|-------|--------|-------------|
| PixelSplat | Deterministic | 25.9 | 0.856 | 0.143 | ~0.1s |
| MVSplat | Deterministic | 26.1 | 0.862 | 0.138 | ~0.1s |
| LVSM | Deterministic | 26.3 | 0.865 | 0.133 | ~0.1s |
| CAT3D | Fully Generative | 25.8 | 0.842 | 0.158 | ~30s |
| SV3D | Fully Generative | 24.5 | 0.821 | 0.192 | ~60s |
| **UMAMI** | **Hybrid** | **26.8** | **0.872** | **0.125** | **~3s** |

### Ablation Study

| Configuration | PSNR | LPIPS | Notes |
|---------------|------|-------|-------|
| Deterministic head only | 26.3 | 0.133 | Blurry occluded regions |
| MAR diffusion head only | 25.5 | 0.145 | Insufficient precision in visible regions |
| **UMAMI (hybrid)** | **26.8** | **0.125** | Best overall |
| Without confidence-based partitioning | 26.1 | 0.135 | No guidance on which regions to generate |

### Multi-View Input Experiment (3→1, 6→1)

| Input Views | Method | PSNR↑ |
|-------------|--------|-------|
| 3 | MVSplat | 28.5 |
| 3 | **UMAMI** | **29.3** |
| 6 | MVSplat | 30.1 |
| 6 | **UMAMI** | **30.7** |

### Key Findings
- UMAMI outperforms both purely deterministic and purely generative methods on PSNR/SSIM/LPIPS, validating the hybrid strategy.
- Rendering speed is **10×** faster than the fully generative baseline (CAT3D), as diffusion is applied only to occluded regions.
- The largest improvements are observed in extrapolation scenarios, where more unseen regions require generation.
- Confidence predictions accurately reflect visibility—high-confidence regions closely align with actually visible areas.

## Highlights & Insights
- **"Not all pixels need to be generated"** is a simple yet profound insight—decomposing NVS into "known rendering" and "unknown generation" subproblems and addressing each with the most suitable approach.
- **Pure data-driven design with no 3D inductive biases**: no assumptions about any 3D representation (NeRF/3DGS/depth maps); the model relies entirely on Plücker rays, large-scale data, and a large Transformer. This scalable design philosophy follows the LVSM/CAT3D line of work.
- The combination of MAR and diffusion loss is elegant: MAR provides an efficient pretraining framework, while the diffusion loss enables high-quality pixel-level conditional generation.
- Confidence prediction enables automatic specialization of the two heads—no manual specification of which regions to generate is required.

## Limitations & Future Work
- Generation remains independent for each target view—simultaneous rendering of multiple target views with 3D-consistent content is not supported.
- Under extreme extrapolation (large viewpoint gaps between input and target), the deterministic head has very low coverage, effectively degrading to near-full generation.
- The current token size of 8×8 patches limits fine-grained detail resolution.
- Integration with 3DGS/NeRF post-processing pipelines has not been explored, which could further improve 3D consistency.

## Related Work & Insights
- **vs LVSM (NeurIPS'24)**: LVSM is a purely deterministic Transformer-based NVS method; UMAMI extends this by adding a generative head to handle occlusions.
- **vs CAT3D (ECCV'24)**: CAT3D applies full diffusion for NVS; UMAMI performs diffusion only on necessary regions, achieving a 10× speedup.
- **vs MAR (ICLR'24)**: MAR was originally designed for image generation; UMAMI adapts it to conditional NVS by leveraging Transformer features as diffusion conditioning.
- The hybrid deterministic-generative paradigm generalizes naturally to tasks with "partially known + partially unknown" structure, such as video editing and virtual try-on.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The hybrid deterministic-generative NVS framework is a first-of-its-kind contribution with a clear and elegantly executed concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on RealEstate10K and DL3DV with multiple input configurations and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete mathematical formalization, clear illustrations, and thorough comparison with related work.
- Value: ⭐⭐⭐⭐⭐ Achieves the best balance between speed and quality, advancing the state of the art in practical NVS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)

</div>

<!-- RELATED:END -->
