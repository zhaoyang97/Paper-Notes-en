---
title: >-
  [Paper Note] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper proposes Hybrid Radiance Fields (HyRF), which combines compact explicit Gaussians (storing only 8 parameters each) with decoupled grid-based neural fields…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Neural Radiance Fields"
  - "Hybrid Representation"
  - "Model Compression"
  - "Real-time Rendering"
date: 2026-05-08
content_hash: 30c9a041c4c58ac7
---

# HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2509.17083](https://arxiv.org/abs/2509.17083)  
**Code**: [Project Page](https://wzpscott.github.io/hyrf/)  
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, Neural Radiance Fields, Hybrid Representation, Model Compression, Real-time Rendering

## TL;DR

This paper proposes Hybrid Radiance Fields (HyRF), which combines compact explicit Gaussians (storing only 8 parameters each) with decoupled grid-based neural fields, achieving 20× model compression while attaining state-of-the-art rendering quality and real-time performance.

## Background & Motivation

**Background**: 3DGS achieves real-time, high-quality novel view synthesis via explicit optimizable 3D Gaussians, but each Gaussian requires 59 parameters (48 for spherical harmonics encoding view-dependent color and 7 for anisotropic shape), resulting in substantial memory overhead.

**Limitations of Prior Work**: NeRF-based methods are compact but slow to render; 3DGS is real-time but model sizes are enormous (hundreds of MB to GB scale); directly encoding Gaussian attributes with neural fields loses high-frequency details due to fixed grid resolution limitations (rapid opacity/scale changes at object boundaries, high-frequency view-dependent effects).

**Key Challenge**: A fundamental conflict between compactness and high-frequency detail preservation — neural fields excel at smooth variations but lose high frequencies, while explicit storage preserves high frequencies at the cost of parameter explosion.

**Goal**: To design a hybrid representation that simultaneously benefits from the compactness of neural fields and the high-frequency detail preservation of explicit Gaussians.

**Key Insight**: Frequency decomposition — low frequencies are handled by neural fields and high frequencies by explicit residual Gaussians; geometry and appearance are modeled separately by two decoupled neural fields.

**Core Idea**: Decompose the scene into compact explicit Gaussians (only position + diffuse color + isotropic scale + opacity = 8 parameters) plus grid-based neural fields that predict the remaining attributes, together with a hybrid rendering pipeline for foreground and background.

## Method

### Overall Architecture

The scene is represented by two components: ① a set of explicit Gaussians, each with only 8 parameters ($\mathbf{p}_e \in \mathbb{R}^3$, $\mathbf{c}_e \in \mathbb{R}^3$, $s_e \in \mathbb{R}$, $\alpha_e \in \mathbb{R}$); and ② dual decoupled neural fields based on multi-resolution hash encoding.

### Key Designs

1. **Decoupled Neural Fields**:

    - Function: Separately predict geometric attributes (scale, opacity, rotation) and appearance attributes (view-dependent color).
    - Design Motivation: Geometry and appearance are weakly correlated; a single network struggles to learn both simultaneously. Experiments show that joint learning significantly degrades quality (PSNR drops by 0.29 dB).
    - Mechanism: Two independent multi-resolution hash encodings $\Theta_{\mathrm{geo}}$ and $\Theta_{\mathrm{rad}}$ encode positions into high-dimensional features, decoded by tiny MLPs (2 layers, 64 neurons):
    $\mathbf{f}_{\mathrm{geo}}^i = \mathrm{enc}(\mathbf{p}_i; \Theta_{\mathrm{geo}}), \quad (\alpha_n, s_n, \mathbf{r}_n) = \mathrm{dec}(\mathbf{f}_{\mathrm{geo}}^i, \Phi_{\mathrm{geo}})$
    $\mathbf{c}_n = \mathrm{dec}(\mathbf{f}_{\mathrm{rad}}^i \oplus \mathbf{f}_{\mathrm{dir}}^i, \Phi_c)$
    - The color branch additionally takes the positional encoding of the view direction $\mathrm{PE}((\mathbf{p}_i - \mathbf{p}_{\mathrm{cam}})/\|\mathbf{p}_i - \mathbf{p}_{\mathrm{cam}}\|_2)$ as input.
    - Novelty: Hash table sizes for geometry and appearance are set at a 2:1 ratio for improved efficiency.

2. **Aggregation with Explicit Gaussians**:

    - Function: Aggregate raw attributes predicted by the neural fields with explicitly stored residuals.
    - Design Motivation: Grid-based neural fields are prone to losing high-frequency signals due to fixed resolution and hash collisions; explicit residuals compensate for critical details.
    - Mechanism: Additive aggregation followed by activation:
    $\alpha = \sigma(\alpha_n + \alpha_e), \quad \mathbf{c} = \sigma(\mathbf{c}_n + \mathbf{c}_e), \quad s = \sigma(s_n + s_e)$
    - Rotation is predicted solely by the neural field: $\mathbf{r} = \mathrm{Normalize}(\mathbf{r}_n)$
    - Novelty: Compared to the similar approach in LocoGS, this work stores complete explicit shape residuals and introduces decoupled neural fields.

3. **Visibility Pre-culling**:

    - Function: Cull Gaussians outside the view frustum before querying the neural fields.
    - Design Motivation: Hash encoding queries and MLP decoding constitute the primary computational bottleneck; culling invisible points substantially accelerates rendering.
    - Mechanism: Points are transformed to camera space; only those projected within the image frame (with a tolerance margin $\mathrm{tol}$) are retained, while points too close to the image plane are discarded.
    - Effect: Achieves 3.9× rendering speedup with no quality loss.

4. **Background Rendering**:

    - Function: Predict background color using a neural field and alpha-composite it with the foreground Gaussian rendering result.
    - Design Motivation: 3DGS struggles to effectively densify and optimize distant objects, leading to blurry backgrounds.
    - Mechanism: A background sphere of radius $r=100$ is constructed; for each ray, its intersection with the sphere $\mathbf{p}_s$ is computed, and the radiance field predicts the background color:
    $C_{\mathrm{bg}} = \prod_{i=1}^{\mathcal{N}}(1-\alpha_i) \cdot \mathbf{c}_s, \quad C = C_{\mathrm{fg}} + C_{\mathrm{bg}}$
    - Background color is computed only for pixels with accumulated transmittance $T < \tau_T = 0.2$, further accelerating rendering.

### Loss & Training

- The same L1 + SSIM loss as the original 3DGS is used: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\mathrm{ssim}}$
- Explicit opacity is periodically reset and low-opacity Gaussians are pruned.
- Scene contraction (following MipNeRF360) constrains coordinates to $(0,1)$.
- Hash encoding uses 16 levels with feature dimension 2; hash table sizes are $2^{18}$ for standard scenes and $2^{21}$ for large-scale scenes.

## Key Experimental Results

### Main Results

**MipNeRF360 + Tanks&Temples + Deep Blending (13 scenes)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Size(MB)↓ |
|--------|-------|-------|--------|------|-----------|
| Mip-NeRF360 | 27.69 | 0.792 | 0.237 | 0.06 | 8.6 |
| 3DGS | 27.21 | 0.815 | 0.214 | 117 | 734 |
| Scaffold-GS | 27.39 | 0.806 | 0.252 | 86 | 244 |
| **HyRF (Ours)** | **27.78** | **0.816** | **0.211** | **102** | **49** |

HyRF surpasses 3DGS in quality, reduces model size by **15×**, and maintains real-time performance at 102 FPS.

**Large-scale Urban Scenes (Mill19 + Urbanscene3D)**

| Method | PSNR↑ | Size(MB)↓ | FPS↑ |
|--------|-------|-----------|------|
| 3DGS | 22.41 / 21.41 | 1566 / 935 | 81 / 84 |
| Scaffold-GS | 22.33 / 20.25 | 560 / 435 | 36 / 34 |
| **HyRF** | **23.52 / 24.68** | **215 / 202** | **75 / 77** |

Advantages are more pronounced on large-scale scenes: significantly superior quality with 4–7× smaller models.

**Compression Comparison (MipNeRF360)**

| Method | PSNR↑ | Size(MB)↓ |
|--------|-------|-----------|
| Papantonakis et al. | 27.1 | 25.40 |
| Chen et al. | 27.59 | 22.50 |
| **HyRF + Compression** | **27.66** | **18.04** |

HyRF with compression outperforms all state-of-the-art compression methods.

### Ablation Study

**Tanks & Temples dataset**

| Configuration | PSNR↑ | Size(MB)↓ | FPS↑ |
|---------------|-------|-----------|------|
| Full model | 24.07 | 41 | 106 |
| w/o Decoupling | 23.78 | 37 | 101 |
| w/o Explicit Gaussians | 23.45 | 27 | 121 |
| w/o Neural Fields | 22.22 | 14 | 127 |
| w/o Background Rendering | 23.43 | 41 | 112 |
| w/o Pre-culling | 24.06 | 41 | 27 |

- Removing decoupling causes a 0.29 dB PSNR drop, validating the necessity of geometry/appearance separation.
- Removing explicit Gaussians causes a 0.62 dB PSNR drop, demonstrating the importance of high-frequency residual compensation.
- Removing pre-culling causes FPS to plummet from 106 to 27.

### Key Findings

- Explicit color residuals most significantly affect modeling of illumination variation; explicit scale residuals most affect reconstruction of thin structures and training stability.
- Background rendering yields substantial improvements for distant scene elements (clouds, sky).
- Training convergence is noticeably faster than baseline methods.

## Highlights & Insights

- **Elegant frequency decomposition**: Low frequencies handled by neural fields and high frequencies by explicit residuals — the principle is clear and the results are significant.
- **Decoupled design is both theoretically motivated and empirically validated**: The weak correlation between geometry and appearance makes it difficult for a single network to handle both effectively.
- **Plug-and-play compression compatibility**: With far fewer explicit Gaussian parameters (8 vs. 59), traditional compression methods such as VQ and Huffman coding can be directly applied.
- **Comprehensive experimental coverage**: 25 scenes, 5 dataset categories, and comparisons against 6 compression methods.
- Pre-culling achieves a 3.9× speedup with virtually no quality loss, offering strong practical utility.

## Limitations & Future Work

- The inherent aliasing problem of 3DGS is not addressed.
- The neural field component requires high-end GPUs to achieve real-time performance; efficiency on web browsers and integrated graphics remains to be improved.
- Surface reconstruction accuracy is sometimes insufficient.
- Future work could integrate anti-aliasing techniques (e.g., Mip-Splatting) and more efficient hash encodings.

## Related Work & Insights

- **Scaffold-GS**: Uses anchors and neural features to predict local Gaussian attributes, but operates locally rather than globally.
- **LocoGS**: A similar approach that stores Gaussian attributes in neural fields, but lacks explicit residuals and decoupled design.
- **Instant-NGP**: The origin of multi-resolution hash encoding, which this work extends to Gaussian attribute prediction.
- **Insight**: Hybrid explicit-implicit representations may represent the optimal trade-off for novel view synthesis; frequency decomposition is a general and transferable design paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ Hybrid representation with decoupled design — the idea is clear, though the general direction of compressed 3DGS is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 25 scenes across 5 datasets, detailed ablations, compression comparisons, and large-scale scene evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear paper structure, rich figures and tables, rigorous ablation logic.
- Value: ⭐⭐⭐⭐ Highly practical; achieving the triple benchmark of 20× compression, quality improvement, and real-time performance carries significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](../../CVPR2026/3d_vision/pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)

</div>

<!-- RELATED:END -->
