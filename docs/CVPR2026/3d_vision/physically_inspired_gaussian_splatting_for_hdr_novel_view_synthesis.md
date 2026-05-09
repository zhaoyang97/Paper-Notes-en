---
title: >-
  [Paper Note] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis
description: >-
  [CVPR 2026][3D Vision][HDR Novel View Synthesis] This paper proposes PhysHDR-GS, a physically inspired HDR novel view synthesis framework that decomposes Gaussian colors into intrinsic reflectance and adjustable ambient illumination. An Image-Exposure (IE) branch and a Gaussian-Illumination (GI) branch complementarily capture HDR details. A cross-branch HDR consistency loss provides explicit HDR supervision without ground-truth HDR data, and illumination-guided gradient scaling addresses gradient starvation caused by exposure bias. The method outperforms HDR-GS by 2.04 dB across multiple benchmarks while maintaining real-time rendering at 76 FPS.
tags:
  - CVPR 2026
  - 3D Vision
  - HDR Novel View Synthesis
  - 3DGS
  - Physically-Inspired Rendering
  - Dual-Branch Architecture
  - Illumination-Guided Gradient Scaling
date: 2026-05-08
content_hash: 6e38bac6edbd3a36
---

# Physically Inspired Gaussian Splatting for HDR Novel View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.28020](https://arxiv.org/abs/2603.28020)
**Code**: [https://huimin-zeng.github.io/PhysHDR-GS/](https://huimin-zeng.github.io/PhysHDR-GS/)
**Area**: 3D Vision / HDR Novel View Synthesis
**Keywords**: HDR Novel View Synthesis, 3DGS, Physically-Inspired Rendering, Dual-Branch Architecture, Illumination-Guided Gradient Scaling

## TL;DR
This paper proposes PhysHDR-GS, a physically inspired HDR novel view synthesis framework that decomposes Gaussian colors into intrinsic reflectance and adjustable ambient illumination. An Image-Exposure (IE) branch and a Gaussian-Illumination (GI) branch complementarily capture HDR details. A cross-branch HDR consistency loss provides explicit HDR supervision without ground-truth HDR data, and illumination-guided gradient scaling addresses gradient starvation caused by exposure bias. The method outperforms HDR-GS by 2.04 dB across multiple benchmarks while maintaining real-time rendering at 76 FPS.

## Background & Motivation

**Background**: HDR novel view synthesis (HDR-NVS) reconstructs high dynamic range scenes by fusing LDR views captured at different exposures. The evolution from NeRF to 3DGS has significantly accelerated HDR-NVS; HDR-GS fits HDR colors via spherical harmonics with an MLP tone mapper, while GaussHDR unifies 3D/2D tone mapping and fuses dual-branch LDR outputs.

**Limitations of Prior Work**: (1) **Appearance entanglement**: Object appearance is jointly determined by material properties and environmental conditions (direct/indirect illumination). Simply scaling sensor exposure time $\Delta t$ cannot disentangle these factors nor capture illumination-dependent appearance changes — exposure variation $\Delta t$ induces global intensity shifts, whereas ambient illumination variation $\Delta L_a$ causes local appearance changes (e.g., specular reflections on a lucky cat nameplate). (2) **Implicit HDR supervision**: HDR ground truth is generally unavailable; supervision of HDR content can only be achieved indirectly by constraining tone-mapped LDR results, but tone mapping compresses dynamic range such that abnormal or saturated HDR values cannot be effectively constrained. (3) **Gradient starvation under exposure bias**: Tone mapping curves have near-zero slopes in extreme (over/under-exposed) regions, causing Gaussian primitives in those regions to accumulate far smaller gradients than those in normally-exposed regions, making it difficult to reach the densification threshold and resulting in under-representation.

**Key Challenge**: Existing HDR-NVS methods follow the conventional HDR imaging pipeline — simulating 2D images at different brightness levels via exposure and tone mapping — without modeling illumination in 3D space, thus ignoring the environment-dependent appearance properties of scenes.

**Goal**: (1) Disentangle the distinct effects of exposure and ambient illumination on appearance; (2) Provide explicit HDR supervision without HDR ground truth; (3) Address gradient starvation and insufficient densification of Gaussians in extreme-exposure regions.

**Key Insight**: Starting from the physically-based rendering equation, Gaussian colors are modeled as a function of intrinsic reflectance $H_r$ and ambient illumination $L_a$, where exposure $t$ and illumination $L_a$ complementarily modulate dynamic range.

**Core Idea**: Decompose 3DGS colors into reflectance and illumination; use an exposure-modulated Image-Exposure (IE) branch and an illumination-modulated Gaussian-Illumination (GI) branch to complementarily capture HDR details; employ a cross-branch consistency loss and illumination-guided gradient scaling to address HDR supervision and gradient starvation.

## Method

### Overall Architecture
PhysHDR-GS decomposes each Gaussian's color into intrinsic reflectance $H_r$ (scene intrinsic property, exposure-invariant) and ambient illumination $L_a$ (adjustable), synthesizing HDR color via an MLP: $\mathbf{c} = g(L_a, H_r)$. The framework comprises two complementary branches: (1) **IE branch**: applies exposure scaling $I_{HDR} \times t$ to the rendered HDR image to simulate standard camera observation; (2) **GI branch**: adjusts the ambient illumination of 3D Gaussians via an illumination modulator to render a relit HDR image $\hat{I}_{HDR}$, capturing illumination-dependent appearance changes. The HDR outputs from both branches are fused by a tone mapper to produce the final LDR result.

### Key Designs

1. **Physically-Inspired Radiance Synthesis (IE + GI Dual Branch)**:

    - **Function**: Complementarily cover a higher dynamic range.
    - **Mechanism**: Based on the simplified rendering equation $L_o(\mathbf{x},\omega_o) = L_e(\mathbf{x}) + L_a(\mathbf{x}) H_r(\mathbf{x},\omega_o)$. **IE branch**: Models $H_r$ and $L_a$ separately, synthesizes Gaussian color $\mathbf{c}=g(L_a,H_r)$ via MLP $g$, and applies exposure $t$ as a global scale after rendering the HDR image — covering different brightness bands and pulling midtone regions into the camera response range. **GI branch**: Introduces an illumination modulator $\hat{L}_a = \varphi(L_a, l)$ that replaces $L_a$ with virtual illumination $\hat{L}_a$ to resynthesize the relit color $\hat{\mathbf{c}}=g(\hat{L}_a, H_r)$ — locally adjusting radiance intensity to avoid saturation.
    - **Design Motivation**: Exposure $t$ provides global dynamic range control, while ambient illumination $L_a$ provides local illumination-dependent variation. Their response patterns are complementary, and combining them covers a higher dynamic range.

2. **Cross-Branch HDR Consistency Loss**:

    - **Function**: Provides explicit supervision for HDR content without HDR ground truth.
    - **Mechanism**: For each view, the illumination level $l$ is set equal to exposure $t$ (making the two branches brightness-comparable). A Gaussian blur is applied to both $I_{HDR} \times t$ from the IE branch and $\hat{I}_{HDR}$ from the GI branch, and an L1 consistency loss is computed: $\mathcal{L}_{\text{cons}} = \|\mathcal{G}(I_{HDR} \times t) - \mathcal{G}(\hat{I}_{HDR})\|_1$. Gaussian blurring avoids penalizing misaligned fine details; the loss constrains both branches to agree on overall illumination and low-frequency structure.
    - **Design Motivation**: LDR supervision alone cannot constrain saturated or abnormal HDR values (compressed by tone mapping). HDR outputs from two different modeling paths (exposure vs. illumination) should be physically consistent — this self-supervised signal compensates for the absence of HDR ground truth.

3. **Illumination-Guided Gradient Scaling**:

    - **Function**: Alleviates gradient starvation of Gaussians in over/under-exposed regions and prevents under-densification.
    - **Mechanism**: The gradient received by a Gaussian is observed to be positively correlated with illumination deviation $\Delta L_a = |L_a - \hat{L}_a|$ (over/under-exposed regions have large illumination deviation but small gradients). A scaling factor $s_a = s \cdot \sigma(|L_a - \hat{L}_a|) + 1$ is proposed ($\sigma$ is sigmoid, $s$ is a hyperparameter), modifying the densification criterion to $\mathbb{I}_i(s_a) \frac{1}{M_i}\sum_k \|\frac{\partial \mathcal{L}_k}{\partial \mu_{i,k}^{\text{ndc}}}\|_2 > \tau_p$. Gaussians with larger illumination deviation receive greater gradient amplification, helping them reach the splitting threshold.
    - **Design Motivation**: Standard 3DGS densification relies on a screen-space gradient threshold, but tone mapping curves have near-zero slopes in extreme regions, causing Gaussians there to accumulate negligible gradients and never be split/cloned — resulting in under-densified blur in over/under-exposed regions. Gradient scaling directly compensates for this systematic bias.

4. **Cross-Fusion Tone Mapper**:

    - **Function**: Fuses the LDR outputs from the IE and GI branches.
    - **Mechanism**: Tone mapper $f$ consists of two lightweight MLPs — $f_{tm}$ applies global and local tone mapping to each HDR input to produce two pairs of LDR predictions; $f_{mix}$ performs cross-fusion of the two pairs: $I_{LDR}^{IG} = f_{mix}(I_{LDR}^{glo}, \hat{I}_{LDR}^{loc})$ and $I_{LDR}^{GI} = f_{mix}(I_{LDR}^{glo}, I_{LDR}^{loc})$; the final LDR output is the sum of both.
    - **Design Motivation**: Global tone mapping preserves overall brightness consistency, local tone mapping retains fine details, and cross-fusion allows the complementary information from both branches to mutually reinforce each other in the LDR domain.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{cons}} + \lambda_3 \mathcal{L}_{\text{unit}}$, where reconstruction loss $\mathcal{L}_{\text{rec}} = \gamma \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{D-SSIM}}$ ($\gamma=0.2$) is computed over three LDR outputs. $\lambda_1=1, \lambda_2=0.5, \lambda_3=0$ (0.5 for synthetic data). During the first 10k iterations, the fusion MLP is frozen and only the tone mapping MLP is trained. Training runs for 30k iterations on a single A6000 GPU.

## Key Experimental Results

### Main Results (HDR-NeRF-Real, exp3 setting)

| Method | LDR-OE PSNR↑ | LDR-NE PSNR↑ | LPIPS↓ |
|--------|-------------|-------------|--------|
| HDR-NeRF | 34.27 | 32.15 | 0.074 |
| HDR-GS | 34.87 | 31.02 | 0.029 |
| GaussHDR | 36.05 | 33.49 | 0.017 |
| GaussHDR† | 36.32 | 33.84 | 0.014 |
| **Ours†** | **36.91** | **34.15** | **0.012** |

Note: Ours† (Scaffold-GS) outperforms GaussHDR† by 0.59 dB on LDR-OE.

### Synthetic Data Results (HDR-NeRF-Syn, exp3 setting)

| Method | LDR-OE PSNR↑ | LDR-NE PSNR↑ | HDR PSNR↑ |
|--------|-------------|-------------|-----------|
| HDR-GS | 40.28 | 27.07 | 17.51 |
| GaussHDR† | 43.87 | 42.74 | 39.08 |
| **Ours†** | **44.26** | **43.19** | **39.21** |

### Ablation Study (HDR-NeRF-Real, exp3)

| Configuration | LDR-OE PSNR | LDR-NE PSNR |
|--------------|-------------|-------------|
| IE branch only | 36.18 | 33.38 |
| + GI branch | 36.27 (+0.09) | 33.46 (+0.08) |
| + HDR-cons | 36.43 (+0.16) | 33.84 (+0.38) |
| + I-GS | **36.91** (+0.48) | **34.15** (+0.31) |

### Efficiency Comparison

| Method | Render (ms) | FPS | Training (min) | VRAM (MB) |
|--------|------------|-----|---------------|----------|
| HDR-NeRF | 4189 | 0.24 | 500 | 11049 |
| HDR-GS | 9 | 117 | 10 | 5014 |
| GaussHDR | 19 | 53 | 28 | 5596 |
| Ours | **13** | **76** | 15 | **3274** |

### Key Findings
- **Illumination-guided gradient scaling (I-GS) contributes the most** — alone delivering a 0.48 dB gain, confirming that gradient starvation in over/under-exposed regions is a critical bottleneck in HDR-NVS.
- **HDR consistency loss yields significant gains** — especially on novel exposure (LDR-NE, +0.38 dB), demonstrating that GT-free HDR self-supervision effectively compensates for the information loss caused by tone mapping.
- **The GI branch has limited standalone contribution (+0.09 dB) but synergizes well with other components** — qualitative analysis shows it primarily improves illumination-dependent appearance (e.g., tabletop reflections) and texture distortion.
- **High efficiency** — Ours is 1.43× faster than GaussHDR (76 FPS vs. 53 FPS), uses only 3274 MB VRAM (vs. GaussHDR's 5596 MB), and trains in 15 minutes.
- Ours† achieves the best LPIPS scores across all benchmarks, indicating that physical modeling benefits perceptual quality.

## Highlights & Insights
- **The dual design of "exposure modulates images, illumination modulates Gaussians"** is the core insight — exposure $t$ is a global 2D-domain scale, while illumination $L_a$ is a local 3D-domain modulation; the two are complementary in covering HDR. This dual-branch design, naturally derived from the physically-based rendering equation, has a stronger theoretical foundation than prior engineering-driven designs.
- **The discovery of gradient starvation and its solution have broad applicability** — any 3DGS optimization involving nonlinear mappings (e.g., gamma correction, tone mapping) may suffer from similar gradient attenuation. The finding that illumination deviation serves as a proxy variable for gradient scaling is transferable to other scenarios.
- **The cross-branch self-supervision idea** — two different paths modeling the same physical quantity (HDR radiance) are forced to be consistent, providing explicit supervision without GT. This design is transferable to other 3D reconstruction tasks lacking ground truth.
- **Real-time efficiency** — 76 FPS and 3274 MB VRAM, 322× faster than HDR-NeRF, and 1.43× faster than GaussHDR with lower memory usage.

## Limitations & Future Work
- Ambient illumination is assumed to follow uniform hemispherical illumination, which is insufficiently accurate for modeling directional strong light sources (e.g., point lights or spotlights).
- The decomposition of reflectance $H_r$ and illumination $L_a$ relies on MLPs and may suffer from inherent ambiguity — the same observation can be explained by multiple $(H_r, L_a)$ pairs.
- The illumination modulator $\varphi$ is data-driven and may have limited generalization to illumination conditions beyond the training exposure range.
- Evaluation is limited to multi-exposure static scenes; performance on dynamic scenes and single-exposure settings remains unknown.
- The GI branch contributes marginally in isolation (+0.09 dB), suggesting that the effectiveness of illumination modulation may be constrained by the diversity of illumination variation in the training data.

## Related Work & Insights
- **vs. HDR-GS**: HDR-GS fits HDR colors via spherical harmonics with an exposure-conditioned MLP tone mapper, without modeling 3D illumination. PhysHDR-GS decomposes color into reflectance and illumination, explicitly modeling illumination in 3D space.
- **vs. GaussHDR**: GaussHDR unifies 3D/2D tone mapping and fuses dual-branch LDR outputs in a purely engineering-oriented design. PhysHDR-GS derives its dual-branch structure from the physically-based rendering equation, with each branch carrying a distinct physical interpretation (exposure = 2D global vs. illumination = 3D local).
- **vs. NeRF-based HDR methods**: Methods such as HDR-NeRF are extremely slow at both training and inference (4189 ms/frame). PhysHDR-GS inherits the efficiency advantages of 3DGS (13 ms/frame).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The IE+GI dual-branch design derived from the physically-based rendering equation is theoretically elegant; the discovery of gradient starvation and the illumination-guided scaling are practically valuable new contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks × two exposure settings × two backbones, complete ablation study, and detailed efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The derivation from the physically-based rendering equation to the method design is clear, with rich and intuitive figures.
- **Value**: ⭐⭐⭐⭐ The gradient starvation finding and illumination-guided scaling have broad applicability to the 3DGS community, though HDR-NVS remains a relatively niche area.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)

</div>

<!-- RELATED:END -->
