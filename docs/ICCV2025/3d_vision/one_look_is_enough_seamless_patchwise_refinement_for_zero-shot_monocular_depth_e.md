---
title: >-
  [Paper Note] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images
description: >-
  [3D Vision] Proposes PRO (Patch Refine Once), which achieves seamless patch-wise depth refinement on high-resolution images via Grouped Patch Consistency Training (GPCT) and Bias-Free Masking (BFM) strategies, eliminating boundary artifacts with a single refinement pass per patch—12× faster inference than PatchRefiner.
tags:
  - ICCV 2025
  - 3D Vision
  - High-Resolution Depth Estimation
  - Zero-Shot
  - Patch-Based Refinement
  - Depth Consistency
  - DepthAnything
date: 2026-05-08
content_hash: d63b98213bf65332
---
## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.22351](https://arxiv.org/abs/2503.22351)
- **Code**: [https://kaist-viclab.github.io/One-Look-is-Enough_site](https://kaist-viclab.github.io/One-Look-is-Enough_site)
- **Area**: 3D Vision / Depth Estimation
- **Keywords**: High-Resolution Depth Estimation, Zero-Shot, Patch-Based Refinement, Depth Consistency, DepthAnything

## TL;DR
This paper proposes PRO (Patch Refine Once), which achieves seamless patchwise depth refinement on high-resolution images through Grouped Patch Consistency Training (GPCT) and a Bias-Free Mask (BFM) strategy. PRO eliminates boundary artifacts with only a single refinement pass per patch and achieves a 12× inference speedup over PatchRefiner.

## Background & Motivation

Zero-shot monocular depth estimation (MDE) models such as DepthAnything V2 exhibit strong generalization but face significant challenges on high-resolution images:

**Resolution Mismatch:**
- Models are trained at low resolutions (384×384 or 518×518)
- Full-resolution inference: degraded depth accuracy and excessive GPU memory consumption
- Downsampled inference: loss of fine edge details

**Limitations of Existing Patch-Based Methods:**

**Depth Discontinuity:** Depth values are locally consistent within each patch but exhibit pronounced boundary artifacts (grid artifacts) across patch borders.

**Inference Inefficiency:** PatchRefiner requires 177 patches for test-time ensemble to mitigate discontinuities, inflating inference time to 16 seconds.

**Synthetic Data Bias:** High-resolution dense depth annotations required for training are typically sourced from synthetic datasets (e.g., UnrealStereo4K), where transparent objects such as windows are incorrectly labeled with the depth of the background, introducing training bias.

Core Motivation: **Can a model be trained to eliminate boundary artifacts with only a single refinement pass (One Look) per patch?**

## Method

### Overall Architecture

PRO adopts a residual prediction architecture:
1. A downsampled image is passed through a pretrained MDE model $\Psi$ to obtain a coarse depth map $\mathbf{D}_c$.
2. The original image is divided into patches; each patch is also processed by $\Psi$ to yield a fine depth map $\mathbf{D}_f^i$.
3. A residual prediction network fuses global and local features to predict the refined depth:
$$\mathbf{D}_{refine}^i = \theta(\text{concat}(\mathbf{P}^i, \text{ROI}(\mathbf{D}_c, \mathbf{P}^i), \mathbf{D}_f^i))$$

### Grouped Patch Consistency Training (GPCT)

GPCT is the core strategy for resolving depth discontinuities:

**During Training:** The image is cropped into four overlapping patches (A, B, C, D). Each patch is refined independently, and a consistency loss is applied over the overlapping regions:

$$\mathcal{L}_{con} = \sum_{i \neq j} \frac{1}{|\Omega|} \sum_{p \in \Omega} (\mathbf{D}_{refine}^i(p) - \mathbf{D}_{refine}^j(p))^2$$

**Key Distinction from Prior Work:**
- PatchFusion applies consistency constraints only between two diagonally adjacent patches (e.g., A–D or B–C).
- GPCT processes all four patches simultaneously, **enforcing consistency across all boundaries at every backward pass**, providing a more comprehensive supervision signal.

**During Testing:** The merged depth is computed by simple averaging:
$$\mathbf{D}_{merged}(p) = \frac{1}{N_o(p)} \sum_{i} \mathbf{D}_{refine}^i(p)$$

No additional test-time ensemble steps are required.

### Bias-Free Mask (BFM)

BFM addresses the annotation bias present in synthetic datasets:

**Problem Source:** In UnrealStereo4K, the depth of transparent objects (e.g., windows) is erroneously annotated as the depth of the background behind them.

**Mechanism:** The prior knowledge of the pretrained MDE model is leveraged to identify unreliable regions.

1. Compute the unreliable region mask:
$$\mathbf{M}_{unreliable} = \left[\max\left(\frac{N(\mathbf{D}_c)}{N(\mathbf{D}_{gt})}, \frac{N(\mathbf{D}_{gt})}{N(\mathbf{D}_c)}\right) > \tau\right]$$

where $N(\cdot)$ denotes min-max normalization and $\tau=2$.

2. Extract an edge mask (intersection of edges from the coarse depth and GT depth, preserving critical edge information):
$$\mathbf{M}_{edge} = \mathbf{E}_c \cap \mathbf{E}_{gt}$$

3. Compute the final reliable mask:
$$\mathbf{M}_{BFM} = \mathbf{M}_{edge} \cup \sim\mathbf{M}_{unreliable}$$

The loss is computed only over reliable regions, preventing the model from learning erroneous depth patterns associated with transparent objects.

### Loss & Training

$$\mathcal{L}_{final} = \mathcal{L}_{masked} + \lambda \mathcal{L}_{con}, \quad \lambda = 4$$

$\mathcal{L}_{masked}$ combines L1, L2, and multi-scale gradient losses (4 scales) in a ratio of 1:1:5.

## Key Experimental Results

### Main Results: Zero-Shot High-Resolution Depth Estimation

| Method | Time (s) | Booster AbsRel↓ | ETH3D AbsRel↓ | Middle14 AbsRel↓ | NuScenes AbsRel↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| DepthAnythingV2 | - | 0.0307 | 0.0465 | 0.0307 | 0.106 |
| BoostingDepth | 12.7 | 0.0330 | 0.0552 | 0.0330 | 0.115 |
| PatchFusion P=177 | 36.8 | 0.0496 | 0.0723 | 0.0448 | 0.139 |
| PatchRefiner P=16 | 1.5 | 0.0348 | 0.0435 | 0.0292 | 0.107 |
| PatchRefiner P=177 | 16.2 | 0.0336 | 0.0430 | 0.0292 | 0.106 |
| **PRO (Ours)** | **1.4** | **0.0304** | **0.0422** | **0.0287** | **0.104** |

PRO achieves state-of-the-art performance across all metrics with the fastest inference speed (1.4s vs. 16.2s for PatchRefiner-177, a 12× speedup).

### Consistency Error Comparison

| Method | Consistency Error CE ↓ |
|:---|:---:|
| PatchFusion | 0.364 |
| PatchRefiner | 0.347 |
| **PRO** | **0.049** |

CE is reduced by 85.9%, demonstrating that GPCT substantially eliminates inter-patch discontinuities.

### Ablation Study

| Model | GPCT | BFM | Booster AbsRel↓ | CE ↓ |
|:---|:---:|:---:|:---:|:---:|
| (a) Baseline | | | 0.0385 | 0.208 |
| (b) +BFM | | ✓ | 0.0303 | 0.117 |
| (c) +GPCT | ✓ | | 0.0313 | 0.058 |
| (d) +Both | ✓ | ✓ | **0.0304** | **0.049** |

Key Findings:
- BFM reduces AbsRel by 21.3% on the Booster dataset (which contains transparent objects).
- GPCT reduces CE by 50.4%.
- The two components are complementary and yield the best results when used together.

### Overlap Size Ablation

| Overlap (px) | CE ↓ | ETH3D AbsRel↓ |
|:---:|:---:|:---:|
| 28 | 0.108 | 0.0423 |
| 112 | 0.065 | 0.0425 |
| 224 | **0.049** | **0.0422** |
| 448 | 0.060 | 0.0423 |

An overlap of 224 pixels is optimal; excessively large overlaps weaken the consistency constraint because patch contents become too similar.

## Highlights & Insights
1. **"One Look" Philosophy:** Consistency is enforced during training, eliminating the need for test-time ensemble — a concise and efficient solution.
2. **Generalizable Insight from BFM:** Leveraging pretrained model priors to filter annotation noise in training data is a broadly applicable strategy.
3. **Four-Patch GPCT:** Enforcing constraints across all four patches simultaneously yields substantially stronger consistency than constraining only two diagonal patches, as evidenced by the large CE improvement.
4. **Plug-and-Play:** PRO can be seamlessly integrated with different base depth estimation models.

## Limitations & Future Work
- The Boundary Recall (BR) metric is slightly lower than BoostingDepth and PatchFusion, indicating room for improvement in edge sharpness.
- The BFM threshold $\tau$ requires empirical tuning.
- Depth estimation for transparent objects is avoided rather than fundamentally resolved (handled via masking).

## Related Work & Insights
- Zero-shot depth estimation: MiDaS, DepthAnything, Marigold, GeoWizard
- High-resolution depth estimation: BoostingDepth, PatchFusion, PatchRefiner
- Training data: UnrealStereo4K, ETH3D, Middlebury

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The designs of GPCT and BFM are elegant, addressing test-time challenges through training-time strategies.
- **Technical Depth**: ⭐⭐⭐⭐ — Thorough problem analysis with clear motivation and validation for each component.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Zero-shot evaluation on four datasets with comprehensive ablations.
- **Value**: ⭐⭐⭐⭐⭐ — A 12× speedup with superior quality makes it directly deployable in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)

</div>

<!-- RELATED:END -->
