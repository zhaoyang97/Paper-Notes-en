---
title: >-
  [Paper Note] MMOne: Representing Multiple Modalities in One Scene
description: >-
  [ICCV 2025][Multimodal VLM][Multimodal scene representation] MMOne is a general framework that addresses property disparity and granularity disparity in multi-modal scene representation through a modality modeling module…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Multimodal scene representation"
  - "3D Gaussian splatting"
  - "modality conflict"
  - "modality decomposition"
  - "thermal imaging"
date: 2026-05-08
content_hash: 281f39640b602d9e
---

# MMOne: Representing Multiple Modalities in One Scene

**Conference**: ICCV 2025
**arXiv**: [2507.11129](https://arxiv.org/abs/2507.11129)
**Code**: [MMOne](https://github.com/Neal2020GitHub/MMOne)
**Area**: Multimodal VLM
**Keywords**: Multimodal scene representation, 3D Gaussian splatting, modality conflict, modality decomposition, thermal imaging

## TL;DR

MMOne is a general framework that addresses property disparity and granularity disparity in multi-modal scene representation through a modality modeling module (with modality indicators) and a multi-modal decomposition mechanism. It jointly models RGB, thermal, and language modalities within a single 3DGS representation, achieving consistent improvements across all modalities.

## Background & Motivation

3D scene representation has achieved great success in RGB rendering, evolving from NeRF to 3D Gaussian Splatting (3DGS). However, integrating multiple modalities (RGB, thermal imaging, language) into a unified scene representation poses fundamental challenges—**Modality Conflicts**:

**Property Disparity**: Different modalities have intrinsically different data characteristics. For example, RGB is a 3-dimensional color vector, whereas language requires a high-dimensional feature space; paper occludes heat sources in RGB/language but not in thermal imaging.

**Granularity Disparity**: Different modalities operate at different information granularities. Thermal imaging is relatively coarse, RGB is finer, and language features remain consistent within object boundaries. Consequently, at object edges, the thermal modality favors fewer large Gaussians, while RGB requires many small ones.

Key limitations of existing methods:
- Use **shared opacity** for all modalities, ignoring inter-modal property disparity
- Use **the same set of Gaussians** for all modalities, contradicting modality-specific granularity requirements
- Modality-specific designs target individual modalities and do not generalize to additional ones

The authors' core question: **How can the intrinsic differences among modalities be resolved when representing multiple modalities simultaneously?**

## Method

### Overall Architecture

MMOne builds upon the 3DGS framework. Given multi-view multi-modal inputs, it progressively constructs a multi-modal scene representation. Each modality is handled by a dedicated modality modeling module, and the densification process integrates a multi-modal decomposition mechanism.

During training, each modality is rendered independently and losses are summed: $\mathcal{L} = \sum_{i=1}^{m} \mathcal{L}_{M_i}$

### Modality Modeling Module (Addressing Property Disparity)

Two components are introduced per modality:

**Modality-specific features** $m_i \in \mathbb{R}^{d_m}$: Different modalities use feature vectors of different dimensions to accommodate their respective physical properties.

**Modality indicator** $\alpha^m \in [0,1]$: Replaces shared opacity by independently controlling opacity for each modality. The rendering equation becomes:

$$M(x) = \sum_{i=1}^{N} T_i^m \cdot \alpha_i^m \cdot g_i^{2D}(x) \cdot m_i$$

$$T_i^m = \prod_{j=1}^{i-1} (1 - \alpha_j^m \cdot g_j^{2D}(x))$$

Key roles of the modality indicator:
- **Weighting mechanism**: Provides different rendering weights for different modalities
- **Switch function**: Selectively deactivates certain modalities during rendering. When a modality is "switched off," the geometric attributes of the Gaussians are influenced only by the remaining active modalities

The switch functionality is implemented in CUDA rasterization by skipping the rendering of specific modalities, thereby freezing their corresponding gradient updates.

### Multi-Modal Decomposition Mechanism (Addressing Granularity Disparity)

**Multi-modal pruning**:
- In vanilla 3DGS, Gaussians with low opacity are directly removed (**Hard Prune**). In multi-modal scenes, directly pruning a Gaussian whose indicator is low in one modality but high in another degrades the other modality.
- The proposed **Soft Prune** only deactivates the specific modality (by setting its modality indicator to "off") rather than deleting the entire Gaussian.
- The pruning threshold for single-modal Gaussians is raised to reduce unimportant single-modal Gaussians and encourage learning of cross-modal shared attributes.

**Multi-modal decomposition**:
During 3DGS densification, gradients from different modalities back-propagated to the same Gaussian may cancel each other, leading to suboptimal results. The solution:

Accumulated gradients $g_{m_i}$ and $g_{m_j}$ per modality are used to compute the gradient discrepancy:

$$gd_{ij} = norm(g_{m_i} - g_{m_j})$$

When the gradient discrepancy exceeds a threshold (0.0002), a multi-modal Gaussian is decomposed into multiple single-modal Gaussians, each optimized independently by its own modality loss.

This disentangles multi-modal information into **shared components** (multi-modal Gaussians) and **modality-specific components** (single-modal Gaussians), yielding a more compact and efficient representation.

### Loss & Training

The total loss is the sum of per-modality losses, with modality-specific formulations:
- **RGB**: Standard 3DGS L1 + SSIM loss
- **Thermal**: L1 + SSIM + smoothness regularization
- **Language**: Semantic feature loss following LangSplat

## Key Experimental Results

### RGB–Thermal Evaluation

| Modality | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|------|------|
| RGB | 3DGS | 23.27 | 0.821 | 0.220 |
| RGB | ThermalGaussian | 24.38 | 0.846 | 0.204 |
| RGB | **MMOne** | **24.89** | **0.854** | 0.209 |
| Thermal | 3DGS | 24.11 | 0.859 | 0.214 |
| Thermal | ThermalGaussian | 25.51 | 0.879 | 0.172 |
| Thermal | **MMOne** | **25.89** | **0.890** | **0.176** |

RGB improves by 0.5 dB and thermal by 0.4 dB, using only **one-third** the number of Gaussians compared to ThermalGaussian.

### RGB–Language Evaluation

| Modality | Method | PSNR↑(R) / mIoU↑(L) | Note |
|------|------|------|------|
| R | LangSplat | 24.02 | Sequential: RGB first, then language |
| R | LS-Joint | 23.23 | Joint training degrades RGB |
| R | **MMOne** | **24.35** | RGB surpasses single-modal LangSplat |
| L | LangSplat | 47.6 | Baseline |
| L | LS-Joint | 55.3 | mIoU +7.7 but at the cost of RGB |
| L | **MMOne** | **56.6** | Best mIoU with no RGB degradation |

**Key finding**: MMOne achieves higher RGB rendering quality than LangSplat trained on RGB alone, demonstrating **mutual enhancement** across modalities.

### RGB–Thermal–Language (Three-Modality) Evaluation

| Method | RGB PSNR | Thermal PSNR | Language mIoU |
|------|------|------|------|
| MM-Joint | 22.32 | 23.38 | 45.1 |
| **MMOne** | **23.19** | **24.24** | **48.1** |

### Modality Conflict Analysis (Key Findings)

| Method | RGB PSNR (2-modal→3-modal) | Thermal PSNR (2-modal→3-modal) |
|------|------|------|
| ThermalGaussian + Language | 22.88 → **22.32** (−0.56) | 23.90 → **23.38** (−0.52) |
| MMOne + Language | 23.12 → **23.19** (+0.07) | 24.17 → **24.24** (+0.07) |

Adding language to the joint-training baseline significantly degrades RGB and thermal performance, whereas adding language to MMOne yields a **slight improvement**, completely resolving modality conflict.

### Ablation Study

| Method | RGB PSNR | Thermal PSNR | Lang mIoU | #Gaussians (×10⁴) |
|------|------|------|------|------|
| MM-Joint | 22.32 | 23.38 | 45.1 | 32.9 |
| + Modality Modeling | 22.38 | 23.73 | 45.3 | 29.0 |
| + Hard Prune | 22.67 | 23.86 | 46.9 | 13.4 |
| + Soft Prune | 22.98 | 23.99 | 47.0 | 10.6 |
| + Decomposition | **23.19** | **24.24** | **48.1** | **9.9** |

Each component contributes consistent improvements. The final model achieves comprehensive superiority using only **30%** of the baseline Gaussian count.

## Highlights & Insights

1. **Identification of fundamental problems**: The first systematic analysis of property disparity and granularity disparity in multi-modal scene representation, accompanied by a unified solution.
2. **Elegant design of the modality indicator**: Serving simultaneously as a weighting coefficient and a switch, a single concise concept addresses both property and granularity disparities.
3. **Mutual enhancement rather than conflict**: Demonstrates that proper disentanglement enables multi-modal learning to be mutually beneficial rather than mutually detrimental.
4. **Compact and efficient**: Superior performance is achieved with one-third the number of Gaussians, indicating genuine information efficiency gains through disentanglement.
5. **General and scalable**: The modality-agnostic framework design readily accommodates additional modalities.

## Limitations & Future Work

1. Validation is limited to RGB, thermal, and language modalities; effectiveness for depth, tactile, and other modalities remains to be verified.
2. The framework relies on RGB camera poses from COLMAP; thermal camera poses require accurate calibration.
3. The gradient discrepancy threshold for multi-modal decomposition (0.0002) is manually set and may require adjustment for different scenes.
4. Dynamic scenes are not addressed.

## Related Work & Insights

- **Single-modal scene representation**: Advances of NeRF/3DGS in RGB; Thermal3D-GS for thermal imaging.
- **Dual-modal representation**: LERF/LangSplat for RGB+language; ThermalGaussian for RGB+thermal.
- **Multi-modal representation**: GLS/LangSurf leverage depth to assist RGB+language but are fundamentally dual-modal.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](../../ICLR2026/multimodal_vlm/multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ICCV 2025\] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models](one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](../../NeurIPS2025/multimodal_vlm/nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)

</div>

<!-- RELATED:END -->
