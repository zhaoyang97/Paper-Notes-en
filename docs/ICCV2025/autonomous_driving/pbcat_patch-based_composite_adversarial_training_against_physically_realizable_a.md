---
title: >-
  [Paper Note] PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection
description: >-
  [ICCV 2025][Autonomous Driving][Adversarial Training] This paper proposes PBCAT (Patch-Based Composite Adversarial Training), which combines small-area gradient-guided adversarial patches with global imperceptible perturbations for adversarial training, providing unified defense against multiple physically realizable attacks (adversarial patches and adversarial textures). PBCAT achieves a 29.7% AP improvement over the previous SOTA defense on pedestrian detection tasks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Adversarial Training
  - Physically Realizable Attacks
  - Object Detection
  - Adversarial Texture
  - Adversarial Patch
date: 2026-05-08
content_hash: 34dc6f0a466e57b2
---

# PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection

**Conference**: ICCV 2025
**arXiv**: [2506.23581](https://arxiv.org/abs/2506.23581)
**Code**: [GitHub](https://github.com/LixiaoTHU/oddefense-PatchAT)
**Area**: Autonomous Driving
**Keywords**: Adversarial Training, Physically Realizable Attacks, Object Detection, Adversarial Texture, Adversarial Patch

## TL;DR

This paper proposes PBCAT (Patch-Based Composite Adversarial Training), which combines small-area gradient-guided adversarial patches with global imperceptible perturbations for adversarial training, providing unified defense against multiple physically realizable attacks (adversarial patches and adversarial textures). PBCAT achieves a 29.7% AP improvement over the previous SOTA defense on pedestrian detection tasks.

## Background & Motivation

Object detection plays a central role in safety-critical applications such as autonomous driving and video surveillance, yet recent studies have shown that detectors are highly vulnerable to physically realizable attacks. These attacks fall into two main categories:

**Adversarial Patches**: Adversarial patterns placed in a fixed local region — small in area but with large perturbation magnitude.

**Adversarial Textures**: Methods such as AdvTexture and AdvCaT that apply adversarial perturbations over clothing surfaces — large in area, effective from multiple viewpoints, and capable of deceiving detectors when worn in the physical world.

Existing defense methods suffer from three key limitations:

- **Narrow coverage**: Most defenses target only patch attacks and are nearly ineffective against texture attacks (non-AT methods such as SAC yield AP near 0 under AdvTexture).
- **Lack of adaptive robustness**: Input preprocessing and feature filtering methods can be bypassed by adaptive attacks.
- **Mismatch of $\ell_\infty$-AT with physical threats**: The threat model of globally imperceptible perturbations fundamentally differs from physical attacks; although some transferability exists, it remains insufficient.

Extending patch-based AT from classification to detection introduces additional challenges: detection scenarios require handling multiple patches across multiple objects, and computational cost grows sharply with the number of candidate patch locations.

## Method

### Overall Architecture

The core idea of PBCAT is to combine two types of adversarial perturbations during training: small-area, high-intensity gradient-guided adversarial patches and globally constrained low-intensity $\ell_\infty$ perturbations. The final adversarial example is:

$$\delta = \text{Apply}(\delta_p \odot \mathbf{M}, \mathbf{x}) + \delta_g$$

where $\delta_p$ is the patch perturbation ($\|\delta_p\|_\infty \leq \beta$), $\delta_g$ is the global perturbation ($\|\delta_g\|_\infty \leq \epsilon$), and $\mathbf{M}$ is a binary mask selected via gradient guidance.

### Key Designs

1. **Extension from $\ell_\infty$-AT to Patch-based AT**: The attack budget is reformulated from a global constraint to a local mask constraint. For detection, each bounding box may contain one adversarial patch. The key formulation is:

$$\theta = \arg\min_\theta \mathbb{E}_\mathbf{x}\left\{\max_{\|\delta_p \odot \mathbf{M}\|_\infty \leq \beta} \mathcal{L}_d(f_\theta(\mathbf{x} + \delta_p \odot \mathbf{M}), y)\right\}$$

Patches are randomly placed within the bounding box (sampled from a Gaussian distribution centered at the bbox center) to prevent the model from exploiting patch location for regression (information leakage). The patch side length is $s = \lambda \cdot \sqrt{w_{bbox}^2 + h_{bbox}^2}$. Each bbox has a 50% probability of receiving a patch to preserve detection capability on clean objects.

2. **Gradient-Guided Patch Segmentation and Selection**: The sampled patch region is divided into $n \times n$ sub-patches (default $N=64$, i.e., $8\times8$). After a single forward and backward pass, the average gradient norm of each sub-patch is computed, and the top 50% regions with the largest gradients are selected to construct the binary mask $\mathbf{M}$. Key advantages of this design:

    - Requires only a single forward/backward pass, adding negligible computational overhead.
    - High-gradient regions correspond to the model's vulnerable areas, where perturbations are most effective.
    - Produces irregularly shaped perturbation regions that better approximate the diversity of real physical attacks (as opposed to simple square patches).

3. **Combination of Local Patch and Global Noise**: This is the most critical innovation in PBCAT. Training with only small patches fails to defend against large-area texture attacks, yet directly enlarging the patch area causes training collapse (severe destruction of object information). The solution is:

    - Introducing $\ell_\infty$-constrained global perturbations ($\epsilon = 4/255$) at low intensity that do not destroy object information.
    - Global noise covers the entire image, compensating for regions not covered by the small patch.
    - The two perturbations are complementary: the patch provides locally strong adversarial signals, while the global noise provides spatial coverage.

### Loss & Training

- Acceleration via FreeAT: gradients are reused across steps to avoid the high cost of a full PGD inner loop.
- An adversarially pretrained backbone from AdvOD is used.
- Patch perturbation step size $\alpha = 8/255$, magnitude $\beta = 64/255$.
- FreeAT replay parameter $r = 8$.
- A general-purpose detector is trained on MS-COCO and directly transferred to safety-critical pedestrian detection tasks.
- Gradients for patch and global perturbations are computed jointly within the same backward pass.

## Key Experimental Results

### Main Results

**Pedestrian Detection AP50 (Adaptive White-Box Attacks)**

| Method | Clean (Inria) | AdvPatch | Clean (Synth) | AdvTexture | AdvCaT |
|--------|--------------|----------|--------------|-----------|--------|
| Vanilla | 96.2 | 37.3 | 86.4 | 0.2 | 0.3 |
| SAC | 96.2 | 57.1 | 85.4 | 0.3 | 0.6 |
| Jedi | 92.3 | 64.4 | 88.1 | 2.3 | 0.7 |
| $\ell_\infty$-AT (AdvOD) | 95.9 | 56.1 | 92.5 | 30.5 | 39.6 |
| **PBCAT** | 95.4 | **77.6** | 92.5 | **60.2** | **56.4** |

### Ablation Study

**Contribution of Each Component (Pedestrian Detection AP50)**

| Patch | Global | Gradient | AdvPatch | AdvTexture | AdvCaT |
|-------|--------|----------|----------|-----------|--------|
| ✓ | | | 35.4 | 1.6 | 0.8 |
| ✓ | ✓ | | 72.8 | 24.9 | 19.5 |
| ✓ | ✓ | ✓ | **77.6** | **63.3** | **56.4** |

**Effect of Sub-Patch Count**

| Sub-patches | AdvPatch | AdvTexture | AdvCaT | Notes |
|-------------|----------|-----------|--------|-------|
| 16 | **78.3** | 50.8 | 46.2 | Too coarse; weak texture defense |
| 64 | 77.6 | **60.2** | 56.4 | Best balance |
| Pixel-level | 67.4 | 20.4 | **59.4** | Too fine; patch defense degrades |

### Key Findings

- Global noise is critical for defending against texture attacks — training with patches alone yields only 1.6% AP on AdvTexture; adding global noise raises it to 24.9%.
- Gradient-guided selection further improves robustness across all attacks by approximately 10–20% over random selection.
- All non-AT defense methods (LGS, SAC, Jedi, etc.) are nearly ineffective against texture attacks under adaptive attacks (AP < 6%).
- PBCAT generalizes to FCOS and DN-DETR, demonstrating detector-agnostic applicability.
- In physical-world video evaluation, PBCAT successfully detects pedestrians wearing adversarially textured clothing.

## Highlights & Insights

- **Unified defense framework**: The first single AT method to effectively defend against both patch and texture physical attacks simultaneously, filling an important gap.
- **Elegant composite perturbation design**: The combination of locally strong and globally weak perturbations avoids training collapse while providing sufficient spatial coverage.
- **Efficient gradient guidance**: Requires only a single forward/backward pass, in contrast to prior methods that perform multiple forward inferences to search for optimal patch locations.
- **Practical security value**: Directly addresses real-world threats of adversarial clothing camouflage, with significant implications for video surveillance and autonomous driving safety.

## Limitations & Future Work

- As with most AT methods, PBCAT incurs a slight clean accuracy drop.
- Whether an inherent trade-off exists between robustness to physically realizable attacks and clean accuracy remains an open question.
- The current patch shape is still a regular square region, whereas physical attacks may exhibit more irregular shapes.
- Future work could explore incorporating style transfer or GAN-generated perturbations to increase training diversity.

## Related Work & Insights

- Bridges the gap between $\ell_\infty$-AT (not targeting physical attacks) and patch-based AT (limited to patches only).
- The FreeAT acceleration strategy brings adversarial training costs close to standard training, advancing the practicality of AT for detection tasks.
- The gradient-guided segmentation idea is generalizable to other adversarial training scenarios requiring spatial selection.
- Provides direct reference value for adversarial robustness research in autonomous driving security.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The composite perturbation strategy and gradient-guided selection are elegantly designed, unifying patch and texture defense.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers diverse attacks, detectors, and datasets; ablations are highly detailed; physical-world validation is included.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, method description is progressive, and experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Directly improves robustness of safety-critical detection systems with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](../../AAAI2026/autonomous_driving/moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)

</div>

<!-- RELATED:END -->
