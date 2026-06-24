---
title: >-
  [Paper Note] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][Talking Head Synthesis] TalkingGaussian is proposed, a deformation-driven talking head synthesis framework based on 3D Gaussian Splatting. It represents facial motion by applying smooth deformations to persistent Gaussian primitives, and decomposes the face and inner mouth regions to address motion inconsistency.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Talking Head Synthesis"
  - "3D Gaussian Splatting"
  - "Deformation Field"
  - "Face-Mouth Decomposition"
  - "Audio-driven"
date: 2026-05-08
content_hash: d34d4a7f465a9dc6
---

# TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2404.15264](https://arxiv.org/abs/2404.15264)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Talking Head Synthesis, 3D Gaussian Splatting, Deformation Field, Face-Mouth Decomposition, Audio-driven

## TL;DR

TalkingGaussian is proposed, a deformation-driven talking head synthesis framework based on 3D Gaussian Splatting. It represents facial motion by applying smooth deformations to persistent Gaussian primitives, and decomposes the face and inner mouth regions to address motion inconsistency.

## Background & Motivation

Existing NeRF-based talking head methods (e.g., RAD-NeRF, ER-NeRF) represent facial motion by directly modifying the color and density of points. However, adjacent facial regions may exhibit significantly different color and structural changes. Mutually continuous and smooth neural fields struggle to fit these discontinuous appearance variations, leading to severe distortions such as mouth blurriness and transparent eyelids. While deformation provides a smoother and more continuous representation of motion, previous deformation-based methods lacked fine-grained point-level control.

## Method

### Overall Architecture

TalkingGaussian consists of: (1) Persistent Gaussian Field—a static head structure that maintains a constant appearance and stable geometry; (2) Grid-based Motion Field—a motion field based on tri-plane hash encoding to predict condition-driven point-wise deformation; (3) Face-Mouth Decomposition—decomposing the model into a facial branch and an inner mouth branch.

### Key Designs

**Deformation Paradigm**: Each Gaussian primitive maintains its color $f$ and opacity $\alpha$ unchanged. It changes only its position, scale, and rotation via deformation $\delta_i = \{\Delta\mu_i, \Delta s_i, \Delta q_i\}$. The deformed parameters are represented as $\theta_D = \{\mu+\Delta\mu, s+\Delta s, q+\Delta q, \alpha, f\}$.

**Incremental Sampling Strategy**: Designed to address the vanishing gradient problem in deformation learning. A sliding window is used to progressively sample training frames according to a facial action metric $m_j$: $m_j \in [B_{lower}+k \times T, B_{upper}+k \times T]$, gradually progressing from mouth-closed to mouth-opened, and eyes-opened to eyes-closed.

**Face-Mouth Decomposition**: The lips and inner mouth are spatially close but exhibit inconsistent motion, making them difficult to represent accurately using a single motion field. A semantic mask is used to separate these two regions. The facial branch takes audio $\mathbf{a}$ and upper-face expression $\mathbf{e}$ as conditions, while the mouth branch takes only audio conditions and predicts translations exclusively. Finally, they are blended based on the front-back occlusion relationship: $\mathcal{C}_{head} = \mathcal{C}_{face} \times \mathcal{A}_{face} + \mathcal{C}_{mouth} \times (1-\mathcal{A}_{face})$.

### Loss & Training

- **Static Initialization**: $\mathcal{L}_C = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$
- **Motion Learning**: $\mathcal{L}_D = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$ (rendered using deformed parameters)
- **Fusion Fine-Tuning**: $\mathcal{L}_F = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM} + \gamma\mathcal{L}_{LPIPS}$ ($\lambda=0.2, \gamma=0.5$)

## Key Experimental Results

### Self-Reconstruction Setting

Average results across 4 portrait videos (Macron/Lieu/Obama/May):

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | LMD↓ | Sync-C↑ | Training Time | FPS |
|------|-------|--------|-------|------|---------|---------|-----|
| AD-NeRF | 31.87 | 0.0942 | 0.877 | 2.791 | 5.353 | 18.7h | 0.11 |
| RAD-NeRF | 33.07 | 0.0530 | 0.887 | 2.761 | 5.052 | 5.3h | 28.7 |
| ER-NeRF | 32.83 | 0.0289 | 0.889 | 2.676 | 5.295 | 2.1h | 31.2 |
| ER-NeRF+e | 33.14 | 0.0271 | 0.902 | 2.623 | 5.754 | - | - |
| **Ours** | **33.61** | **0.0259** | **0.910** | **2.586** | **6.516** | **0.5h** | **108** |

### Lip-Sync Setting

Cross-domain audio-driven Sync-C/Sync-E evaluation (Audio A driving Obama):

| Method | Obama Sync-E↓ | Obama Sync-C↑ | May Sync-E↓ | May Sync-C↑ |
|------|---------------|---------------|-------------|-------------|
| AD-NeRF | 9.742 | 5.195 | 9.517 | 4.757 |
| ER-NeRF | - | - | - | - |
| **TalkingGaussian** | Best | Best | Best | Best |

### Key Findings

- Training requires only 0.5 hours vs. 2.1 hours for ER-NeRF, and rendering achieves 108 FPS vs. 31.2 FPS, representing a massive efficiency gain.
- LPIPS decreases from 0.0289 to 0.0259, demonstrating that the deformation paradigm produces clearer and sharper facial details.
- Sync-C increases from 5.295 to 6.516 (GT is 7.584), showing significant improvement in lip-sync quality.

## Highlights & Insights

1. The comparative analysis of **deformation vs. appearance modification** is highly convincing—the deformation space is smoother and more continuous, preventing distortions caused by color jumps.
2. The Face-Mouth decomposition addresses an often-overlooked key challenge: the external lips and the inner mouth can exhibit completely different motions.
3. The incremental sampling strategy successfully resolves the vanishing gradient problem in deformation learning, offering a highly reusable methodology.

## Limitations & Future Work

- Depends on a pre-trained face parsing model to obtain the mouth mask.
- Only supports person-specific training on individual scenarios.
- Limited generalization capability to extreme head pose variations.

## Related Work & Insights

The approach of applying 3DGS to dynamic talking heads was further developed by subsequent works like GaussianTalker. The concept of face-mouth decomposition can potentially be extended to region-based decomposition for full-body animation.

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [\[CVPR 2026\] EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization](../../CVPR2026/3d_vision/emotag_emotion-aware_talking_head_synthesis_on_gaussian_splatting_with_few-shot_.md)
- [\[CVPR 2025\] MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation](../../CVPR2025/3d_vision/monocular_and_generalizable_gaussian_talking_head_animation.md)

</div>

<!-- RELATED:END -->
