---
title: >-
  [Paper Note] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image
description: >-
  [CVPR2026][3D Vision][3D human reconstruction] DynaAvatar presents the first zero-shot framework for reconstructing animatable 3D human avatars with motion-dependent cloth dynamics from a single image. Through a static-to-dynamic knowledge transfer strategy and a optical flow-guided DynaFlow loss, the method achieves realistic garment dynamics under limited dynamic training data, surpassing all existing approaches across the board.
tags:
  - CVPR2026
  - 3D Vision
  - 3D human reconstruction
  - animatable avatar
  - cloth dynamics
  - 3D Gaussian Splatting
  - single-image reconstruction
date: 2026-05-08
content_hash: 1f4e42698214f8d2
---

# Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image

**Conference**: CVPR2026
**arXiv**: [2603.14772](https://arxiv.org/abs/2603.14772)
**Code**: [https://juhyeon-kwon.github.io/DynaAvatar.github.io/](https://juhyeon-kwon.github.io/DynaAvatar.github.io/) (Project Page)
**Area**: 3D Vision
**Keywords**: 3D human reconstruction, animatable avatar, cloth dynamics, 3D Gaussian Splatting, single-image reconstruction

## TL;DR
DynaAvatar presents the first zero-shot framework for reconstructing animatable 3D human avatars with motion-dependent cloth dynamics from a single image. Through a static-to-dynamic knowledge transfer strategy and a optical flow-guided DynaFlow loss, the method achieves realistic garment dynamics under limited dynamic training data, surpassing all existing approaches across the board.

## Background & Motivation

**State of the Field**: Single-image animatable 3D human avatar reconstruction is a central goal in computer vision and graphics. Existing zero-shot methods (IDOL, LHM) primarily rely on skeleton-based rigid transformations (LBS) to drive animation, which enables body joint motion but is fundamentally incapable of modeling non-rigid cloth dynamics. Another category of personalized methods (ExAvatar, GaussianAvatar) can capture subject-specific garment deformations, but requires per-subject multi-view video capture and optimization, making generalization to arbitrary new subjects infeasible.

**Limitations of Prior Work**:
   - **Rigid animation**: Zero-shot methods produce overly stiff animations where garments such as skirts and jackets fail to naturally flutter during motion, severely degrading visual realism.
   - **Personalization dependency**: Methods capable of modeling cloth dynamics (PERSONA, SeqAvatar) require per-subject data capture and optimization, lacking scalability.
   - **Scarcity of dynamic data**: Large-scale dynamic capture data is prohibitively expensive to collect (multi-view synchronization, temporal calibration, garment diversity), and SMPL-X annotations in existing datasets are commonly missing or noisy.

**Root Cause**: Learning motion-dependent cloth dynamics requires large-scale dynamic capture data, which is extremely scarce. Meanwhile, conventional image reconstruction losses fail to provide effective supervision under large-magnitude garment deformations due to limited receptive fields and color-geometry coupling.

**Paper Goals**:
   - How to achieve motion-dependent cloth dynamics in a zero-shot setting (without per-subject optimization)?
   - How to learn effective dynamic deformation priors under limited dynamic data?
   - How to provide reliable geometric supervision signals for large-scale cloth motion?

**Starting Point**: The authors observe that large-scale static capture data, despite lacking temporal deformation information, contains rich priors on human geometry and appearance. Furthermore, optical flow can establish pixel-level correspondences between rendered and real images across large deformations, providing purely geometric displacement supervision.

**Core Idea**: Combine LoRA fine-tuning of a statically pre-trained Transformer for knowledge transfer with a flow-guided DynaFlow loss for geometry-level deformation supervision, enabling single-image avatars to exhibit realistic motion-dependent cloth dynamics in a zero-shot setting.

## Method

### Overall Architecture

DynaAvatar adopts a Transformer-based feed-forward architecture. Given a single person image and a motion history sequence (1 second / 15 frames), it outputs a 3D Gaussian avatar in canonical space exhibiting motion-dependent cloth dynamics. The overall pipeline consists of five stages:

1. **Image feature extraction**: A frozen pre-trained encoder (Sapiens + DINOv2) extracts image tokens $\mathbf{T}_I$.
2. **Static Transformer**: Extracts detailed geometric and appearance features without considering cloth dynamics.
3. **Motion Encoder**: Encodes motion history into motion tokens $\mathbf{T}_M$.
4. **Dynamic Transformer**: Fuses motion information to superimpose motion-dependent cloth deformations onto static features.
5. **Gaussian Decoder + LBS animation + 3DGS rendering**: Outputs the final animatable avatar.

### Key Designs

1. **Static Transformer**:

    - **Function**: Extracts detailed geometry and appearance features from the input image, without cloth dynamics.
    - **Mechanism**: Employs a Multimodal Transformer Block (MM), treating position encodings of SMPL-X template vertices as 3D point tokens $\mathbf{T}_{3D}$ (queries) and image tokens $\mathbf{T}_I$ (keys/values). Cross-attention updates 3D point features: $\mathbf{T}_{3D}, \mathbf{T}_I \leftarrow \text{MM}(\mathbf{T}_{3D}, \mathbf{T}_I; \mathbf{F}_I)$. The global context feature $\mathbf{F}_I$ is obtained by averaging Sapiens image tokens and used for AdaLN modulation.
    - **Design Motivation**: Reuses weights pre-trained on large-scale static datasets (from LHM) to provide strong priors for subsequent dynamic learning. Freezing the image encoder prevents catastrophic forgetting.

2. **Motion Encoder**:

    - **Function**: Encodes the past 1-second (15-frame) motion history into motion tokens $\mathbf{T}_M$.
    - **Mechanism**: Motion history includes 3D pose (6D rotation parameterization), pose velocity, pose acceleration, and 3D keypoint velocity. Features are positionally encoded and processed through multi-layer MLPs. All motion histories are transformed into a canonical world coordinate system (using dataset-provided up vectors or a default camera $y$-axis) to ensure consistent motion semantics.
    - **Design Motivation**: Cloth dynamics depend not only on the current pose but also on motion direction and velocity. Under the same pose, garment behavior differs entirely between jumping upward and falling downward. Including velocity and acceleration allows the model to distinguish motion direction and intensity.

3. **Dynamic Transformer**:

    - **Function**: Injects motion information into static features, enabling the avatar to exhibit motion-dependent cloth dynamics.
    - **Mechanism**: Motion tokens $\mathbf{T}_M$ serve as keys/values, while $\mathbf{T}_{3D}$ from the Static Transformer output serves as queries. Fusion is performed via an MM block: $\mathbf{T}_{3D}, \mathbf{T}_M \leftarrow \text{MM}(\mathbf{T}_{3D}, \mathbf{T}_M; \mathbf{F}_M)$, where $\mathbf{F}_M$ is the last element of $\mathbf{T}_M$, used as an AdaLN condition.
    - **Design Motivation**: The Dynamic Transformer, trained from scratch, specializes in motion-dependent deformation modeling with a clear division of labor from the pre-trained Static Transformer — the static component preserves geometry/appearance priors while the dynamic component focuses on learning temporal deformations.

4. **Static-to-Dynamic Knowledge Transfer**:

    - **Function**: Leverages pre-training knowledge from large-scale static data to accelerate and improve dynamic deformation learning.
    - **Mechanism**: The Static Transformer is initialized with weights pre-trained on large-scale static datasets (from LHM). During dynamic task training, it is not fully fine-tuned; instead, lightweight LoRA adapters are used for adaptation. The Dynamic Transformer is trained from random initialization.
    - **Design Motivation**: Experiments (Fig. 7) show that training the entire model from scratch fails to preserve input image texture details, while full fine-tuning tends to overwrite the original pre-trained knowledge. Only LoRA fine-tuning can retain rich static knowledge while allowing the Dynamic Transformer to effectively learn motion-dependent dynamics.

5. **Gaussian Decoder and Animation Rendering**:

    - **Function**: Converts Dynamic Transformer outputs into a 3DGS representation for animation and rendering.
    - **Mechanism**: A single linear layer decoder predicts per-Gaussian mean, scale, rotation, opacity, color, and skinning weight offsets. LBS (combined with predicted offsets and diffusion skinning weights) drives the canonical-space avatar to target poses, rendered via a 3DGS renderer.
    - **Design Motivation**: The canonical-space avatar already encodes motion-dependent cloth dynamics; LBS naturally preserves these effects. Skinning weight offsets allow each Gaussian to adjust its animation behavior based on motion history.

### Loss & Training

The **DynaFlow loss** is the most important supervisory design innovation in this work:

- **Problem**: Conventional image reconstruction losses (L1, SSIM) entangle geometry and color, producing ambiguous geometric supervision. Patch-based losses cannot establish cross-region correspondences for large-scale cloth deformations.
- **Solution**: In addition to RGB images, the renderer produces an $xy$ coordinate map $\mathbf{M} \in \mathbb{R}^{H \times W \times 2}$ (rendering each Gaussian's screen-space projected coordinates in place of color). LightGlue computes optical flow matches between rendered and real images, yielding $N$ source-target pixel coordinate pairs $(\mathbf{p}_{src}, \mathbf{p}_{tgt})$, subject to the constraint:

$$\mathcal{L}_{flow} = \frac{1}{N} \sum \|\mathbf{M}(\mathbf{p}_{src}) - \mathbf{p}_{tgt}\|_1$$

- **Key Details**: The match count $N$ is capped at 1024 for stability. DynaFlow is activated only in the latter half of training (early rendering quality is insufficient for reliable flow estimation). Gradients back-propagate through $\mathbf{M}$, directly correcting Gaussian 2D positions.
- **Full Loss**: $\mathcal{L} = \mathcal{L}_{L1} + \mathcal{L}_{SSIM} + \mathcal{L}_{mask} + \mathcal{L}_{LPIPS} + \mathcal{L}_{flow} + \mathcal{L}_{reg}$ (including Laplacian regularization for face and hands).

**Dataset Re-annotation**: Unified SMPL-X re-fitting is performed on three datasets — DNA-Rendering, 4D-Dress, and Actors-HQ — using DWPose for 2D keypoint prediction, SMPLest-X initialization, multi-view L1 loss optimization, and Savitzky-Golay temporal smoothing, yielding 11M+ high-quality image supervision samples.

## Key Experimental Results

### Main Results

| Method | DNA-Rendering PSNR↑ | DNA-Rendering SSIM↑ | DNA-Rendering LPIPS↓ | 4D-Dress PSNR↑ | 4D-Dress SSIM↑ | 4D-Dress LPIPS↓ | Actors-HQ PSNR↑ | Actors-HQ SSIM↑ | Actors-HQ LPIPS↓ |
|------|------|------|------|------|------|------|------|------|------|
| IDOL | 17.84 | 0.902 | 0.155 | 21.31 | 0.948 | 0.077 | 20.93 | 0.910 | 0.138 |
| PERSONA | 14.91 | 0.883 | 0.207 | 19.46 | 0.943 | 0.098 | 19.08 | 0.904 | 0.171 |
| LHM | 17.42 | 0.901 | 0.169 | 21.03 | 0.950 | 0.085 | 20.29 | 0.908 | 0.151 |
| **DynaAvatar** | **19.45** | **0.916** | **0.136** | **23.74** | **0.960** | **0.064** | **21.38** | **0.916** | **0.128** |

DynaAvatar comprehensively outperforms existing single-image methods on all three datasets. On 4D-Dress, PSNR improves by +2.43 dB (vs. IDOL) and LPIPS decreases by 16.9%. Leading performance on the cross-domain Actors-HQ benchmark further demonstrates generalization capability.

### Ablation Study

| Configuration | 4D-Dress PSNR↑ | 4D-Dress SSIM↑ | 4D-Dress LPIPS↓ | Note |
|------|---------|---------|---------|------|
| w/o Dynamic Transformer | 22.57 | 0.952 | 0.068 | No motion history; cloth dynamics lost |
| w/o Knowledge Transfer (train from scratch) | — | — | — | Severe loss of texture detail (Fig. 7b) |
| Full fine-tuning (no LoRA) | — | — | — | Pre-trained knowledge overwritten (Fig. 7c) |
| w/o DynaFlow | — | — | — | Missing large-scale cloth motion; blurry boundaries (Fig. 8) |
| **Full DynaAvatar** | **23.62** | **0.958** | **0.062** | Complete model |

### Key Findings
- **Dynamic Transformer is key to cloth dynamics**: Removing it drops PSNR from 23.62 to 22.57 (−1.05 dB), with garments degenerating to static copies.
- **LoRA is the critical format for knowledge transfer**: Both full fine-tuning and training from scratch lose the texture patterns of the input image; only LoRA strikes the right balance between preserving static priors and learning dynamics.
- **DynaFlow resolves the large-deformation supervision problem**: Without DynaFlow, garments remain nearly static during fast motion with blurry boundaries; DynaFlow's pixel-level displacement supervision directly informs each Gaussian where to move.
- **Same pose, different motion histories yield different cloth behavior** (Fig. 6): Similar poses but different motion histories (falling vs. jumping upward) produce clearly distinct garment deformations, confirming that the Dynamic Transformer genuinely utilizes motion information.

## Highlights & Insights
- **DynaFlow loss design is highly elegant**: By rendering an $xy$ coordinate map instead of RGB and using optical flow to establish correspondences across large deformations, it completely decouples color-geometry entanglement into purely geometric supervision. This idea is not limited to avatars and can transfer to any 3DGS scenario requiring large-displacement supervision (e.g., dynamic scene reconstruction, object deformation modeling).
- **Static-dynamic decoupled architecture**: The combination of a pre-trained Static Transformer (with LoRA-frozen backbone) and a from-scratch Dynamic Transformer achieves an elegant balance between "preserving pre-trained knowledge" and "learning new capabilities." This dual-tower + LoRA paradigm is reusable in other settings where new tasks must be learned from limited data.
- **Data engineering contribution is underappreciated**: Re-annotating SMPL-X parameters across three datasets to obtain 11M+ training samples — optimizing directly from multi-view 2D keypoints rather than triangulated 3D keypoints — constitutes a reusable contribution in its own right.

## Limitations & Future Work
- **Extremely loose garments**: The authors acknowledge that subjects wearing very loose clothing are excluded due to difficulties in 2D keypoint detection, indicating limitations under severe occlusion.
- **Computational overhead**: LightGlue optical flow matching, while efficient, still adds training cost; the motion encoder requires 15 frames of history, necessitating zero-padding for the first frame.
- **Lack of physical consistency**: Purely data-driven cloth deformation lacks physical constraints, potentially producing physically implausible penetrations or floating artifacts.
- **Single-image geometric ambiguity**: Reconstructing occluded regions from a single image remains inherently ambiguous; multi-view input could further improve quality.

## Related Work & Insights
- **vs. LHM**: LHM is also a zero-shot single-image Transformer method but relies solely on rigid LBS animation. DynaAvatar's Static Transformer directly reuses LHM's pre-trained weights, augmenting them with a Dynamic Transformer and DynaFlow to achieve a fundamental leap from "statically copied garments" to "dynamically deforming garments."
- **vs. PERSONA**: PERSONA supports cloth dynamics but requires per-subject optimization (generating richly posed video sequences), making it slow and non-scalable. DynaAvatar performs feed-forward inference, offering far superior speed and scalability.
- **vs. physics-based simulation methods (HOOD/ContourCraft)**: Physics-based methods require clean garment meshes or multi-view calibration and are sensitive to pose accuracy, making them prone to failure. DynaAvatar learns data-driven deformation priors and is more robust to pose noise.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First zero-shot single-image cloth-dynamic avatar; DynaFlow loss is a novel design, though the overall framework represents an incremental extension of LHM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation on three datasets with cross-domain generalization and detailed ablations; qualitative comparisons are intuitive and convincing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated with sound logic; method description is thorough; figures and tables are well designed.
- **Value**: ⭐⭐⭐⭐ Addresses an important limitation of single-image avatars; the data re-annotation pipeline has independent value; provides a clear contribution to future 3D human reconstruction research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars](progressiveavatars_progressive_animatable_3d_gaussian_avatars.md)
- [\[CVPR 2026\] STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction](stavatar_soft_binding_and_temporal_density_control_for_monocular_3d_head_avatars.md)
- [\[ICLR 2026\] CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions](../../ICLR2026/3d_vision/clods_visual-only_unsupervised_cloth_dynamics_learning_in_unknown_conditions.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
