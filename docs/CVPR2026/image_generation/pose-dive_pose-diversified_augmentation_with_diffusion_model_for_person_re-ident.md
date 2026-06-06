---
title: >-
  [Paper Note] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification
description: >-
  [CVPR 2026][Image Generation][Person Re-Identification] Pose-dIVE leverages the SMPL model to jointly control human body pose and camera viewpoint…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Person Re-Identification"
  - "Data Augmentation"
  - "Diffusion Model"
  - "SMPL"
  - "Pose Diversification"
date: 2026-05-08
content_hash: 4fd13b3a51dc707c
---

# Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification

**Conference**: CVPR 2026
**arXiv**: [2406.16042](https://arxiv.org/abs/2406.16042)  
**Code**: [https://cvlab-kaist.github.io/Pose-dIVE](https://cvlab-kaist.github.io/Pose-dIVE)  
**Area**: Diffusion Models / Image Generation
**Keywords**: Person Re-Identification, Data Augmentation, Diffusion Model, SMPL, Pose Diversification

## TL;DR
Pose-dIVE leverages the SMPL model to jointly control human body pose and camera viewpoint, using a diffusion model to generate person images with diversified poses and viewpoints. This approach systematically alleviates distributional bias in Re-ID training data, consistently improving the generalization capability of arbitrary Re-ID models across multiple benchmarks.

## Background & Motivation

**Background**: Person Re-Identification (Re-ID) has achieved remarkable progress in tracking and recognizing individuals across multi-camera networks. Methods such as CLIP-ReID and SOLIDER demonstrate strong performance on standard benchmarks, yet a significant gap persists between training conditions and real-world deployment scenarios.

**Limitations of Prior Work**: Existing Re-ID datasets suffer from severe lack of diversity in body pose and camera viewpoint — typically containing only limited walking/standing poses and 2–3 fixed viewpoints. This unimodal distribution makes it difficult for models to learn identity features invariant to pose and viewpoint.

**Key Challenge**: Collecting richer and more diverse datasets is constrained by privacy concerns and the high cost of deploying large-scale multi-viewpoint camera systems. Existing data augmentation methods (GAN-based or simple geometric transformations) either exploit only the pose/viewpoint variations already present within the dataset, or treat pose and viewpoint as independent factors to be handled separately.

**Goal**: (1) How to systematically inject sparse and underrepresented pose and viewpoint samples into training data? (2) How to handle pose and viewpoint variations jointly rather than independently? (3) How to ensure that generated augmentation data preserves identity consistency?

**Key Insight**: The SMPL 3D human body model is used to simultaneously encode pose and viewpoint information (implicitly encoding camera viewpoint via depth maps and normal maps), with pretrained Stable Diffusion's powerful priors leveraged to generate high-fidelity augmented images.

**Core Idea**: Uniformly sample poses and viewpoints from external data sources, render them into 2D conditioning signals via SMPL, and guide the diffusion model to generate identity-consistent, diversified training samples.

## Method

### Overall Architecture
Pose-dIVE consists of three stages: (1) training the generative model (two-phase: fashion video pretraining followed by Re-ID data fine-tuning); (2) augmenting training data using the generative model (sampling pose/viewpoint from external sources → rendering SMPL → generating new images); (3) training an arbitrary Re-ID model on the augmented dataset.

### Key Designs

1. **SMPL-Driven Joint Pose–Viewpoint Conditioning**:

    - Function: Simultaneously control human body pose and camera viewpoint.
    - Mechanism: The SMPL model function $\{V, F\} \leftarrow M(\beta, \theta)$ generates a 3D mesh from shape parameters $\beta$ and pose parameters $\theta$. The rendering function $\{I_d, I_n, I_s\} \leftarrow R(\{V,F\}, \phi)$ projects the mesh into a depth map $I_d$, a normal map $I_n$, and a skeleton image $I_s$. The depth map implicitly encodes camera viewpoint information, while the normal map captures surface details.
    - Design Motivation: Pure 2D skeleton projection lacks depth information — when the camera is positioned above the subject, the skeleton is vertically compressed, yet the model cannot distinguish whether this is due to camera position or a shorter individual. The 3D depth map from SMPL resolves this ambiguity.

2. **Pose and Viewpoint Diversification Strategy**:

    - Function: Introduce poses and viewpoints absent from the training data by drawing from external data sources.
    - Mechanism: Camera viewpoint is controlled by uniformly sampling elevation $\alpha \sim U(\alpha_{min}, \alpha_{max})$ and azimuth $\gamma \sim U(\gamma_{min}, \gamma_{max})$ ($\alpha$ range: 0–30°, $\gamma$ range: 0–360°). Human body poses are uniformly sampled from an external dataset $P_{ext}$ (e.g., dance videos from Everybody Dance Now).
    - Design Motivation: Viewpoints in Re-ID datasets are heavily concentrated in the horizontal direction with limited angular coverage, and poses are predominantly walking/standing. External poses introduce motion patterns entirely absent from the dataset.

3. **Dual-Branch Diffusion Generation Architecture**:

    - Function: Generate new images conditioned on pose while preserving identity information.
    - Mechanism: Based on the AnimateAnyone architecture, the pretrained SD is cloned into two U-Net branches. The Reference U-Net receives the reference identity image; the Denoising U-Net performs denoising. Identity information is shared between the two branches via self-attention: Q comes from the denoising branch, while K/V come from both the reference and denoising branches. The pose condition is encoded by the Pose Guider network $c_{pose} = G([I_d, I_n, I_s])$ (a stack of convolutional layers) and injected into the denoising U-Net.
    - Design Motivation: Leveraging the broad human appearance knowledge of pretrained SD ensures plausible image generation even for extreme poses unseen during training.

### Loss & Training
- Two-phase training: Phase 1 trains on fashion video datasets to learn general human body representations (15 hours, single A6000 GPU); Phase 2 fine-tunes on Re-ID datasets (resolution 192×384, batch size=4).
- MSE loss with Adam optimizer, lr=1e-5, weight decay=0.01.
- VAE encoder and CLIP image encoder weights are frozen; only the Reference U-Net, Denoising U-Net, and Pose Guider are trained.

## Key Experimental Results

### Main Results

| Method | MSMT17 mAP | MSMT17 R1 | Market1501 mAP | CUHK03(L) mAP |
|---|---|---|---|---|
| CLIP-ReID (baseline) | 68.0 | 85.8 | 89.6 | 95.5 |
| + Pose-dIVE | **71.0** | **87.5** | **90.3** | **97.2** |
| SOLIDER (baseline) | 67.4 | 85.9 | 91.6 | 97.4 |
| + Pose-dIVE | **68.3** | 85.9 | **92.3** | **97.6** |

### Ablation Study

| Pose Aug. | Viewpoint Aug. | MSMT17 mAP | Market1501 mAP | CUHK03(D) mAP |
|---|---|---|---|---|
| × | × | 68.0 | 89.6 | 93.7 |
| × | ✓ | 70.9 | 90.1 | 93.8 |
| ✓ | × | 70.9 | 90.2 | 94.6 |
| ✓ | ✓ | **71.0** | **90.3** | **95.5** |

Data diversity vs. data volume comparison (Market1501, fixed at 30,453 images):

| Training Data | ResNet-50 mAP | SOLIDER mAP |
|---|---|---|
| Original data (11,883 images) | 74.7 | 91.6 |
| + Real images (30,453 images) | 77.8 | 91.8 |
| + Pose-dIVE augmentation (30,453 images) | **80.2** | **92.3** |

### Key Findings
- Pose augmentation and viewpoint augmentation each independently contribute gains; combining both yields the best performance (+3.0 mAP on MSMT17).
- Given the same data volume, Pose-dIVE augmented data outperforms simply adding real images (80.2 vs. 77.8 mAP), demonstrating that diversity matters more than quantity.
- Improvements on real-world test sets are more pronounced (+13.6 mAP, +11.0 R1), confirming that augmentation substantially enhances out-of-domain generalization.

## Highlights & Insights
- **Diversity > Quantity**: Under controlled data-scale experiments, Pose-dIVE augmented data outperforms an equal volume of real images. This finding carries broad implications for data augmentation research — the key lies not in more samples, but in more uniform distributional coverage.
- **SMPL as a Unified Conditioning Bridge**: Using SMPL's 3D mesh as a unified representation for both pose and viewpoint avoids the ambiguity inherent in 2D skeleton projections while enabling decoupled, independent control of both factors. This design paradigm of 3D intermediate representation → 2D conditioning signal is transferable to other generation tasks requiring precise control.
- **General-Purpose Augmentation Framework**: Pose-dIVE can be applied in a plug-and-play manner to arbitrary Re-ID models, consistently yielding improvements on both CLIP-ReID and SOLIDER baselines.

## Limitations & Future Work
- Generated image quality may degrade under extreme poses; despite leveraging SD priors, the approach remains constrained by the training data distribution.
- Elevation angle is restricted to 0–30°; performance under larger elevation angles (e.g., drone viewpoints) has not been validated.
- Augmentation is limited to the appearance domain, without addressing other domain shift factors such as occlusion and lighting variation.
- Future work could explore using video diffusion models to generate temporally consistent augmentation sequences for video-based Re-ID.

## Related Work & Insights
- **vs. GCL**: GCL uses 3D mesh horizontal rotation to vary viewpoints while keeping the original pose unchanged; Pose-dIVE simultaneously diversifies both pose and viewpoint.
- **vs. LSRO/PoseTransfer GANs**: These methods transfer poses only within the dataset; Pose-dIVE introduces entirely new poses from external data sources.
- **vs. 3DInvarReID**: Focuses on 3D body shape reconstruction for long-term Re-ID, which differs from this paper's objective of enhancing training data diversity.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of jointly controlling pose and viewpoint via SMPL is well-conceived, though the overall framework builds upon AnimateAnyone.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four Re-ID benchmarks, two baselines, extensive ablation studies, and real-world testing — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method is described in detail, though some formulations are routine.
- Value: ⭐⭐⭐⭐ A general-purpose Re-ID augmentation framework with strong practical utility and reference value for other pose-sensitive tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[CVPR 2026\] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective](accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)
- [\[CVPR 2026\] FDeID-Toolbox: Face De-Identification Toolbox](fdeidtoolbox_face_deidentification_toolbox.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
