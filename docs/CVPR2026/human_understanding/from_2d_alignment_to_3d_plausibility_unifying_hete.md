---
title: >-
  [Paper Note] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction
description: >-
  [CVPR 2026][Human Understanding][two-hand reconstruction] The paper decouples two-hand reconstruction into 2D structural alignment and 3D spatial interaction alignment. Stage 1 employs a Fusion Alignment Encoder (FAE) to implicitly distill three 2D priors from Sapiens (keypoints, segmentation, depth), eliminating the need for the foundation model at inference (56 fps). Stage 2 maps penetrating poses to physically plausible configurations via a penetration-aware diffusion model with collision gradient guidance. On InterHand2.6M, MPJPE is reduced to 5.36 mm (surpassing SOTA 4DHands by 2.13 mm) and penetration volume is reduced by 7×.
tags:
  - CVPR 2026
  - Human Understanding
  - two-hand reconstruction
  - fusion alignment encoder
  - penetration-free diffusion
  - MANO
  - Sapiens
date: 2026-05-08
content_hash: ed673bf46d0cba40
---

# A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2503.17788](https://arxiv.org/abs/2503.17788)
**Code**: [Project Page](https://gaogehan.github.io/A2P/)
**Area**: Human Understanding / Hand Reconstruction
**Keywords**: two-hand reconstruction, fusion alignment encoder, penetration-free diffusion, MANO, Sapiens

## TL;DR

The paper decouples two-hand reconstruction into 2D structural alignment and 3D spatial interaction alignment. Stage 1 employs a Fusion Alignment Encoder (FAE) to implicitly distill three 2D priors from Sapiens (keypoints, segmentation, depth), eliminating the need for the foundation model at inference (56 fps). Stage 2 maps penetrating poses to physically plausible configurations via a penetration-aware diffusion model with collision gradient guidance. On InterHand2.6M, MPJPE is reduced to 5.36 mm (surpassing SOTA 4DHands by 2.13 mm) and penetration volume is reduced by 7×.

## Background & Motivation

**State of the Field**: Monocular 3D reconstruction of two interacting hands is a critical capability for AR/VR, robotics, and character animation. Large-scale hand datasets (InterHand2.6M/Re:InterHand) have driven advances through data scaling, stronger backbones, and attention-based inter-hand relation modeling (IntagHand/ACR/4DHands). Meanwhile, the effectiveness of foundation-model 2D priors (keypoints/segmentation/depth) and diffusion-based generative priors has been validated in body reconstruction.

**Limitations of Prior Work**: (1) Existing two-hand methods (IntagHand/ACR/4DHands) lack explicit 2D-3D alignment mechanisms, leading to spatial inconsistency and unnatural interactions; (2) 2D cues are unreliable under mutual occlusion, causing frequent finger penetration; (3) Directly deploying foundation models (e.g., Sapiens with 1B parameters) is computationally prohibitive (3 fps), and 2D-3D feature alignment across multi-task predictions is ambiguous; (4) Diffusion priors (InterHandGen) serve only as output regularizers without explicitly modeling 3D spatial interaction.

**Root Cause**: 2D priors are unreliable in occluded regions, necessitating 3D interaction priors; yet 3D generative priors require accurate 2D alignment as an anchor, otherwise drifting to implausible states. The two are mutually dependent yet individually limited.

**Paper Goals**: (1) How to leverage multimodal 2D priors for structural alignment under inference-efficiency constraints; (2) How to achieve physically plausible 3D spatial interaction (eliminating penetration) via generative models.

**Starting Point**: The problem is decoupled into two complementary stages — 2D structural alignment (prior distillation, addressing pose estimation under occlusion) and 3D spatial interaction alignment (conditional diffusion, addressing physical penetration) — with progressive correction resolving failures at their source.

**Core Idea**: During training, Sapiens provides 2D prior guidance; at inference, a distilled lightweight model replaces it (18.7× speedup). A conditional diffusion model with collision gradient guidance then maps penetrating poses to plausible configurations.

## Method

### Overall Architecture

A two-stage pipeline. **Stage 1 (2D Alignment)**: ResNet-50 extracts image features $\mathbf{F}_i$ → Sapiens extracts keypoint/segmentation/depth prior features $\mathbf{F}_k, \mathbf{F}_s, \mathbf{F}_d$ → Projection fuses them into $\mathbf{F}_p$ → Fusion Alignment Encoder (ResNet-50) learns to distill $\mathbf{F}_p$ via MSE → Transformer Encoder fuses $\langle\mathbf{F}_i, \mathbf{F}_p\rangle$ → MANO regressor predicts hand parameters. At inference, Sapiens is removed; only the FAE is used. **Stage 2 (3D Interaction)**: If bilateral hand IoU > 0 and penetration is detected → penetrating MANO parameters are fed as conditions to the diffusion model → DDIM denoising with collision gradient guidance → physically plausible configuration is output.

### Key Designs

1. **Fusion Alignment Encoder (FAE)**

    - **Function**: Leverages multimodal 2D priors from a foundation model during training; replaced by a lightweight distilled model at inference.
    - **Mechanism**: During training, Sapiens (1B parameters) extracts three prior features → a Projection layer fuses them as $\mathbf{F}_p = \text{Proj}(\mathbf{F}_k, \mathbf{F}_s, \mathbf{F}_d)$ → the FAE (lightweight ResNet-50, 52.6M parameters) learns to align with $\mathbf{F}_p$ via MSE loss → at inference, only the FAE replaces Sapiens, achieving 18.7× speedup (3 fps → 56 fps) with only a 0.47 mm increase in MRRPE. Critically, **implicit feature distillation is used rather than extracting explicit prior predictions** (keypoint coordinates/segmentation maps/depth maps), avoiding cascading prediction errors.
    - **Design Motivation**: Directly deploying the foundation model at inference is too slow (1B parameters, 3 fps), while conventional explicit prior prediction with input augmentation accumulates prediction errors. Implicit distillation preserves structural knowledge while drastically reducing inference cost — "foundation-level guidance without foundation-level cost."

2. **Penetration-Aware Diffusion Model**

    - **Function**: Learns a generative mapping from penetrating poses to physically plausible configurations.
    - **Mechanism**: Transformer-based architecture following the MDM-style diffusion process (1000 steps, cosine noise schedule). **Training data construction**: (i) Penetrating outputs from a lower-performance model serve as conditions $\mathbf{X}_c$, with ground truth as targets $\mathbf{X}_0$; (ii) GT MANO parameters are noise-perturbed until penetration occurs, forming paired data. Denoising loss: $\mathcal{L}_{diffusion} = \|\mathbf{X}_0 - \mathcal{D}(\mathbf{X}_t, \mathbf{X}_c)\|_2$. At inference, the model is activated only when bilateral hand IoU > 0 and penetration is detected (most frames are skipped).
    - **Design Motivation**: Unlike InterHandGen (diffusion as output regularization) and Zuo et al. (CNN-based interaction feature extraction), explicitly modeling the "penetration → plausible" mapping is a more direct and effective approach. Conditional diffusion performs "repair" rather than "generation" — taking penetrating poses as input and producing plausible poses as output, which is more stable than generating hand interactions from scratch.

3. **Collision Gradient Guidance**

    - **Function**: Introduces physical collision constraints into the diffusion denoising process.
    - **Mechanism**: After each DDIM denoising step, the estimated $\hat{\mathbf{X}}_0$ is passed through MANO to obtain mesh vertices → (i) Chamfer distances between vertices of both hands are computed as $\mathbf{N}_{ij} = |\mathbf{V}_{t-1}^i - \mathbf{V}_c^j|^2$, retaining near-neighbor pairs where $\mathbf{N}_{ij} < d_{threshold}$; (ii) Normal cosine similarity is checked: $\cos(\theta_{ij}) < \cos(\theta_{thre})$ indicates opposing normals (penetration), while aligned normals indicate normal contact; (iii) A GMoF robust collision loss computes gradients to update: $\hat{\mathbf{X}}_0 = \hat{\mathbf{X}}_0 - \lambda \nabla \mathcal{L}_{collision}$.
    - **Design Motivation**: The hybrid distance–direction criterion precisely distinguishes penetration from natural contact — close distance with opposing normals signals penetration requiring correction, while close distance with aligned normals signals natural contact that should not be disturbed. The GMoF function provides robustness, preventing outlier vertices from dominating the gradient.

### Loss & Training

**Stage 1**: $\mathcal{L}_{total} = \mathcal{L}_{hand}$ (MANO parameters + 3D/2.5D joint L1) $+ \mathcal{L}_{prior}$ (MSE between FAE and fused prior features). Trained on 4× A100, AdamW lr=1e-4 (reduced by 10× at epoch 4), batch size 48. Training data: InterHand2.6M + Re:InterHand + COCO + FreiHAND + HO-3D (substantially fewer datasets than 4DHands). **Stage 2**: L2 denoising loss, 1000-step cosine schedule.

## Key Experimental Results

### Main Results — InterHand2.6M (5fps test)

| Method | MRRPE↓ | MPJPE↓ | MPVPE↓ | IH MPJPE↓ | SH MPJPE↓ |
|--------|--------|--------|--------|-----------|-----------|
| IntagHand | - | 9.95 | 10.29 | 10.27 | 9.67 |
| ACR | - | 8.09 | 8.29 | 9.08 | 6.85 |
| InterWild | 26.74 | 7.85 | 8.16 | 8.24 | 6.72 |
| InterHandGen | 25.42 | 7.50 | 7.78 | 8.13 | 6.47 |
| 4DHands | 24.58 | 7.49 | 7.72 | - | - |
| **Ours** | **21.60** | **5.36** | **5.58** | **5.93** | **4.84** |

### Ablation Study — Incremental Module Addition (InterHand2.6M)

| Configuration | MRRPE↓ | MPJPE↓ | MPJPE-XY↓ | MPJPE-Z↓ |
|---------------|--------|--------|-----------|----------|
| Baseline | 25.30 | 7.77 | 5.21 | 4.54 |
| + Keypoint Prior | 24.71 | 6.48 (−1.29) | 4.28 | 4.43 |
| + Segmentation Prior | 24.52 | 6.19 (−0.29) | 4.21 | 4.40 |
| + Depth Prior | 22.38 | 5.74 (−0.45) | 4.13 | **3.37** |
| + Penetration Diffusion | **21.60** | **5.36** (−0.38) | **3.87** | **3.01** |

### Key Findings

- The three priors are complementary: keypoints contribute most (−1.29 MPJPE); depth prior primarily improves the Z dimension (4.54 → 3.37); segmentation prior provides reliable 2D contours under occlusion.
- On HIC in-the-wild data (excluded from training): MPJPE improves from 9.32 to 6.67 mm versus 4DHands, demonstrating strong generalization.
- Penetration metrics: PenVol decreases from 0.76 to 0.11 (↓7×), PenDist from 0.04 to 0.01, confirming substantial penetration elimination.
- FAE efficiency: 52.6M parameters (vs. 1B), 56 fps (vs. 3 fps), with only +0.47 mm MRRPE degradation.

## Highlights & Insights

- **Large model at training, distilled small model at inference**: The FAE's implicit distillation strategy is a practical realization of "foundation-level guidance without foundation-level cost," achieving 18.7× speedup with negligible accuracy loss.
- **Conditional diffusion as "repair" rather than "generation"**: Mapping penetrating poses to plausible ones is more stable than generating hand interactions from scratch. IoU-based activation ensures inference overhead is incurred only when necessary.
- **Hybrid distance–direction criterion for collision gradient guidance**: Close distance with opposing normals indicates penetration; close distance with aligned normals indicates natural contact. This design precisely distinguishes the two cases and avoids erroneous corrections.
- **Surpassing SOTA with less training data**: 4DHands uses 3 two-hand and 9 single-hand datasets, while this work uses substantially fewer yet achieves a 2.13 mm reduction in MPJPE, demonstrating the intrinsic effectiveness of the proposed method.

## Limitations & Future Work

- 2D priors are unreliable under motion blur, which also degrades the quality of FAE-distilled features.
- Temporal information from video sequences is not exploited; integration with 4DHands-style spatiotemporal modeling is a potential direction.
- The diffusion model introduces additional inference overhead (though activated only upon penetration detection), limiting real-time applicability.
- Collision gradient guidance requires MANO mesh reconstruction and is not directly applicable to non-MANO representations (e.g., implicit hand models).
- Only ResNet-50 is evaluated as the FAE backbone; the effectiveness of lighter alternatives (e.g., MobileNet) remains unexplored.

## Related Work & Insights

- **vs. 4DHands**: 4DHands models inter-hand relations via RAT+SIR but lacks explicit penetration handling. A2P explicitly learns the penetration-to-plausible mapping via diffusion and achieves superior performance with fewer training data.
- **vs. InterHandGen**: The diffusion model serves only as a regularizer, yielding insufficient penetration suppression (PenVol 0.76). A2P explicitly models conditional penetration removal with collision gradient guidance (PenVol 0.11).
- **vs. Zuo et al.**: CNN encoders extract interaction features without strong geometric constraints. A2P's diffusion model operates directly in MANO parameter space.
- The FAE distillation paradigm and conditional diffusion repair strategy are transferable to related tasks such as body reconstruction and human-object interaction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-stage decoupled design of 2D prior distillation and penetration diffusion is novel; the hybrid distance–direction criterion for collision gradient guidance is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on InterHand2.6M/HIC/FreiHAND and in-the-wild data, with detailed ablations on priors, diffusion, FAE efficiency, and penetration metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; pipeline figures are information-rich; the two-stage design logic is internally consistent.
- **Value**: ⭐⭐⭐⭐ A substantial 2.13 mm MPJPE improvement with pronounced penetration elimination represents a meaningful practical advance for hand interaction reconstruction.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)
- [\[CVPR 2026\] LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [\[CVPR 2026\] HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)

<!-- RELATED:END -->
