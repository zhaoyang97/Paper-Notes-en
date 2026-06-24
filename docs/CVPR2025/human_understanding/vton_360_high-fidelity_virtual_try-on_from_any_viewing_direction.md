---
title: >-
  [Paper Note] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction
description: >-
  [CVPR 2025][Human Understanding][3D Virtual Try-On] VTON 360 is proposed, which reformulates 3D virtual try-on as a multi-view consistent 2D virtual try-on expansion problem. By combining pseudo-3D pose representation, multi-view spatial attention, and multi-view CLIP embedding, it achieves high-fidelity virtual try-on from arbitrary viewing directions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D Virtual Try-On"
  - "Multi-View Consistency"
  - "Gaussian Splatting"
  - "Diffusion Models"
  - "Garment Texture Preservation"
date: 2026-05-08
content_hash: 2838c8f64833c30e
---

# VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction

**Conference**: CVPR 2025  
**arXiv**: [2503.12165](https://arxiv.org/abs/2503.12165)  
**Code**: [Project Page](https://scnuhealthy.github.io/VTON360)  
**Area**: Human Understanding  
**Keywords**: 3D Virtual Try-On, Multi-View Consistency, Gaussian Splatting, Diffusion Models, Garment Texture Preservation

## TL;DR

VTON 360 is proposed, which reformulates 3D virtual try-on as a multi-view consistent 2D virtual try-on expansion problem. By combining pseudo-3D pose representation, multi-view spatial attention, and multi-view CLIP embedding, it achieves high-fidelity virtual try-on from arbitrary viewing directions.

## Background & Motivation

Virtual try-on (VTON) is in high demand in e-commerce and fashion design. Although 2D VTON methods have achieved significant progress, they cannot support multi-view rendering. Traditional 3D VTON methods either rely on expensive 3D scanning equipment or reconstruct 3D garment models from 2D images, which lacks multi-view information and leads to insufficient fidelity.

SDS-based methods like DreamVTON can render from arbitrary views, but text-to-image (T2I) models learn semantic-level "concepts" rather than pixel-level precision, limiting their fidelity. GaussianVTON models 3D VTON as a scene editing task; however, because no existing 2D VTON method can generate multi-view 3D-consistent images, garment fidelity and consistency remain problematic.

The core insight of this work is that an equivalence relation exists between a 3D model and its multi-view 2D rendered images. Thus, 3D VTON can be transformed into a consistent editing problem on multi-view 2D images, with the edited 3D model subsequently restored through 3D reconstruction.

## Method

### Overall Architecture

Given an input 3D human model $\mathbf{G}_{\text{src}}$ and a pair of front-and-back view garment images $(g_f, g_b)$, the method consists of three steps: (1) rendering the 3D model into multi-view 2D images; (2) using an enhanced 2D VTON network to perform 3D-consistent editing on the multi-view images; and (3) reconstructing the edited 3D model $\mathbf{G}_{\text{VTON}}$ using Gaussian Splatting. The core innovation lies in the second step, which extends a typical 2D VTON framework (Main UNet + GarmentNet) to support multi-view consistent generation.

### Key Design 1: Pseudo-3D Pose Representation

**Function**: Replaces traditional DensePose to provide cross-view geometrically consistent human pose representation.

**Mechanism**: Uses normal maps $\mathbf{N}$ derived from the SMPL-X 3D human body model as the pose condition, which are encoded by a lightweight PoseEncoder $\mathcal{E}'$ and fed into the Main UNet. Normal maps capture fine-grained surface orientation information, maintaining geometric structure consistency across different viewpoints.

**Design Motivation**: DensePose assigns uniform semantic labels to each body part and lacks 3D geometric consistency, which causes artifacts and temporal inconsistency (such as improper handling of limb boundaries) under multiple views. Normal maps provide smoother, geometrically consistent transitions, supporting realistic shading effects.

### Key Design 2: Multi-View Spatial Attention (MVAttention)

**Function**: Models correlations between features of different views to ensure 3D consistency in multi-view generation.

**Mechanism**: Inspired by temporal attention in video generation, a spatial attention layer is designed. The Query comes from the multi-view features $\mathbf{F}^l$, and the Key/Value is the concatenation of multi-view features with front and back garment features $[\mathbf{F}^l \oplus F_f^l \oplus F_b^l]$. The key innovation is the introduction of a correlation matrix $C$ based on the angular differences of camera rotation matrices:

$$C_{ij} = ((\text{trace}(R_i^T R_j) - 1) / 2 + 1) / 2$$

Similar viewpoints receive higher correlation weights, while distant viewpoints receive lower weights, modulating the attention scores by $C_{ij}$.

**Design Motivation**: Multi-view inputs originate from random azimuth angles with non-uniform spatial intervals; features from similar views are highly correlated, whereas disparate views are relatively independent. Directly applying standard attention fails to model this spatial relationship.

### Key Design 3: Multi-View CLIP Embedding

**Function**: Injects camera viewpoint information into CLIP features, enabling the network to learn garment features related to specific viewpoints.

**Mechanism**: The camera rotation matrix is extracted as a 9-dimensional tensor $\mathbf{r}_i$, which is positionally encoded and projected via an MLP to the same dimension as the CLIP embedding. It is then concatenated with the garment CLIP feature $F^g$ along the token axis to form $Y_i = F^g \oplus \text{MLP}(F_i^c)$, which is utilized in the cross-attention layers of the Main UNet.

**Design Motivation**: The CLIP embedding in standard 2D VTON contains no viewpoint information, failing to distinguish garment details that should appear under different views (e.g., front logo vs. back tag). Injecting camera conditioning improves viewpoint awareness.

### Loss & Training

The standard LDM denoising loss is used: $\mathcal{L}_{\text{ldm}} = \mathbb{E}[\|\epsilon - \hat{\epsilon}_\theta(z_t, t, \eta, \psi, \zeta)\|_2^2]$, where $\eta$ represents the latent representation of the garment and normal map, $\psi$ is the multi-view CLIP embedding, and $\zeta$ represents the agnostic human image. Training is conducted in two stages: first, single-view training for basic capabilities, followed by multi-view training for the MVAttention module.

## Key Experimental Results

### Main Results: Comparison with SOTA Methods

| Method | CLIP_cons ↑ | DINO_sim ↑ | Vote_quality | Vote_align |
|------|------------|-----------|-------------|-----------|
| DreamWaltz | 0.887 | 0.556 | 0.46% | 1.54% |
| TIP-Editor | 0.939 | 0.569 | 0.92% | 0.62% |
| GaussCtrl | 0.931 | 0.577 | 1.08% | 1.38% |
| **VTON 360** | 0.923 | **0.633** | **97.54%** | **96.46%** |

(Thuman2.0 dataset; the trend is consistent on MVHumanNet, with DINO_sim at 0.623 vs. 0.521 for the second-best)

### Ablation Study: Step-by-step Contribution of Three Techniques (Thuman2.0)

| Configuration | CLIP_cons ↑ | DINO_sim ↑ |
|------|-----------|-----------|
| 2D-VTON baseline | 0.892 | 0.609 |
| + Pseudo-3D pose | 0.910 | 0.626 |
| + Multi-view CLIP embedding | 0.913 | 0.631 |
| + MVAttention | **0.923** | **0.633** |

### Key Findings

- DINO similarity (garment texture preservation ability) significantly outperforms all baseline methods, primarily benefiting from pixel-level detail transfer rather than semantic-level concepts.
- In user studies, the method achieves a 97.54% quality preference and a 96.46% alignment preference, demonstrating an overwhelming advantage in subjective perception.
- Good generalization capability is demonstrated on garment images from e-commerce platforms (YOOX, Taobao, TikTok), accurately preserving details such as stripes, logos, and buttons.
- Pseudo-3D pose improves limb generation quality the most, while MVAttention further enhances cross-view consistency.

## Highlights & Insights

1. **Clever reformulation of the problem**: Converting 3D VTON into a "multi-view consistent 2D VTON expansion" leverages the mature 2D VTON technology stack, avoiding the need to build a 3D pipeline from scratch.
2. **Exquisite design of the correlation matrix**: Constructing inter-view correlation weights using the trace relationship of camera rotation matrices is physically intuitive and computationally simple.
3. **Staged training strategy**: Learning basic VTON capability using single views first, followed by multi-view consistency learning, reduces training difficulty.

## Limitations & Future Work

- Requires front and back garment images as input, limiting application scenarios where only a single garment image is available.
- Training is constrained by GPU memory; only 8 views were used during training (compared to 16 during testing). Using more views might further improve the results.
- Relies on the fitting quality of SMPL-X, and the robustness to non-standard body shapes or complex poses remains to be validated.
- The input resolution is fixed at $768 \times 576$, which affects high-resolution detail preservation.
- Try-on scenarios involving complex accessories (e.g., scarves, hats) and lower garments are not yet handled.

## Related Work & Insights

- **CatVTON / OOTDiffusion**: Representative 2D VTON methods. This work extends their GarmentNet + Main UNet frameworks for 3D.
- **GaussianVTON**: Concurrent work that also follows a 2D VTON + 3DGS pipeline but lacks multi-view consistency design, leading to suboptimal results.
- **GaussCtrl**: Implements 3D-aware editing using depth conditions and attention alignment, inspiring the multi-view consistent editing approach.

## Rating

⭐⭐⭐⭐ — Clear problem formulation. The three technical designs have explicit motivations and show significant impact, with the user study demonstrating overwhelming superiority. Limitations lie in the requirement of dual-view garment inputs and dependency on SMPL-X.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[ECCV 2024\] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](../../ECCV2024/human_understanding/wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[CVPR 2025\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)

</div>

<!-- RELATED:END -->
