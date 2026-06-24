---
title: >-
  [Paper Note] LUCAS: Layered Universal Codec Avatars
description: >-
  [CVPR 2025][3D Vision][codec avatar] Proposes LUCAS, the first universal prior Avatar model that disentangles the face and hair into layered meshes. By employing a shared expression code and independent decoders, it enables natural face-hair interactions while supporting both real-time mesh rendering (45 FPS on mobile devices) and high-fidelity Gaussian rendering, achieving state-of-the-art performance in cross-identity zero-shot driving.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "codec avatar"
  - "layered representation"
  - "universal prior model"
  - "face-hair disentanglement"
  - "Pixel Codec Avatar"
  - "Gaussian splatting"
  - "real-time rendering"
date: 2026-05-08
content_hash: e510ee162d7de382
---

# LUCAS: Layered Universal Codec Avatars

**Conference**: CVPR 2025  
**arXiv**: [2502.19739](https://arxiv.org/abs/2502.19739)  
**Code**: [https://lsn33096.github.io/LUCAS/](https://lsn33096.github.io/LUCAS/)  
**Area**: 3D Vision  
**Keywords**: codec avatar, layered representation, universal prior model, face-hair disentanglement, Pixel Codec Avatar, Gaussian splatting, real-time rendering

## TL;DR

Proposes LUCAS, the first universal prior Avatar model that disentangles the face and hair into layered meshes. By employing a shared expression code and independent decoders, it enables natural face-hair interactions while supporting both real-time mesh rendering (45 FPS on mobile devices) and high-fidelity Gaussian rendering, achieving state-of-the-art performance in cross-identity zero-shot driving.

## Background & Motivation

**Background**: Codec Avatars achieve high-fidelity 3D head avatar reconstruction through VAE architectures. Pixel Codec Avatar (PiCA) enables highly efficient pixel-level rendering but is limited to personalized training. Universal models such as URAvatar achieve cross-identity generalization but suffer from poor hair modeling quality.

**Limitations of Prior Work**: (1) Fundamental limitations of **single-mesh representations**: The face and hair share the same UV space, where the hair is allocated to an extremely small UV region, resulting in severely degraded reconstruction of long hair. (2) Smoothness regularization forces coupled motion of the face and hair—for instance, hair should naturally drape, but its deformation is constrained under a single mesh. (3) PiCA is a personalized model trained per individual, offering poor scalability. (4) The guide meshes of existing Universal Prior Models (UPMs) are inaccurate, leading to drifted Gaussian anchors.

**Key Challenge**: The face and hair possess fundamentally different geometric and motion characteristics—the face exhibits rigid skeletal motion with expression deformations, while the hair exhibits soft, hanging, physical dynamics. Modeling both within a single mesh prevents simultaneous optimization.

**Key Insight**: Extend PiCA into a universal model (uPiCA), then utilize a layered representation to disentangle the face and hair, assigning independent hypernetworks, decoders, and pixel decoders to each.

## Method

### Overall Architecture

Three core components:
1. **Identity Hypernetwork** ($\mathcal{E}_{id}^{face}$ / $\mathcal{E}_{id}^{hair}$): Generates identity-specific biases from neutral textures and geometries.
2. **Shared Expression Encoder** ($\mathcal{E}_{exp}$): Universally encodes expression changes.
3. **Layered Decoder** ($\mathcal{D}^{face}$ / $\mathcal{D}^{hair}$): Independently decodes the geometry and appearance of the face and hair.

### Key Designs

**1. Avatar Dehairing**
- **Function**: Constructs a dehaired bald geometry baseline to serve as the foundation for the independent hair layer.
- **Mechanism**: Starting with 5 naturally bald subjects, a linear morphable model is iteratively constructed (via EM + factor analysis) and incrementally extended to subjects with hair, inferring the bald geometry of the occluded scalp using known non-hair regions.
- **Design Motivation**: Accurate bald geometry forms the foundation of layered representation—the hair mesh must be built upon an accurate scalp surface to deform correctly.

**2. Universal Layered Prior Model**
- **Function**: Extends uPiCA to a layered architecture where the face and hair are modeled independently.
- **Mechanism**:
    - A shared expression encoder extracts a unified expression code $z$ (ensuring synchronized face-hair control).
    - Face decoder: $\mathbf{g}^{face} = \mathcal{D}_g^{face}(z, \eta)$, $\mathbf{e}^{face} = \mathcal{D}_e^{face}(z, \omega, \eta)$.
    - Hair decoder receives an additional head pose $h$: $\mathbf{g}^{hair} = \mathcal{D}_g^{hair}(z, \eta, h)$, $\mathbf{e}^{hair} = \mathcal{D}_e^{hair}(z, \omega, \eta, h)$.
    - Multi-mesh joint rendering: The two meshes are concatenated and rasterized jointly, then passed to independent pixel decoders using face/hair masks, respectively.
- **Design Motivation**: Hair movement is jointly affected by head pose and facial expression (e.g., frowning causes the hair to drape). Thus, the hair decoder must receive both $z$ (expression) and $h$ (head pose).

**3. Gaussian Rendering on Layered Meshes**
- **Function**: Leverages the accurate geometry of layered meshes as Gaussian anchors to improve high-fidelity rendering quality.
- **Mechanism**: Gaussians are parameterized on the vertices of the layered PiCA guide mesh, where the face and hair possess independent hypernetworks and decoders. The decoders output the position offset $\delta t_k$, rotation $q_k$, scale $s_k$, color $d_k^c$, and opacity $o_k$ for each Gaussian.
- **Key Regularization**: The delta position loss $\mathcal{L}_\Delta$ prevents hair Gaussians from drifting into facial regions, and bald face area Gaussians from drifting into hair regions.

### Loss & Training

$$\mathcal{L}_{total} = \lambda_{pica}\mathcal{L}_{pica} + \lambda_{gs}\mathcal{L}_{gs} + \lambda_{dehair}\mathcal{L}_{dehair}$$

- $\mathcal{L}_{pica}$: Layered reconstruction loss (photometric $\mathcal{L}_I$ + depth $\mathcal{L}_D$ + normal $\mathcal{L}_N$ + mesh tracking $\mathcal{L}_M$ + smoothness $\mathcal{L}_S$ + KL $\mathcal{L}_{KL}$ + segmentation $\mathcal{L}_{seg}$)
- $\mathcal{L}_{gs}$: Gaussian rendering loss + scale regularization + delta position loss
- $\mathcal{L}_{dehair}$: Dehairing geometry loss (heavy weight in the early stage with decay during training)

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| †PiCA (mesh, personalized) | 32.05 | 0.8895 | 0.2678 |
| †LUCAS (mesh, personalized) | **33.52** | **0.9044** | **0.2479** |
| †LUCAS (gs, personalized) | **35.20** | **0.9286** | **0.2407** |
| *uPiCA (mesh, universal) | 32.56 | 0.8971 | 0.2594 |
| *LUCAS (mesh, universal) | 33.03 | 0.9073 | 0.2537 |
| *URAvatar (gs, universal) | 33.12 | 0.9034 | 0.2464 |
| ***LUCAS (gs, universal)** | **34.56** | **0.9201** | **0.2394** |

LUCAS gs universal model outperforms URAvatar by +1.4 dB PSNR.

### Ablation Study

| Configuration | Train PSNR↑ | Unseen PSNR↑ |
|---|---|---|
| w/o Expression Code (Hair) | 34.10 | 31.91 |
| w/o Hair Segmentation | 34.03 | 31.80 |
| **Full Model** | **34.50** | **32.58** |

Expression code gains +0.4 dB for hair; hair segmentation regularization is crucial for fine hair strand reconstruction.

### Key Findings

1. **Layered > Single Mesh**: LUCAS mesh outperforms uPiCA (on the same architectural baseline) across all metrics, especially in long hair scenarios.
2. **Universal > Personalized**: The Gaussian rendering of the universal LUCAS model (34.56 PSNR) approaches or even exceeds the level of personalized PiCA + Gaussian models.
3. **Expression Code is Essential for Hair**: Without the expression code, hair cannot deform naturally with expressions (e.g., draping when frowning).
4. **Layered Meshes Improve Gaussian Quality**: Precise anchor geometry allows Gaussians to fit the target without requiring large spatial offsets, thereby reducing artifacts.

## Highlights & Insights

1. **First Mesh-Based UPM**: While previous universal models mostly relied on volumetric or Gaussian representations, LUCAS is the first to achieve a mesh-based UPM, supporting real-time rendering at 45 FPS on mobile devices.
2. **Physical Soundness of Layered Disentanglement**: The face exhibits skeletal-muscle-driven rigid deformation, whereas the hair undergoes gravity/inertia-driven soft deformation. Layered modeling aligns with physical priors.
3. **Shared Expression Code + Independent Decoding**: Elegantly balances face-hair coordination (synchronized control) and independence (separate deformation).
4. **Iterative Expansion from 5 Bald Subjects**: The progressive strategy for dehairing is highly practical.

## Limitations & Future Work

1. Degradation still occurs during extreme hair deformations (e.g., head shaking), particularly with unseen poses in zero-shot driving.
2. The model was trained exclusively on Meta's internal multi-view capture system (110 cameras, 76 identities), representing an extremely high barrier to data acquisition.
3. Relighting capabilities are not addressed.
4. Dehairing depends heavily on the accuracy of HRNet segmentation, which may fail on hairstyles with blurry hair lines.
5. The dual-branch architecture increases parameter volume and training complexity, making it less friendly for resource-constrained scenarios.

## Related Work & Insights

- **PiCA (Ma et al.)**: Pixel-level decoding + real-time rendering $\rightarrow$ LUCAS inherits its high-efficiency rendering while expanding it to a universal and layered model.
- **URAvatar (Cao et al.)**: Gaussian UPM $\rightarrow$ Offers good rendering quality but poor underlying mesh geometry; LUCAS provides superior Gaussian anchors using layered meshes.
- **MEGANE (Lombardi et al.)**: Layered modeling of face + glasses $\rightarrow$ LUCAS generalizes layered modeling to the face + hair for the first time, which is significantly more challenging due to the non-rigid nature of hair.
- **Insight**: The concept of layered disentanglement can be extended to full-body avatars (independent layers for body, clothing, and accessories) and animal avatars (layered body and fur).

## Rating

⭐⭐⭐⭐ — The first mesh-based UPM with layered face-hair disentanglement holds great engineering value. The experiments are thorough (covering personalized/universal/ablation/driving setups), and it supports real-time rendering on mobile devices. However, the data acquisition threshold is high, and handling of extreme deformations remains insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](../../ICCV2025/3d_vision/haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[CVPR 2025\] Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior](vid2avatar-pro_authentic_avatar_from_videos_in_the_wild_via_universal_prior.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
