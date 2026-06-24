---
title: >-
  [Paper Note] LaRa: Efficient Large-Baseline Radiance Fields
description: >-
  [ECCV 2024][3D Vision][Large-baseline reconstruction] The LaRa feed-forward reconstruction model is proposed, which unifies local and global reasoning through a **Gaussian Volume** representation and a **Group Attention Layer**. It reconstructs 360° radiance fields from large-baseline views using only 4 images, and outperforms more computationally demanding methods like LGM while requiring only **4×A100 training for 2 days**.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Large-baseline reconstruction"
  - "feed-forward radiance fields"
  - "Gaussian volume"
  - "group attention"
  - "2D Gaussian Splatting"
date: 2026-05-08
content_hash: 691331543551b3db
---

# LaRa: Efficient Large-Baseline Radiance Fields

**Conference**: ECCV 2024  
**arXiv**: [2407.04699](https://arxiv.org/abs/2407.04699)  
**Code**: [https://apchenstu.github.io/LaRa/](https://apchenstu.github.io/LaRa/)  
**Area**: 3D Vision  
**Keywords**: Large-baseline reconstruction, feed-forward radiance fields, Gaussian volume, group attention, 2D Gaussian Splatting

## TL;DR

The LaRa feed-forward reconstruction model is proposed, which unifies local and global reasoning through a **Gaussian Volume** representation and a **Group Attention Layer**. It reconstructs 360° radiance fields from large-baseline views using only 4 images, and outperforms more computationally demanding methods like LGM while requiring only **4×A100 training for 2 days**.

## Background & Motivation

**Background**: Neural radiance fields have achieved excellent results in per-scene optimization and small-baseline settings, but feed-forward large-baseline reconstruction remains challenging.

**Limitations of Prior Work**: 
   - Feature matching-based methods (MVSNeRF, MuRF) rely on overlapping image regions and cannot handle large baselines.
   - Global attention-based methods (LGM, GRM) ignore the locality of 3D reconstruction and require massive GPU resources in the range of 32×A100.
   - A lack of 3D inductive biases leads to blurry reconstructions.

**Key Challenge**: Global attention can model long-range dependencies but is computationally expensive and ignores local geometric constraints, whereas local matching is efficient but fails to handle large viewpoint differences.

**Goal**: Achieve high-quality, large-baseline, feed-forward 3D reconstruction using limited computational resources (academic-grade GPUs).

**Key Insight**: Divide the volume into local groups for intra-group cross-attention (simulating local matching), and then propagate inter-group information using a 3D CNN (achieving global coordination).

**Core Idea**: Unify local feature matching and global information propagation within a transformer using group attention, enabling efficient large-baseline radiance field reconstruction.

## Method

### Overall Architecture

Given $M=4$ images and camera parameters, LaRa outputs a Gaussian Volume in three steps:

1. **Feature Extraction**: A DINO encoder extracts 2D features, injects camera information via Plücker rays, and back-projects them into a 3D feature volume $\mathbf{V}_f$.
2. **Volume Transformer**: A group attention layer progressively updates a learnable embedding volume $\mathbf{V}_e$, outputting a Gaussian volume $\mathbf{V}_{\mathcal{G}}$.
3. **Coarse-to-Fine Decoding**: 2D Gaussian parameters are decoded from the volume features, and high-resolution images are rendered via efficient rasterization.

$$\mathbf{V}_{\mathcal{G}} = \{\mathcal{G}_i^k\}_{k=1}^K = \mathbf{f}(\mathbf{v}; \mathbf{I}, \boldsymbol{\pi})$$

### Key Designs

1. **Gaussian Volume**: Each voxel stores $K=2$ 2D Gaussian primitives, each containing opacity $\alpha$, tangent vector $\mathbf{t}$, scale $\mathbf{S}$, spherical harmonics coefficients, and a displacement offset $\Delta \in [-1,1]^3$. The primitive location is defined as $\mathbf{p}_i^k = \mathbf{v}_i + r \cdot \Delta_i^k$, where $r = 1/32$ is the maximum displacement range. This structures the unordered point set prediction problem as local offset prediction within voxels, reducing the learning difficulty. Additionally, 2D Gaussian Splatting (rather than 3DGS) is used to facilitate surface regularization and mesh extraction.

2. **Group Attention Layer**: The volume is unfolded into $G=16$ local groups. Cross-attention is performed only within each group, and then a 3D CNN propagates information across groups. The core equations are:

$$\dot{\mathbf{V}}_e^{g,j} = \text{GroupCrossAttn}(\text{LN}(\mathbf{V}_e^{g,j}), \mathbf{V}_f^g) + \mathbf{V}_e^{g,j}$$

$$\ddot{\mathbf{V}}_e^{g,j} = \text{MLP}(\text{LN}(\dot{\mathbf{V}}_e^{g,j})) + \dot{\mathbf{V}}_e^{g,j}$$

$$\mathbf{V}_e^{j+1} = \text{3DCNN}(\text{LN}(\ddot{\mathbf{V}}_e^j)) + \ddot{\mathbf{V}}_e^j$$

These three sub-layers each have residual connections, and 12 layers are stacked in total. Different groups are processed in parallel along the batch dimension, significantly boosting training efficiency. Key Insight: While $G=1$ (global attention) requires 22 days to train for 30 epochs, $G=16$ requires only 2 days and achieves better performance.

3. **Coarse-to-Fine Decoding**: 

    - **Coarse Module**: A lightweight MLP decodes voxel features into 2D Gaussian parameters, rendering RGB, depth, and opacity maps.
    - **Fine Module**: Projects the Gaussian primitive centers onto the coarse rendering results and the upsampled features of the original images. It uses a displacement feature $|\hat{\mathbf{D}}_{\mathbf{p}} - z_{\mathbf{p}}|$ (the difference between the rendered depth and the primitive depth) to achieve occlusion-aware reasoning. It then predicts the residual spherical harmonics coefficients via cross-attention and MLP:

$$\text{SH}_{i,k}^{\text{fine}} = \text{SH}_{i,k}^{\text{coarse}} + \text{SH}_{i,k}^{\text{residuals}}$$

Design Motivation: The DINO encoder and attention layers tend to lose high-frequency texture information. The fine module compensates for this by directly querying the original image features.

4. **Plücker Ray Modulation**: Instead of extrinsic/intrinsic matrices, Plücker rays (the cross product of camera position and ray direction) are used to encode camera information, which is injected into 2D features via AdaLN. The advantage of this parametrization is its independence from object scale, camera position, and focal length, thereby enhancing generalization.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\mathcal{I}, \hat{\mathcal{I}}) + \mathcal{L}_{\text{SSIM}}(\mathcal{I}, \hat{\mathcal{I}}) + \mathcal{L}_{\text{Reg}}$$

The regularization term (enabled after epoch 15):

$$\mathcal{L}_{\text{Reg}} = \gamma_d \mathcal{L}_d + \gamma_n \mathcal{L}_n$$

- $\mathcal{L}_d = \sum_{i,j} \omega_i \omega_j |z_i - z_j|$: Distillation loss, which concentrates ray weights on the surface ($\gamma_d=1000$).
- $\mathcal{L}_n = \sum_i \omega_i(1 - \mathbf{n}_i^\top \mathbf{N})$: Normal consistency loss ($\gamma_n=0.2$).

Training details: AdamW, lr=$2\times10^{-4}$ with cosine annealing, 50 epochs (50K iterations per epoch), trained on 4×A100-40G GPUs. Features 125M trainable parameters. Dataset: Objaverse with 264K scenes, where 4 input views are selected via K-means, and 8 views are used for supervision.

## Key Experimental Results

### Main Results - Novel View Synthesis (4 Input Views)

| Method | Gobjaverse PSNR↑ | GSO PSNR↑ | Co3D PSNR↑ | Gobjaverse LPIPS↓ |
|------|------------------|-----------|------------|-------------------|
| MVSNeRF | 14.48 | 15.21 | 12.94 | 0.1856 |
| MuRF | 14.05 | 12.89 | 11.60 | 0.3018 |
| LGM (32×A100-80G) | 19.67 | 23.67 | 13.81 | 0.1576 |
| **Ours-fast (2 days, 4×A100-40G)** | **25.30** | **26.79** | **21.56** | **0.1027** |
| **Ours (3.5 days)** | **26.14** | **27.65** | **21.64** | **0.0932** |

### Ablation Study - Number of Groups and Module Effects

| Design Variant | Gobjaverse PSNR↑ | GSO PSNR↑ | Geometric Accuracy (0.01)↑ |
|---------|------------------|-----------|----------------|
| G=4 (Large groups / fewer groups) | 22.27 | 23.06 | 31.0% |
| G=8 | 23.80 | 25.30 | 42.8% |
| w/o $\mathcal{L}_{\text{Reg}}$ | 26.16 | 27.71 | 45.6% |
| Coarse module only | 25.06 | 26.28 | 52.2% |
| **Full model (G=16)** | **25.30** | **26.79** | **52.2%** |

### Key Findings

- Outperforms all baselines on all datasets by a significant margin: PSNR of 26.14 vs LGM's 19.67 (+32.9% relative gain) on Gobjaverse.
- Extremely computationally efficient: 4×A100-40G (2 days) vs LGM's 32×A100-80G (a 32× difference in GPU hours).
- Generalizes well on real-world Co3D data (21.64), whereas LGM only achieves 13.81 due to its reliance on fixed camera-to-object distances.
- Using group attention with $G=16$ significantly outperforms $G=4$ (+3 PSNR), as local attention aligns better with the nature of 3D matching.
- Removing the regularization term slightly improves rendering metrics but degrades geometric quality and introduces floaters.
- Coarse-to-fine decoding yields around +0.5 PSNR improvement in texture details.

## Highlights & Insights

- The group attention layer is a key contribution: it embeds the local matching nature of 3D reconstruction into the transformer design, making it more efficient and effective than brute-force global attention.
- Plücker ray modulation renders the model robust to scene scale and focal length variations, which is critical for generalization on unconstrained data like Co3D.
- The displacement feature $|\hat{D} - z|$ used in coarse-to-fine decoding elegantly addresses the occlusion reasoning problem.
- Multi-voxel offset representation instead of absolute coordinate prediction is an efficient design, transforming unordered point set generation into structured regression.
- The extremely low training resource requirement (academically reproducible) is a powerful response to industrial-level approaches like LGM.

## Limitations & Future Work

- The volume resolution is fixed at $64^3$, which may limit the representation capacity for large scenes or high details.
- Only bounded object reconstruction is demonstrated, with unbounded/outdoor scenes remaining untested.
- Using only two Gaussian primitives per voxel may be insufficient for thin structures and semi-transparent objects.
- Although training is efficient, the GPU memory consumption of the volume transformer during inference is still constrained by the resolution.
- Reliance on DINO features as the image encoder may degrade performance when handling stylized images outside of DINO's training distribution.

## Related Work & Insights

- **MVSNeRF**: A representative cost volume + volume rendering method, limited to small baselines.
- **LGM / GRM**: Concurrent works that generate 3DGS using a global transformer; they yield good results but require massive computational resources.
- **2D Gaussian Splatting**: Adopted as the rendering primitive in this paper, which is more beneficial for surface modeling and mesh extraction than 3DGS.
- **Insights**: The group attention strategy can be generalized to other 3D tasks, such as voxel transformers in point cloud segmentation or 3D detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of group attention and Gaussian volume yields significant improvements in large-baseline reconstruction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets + zero-shot generalization + mesh extraction + detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework diagrams and coherent logic across modules.
- **Value**: ⭐⭐⭐⭐⭐ — Strong performance, highly economical training, and excellent reproducibility, making it highly valuable for feed-forward 3D reconstruction research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)
- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[ECCV 2024\] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [\[ECCV 2024\] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)

</div>

<!-- RELATED:END -->
