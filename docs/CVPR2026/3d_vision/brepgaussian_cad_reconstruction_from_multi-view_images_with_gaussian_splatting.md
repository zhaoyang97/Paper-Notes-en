---
title: >-
  [Paper Note] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][CAD reconstruction] BRepGaussian is the first method to reconstruct complete B-rep CAD models directly from multi-view images. It employs a two-stage 2D Gaussian splatting framework to learn edge and patch features, followed by parametric fitting to produce watertight boundary representations, without requiring point cloud supervision.
tags:
  - CVPR 2026
  - 3D Vision
  - CAD reconstruction
  - B-rep
  - Gaussian splatting
  - parametric surface fitting
  - contrastive learning
date: 2026-05-08
content_hash: 960b4d57c3194965
---

# BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2602.21105](https://arxiv.org/abs/2602.21105)
**Code**: Coming soon (to be released upon acceptance)
**Area**: 3D Vision
**Keywords**: CAD reconstruction, B-rep, Gaussian splatting, parametric surface fitting, contrastive learning

## TL;DR

BRepGaussian is the first method to reconstruct complete B-rep CAD models directly from multi-view images. It employs a two-stage 2D Gaussian splatting framework to learn edge and patch features, followed by parametric fitting to produce watertight boundary representations, without requiring point cloud supervision.

## Background & Motivation

**Background**: CAD reconstruction (reverse engineering) is a classical problem in computer vision and graphics. Conventional methods primarily take high-quality point clouds as input, perform semantic segmentation to obtain patch labels, and then fit parametric primitives. Learning-based methods such as SPFN and ParSeNet have achieved considerable success.

**Limitations of Prior Work**: Acquiring high-quality point clouds is costly and requires specialized equipment. Existing methods demand extensive manual annotation and exhibit limited generalization to novel shapes, heavily relying on dataset-specific network designs.

**Key Challenge**: Image data is far more accessible and scalable than point clouds, yet a significant gap exists between images and parametric 3D modeling. All prior methods require high-quality point clouds as an intermediate step that cannot be bypassed.

**Goal**: To recover a complete B-rep representation—including parametric faces, edges, vertices, and their topological connections—directly from multi-view RGB images.

**Key Insight**: 2D Gaussian Splatting (2DGS) is used as an intermediate representation. Its flat, disk-shaped primitives are naturally aligned with the planar and low-curvature surfaces prevalent in CAD models, and each Gaussian can carry learnable semantic features.

**Core Idea**: 2DGS is extended into an edge- and patch-aware representation. A two-stage training scheme decouples geometry/edge learning from patch instance learning, after which parametric fitting on the labeled point cloud yields the final B-rep model.

## Method

### Overall Architecture

The input consists of multi-view RGB images of a CAD object. The full pipeline comprises four steps: (1) extracting edge masks and patch masks from 2D images using an edge detector and SAM; (2) two-stage 2DGS training—first learning geometry and edge features, then learning patch instance features; (3) converting Gaussian primitives into a dense labeled point cloud; and (4) a constraint-guided parametric fitting module that assembles the point cloud into a watertight B-rep model. The output is a complete B-rep CAD model containing parametric faces (planes/cylinders/spheres), edges (lines/curves), and vertices.

### Key Designs

1. **Two-Stage Gaussian Splatting Training**

    - **Function**: Decouples geometry/edge learning and patch instance learning into two independent stages.
    - **Mechanism**: In Stage 1, each 2DGS Gaussian is augmented with a scalar edge value $e_i \in [0,1]$. An edge map $E(u) = \sum_i w_i e_i$ is rendered via alpha compositing and supervised against 2D edge detection results using an L2 loss. In Stage 2, all geometric parameters (position xyz, spherical harmonic coefficients, etc.) are frozen, and only a 16-dimensional feature vector $\mathbf{f}_i \in \mathbb{R}^{16}$ per Gaussian is trained.
    - **Design Motivation**: In joint training, the complex gradients from patch contrastive learning disrupt geometry reconstruction quality. Freezing geometric parameters allows each stage to focus on a single learning objective, which experiments confirm to be the most stable and accurate strategy.

2. **Contrastive Learning for Patch Instances**

    - **Function**: Learns cross-view consistent 3D patch instance labels in the absence of cross-view mask correspondences.
    - **Mechanism**: A triplet loss is employed. For each mask region $\mathcal{M}_k$, an anchor $\mathbf{p}_a$, a positive sample $\mathbf{p}_p$ (within the same mask), and the hardest negative sample $\mathbf{p}_n$ (the pixel with the smallest feature distance from other masks) are sampled. The triplet loss is constructed using cosine distance $d(\mathbf{p}_i, \mathbf{p}_j) = 1 - \tilde{\mathbf{f}}_{\mathbf{p}_i} \cdot \tilde{\mathbf{f}}_{\mathbf{p}_j}$: $\mathcal{L}_{\text{tri}} = \max(0, d(\mathbf{p}_a, \mathbf{p}_p) - d(\mathbf{p}_a, \mathbf{p}_n) + m)$.
    - **Design Motivation**: Patches carry instance-level rather than semantic-category labels, and SAM masks across different views cannot be directly matched in a one-to-one manner. Contrastive learning enables automatic cross-view patch clustering in feature space.

3. **Adaptive Sampling from Gaussians to Point Cloud**

    - **Function**: Converts trained Gaussian primitives into a dense point cloud with edge and patch labels.
    - **Mechanism**: Edge regions require many elongated ellipsoidal Gaussians, while flat regions require only a few near-spherical ones. The center of each Gaussian is sampled; for ellipsoidal Gaussians whose major-to-minor axis ratio is not extreme, four additional points are sampled along the ellipse, ensuring that the sampled points align with the true surface distribution.
    - **Design Motivation**: Using only Gaussian centers leads to undersampling in edge regions, which degrades subsequent parametric fitting accuracy.

4. **Constraint-Guided Parametric Fitting**

    - **Function**: Fits parametric primitives (planes/cylinders/spheres) from the labeled point cloud and assembles them into a watertight B-rep.
    - **Mechanism**: A five-step procedure is followed: (a) RANSAC is applied to fit three primitive types for each patch; (b) intersection lines/curves between primitive pairs are computed; (c) edge point clouds constrain the valid parameter range of line/curve segments; (d) three-plane and two-line intersection point clustering yields vertices; (e) bottom-up assembly with Boolean operations produces a clean watertight B-rep.
    - **Design Motivation**: The hierarchical extraction (face → edge → vertex → assembly) fully exploits the patch and edge label information obtained from Gaussian training.

### Loss & Training

- **Stage 1**: $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{geo}} + 0.1 \mathcal{L}_{\text{edge}}$, where $\mathcal{L}_{\text{geo}} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$
- **Stage 2**: Triplet loss $\mathcal{L}_{\text{tri}}$ with hard negative mining and margin hyperparameter $m$

## Key Experimental Results

### Main Results

Patch segmentation evaluation (Precision/Recall/F1) on the ABC-NEF dataset:

| Method | Input | Prec ↑ | Rec ↑ | F1 ↑ |
|--------|-------|--------|-------|------|
| ParSeNet | GT point cloud | 0.511 | 0.265 | 0.349 |
| PCER-Net | GT point cloud | 0.876 | 0.912 | 0.894 |
| SED-Net | GT point cloud | 0.949 | 1.000 | 0.974 |
| ParSeNet | Densified point cloud | 0.623 | 0.236 | 0.343 |
| PCER-Net | Densified point cloud | 0.536 | 0.792 | 0.639 |
| **BRepGaussian** | **Multi-view images** | **0.890** | **0.918** | **0.904** |

CAD reconstruction quality comparison ($D_c$: Chamfer Distance $\times 10^{-2}$, $D_h$: Hausdorff Distance $\times 10^{-1}$):

| Method | Input | CD(Surface) | CD(Curve) | HD(Surface) | HD(Curve) |
|--------|-------|-------------|-----------|-------------|-----------|
| Point2CAD | Ours labels | 3.38 | 5.42 | 2.413 | 3.858 |
| Point2CAD | PCER-Net | 7.08 | 20.45 | 3.394 | 7.276 |
| Split-and-Fit | Densified | 6.23 | 13.98 | 3.523 | 4.962 |
| **BRepGaussian** | **Ours point cloud** | 4.90 | **5.01** | **3.351** | **3.626** |

### Ablation Study

| Configuration | Result | Remarks |
|---------------|--------|---------|
| Two-stage training | Best | Geometry is not disrupted by patch learning |
| Single-stage joint training | Degraded | Patch gradients interfere with geometry reconstruction |
| Feature dimension d=16 | Best | Optimal feature space size for patch instances |
| Center-only sampling | Degraded | Undersampling in edge regions |
| Ellipse-adaptive sampling | Best | Sufficient edge coverage |

### Key Findings

- BRepGaussian's patch segmentation F1 from images (0.904) surpasses PCER-Net's result using GT point clouds (0.894), demonstrating that features learned from multi-view images via contrastive learning are more effective than direct point cloud segmentation.
- Curve reconstruction metrics are globally best (CD=5.01, HD=3.626), confirming that edge detection provides decisive guidance for subsequent parametric fitting.
- Point2CAD using the paper's labels achieves a slightly lower surface CD (3.38 vs. 4.90), but qualitative analysis reveals redundant patches, making the actual reconstruction quality inferior to BRepGaussian's more compact output.

## Highlights & Insights

- **First end-to-end pipeline from images to B-rep**: The method entirely bypasses point cloud acquisition and extends the advantages of Gaussian splatting to structured 3D modeling. This paradigm shift demonstrates that GS is capable not only of rendering but also of engineering-grade parametric reconstruction.
- **Two-stage strategy of frozen geometry and feature-only training**: A simple yet effective decoupling of geometric and semantic learning that avoids gradient conflicts in multi-task training. This approach is transferable to any GS task requiring semantic annotation on top of existing geometry.
- **Judicious choice of 2DGS**: Flat, disk-shaped primitives are naturally aligned with the planar and low-curvature surfaces of CAD models, yielding surface sampling quality far superior to that of 3DGS.

## Limitations & Future Work

- Only three primitive types are supported (planes, cylinders, spheres); free-form surfaces such as B-splines and NURBS cannot be handled.
- SAM mask quality on low-texture CAD images is limited and requires manual correction (~3 minutes per object), leaving room for improved automation.
- Evaluation is conducted only on the ABC-NEF subset; generalization to real-world captured CAD parts has not been validated.
- Primitive fitting relies on traditional RANSAC; differentiable fitting methods could be explored to enable end-to-end optimization.

## Related Work & Insights

- **vs. Point2CAD**: A pure fitting method that depends on external labels. When supplied with the paper's labels, it achieves the lowest surface CD but produces redundant patches; BRepGaussian's constraint-guided fitting yields cleaner results.
- **vs. SED-Net**: Achieves the highest F1 (0.974) using GT point clouds but fails to generalize to densified point clouds reconstructed from images. BRepGaussian, starting from images, exhibits stronger generalization.
- **vs. Curve-Aware GS**: That work recovers only parametric curves, whereas BRepGaussian further recovers complete face + edge + vertex + topology structures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First complete pipeline from images to B-rep; a pioneering contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough comparisons on ABC-NEF, but real-world validation is absent
- Writing Quality: ⭐⭐⭐⭐ Clear structure with an intuitive pipeline figure
- Value: ⭐⭐⭐⭐⭐ Opens a new image-based paradigm for CAD reverse engineering

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniSplat: Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images](unisplat_3d_representations_unposed.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_human-scene_reconstruction_from_multi-person_multi-view_video_in_a_sing.md)
- [\[CVPR 2026\] MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)

</div>

<!-- RELATED:END -->
