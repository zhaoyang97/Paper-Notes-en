---
title: >-
  [Paper Note] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling
description: >-
  [ICCV 2025][Human Understanding][3D morphable model] imHead proposes the first large-scale implicit 3D head morphable model. Through a global-local decoupled architecture trained on a dataset of 4,000 identities, it achieves both a compact implicit representation and localized facial editing, surpassing existing methods in reconstruction accuracy and editing flexibility.
tags:
  - ICCV 2025
  - Human Understanding
  - 3D morphable model
  - implicit function
  - head modeling
  - local editing
  - large-scale dataset
date: 2026-05-08
content_hash: d516f5285220c193
---

# ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling

**Conference**: ICCV 2025  
**arXiv**: [2510.10793](https://arxiv.org/abs/2510.10793)  
**Code**: [Project Page](https://rolpotamias.github.io/imHead/)  
**Area**: Human Understanding  
**Keywords**: 3D morphable model, implicit function, head modeling, local editing, large-scale dataset

## TL;DR

imHead proposes the first large-scale implicit 3D head morphable model. Through a global-local decoupled architecture trained on a dataset of 4,000 identities, it achieves both a compact implicit representation and localized facial editing, surpassing existing methods in reconstruction accuracy and editing flexibility.

## Background & Motivation

3D Morphable Models (3DMMs) are a core technology in face modeling, with broad applications in gaming, computer graphics, and virtual reality. However, traditional PCA-based 3DMMs suffer from two fundamental limitations:

**Insufficient expressiveness of linear models**: PCA models fail to capture complex local facial variations, producing overly smooth surfaces that lack high-frequency detail (e.g., hair strands, wrinkles). Although nonlinear approaches (e.g., graph neural networks) offer improvements, they still fall short of photorealistic quality.

**Topological consistency constraints**: 3DMMs require all scans in a dataset to share a consistent topology and precise dense correspondences. Establishing such correspondences is extremely time-consuming and error-prone, restricting 3DMMs to the facial region and making it difficult to extend them to the full head.

Deep Implicit Functions (DIFs) estimate the signed distance from a spatial point to a surface via neural networks, providing a continuous, topology-free 3D representation. Existing implicit 3DMMs (e.g., NPHM) can model the full head but face two issues:

- **Globally entangled implicit space**: NPHM partitions the implicit space into local components and appends a global identity code, but the global information is "baked" into the local networks, preventing local editing.
- **Small-scale datasets**: Existing methods are trained on fewer than 300 identities, severely limiting identity diversity and failing to capture real-world distributions.

The core insight of imHead is: **retain a single compact global identity space and achieve local editability through region-specific intermediate representations**, rather than directly partitioning the implicit space.

## Method

### Overall Architecture

imHead is an auto-decoder-style implicit model $\mathcal{M}: (\mathbf{x}, \mathbf{z}_{id}, \mathbf{z}_{exp}) \mapsto y \in \mathbb{R}$, composed of three core modules:

1. **Decomposition Network (DecNet)**: Decomposes the global identity code into local region embeddings.
2. **Structure Blending Fusion Network (FusionNet)**: Aggregates local features and predicts the global implicit field.
3. **Expression Warping Module**: Learns the mapping from observation space to canonical space to model expression deformation.

### Key Designs

1. **Dataset Construction (4,000 Identities)**

   Using raw scans from the MimicMe dataset (5,000 subjects, 20 expressions), a large-scale complete head dataset is constructed through the following pipeline:

   - Multi-view rendering + RetinaFace detection of 2D landmarks → triangulation to extract 3D landmarks
   - ICP rigid registration to FLAME canonical space → fitting optimization to obtain soft correspondences
   - NPHM model fitting to complete the full head
   - Non-rigid ICP (NICP) registration to recover identity-specific details

   After filtering, 4,000 identities (~50,000 scans) are retained — **10×** larger than previous implicit head datasets. Demographic coverage includes 57% male / 43% female, ages 1–81, and multiple ethnicities.

2. **Global-Local Decoupled Identity Network**

   **DecNet $\mathcal{T}_\theta$**: Maps the global identity code $\mathbf{z}_{id} \in \mathbb{R}^{256}$ to $K=39$ local embeddings $\{\mathbf{z}_{id}^j \in \mathbb{R}^{32}\}$ via a simple linear projection layer. The elegance of this design lies in maintaining a compact global implicit space (only 256 dimensions, 8.5× smaller than NPHM's 2,176 dimensions) while enabling local editability through intermediate representations.

   **Local-Part Networks $\{g_j\}_{j=0}^K$**: $K$ independent local networks, each receiving query coordinate $\mathbf{x}$ and local embedding $\mathbf{z}_{id}^j$, and extracting high-dimensional features $\mathbf{f}_x^j = g_j(\mathbf{x} - \mathbf{k}_j, \mathbf{z}_{id}^j)$, where $\mathbf{k}_j$ is the keypoint of the corresponding region (regressed by LandmarkNet), serving as the origin of the local coordinate frame. Symmetric regions share network parameters. Positional encoding $\gamma(\mathbf{x} - \mathbf{k}_j)$ is used to capture high-frequency details.

   **FusionNet $\mathcal{F}_\theta$**: Aggregates local features via distance-based weighting: $\hat{\mathbf{f}}_x = \sum_j^K w(\mathbf{x}, \mathbf{k}_j) \mathbf{f}_x^j$, with weights $w(\mathbf{x}, \mathbf{k}_j) = \frac{e^{-\|\mathbf{x}-\mathbf{k}_j\|_2/\sigma}}{\sum_j^K e^{-\|\mathbf{x}-\mathbf{k}_j\|_2/\sigma}}$, and the fusion network regresses the SDF value as $y = \mathcal{F}_\theta(\mathbf{x}, \hat{\mathbf{f}}_x)$.

   Key distinction from NPHM: imHead does not directly blend local neural fields (which would cause discontinuities during editing); instead, it guides the global implicit field by fusing intermediate features, ensuring smoothness during editing.

3. **Backward Expression Warping**

   Unlike NPHM's forward warping, imHead adopts backward warping: $\Delta \mathbf{x} = \mathcal{E}(\mathbf{x}_{obs}, \mathbf{z}_{id}, \mathbf{z}_{exp})$, $\mathbf{x}_{can} = \mathbf{x}_{obs} + \Delta \mathbf{x}$.

   Advantage: Forward warping requires iterative root-finding to establish soft correspondences, which is sensitive to initialization and computationally expensive. Backward warping directly maps observation-space points to canonical space, yielding a smoother fitting process and a **3× speedup** (40s vs. 138s).

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{eik} + \lambda_{kpt}\mathcal{L}_{kpt} + \lambda_{sym}\mathcal{L}_{sym} + \lambda_{reg}\mathcal{L}_{reg}$$

- $\mathcal{L}_{rec}$: SDF drives surface points to zero + gradient matches surface normals
- $\mathcal{L}_{eik}$: Eikonal regularization (unit-norm gradient)
- $\mathcal{L}_{kpt}$: Keypoint regression loss
- $\mathcal{L}_{sym}$: Symmetry constraint on implicit codes of symmetric regions
- $\mathcal{L}_{reg}$: Regularization on identity and expression codes

## Key Experimental Results

### Main Results

**Identity Reconstruction (Neutral Expression, Single Scan)**:

| Method | NPHM CD↓ | NPHM NC↑ | NPHM F@5mm↑ | MimicMe CD↓ | MimicMe NC↑ | MimicMe F@5mm↑ |
|--------|----------|----------|-------------|-------------|-------------|----------------|
| FLAME | 1.244 | 0.943 | 0.632 | 1.336 | 0.929 | 0.606 |
| NPHM† | 0.514 | 0.980 | 0.866 | 0.598 | 0.967 | 0.827 |
| monoNPHM† | 0.514 | 0.980 | 0.866 | 0.593 | 0.968 | 0.829 |
| **imHead-Full†** | **0.459** | **0.988** | **0.898** | **0.533** | **0.986** | **0.873** |

imHead outperforms all baselines on both datasets, while its implicit space is only 256 dimensions (8.5× more compact than NPHM's 2,176 dimensions).

**Expression Reconstruction**: imHead-Full achieves CD=0.485 and F@5mm=0.912 on the NPHM dataset, and CD=0.563 and F@5mm=0.878 on MimicMe, again leading all baselines. Fitting speed is 40s vs. NPHM's 138s.

### Ablation Study

| Configuration | NPHM CD↓ | NPHM NC↑ | NPHM F@5mm↑ | MimicMe CD↓ | Note |
|---------------|----------|----------|-------------|-------------|------|
| w. Local Lat. (d=312) | 0.876 | 0.915 | 0.689 | 0.874 | Pure local implicit space, severely underperforms |
| w. Local Lat. (d=1248) | 0.775 | 0.948 | 0.743 | 0.767 | Large local space, still inferior to global |
| w. Local+Global (d=1344) | 0.494 | 0.964 | 0.841 | 0.569 | NPHM-style design, comparable but 5× larger space |
| w/o FusionNet | 0.595 | 0.954 | 0.808 | 0.674 | Direct local SDF regression, non-smooth normals |
| w/o Local Canonical | 0.723 | 0.934 | 0.723 | 0.884 | Missing local coordinate frame, large performance drop |
| **imHead-Full** | **0.459** | **0.988** | **0.898** | **0.533** | Full model |

### Key Findings

- A global implicit space captures global distributional patterns better than a local space, even when the local space dimensionality is as large as 1,248 — compared to only 256 for the global space.
- Intermediate feature fusion via FusionNet is critical for avoiding discontinuities during editing.
- Introducing the large-scale dataset (imHead-MimicMe vs. imHead-NPHM) reduces CD from 0.571 to 0.546 on the MimicMe test set (~20% improvement), validating the importance of data scale.
- imHead is robust to noisy inputs, maintaining reasonable identity reconstruction under Gaussian noise at 1.5× standard deviation.

## Highlights & Insights

- **Elegant global-local balance**: DecNet decomposes a compact global space into local intermediate representations, simultaneously achieving compression efficiency and editing flexibility — an exceptionally elegant design.
- **Natural emergence of local editing**: The model is trained purely for reconstruction yet naturally supports independent editing of facial regions and region swapping (e.g., exchanging the nose or hair between two identities) without any additional constraints.
- **Correspondence preservation in canonical space**: Even under extreme expressions (e.g., open mouth), backward warping maintains facial topological consistency — analogous to the advantages of traditional 3DMMs.

## Limitations & Future Work

- Implicit model inference is slower than explicit 3DMMs, requiring extensive point sampling and marching cubes post-processing.
- High-frequency thin structures such as individual hair strands are difficult to model precisely.
- Local editing is constrained by the fixed number of anchor points, and boundaries may be influenced by neighboring local networks.
- The dataset exhibits racial bias (73% Caucasian), with insufficient diversity particularly in the hair region.
- Dense 1-to-1 correspondences are not available (unlike explicit models).

## Related Work & Insights

- The part-level intermediate representation concept from SPAGHETTI directly inspired this work, though the paper applies it to the more challenging domain of human faces.
- The design choice of backward warping elegantly circumvents the root-finding problem of forward warping, effectively simplifying the entire fitting pipeline.
- The large-scale dataset construction pipeline (raw scan → FLAME registration → NPHM completion → NICP detail recovery) offers important reference value for the community.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Global-local decoupled architecture is novel; backward warping simplifies the pipeline
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers reconstruction, generation, editing, and correspondence across multiple dimensions with detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ Technical details are clear with complete derivations
- **Value**: ⭐⭐⭐⭐⭐ Large-scale implicit head model with local editing capability — a foundational contribution to the 3D face/head modeling field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](../../CVPR2026/human_understanding/lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)
- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)

</div>

<!-- RELATED:END -->
