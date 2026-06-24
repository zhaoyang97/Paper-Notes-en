---
title: >-
  [Paper Note] Hiding Imperceptible Noise in Curvature-Aware Patches for 3D Point Cloud Attack
description: >-
  [ECCV 2024][3D Vision][3D point cloud attack] Proposes the Wavelet Patches Attack (WPA) method, which employs wavelet transform to analyze local curvature structures of point clouds and hides adversarial perturbations within curvature-consistent patches—perturbing along tangent planes in flat regions and along normal vectors in sharp regions—achieving a more imperceptible 3D point cloud attack compared to existing methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D point cloud attack"
  - "adversarial perturbation"
  - "wavelet transform"
  - "curvature-aware"
  - "geometric imperceptibility"
date: 2026-05-08
content_hash: 439430167de7f768
---

# Hiding Imperceptible Noise in Curvature-Aware Patches for 3D Point Cloud Attack

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision / Adversarial Attack  
**Keywords**: 3D point cloud attack, adversarial perturbation, wavelet transform, curvature-aware, geometric imperceptibility

## TL;DR

Proposes the Wavelet Patches Attack (WPA) method, which employs wavelet transform to analyze local curvature structures of point clouds and hides adversarial perturbations within curvature-consistent patches—perturbing along tangent planes in flat regions and along normal vectors in sharp regions—achieving a more imperceptible 3D point cloud attack compared to existing methods.

## Background & Motivation

**Background**: With the ubiquity of 3D sensors (e.g., LiDAR, RGB-D cameras), point clouds have been widely applied in safety-critical scenarios such as autonomous driving and robotic navigation. However, deep point cloud classification models (e.g., PointNet, DGCNN) have been proven vulnerable to adversarial attacks—minor point displacements can lead to misclassification. Existing 3D attack methods mainly rely on global distance constraints (e.g., Chamfer distance, Hausdorff distance) to restrict the magnitude of perturbations.

**Limitations of Prior Work**: (1) **Global distance constraints overlook local geometric properties**—point clouds are highly structured 3D data, and geometric properties vary significantly across different regions (flat surfaces vs. sharp edges). Global distance constraints restrict perturbation magnitudes uniformly across all points, which leads to perturbations in flat regions being visually highly conspicuous (due to disruption of smoothness), despite having small distances, while restricting perturbations in sharp regions too conservatively. (2) **Lack of explicit modeling of 3D geometric structures**—existing methods optimize perturbations directly in the Cartesian coordinate space, without understanding the underlying geometric manifold structure of the point cloud.

**Key Challenge**: Adversarial attacks require perturbations to be large enough to alter the model's predictions, yet imperceptible enough to avoid detection. In 3D point clouds, "imperceptibility" is not just about small distance metrics; more importantly, perturbations must not destroy local geometric features—flat areas should remain flat, and sharp corners should retain their distinct edges.

**Goal**: (1) How to identify local regions with different curvature properties in a point cloud; (2) How to customize directional perturbation strategies based on local geometric characteristics; (3) How to guarantee attack success rate while maintaining geometric imperceptibility.

**Key Insight**: The authors introduce the wavelet transform to analyze the spectral properties of point clouds—low-frequency wavelet coefficients correspond to flat surface regions, and high-frequency wavelet coefficients correspond to sharp edge/corner regions. Utilizing wavelet coefficients allows a natural decomposition of the point cloud into patches of varying curvature levels. Directional perturbations are then tailored to the curvature properties of each patch: flat patches are perturbed along the tangent planes (maintaining surface smoothness), and sharp patches are perturbed along normal vectors (preserving clean edges).

**Core Idea**: Decompose the point cloud into curvature-aware patches using the wavelet transform, and add adversarial perturbations along tangent planes in flat patches and along normal directions in sharp patches, thus "hiding" the noise in alignment with local geometric properties.

## Method

### Overall Architecture

Given an input 3D point cloud $\mathcal{P} = \{p_i\}_{i=1}^N$, the WPA method consists of three steps: (1) Wavelet transform—transform the point cloud into the spectral domain using a graph wavelet operator to obtain the wavelet coefficients of each point; (2) Curvature-aware patch decomposition—group points into flat patches (slow-variation) and sharp patches (fast-variation) according to the magnitude of their wavelet coefficients; (3) Directional perturbation—apply different directional adversarial perturbations to different types of patches and optimize via PGD to mislead the classification model.

### Key Designs

1. **Graph Wavelet Transform**:

    - **Function**: Transforms point clouds from the spatial domain to the spectral domain, capturing geometric structural information across different scales.
    - **Mechanism**: First, a k-NN graph $G$ is constructed on the point cloud to compute the graph Laplacian matrix $L$. Then, the spectral graph wavelet transform $\Psi_s = U g(s\Lambda) U^T$ is applied, where $U$ is the eigenvector matrix of $L$, $\Lambda$ is the diagonal matrix of eigenvalues, $g$ is the wavelet kernel function, and $s$ is the scale parameter. The wavelet coefficient $w_i$ of each point reflects the degree of geometric variation in its neighborhood—points with large $|w_i|$ are located in high-curvature regions (edges, corners), while those with small $|w_i|$ lie in low-curvature regions (planes, smooth surfaces).
    - **Design Motivation**: Compared to directly computing curvature (such as principal or Gaussian curvature), the wavelet transform offers two main advantages: (a) multi-scale analysis can capture geometric structures at different granularities; (b) spectral decomposition is more robust and insensitive to point cloud sampling density. Moreover, wavelet coefficients provide a purely data-driven measure of the "intensity of geometric variation".

2. **Curvature-Aware Patch Decomposition**:

    - **Function**: Decomposes the point cloud into two types of patches: flat (slow-variation) and sharp (fast-variation).
    - **Mechanism**: A threshold $\tau$ is set to classify points with wavelet coefficients $|w_i| < \tau$ into flat patches $\mathcal{P}_{slow}$, and points with $|w_i| \geq \tau$ into sharp patches $\mathcal{P}_{fast}$. The threshold $\tau$ is adaptively determined based on the distribution of wavelet coefficients of all points (e.g., using the median or mean + standard deviation). For each patch, the local coordinate system—tangent plane basis vectors $\{t_1, t_2\}$ and normal vector $n$—is computed for each point via PCA on its neighborhood.
    - **Design Motivation**: The key to processing different curvature regions separately lies in human visual sensitivity: minor bumps on flat surfaces are highly conspicuous, whereas minor displacements on sharp edges are virtually unnoticed. Patch-level processing allows the perturbation strategy to adapt to different visual sensitivities.

3. **Directional Adversarial Perturbation**:

    - **Function**: Designs perturbations in different directions based on patch types to hide the noise geometrically.
    - **Mechanism**: For a point $p_i$ in a flat patch $\mathcal{P}_{slow}$, the perturbation is constrained within the tangent plane: $\delta_i = \alpha_1 t_1 + \alpha_2 t_2$ (sliding along the tangent plane to maintain surface smoothness). For a point $p_i$ in a sharp patch $\mathcal{P}_{fast}$, the perturbation is primarily aligned along the normal vector: $\delta_i = \beta \cdot n$ (perturbations along the normal direction have minimal visual impact in sharp regions). The final optimization objective is to maximize the classification loss while satisfying the directional constraints: $\max_\delta L_{cls}(f(\mathcal{P} + \delta), y)$, subject to $\delta_i \in \text{TangentPlane}(p_i)$ for slow patches, $\delta_i \parallel n_i$ for fast patches. This is optimized iteratively via Project Gradient Descent (PGD).
    - **Design Motivation**: Perturbing flat surfaces along the tangent plane does not create new bumps or depressions, thereby preserving surface smoothness. Perturbing sharp regions along the normal direction merely slightly adjusts the sharpness of the edges, which is visually imperceptible due to the pre-existing high curvature variation. This directional constraint ensures that perturbations align with local geometric properties and are "hidden" within the original geometric structures.

### Loss & Training

The attack loss uses the negative value of cross-entropy loss (for untargeted attacks) or cross-entropy of the target class (for targeted attacks). An additional geometric-preserving regularization term is added: $L_{geo} = \sum_i \|K(p_i + \delta_i) - K(p_i)\|$ (local curvature change penalty). The total optimization objective is $\max_\delta L_{cls} - \lambda L_{geo}$. Using the PGD algorithm, the perturbation is projected back to the directional constraint set after each update step.

## Key Experimental Results

### Main Results

| Defense Method | Metric | WPA (Ours) | 3D-Adv | GeoA3 | kNN Attack |
|----------------|--------|------------|--------|-------|------------|
| No Defense | ASR↑ | 100.0 | 100.0 | 100.0 | 100.0 |
| SRS | ASR↑ | 94.2 | 87.3 | 89.1 | 85.6 |
| SOR | ASR↑ | 91.5 | 82.4 | 84.7 | 79.3 |
| DUP-Net | ASR↑ | 88.3 | 76.5 | 79.2 | 72.1 |
| IF-Defense | ASR↑ | 85.7 | 71.2 | 73.8 | 68.4 |

### Ablation Study

| Configuration | ASR↑ | Chamfer↓ | Subjective Score↑ | Description |
|---------------|------|----------|-------------------|-------------|
| WPA (Full) | 94.2 | 0.0023 | 4.3/5 | Complete method |
| Global constraint (no directionality) | 95.1 | 0.0021 | 2.8/5 | High attack success rate but visually conspicuous |
| Tangent-plane perturbation only | 88.5 | 0.0019 | 4.1/5 | Insufficient attack strength |
| Normal perturbation only | 90.3 | 0.0025 | 3.6/5 | Bumps appear in flat regions |
| Random direction constraint | 91.7 | 0.0022 | 3.2/5 | Random directions are inferior to curvature-awareness |

### Key Findings
- WPA achieves a significantly higher attack success rate under defense methods compared to other baselines (+7.1% against SOR defense compared to the second-best method), indicating that geometrically consistent perturbations are indeed harder to detect and defend against.
- The subjective rating experiment serves as an important complement—while global constraint methods yield smaller Chamfer distances, human evaluation considers WPA to have superior imperceptibility (4.3 vs 2.8/5), proving that relying solely on distance metrics is insufficient.
- Ablation studies demonstrate that the selection of directional constraints is critical: pure tangent-plane perturbation exhibits insufficient attack capabilities (as the normal directions may be more sensitive to classifiers), while pure normal perturbation lacks concealment in flat regions.
- On objects with rich geometric structures (e.g., airplane wings, car body surfaces), the imperceptibility advantage of WPA becomes even more pronounced.

## Highlights & Insights

- **Using wavelet transform for point cloud curvature analysis is an ingenious technical choice**: Compared to traditional curvature computation methods (requiring surface fitting and second-order derivative calculation), the wavelet transform is more robust and offers multi-scale analysis capabilities. This tool can be transferred to other tasks such as point cloud denoising and feature extraction.
- **The concept of "applying different perturbation directions to different geometric regions"** has broad applicability: it is not limited to adversarial attack tasks but can also be leveraged in scenarios like point cloud data augmentation and deformation synthesis using similar geometric-aware strategies.
- The introduction of subjective score experiments is compelling—the "imperceptibility" of 3D adversarial attacks cannot be measured solely by distance metrics; human perception is the ultimate standard.

## Limitations & Future Work

- The computational overhead of the graph wavelet transform is relatively high (requiring eigendecomposition), which could become a bottleneck for large-scale point clouds. Approximation algorithms or fast wavelet transform techniques can be explored for acceleration.
- Although directional constraints enhance imperceptibility, they also restrict the degrees of freedom for perturbations, which may reduce the attack success rate in some scenarios. Adaptive constraint strengths—relaxing constraints in regions highly sensitive to detection—could be investigated.
- Evaluations are primarily conducted on classification tasks, lacking studies on more complex tasks such as detection and segmentation.
- Adaptive defense strategies specifically targeting the WPA strategy have not been considered—future work could design methods to detect WPA attacks using curvature consistency checks.
- The method can be extended to mesh attacks—meshes have more explicit surface structures, making tangent planes and normal vectors easier to calculate precisely.

## Related Work & Insights

- **vs 3D-Adv**: 3D-Adv uses Chamfer distance as a global constraint and ignores local geometric properties. WPA's geometric-aware strategy comprehensively outperforms 3D-Adv in terms of imperceptibility.
- **vs GeoA3**: GeoA3 constrains perturbations using normal consistency and curvature preservation, but still applies global constraints. WPA achieves finer region-adaptive strategies via patch decomposition.
- **vs kNN Attack**: kNN Attack employs k-NN distance for local constraints but does not differentiate perturbation directions. WPA's directional constraint maintains geometric invariance more fundamentally.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of wavelet transform, curvature awareness, and directional perturbation is highly novel, elegantly merging signal processing theory with adversarial attacks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and defenses, including human subjective ratings.
- Writing Quality: ⭐⭐⭐⭐ The technical description is clear, and the method motivation is fully articulated.
- Value: ⭐⭐⭐⭐ Offers a new perspective for studying 3D adversarial robustness, with wavelet analysis tools transferable to other 3D tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FLAT: Flux-Aware Imperceptible Adversarial Attacks on 3D Point Clouds](flat_flux-aware_imperceptible_adversarial_attacks_on_3d_point_clouds.md)
- [\[ECCV 2024\] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)
- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](../../CVPR2025/3d_vision/sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[ECCV 2024\] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)
- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)

</div>

<!-- RELATED:END -->
