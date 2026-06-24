---
title: >-
  [Paper Note] GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][Unsigned Distance Function] This paper proposes GaussianUDF, which fits 2D Gaussian planes to surfaces and leverages self-supervision and gradient inference to provide unsigned distance supervision for near-field and far-field regions respectively. This achieves efficient continuous UDF inference within the 3DGS framework for the first time, enabling high-quality open surface reconstruction.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Unsigned Distance Function"
  - "3D Gaussian Splatting"
  - "Open Surface Reconstruction"
  - "2D Gaussian"
  - "Self-supervision"
date: 2026-05-08
content_hash: 2014a1fdd78520e8
---

# GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.19458](https://arxiv.org/abs/2503.19458)  
**Code**: [https://lisj575.github.io/GaussianUDF/](https://lisj575.github.io/GaussianUDF/)  
**Area**: 3D Vision  
**Keywords**: Unsigned Distance Function, 3D Gaussian Splatting, Open Surface Reconstruction, 2D Gaussian, Self-supervision

## TL;DR
This paper proposes GaussianUDF, which fits 2D Gaussian planes to surfaces and leverages self-supervision and gradient inference to provide unsigned distance supervision for near-field and far-field regions respectively. This achieves efficient continuous UDF inference within the 3DGS framework for the first time, enabling high-quality open surface reconstruction.

## Background & Motivation

**Background**: Reconstructing open surfaces (such as clothing, leaves, and other non-closed objects) from multi-view images is an important task in digitization. The mainstream strategy is to learn an Unsigned Distance Function (UDF) via volume rendering and then extract the surface from its zero level set. The advantage of UDF over SDF is its ability to handle arbitrary topologies without requiring closed surfaces.

**Limitations of Prior Work**: Existing UDF methods (such as NeuralUDF, 2S-UDF, VRPrior) are based on NeRF-style volume rendering, which requires ray tracing to find intersections, leading to low training efficiency (usually taking 7-9 hours). Although 3DGS is a promising candidate to accelerate reconstruction with its explicit representation and efficient rasterization, the core bottleneck is the fundamental gap between the discrete explicit representation of 3DGS (a collection of Gaussian primitives) and the continuous implicit representation of UDF.

**Key Challenge**: The gradient field of UDF near the zero level set is extremely complex, as the gradients are undefined (discontinuous) on the surface. Consequently, directly using gradient projections to learn UDF leads to unstable optimization. Meanwhile, because the Gaussian center points are sparse and non-uniform, relying solely on Chamfer Distance of the points cannot provide sufficient surface details.

**Goal**: To bridge the gap between discrete Gaussians and continuous UDFs within the 3DGS framework, achieving efficient and accurate open surface reconstruction.

**Key Insight**: Approximate the surface using 2D Gaussians (thin planes) and leverage the entire Gaussian plane (instead of just the center point) to provide distance supervision near the surface. The far-field is pulled toward the center points using gradient projection, while the near-field is supervised self-supervisedly using sampling points along the Gaussian normal direction.

**Core Idea**: By combining 2D Gaussian surface fitting with self-supervised near-field supervision and gradient-inferred far-field supervision, a differentiable bridge is established from discrete Gaussians to continuous UDFs within 3DGS.

## Method

### Overall Architecture
GaussianUDF jointly optimizes two components: a set of 2D Gaussian primitives $\{g_i\}_{i=1}^I$ and a MLP-parameterized UDF network $f$. 2D Gaussians learn the scene appearance by rendering images via differentiable splatting and minimizing rendering errors, while being constrained to fit the zero level set of the UDF. Based on the fitted Gaussians, supervision signals for the UDF are obtained in the near-field and far-field respectively. The input consists of multi-view RGB images, the output is a continuous UDF field, and the final open surface is extracted using the MeshUDF algorithm.

### Key Designs

1. **Gradient-based Far-field Inference**:

    - **Function**: To provide coarse-grained UDF supervision for spatial regions far from the surface.
    - **Mechanism**: Random query points $\{q_j\}$ are sampled near the Gaussian centers and projected onto the zero level set along the UDF gradient direction in a Neural-Pull manner, yielding projection points $q_j' = q_j - f(q_j) \cdot \nabla f(q_j) / |\nabla f(q_j)|$. The Chamfer Distance $L_{far}$ between these projection points and the Gaussian centers is used as the loss. This encourages the zero level set of the UDF to align with the Gaussian center points.
    - **Design Motivation**: Gaussian centers provide a sparse sampling of the surface. While gradient projection can "pull" distant query points toward the surface, relying solely on this loss would cause holes and noise due to the sparsity and non-uniformity of the centers.

2. **Self-supervised Near-field Supervision**:

    - **Function**: To density-provide UDF ground truth near the surface using the full structure of the 2D Gaussian planes.
    - **Mechanism**: Leveraging the property that 2D Gaussians are inherently thin planes. Root points $\{r_i^h\}$ are sampled on each Gaussian plane, and sampling points $e_{i,h}^b = r_i^h + t_b \cdot n_i / \|n_i\|$ are generated along the normal direction with a random offset $t_b$. Key insight: if the Gaussian plane is fitted to the surface, the offset $|t_b|$ is exactly the unsigned distance from the sampled point to the surface. Therefore, $L_{near} = \|f(e_i^b) - |t_b|\|_1$ is used as the self-supervision signal.
    - **Design Motivation**: Far-field supervision only utilizes Gaussian centers (0D), whereas self-supervision utilizes the entire Gaussian plane (2D), which covers sufficient surface area to fill the gaps between the centers. This is the core mechanism to compensate for Gaussian sparsity.

3. **Gaussian Projection to Zero Level Set**:

    - **Function**: To ensure that the 2D Gaussian centers lie precisely on the zero level set of the UDF, so that the Gaussians truly fit the surface.
    - **Mechanism**: Gaussian centers are projected to the zero level set using the UDF gradient as $\mu_i' = \mu_i - f(\mu_i) \cdot \nabla f / |\nabla f|$, but the **gradient flow through $f$ is blocked** during backpropagation. Then, $L_{proj} = \|\mu_i' - \mu_i\|_2$ is used to directly update the Gaussian positions. Unlike methods like GSPull, this avoids passing gradients through the UDF network, preventing optimization instability caused by the complex gradient field near the zero level set.
    - **Design Motivation**: Unlike SDF, the gradient of UDF is discontinuous and complex at the zero level set, meaning direct backpropagation of gradients often leads to oscillation. By using stop-gradient to treat the projection target as a fixed anchor, the optimization process is stabilized.

### Loss & Training
The total loss is: $L = (1-\lambda_1)L_{rgb} + \lambda_1 L_{ssim} + \lambda_2 L_{far} + \lambda_3 L_{near} + \lambda_4 L_{proj} + \lambda_5 L_{depth} + \lambda_6 L_{norm}$, where $L_{depth}$ is the depth distillation loss (constraining the spacing of Gaussian intersection points along the same ray), and $L_{norm}$ is the normal consistency loss (aligning Gaussian normals with normals derived from the depth map). Training runs for 30k iterations; $L_{far}$ is enabled between iterations 9k-12k, after which $L_{near}$ and $L_{proj}$ are introduced. The UDF network is an 8-layer MLP with 256 hidden units, positional encoding, and ReLU activations, with an absolute value applied to the final layer to ensure non-negativity.

## Key Experimental Results

### Main Results
Chamfer Distance ($\times 10^{-3}$) on the DF3D dataset (open surfaces, 12 clothing items):

| Method | Type | Average CD↓ | Training Time |
|------|------|---------|---------|
| NeuS | SDF | 4.36 | 5.7h |
| 2DGS | SDF | 3.81 | 6min |
| GOF | SDF | 2.49 | 47min |
| NeuralUDF | UDF | 2.15 | 8.6h |
| 2S-UDF | UDF | 1.98 | 7.8h |
| VRPrior | UDF | 1.71 | 9.2h |
| **GaussianUDF** | **UDF** | **1.60** | **1.6h** |

Average CD on the DTU dataset (closed surfaces, 15 scenes):

| Method | Average CD↓ |
|------|---------|
| 2DGS | 0.80 |
| GOF | 0.74 |
| GSPull | 0.75 |
| VRPrior | 0.71 |
| **GaussianUDF** | **0.68** |

### Ablation Study

| Configuration | DTU CD↓ |
|------|---------|
| Only Far | 0.99 |
| Far + Near | 0.78 |
| Far + Proj | 0.88 |
| w/o Warp | 0.74 |
| w/o Near | 0.77 |
| w/o Proj | 0.76 |
| **Full Model** | **0.68** |

### Key Findings
- $L_{near}$ (self-supervised near-field loss) contributes the most: removing it domesticates a jump in CD from 0.68 to 0.77, validating the core value of using Gaussian planes to provide dense supervision.
- Using only far-field loss (Only Far) performs the worst (0.99), as the Gaussian point cloud itself is sparse and noisy, failing to provide reliable geometric information.
- The contribution of $L_{proj}$ (projection constraint) to registration accuracy (0.76 $\rightarrow$ 0.68) reflects the importance of precisely aligning Gaussians to the zero level set.
- For open surface tasks, SDF methods (NeuS, 2DGS, GOF) produce 1.5 to 2.7 times higher errors than UDF methods due to their attempts to construct closed meshes.
- On closed surfaces (DTU), GaussianUDF outperforms dedicated SDF methods, despite assuming open surfaces.

## Highlights & Insights
- **Ingenious Self-Supervision Mechanism**: By utilizing the geometric property that a 2D Gaussian "is inherently a thin plane", ground truth distances are obtained by sampling along the normal, eliminating the need for external annotations. This "representation as supervision" concept can be generalized to other implicit field learning tasks.
- **Stabilization via Stop-gradient**: The complex gradient field near the zero level set of UDF is a widely recognized challenge. The authors address this by using stop-gradient to freeze the projection target and directly move the Gaussian positions instead, replacing implicit gradient flow with explicit constraints, which is simple but highly effective.
- **Point Cloud Deformation Verification**: Morphing an arbitrary point cloud (e.g., an apple or donut) into target clothing shapes using the learned UDF intuitively demonstrates the global correctness of the distance field, rather than just its correctness near the zero level set.

## Limitations & Future Work
- Reconstructing textureless structures is weaker compared to SDF methods, as the higher flexibility of UDF increases optimization difficulty.
- Mesh extraction from UDF remains an open problem (the quality of final reconstruction is constrained by algorithms like MeshUDF).
- Although the training time of 1.6 hours is 5 times faster than NeRF-based methods, it is still much slower than 2DGS (6 minutes), with the main bottlenecks being the UDF MLP and sampling.
- Lack of experiments on mask-supervised setups, which might yield further improvements in foreground/background separated scenes.

## Related Work & Insights
- **vs VRPrior**: VRPrior requires additional depth prior data to learn the UDF, while GaussianUDF is purely self-supervised and achieves higher accuracy (1.60 vs 1.71).
- **vs 2DGS/GOF**: These SDF methods model closed surfaces, which leads to double-layer issues on open surfaces. GaussianUDF naturally supports open topologies through UDF.
- **vs GSPull**: GSPull uses a similar gradient projection to learn SDFs, but the gradient of the zero level set in SDFs is more stable. The stop-gradient strategy in GaussianUDF is a specialized design tailored for the unstable gradients of UDFs.
- This work shows that 3DGS and implicit fields can complement each other: Gaussians provide efficient rendering and sparse surface sampling, while MLP provides a continuous distance field.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ This work realizes UDF inference within the 3DGS framework for the first time, featuring ingenious designs of self-supervision and stop-gradient mechanisms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ It covers both open/closed surfaces and synthetic/real data with thorough ablations, though it lacks large-scale scene testing.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly described and well-illustrated, though there are somewhat many equations.
- **Value**: ⭐⭐⭐⭐ It offers a highly efficient novel approach for open surface reconstruction and bridges the gap between explicit and implicit representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Distilling Unsigned Distance Function for Surface Reconstruction from 3D Gaussian Splatting](../../CVPR2026/3d_vision/distilling_unsigned_distance_function_for_surface_reconstruction_from_3d_gaussia.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)
- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
