---
title: >-
  [Paper Note] HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition
description: >-
  [CVPR 2025][Signed Distance Function] This paper proposes HotSpot, which utilizes the classical relationship between the screened Poisson equation and distance fields to design a new heat loss. It provides an asymptotically sufficient condition for optimizing neural signed distance functions, ensuring that the implicit function converges to the true distance field. It significantly outperforms existing methods in 2D/3D surface reconstruction with complex topologies.
tags:
  - "CVPR 2025"
  - "Signed Distance Function"
  - "Neural Implicit Surfaces"
  - "Point Cloud Reconstruction"
  - "Heat Equation"
  - "Surface Reconstruction"
date: 2026-05-08
content_hash: 0f3f3306dc08dab8
---

# HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition

**Conference**: CVPR 2025  
**arXiv**: [2411.14628](https://arxiv.org/abs/2411.14628)  
**Code**: [https://github.com/zimo-wang/HotSpot](https://github.com/zimo-wang/HotSpot)  
**Area**: Other  
**Keywords**: Signed Distance Function, Neural Implicit Surfaces, Point Cloud Reconstruction, Heat Equation, Surface Reconstruction

## TL;DR
This paper proposes HotSpot, which utilizes the classical relationship between the screened Poisson equation and distance fields to design a new heat loss. It provides an asymptotically sufficient condition for optimizing neural signed distance functions, ensuring that the implicit function converges to the true distance field. It significantly outperforms existing methods in 2D/3D surface reconstruction with complex topologies.

## Background & Motivation

**Background**: Neural signed distance functions (neural SDFs) are the dominant scheme for implicit surface representation, widely used in 3D reconstruction, inverse rendering, and collision detection. Existing methods primarily optimize neural networks to approximate the true SDF using an eikonal loss (constraining the gradient norm to be 1) and a boundary loss.

**Limitations of Prior Work**: The eikonal loss is only a necessary condition but not a sufficient condition for an SDF—even if the implicit function satisfies a gradient norm of 1 almost everywhere, it may still not be a true distance function. Furthermore, the eikonal loss faces instability during optimization (backward diffusion causing divergence). The surface area regularization added by existing methods to alleviate ill-posedness distorts the distance field, making it difficult to balance detail preservation and redundant boundary elimination.

**Key Challenge**: All existing loss functions depending solely on gradient norms fail to rule out non-SDF solutions, as discontinuous jumps in gradients can satisfy the eikonal equation while not representing a true distance function. This fundamental theoretical defect causes the optimization to easily fall into local optima, failing in reconstructions of complex topologies (high-genus shapes).

**Goal**: Design a new loss function whose minimization provides an asymptotically sufficient condition, ensuring that the output converges to the true distance function while offering optimization stability and naturally penalizing excess surface area.

**Key Insight**: The authors leverage the classical mathematical relationship between the screened Poisson equation and distance fields—where the logarithmic transformation of the heat field asymptotically converges to the true distance field as the absorption coefficient $\lambda \to \infty$. This relationship has been used in computer graphics to approximate geodesic distances but has never been applied to optimize neural SDFs.

**Core Idea**: Transform the screened Poisson equation of heat conduction into a heat loss that can directly act on the implicit function. Combined with boundary and eikonal losses, this constructs an asymptotically sufficient optimization framework.

## Method

### Overall Architecture
Given an input point cloud without normal information, the goal is to find a neural network $u(\mathbf{x})$ that approximates the true SDF $f(\mathbf{x})$. Building upon the standard boundary loss $L_{\text{boundary}}$ and eikonal loss $L_{\text{eikonal}}$, HotSpot introduces a new heat loss $L_{\text{heat}}$ derived from the screened Poisson equation, optimizing a weighted combination of the three.

### Key Designs

1. **Heat Loss**:

    - **Function**: Provides an asymptotically sufficient condition to guarantee that the implicit function converges to the true distance function.
    - **Mechanism**: Based on the relationship between the screened Poisson equation $\nabla^2 h - \lambda^2 h = 0$ and distance $\lim_{\lambda \to \infty} \frac{1}{\lambda}\ln(h_\lambda) = -d_\Gamma$, it transforms the heat field equation into an energy functional with respect to the implicit function $u$ using the change of variable $h = e^{-\lambda|u|}$. The resulting heat loss is $L_{\text{heat}} = \frac{1}{2}\int_\Omega e^{-2\lambda|u|}(\|\nabla u\|^2 + 1) d\mathbf{x}$. Theoretically, it is proven that the distance approximation error converges linearly to zero at a rate of $O(1/\lambda)$.
    - **Design Motivation**: Unlike the eikonal loss, the heat loss provides a bounded distance error even with a finite $\lambda$, and the error vanishes as $\lambda$ increases, fundamentally addressing the lack of a sufficient condition.

2. **$\lambda$ Scheduler (Increasing Strategy for Absorption Coefficient)**:

    - **Function**: Gradually increases the absorption coefficient $\lambda$ during training to balance the optimization of near-field and far-field regions.
    - **Mechanism**: An excessively large $\lambda$ causes $e^{-2\lambda|u|}$ to underflow below floating-point precision limits, resulting in vanishing gradients for the heat loss in regions far from the surface. Therefore, an increasing scheduling strategy is adopted, utilizing a small $\lambda$ first to shape global structures, followed by increasing $\lambda$ to refine distance accuracy. Far from the surface, the optimization is dominated by the eikonal loss.
    - **Design Motivation**: Resolves the numerical instability and the degradation of far-field optimization capability when $\lambda$ is large.

3. **Spatio-Temporal Stability Guarantee**:

    - **Function**: Ensures stable convergence of the optimization process.
    - **Mechanism**: For spatial stability, it is proven that a single-point error in the eikonal equation propagates unboundedly along rays, whereas in the screened Poisson equation, the error decays exponentially as $e^{\lambda(\epsilon - r)}$. For temporal stability, von Neumann analysis proves that the decay rate of all modes in the frequency domain of the heat loss gradient flow is $-(|\omega|^2 + \lambda^2)$, which is always negative, guaranteeing stable convergence.
    - **Design Motivation**: Addresses the backward diffusion divergence issue that arises with the eikonal loss when $\kappa_e < 0$.

### Loss & Training
The total loss is $L = w_b L_{\text{boundary}} + w_e L_{\text{eikonal}} + w_h L_{\text{heat}}$. The network employs an MLP with 5 hidden layers and 128 channels. Importance sampling is used to improve the integration efficiency of the heat loss. Geometric initialization and a linear network structure are used in training.

## Key Experimental Results

### Main Results

| Dataset / Metric | HotSpot | DiGS | StEik | SAL |
|------------|---------|------|-------|-----|
| ShapeNet IoU ↑ | **0.9796** | 0.9636 | 0.9641 | 0.7400 |
| ShapeNet Chamfer ↓ | **0.0029** | 0.0031 | 0.0032 | 0.0074 |
| ShapeNet Hausdorff ↓ | **0.0250** | 0.0435 | 0.0368 | 0.0851 |
| ShapeNet SMAPE ↓ | **0.0540** | 0.2140 | 0.0931 | 0.1344 |
| 2D IoU ↑ | **0.9870** | 0.7882 | 0.6620 | - |
| 2D Chamfer ↓ | **0.0014** | 0.0055 | 0.0073 | - |

### Ablation Study

| Loss Combination | IoU ↑ | Chamfer ↓ | SMAPE ↓ |
|---------|-------|-----------|---------|
| Boundary + Eikonal (IGR) | 0.8192 | 0.0068 | 0.1315 |
| Boundary + SAL | 0.8936 | 0.0029 | 0.1205 |
| Boundary + Eikonal + Area (DiGS) | 0.6338 | 0.0566 | 1.0919 |
| Boundary + Eikonal + Heat (**Ours**) | **0.9851** | **0.0016** | **0.0754** |

### Key Findings
- Heat loss contributes the most to reconstruction accuracy: removing the heat loss drops the IoU from 0.9851 to 0.8192 (keeping only boundary + eikonal), proving that the asymptotically sufficient condition is crucial.
- Area regularization is actually counterproductive: adding area loss and divergence loss sharply degrades performance (IoU drops to 0.3-0.6), verifying the theoretical analysis that area penalties distort the distance field.
- The advantage is particularly pronounced on complex shapes with high genus: while other methods produce redundant boundaries and get trapped in local optima, HotSpot correctly reconstructs the topology.
- It requires the fewest sphere tracing iterations, indicating the highest distance field accuracy.

## Highlights & Insights
- Designing a loss function starting from the mathematical relationship between heat conduction and distance elevates the problem from a necessary condition to an asymptotically sufficient condition. This approach of solving problems from theoretical foundations is highly elegant.
- Using only first-order derivative information (eliminating the need for second-order derivatives) makes it computationally more efficient than methods like DiGS and StEik that require Hessians.
- The weight term $e^{-2\lambda|u|}$ naturally focuses the loss near the surface, achieving targeted optimization of critical regions automatically without requiring extra sampling strategies.

## Limitations & Future Work
- The selection of $\lambda$ relies on empirical tuning and data scale, lacking an adaptive mechanism; a spatially adaptive $\lambda$ might further improve performance.
- Under a large $\lambda$, the gradient of the heat loss vanishes in regions far from the surface, requiring the eikonal loss as a fallback. Their coupling relationship warrants further theoretical analysis.
- Experiments are primarily validated on ShapeNet and SRB, lacking evaluation on large-scale real scanned data.
- Future research can explore incorporating the heat loss into inverse rendering frameworks (e.g., NeuS), combining it with photometric losses to further improve reconstruction quality.

## Related Work & Insights
- **vs DiGS**: Uses divergence loss + area loss for regularization, which remains inherently a necessary condition, and the area loss distorts the distance field. HotSpot approaches from the perspective of sufficient conditions, making it theoretically more sound.
- **vs StEik**: Uses directional divergence loss to stabilize eikonal training but still cannot guarantee convergence to the true distance. HotSpot's spatio-temporal stability analysis proves a stronger convergence guarantee.
- **vs PHASE**: Starts from different mathematical principles but is related in form. PHASE suggests small boundary weights, whereas HotSpot's theoretical analysis indicates that strong boundary conditions are required and avoids unbounded network weight derivatives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Derive asymptotically sufficient condition loss from heat conduction equations; outstanding theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2D/3D datasets represent comprehensive coverage and thorough ablations, but lack verification on large-scale real-world data.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear; excellent figure and chart visualization.
- Value: ⭐⭐⭐⭐ Important theoretical advancement in the field of neural SDF optimization, transferable to downstream tasks like inverse rendering.
---
title: >-
  [论文解读] HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition
description: >-
  [CVPR 2025][人体理解][符号距离函数] 提出 HotSpot，基于 screened Poisson 方程与距离的经典关系设计新的 SDF 优化损失，提供了渐近充分条件保证收敛到真正的距离函数（而非仅满足 Eikonal 的伪解），同时自然惩罚多余表面积，在复杂形状上显著优于 SAL/DiGS/StEik。
tags:
  - CVPR 2025
  - 人体理解
  - 符号距离函数
  - 表面重建
  - Poisson方程
  - Eikonal约束
  - 距离场优化
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](sufficient_invariant_learning_for_distribution_shift.md)
- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](../../AAAI2026/others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](../../NeurIPS2025/others/learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

</div>

<!-- RELATED:END -->
