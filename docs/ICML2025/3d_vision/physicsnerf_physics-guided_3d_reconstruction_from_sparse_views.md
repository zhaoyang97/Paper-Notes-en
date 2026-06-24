---
title: >-
  [Paper Note] PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views
description: >-
  [ICML 2025][3D Vision][NeRF] PhysicsNeRF proposes a physics-prior-based sparse-view NeRF framework. By leveraging four complementary constraints—depth ranking, cross-view consistency, sparsity regularization, and progressive training—it achieves a PSNR of 21.4 dB with only 8 views while providing an in-depth theoretical analysis of the nature of overfitting under sparse-view conditions.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "NeRF"
  - "Sparse Views"
  - "Physics Priors"
  - "3D Reconstruction"
  - "Generalization Analysis"
date: 2026-05-08
content_hash: 1ef32e3705888978
---

# PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views

**Conference**: ICML 2025  
**arXiv**: [2505.23481](https://arxiv.org/abs/2505.23481)  
**Code**: [Available](https://github.com/bmrayan/PhysicsNeRF)  
**Area**: 3D Vision / Neural Radiance Fields  
**Keywords**: NeRF, Sparse Views, Physics Priors, 3D Reconstruction, Generalization Analysis

## TL;DR

PhysicsNeRF proposes a physics-prior-based sparse-view NeRF framework. By leveraging four complementary constraints—depth ranking, cross-view consistency, sparsity regularization, and progressive training—it achieves a PSNR of 21.4 dB with only 8 views while providing an in-depth theoretical analysis of the nature of overfitting under sparse-view conditions.

## Background & Motivation

### Background
Neural Radiance Fields (NeRF) have become the standard methodology for view synthesis, but typically rely on dense views (hundreds of images). Existing sparse-view approaches like RegNeRF, DietNeRF, SparseNeRF, and Instant-NGP improve regularization and encoding strategies, yet they either still require relatively dense views or lack physically-grounded priors. Physics-aware extensions such as PAC-NeRF and PIE-NeRF introduce constraints, but fall short of addressing generalization challenges under extremely sparse views.

### Limitations of Prior Work
Sparse-view reconstruction is a severely underdetermined inverse problem—the limited $N \times K$ pixel-level color constraints cannot uniquely resolve a continuous 3D radiance field, leading to exponentially many 3D solutions consistent with the limited observations. Overfitting is not a mere technical flaw but rather a reflection of this inherent ambiguity.

### Ours
This paper proposes PhysicsNeRF, a compact NeRF variant with only 0.67M parameters. It leverages four complementary physics-based constraints to regularize 3D reconstruction under sparse views, alongside an in-depth theoretical analysis of the nature of the generalization gap.

## Method

### Overall Architecture

PhysicsNeRF employs dual-scale coordinate encoding ($1\times$ and $2\times$ scales), where each branch utilizes a 7-layer MLP (192 hidden units), totaling only 0.67M parameters. Inspired by Instant-NGP and Plenoxels, this design aims to balance model capacity with generalization capability under sparse supervision.

### Key Designs

1. **Depth Ranking Consistency**: Utilizing relative depth supervision provided by monocular depth estimators such as MiDaS, a ranking loss is imposed on selected pixel pairs $(i,j) \in \mathcal{P}$:
$$\mathcal{L}_{\text{depth}} = \sum_{(i,j)\in\mathcal{P}} \ell_{\text{rank}}\big(\text{sgn}(D_M(i)-D_M(j)),\; \text{sgn}(D_P(i)-D_P(j))\big)$$
The Mechanism is to utilize the ordinal relationships (rather than absolute depth values) provided by pretrained depth models to guide geometric learning, thereby avoiding inaccuracies associated with absolute depth estimation.

2. **Cross-View Geometric Consistency**: Constraining rays projected from different camera poses to the same 3D point to yield consistent radiance field outputs:
$$\mathcal{L}_{\text{cv}} = \sum_k \|F_\theta(\mathbf{r}_{k,1}) - F_\theta(\mathbf{r}_{k,2})\|_2^2$$
The Design Motivation is to introduce the geometric consistency principles of multi-view stereo into NeRF training, enhancing cross-view geometric coherence.

3. **Sparsity Regularization**: Natural scenes exhibit spatial sparsity. A volumetric prior is applied to the density field:
$$\mathcal{L}_{\text{sparse}} = \mathbb{E}_{\mathbf{x}\sim\mathcal{U}(\Omega)}[\text{softplus}(\sigma(\mathbf{x}))]$$
Concurrently, a gradient regularization $\mathcal{L}_{\text{reg}} = \|\nabla_{\mathbf{r}} F_\theta(\mathbf{r})\|_2^2$ is incorporated to promote smoothness and prevent excessive local variation.

4. **Progressive Training**: Inspired by curriculum learning, physical constraints are progressively introduced via a scheduling function $\alpha(t)$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rgb}} + \alpha(t)\sum_i \lambda_i \mathcal{L}_i$$
where $\alpha(t)$ is a piecewise constant: 0.008 when $t<5k$, 0.025 when $5k \leq t < 15k$, and 0.08 thereafter.

### Loss & Training

The total loss is a weighted combination of the RGB reconstruction loss and the four physical constraint losses. The model is optimized using the Adam optimizer with an initial learning rate of $5\times10^{-4}$ and an exponential decay factor of $\gamma=0.998$ over 150,000 training iterations, utilizing mixed-precision training and adaptive batch sizes.

## Key Experimental Results

### Main Results

| Dataset/Scene | Metric | PhysicsNeRF | NeRF | RegNeRF | DietNeRF | SparseNeRF |
|------------|------|-------------|------|---------|----------|------------|
| Chair | Train/Test/Gap | 23.2/18.5/4.7 | 16.2/9.1/7.1 | 21.0/12.6/8.4 | 20.4/13.8/6.6 | 21.3/12.9/8.4 |
| Lego | Train/Test/Gap | 21.7/15.0/6.7 | 15.0/8.5/6.5 | 19.8/11.5/8.3 | 19.5/13.0/6.5 | 20.1/11.7/8.4 |
| Drums | Train/Test/Gap | 19.2/12.0/7.2 | 14.4/8.5/5.9 | 19.5/11.3/8.2 | 19.5/12.8/6.7 | 20.1/12.8/8.4 |
| **Average** | Train/Test/Gap | **21.4/15.2/6.2** | - | - | - | - |

### Ablation Study

| Configuration | Train PSNR | Test PSNR | Gap | Explanation |
|------|-----------|-----------|-----|------|
| RGB only | 23.3 | 9.8 | 13.5 | Best training but worst generalization |
| + Depth ranking | 23.0 | 11.2 | 11.8 | Gap reduced by 1.7 dB |
| + Cross-view | 22.7 | 12.8 | 9.9 | Further narrows the Gap |
| + Sparsity | 22.4 | 13.9 | 8.5 | Approaching final performance |
| + All (Full) | 21.7 | 15.0 | 6.7 | Optimal generalization |

### Key Findings

1. **Collapse-Recovery Dynamics**: A consistent PSNR collapse-recovery pattern is observed during training at approximately 20k iterations, precisely corresponding to the activation times of the progressive constraints.
2. **Generalization Gap Correlates Positively with Complexity**: As the geometric complexity of the scene increases, the generalization gap increases from 4.7 → 6.7 → 7.2 dB.
3. **Overfitting is an Inherent Feature of Sparse-View Reconstruction**: Theoretical analysis demonstrates that the structural magnitude of the generalization gap is $O(\sqrt{|\theta|/N})$.

## Highlights & Insights

- Deep theoretical analysis of the nature of overfitting in sparse-view reconstruction, casting overfitting as a structural property rather than an implementation flaw.
- A compact design with only 0.67M parameters, demonstrating that physical priors are more critical than model scale.
- The discovery of collapse-recovery dynamics reveals the underlying mechanism of physical constraints in shaping the optimization landscape.
- Insights for world model construction: physically consistent representation under limited observations remains an open problem.

## Limitations & Future Work

- The generalization gap remains at 5.7-6.2 dB, indicating that fixed-form physical constraints struggle to fully resolve the underdetermined nature of the problem.
- Experiments are limited to the NeRF synthetic dataset, lacking validation on real-world scenes.
- Future directions include: learnable adaptive physical constraints, multi-modal information fusion (semantics + geometry + temporal), and hierarchical scene decomposition.
- Lack of comparison with more advanced 3DGS-based or diffusion-based methods.

## Related Work & Insights

- Belongs to the line of sparse-view NeRF work alongside RegNeRF, DietNeRF, and SparseNeRF, but places a heavier emphasis on physical priors.
- Borrows from the PINN paradigm of incorporating physical constraints into neural networks.
- The collapse-recovery dynamics are similar to the phase transition studies of loss landscapes in training.

## Rating
- Novelty: ⭐⭐⭐⭐ The physical constraints themselves are not entirely novel, but the theoretical analysis of overfitting behavior provides a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments are conducted only on the NeRF synthetic dataset (3 scenes), indicating a limited evaluation scale.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical analysis is clear and in-depth, with a complete structure.
- Value: ⭐⭐⭐⭐ The theoretical insights are valuable, though the practical application scenarios remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](../../CVPR2025/3d_vision/mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](../../ICCV2025/3d_vision/reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](../../ICCV2025/3d_vision/no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[CVPR 2026\] GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views](../../CVPR2026/3d_vision/gifsplat_generative_prior-guided_iterative_feed-forward_3d_gaussian_splatting_fr.md)
- [\[CVPR 2025\] PhysAnimator: Physics-Guided Generative Cartoon Animation](../../CVPR2025/3d_vision/physanimator_physics-guided_generative_cartoon_animation.md)

</div>

<!-- RELATED:END -->
