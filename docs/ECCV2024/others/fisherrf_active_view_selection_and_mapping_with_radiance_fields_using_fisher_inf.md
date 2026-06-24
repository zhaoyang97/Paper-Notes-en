---
title: >-
  [Paper Note] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information
description: >-
  [ECCV 2024][Fisher Information] This paper proposes FisherRF, which utilizes Fisher Information to directly quantify the observed information of radiance fields model parameters. By maximizing the Expected Information Gain (EIG), it selects the optimal view. The proposed method achieves state-of-the-art (SOTA) performance across three tasks: view selection, active mapping, and uncertainty quantification. Furthermore, by leveraging sparsity and custom CUDA kernels…
tags:
  - "ECCV 2024"
  - "Fisher Information"
  - "Active View Selection"
  - "3D Gaussian Splatting"
  - "Uncertainty Quantification"
  - "Active Mapping"
date: 2026-05-08
content_hash: 69cb1f3233a9077a
---

# FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information

**Conference**: ECCV 2024  
**arXiv**: [2311.17874](https://arxiv.org/abs/2311.17874)  
**Code**: None  
**Area**: Others  
**Keywords**: Fisher Information, Active View Selection, 3D Gaussian Splatting, Uncertainty Quantification, Active Mapping

## TL;DR

This paper proposes FisherRF, which utilizes Fisher Information to directly quantify the observed information of radiance fields model parameters. By maximizing the Expected Information Gain (EIG), it selects the optimal view. The proposed method achieves state-of-the-art (SOTA) performance across three tasks: view selection, active mapping, and uncertainty quantification. Furthermore, by leveraging sparsity and custom CUDA kernels, it achieves a view evaluation speed of 70 fps.

## Background & Motivation

1. **Background**: Technologies such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have significantly advanced image rendering and 3D reconstruction. However, training high-quality radiance field models requires a large number of images from diverse viewpoints, and image acquisition remains costly. Thus, efficient selection of the most informative viewpoints for acquisition is a key problem. Existing active view selection methods are mainly divided into two categories: white-box methods (modifying model architectures to embed Bayesian models for uncertainty approximation) and black-box methods (indirectly evaluating uncertainty via predicted distributions).

2. **Limitations of Prior Work**: White-box methods depend on specific model architectures and suffer from slow training speeds. Black-box methods (such as ActiveNeRF via adding variance outputs, and BayesRays via assuming perturbation fields) can only indirectly approximate model uncertainty; selecting views based on indirect approximations does not guarantee optimal information gain for the model. Additionally, these methods are incompatible with newer radiance field representations like 3D Gaussian Splatting, as the latter do not use query points as input.

3. **Key Challenge**: The parameter space of radiance field models is enormous (typically exceeding 200 million optimizable parameters), making the direct computation of the full Fisher Information Matrix or Hessian matrix intractable. Quantifying the information gain computably while maintaining theoretical rigor is the key technical challenge.

4. **Goal**: Provide a theoretically-grounded and highly efficient method to quantify observed information in radiance fields and select the optimal viewpoints. The method should be compatible with multiple radiance field representations (NeRF, 3DGS, Plenoxels) and extensible to batch view selection, active mapping, and pixel-level uncertainty quantification.

5. **Key Insight**: Fisher Information is fundamentally the Hessian matrix of the log-likelihood function with respect to model parameters. In volume rendering, this Hessian does not depend on ground-truth observations (ground truth images); it only requires the camera parameters of candidate viewpoints for computation. This implies that the information gain of each candidate view can be evaluated without actually capturing the image. Furthermore, due to the local parameter structure of radiance field models, the Hessian matrix is highly sparse, allowing for highly efficient computation by leveraging this sparsity.

6. **Core Idea**: Directly measure the information gain of radiance field parameters via Fisher Information to select the next best viewpoint, without modifying the model architecture, relying on indirect uncertainty approximations, and maintaining computational efficiency.

## Method

### Overall Architecture

The core pipeline of FisherRF: (1) train the radiance field model with a small initial set of viewpoints to obtain parameters $\mathbf{w}^*$; (2) calculate the Expected Information Gain (EIG) relative to the training set for each candidate viewpoint using Fisher Information; (3) select the candidate viewpoint with the highest EIG as the next capture position; (4) add the ground-truth image of this viewpoint to the training set and retrain the model; (5) repeat the process until the target number of viewpoints is met. This framework is implemented on both 3D Gaussian Splatting and Plenoxels.

### Key Designs

1. **Fisher Information Derivation in Volume Rendering**: Core theoretical contribution. In radiance fields, the negative log-likelihood is $-\log p(\mathbf{y}|\mathbf{x},\mathbf{w}) = (\mathbf{y}-f(\mathbf{x},\mathbf{w}))^T(\mathbf{y}-f(\mathbf{x},\mathbf{w}))$, and its Fisher Information (the Hessian matrix) is $\mathbf{H}'' = \nabla_\mathbf{w}f^T \nabla_\mathbf{w}f$. The key insight is that this Hessian depends only on the Jacobian of the rendering function, not the ground-truth image $\mathbf{y}$—enabling the information content of each candidate view to be evaluated without any real capture. The optimization objective for Expected Information Gain is $\arg\max_{\mathbf{x}_i^{acq}} \text{tr}(\mathbf{H}''[\mathbf{y}_i|\mathbf{x}_i,\mathbf{w}^*] \cdot \mathbf{H}''[\mathbf{w}^*|\mathcal{D}_{train}]^{-1})$. Due to the enormous parameter space, a Laplace approximation is used to diagonalize the Hessian matrix: $\mathbf{H}'' \approx \text{diag}(\nabla_\mathbf{w}f^T\nabla_\mathbf{w}f) + \lambda I$.

2. **Greedy Algorithm for Batch View Selection**: In practice, selecting multiple views simultaneously is often required. Simply maximizing EIG independently for each view leads to selecting similar views (information redundancy). Leveraging the additivity of Fisher Information, FisherRF designs a greedy optimization algorithm: after choosing each optimal view, its Hessian is added to the existing training Hessian before selecting the next one. Since each parameter is only affected by rays within a localized spatial region, the Hessian matrix is highly sparse, allowing matrix product computation to be as efficient as backpropagation. The diagonal Hessian calculation implemented via a custom CUDA kernel takes only 11.3ms, which is about 100x faster compared to the 1.1s required by the traditional PyTorch backward engine.

3. **Pixel-Level Uncertainty Quantification**: FisherRF can also derive pixel-level uncertainty metrics. In 3D Gaussian Splatting, each parameter directly corresponds to a position in 3D space. For each rendered pixel, the Fisher Information of the 3D Gaussian parameters contributing to the color along the ray direction can be aggregated via the volume rendering equation: $\mathbf{U}(\mathbf{r}) = \sum_{n=1}^{N_s} T_i(1-\exp(-\sigma_n\delta_n))\text{tr}(\mathbf{G}_n)$, where $\mathbf{G}_n$ is the Hessian submatrix of the parameters associated with the $n$-th sample point. Regions with low Fisher Information correspond to areas of high uncertainty—indicating regions where the model has captured less information.

### Loss & Training

- Radiance field training uses a standard L2 rendering loss (MSE between rendered and ground-truth images).
- View selection is performed at intervals during training iterations (e.g., once every 100 epochs).
- 3DGS training follows the original configuration, with opacity reset after each view addition to prevent degradation.
- Initial training uses 4 uniformly distributed views, which are then incrementally increased to 20 views.
- The active mapping system is built upon the SplaTAM framework, utilizing frontier-based exploration to generate candidate trajectories.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Blender (20 views) | 3DGS + Random | 28.73 | 0.939 | 0.053 |
| Blender (20 views) | 3DGS + ActiveNeRF | 26.61 | 0.905 | 0.081 |
| Blender (20 views) | **3DGS + Ours** | **29.53** | **0.944** | **0.043** |
| Blender (10 views) | 3DGS + Random | 22.49 | 0.873 | 0.112 |
| Blender (10 views) | **3DGS + Ours** | **23.68** | **0.883** | **0.102** |
| Mip360 (seq) | 3DGS + Random | 17.91 | 0.564 | 0.430 |
| Mip360 (seq) | 3DGS + ActiveNeRF | 17.89 | 0.533 | 0.414 |
| Mip360 (seq) | **3DGS + Ours** | **20.35** | **0.601** | **0.361** |

### Active Mapping Results

| Method | Gibson Comp.(%)↑ | Gibson Comp.(cm)↓ | MP3D Comp.(%)↑ | MP3D Comp.(cm)↓ |
|------|------------------|-------------------|----------------|-----------------|
| FBE | 68.91 | 14.42 | 71.18 | 9.78 |
| Active Neural Mapping | 80.45 | 7.44 | 73.15 | 9.11 |
| **Ours** | **92.89** | **5.64** | **89.41** | **2.91** |

### Uncertainty Quantification

| Method | AUSE (LF Dataset, avg)↓ |
|------|------------------------|
| CF-NeRF | 0.38 |
| ActiveNeRF | 0.33 |
| BayesRays | 0.23 |
| **Ours** | **0.22** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Random Selection | Baseline PSNR | Fisher Information yields significant improvement |
| ActiveNeRF (Learned Variance) | Below Random | Indirect uncertainty approximation is unreliable |
| BayesRays (Perturbation Field) | Slightly better than Random | Uncertainty $\neq$ information gain |
| FisherRF (EIG) | Optimal | Clear advantage of directly optimizing information gain |
| Plenoxels Backend | Equally effective | Framework generalizes across different radiance field models |

### Key Findings

- **EIG vs. Uncertainty**: Selecting views using Expected Information Gain (EIG) significantly outperforms using uncertainty alone. Simply selecting the most uncertain views does not guarantee the maximum global information gain for the model parameters.
- **ActiveNeRF < Random**: On the 3DGS backend, ActiveNeRF performs worse than random selection, demonstrating that modifying model architectures to estimate uncertainty is unreliable for new representations.
- **Effectiveness of Batch Selection**: The greedy batch selection algorithm effectively avoids informational redundancy.
- **Extremely Sparse Scenarios**: With only 10 training views, the advantage of FisherRF becomes even more pronounced.
- **Active Mapping Gains**: Scene completeness increases significantly, from 80.45% to 92.89% (Gibson) and from 73.15% to 89.41% (MP3D).

## Highlights & Insights

1. **Theoretical Elegance**: Deriving the view selection objective from the first principles of information theory, where Fisher Information provides a concrete mathematical framework. The property that the Hessian does not depend on ground-truth images makes "selection without acquisition" theoretically self-consistent.
2. **Sparsity Insights**: The discovery that the local parameter structure of radiance fields leads to highly sparse Hessians makes Fisher Information computation feasible for 200M+ parameters. The custom CUDA kernels achieve a 100x speedup.
3. **Unified Multi-task Framework**: View selection, batch selection, path planning, and pixel-level uncertainty quantification are all naturally derived from the same Fisher Information framework.
4. **Strong Experimental Results**: Outperforms the SOTA across three different tasks and five datasets by a significant margin.

## Limitations & Future Work

1. Restricted to static scenes; Fisher Information quantification in dynamic radiance fields (e.g., D-NeRF, 4D Gaussian Splatting) remains an open challenge.
2. The diagonal Laplace approximation discards correlation information between parameters. More precise block-diagonal or low-rank approximations could further improve performance.
3. Active mapping relies on frontier-based exploration to provide candidate trajectories; more advanced path-planning strategies or reinforcement learning methods may bring additional benefits.
4. Pixel-level uncertainty is relative (based on observation information) and not an absolute metric, which limits its utility in applications requiring calibrated uncertainty.
5. When the number of training views is extremely small (<5), the initial model quality may be too low, leading to inaccurate Fisher Information estimation.

## Related Work & Insights

- **ActiveNeRF (Pan et al., ECCV 2022)**: Introduces a variance-output head in NeRF for active view selection; a direct competitor.
- **BayesRays (Goli et al., 2023)**: Quantifies NeRF uncertainty by assuming spatial perturbation fields; inapplicable to 3DGS.
- **3D Gaussian Splatting (Kerbl et al., 2023)**: FisherRF is implemented on top of this, exploiting its explicit parameterization to efficiently compute Fisher Information.
- **SplaTAM (Keetha et al., 2023)**: A 3D-Gaussian-based SLAM system upon which FisherRF constructs its active mapping system.
- **Kirsch & Gal, 2022**: Unifies various deep active learning methods from the perspective of Fisher Information, providing the theoretical inspiration for FisherRF.
- **Insight**: Fisher Information as a parameter-level information metric holds great potential to play a larger role in other vision tasks requiring active data acquisition (e.g., active SLAM, robotic exploration).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to directly apply Fisher Information to parameter-level information quantification in radiance fields, offering an elegant and complete theoretical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive evaluation across three tasks, five datasets, and multiple radiance field backends, complete with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations with structurally logical sequencing from motivation to methodology and experiments.
- **Value**: ⭐⭐⭐⭐⭐ Lays down both theoretical foundations and efficient implementations for active learning in radiance fields; the multi-task framework possesses substantial potential for broader applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ICML 2025\] Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](../../ICML2025/others/fishers_for_free_approximating_the_fisher_information_matrix_by_recycling_the_sq.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](../../CVPR2025/others/which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)

</div>

<!-- RELATED:END -->
