---
title: >-
  [Paper Note] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis
description: >-
  [ICLR 2026][Autonomous Driving][LiDAR novel view synthesis] SG-NLF proposes a pose-free LiDAR NeRF framework that integrates spectral information with geometric consistency. It leverages a hybrid spectral-geometric repre…
tags:
  - "ICLR 2026"
  - "Autonomous Driving"
  - "LiDAR novel view synthesis"
  - "pose-free NeRF"
  - "spectral embedding"
  - "geometric consistency"
  - "adversarial learning"
date: 2026-05-08
content_hash: 1889a4c3c738cdad
---

# Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis

**Conference**: ICLR 2026
**arXiv**: [2603.12903](https://arxiv.org/abs/2603.12903)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: LiDAR novel view synthesis, pose-free NeRF, spectral embedding, geometric consistency, adversarial learning

## TL;DR
SG-NLF proposes a pose-free LiDAR NeRF framework that integrates spectral information with geometric consistency. It leverages a hybrid spectral-geometric representation for continuous smooth geometry reconstruction, a confidence-aware pose graph for global pose optimization, and an adversarial learning strategy to enforce cross-frame consistency, achieving improvements of 35.8% in reconstruction quality and 68.8% in pose accuracy over the previous state of the art.

## Background & Motivation
1. **Background**: Novel view synthesis (NVS) is a critical task in 3D perception, with broad applications in scene understanding and autonomous driving. NeRF-based methods have recently been successfully extended to LiDAR NVS, surpassing conventional simulation approaches.
2. **Limitations of Prior Work**:
    - Most LiDAR NVS methods rely on accurate camera poses, which are often unavailable in real-world scenarios.
    - Both pose-dependent and pose-free methods employ geometric interpolation (e.g., multi-resolution hash encoding) for neural field rendering. Due to the sparsity and irregularity of LiDAR data, interpolated features fail to reconstruct continuous surfaces, leading to geometric inconsistencies (geometric holes) in textureless regions.
    - The only existing pose-free method, GeoNLF, relies on pairwise alignment and cannot guarantee global pose accuracy.
3. **Key Challenge**: The inherent sparsity and lack of texture in LiDAR data prevent interpolation-based representations from reconstructing smooth, continuous surfaces. Larger inter-frame motion and reduced overlap in low-frequency LiDAR sequences further degrade multi-view consistency.
4. **Goal**: Simultaneously achieve high-quality LiDAR novel view synthesis and accurate pose estimation, particularly in challenging low-frequency scenarios.
5. **Key Insight**: Introduce spectral embedding to provide global structural priors that compensate for the limitations of local geometric interpolation.
6. **Core Idea**: Learn the eigenfunctions of the Laplace-Beltrami operator in a differentiable manner and fuse them as spectral embeddings combined with geometric encodings to form a hybrid representation; construct a confidence-aware pose graph based on feature compatibility for global pose optimization.

## Method

### Overall Architecture
Multi-view LiDAR sequence $\{\mathcal{S}_i\}_{i=0}^{N}$ → feature extraction via hybrid spectral-geometric representation → global pose optimization via confidence-aware pose graph → NeRF volume rendering to synthesize novel views $\hat{\mathcal{S}}$ → adversarial learning to enforce cross-frame consistency.

### Key Designs
1. **Hybrid Spectral-Geometric Representation**:
    - **Geometric Encoding**: Multi-resolution hash grid encoding $\boldsymbol{f}_{\text{geo}}(\mathbf{x})$ captures local structure and high-frequency details.
    - **Spectral Embedding**: The first $K$ eigenfunctions of the Laplace-Beltrami operator are learned as $\boldsymbol{f}_{\text{spe}}(\mathbf{x}) = [\Psi_0(\mathbf{x}), \ldots, \Psi_K(\mathbf{x})]^\top$, endowing the representation with intrinsic isometric invariance and surface-awareness.
    - Eigenfunctions are optimized in a differentiable manner by minimizing the Rayleigh quotient $\mathcal{R}_\Sigma(\Psi_i) = \frac{\sum_j \|\nabla_\Sigma \Psi_i(\hat{\mathbf{x}}_j)\|^2 dA_j}{\sum_j \Psi_i^2(\hat{\mathbf{x}}_j) dA_j}$.
    - Orthogonality $\mathcal{L}_{\text{ortho}}$ and normalization $\mathcal{L}_{\text{norm}}$ constraints are imposed to ensure the learned eigenfunctions form a valid orthonormal basis.
    - The two encodings are progressively fused during training into a hybrid representation $\boldsymbol{f}_{\text{hyb}}(\mathbf{x})$.
    - **Design Motivation**: Spectral embeddings provide global smoothness priors, while geometric encodings supply high-frequency details; their complementary fusion addresses geometric holes arising from sparse LiDAR data.

2. **Confidence-aware Global Pose Optimization**:
    - A pose graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is constructed, where vertices represent LiDAR frames and poses, and edges represent inter-frame constraints.
    - The edge set includes not only temporally adjacent edges but also non-adjacent edges with high feature compatibility.
    - Feature correspondences are established via a coarse-to-fine mutual nearest neighbor (MNN) strategy.
    - Edge compatibility score $E^{ij}$ is computed as the mean cosine similarity of feature pairs in the refined correspondence set.
    - Each edge weight $\alpha^{ij}$ is derived from the spatial consistency $P_{mn}$ of the correspondences.
    - Pose graph loss: $\mathcal{L}_{\text{graph}} = \sum_{(i,j) \in \mathcal{E}} \alpha^{ij} \cdot \mathcal{L}_{\text{cd}}^{ij}$.
    - **Design Motivation**: Pairwise alignment alone cannot guarantee global accuracy; global graph optimization with adaptive edge weights effectively suppresses the influence of inaccurate alignments.

3. **Cross-frame Consistency via Adversarial Learning**:
    - "Real" depth map pairs $\mathbf{I}_{\text{real}} = [D_{ij}, D_j]$ (projections of ground-truth point clouds after transformation) and "fake" pairs $\mathbf{I}_{\text{fake}} = [\hat{D}_{ij}, D_j]$ (projections of synthesized point clouds after transformation) are constructed.
    - A multi-scale PatchGAN discriminator detects geometric misalignment at both global and local levels.
    - Hinge loss: $\mathcal{L}_{\text{con}} = \max(0, 1 - \mathbf{\Phi}(\mathbf{I}_{\text{real}})) + \max(0, 1 + \mathbf{\Phi}(\mathbf{I}_{\text{fake}}))$.
    - **Design Motivation**: Pixel-level supervision only penalizes single-frame photometric error, neglecting structural information; adversarial learning jointly evaluates reconstruction quality and pose accuracy within a unified discriminative framework.

### Loss & Training
- The overall training objective combines the consistency loss, range image loss, and spectral loss.
- Spectral loss: $\mathcal{L}_{\text{spe}} = \sum_i \mathcal{R}_\Sigma(\Psi_i) + \lambda_n \mathcal{L}_{\text{norm}} + \lambda_o \mathcal{L}_{\text{ortho}}$.
- Poses are parameterized and optimized in Lie algebra space; the Jacobian $\boldsymbol{J}$ is omitted for more stable convergence.
- Training runs for 60K iterations with a batch size of 4096 rays, using the Adam optimizer with a linearly decaying learning rate of 0.01.
- Training is performed on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results

| Dataset / Method | CD↓ | F-score↑ | Depth PSNR↑ | Intensity PSNR↑ | Notes |
|---|---|---|---|---|---|
| **KITTI-360 Low-freq.** | | | | | |
| LiDAR4D (GT pose) | 0.2760 | 0.8843 | 24.73 | 16.95 | pose-dependent |
| GeoNLF (pose-free) | 0.2363 | 0.9178 | 25.28 | 16.58 | Prev. SOTA |
| **SG-NLF (Ours)** | **0.1695** | **0.9191** | **28.71** | **19.27** | CD ↓28.3% |
| **nuScenes Low-freq.** | | | | | |
| GeoNLF | 0.2408 | 0.8647 | 22.95 | 28.61 | Prev. SOTA |
| **SG-NLF (Ours)** | **0.1545** | **0.9097** | **28.41** | **30.50** | CD ↓35.8% |
| **Pose Estimation ATE(m)↓** | | | | | |
| GeoNLF (KITTI-360) | 0.170 | - | - | - | |
| **SG-NLF (KITTI-360)** | **0.074** | - | - | - | ↓56.4% |
| GeoNLF (nuScenes) | 0.228 | - | - | - | |
| **SG-NLF (nuScenes)** | **0.071** | - | - | - | ↓68.8% |

### Ablation Study

| Configuration | CD↓ | Depth PSNR↑ | Intensity PSNR↑ | ATE(m)↓ | Notes |
|---|---|---|---|---|---|
| Baseline (no components) | 0.618 | 21.32 | 25.86 | 1.328 | Same baseline as GeoNLF |
| w/o Hybrid Repr. (HR) | 0.217 | 25.10 | 28.43 | 0.204 | Geometric encoding only |
| w/o Global Pose Optim. (GP) | 0.463 | 23.94 | 27.55 | 0.798 | No graph optimization |
| w/o Cross-frame Consist. (CFC) | 0.182 | 26.60 | 29.30 | 0.076 | No adversarial learning |
| **Full SG-NLF** | **0.155** | **28.41** | **30.50** | **0.071** | All components |
| Spectral emb. only (w/o GE) | 0.181 | 26.85 | 29.03 | - | Smooth but lacks high-freq. detail |

### Key Findings
- The global structural prior provided by spectral embedding is essential for resolving geometric inconsistencies in sparse LiDAR data.
- SG-NLF outperforms LiDAR4D (which uses GT poses) even without ground-truth poses (CD: 0.1695 vs. 0.2760).
- All three core components—hybrid representation, global pose optimization, and cross-frame consistency—contribute substantially to the final performance.
- State-of-the-art results are also achieved in standard-frequency scenarios, demonstrating strong generalization.
- Cross-frame consistency effectively regularizes training even in the absence of pose optimization.

## Highlights & Insights
- This work is the first to introduce spectral methods (LBO eigenfunctions) into LiDAR NeRF, offering a novel perspective for continuous surface reconstruction from sparse point clouds.
- The hybrid representation elegantly balances low-frequency global structure and high-frequency local detail, providing a principled solution to LiDAR sparsity.
- The confidence-aware pose graph design comprehensively outperforms naive pairwise alignment approaches.
- The adversarial learning strategy elegantly unifies reconstruction quality and pose accuracy within a single discriminative framework.
- Performance is particularly notable in low-frequency scenarios characterized by large inter-frame motion and limited overlap.

## Limitations & Future Work
- Only a single implementation of SG-NLF is presented; exploring additional technical variants for different application scenarios remains open.
- Training requires 60K iterations; inference speed, while improved, still has room for further optimization.
- Validation is primarily conducted on KITTI-360 and nuScenes; generalization to a broader range of driving environments warrants further investigation.
- The effect of the number of spectral embedding eigenfunctions $K$ on performance merits more systematic study.
- Extension to dynamic scenes is a potential direction, as the current method primarily handles static environments.

## Related Work & Insights
- GeoNLF is the direct predecessor in the pose-free paradigm; SG-NLF substantially surpasses it through global graph optimization and the hybrid representation.
- Dynamic scene modeling in LiDAR4D and STGC could be integrated with the proposed method.
- Neural spectral methods (SNS) provide the technical foundation for learning LBO eigenfunctions.
- Spectral embeddings have been widely used in 3D shape analysis; this work is the first to introduce them into the NeRF framework.
- The proposed approach is generalizable to other sparse data reconstruction tasks that require global structural priors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)
- [\[ICLR 2026\] Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](multi-head_low-rank_attention.md)

</div>

<!-- RELATED:END -->
