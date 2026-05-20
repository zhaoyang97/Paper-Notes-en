---
title: >-
  [Paper Note] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks
description: >-
  [AAAI 2026][Video Understanding][Grassmannian] This paper proposes GMSF-Net, a topology-driven multi-subspace fusion network on the Grassmann manifold. By introducing adaptive multi-subspace construction and a Fréchet me…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Grassmannian"
  - "manifold learning"
  - "subspace fusion"
  - "3D action recognition"
  - "Riemannian neural networks"
date: 2026-05-08
content_hash: 43872567c34e00d7
---

# Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks

**Conference**: AAAI 2026
**arXiv**: [2511.08628](https://arxiv.org/abs/2511.08628)  
**Code**: [GitHub](https://github.com/Xua-Yu/GMSF-Net)  
**Area**: Video Understanding / Manifold Learning
**Keywords**: Grassmannian, manifold learning, subspace fusion, 3D action recognition, Riemannian neural networks

## TL;DR

This paper proposes GMSF-Net, a topology-driven multi-subspace fusion network on the Grassmann manifold. By introducing adaptive multi-subspace construction and a Fréchet mean-based subspace interaction mechanism, it successfully transfers the multi-channel interaction paradigm from Euclidean space to non-Euclidean geometry, achieving state-of-the-art performance on 3D action recognition, EEG classification, and graph tasks.

## Background & Motivation

### Core Problem

The Grassmann manifold is a powerful tool for geometric representation learning, enabling high-dimensional data to be modeled as low-dimensional subspaces. However, existing methods suffer from two critical limitations:

**Static single-subspace representation**: Methods such as GrNet and GDLNet model input data using a single fixed orthogonal subspace, making it difficult to capture local geometric variations and multimodal distributions, thereby limiting representational capacity.

**Lack of subspace interaction**: Much of the success of deep learning in Euclidean space is attributable to multi-channel interactions and nonlinear activations (e.g., multi-channel convolutions in LeNet-5), yet multi-subspace interaction on the Grassmann manifold has been largely overlooked.

### Key Challenges

- How to perform effective subspace interaction on the Grassmann manifold?
- How to design stackable deep architectures to expand model capacity?
- How to guarantee optimization convergence on Riemannian manifolds?

## Method

### Overall Architecture: GMSF-Net

GMSF-Net consists of three core modules: the Adaptive Multi-Subspace Encoder (AMSE), the Multi-Subspace Interaction Block, and Riemannian Batch Normalization.

### 1. Adaptive Multi-Subspace Construction (AdaMSC)

This is the core of the AMSE, inspired by the Kolmogorov–Arnold representation theorem. The procedure is as follows:

1. **Extract frame-level features** and compute the covariance matrix $X \in \mathbb{R}^{n \times n}$, modeling statistical dependencies of features along the temporal dimension.
2. **Schmidt orthogonalization**: Map $X$ to a set of low-dimensional orthogonal subspaces $\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$, where $S_i^\top S_j = 0$ ($i \neq j$) and each $S_j \in \mathcal{G}(n,1)$.
3. **Learnable weight selection**: Initialize learnable weights $\mathcal{W}^{(m')}$ for each new subspace $S'_{m'}$, apply Softmax normalization, and select the top-$p$ most important atomic subspaces.
4. **Weighted combination**: $S'_{m'} = [\tilde{w}_{j_1}^{(m')} S_{j_1}, \tilde{w}_{j_2}^{(m')} S_{j_2}, \ldots, \tilde{w}_{j_p}^{(m')} S_{j_p}]$

This design allows each new subspace to be composed of distinct key atoms, enabling adaptive adjustment to different tasks.

### 2. Topology-Driven Convergence Analysis

The paper rigorously proves the convergence of multi-subspace construction from a topological perspective:

- The projection metric $d_p(X_1, X_2) = 2^{-1/2} \|X_1 X_1^T - X_2 X_2^T\|_F$ is used to define distances on the Grassmann manifold.
- Subspaces are iteratively updated via Riemannian gradient descent.
- It is proved that under the topology induced by the projection metric, the subspace sequence converges to a stable subspace $S^*$: $d(S'(t), S^*) \to 0$.

### 3. Multi-Subspace Interaction Block

This block comprises two sub-components:

**Grassmann Multi-Subspace Representation (GMSR)**: A shared learnable mapping matrix $W_c$ (constrained to the Stiefel manifold) is applied to all input subspaces, generating representations under different geometric frames: $X_{GMSR}^{c,m'} = W_c^T S'_{m'}$. QR or SVD decomposition is subsequently applied to maintain orthogonality.

**Grassmann Subspace Interaction (GSI)**: Multiple subspaces are fused using the Fréchet mean:

- When $m'=2$, a closed-form solution exists (geodesic interpolation).
- When $m'>2$, the Karcher flow algorithm is used to iteratively optimize in the tangent space.

### 4. Optimization Strategy

- **Riemannian Batch Normalization**: Maps statistical features to the SPD manifold and normalizes them to enhance discriminability.
- **Mutual Information Regularization**: Maximizes information complementarity across different subspaces.
- **Total Loss**: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_C$

## Key Experimental Results

### Main Results I: 3D Action Recognition (FPHA Dataset)

| Method | Accuracy (%) | Model Size (MB) | FLOPs (M) |
|--------|-------------|-----------------|-----------|
| GrNet | 78.79±1.82 | 6.73 | 38.60 |
| SPDNet | 87.65±1.02 | 13.60 | 1595.50 |
| MATT | 87.70±0.68 | 1.83 | 142.07 |
| SPDNetBN | 89.33±0.49 | 13.63 | 1902.97 |
| GDLNet | 87.60±0.69 | 1.83 | 33.69 |
| **GMSF-Net-1Block** | **90.43±0.74** | **1.20** | 48.42 |
| **GMSF-Net-3Blocks** | **91.22±0.53** | **1.30** | 81.07 |

GMSF-Net-3Blocks outperforms GrNet by 12.43% while using a smaller model (1.30 MB vs. 6.73 MB).

### Main Results II: EEG Signal Classification (MAMEM-SSVEP-II Dataset)

| Method | Accuracy (%) | Model Size (MB) |
|--------|-------------|-----------------|
| EEGNet | 53.72±7.23 | 0.075 |
| SCCNet | 62.11±7.70 | 0.55 |
| SPDNet | 62.30±3.12 | 2.81 |
| GrNet | 61.23±3.56 | 1.95 |
| MATT | 65.19±3.14 | 1.97 |
| GDLNet | 65.52±2.86 | 1.95 |
| **GMSF-Net-3Blocks** | **66.87±1.46** | **1.94** |

GMSF-Net surpasses GrNet by 5.64% and GDLNet by 1.35%, with a notably smaller standard deviation (1.46 vs. 2.86), indicating significantly improved stability.

### Ablation Study

| Configuration | HDM05 | FPHA | SSVEP |
|---------------|-------|------|-------|
| Adaptive subspace + interaction | **63.64%** | **90.43%** | **66.74%** |
| Adaptive subspace (no interaction) | 56.49% | 80.68% | 59.83% |
| Random subspace + interaction | 50.29% | 72.47% | 56.01% |
| Fixed subspace + interaction | 53.04% | 83.06% | 66.05% |

The ablation results clearly demonstrate: (1) the adaptive mechanism substantially outperforms random and fixed subspace alternatives; (2) the interaction mechanism is effective only on high-quality subspaces—applying it to random subspaces instead introduces noise.

## Highlights & Insights

1. **First introduction of deep Grassmann subspace interaction in Riemannian neural networks**, successfully transferring the multi-channel interaction philosophy from Euclidean to non-Euclidean geometry.
2. **Topology-driven theoretical guarantees**: The convergence of adaptive subspace construction is rigorously proved under the projection metric topology, providing a solid theoretical foundation.
3. **Significant reduction in model overhead**: A 1.30 MB model surpasses SPDNetBN (13.63 MB) on FPHA, demonstrating a clear efficiency advantage.
4. **Learnable subspace selection mechanism** (analogous to KAN) makes subspace construction fully data-driven.

## Limitations & Future Work

1. **Task scope**: Validation is primarily conducted on small-to-medium-scale datasets (HDM05, FPHA, SSVEP); performance on large-scale video understanding tasks remains untested.
2. **Grassmannian assumption dependency**: Performance gains are limited on datasets that do not conform to an ideal subspace structure (e.g., PubMed), indicating strong geometric assumptions about the data.
3. **Computational complexity**: The cost of computing the Fréchet mean via Karcher flow iteration grows with the number of subspaces and their dimensionality.
4. **Block saturation**: Performance gains diminish progressively from 1 Block to 3 Blocks, suggesting limited depth-scalability potential of the stackable architecture.

## Related Work & Insights

- **GrNet** (Huang et al., 2018): The first deep network on the Grassmann manifold, introducing layers such as FRMap, OrthMap, and ProjMap, but restricted to a single static subspace.
- **GDLNet** (Wang et al., 2024): Introduces self-attention on the Grassmann manifold to capture inter-subspace dependencies, yet still constrained by single-subspace modeling.
- **SPDNet/SPDNetBN**: Deep networks based on the SPD manifold; relatively large models with high computational cost.
- **MATT** (Pan et al., 2022): A manifold attention-based method.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Overall Rating**: ⭐⭐⭐⭐ (4/5)

> Theoretically rigorous with clearly articulated contributions, the paper successfully transfers multi-channel interaction to the Grassmann manifold. However, the application scenarios remain primarily academic, and large-scale practical utility has yet to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](../../CVPR2026/video_understanding/spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](../../CVPR2026/video_understanding/learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](../../ICLR2026/video_understanding/nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)

</div>

<!-- RELATED:END -->
