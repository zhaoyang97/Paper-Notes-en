---
title: >-
  [Paper Note] Hyperbolic Busemann Neural Networks
description: >-
  [CVPR 2026][Graph Learning][Hyperbolic neural networks] This paper intrinsically lifts multinomial logistic regression (MLR) and fully connected (FC) layers to hyperbolic space via Busemann functions, proposing two unified components—BMLR and BFC—applicable to both the Poincaré ball and the Lorentz model. The proposed components outperform existing hyperbolic layers across four task categories: image classification, genomic sequence classification, node classification, and link prediction.
tags:
  - CVPR 2026
  - Graph Learning
  - Hyperbolic neural networks
  - Busemann function
  - hyperbolic classification
  - fully connected layer
  - manifold learning
date: 2026-05-08
content_hash: 93913ab2a75f3072
---

# Hyperbolic Busemann Neural Networks

**Conference**: CVPR 2026
**arXiv**: [2602.18858](https://arxiv.org/abs/2602.18858)
**Code**: [Available](https://github.com/GitZH-Chen/HBNN)
**Area**: Graph Learning
**Keywords**: Hyperbolic neural networks, Busemann function, hyperbolic classification, fully connected layer, manifold learning

## TL;DR

This paper intrinsically lifts multinomial logistic regression (MLR) and fully connected (FC) layers to hyperbolic space via Busemann functions, proposing two unified components—BMLR and BFC—applicable to both the Poincaré ball and the Lorentz model. The proposed components outperform existing hyperbolic layers across four task categories: image classification, genomic sequence classification, node classification, and link prediction.

## Background & Motivation

### 1. State of the Field

Hyperbolic space, owing to its exponentially growing volume, can embed tree-structured and hierarchical data with low distortion. It has achieved broad success in computer vision, graph learning, multimodal learning, recommendation systems, genomics, and NLP. To support hyperbolic deep learning, MLR and FC layers—two fundamental building blocks—have been repeatedly extended to both the Poincaré ball and the Lorentz model.

### 2. Limitations of Prior Work

Existing hyperbolic MLR and FC layers share several common issues:

- **Over-parameterization**: Ganea et al.'s Poincaré MLR requires an additional manifold-valued parameter $p_k \in \mathbb{P}_K^n$ per class, doubling the parameter count.
- **Poor batch computation efficiency**: Certain methods (e.g., PBMLR-P) require per-class loop computation and cannot be efficiently vectorized.
- **Model specificity**: Poincaré FC applies only to the Poincaré model, and Lorentz FC only to the Lorentz model, with no unified framework.
- **Geometric distortion**: Möbius FC and Lorentz FC perform Euclidean transformations in the tangent space or ambient Minkowski space before projecting back, distorting the intrinsic geometry.

### 3. Root Cause

Practice demands an **intrinsic, efficient, and unified** hyperbolic MLR/FC layer, yet existing approaches are either non-intrinsic (relying on tangent/ambient space approximations), inefficient (over-parameterized or lacking batch support), or non-unified (tied to a single model).

### 4. Paper Goals

To provide unified, parameter-efficient, and batch-efficient MLR and FC layers applicable to both the Poincaré ball and the Lorentz model while preserving authentic geometric distance interpretations.

### 5. Starting Point

**Busemann functions**—the intrinsic generalization of inner products in hyperbolic space. The Euclidean inner product $\langle v, x \rangle$ corresponds to the Busemann function $-B^v(x)$ in hyperbolic space; the Euclidean hyperplane corresponds to the **horosphere**. Both concepts admit closed-form expressions on the Poincaré ball and the Lorentz model.

### 6. Core Idea

By directly replacing the inner product in Euclidean MLR/FC with Busemann functions, this paper derives **BMLR** (Busemann MLR) and **BFC** (Busemann FC). A single set of equations covers both hyperbolic models and naturally reduces to the Euclidean counterparts as the curvature $K \to 0^-$.

## Method

### Overall Architecture

The paper proposes two core components:

1. **BMLR**: Replaces the final classification head, generalizing the Euclidean softmax logit $u_k(x) = \langle a_k, x \rangle + b_k$ to $u_k(x) = -\alpha_k B^{v_k}(x) + b_k$.
2. **BFC**: Replaces intermediate FC layers, generalizing the element-wise Euclidean FC output $y_k = \langle a_k, x \rangle + b_k$ via the signed-distance equation from a point to a horosphere, implicitly defining the output.

Both share the same mathematical framework: Euclidean inner product → Busemann function; Euclidean hyperplane → horosphere.

### Key Designs

#### Design 1: Busemann MLR (BMLR)

**Function**: Lifts logit computation for multi-class classification from Euclidean to hyperbolic space.

**Mechanism**: In Euclidean MLR, the logit $u_k(x) = \alpha_k \langle v_k, x \rangle + b_k$ involves an inner product. By the correspondence between Busemann functions and inner products ($B^v(x) = -\langle x, v \rangle$ in Euclidean space), the hyperbolic logit is defined as:

$$u_k(x) = -\alpha_k B^{v_k}(x) + b_k$$

where $\alpha_k > 0$, $v_k \in \mathbb{S}^{n-1}$, and $b_k \in \mathbb{R}$. On the Poincaré ball, $B^v(x) = \frac{1}{\sqrt{-K}} \log \frac{\|v - \sqrt{-K}x\|^2}{1 + K\|x\|^2}$; on the Lorentz model, $B^v(x) = \frac{1}{\sqrt{-K}} \log(\sqrt{-K}(x_t - \langle x_s, v \rangle))$.

**Design Motivation**:
- **Parameter efficiency**: Each class requires only $(\alpha_k, v_k, b_k)$, totaling $C(n+2)$ parameters, with no additional manifold-valued parameters.
- **Geometric fidelity**: The logit is equivalent to the true geodesic distance from a point to a horosphere (not a pseudo-distance).
- **Batch efficiency**: Logits for all classes can be computed in a single matrix operation.
- **Correct limiting behavior**: As $K \to 0^-$, Poincaré BMLR → $2\alpha_k \langle v_k, x \rangle + b_k$ and Lorentz BMLR → $\alpha_k \langle v_k, x_s \rangle + b_k$, both recovering Euclidean MLR.

#### Design 2: Point-to-Horosphere Distance Interpretation

**Function**: Provides geometric meaning for BMLR logits.

**Mechanism**: In Hadamard spaces (a broader class of metric spaces encompassing both Euclidean and hyperbolic spaces), horospheres—level sets of Busemann functions—are equidistant: $d(H_{\tau_1}^\gamma, H_{\tau_2}^\gamma) = |\tau_2 - \tau_1|$. The distance from a point to a horosphere is thus $d(x, H_\tau^v) = |B^v(x) - \tau|$, and the BMLR logit is exactly the signed distance from the point to the horosphere, scaled by $\alpha_k$.

**Design Motivation**: Analogous to the point-to-hyperplane distance interpretation of Euclidean MLR (Lebanon & Lafferty), this endows the classification decision with a clear geometric meaning—the closer a sample is to the horosphere of a given class, the higher the probability of belonging to that class.

#### Design 3: Busemann FC (BFC) Layer

**Function**: Lifts the FC layer from Euclidean to hyperbolic space.

**Mechanism**: Euclidean FC can be written as $\bar{d}(y, H_{e_k, 0}) = \langle a_k, x \rangle + b_k$, i.e., the $k$-th output dimension is the signed distance to a coordinate hyperplane. Replacing the right-hand side with the Busemann logit and the left-hand side with the hyperbolic point-to-hyperplane distance yields the implicit equation $\bar{d}(y, H_{e_k, e}) = u_k(x)$, from which $y$ is solved.

**Explicit solutions**:
- **Poincaré BFC**: $y = \omega / (1 + \sqrt{1 - K\|\omega\|^2})$, where $\omega_k = \sinh(\sqrt{-K} \cdot u_k(x)) / \sqrt{-K}$
- **Lorentz BFC**: $y_s = \sinh(\sqrt{-K} \cdot u(x)) / \sqrt{-K}$, $y_t = \sqrt{1/(-K) + \|y_s\|^2}$

**Design Motivation**:
- **Intrinsic**: Operates directly on the hyperbolic manifold without tangent or ambient space approximations.
- **Unified**: The same framework applies to both Poincaré and Lorentz models.
- **Extensible**: Supports activation functions $\phi$ by replacing $u_k(x)$ with $\phi(-\alpha_k B^{v_k}(x) + b_k)$; also supports gyroaddition bias terms.
- **Complexity**: FLOPs are $O(nm)$, comparable to existing methods; the Lorentz version requires only $O(2nm)$.

### Loss & Training

- **Classification** (BMLR): Standard cross-entropy loss.
- **Link prediction** (BFC): Fermi-Dirac decoder with cross-entropy, following the original HGCN setup.
- **Parameter constraints**: $v_k$ is maintained on the unit sphere $v_k \in \mathbb{S}^{n-1}$ via normalization; $\alpha_k > 0$ is enforced via softplus.
- **Curvature**: The curvature $K$ is either learned or selected via cross-validation depending on the task.
- **Feature mapping**: In hybrid architectures, Euclidean backbone features are projected to hyperbolic space via the exponential map before being passed to BMLR/BFC.

## Key Experimental Results

### Main Results

#### Table 1: Image Classification Accuracy (ResNet-18 backbone, Top-1 %)

| Space | Method | CIFAR-10 (10 cls) | CIFAR-100 (100 cls) | Tiny-ImageNet (200 cls) | ImageNet-1k (1000 cls) |
|-------|--------|-------------------|---------------------|-------------------------|------------------------|
| $\mathbb{R}^n$ | MLR | 95.14 | 77.72 | 65.19 | 71.87 |
| $\mathbb{P}_K^n$ | PMLR | 95.04 | 77.19 | 64.93 | 71.77 |
| $\mathbb{P}_K^n$ | PBMLR-P | 95.23 | 77.78 | 65.43 | 71.46 |
| $\mathbb{P}_K^n$ | **BMLR-P** | **95.32** | **78.10** | **66.16** | **73.36** |
| $\mathbb{L}_K^n$ | LMLR | 94.98 | 78.03 | 65.63 | 72.46 |
| $\mathbb{L}_K^n$ | **BMLR-L** | **95.25** | **78.07** | **65.99** | **73.24** |

**Key Findings**: The advantage of BMLR over existing hyperbolic MLR methods grows with the number of classes. On ImageNet-1k (1,000 classes), BMLR-P outperforms PMLR by **1.59%** and PBMLR-P by **1.90%**. PBMLR-P has twice the parameter count of other methods and the slowest training speed.

#### Table 2: Node Classification F1 (HGCN backbone) and Link Prediction AUC

| Space | Method | Disease ($\delta=0$) | Airport ($\delta=1$) | PubMed ($\delta=3.5$) | Cora ($\delta=11$) |
|-------|--------|----------------------|----------------------|------------------------|---------------------|
| **Node Classification F1** | | | | | |
| $\mathbb{P}_K^n$ | HGCN (tangent) | 86.87 | 85.34 | 76.29 | 76.56 |
| $\mathbb{P}_K^n$ | HGCN-BMLR-P | **92.45** | **86.02** | **77.36** | **78.48** |
| $\mathbb{L}_K^n$ | HGCN-LMLR | 89.72 | 82.61 | 75.44 | 69.91 |
| $\mathbb{L}_K^n$ | HGCN-BMLR-L | **90.80** | **85.27** | **77.30** | **77.65** |
| **Link Prediction AUC** | | | | | |
| $\mathbb{P}_K^n$ | Poincaré FC | 79.45 | 94.31 | 94.24 | 88.21 |
| $\mathbb{P}_K^n$ | BFC-P | **80.45** | **94.88** | **94.85** | **91.94** |
| $\mathbb{L}_K^n$ | Lorentz FC | 72.78 | 92.99 | 94.20 | 92.06 |
| $\mathbb{L}_K^n$ | BFC-L | **78.36** | **95.37** | **94.90** | **92.28** |

### Ablation Study

- **Effect of class count**: As the number of classes increases from CIFAR-10 (10) to ImageNet-1k (1,000), the advantage of BMLR grows from ~0.2% to ~1.6%, demonstrating the superior expressivity of Busemann functions for high-class-count classification.
- **Effect of graph hyperbolicity**: In node classification, LMLR degrades severely on Cora ($\delta=11$, least hyperbolic), dropping to 69.91 vs. 77.37 for the tangent baseline, whereas BMLR-L maintains 77.65, demonstrating robustness to varying graph hyperbolicity.
- **Link prediction on Disease ($\delta=0$, most hyperbolic)**: BFC-L outperforms Lorentz FC by 5.58%, showing that the geometric advantage of Busemann functions is greatest on the most hyperbolic data.

### Key Findings

1. **Advantage scales with class count**: On ImageNet-1k, BMLR outperforms PMLR by 1.59% and LMLR by 0.78%.
2. **Fastest training speed**: Lorentz BMLR achieves the lowest FLOPs and shortest fitting time among all hyperbolic MLR methods; PBMLR-P, lacking batch computation support, is consistently the slowest across 16 genomic datasets.
3. **Greater gain in more hyperbolic settings**: On Disease ($\delta=0$), BFC-L outperforms Lorentz FC by 5.58%, while the gap narrows to 0.22% on the flatter Cora ($\delta=11$).
4. **Robustness**: While existing hyperbolic MLR methods can underperform the tangent baseline on less hyperbolic graphs (e.g., LMLR degrades substantially on Cora), BMLR achieves the best performance across all values of $\delta$.

## Highlights & Insights

- **Mathematical elegance**: Busemann functions provide a unified generalization from Euclidean inner products to hyperbolic space, with a single formula covering both the Poincaré ball and the Lorentz model.
- **Theoretical completeness**: The paper proves the equidistance property of horospheres in Hadamard spaces (Thm 3.3), provides a point-to-horosphere distance interpretation for BMLR, and establishes the limiting behavior as $K \to 0^-$.
- **Practical efficiency**: BMLR-L requires $C(2n+12)$ FLOPs, close to the $C(2n)$ of Euclidean MLR, incurring nearly zero overhead.
- **Cross-domain validation**: The method is evaluated across four task categories (vision, genomics, graph node classification, graph link prediction) covering 20+ datasets, demonstrating its generality.

## Limitations & Future Work

1. **Limited to MLR and FC**: Attention, normalization, residual connections, and other network components have not been reformulated using Busemann functions. Whether a complete Busemann network can be constructed remains an open question.
2. **Fixed or manually selected curvature**: Although learnable curvature is mentioned, experiments primarily rely on cross-validation for curvature selection; adaptive curvature learning warrants further exploration.
3. **Restricted to constant-curvature spaces**: Real-world data may exhibit variable-curvature structure (e.g., product spaces $\mathbb{H} \times \mathbb{E}$). Extending Busemann functions to mixed-curvature spaces is a promising research direction.
4. **Limited large-scale GNN experiments**: Graph learning experiments use only small-scale datasets (the largest being PubMed with ~20K nodes); performance on million-scale graphs remains unverified.

## Related Work & Insights

- **Builds upon**: Ganea et al. (NeurIPS'18) Poincaré MLR/FC → Shimizu et al. (NeurIPS'21) reparameterization → Bdeir et al. (ICLR'24) Lorentz MLR/CNN.
- **Busemann functions in ML**: Fan et al. hyperbolic SVM; Chami et al. hyperbolic PCA; Bonet et al. Sliced-Wasserstein.
- **Broader inspiration**: The role of Busemann functions as an "intrinsic inner product" can be analogized to other Hadamard manifolds (e.g., the space of SPD matrices), providing a template for designing more general manifold neural network components.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Using Busemann functions as a unifying tool for constructing hyperbolic MLR and FC layers is mathematically well-motivated and theoretically elegant, though the core idea is a combination of existing tools.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Systematic comparisons across 4 task categories, 20+ datasets, and two hyperbolic models with efficiency analysis are provided; however, graph experiments use only classic small-scale benchmarks and lack large-scale evaluations such as OGB.
- ⭐⭐⭐⭐⭐ **Writing Quality**: The theorem–proof structure is rigorous, comparison tables are clear and comprehensive, and the Euclidean-to-hyperbolic analogy is presented in a coherent and accessible narrative.
- ⭐⭐⭐⭐ **Value**: Code is publicly available; BMLR/BFC are plug-and-play components; Lorentz BMLR achieves speed comparable to Euclidean MLR, with low barriers to practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)

<!-- RELATED:END -->
