---
title: >-
  [Paper Note] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement
description: >-
  [ICLR 2026][Interpretability][rich dynamics] This paper proposes a computationally efficient, performance-agnostic measure of dynamical richness, $\mathcal{D}_{LR}$…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "rich dynamics"
  - "lazy training"
  - "neural collapse"
  - "feature learning"
  - "CKA"
date: 2026-05-08
content_hash: 8a1335219d336d4c
---

# Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement

**Conference**: ICLR 2026
**arXiv**: [2410.04264](https://arxiv.org/abs/2410.04264)  
**Code**: Available (provided in appendix)  
**Area**: Interpretability
**Keywords**: rich dynamics, lazy training, neural collapse, feature learning, CKA

## TL;DR
This paper proposes a computationally efficient, performance-agnostic measure of dynamical richness, $\mathcal{D}_{LR}$, which quantifies rich/lazy training dynamics by comparing activations before and after the last layer, and demonstrates that neural collapse is a special case of this measure.

## Background & Motivation

**Background**: Feature learning in deep learning can be viewed from two perspectives — representation quality (how useful features are for downstream tasks) and a dynamics perspective (rich vs. lazy training). Rich training refers to features undergoing nonlinear dynamic transformations, whereas lazy training approximates linear model behavior.

**Limitations of Prior Work**: Existing richness measures each suffer from distinct drawbacks. NTK-based change metrics are computationally prohibitive (scaling quadratically with the number of parameters); the initial kernel similarity $\mathcal{S}_{init}$ depends on the initial kernel and sometimes produces incorrect judgments (e.g., weight decay causes kernel changes without constituting genuinely rich training); the parameter norm $\|\theta\|_F^2$ is merely correlational rather than causal; and the NC1 metric from neural collapse is unbounded and sensitive to output scaling.

**Key Challenge**: Rich dynamics and better representations are frequently conflated, with accuracy used as a proxy for richness. In practice, rich dynamics do not always imply better generalization — the authors demonstrate on MNIST that a richly trained model achieves only 10% test accuracy, while a lazily trained model reaches 74.4%.

**Goal**: (1) A richness measure independent of model performance; (2) high computational efficiency; (3) a unified account of known phenomena such as neural collapse.

**Key Insight**: The low-rank bias of rich training — under rich dynamics, features prior to the last layer should span only the minimal number of dimensions required to express the learned function (low-rank structure).

**Core Idea**: Define a minimal projection operator $\mathcal{T}_{MP}$, and use CKA to measure the distance between the actual feature kernel and this ideal low-rank projection; smaller values indicate richer training.

## Method

### Overall Architecture
Input images are passed through the network, extracting penultimate-layer features $\Phi(x) \in \mathbb{R}^p$ and final outputs $\hat{f}(x) \in \mathbb{R}^C$. A feature kernel operator $\mathcal{T}$ and an ideal minimal projection operator $\mathcal{T}_{MP}$ are constructed; CKA is used to compare the two, yielding the richness measure $\mathcal{D}_{LR} \in [0,1]$.

### Key Designs

1. **Feature Kernel Operator $\mathcal{T}$**:

    - Function: Maps feature representations into a kernel operator in function space.
    - Mechanism: $\mathcal{T} = \sum_{k=1}^{p} |\Phi_k\rangle\langle\Phi_k|$, i.e., the sum of outer products over all feature dimensions. Eigenvalues $\rho_k$ and eigenfunctions $e_k$ are obtained via Mercer's theorem.
    - Design Motivation: Operating in function space rather than vector space makes the measure independent of specific training samples.

2. **Minimal Projection Operator $\mathcal{T}_{MP}$ (Definition 1)**:

    - Function: Defines the ideal state of rich training.
    - Mechanism: $\mathcal{T}_{MP}[u] = a_1\langle \mathbf{1}|u\rangle\mathbf{1} + a_2 P_{\hat{\mathcal{H}}}(u)$, where $P_{\hat{\mathcal{H}}}$ is the orthogonal projection onto the learned function space $\hat{\mathcal{H}} = \text{span}\{\hat{f}_1, \ldots, \hat{f}_C\}$. When the actual $\mathcal{T}$ coincides with $\mathcal{T}_{MP}$, features span only a $C$-dimensional space (matching the output dimensionality), perfectly embodying the low-rank bias of rich training.
    - Design Motivation: Under ideal rich dynamics, only the minimal number of features are learned and utilized, requiring no additional processing by the last layer.

3. **Low-Rank Measure $\mathcal{D}_{LR}$**:

    - Function: Quantifies the degree of training richness.
    - Mechanism: $\mathcal{D}_{LR} = 1 - \text{CKA}(\mathcal{T}, \mathcal{T}_{MP})$, with range $[0,1]$; 0 denotes maximally rich and 1 denotes maximally lazy training.
    - Connection to Neural Collapse: When $\mathcal{T} = \mathcal{T}_{MP}$, NC1 (within-class variability collapse) and NC2 (feature convergence to a simplex equiangular tight frame) follow automatically.

4. **Feature Decomposition Visualization (Eq. 5)**:

    - Function: Provides richer diagnostic information beyond a single scalar.
    - Three complementary views: (i) cumulative quality $\Pi^*(k)$ — how well the top-$k$ features express the target function; (ii) cumulative utilization $\hat{\Pi}(k)$ — how much of the top-$k$ features are exploited by the last layer; (iii) relative eigenvalues $\rho_k/\rho_1$ — the relative importance of each feature.
    - Eigenfunctions are approximated from finite samples via the Nyström method, with computational complexity $\mathcal{O}(p^2 C)$.

### Computational Efficiency
Key advantage: Only $n$ forward passes are required to obtain penultimate-layer ($n \times p$) and output-layer ($n \times C$) activations, followed by $\mathcal{O}(npC)$ computation. For standard models with $p \approx 10^3$ and $n \approx \mathcal{O}(p)$, the total cost is $\mathcal{O}(p^2 C)$, far superior to NTK-based methods that scale quadratically with the total number of parameters.

## Key Experimental Results

### Main Results: Comparison with Existing Richness Measures

| Measure | Dependencies | Weight Decay Misclassification | Target Downscaling Alignment | Complexity |
|------|------|------|------|------|
| $\mathcal{D}_{LR}$ (Ours) | None (no labels/initial kernel/performance) | ✓ Correct | ✓ Varies consistently with $\alpha$ | $\mathcal{O}(p^2 C)$ |
| $\mathcal{S}_{init}$ (initial kernel distance) | Initial kernel | ✗ Misclassifies | ✗ Invariant to $\alpha$ | $\mathcal{O}(n^2 p)$ |
| $\|\theta\|_F^2$ (parameter norm) | Initial parameters | ✗ Misclassifies | ✗ Invariant to $\alpha$ | $\mathcal{O}(D)$ |
| NC1 (neural collapse) | Class labels | ✗ Unstable magnitude | ✗ Opposite direction | $\mathcal{O}(np^2)$ |

### Relationship Between Training Factors and Richness

| Task | Architecture | Condition | Test Acc.↑ | $\mathcal{D}_{LR}$↓ |
|------|------|------|------|------|
| Mod 97 | 2-layer Transformer | Before grokking (step 200) | 5.2% | 0.51 |
| Mod 97 | 2-layer Transformer | After grokking (step 3000) | 99.8% | 0.11 |
| CIFAR-100 | ResNet18 | lr=0.005 | 66.3% | 0.053 |
| CIFAR-100 | ResNet18 | lr=0.05 (optimal) | 78.3% | 0.025 |
| CIFAR-100 | ResNet18 | lr=0.2 | 74.5% | 0.039 |
| CIFAR-100 | VGG-16 | Without BN | 21.7% | 0.66 |
| CIFAR-100 | VGG-16 | With BN | 72.0% | 0.073 |

### Ablation Study

| Setting | Test Acc. | $\mathcal{D}_{LR}$ | Notes |
|------|---------|------|------|
| MNIST rich (full backprop) | 10.0% | 0.0087 | Rich ≠ good generalization |
| MNIST lazy (last layer only) | 74.4% | 0.63 | Lazy but better generalization |
| CIFAR-10 no label shuffling | 95.0% | 0.031 | Rich + good generalization |
| CIFAR-10 fully shuffled labels | 9.5% | 0.034 | Rich but no generalization |

### Key Findings
- Grokking represents a lazy→rich transition: $\mathcal{D}_{LR}$ decreases from 0.51 to 0.11, verified independently of performance for the first time.
- The role of batch normalization is reframed: VGG-16 without BN exhibits lazy dynamics (0.66), while adding BN induces rich dynamics (0.073), offering a new perspective on the mechanisms of BN.
- The optimal learning rate corresponds to the richest training: ResNet18 on CIFAR-100 achieves the smallest $\mathcal{D}_{LR}$ at lr=0.05.
- Feature quality and feature magnitude are correlated during training: features associated with larger eigenvalues improve in quality more rapidly.

## Highlights & Insights
- Neural collapse is unified as a special case of richness — both NC1 and NC2 can be derived from $\mathcal{T} = \mathcal{T}_{MP}$, implying that neural collapse is fundamentally a dynamic phenomenon rather than a generalization indicator.
- The empirical demonstration that "rich ≠ better" clearly shows that rich training can concentrate features on spurious encodings, leading to generalization failure, thereby challenging the assumption that rich training is inherently superior.
- The finding that BN promotes rich dynamics can be transferred to the study of how other normalization techniques affect training dynamics.

## Limitations & Future Work
- Only last-layer features are considered; intermediate-layer dynamics are not addressed (the authors discuss this in the appendix).
- The method assumes orthogonal and isotropic target functions, and does not cover class-imbalanced scenarios.
- The visualization approach relies on the Nyström approximation and requires $n > p$ samples.
- The framework could be extended to richness analysis in regression tasks and generative models.

## Related Work & Insights
- **vs. NTK-based metrics**: NTK methods are theoretically more complete but computationally infeasible; this paper approximates using the last-layer kernel, substantially improving practicality.
- **vs. Neural Collapse (Papyan et al., 2020)**: Neural collapse is a special case of the proposed method, but NC relies on class labels and yields an unbounded metric.
- **vs. Feature Learning Theory (Yang & Hu, 2021)**: The μP framework theoretically predicts rich/lazy transitions; this paper provides an empirical measurement tool.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical framework unifying richness measures and neural collapse is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experiments across diverse architectures, datasets, and training factors.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear paper structure, high-quality figures, and well-motivated intuitive explanations.
- Value: ⭐⭐⭐⭐ Practically valuable as a diagnostic tool, and theoretically bridges rich dynamics and representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)

</div>

<!-- RELATED:END -->
