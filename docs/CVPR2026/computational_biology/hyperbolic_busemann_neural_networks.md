---
title: >-
  [Paper Note] Hyperbolic Busemann Neural Networks
description: >-
  [CVPR 2026][Computational Biology][Hyperbolic Neural Networks] This work utilizes Busemann functions to intrinsically lift Multinomial Logistic Regression (MLR) and Fully Connected (FC) layers into hyperbolic space. It introduces two unified components, BMLR and BFC, which are applicable to both the Poincaré ball and Lorentz models. These components outperform existing hyperbolic layers across four task categories: image classification, genomic sequences, node classification…
tags:
  - "CVPR 2026"
  - "Computational Biology"
  - "Hyperbolic Neural Networks"
  - "Busemann functions"
  - "Hyperbolic classification"
  - "Fully connected layers"
  - "Manifold learning"
date: 2026-05-08
content_hash: 48c1cadf5eae1071
---

# Hyperbolic Busemann Neural Networks

**Conference**: CVPR 2026  
**arXiv**: [2602.18858](https://arxiv.org/abs/2602.18858)  
**Code**: [Available](https://github.com/GitZH-Chen/HBNN)  
**Area**: Computational Biology  
**Keywords**: Hyperbolic Neural Networks, Busemann functions, Hyperbolic classification, Fully connected layers, Manifold learning  

## TL;DR

This work utilizes Busemann functions to intrinsically lift Multinomial Logistic Regression (MLR) and Fully Connected (FC) layers into hyperbolic space. It introduces two unified components, BMLR and BFC, which are applicable to both the Poincaré ball and Lorentz models. These components outperform existing hyperbolic layers across four task categories: image classification, genomic sequences, node classification, and link prediction.

## Background & Motivation

### 1. Background

Due to their exponential volume growth, hyperbolic spaces can embed tree-like and hierarchical structures with low distortion. In recent years, they have achieved widespread success in computer vision, graph learning, multimodal learning, recommendation systems, genomics, and NLP. To support hyperbolic deep learning, core components such as MLR and FC layers have been generalized several times to the Poincaré ball and Lorentz models.

### 2. Limitations of Prior Work

Existing hyperbolic MLR and FC layers suffer from several common issues:

- **Over-parameterization**: The Poincaré MLR by Ganea et al. requires an additional manifold parameter $p_k \in \mathbb{P}_K^n$ for each class, doubling the parameter count.
- **Low batch computation efficiency**: Some methods (e.g., PBMLR-P) require looping through classes individually, preventing efficient matrixization.
- **Model specificity**: Poincaré FC layers only apply to the Poincaré model, while Lorentz FC layers only apply to the Lorentz model, lacking a unified framework.
- **Geometric distortion**: Möbius FC and Lorentz FC perform Euclidean transformations in tangent spaces or ambient Minkowski spaces followed by projection, distorting the intrinsic geometry.

### 3. Key Challenge

Practice requires an **intrinsic, efficient, and unified** hyperbolic MLR/FC layer. However, current solutions are either non-intrinsic (relying on tangent/ambient space approximations), inefficient (over-parameterized/unsupported multi-class batching), or non-unified (bound to a single model).

### 4. Goal

The goal is to provide unified, parameter-compact, and batch-efficient MLR and FC layers for both the Poincaré and Lorentz hyperbolic models, while retaining a true geometric distance interpretation.

### 5. Key Insight

**Busemann functions** serve as the intrinsic generalization of the inner product in hyperbolic space. The hyperbolic counterpart of the Euclidean inner product $\langle v, x \rangle$ is the Busemann function $-B^v(x)$, and the counterpart of Euclidean hyperplanes is the **horosphere**. These concepts have analytical closed-form expressions in both Poincaré and Lorentz models.

### 6. Core Idea

The Euclidean inner product operation in MLR/FC is directly replaced with Busemann functions, resulting in **BMLR** (Busemann MLR) and **BFC** (Busemann FC). A single set of formulas covers both hyperbolic models and naturally recovers Euclidean counterparts as the curvature $K \to 0^-$.

## Method

### Overall Architecture

This paper addresses a fundamental gap in hyperbolic neural networks: the lack of a formulation for classification heads (MLR) and fully connected layers (FC) that is intrinsic, compact, and unified across both Poincaré and Lorentz models. The authors observe that in Euclidean space, these components are fundamentally built upon the "inner product $\langle v, x\rangle$" and "hyperplanes." Since the intrinsic hyperbolic counterpart of the inner product is the **Busemann function** $-B^v(x)$ and the counterpart of hyperplanes is the **horosphere**, they shift the entire framework to hyperbolic space by substituting these terms.

Based on this approach, two components are developed: **BMLR** is used at the end of the network for classification, rewriting the Euclidean logit $u_k(x)=\langle a_k,x\rangle+b_k$ using Busemann functions. **BFC** replaces intermediate fully connected layers by substituting the Euclidean FC definition ("output dimension $k$ = signed distance to a coordinate hyperplane") with Busemann logits and solving for the hyperbolic point $y$. Since both share the "inner product $\to$ Busemann, hyperplane $\to$ horosphere" dictionary, the formulas apply to both models and automatically revert to Euclidean versions when $K\to 0^-$.

### Key Designs

**1. Busemann MLR: Replacing Inner Products with Busemann Functions**

The inner product in the Euclidean MLR logit $u_k(x)=\alpha_k\langle v_k,x\rangle+b_k$ has no direct hyperbolic equivalent; previous hyperbolic MLR designs compensated by adding an extra manifold parameter $p_k$ per class, doubling parameters. This work reinterprets the inner product: since $B^v(x)=-\langle x,v\rangle$ in Euclidean space, the Busemann function is effectively the negative inner product. Thus, the inner product is replaced directly to obtain:

$$u_k(x) = -\alpha_k B^{v_k}(x) + b_k$$

where $\alpha_k>0$, $v_k\in\mathbb{S}^{n-1}$, and $b_k\in\mathbb{R}$. Each class requires only $(\alpha_k,v_k,b_k)$, totaling $C(n+2)$ parameters, with no manifold-valued parameters. $B^v(x)$ has closed forms for both models: in the Poincaré ball, $B^v(x)=\frac{1}{\sqrt{-K}}\log\frac{\|v-\sqrt{-K}x\|^2}{1+K\|x\|^2}$, and in the Lorentz model, $B^v(x)=\frac{1}{\sqrt{-K}}\log(\sqrt{-K}(x_t-\langle x_s,v\rangle))$. These can be batched as matrices, computing all class logits simultaneously—unlike PBMLR-P, which requires per-class loops. Furthermore, as $K\to 0^-$, the Poincaré version approaches $2\alpha_k\langle v_k,x\rangle+b_k$ and the Lorentz version approaches $\alpha_k\langle v_k,x_s\rangle+b_k$, ensuring a clean generalization.

**2. Distance-to-Horosphere Interpretation: Geometric Meaning of Logits**

By replacing the inner product with the Busemann function, the resulting logit gains a geometric meaning: it represents the "signed geodesic distance from a point to a horosphere." In Hadamard spaces (including Euclidean and hyperbolic), the level sets of Busemann functions—horospheres—are equidistant: $d(H_{\tau_1}^\gamma,H_{\tau_2}^\gamma)=|\tau_2-\tau_1|$. Thus, the distance to any horosphere is $d(x,H_\tau^v)=|B^v(x)-\tau|$. Consequently, the BMLR logit equals the signed distance multiplied by $\alpha_k$. This ports the Euclidean MLR interpretation ("logit = signed distance to decision hyperplane" by Lebanon & Lafferty) directly to hyperbolic space using true geodesic distances.

**3. Busemann FC: Applying the Same Dictionary to FC Layers**

The FC layer requires outputting a new point rather than just a scalar logit. The Euclidean FC is first rewritten as a distance equality $\bar{d}(y,H_{e_k,0})=\langle a_k,x\rangle+b_k$, where the $k$-th dimension of output $y$ equals the signed distance to the $k$-th coordinate hyperplane. Both sides are then replaced: the inner product is swapped for the Busemann logit $u_k(x)$, and the Euclidean distance is swapped for the hyperbolic distance to a horosphere, forming the implicit equation $\bar{d}(y,H_{e_k,e})=u_k(x)$. Closed-form solutions for $y$ exist: in Poincaré, $y=\omega/(1+\sqrt{1-K\|\omega\|^2})$ with $\omega_k=\sinh(\sqrt{-K}\,u_k(x))/\sqrt{-K}$; in Lorentz, $y_s=\sinh(\sqrt{-K}\,u(x))/\sqrt{-K}$ and $y_t=\sqrt{1/(-K)+\|y_s\|^2}$. The process remains on the hyperbolic manifold, avoiding geometric distortions introduced by tangent space approximations.

### Loss & Training

- **Classification Tasks** (BMLR): Standard cross-entropy loss.
- **Link Prediction** (BFC): Fermi-Dirac decoder with cross-entropy, following HGCN settings.
- **Parameter Constraints**: $v_k$ is constrained to the unit sphere $\mathbb{S}^{n-1}$ via normalization; $\alpha_k > 0$ is ensured via softplus.
- **Curvature**: Curvature $K$ is treated as a learnable parameter or selected via cross-validation.
- **Feature Mapping**: In hybrid architectures, Euclidean backbone outputs are projected to hyperbolic space via the exponential map before being fed into BMLR/BFC.

## Key Experimental Results

### Main Results

**Table 1: Image Classification Accuracy (ResNet-18 backbone, Top-1 %)**

| Space | Method | CIFAR-10 (10 classes) | CIFAR-100 (100 classes) | Tiny-ImageNet (200 classes) | ImageNet-1k (1000 classes) |
|-------|--------|-----------------------|-------------------------|-----------------------------|----------------------------|
| $\mathbb{R}^n$ | MLR | 95.14 | 77.72 | 65.19 | 71.87 |
| $\mathbb{P}_K^n$ | PMLR | 95.04 | 77.19 | 64.93 | 71.77 |
| $\mathbb{P}_K^n$ | PBMLR-P | 95.23 | 77.78 | 65.43 | 71.46 |
| $\mathbb{P}_K^n$ | **BMLR-P** | **95.32** | **78.10** | **66.16** | **73.36** |
| $\mathbb{L}_K^n$ | LMLR | 94.98 | 78.03 | 65.63 | 72.46 |
| $\mathbb{L}_K^n$ | **BMLR-L** | **95.25** | **78.07** | **65.99** | **73.24** |

**Key Findings**: The lead of BMLR over existing hyperbolic MLR layers increases as the number of classes grows—on ImageNet-1k (1000 classes), BMLR-P is **1.59%** higher than PMLR and **1.90%** higher than PBMLR-P. PBMLR-P has twice the parameters and the slowest training speed.

**Table 2: Node Classification F1 (HGCN backbone) and Link Prediction AUC**

| Space | Method | Disease (δ=0) | Airport (δ=1) | PubMed (δ=3.5) | Cora (δ=11) |
|-------|--------|---------------|---------------|-----------------|-------------|
| **Node Class F1** | | | | | |
| $\mathbb{P}_K^n$ | HGCN (tangent) | 86.87 | 85.34 | 76.29 | 76.56 |
| $\mathbb{P}_K^n$ | HGCN-BMLR-P | **92.45** | **86.02** | **77.36** | **78.48** |
| $\mathbb{L}_K^n$ | HGCN-LMLR | 89.72 | 82.61 | 75.44 | 69.91 |
| $\mathbb{L}_K^n$ | HGCN-BMLR-L | **90.80** | **85.27** | **77.30** | **77.65** |
| **Link Pred AUC** | | | | | |
| $\mathbb{P}_K^n$ | Poincaré FC | 79.45 | 94.31 | 94.24 | 88.21 |
| $\mathbb{P}_K^n$ | BFC-P | **80.45** | **94.88** | **94.85** | **91.94** |
| $\mathbb{L}_K^n$ | Lorentz FC | 72.78 | 92.99 | 94.20 | 92.06 |
| $\mathbb{L}_K^n$ | BFC-L | **78.36** | **95.37** | **94.90** | **92.28** |

### Ablation Study

- **Number of Classes Effect**: From CIFAR-10 to ImageNet-1k, BMLR's advantage expands from ~0.2% to ~1.6%, demonstrating the representational superiority of Busemann functions in high-dimensional classification.
- **Hyperbolicity Effect**: In node classification, LMLR degrades significantly on Cora ($\delta=11$, least hyperbolic; 69.91 vs 77.37 for tangent), but BMLR-L remains robust at 77.65.
- **Disease Dataset ($\delta=0$, most hyperbolic)**: In link prediction, BFC-L is 5.58% higher than Lorentz FC, showing Busemann geometry provides the greatest gain on the most hyperbolic data.

### Key Findings

1. **Gains scale with class count**: BMLR outperforms PMLR by 1.59% and LMLR by 0.78% on the 1000-class ImageNet-1k.
2. **Fastest training speed**: Lorentz BMLR has the lowest FLOPs and shortest fit time among hyperbolic MLRs; PBMLR-P is consistently the slowest due to lack of batching support.
3. **Larger gains in more hyperbolic geometry**: BFC-L is 5.58% higher than Lorentz FC on Disease ($\delta=0$), while the gap narrows to 0.22% on the flatter Cora ($\delta=11$).
4. **Robustness**: While prior hyperbolic MLRs may perform worse than tangent baselines on non-hyperbolic graphs (e.g., LMLR on Cora), BMLR remains optimal across all $\delta$.

## Highlights & Insights

- **Mathematical Elegance**: Uses Busemann functions to unify the generalization of Euclidean inner products to hyperbolic space across both Poincaré and Lorentz models.
- **Theoretical Completeness**: Proves the equidistance of horospheres in Hadamard space (Thm 3.3) and provides the signed-distance-to-horosphere interpretation and curvature limit theorems.
- **High Practicality**: The FLOPs for BMLR-L are $C(2n+12)$, close to the Euclidean MLR's $C(2n)$, representing negligible overhead.
- **Cross-Domain Validation**: Demonstrates generalizability across vision, genomics, node classification, and link prediction.

## Limitations & Future Work

1. **Component Coverage**: Only MLR and FC are covered. Whether attention, normalization, and residual layers can be reconstructed with Busemann functions to build a complete Busemann network remains to be seen.
2. **Curvature Selection**: While learnable curvature is mentioned, experiments mainly use cross-validation. Adaptive curvature learning requires further exploration.
3. **Constant Curvature**: Real data may have variable curvature. Investigating Busemann functions in product spaces (e.g., $\mathbb{H} \times \mathbb{E}$) is a valuable direction.
4. **Large-Scale GNNs**: Graph experiments were limited to small datasets (max PubMed ~20K nodes). Performance on million-scale benchmarks (e.g., OGB) is unverified.

## Related Work & Insights

- **Succession**: Extends Ganea et al. (NeurIPS'18) Poincaré MLR/FC $\to$ Shimizu et al. (NeurIPS'21) reparameterization $\to$ Bdeir et al. (ICLR'24) Lorentz MLR/CNN.
- **Busemann Functions in ML**: Links to Fan et al. Hyperbolic SVM, Chami et al. Hyperbolic PCA, and Bonet et al. Sliced-Wasserstein.
- **Inspiration**: The role of Busemann functions as an "intrinsic inner product" can be analogized to other Hadamard manifolds (e.g., SPD matrix spaces), providing a template for designing manifold neural network components.

## Rating

- ⭐⭐⭐⭐ Novelty: Uses Busemann functions to unify hyperbolic MLR and FC; clear mathematical motivation and elegant framework, though based on existing tools.
- ⭐⭐⭐⭐ Experimental Thoroughness: Systematic comparison across 4 task types, 20+ datasets, and 2 models including efficiency; lacks large-scale OGB-style benchmarks.
- ⭐⭐⭐⭐⭐ Writing Quality: Rigorous theorem-proof structure, comprehensive tables, and clear narrative on Euclidean-Hyperbolic analogies.
- ⭐⭐⭐⭐ Value: Open-sourced code; BMLR/BFC are plug-and-play, and Lorentz BMLR speed is nearly Euclidean, ensuring a low barrier for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyperbolic Neural Population Geometry Benefits Computation](../../ICML2026/computational_biology/hyperbolic_neural_population_geometry_benefits_computation.md)
- [\[CVPR 2026\] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)
- [\[CVPR 2026\] CryoKRAQEN: Kernel-Regularized Annealing for Quantized Embedding Networks in Cryo-EM Heterogeneous Reconstruction](cryokraqen_kernel-regularized_annealing_for_quantized_embedding_networks_in_cryo.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)

</div>

<!-- RELATED:END -->
