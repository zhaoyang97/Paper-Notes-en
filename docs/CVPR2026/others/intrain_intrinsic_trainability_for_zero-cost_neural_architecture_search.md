---
title: >-
  [Paper Note] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search
description: >-
  [CVPR 2026][Zero-cost proxies] InTrain formalizes "whether an architecture can be trained effectively" as an intrinsic invariant independent of the training process. By combining the **Geometric Capacity** (via Participation Ratio) of forward activations and the **Optimization Resilience** (via Gradient Health) of backward gradients through scale-invariant **multiplicative coupling**, it achieves ranking correlations on NAS-Bench-101/201 that match ensemble proxies and surpas…
tags:
  - "CVPR 2026"
  - "Zero-cost proxies"
  - "Neural Architecture Search"
  - "Trainability"
  - "Participation Ratio"
  - "Gradient Health"
date: 2026-05-08
content_hash: 73f0fe89703e4655
---

# InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_InTrain_Intrinsic_Trainability_for_Zero-Cost_Neural_Architecture_Search_CVPR_2026_paper.html)  
**Code**: None  
**Area**: AutoML / Neural Architecture Search  
**Keywords**: Zero-cost proxies, Neural Architecture Search, Trainability, Participation Ratio, Gradient Health

## TL;DR
InTrain formalizes "whether an architecture can be trained effectively" as an intrinsic invariant independent of the training process. By combining the **Geometric Capacity** (via Participation Ratio) of forward activations and the **Optimization Resilience** (via Gradient Health) of backward gradients through scale-invariant **multiplicative coupling**, it achieves ranking correlations on NAS-Bench-101/201 that match ensemble proxies and surpass all single-index proxies.

## Background & Motivation
**Background**: Neural Architecture Search (NAS) can automatically discover networks that rival or even exceed human-designed ones. However, traditional NAS requires training thousands of candidate architectures to convergence, often consuming thousands of GPU days. To bypass this cost, **zero-cost (training-free) proxies** have emerged—predicting the final accuracy of an architecture by calculating a scalar score at initialization, reducing evaluation costs by several orders of magnitude.

**Limitations of Prior Work**: Existing zero-cost proxies (such as SNIP, GraSP, SynFlow, NASWOT, ZiCo, Zen-score, etc.) each focus on a single "symptom" of trainability: either gradient magnitude/connection sensitivity or expression diversity of activations. Gradient-based metrics ignore representation geometry, while activation-based metrics ignore training dynamics. Comprehensive benchmark studies also find that many proxies exhibit inconsistent correlations across different datasets and search spaces.

**Key Challenge**: These methods are pieced together using scattered heuristics but fail to answer the fundamental question—**what exactly makes an architecture "trainable"**? In other words, there is a lack of a theoretical criterion that simultaneously unifies "geometry (representational power)" and "gradients (optimizability)" while remaining independent of specific optimizers and hyperparameters.

**Goal**: To define **intrinsic trainability**—an architectural invariant determined solely by network topology and initial parameters, independent of the training pipeline—and implement it as a zero-cost proxy that can be computed in seconds.

**Key Insight**: The authors view a deep network as a bidirectional information processor: the forward pass transforms input into increasingly abstract representations layer by layer, while the backward pass allows error signals to drive optimization through the parameter space. A trainable architecture must be strong at both ends—possessing sufficient **geometric capacity** to represent complex functions and sufficient **optimization resilience** to allow stable gradient flow.

**Core Idea**: Intrinsic trainability is characterized by two measurable components: "geometric capacity $\gamma$" and "optimization resilience $o$". The paper asserts that their relationship is not additive but governed by **multiplicative gating** (if one collapses, the other becomes useless), finally synthesizing them into a single score $I(A)$.

## Method

### Overall Architecture
InTrain does not train any weights; it performs only **one forward pass + one backward pass** on the initialized network. The forward pass collects activations from each layer to calculate the "geometric capacity" $\gamma(A)$, and the backward pass uses a synthetic loss to activate all parameter paths to calculate the "optimization resilience" $o(A)$. Finally, the two are coupled multiplicatively and log-normalized by depth to obtain the intrinsic trainability score $I(A)$. This metric is built on three design principles: **depth invariance** (allowing fair comparison across depths), **compositionality** (information is transmitted multiplicatively across layers, meaning a bottleneck in any layer blocks downstream flow, hence log-product aggregation), and **bidirectionality** (accounting for both forward and backward passes). This is a purely analytical calculation chain involving no multi-module coordination or feedback loops, thus expressed more clearly via formulas than a framework diagram.

### Key Designs

**1. Geometric Capacity: Quantifying representation manifold dimensions via Participation Ratio**

Addressing the issue where "activation proxies focus on diversity but lack the essence of expressivity," InTrain treats the activation of the $\ell$-th layer $A_\ell\in\mathbb{R}^{N\times C}$ as a point cloud in feature space, using the eigen-spectrum of its covariance matrix $C_\ell$ to measure the dimensionality over which information spreads. Instead of matrix rank (which treats all non-zero singular values equally), it uses the **Participation Ratio (PR)**:

$$\mathrm{PR}(C_\ell)=\frac{(\operatorname{Tr} C_\ell)^2}{\operatorname{Tr}(C_\ell^2)}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

PR is equivalent to the exponential of the second-order Rényi entropy of the normalized eigenvalue distribution $\mathrm{PR}=\exp(H_2)$ (where $p_i=\lambda_i/\operatorname{Tr}C_\ell$ and $H_2=-\log\sum_i p_i^2$). A high PR indicates that variance is uniformly distributed across many dimensions, signifying no representation collapse. Convolutional activations of shape $(N,C,H,W)$ are reshaped to $(N\cdot H\cdot W, C)$ to compute covariance along the channel dimension, with $\epsilon I$ ($\epsilon=10^{-10}$) added for numerical stability. Since information is subject to multiplicative constraints across layers (collapse in one layer kills all subsequent layers), geometric capacity is aggregated using a **log-product**:

$$\gamma(A)=\sum_{\ell=1}^{L}\log\big(\mathrm{PR}(C_\ell)\big)$$

**2. Optimization Resilience: Quantifying backpropagation stability via Gradient Health**

Geometric capacity describes "what can be represented," but even with high expressivity, a network cannot learn if gradients are pathological (vanishing or exploding). InTrain requires a gradient health indicator that is **independent of training dynamics and architectural assumptions (e.g., ResNet, BN)**. For each parameter $\theta_i$, it calculates the "variance-to-maximum ratio" of the gradient elements:

$$h(\theta_i)=\min\!\Big(1,\ \frac{\sigma(\nabla_{\theta_i})}{\max(|\nabla_{\theta_i}|)+\epsilon}\Big)$$

where $\sigma(\cdot)$ is the standard deviation of gradient elements. A high ratio indicates a uniform distribution of gradient magnitudes (many elements contributing to stable optimization), while a low ratio suggests concentration in a few components, leading to instability. Capping at 1 prevents outliers from dominating and maintains interpretability. Unlike the multiplicative nature of geometric capacity (serial/bottleneck-constrained), the authors consider optimization resilience as an **additive, parallel** property—each healthy parameter path independently contributes "capacity to receive stable gradient updates," hence aggregated via **summation**:

$$o(A)=\sum_{i=1}^{|\Theta|}h(\theta_i)$$

To compute gradients, the authors use a **synthetic loss**: input $X\sim\mathcal{N}(0,1)$ with random labels, using Cross-Entropy for classification and MSE for spatial outputs (e.g., segmentation). This ensures gradients depend only on the architecture itself, not the data distribution, and guarantees activation of all parameter paths.

**3. Multiplicative Gating: Coupling capacity and resilience into a single score**

Addressing the flaw where addition might yield a moderate score even if one component collapses, the authors propose a **multiplicative gating hypothesis**: capacity and resilience act as gates for each other. Large capacity with resilience $o\approx0$ results in an inability to transmit gradients; high resilience with degraded capacity results in stable optimization of a trivial mapping. Thus, $\gamma\times(1+o)$ is used instead of $\gamma+o$. The final intrinsic trainability is log-normalized by depth:

$$I(A)=\frac{\gamma(A)\times\big(1+o(A)\big)}{\log(L+1)}$$

The term $(1+o)$ ensures positivity and prevents degradation when $o=0$. Using $\log(L+1)$ instead of $L$ for normalization accounts for the fact that both $\gamma$ and $o$ grow with depth, and direct division by $L$ would excessively penalize deep networks. The logarithmic form better fits the reality that information processing capacity grows sub-linearly with depth.

### Loss & Training
InTrain itself **does not involve training**; it only uses a synthetic loss to drive a single backward pass for collecting gradient statistics. Implementation uses 64 synthetic images ($64\times64$) independently sampled pixel-wise from a standard Gaussian as input. For full NAS implementation, the authors embed InTrain as a scoring metric within an evolutionary search framework, resulting in **InTrain-NAS**, with hyperparameters following standard evolutionary NAS configurations.

## Key Experimental Results

### Main Results
Ranking correlation of InTrain versus various zero-cost proxies against true test accuracy on NAS-Bench-101 (KT=Kendall's τ, SPR=Spearman ρ, higher is better):

| Proxy | NAS-Bench-101 KT | NAS-Bench-101 SPR |
|------|------|------|
| Grad_norm | -0.17 | -0.25 |
| SynFlow | 0.20 | 0.29 |
| Zen-score | 0.31 | 0.44 |
| ZiCo | 0.46 | 0.63 |
| FLOPs | 0.31 | 0.44 |
| #Params | 0.31 | 0.43 |
| **Ours (InTrain)** | **0.56** | **0.75** |

KT on NAS-Bench-201 across three datasets (and SPR on ImageNet16-120):

| Proxy | CIFAR-10 KT | CIFAR-100 KT | ImageNet16-120 KT |
|------|------|------|------|
| SynFlow | 0.561 | 0.553 | 0.531 |
| Jacov | 0.616 | 0.639 | 0.602 |
| ZiCo | 0.607 | 0.614 | 0.587 |
| VKDNW_single | 0.618 | 0.634 | 0.622 |
| AZ-NAS (Ensemble) | **0.712** | **0.696** | 0.673 |
| **Ours (InTrain)** | 0.669 | 0.671 | **0.675** |

InTrain is the strongest among all **single-index** proxies; the only higher one, AZ-NAS, is an ensemble method (lacking a unified theory) and is outperformed by InTrain on ImageNet16-120. Top-1 accuracy of InTrain-NAS on ImageNet-1K under various FLOPs budgets:

| FLOPs Budget | Method | Top-1 (%) | Search Cost (GPU Days) |
|------|------|------|------|
| 450M | ZiCo / AZ-NAS | 78.1 / 78.6 | 0.4 |
| 450M | **InTrain-NAS** | **78.9** | 0.4 |
| 600M | AZ-NAS | 79.9 | 0.6 |
| 600M | **InTrain-NAS** | **79.9** | 0.4 |
| 1000M | AZ-NAS | 81.1 | 0.7 |
| 1000M | **InTrain-NAS** | **81.3** | 0.4 |

### Ablation Study
Decomposition of the two components and coupling methods on NAS-Bench-201 (KT / SPR):

| Configuration | Kendall's τ | Spearman ρ | Description |
|------|------|------|------|
| PR-only | 0.61 | 0.82 | Geometric Capacity only |
| Grad-only | 0.63 | 0.83 | Optimization Resilience only |
| PR + Grad (Addition) | 0.59 | 0.80 | Naive addition, worse than single components |
| **InTrain (Multiplicative)** | **0.67** | **0.86** | Full model |

### Key Findings
- **Additive coupling leads to performance degradation**: Summing PR + Grad yields KT=0.59, lower than PR-only (0.61) or Grad-only (0.63). This confirms that naive addition creates interference, supporting the multiplicative gating hypothesis.
- **Dataset Stability**: Most proxies show significant fluctuations in correlation across datasets, whereas InTrain maintains stable ranking across CIFAR-10/100/ImageNet16-120. The authors attribute this to its reliance on intrinsic properties rather than data-dependent heuristics.
- **Both components are effective independently**: PR-only and Grad-only provide meaningful correlations (0.61, 0.63), but multiplicative coupling yields significant synergistic gains (0.67).

## Highlights & Insights
- **Unifying "Trainability" from Scattered Symptoms**: Simultaneously covering "representation geometry" and "optimization dynamics"—two dimensions long handled separately—is the most compelling aspect of this work.
- **Clean Physical Intuition of Multiplicative Gating**: The use of extreme cases (high capacity/zero resilience and vice versa) to justify $\gamma\times(1+o)$ over addition is logically sound and directly validated by ablation studies.
- **Aggregation Logic Distinguishes Serial/Parallel**: Geometric capacity uses log-product (serial bottleneck), while optimization resilience uses summation (parallel paths). This approach of "selecting aggregation operators based on physical properties" can be transferred to other proxy designs.
- **Participation Ratio Interpretation**: Linking PR to the exponential of the second-order Rényi entropy provides an information-theoretic explanation that is more nuanced than simple matrix rank.

## Limitations & Future Work
- **Gap with Ensemble Proxies**: AZ-NAS (ensemble) still achieves higher correlation on CIFAR-10/100; single theoretical proxies have yet to fully dominate in all settings.
- **Dependence on Synthetic Data**: Gradient statistics are derived from synthetic Gaussian inputs and random labels. While the paper emphasizes this isolates architectural effects, whether trainability remains identical under real data distributions remains an open question.
- **Scalability**: The computational cost of covariance eigen-spectra and per-parameter gradient statistics for large-scale models warrants further evaluation.
- **Potential Improvements**: Generalizing multiplicative gating to more than two components or replacing the $\log(L+1)$ term with a learnable depth correction.

## Related Work & Insights
- **vs ZiCo / NASWOT (Gradient-based)**: These only characterize gradient statistics or NTK condition numbers, ignoring representation geometry. InTrain treats gradient health as one of two components and couples it with geometric capacity, resulting in higher and more stable correlations.
- **vs Zen-score / TE-NAS (Activation/Expression-based)**: These evaluate expression diversity but ignore training dynamics. InTrain quantifies geometric capacity via PR and supplements it with optimization resilience for more comprehensive coverage.
- **vs AZ-NAS (Ensemble)**: AZ-NAS achieves high correlation by combining multiple heterogeneous proxies but lacks a unified theory, has lower interpretability, and higher computational overhead. InTrain approaches its performance with a single theoretically grounded proxy and exceeds it on ImageNet16-120.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing trainability as a multiplicative invariant of geometric capacity and optimization resilience is a rare theoretical unification in zero-cost NAS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive results on NAS-Bench-101/201 and MobileNetV2/ImageNet, though some variance information and robustness tables were omitted.
- Writing Quality: ⭐⭐⭐⭐⭐ The derivation chain—three principles → two components → multiplicative coupling—is clear and well-motivated.
- Value: ⭐⭐⭐⭐ Provides a single-index proxy that is fast to compute and stable across datasets, offering practical value for reducing NAS costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Progressive Neural Architecture Generation](progressive_neural_architecture_generation.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[CVPR 2026\] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork](hypernas_enhancing_architecture_representation_for_nas_predictor_via_hypernetwor.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)

</div>

<!-- RELATED:END -->
