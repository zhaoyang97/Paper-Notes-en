---
title: >-
  [Paper Note] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift
description: >-
  [CVPR2026][Optimization][federated learning] This paper proposes the Fed-ADE framework, which adaptively adjusts the learning rate for each client at each time step using two lightweight distribution shift signals — unce…
tags:
  - "CVPR2026"
  - "Optimization"
  - "federated learning"
  - "distribution shift"
  - "adaptive learning rate"
  - "online adaptation"
  - "unsupervised adaptation"
date: 2026-05-08
content_hash: f3437d7bb5e7d461
---

# Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift

**Conference**: CVPR2026
**arXiv**: [2603.01040](https://arxiv.org/abs/2603.01040)  
**Code**: [h2w1/Fed-ADE](https://github.com/h2w1/Fed-ADE)  
**Area**: Optimization
**Keywords**: federated learning, distribution shift, adaptive learning rate, online adaptation, unsupervised adaptation

## TL;DR

This paper proposes the Fed-ADE framework, which adaptively adjusts the learning rate for each client at each time step using two lightweight distribution shift signals — uncertainty dynamics estimation and representation dynamics estimation — enabling unsupervised post-deployment adaptation in federated settings.

## Background & Motivation

**Ubiquity of post-deployment distribution shift**: Edge devices (smartphones, IoT, autonomous vehicles) continuously receive non-stationary data streams, causing pretrained models to degrade rapidly under distribution shift.

**Dual heterogeneity challenge**: Federated learning simultaneously faces shift heterogeneity (each client experiences distribution shift with different temporal dynamics) and data heterogeneity (clients' local data differ inherently in scale and domain).

**Fatal flaw of fixed learning rates**: A learning rate that is too small leads to underfitting, while one that is too large causes divergence; applying a single fixed learning rate across hundreds of heterogeneous clients cannot accommodate their varying rates of distribution shift.

**Unsupervised constraint**: Ground-truth labels are unavailable after deployment, making learning rate selection even more challenging and rendering conventional loss-based scheduling strategies inapplicable.

**Limitations of existing methods**: Methods such as Fed-POE rely on multi-model ensembles or costly hyperparameter search, introducing additional communication and computation overhead; centralized methods (ATLAS, FTH) cannot exploit shared knowledge across clients.

**Core problem**: How to automatically select appropriate learning rates for each client in a federated setting with unlabeled, heterogeneous, and time-varying distribution shift?

## Method

### Overall Architecture

Fed-ADE adopts a partial-sharing personalized federated learning architecture: each client model $\theta_c$ is partitioned into shared layers $\psi_c$ (communicated with the server) and personalized layers $\phi_c$ (retained locally). In each communication round: (1) the client updates all parameters using the adaptive learning rate $\eta_c^t$; (2) it uploads $\psi_c$ to the server for weighted aggregation; (3) it receives the aggregated $\bar{\psi}$, freezes the shared layers, and updates only the personalized layers using the adaptive learning rate. The core innovation lies in the adaptive computation of the learning rate:

$$\eta_c^t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \mathcal{S}_c^t$$

where $\mathcal{S}_c^t \in [0,1]$ is a composite distribution shift signal formed by combining two complementary estimators: $\mathcal{S}_c^t = \frac{1}{2}(\mathcal{S}_{\text{unc}}^t + \mathcal{S}_{\text{rep}}^t)$.

### Key Design 1: Uncertainty Dynamics Estimation

- **Function**: Captures label distribution shift by tracking temporal changes in model prediction uncertainty.
- **Mechanism**: Computes the mean softmax vector over the current batch $\mathbf{q}_c^t = \frac{1}{|\mathbf{x}_c^t|} \sum_{x} \mathcal{H}(\theta_c; x)$, then measures the change between consecutive time steps via cosine distance: $\mathcal{S}_{\text{unc}}^t = 1 - \cos(\mathbf{q}_c^{t-1}, \mathbf{q}_c^t)$.
- **Design Motivation**: The mean softmax vector serves as an unlabeled class-level confidence summary and acts as an entropy proxy; batch-level averaging eliminates single-sample randomness; only the previous step's $\mathbf{q}_c^{t-1}$ needs to be cached, with memory overhead of only $O(|\mathcal{I}|)$ (number of classes).

### Key Design 2: Representation Dynamics Estimation

- **Function**: Captures feature-level covariate shift in the embedding space.
- **Mechanism**: Computes the batch-averaged feature vector after $\ell_2$ normalization $\mathbf{z}_c^t = \frac{1}{|\mathbf{x}_c^t|} \sum_x \frac{h_{\psi_c}(x)}{\|h_{\psi_c}(x)\|_2}$, then applies a scaled cosine distance: $\mathcal{S}_{\text{rep}}^t = \frac{1}{2}(1 - \cos(\mathbf{z}_c^{t-1}, \mathbf{z}_c^t))$.
- **Design Motivation**: $\ell_2$ normalization ensures that the cosine distance reflects only directional change rather than scale differences; the $\frac{1}{2}$ scaling normalizes the range from $[0,2]$ to $[0,1]$, aligning it with the uncertainty signal; the estimator is entirely label-free and computed locally, with memory overhead of only $O(d)$ (feature dimension).

### Key Design 3: Unsupervised Risk Estimation

- **Function**: Estimates the expected risk for each client without labels, serving as the optimization objective for model updates.
- **Mechanism**: Employs Black-box Shift Estimation (BBSE), using a confusion matrix $\mathbf{M}$ precomputed on the server and the client's pseudo-label distribution $\mathbf{Q}_{c,\hat{y}}^t$ to estimate the current label distribution: $\mathbf{Q}_{c,y}^t \approx \mathbf{M}^{-1} \mathbf{Q}_{c,\hat{y}}^t$, yielding the unsupervised risk estimate $\widehat{\mathcal{F}}_c^t(\theta_c)$.
- **Design Motivation**: The supervised risk is decomposed into a weighted sum of class-level sub-risks, where the weights (label distributions) can be estimated unsupervisedly via BBSE, and the initial sub-risks can be substituted with empirical estimates from pretraining data.

### Key Design 4: Theoretical Guarantee (Dynamic Regret Bound)

- **Function**: Proves that the adaptive learning rate of Fed-ADE achieves min-max optimal dynamic regret in non-stationary environments.
- **Core Result**: With the choice $\eta^* = \Theta(T^{-1/3} \bar{\mathcal{S}}_c^{1/3})$, the dynamic regret satisfies $\mathbb{E}[\text{Reg}_T] = \mathcal{O}(\bar{\mathcal{S}}_c^{1/3} T^{2/3})$, matching the min-max optimal bound for online learning under unsupervised label shift.
- **Design Motivation**: Theoretical analysis (Theorems 1 & 2) shows that the cumulative shift proxy $\bar{\mathcal{S}}_c$ accurately approximates the true distribution shift, providing a theoretical foundation for the adaptive learning rate.

## Key Experimental Results

### Table 1: Label Shift Setting (Mean Accuracy %)

| Dataset | Shift Type | FTH | ATLAS | Fed-POE | FedCCFA | FixLR(Mid) | **Fed-ADE** |
|--------|---------|-----|-------|---------|---------|------------|-------------|
| Tiny ImageNet | Lin. | 78.2 | 76.5 | 87.1 | 84.7 | 88.2 | **89.1** |
| Tiny ImageNet | Sin. | 77.9 | 76.8 | 87.5 | 84.8 | 88.0 | **88.9** |
| CIFAR-10 | Lin. | 31.4 | 36.5 | 71.3 | 65.8 | 70.8 | **73.8** |
| CIFAR-10 | Sin. | 40.3 | 43.7 | 71.4 | 65.8 | 70.5 | **73.6** |
| LAMA | Lin. | 68.3 | 79.5 | 85.4 | 95.6 | 95.2 | **95.8** |
| LAMA | Squ. | 70.5 | 79.8 | 84.2 | 92.0 | 95.4 | **96.4** |

**Key Takeaway**: Fed-ADE achieves the best performance across all label shift settings; it improves over FixLR by approximately 1–3% on average and over Fed-POE by approximately 2–4%.

### Table 2: Covariate Shift Setting + Ablation Study

| Dataset | Shift Type | FTH | ATLAS | Fed-POE | FixLR(Mid) | **Fed-ADE** |
|--------|---------|-----|-------|---------|------------|-------------|
| CIFAR-10-C | Lin. | 23.7 | 13.9 | 44.5 | 63.9 | **64.4** |
| CIFAR-10-C | Squ. | 23.8 | 14.1 | 48.5 | 64.5 | **65.4** |
| CIFAR-100-C | Lin. | 9.2 | 3.5 | 27.3 | 43.4 | **45.8** |
| CIFAR-100-C | Sin. | 7.6 | 2.9 | 27.5 | 42.1 | **46.7** |

**Ablation** (CIFAR-10 Lin.): Fed-ADE (full) 73.8% > w/o $\mathcal{S}_{\text{unc}}$ 71.3% > w/o $\mathcal{S}_{\text{rep}}$ 73.1% > Fixed LR 70.8%. The two signals are complementary: $\mathcal{S}_{\text{unc}}$ is more sensitive to label shift, while $\mathcal{S}_{\text{rep}}$ is more sensitive to covariate shift.

**Computational Efficiency**: Fed-ADE achieves a mean wall time of approximately 109 seconds, which is 17–24× faster than localized methods and approximately 2× faster than FedCCFA.

## Highlights & Insights

1. **Extremely lightweight design**: The two shift estimators require only caching the mean vector from the previous step ($O(|\mathcal{I}|) + O(d)$), with no additional communication, no labels, and no multi-model ensembles.
2. **Unification of theory and practice**: The paper proves that $\mathcal{S}_c^t$ can approximate the true distribution shift and derives a min-max optimal dynamic regret of $\mathcal{O}(\bar{\mathcal{S}}_c^{1/3} T^{2/3})$, which is relatively rare in the federated unsupervised adaptation literature.
3. **Superiority of cosine similarity**: Ablation experiments show that cosine similarity outperforms KL divergence, Wasserstein distance, and Bayesian CPD, owing to its bounded and direction-based nature, which confers greater robustness to pseudo-label noise and class imbalance.
4. **Cross-modal generalization**: The method demonstrates strong performance on both image benchmarks (Tiny ImageNet, CIFAR-10/100, CIFAR-C) and text benchmarks (LAMA), confirming that the approach is not modality-specific.
5. **Robustness to pretraining distribution**: Fed-ADE maintains stable performance when the pretraining data follows Gaussian or exponentially decaying distributions rather than a uniform distribution.

## Limitations & Future Work

1. **Equal-weight combination of two signals**: The simple average $\mathcal{S}_c^t = \frac{1}{2}(\mathcal{S}_{\text{unc}}^t + \mathcal{S}_{\text{rep}}^t)$ does not adaptively weight the signals according to the current shift type, and may underperform relative to an attention-based automatic weighting scheme.
2. **Only label shift and covariate shift are evaluated**: More complex shift types such as concept drift (changes in $P(y|x)$) and prior probability shift are not addressed.
3. **$\eta_{\min}$ and $\eta_{\max}$ still require manual specification**: Although the paper demonstrates low sensitivity to these hyperparameters, different boundary values are used for image and text benchmarks, and no automatic selection strategy is provided.
4. **Fixed at 100 clients**: The behavior and convergence rate in large-scale (1000+) or very small client settings remain unexplored.
5. **Dependence on the BBSE confusion matrix**: Pretraining data are required to compute $\mathbf{M}$; if such data are unavailable or poorly representative, the quality of the estimation may be compromised.

## Related Work & Insights

- **Comparison with ATLAS/FTH**: These localized methods perform extremely poorly in federated settings (only 30–40% on CIFAR-10), demonstrating the critical importance of cross-client knowledge sharing.
- **Comparison with Fed-POE**: Fed-POE improves adaptability through ensemble strategies but incurs high computational cost and lacks awareness of shift magnitude; Fed-ADE more efficiently controls the learning rate directly via lightweight shift signals.
- **Implications for TTA / continual learning**: The dual-signal paradigm of using uncertainty and representation signals for shift detection is transferable to test-time adaptation and continual learning settings.
- **A new paradigm for adaptive learning rates**: Unlike gradient-statistics-based adaptive methods such as Adam, Fed-ADE adjusts the learning rate based on signals derived from data distribution changes, representing a novel design direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of driving adaptive learning rates with two complementary lightweight shift signals is novel, and the exploration of unsupervised post-deployment adaptation in federated settings is valuable; however, the core techniques (cosine distance, BBSE) are combinations of existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 5 datasets, 2 shift types, 4 temporal schedules, and multiple ablation groups (similarity metrics, estimators, pretraining distributions, learning rate boundaries), making it highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, notation is consistent, and the theoretical sections are rigorous; however, the high information density of Table 1 slightly reduces readability.
- **Value**: ⭐⭐⭐⭐ — Addresses the practical pain point of learning rate selection in federated post-deployment adaptation; the method is lightweight, efficient, and theoretically grounded, offering clear reference value to the federated learning and online adaptation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](../../ICLR2026/optimization/a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[NeurIPS 2025\] Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable](../../NeurIPS2025/optimization/exact_and_linear_convergence_for_federated_learning_under_arbitrary_client_parti.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
