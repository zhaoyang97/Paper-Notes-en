---
title: >-
  [Paper Note] Private Model Personalization Revisited
description: >-
  [ICML 2025][AI Safety][Differential Privacy] Proposed the Private FedRep algorithm, which learns a shared low-dimensional embedding $U^* \in \mathbb{R}^{d \times k}$ ($k \ll d$) via an alternating minimization framework under user-level differential privacy (DP) constraints. This reduces the privacy error term by a factor of $\widetilde{O}(dk)$ compared to the prior work by Jain et al., and generalizes to a broader class of sub-Gaussian distributions (rather than being restri…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Differential Privacy"
  - "Model Personalization"
  - "Shared Representation Learning"
  - "Federated Learning"
  - "Johnson-Lindenstrauss"
date: 2026-05-08
content_hash: 1a7014bca5e08d2e
---

# Private Model Personalization Revisited

**Conference**: ICML 2025  
**arXiv**: [2506.19220](https://arxiv.org/abs/2506.19220)  
**Code**: To be confirmed  
**Area**: AI Safety  
**Keywords**: Differential Privacy, Model Personalization, Shared Representation Learning, Federated Learning, Johnson-Lindenstrauss

## TL;DR

Proposed the Private FedRep algorithm, which learns a shared low-dimensional embedding $U^* \in \mathbb{R}^{d \times k}$ ($k \ll d$) via an alternating minimization framework under user-level differential privacy (DP) constraints. This reduces the privacy error term by a factor of $\widetilde{O}(dk)$ compared to the prior work by Jain et al., and generalizes to a broader class of sub-Gaussian distributions (rather than being restricted to Gaussian distributions). Additionally, dimension-free classification risk bounds are provided using the Johnson-Lindenstrauss transform.

## Background & Motivation

**Background**: Model personalization is a core strategy for addressing the statistical heterogeneity of user data. In the shared representation learning framework, $n$ users collaboratively learn a shared low-dimensional embedding $U^* \in \mathbb{R}^{d \times k}$ ($k \ll d$), and each user subsequently trains their own local model $v_i \in \mathbb{R}^k$ such that $w_i^* = U^* v_i^*$. FedRep (Collins et al., 2021) is a representative non-private algorithm within this framework.

**Limitations of Prior Work**: Jain et al. (2021) proposed the first model personalization algorithm with privacy guarantees, but it suffers from three key limitations: (1) it requires user data to follow standard Gaussian distributions, which is an overly strong assumption; (2) it requires centralized processing—the embedding update involves joint optimization over all user data, making it difficult to adapt into a federated algorithm; (3) the privacy error term contains high-dimensional dependencies of $\widetilde{O}(d^3 k^2)$, leading to severe utility loss in high-dimensional scenarios.

**Key Challenge**: The trade-off between privacy protection and personalization utility is particularly acute in high-dimensional settings—the higher the embedding dimension $d$, the more DP noise needs to be added. Existing methods lack tight noise calibration, resulting in an excessively high privacy cost.

**Goal**: (1) Design a private model personalization algorithm that is natively federated; (2) provide tighter utility guarantees under more relaxed data distribution assumptions (sub-Gaussian); (3) derive classification risk bounds that are independent of the input dimension $d$.

**Key Insight**: The authors observe that the alternating minimization structure of FedRep is naturally suited for federated settings—users update local vectors $v_i$, while the server aggregates embedding gradients. The key innovation is using disjoint data batches to update local vectors and compute embedding gradients respectively, thereby tightening the upper bound on the Frobenius norm of the gradient and reducing the required amount of DP noise.

**Core Idea**: Through a federated alternating minimization framework with disjoint batches + QR orthonormalization + clipped aggregation, a privacy-utility trade-off that is $\widetilde{O}(dk)$ times tighter than the SOTA is achieved under sub-Gaussian assumptions.

## Method

### Overall Architecture

The input consists of datasets $S_i = \{(x_{i,j}, y_{i,j})\}_{j=1}^m$ for $n$ users, and the output consists of the shared embedding matrix $U^{\text{priv}} \in \mathbb{R}^{d \times k}$ (with orthogonal columns) and the local vector $v_i^{\text{priv}} \in \mathbb{R}^k$ for each user. The algorithm consists of three stages: (1) private initialization (Algorithm 2) to obtain the initial embedding $U_{\text{init}}$; (2) Private FedRep iterations (Algorithm 1) to alternately update local vectors and embeddings; (3) each user independently solving their own final local model.

### Key Designs

1. **Alternating Minimization with Disjoint Batches**:

    - **Function**: In each iteration, each user samples two disjoint batches $B_{i,t}$ and $B'_{i,t}$ from their data, which are used to update the local vector $v_{i,t}$ and compute the embedding gradient $\nabla_{i,t}$, respectively.
    - **Mechanism**: The user first solves $v_{i,t} = \arg\min_v \hat{L}(U_t, v; B_{i,t})$ on $B_{i,t}$, and then computes the gradient $\nabla_{i,t} = \nabla_U \hat{L}(U_t, v_{i,t}; B'_{i,t})$ on the independent batch $B'_{i,t}$. Since the two batches are independent, the upper bound on the gradient norm is tighter: $\|\nabla_{i,t}\|_F \leq \widetilde{O}((R+\Gamma)\Gamma\sqrt{dk})$.
    - **Design Motivation**: The original FedRep reuses the same batch, introducing dependencies between the local vector and the gradient, which yields a looser high-probability upper bound on the gradient norm. Disjoint batches break this dependency, directly reducing the amount of DP noise injected.

2. **Clipped Aggregation + QR Orthonormalization**:

    - **Function**: After the server receives the gradients from each user, it clips them within a $\psi$-norm ball, aggregates them, adds Gaussian noise $\xi_{t+1} \sim \mathcal{N}^{d \times k}(0, \hat{\sigma}^2)$, and then re-orthonormalizes the updated embedding via QR decomposition.
    - **Mechanism**: The embedding is updated as $\hat{U}_{t+1} = U_t - \eta(\frac{1}{n}\sum_i \text{clip}(\nabla_{i,t}, \psi) + \xi_{t+1})$, followed by $U_{t+1}, P_{t+1} = \text{QR}(\hat{U}_{t+1})$. The noise standard deviation is configured as $\hat{\sigma} = C\psi\sqrt{T\log(1/\delta)}/(n\epsilon)$.
    - **Design Motivation**: QR orthonormalization ensures that $U_t$ always has orthogonal columns. This property prevents noise from propagating and accumulating across iterations—since $\|U_t\|_2 = 1$ always holds, the upper bound of the gradient norm in each round is unaffected by historical noise.

3. **Private Initialization (Based on Subspace Recovery)**:

    - **Function**: Based on the subspace recovery estimator from Duchi et al., the second-moment matrix is estimated using the U-statistic $Z_i = \frac{2}{m(m-1)}\sum_{j_1 \neq j_2} y_{i,j_1} y_{i,j_2} x_{i,j_1} x_{i,j_2}^\top$, and the initial embedding is obtained via top-$k$ SVD after adding noise.
    - **Mechanism**: The expectation of this U-statistic is $U^* V^{*\top} V^* U^{*\top}$ (plus noise and identity matrix contributions), whose top-$k$ eigenvectors span the column space of $U^*$. Gaussian noise is injected to guarantee DP after clipping at $\psi_{\text{init}} = \widetilde{O}((R^2 + \Gamma^2)d)$.
    - **Design Motivation**: The convergence of Algorithm 1 requires the principal angle distance between the initial embedding $U_0$ and the true $U^*$ to be strictly less than $\sqrt{1-c}$. A random initialization in high-dimensional space is almost orthogonal to $U^*$, failing to satisfy this condition.

### Loss & Training

For the regression setting, the quadratic loss $\ell(U, v, (x,y)) = (y - \langle x, Uv \rangle)^2$ is used. The label generation model is $y = x^\top U^* v_i^* + \zeta$, where $\zeta$ is $R$-sub-Gaussian noise. The learning rate is set to $\eta = 1/(2\Lambda^2)$, the number of iterations to $T = \Theta(\Lambda^2 \log(n^3)/\lambda^2)$, and the batch size to $b = \lfloor m/(2T) \rfloor$. The entire algorithm achieves user-level DP via the billboard model: the server broadcasts $U^{\text{priv}}$, and each user independently solves their local model.

## Key Experimental Results

### Main Results

| Method | Privacy Error Term | Data Assumption | Federated Support |
|------|-----------|---------|---------|
| Jain et al. (2021) | $\widetilde{O}(d^3 k^2 / (n^2 \epsilon^2 \sigma_{\min,*}^4))$ | Standard Gaussian | No (Centralized) |
| **Private FedRep (Ours)** | $\widetilde{O}(d^2 k / (n^2 \epsilon^2 \sigma_{\min,*}^4))$ | sub-Gaussian | **Yes** |
| **Gain** | $\widetilde{O}(dk)$ | More relaxed | — |

### Ablation Study

| Configuration | Key Excess Risk Term | Description |
|------|-------------------|------|
| Full Private FedRep | $\widetilde{O}(d^2k/(n^2\epsilon^2\sigma_{\min,*}^4) + d/(nm\sigma_{\min,*}^4)) + k/m$ | Privacy Term + Statistical Term + Local Term |
| Removing Disjoint Batches | Gradient norm upper bound loosens by $\widetilde{O}(dk)$ | Requires more noise |
| Removing QR Orthonormalization | Noise accumulates across iterations | Unstable convergence |
| Classification Scenario (JL Transform) | $d$-independent margin-based risk bound | Via dimensionality reduction to $O(\log n/\epsilon^2)$ |

### Key Findings

- **The primary improvement in privacy error stems from disjoint batches**: Tightening the upper bound of the gradient Frobenius norm from $\widetilde{O}(d^{3/2}k)$ to the scale of $\widetilde{O}(\sqrt{dk})$ (eliminating the statistical dependency between $v$ and the gradient) directly reduces the clipping threshold $\psi$ and the required noise standard deviation $\hat{\sigma}$.
- **Extension to sub-Gaussian distributions is non-trivial**: Under non-Gaussian distributions, geometric contraction of distance no longer holds with high probability in every round (due to weaker concentration of the covariance matrix). The authors employ an induction-based argument to handle this intermittent non-contraction.
- **Dimension-free classification results rely on the JL transform**: By randomly projecting the $d$-dimensional space into $O(\log n)$ dimensions, a risk bound completely independent of $d$ is achieved under margin loss, at the cost of requiring a bounded norm assumption.

## Highlights & Insights

- **The ingenuity of disjoint batches**: This is a seemingly minor yet highly effective modification—simply splitting a batch to use the components independently eliminates the key statistical dependency, yielding an $\widetilde{O}(dk)$-fold improvement in utility. This "decoupling" philosophy could be highly useful in other scenarios involving private gradient clipping.
- **QR orthonormalization to prevent noise propagation**: In private federated optimization, the accumulation of noise across iterations is a core challenge. By projecting back onto the Stiefel manifold (the set of orthogonal matrices) at each step, the impact of noise is elegantly restricted to the current iteration. This trick is worth borrowing in other low-rank optimization problems.
- **Practicality of the billboard model**: Compared to schemes requiring secure aggregation, the billboard model allows the server to publicly broadcast the results, enabling each user to train local models independently. It is simple to implement and does not leak local vectors.

## Limitations & Future Work

- **Analysis restricted to linear regression and binary classification**: Practical federated personalization scenarios typically involve deep networks and non-convex optimization; the applicability of these theoretical results requires further validation.
- **Dependence on $\sigma_{\min,*}$**: The $\sigma_{\min,*}^{-4}$ dependence in the excess risk implies that utility can degenerate dramatically when user diversity is low (i.e., the condition number of $V^*$ is large). This could be a bottleneck in scenarios where user tasks are highly homogeneous.
- **Initialization requires additional privacy budget**: The initialization step in Algorithm 2 consumes a privacy budget of $(\epsilon/2, \delta/2)$, halving the budget left for the main algorithm. Exploring initialization-free schemes or shared-budget schemes is a valuable direction.
- **Lack of empirical experiments**: The entire work consists of purely theoretical results, without validation of the algorithm's practical performance on real-world datasets. Experiments on synthetic data or federated benchmarks like FEMNIST would greatly strengthen the work.

## Related Work & Insights

- **vs Jain et al. (2021)**: Both works focus on DP personalization within the shared representation learning framework. However, Jain's algorithm requires centralized exact minimization (embedding updates require access to all users' data), has a privacy error term that is $\widetilde{O}(dk)$ times larger, and is restricted to Gaussian distributions. This work completely outperforms it via federated gradients combined with disjoint batches.
- **vs Collins et al. (2021) FedRep**: This work is a privatized version of FedRep. In addition to integrating DP mechanisms, it extends the framework to the non-realizable case with label noise. The original analysis of FedRep was only applicable to noiseless labels.
- **vs JL-based private learning**: Prior works like Bassily et al. (2022) utilize the JL transform for dimensionality reduction to achieve private learning. This work introduces this technique into the classification setting of the federated personalization scenario, obtaining dimension-independent risk guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of decoupling via disjoint batches and JL-based classification stands as a meaningful technical innovation.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work, completely lacking empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rigorous theorem statements and clearly defined positioning of technical contributions.
- Value: ⭐⭐⭐⭐ Advances the theoretical frontier of private federated personalization, though its practical impact remains to be validated empirically.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2025\] Faster Rates for Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)

</div>

<!-- RELATED:END -->
