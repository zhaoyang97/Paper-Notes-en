---
title: >-
  [Paper Note] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes
description: >-
  [NeurIPS 2025][Optimization][Neural Tangent Kernel] This paper establishes explicit quantitative upper bounds on the convergence of gradient-descent-trained shallow neural networks to Gaussian processes at any positive t…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Neural Tangent Kernel"
  - "Gaussian Process"
  - "Wasserstein Distance"
  - "Finite Width"
  - "Infinite-Width Limit"
date: 2026-05-08
content_hash: e005f5a98f1158e1
---

# Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes

**Conference**: NeurIPS 2025
**arXiv**: [2509.24544](https://arxiv.org/abs/2509.24544)  
**Code**: None  
**Area**: Optimization Theory / Neural Network Theory
**Keywords**: Neural Tangent Kernel, Gaussian Process, Wasserstein Distance, Finite Width, Infinite-Width Limit

## TL;DR

This paper establishes explicit quantitative upper bounds on the convergence of gradient-descent-trained shallow neural networks to Gaussian processes at any positive training time $t \geq 0$, proving that the squared 2-Wasserstein distance decays polynomially at rate $O(\log n_1 / n_1)$.

## Background & Motivation

The theoretical understanding of deep learning has been a central research direction. In the overparameterized regime, Neal (1996) and de G. Matthews et al. (2018) showed that network outputs converge to a Gaussian process as width tends to infinity, when parameters are initialized from a Gaussian distribution. The **Neural Tangent Kernel (NTK)** framework introduced by Jacot et al. (2018) further characterizes the training dynamics of infinitely wide networks under gradient descent: the network evolves approximately linearly around initialization, and training can be understood as kernel regression with a fixed kernel.

**Core Problem**: The practical relevance of NTK analysis depends on its approximation accuracy at finite width. Although Lee et al. (2020) established qualitative convergence under the NTK framework, **rigorous quantitative results providing explicit finite-width error bounds remain extremely scarce**.

This gap gives rise to two practical issues: (1) the inability to quantify the discrepancy between finite-width network predictions and their infinite-width NTK counterparts; (2) the inability to reveal how network width, depth, initialization, and training hyperparameters affect the validity of the linear approximation.

Prior works such as Basteri & Trevisan (2024) and Favaro et al. (2025) provide quantitative convergence rates only **at initialization**. The central contribution of this paper is to **extend these results to arbitrary positive training times**.

## Method

### Overall Architecture

Consider a single-hidden-layer (shallow) fully connected neural network:
$$f(x; \theta) = \frac{1}{\sqrt{n_1}} \Phi\left(\frac{1}{\sqrt{n_0}} x \theta^{(0)}\right) \theta^{(1)}$$

where $n_0$ is the input dimension, $n_1$ is the hidden layer width, $\theta^{(0)} \in \mathbb{R}^{n_0 \times n_1}$, and $\theta^{(1)} \in \mathbb{R}^{n_1}$.

For the associated Gaussian process $G_t(x)$, the mean and covariance are determined by the analytic NTK $k_\infty$:
$$\mu_t(x) = k_\infty(x, \mathcal{X}) I_t(k_\infty) y$$
$$\Sigma_t(x, x') = \mathcal{K}(x,x') - \text{(training correction term)}$$

The goal is to quantify the decay behavior of $\mathcal{W}_2^2(f(x; \theta_t), G_t(x))$ with respect to $n_1$.

### Key Designs

1. **Triangle Inequality Decomposition**: The total Wasserstein distance is decomposed into two terms:
    $$\mathcal{W}_2(f(x;\theta_t), G_t(x)) \leq \underbrace{\mathcal{W}_2(f, f^{\text{lin}})}_{\text{nonlinearity error}} + \underbrace{\mathcal{W}_2(f^{\text{lin}}, G_t)}_{\text{CLT error}}$$
    - First term: controls the deviation between the true network and its linearized version.
    - Second term: controls the convergence of the linearized network (CLT) to the Gaussian process.

2. **Linearized Network Analysis**: Define the linearized network $f^{\text{lin}}(x;\theta_t) = f(x;\theta_0) + \nabla_\theta f(x;\theta_0)|_{\theta_0} \omega_t$, whose gradient flow equation admits a closed-form solution:
    $$f^{\text{lin}}(x;\bar{\theta}_t) = f(x;\theta_0) - k_{x\mathcal{X}} I_t(k_{\mathcal{X}\mathcal{X}})(f(x;\theta_0) - y)$$
   where $I_t(B) = (\mathbb{1}_n - e^{-Bt})B^{-1}$ is an auxiliary operator.

3. **Good-Event Decomposition**: The parameter space is partitioned into a "good event" $S$ (satisfying key properties such as NTK condition number bounds) and its complement $S^C$. On $S$, Proposition B.9 is used to control the $L^2$ distance between $f$ and $f^{\text{lin}}$; on $S^C$, Lemma B.12 controls the tail event. The $t^8$ term arises from controlling the tail event.

### Main Theorem

**Theorem 3.4**: Under Assumptions 1–4, for each test point $x \in \mathbb{R}^{n_0}$, there exist positive constants $a_1, a_2$ independent of $n_0, n_1, t$ such that:

$$\mathcal{W}_2^2(f(x;\theta_t), G_t(x)) \leq r\left(\frac{a_1 \log n_1}{(\lambda_{\min}^\infty)^3 n_1 n_0} + \frac{a_2 n_0}{(\lambda_{\min}^\infty)^r n_1^{r/4}}(1 + t^8)\right)$$

Key properties:
- For fixed $t$, the leading term is $O(\log n_1 / n_1)$, i.e., polynomial decay.
- As long as $t$ grows polynomially in $n_1$, choosing sufficiently large $r$ makes the right-hand side tend to zero.
- $\lambda_{\min}^\infty$ is the minimum eigenvalue of the analytic NTK; positive definiteness is required (a mild assumption).

### Assumptions

- Assumption 1: Gaussian initialization.
- Assumption 2: Analytic NTK is positive definite (holds when training data are in general position and $\Phi$ is non-polynomial).
- Assumption 3: $\Phi$ and $\Phi'$ are Lipschitz continuous and bounded (satisfied by sigmoid, tanh, Gaussian, etc.; ReLU does not satisfy this, though the authors expect the conclusions still hold).
- Assumption 4: Sufficient overparameterization condition (the left-hand side tends to zero as $\min\{n_0, n_1\}$ grows).

## Key Experimental Results

### Numerical Validation

| Network Width $n_1$ | $\mathcal{W}_2^2$ (empirical) | Theoretical Bound Trend | Remark |
|---------------------|-------------------------------|-------------------------|--------|
| Small → Large | Monotonically decreasing | $O(\log n_1/n_1)$ | Convergence rate consistent with theoretical prediction |

### Validation Experiment Design

| Experimental Setting | Description | Verified Conclusion |
|----------------------|-------------|---------------------|
| Varying $n_1$ | Fixed $n_0$, vary hidden layer width | $\mathcal{W}_2^2$ decays polynomially with $n_1$ |
| Varying training time $t$ | Fixed width, vary training steps | Error remains bounded over polynomial time |
| Varying activation functions | sigmoid, tanh, etc. | All activations satisfying Assumption 3 pass validation |

### Key Findings

- Quantitative convergence rates hold throughout training, not only at initialization.
- The convergence rate with respect to $n_1$ is nearly optimal (close to the known $n^{-1}$ optimal rate, up to a logarithmic factor).
- The error bound is maintained on time scales where $t$ grows polynomially in $n_1$, covering practical training durations.

## Highlights & Insights

- This is the first work to provide explicit Wasserstein-2 quantitative convergence rates for shallow networks during training, bridging the theoretical gap between initialization and the training process.
- The paper reveals the precise interplay among network width, input dimension, minimum eigenvalue of the NTK, and training time.
- The good-event/bad-event decomposition and the elegant definition of the auxiliary operator $I_t(B)$ gracefully handle the invertibility of random matrices.

## Limitations & Future Work

- Applicable only to shallow (single-hidden-layer) networks; extension to deep networks is an important but challenging direction.
- Not applicable to ReLU activations (Lipschitz continuity is required), although the authors expect the conclusions to still hold.
- The $t^8$ term stems from a rough control of tail events; refining tail estimates may improve the time dependence.
- In practical deep networks, the NTK does not remain constant (a limitation of the lazy training assumption).

## Related Work & Insights

- **vs. Basteri & Trevisan (2024)**: The latter provides quantitative rates only at initialization; this paper extends the result to arbitrary positive training times.
- **vs. Lee et al. (2020)**: The latter establishes qualitative convergence under the NTK framework; this paper provides quantitative finite-width error bounds.
- **vs. Bordino et al. (2025)**: The latter obtains suboptimal convergence rates via second-order Poincaré inequalities; this paper achieves a sharper rate in the shallow network setting.
- **vs. de G. Matthews et al. (2018)**: The latter treats more general deep networks but under a weaker convergence metric (the $\rho_F$ metric); this paper establishes convergence under the stronger $\mathcal{W}_2$ metric for shallow networks.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending quantitative results from initialization to training is a non-trivial contribution, though it primarily represents a refinement of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐ As a theoretical work, numerical validation is present but limited in scale; no verification on large-scale practical networks.
- Writing Quality: ⭐⭐⭐⭐ Theoretical statements are rigorous, notation is consistent, and the proof roadmap is clear.
- Value: ⭐⭐⭐⭐ Adds an important quantitative tool to NTK theory, contributing meaningfully to the understanding of the theoretical properties of finite-width networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](../../ICML2026/optimization/transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)

</div>

<!-- RELATED:END -->
