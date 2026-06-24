---
title: >-
  [Paper Note] Universal Neural Optimal Transport
description: >-
  [ICML2025][Physics & Scientific Computing][Optimal Transport] This work proposes Universal Neural Optimal Transport (UNOT), which utilizes Fourier Neural Operators to learn entropy-regularized optimal transport dual potentials across datasets and resolutions, achieving up to a 7.4× initialization speedup for the Sinkhorn algorithm.
tags:
  - "ICML2025"
  - "Physics & Scientific Computing"
  - "Optimal Transport"
  - "Fourier Neural Operator"
  - "Sinkhorn Algorithm"
  - "Dual Potential"
  - "Adversarial Training"
  - "Bootstrap Loss"
date: 2026-05-08
content_hash: 20efef7017fe19d0
---

# Universal Neural Optimal Transport

**Conference**: ICML2025  
**arXiv**: [2212.00133](https://arxiv.org/abs/2212.00133)  
**Code**: [GregorKornhardt/UNOT](https://github.com/GregorKornhardt/UNOT)  
**Area**: Optimal Transport  
**Keywords**: Optimal Transport, Fourier Neural Operator, Sinkhorn Algorithm, Dual Potential, Adversarial Training, Bootstrap Loss

## TL;DR

This work proposes Universal Neural Optimal Transport (UNOT), which utilizes Fourier Neural Operators to learn entropy-regularized optimal transport dual potentials across datasets and resolutions, achieving up to a 7.4× initialization speedup for the Sinkhorn algorithm.

## Background & Motivation

Optimal Transport (OT) is a fundamental tool in machine learning, widely used in domain adaptation, single-cell genomics, image processing, flow matching, and other scenarios. Given probability measures $\mu, \nu$ and a cost function $c$, the entropy-regularized OT problem is defined as:

$$\text{OT}_\epsilon(\mu,\nu) = \inf_{\pi \in \Pi(\mu,\nu)} \int c \, d\pi - \epsilon \text{KL}(\pi \| \mu \otimes \nu)$$

The Sinkhorn algorithm can solve this problem iteratively, but its computational cost is heavy, posing a significant bottleneck especially in scenarios where OT needs to be solved repeatedly. Existing acceleration schemes include:

- **Gaussian initialization** (Thornton & Cuturi, 2022): Initializing Sinkhorn with the solution of a Gaussian OT problem.
- **Meta OT** (Amos et al., 2023): Training a neural network to predict transport plans, but restricted to fixed dimensions.

Neither can handle variable resolution inputs, nor can they generalize across datasets. The goal of UNOT is to build a **universal** neural OT solver that directly predicts the dual potentials given any pair of discrete measures.

## Method

### Mechanism: Predicting Dual Potentials

Instead of directly predicting the $m \times n$ dimensional transport plan matrix $\Pi$, UNOT predicts the $n$-dimensional dual potential $\boldsymbol{g}$. According to dual OT theory, the optimal transport plan can be recovered from the dual potentials:

$$\Pi = \text{diag}(\boldsymbol{u}) K \text{diag}(\boldsymbol{v}), \quad K = \exp(-C/\epsilon)$$

where $(\boldsymbol{u}, \boldsymbol{v}) = (\exp(\boldsymbol{f}/\epsilon), \exp(\boldsymbol{g}/\epsilon))$. Given $\boldsymbol{g}$, the other potential can be recovered with a single Sinkhorn iteration step: $\boldsymbol{u} = \boldsymbol{\mu} ./ K\boldsymbol{v}$, reducing the problem's dimensionality to $n$.

### Network Architecture: Fourier Neural Operator

The theoretical justification for choosing FNO is the **convergence of discrete dual potentials** (Proposition 2): when discrete measures $(\mu_n, \nu_n) \to (\mu, \nu)$, the corresponding dual potentials converge uniformly. This implies that discrete measures and their potentials can be viewed as discretizations of continuous functions, which naturally matches the discretization-invariance of FNO.

The network $S_\phi$ takes a pair of measures $(\boldsymbol{\mu}, \boldsymbol{\nu})$ as input and outputs the dual potential $\boldsymbol{g}$:

$$S_\phi(\boldsymbol{\mu}, \boldsymbol{\nu}) = \boldsymbol{g}$$

FNO is implemented with $L$ Fourier neural operator layers: each layer performs a discrete Fourier transform on the input, retains a fixed number of low-frequency features, applies a complex linear transform, and then transforms back to the spatial domain. SϕS_\phi has 26M parameters in total. For spherical cost functions, Spherical FNO (SFNO) is used instead.

### Adversarial Training Generator

The generator $G_\theta$ generates training distribution pairs from Gaussian noise $\boldsymbol{z} \sim \mathcal{N}(0, I)$:

$$G_\theta(\boldsymbol{z}) = R[\text{ReLU}(\text{NN}_\theta(\boldsymbol{z}) + \lambda I_{d,d'}(\boldsymbol{z})) + \delta]$$

where $R$ is a normalization and random downsampling operator, $\lambda$ is the residual connection coefficient, and $\delta > 0$ ensures positive density.

**Theoretical Guarantee** (Theorem 3): When $\text{Lip}(\text{NN}_\theta) < \lambda$, $G_\theta$ yields positive density on all non-negative vectors, meaning it can generate any pair of discrete probability measures. This conclusion generalizes to general residual network compositions (Corollary 4).

### Bootstrap Loss Function

Directly using converged Sinkhorn solutions as supervision is computationally prohibitive. UNOT uses a **bootstrap loss**: starting from the network prediction $\boldsymbol{g}_\phi$ as initialization, it runs only $k=5$ steps of Sinkhorn to obtain $\boldsymbol{g}_{\tau_k}$, and then minimizes the distance between the two:

$$\mathcal{L} = \|\boldsymbol{g}_\phi - \boldsymbol{g}_{\tau_k}\|_2^2$$

**Theoretical Guarantee** (Proposition 5): Minimizing the bootstrap loss is equivalent to minimizing the distance to the true potential function, bounded from above by $c(K,k,n) \cdot \|\boldsymbol{g}_\phi - \boldsymbol{g}_{\tau_k}\|_2^2$.

### Full Training Objective

Formulated as an adversarial game, the solver $S_\phi$ minimizes while the generator $G_\theta$ maximizes:

$$\max_\theta \min_\phi \mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{g}_{\tau_k} - S_\phi(G_\theta(\boldsymbol{z}))\|_2^2]$$

## Key Experimental Results

### Experimental Settings

- Cost functions: $\|x-y\|^2$ (squared Euclidean), $\|x-y\|$ (Euclidean), $\arccos(\langle x,y\rangle)$ (spherical)
- Training samples: 200M, resolutions from $10\times10$ to $64\times64$, $\epsilon=0.01$
- Training time: ~35h (H100 GPU)
- Test sets: MNIST (28²), CIFAR10 (28²), LFW (64²), Bears (64²), and cross-dataset evaluation

### Sinkhorn Iterations (To reaching 1% relative error, $c=\|x-y\|^2$)

| Dataset | UNOT | Ones Initialization | Gaussian Initialization |
|--------|------|-----------|----------------|
| MNIST | **3±5** | 16±9 | 10±7 |
| CIFAR | **3±6** | 80±22 | 52±19 |
| LFW | **7±8** | 78±20 | 35±14 |
| Bear | **4±6** | 41±16 | 25±13 |
| LFW-Bear | **4±6** | 53±18 | 29±13 |

### Actual Speedup ($c=\|x-y\|^2$, wall-clock time to reach 1% error)

| Dataset | UNOT (s) | Ones (s) | Speedup |
|--------|----------|----------|--------|
| CIFAR | 9.5e-4 | 7.1e-3 | **7.4×** |
| LFW | 3.0e-3 | 1.5e-2 | **5.0×** |
| Bear | 2.6e-3 | 1.0e-2 | **3.8×** |
| LFW-Bear | 2.7e-3 | 1.2e-2 | **4.4×** |

UNOT significantly reduces the number of Sinkhorn iterations required for convergence across all datasets, achieving up to 7.4× actual speedup on CIFAR.

## Highlights & Insights

- **First universal neural OT solver across datasets and resolutions**: Perfectly combines the discretization invariance of FNO with the convergence theory of dual potentials.
- **Solid theoretical foundation**: Rigorous proofs are provided for the universal approximation property of the generator (Theorem 3), the correctness of the bootstrap loss (Proposition 5), and the discrete-continuous potential convergence (Proposition 2).
- **Clever training paradigm**: The adversarial bootstrap training avoids expensive computations of ground-truth Sinkhorn labels, requiring only $k=5$ steps to provide effective gradient signals.
- **Preserves favorable properties of Sinkhorn**: The output of UNOT can be directly used as initialization for Sinkhorn, remaining parallelizable and differentiable.
- **Support for non-Euclidean geometry**: Seamlessly extends to spherical OT problems using SFNO.

## Limitations & Future Work

- Needs to train a separate model for each cost function $c$ (with a fixed $\epsilon=0.01$). Generalizing to different $\epsilon$ or different $c$ requires additional training.
- High training cost (200M samples, 35h on H100), leading to substantial up-front deployment investment.
- Currently only validated in grid-based discretization scenarios (image-like), whereas the performance on unstructured point clouds remains under-explored.
- The capability of FNO to capture high-frequency potential functions may be limited by the number of retained Fourier modes.
- The speedup on simpler datasets such as MNIST is relatively small (1.25×), with the main gains concentrated on complex distributions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The OT solver framework combining FNO and adversarial bootstrapping is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The experiments covering multiple cost functions, multiple datasets, generalizations, and ablations are relatively comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — The structure is clear across theory, methodology, and experiments, with rigorous proofs.
- Value: ⭐⭐⭐⭐ — Offers practical acceleration value for scenarios requiring large-scale, repetitive OT computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/physics/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](../../NeurIPS2025/physics/feat_free_energy_estimators_with_adaptive_transport.md)
- [\[ICLR 2026\] Tucker-FNO: Tensor Tucker-Fourier Neural Operator and its Universal Approximation Theory](../../ICLR2026/physics/tucker-fno_tensor_tucker-fourier_neural_operator_and_its_universal_approximation.md)
- [\[NeurIPS 2025\] FlashMD: Long-Stride, Universal Prediction of Molecular Dynamics](../../NeurIPS2025/physics/flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/physics/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)

</div>

<!-- RELATED:END -->
