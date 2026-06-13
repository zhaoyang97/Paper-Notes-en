---
title: >-
  [Paper Note] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][dimensionality reduction] This paper proposes a variational manifold embedding framework that formalizes dimensionality reduction as an optimization problem over smooth embe…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "dimensionality reduction"
  - "manifold embedding"
  - "variational framework"
  - "PCA"
  - "Euler-Lagrange"
  - "Noether's theorem"
  - "score vector"
date: 2026-05-08
content_hash: 660f73638eaf898c
---

# A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction

**Conference**: NeurIPS 2025
**arXiv**: [2511.22128](https://arxiv.org/abs/2511.22128)  
**Code**: [GitHub](https://github.com/john-vastola/varembed-neurreps25)  
**Authors**: John J. Vastola, Samuel J. Gershman, Kanaka Rajan (Harvard University)
**Area**: Medical Imaging
**Keywords**: dimensionality reduction, manifold embedding, variational framework, PCA, Euler-Lagrange, Noether's theorem, score vector

## TL;DR

This paper proposes a variational manifold embedding framework that formalizes dimensionality reduction as an optimization problem over smooth embedding maps (minimizing the KL divergence between a prior distribution and the pullback of the data distribution), theoretically unifying PCA and nonlinear dimensionality reduction methods, and leverages the calculus of variations (Euler-Lagrange equations) and Noether's theorem to derive interpretable constraints on optimal embeddings.

## Background & Motivation

**Limitations of Prior Work**:

- **Limitations of PCA**: PCA is the most classical dimensionality reduction method — computationally efficient and interpretable — but is inherently linear and unable to capture the nonlinear structure of data manifolds.
- **Poor interpretability of autoencoders**: Neural network-based autoencoders (AE/VAE) are flexible but typically lack interpretability and rigorous theoretical analysis tools.
- **Geometric distortion in graph-based methods**: Methods such as t-SNE and UMAP are nonlinear and relatively intuitive, yet have been observed to produce pathological distortions in downstream statistical and clustering analyses.
- **Disconnect between geometric and probabilistic perspectives**: The geometric view of dimensionality reduction (learning the data manifold structure) and the probabilistic generative modeling view (minimizing distributional divergence) have long lacked a unified theoretical framework.
- **Absence of mathematical analysis tools**: Most existing nonlinear dimensionality reduction methods are algorithmic in nature, lacking the intervention of tools from variational calculus and PDEs, making it difficult to derive analytic properties of solutions.
- **Unexploited symmetry constraints**: Continuous symmetries in data distributions (translation invariance, rotation invariance, etc.) should constrain the form of dimensionality reduction outputs, yet existing methods do not systematically exploit these structures.

## Method

### Overall Architecture

The core idea is to define dimensionality reduction as finding an **optimal smooth embedding map** $\vec{\phi}: \mathbb{R}^d \to \mathbb{R}^D$ (from the low-dimensional latent space to the high-dimensional ambient space) such that the latent prior distribution $q(\vec{z})$ and the data distribution $p_{\text{data}}$ are made as consistent as possible via the pullback of $\vec{\phi}$. The objective functional is:

$$J[\vec{\phi}] = \int_{\mathbb{R}^d} d\vec{z}\, q(\vec{z}) \left[ \frac{1}{2} \log\det(\vec{J}^T \vec{J}) + \log p_{\text{data}}(\vec{\phi}(\vec{z})) - \log q(\vec{z}) \right]$$

This is the **negative KL divergence** $J = -D_{\text{KL}}(q \| \vec{\phi}^* p_{\text{data}})$, upper bounded by zero (perfect embedding).

### Key Design 1: Physical Intuition Behind the Two-Term Objective

The objective functional comprises two complementary optimization signals:

- **Log-likelihood term** $\log p_{\text{data}}(\vec{\phi}(\vec{z}))$: encourages the embedding map to place latent points in high-density regions of the data distribution, analogous to "potential energy" in physics.
- **Log-determinant term** $\frac{1}{2} \log\det(\vec{J}^T \vec{J})$: encourages the embedding to have non-trivial volume (preventing all points from collapsing to a single global maximum), analogous to "kinetic energy" in physics.

This "kinetic + potential energy" structure enables direct application of variational calculus tools.

### Key Design 2: Euler-Lagrange Equations and PDE Constraints

The optimal embedding satisfies the Euler-Lagrange equations, forming a system of **$D$ coupled nonlinear PDEs**:

$$\sum_j \left[ \partial_j(J_{ji}^+) + J_{ji}^+ \partial_j[\log q(\vec{z})] \right] = \frac{\partial \log p_{\text{data}}(\vec{\phi}(\vec{z}))}{\partial \phi_i}$$

where $\vec{J}^+ = (\vec{J}^T \vec{J})^{-1} \vec{J}^T$ is the Moore-Penrose pseudoinverse. For the one-dimensional case ($d=1$), the EL equations reduce to an ODE system, and the optimal trajectory moves along the **score vector** ($\nabla \log p_{\text{data}}$) of the data distribution, with dynamics analogous to a particle undergoing damped motion.

### Key Design 3: Noether's Theorem and Symmetry-Induced Conservation Laws

Noether's theorem is employed to systematically convert continuous symmetries of the data distribution into conserved quantities of the optimal embedding:

- **Reparameterization invariance** → **Conservation of embedding energy**: $\mathcal{E} = -\frac{1}{2}\log\det(\vec{J}^T\vec{J}) - \log p_{\text{data}} + \log q$ is constant.
- **Translation invariance** → **Conservation of momentum**: if $p_{\text{data}}$ is uniform along some direction, the canonical momentum in that direction is conserved.
- **Rotation invariance** → **Conservation of angular momentum**: if $p_{\text{data}}$ is rotationally invariant in some plane, the corresponding angular momentum is conserved (e.g., the optimal embedding of a ring attractor is a circle).

### Loss & Training

The loss function is the negative KL divergence $J[\vec{\phi}]$ (maximized), equivalent to minimizing the KL divergence between the prior $q$ and the pullback distribution $\vec{\phi}^* p_{\text{data}}$. The framework also provides a Bayesian interpretation: maximizing $J$ is equivalent to computing the **maximum a posteriori (MAP) estimate** of the embedding map in the small-noise limit.

## Key Experimental Results

The paper is primarily theoretical; experiments serve as validating demonstrations:

| Experimental Setting | Result |
|---|---|
| 1D embedding + linearly arranged Gaussian mixture | Optimal embedding traverses Gaussian centers along a straight line |
| 1D embedding + circularly arranged Gaussian mixture | Optimal embedding follows a circular arc through the centers |
| 1D embedding + sinusoidally arranged Gaussian mixture | Optimal embedding follows a sinusoidal path through the centers |
| Gaussian prior + Gaussian likelihood (PCA-solvable case) | Linear analytic solution exactly recovers PCA (selects the top $d$ eigenvectors of the covariance matrix) |
| Uniform likelihood + 1D embedding | Optimal embedding $\phi_i(z) = \int_{-\infty}^z q(y)dy$, i.e., a CDF transform (consistent with efficient coding) |

### Key Findings

- **PCA as a special case**: When the prior is an isotropic Gaussian and the likelihood is multivariate Gaussian, the solution to the EL equations exactly corresponds to PCA (selecting the $d$ eigenvectors with the largest eigenvalues). Energy conservation proves the absence of nonlinear solutions.
- **Connection to diffusion models**: Under a specific prior choice ($q(z) = \gamma e^{\gamma z}$, $\gamma \to \infty$), the optimal 1D embedding trajectory exactly follows the score vector field, consistent with the behavior of the probability flow ODE at a fixed noise scale.
- **Symmetry determines embedding form**: The optimal embedding of a rotationally invariant data distribution in 2D ambient space is a circle; a translation-invariant distribution corresponds to a CDF mapping.

## Highlights & Insights

1. **Strong theoretical unification**: PCA, autoencoders, UMAP/t-SNE, and related dimensionality reduction methods are unified within a single variational optimization framework, with PCA exactly recovered as the Gaussian special case.
2. **Novelty in importing physics tools**: The first systematic application of variational calculus (EL equations) and Noether's theorem to the analysis of dimensionality reduction, providing a formal mathematical language for the symmetries of data manifolds.
3. **Score vector as a bridge**: The paper reveals a deep structural connection between optimal dimensionality reduction embeddings and the score function of diffusion models, establishing a theoretical bridge between two seemingly unrelated fields.
4. **Analytically solvable regimes**: Complete analytic proofs are provided for the Gaussian case (including the argument ruling out nonlinear solutions), which is remarkably rare in dimensionality reduction theory.
5. **Bayesian interpretation of the framework**: The objective functional arises naturally from MAP estimation in Bayesian inference (small-noise limit), endowing the framework with statistical grounding.

## Limitations & Future Work

1. **Restricted to continuous distributions**: The framework assumes $q$ and $p_{\text{data}}$ are distributions over continuous spaces and cannot be directly applied to discrete data (e.g., Poisson spike data).
2. **Scalability unverified**: Although the framework is theoretically applicable to data manifolds of arbitrary dimensionality, its scalability to high-dimensional real-world datasets has not been validated.
3. **Diminishing effect of symmetry constraints in high dimensions**: Symmetry-induced conservation laws strongly constrain low-dimensional embeddings but may provide limited constraint in high-dimensional settings.
4. **EL equations generally lack analytic solutions**: Beyond the Gaussian case, the general nonlinear EL system is difficult to solve analytically; numerical optimization still relies on MLP parameterization and gradient descent.
5. **Lack of large-scale empirical evaluation**: The paper does not perform systematic experimental comparisons on real datasets (e.g., single-cell RNA-seq or neural recordings).
6. **Injectivity constraint on the embedding map**: Enforcing the injectivity of $\vec{\phi}$ in practice is difficult to guarantee during optimization and may lead to artifacts such as "wiggling" in the learned embedding.

## Related Work & Insights

| Method Category | Representative Methods | Comparison with Ours |
|---|---|---|
| Linear dimensionality reduction | PCA, probabilistic PCA | The proposed framework exactly recovers PCA under Gaussian assumptions; PCA cannot capture nonlinearity |
| Geometric/graph embedding | LLE, Isomap, Diffusion Maps | Exploit geometric structure but lack a unified variational framework |
| Graph embedding visualization | t-SNE, UMAP | Nonlinear but may introduce geometric distortions; the proposed framework is theoretically more rigorous |
| Deep generative models | VAE, AE | Flexible but lack interpretability; the proposed framework retains variational structure while constraining the embedding form |
| Diffusion models | Score-based models | This work reveals the dynamical connection between optimal embeddings and score vectors |
| Physics-inspired ML | Noether for learning | This work is the first to apply Noether's theorem to symmetry analysis in dimensionality reduction |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Revisiting dimensionality reduction from the perspective of variational calculus and physics; the introduction of Noether's theorem is highly original
- **Experimental Thoroughness**: ⭐⭐ — Primarily theoretical, with only validating numerical demonstrations; lacks systematic experiments on real datasets
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and self-consistent, physical intuition is clearly conveyed, and the presentation progresses systematically from the PCA special case to the general theory
- **Value**: ⭐⭐⭐⭐ — Provides a unified mathematical foundation for dimensionality reduction theory with far-reaching implications for understanding nonlinear dimensionality reduction and diffusion models, though practical applicability remains to be validated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[ICML 2026\] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](../../ICML2026/physics/mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)

</div>

<!-- RELATED:END -->
