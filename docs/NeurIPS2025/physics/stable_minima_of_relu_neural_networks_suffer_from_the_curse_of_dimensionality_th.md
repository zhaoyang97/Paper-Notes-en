---
title: >-
  [Paper Note] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][stable minima] This paper investigates the generalization properties of stable minima (flat minima) in two-layer overparameterized ReLU networks. It proves that while flatne…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "stable minima"
  - "ReLU networks"
  - "curse of dimensionality"
  - "implicit bias"
  - "nonparametric estimation"
date: 2026-05-08
content_hash: 53b44455c1cf6485
---

# Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon

**Conference**: NeurIPS 2025
**arXiv**: [2506.20779](https://arxiv.org/abs/2506.20779)  
**Code**: None  
**Area**: Scientific Computing
**Keywords**: stable minima, ReLU networks, curse of dimensionality, implicit bias, nonparametric estimation

## TL;DR
This paper investigates the generalization properties of stable minima (flat minima) in two-layer overparameterized ReLU networks. It proves that while flatness does imply generalization, the convergence rate deteriorates exponentially with input dimension (i.e., the curse of dimensionality applies), forming an exponential separation from low-norm solutions (weight decay) that are immune to this curse. The paper further identifies the "neural shattering" phenomenon as the geometric mechanism underlying failure in high dimensions.

## Background & Motivation
**Background**: Overparameterized models in deep learning admit infinitely many global minima, yet gradient descent (GD) appears to find well-generalizing solutions. A large body of work studies the implicit bias of GD through the lens of dynamical stability — stable minima are those to which GD can "stably converge," characterized by bounded loss curvature (i.e., the maximum eigenvalue of the Hessian $\leq 2/\eta$).

**Limitations of Prior Work**: Existing work (Mulayoff et al., 2021; Qiao et al., 2024) either assumes interpolation or restricts to univariate inputs. Qiao et al. (2024) derive favorable risk bounds in the univariate setting, but only for intervals strictly inside the data support and without addressing the multivariate or high-dimensional regime.

**Key Challenge**: Flat minima behave well in low dimensions ($d=1$), but their behavior in high dimensions is entirely unknown. Intuitively, stability/flatness should imply some regularity that aids generalization — but is this regularity sufficient in high dimensions?

**Goal**: To precisely answer the fundamental question of how stable minima of two-layer overparameterized ReLU networks generalize in the high-dimensional non-interpolation regime.

**Key Insight**: Establish a function-space characterization of stable minima via a weighted variation norm, derive upper and lower bounds on both the generalization gap and MSE, and reveal the geometric mechanism through a novel minimax lower bound construction based on spherical coverings.

**Core Idea**: Exponentially many "directional caps" (spherical caps) exist on the high-dimensional sphere, allowing ReLU neurons to activate on only a tiny fraction of data points while maintaining large weights, thereby "fooling" the flatness criterion — this is neural shattering, which causes stable minima to inevitably suffer from the curse of dimensionality in high dimensions.

## Method

### Overall Architecture
This is a theoretical analysis paper. The core contributions are:
1. Establishing a function-space characterization of stable minima via a weighted variation space $V_g$
2. Deriving upper and lower bounds on the generalization gap (Theorem 3.5)
3. Deriving upper and lower bounds on nonparametric estimation MSE (Theorems 3.6, 3.7)
4. Empirically validating the curse of dimensionality and the neural shattering phenomenon

### Key Designs
1. **Weighted variation (semi-)norm**: A data-dependent weight function $g(\mathbf{u}, t)$ is defined, incorporating the probability that data exceeds threshold $t$ in direction $\mathbf{u}$, the conditional expected distance, and the conditional expected position. When inputs are uniformly distributed on the unit ball, $g(\mathbf{u}, t) \asymp (1-|t|)^{d+2}$, meaning the weight decays sharply near the boundary. This implies that neurons with large weights but activations confined near the boundary incur low cost under this norm.
2. **Stability implies regularity (Theorem 3.2)**: For any two-layer ReLU network, $|f_\theta|_{V_g} \leq \frac{\lambda_{max}(\nabla^2 \mathcal{L}(\theta))}{2} - \frac{1}{2} + (R+1)\sqrt{2\mathcal{L}(\theta)}$. Combined with the stability condition (Proposition 2.1), stable minima necessarily belong to a bounded subset of $V_g$.
3. **Generalization gap upper bound (Theorem 3.5)**: For stable minima, the generalization gap converges to zero at rate $n^{-1/(2d+4)}$. While generalization does occur, the rate deteriorates exponentially with dimension $d$.
4. **Generalization gap lower bound (Theorem 3.5)**: An unavoidable lower bound of $n^{-2/(d+1)}$ confirms that the curse of dimensionality is an intrinsic property, not an artifact of the analysis.
5. **MSE upper bound (Theorem 3.6)**: The MSE of optimized stable minima is upper bounded by $\tilde{O}(n^{-1/(2d+4)})$.
6. **Minimax lower bound (Theorem 3.7)**: For $d > 1$, the minimax MSE over $V_g$ is lower bounded by $\Omega(n^{-2/(d+1)})$ for any estimator; for $d = 1$, the lower bound is $\Omega(n^{-1/2})$.

### Core Construction (Theoretical Basis for Neural Shattering)
The key to the lower bound proof is a novel packing argument:
- Pack $M = \exp(\Omega(d))$ mutually disjoint spherical caps on the $(d-1)$-dimensional unit sphere $\mathbb{S}^{d-1}$
- Each cap corresponds to a ReLU neuron $\varphi_i(\mathbf{x}) = c \cdot \phi(\mathbf{u}_i^T \mathbf{x} - t)$, activated only in the directions covered by that cap
- Due to the sharp boundary decay of the weight function $g$, these high-weight neurons near the boundary are "cheap" under the variation norm constraint
- This constructs exponentially many "hard-to-learn" functions, yielding the minimax lower bound

### Loss & Training
- Theoretical analysis is based on the squared loss $\mathcal{L}(\theta) = \frac{1}{2n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2$
- Stability condition: $\lambda_{max}(\nabla^2_\theta \mathcal{L}(\theta)) \leq 2/\eta$ (equivalent to dynamical stability of GD)
- Experiments use standard GD with Kaiming initialization and gradient clipping threshold of 50

## Key Experimental Results

### Main Results: Validation of the Curse of Dimensionality

| Dimension $d$ | GD (large lr $\eta=0.2$) log-MSE slope | Weight Decay ($\eta=0.01$, $\lambda=0.1$) log-MSE slope |
|---|---|---|
| 1 | ~−0.35 | >−0.50 |
| 2 | ~−0.25 | >−0.50 |
| 3 | ~−0.18 | >−0.50 |
| 4 | ~−0.13 | >−0.50 |
| 5 | ~−0.10 | >−0.50 |

The MSE convergence slope (in log-log scale) under GD training drops sharply with dimension, declining from approximately −0.35 at $d=1$ to approximately −0.10 at $d=5$, corroborating the theoretically predicted curse of dimensionality. Weight decay training maintains slopes above −0.5 across all dimensions, remaining unaffected by dimension.

### Ablation Study: Neural Shattering Phenomenon

| Training Setting | Mean Activation Rate per Neuron | Weight Norm | Training MSE |
|---|---|---|---|
| GD ($\eta=0.9$, no WD) | ≤ 10% | Large | ≈ 1.105 |
| GD ($\eta=0.01$, WD=0.1) | High (nearly all activated) | Small and bounded | ≈ 0.055 |

A network of width 2048 is trained on noisy samples from 512 linear targets in 10 dimensions. After entering the edge-of-stability regime under large learning rate GD, the maximum Hessian eigenvalue stabilizes near $2/\eta \approx 2.2$, yet each neuron activates on fewer than 10% of data points while exhibiting large weight norms — this is precisely neural shattering.

### Key Findings
1. The generalization performance of stable minima (flat minima) does deteriorate with dimension, with the MSE convergence rate declining from $n^{-1/6}$ at $d=1$ to near zero at $d=5$
2. Low-norm solutions (weight decay) are immune to the curse of dimensionality, consistently achieving convergence slopes ≥ −0.5
3. Neural shattering is an inherent phenomenon in high dimensions: GD satisfies the flatness constraint by making neurons sparsely activated rather than by shrinking weights
4. The shattering patterns observed experimentally align precisely with the "hard-to-learn" functions constructed in the lower bound proof

## Highlights & Insights
1. **Flatness ≠ a universal key to generalization**: It has long been believed that flat minima necessarily generalize well. This paper provides the first theoretical proof of their fundamental limitations in high dimensions, establishing an exponential separation between flat solutions and low-norm solutions.
2. **Discovery of neural shattering**: A previously unrecognized phenomenon is revealed — in high dimensions, ReLU neurons can "fool" the flatness criterion by sparsely activating across exponentially many directions near the sphere boundary, constituting the geometric root of the curse of dimensionality.
3. **Data-dependent function-space characterization**: The weighted variation space $V_g$ precisely characterizes the inductive bias of stable minima; the boundary decay behavior of the weight function, $(1-|t|)^{d+2}$, is the key quantity.
4. **Consistency between theory and experiment**: The geometric intuition underlying the lower bound construction (spherical packing → sparse activation) is perfectly validated experimentally.
5. **Global domain analysis**: Unlike Qiao et al. (2024), whose analysis is confined to interior intervals, this paper covers the full input domain and characterizes the extrapolation behavior of the network.

## Limitations & Future Work
1. Only two-layer ReLU networks are analyzed; the behavior of stable minima in deeper architectures remains unknown.
2. The input distribution is assumed to be uniform on the unit ball; the weight function $g$ becomes more complex under more general distributions.
3. A gap remains between the upper bound ($n^{-1/(2d+4)}$) and the lower bound ($n^{-2/(d+1)}$); the tight rate has yet to be determined.
4. The effects of practical training techniques such as SGD, batch normalization, and others are not considered.
5. Adaptive learning rate optimizers (e.g., Adam) are not addressed, and these may alter the stability characteristics.
6. Whether the curse of dimensionality can be mitigated through structured priors (e.g., low-dimensional manifold assumptions) warrants further investigation.

## Related Work & Insights
- **Mulayoff et al. (2021)**: First established a function-space characterization of stable minima in the univariate interpolation setting; the present work extends this to high-dimensional non-interpolation regimes.
- **Qiao et al. (2024)**: Derived generalization and risk bounds in the univariate non-interpolation setting (restricted to interior intervals); the present work generalizes to high dimensions and the full domain.
- **Nacson et al. (2023)**: Characterized stable minima in the multivariate interpolation setting; the present work removes the interpolation assumption.
- **Bach (2017)**: Proved that low-norm (weight decay) solutions are immune to the curse of dimensionality; the present work establishes an exponential separation from flat solutions.
- **Edge-of-stability (Cohen et al., 2020)**: Empirically observed that Hessian eigenvalues oscillate near $2/\eta$ during GD training; the stability definition adopted in this paper directly corresponds to this phenomenon.
- **Implication**: Relying solely on implicit regularization (flatness) may be insufficient; explicit regularization (e.g., weight decay) may be necessary for high-dimensional problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof of the curse of dimensionality for flat minima + discovery and theoretical explanation of neural shattering
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments thoroughly validate theoretical predictions, but validation on real data and deeper networks is lacking
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically rigorous, clearly motivated, with geometric intuition and formal proofs seamlessly integrated
- Value: ⭐⭐⭐⭐⭐ Makes a foundational contribution to understanding generalization in deep learning, challenging the prevailing view that "flat minima = good generalization"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)

</div>

<!-- RELATED:END -->
