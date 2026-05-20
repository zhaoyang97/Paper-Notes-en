---
title: >-
  [Paper Note] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization
description: >-
  [NeurIPS 2025][Optimization][minimax optimization] This paper proposes Ada-Minimax and Ada-BiO, two adaptive algorithms that combine momentum normalization with a novel online noise estimation strategy to achieve…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "minimax optimization"
  - "bilevel optimization"
  - "adaptive algorithm"
  - "momentum normalization"
  - "convergence rate"
date: 2026-05-08
content_hash: f35d752373879cd4
---

# Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2509.15399](https://arxiv.org/abs/2509.15399)  
**Code**: Available (submitted with paper appendix)  
**Area**: Optimization Theory / Hierarchical Optimization
**Keywords**: minimax optimization, bilevel optimization, adaptive algorithm, momentum normalization, convergence rate

## TL;DR
This paper proposes Ada-Minimax and Ada-BiO, two adaptive algorithms that combine momentum normalization with a novel online noise estimation strategy to achieve, for the first time, sharp convergence rates of $\tilde{O}(1/\sqrt{T} + \sqrt{\bar{\sigma}}/T^{1/4})$ for nonconvex-strongly-concave minimax and nonconvex-strongly-convex bilevel optimization without requiring prior knowledge of the gradient noise level.

## Background & Motivation

**Background**: Hierarchical optimization (minimax and bilevel) is a central problem formulation in machine learning—adversarial learning and AUC maximization are modeled as minimax problems, while meta-learning and hyperparameter optimization are modeled as bilevel problems. A large number of algorithms have been proposed with established convergence guarantees.

**Limitations of Prior Work**: All existing methods require prior knowledge of the stochastic gradient noise magnitude $\bar{\sigma}$ to set hyperparameters such as step sizes. When the noise level is unknown or varies during training, these methods cannot adapt automatically—conservative step sizes in low-noise regimes lead to slow convergence, while excessively large step sizes in high-noise regimes cause divergence.

**Key Challenge**: Standard AdaGrad-style adaptive methods have already resolved the noise-adaptive problem in single-level optimization, but directly applying them to hierarchical optimization introduces complex stochastic dependency issues—the AdaGrad step sizes for upper- and lower-level variables are mutually coupled, making analysis intractable.

**Key Insight**: Instead of standard AdaGrad, the paper employs normalized SGD with momentum (NSGDM) for the upper-level variable and AdaGrad-Norm for the lower-level variable, with the key innovation being that the momentum parameter itself is also adaptively adjusted.

## Method

### Overall Architecture
Ada-Minimax (Algorithm 1) and Ada-BiO (Algorithm 2) share the same algorithmic framework: the upper-level variable is updated via normalized SGD with momentum ($x_{t+1} = x_t - \eta_{x,t} \cdot m_t / \|m_t\|$), while the lower-level variable is updated via AdaGrad-Norm gradient descent/ascent. The core distinction between the two algorithms lies solely in the gradient estimation scheme—Ada-Minimax computes stochastic gradients directly, whereas Ada-BiO approximates the hypergradient using a Neumann series.

### Key Designs

1. **Adaptive Momentum Normalization (Adaptive NSGDM)**:

    - **Function**: Updates the upper-level variable with automatic adaptation to the noise level.
    - **Mechanism**: Momentum update $m_t = \beta_t \cdot m_{t-1} + (1-\beta_t) \cdot g_{x,t}$, followed by the normalization step $x_{t+1} = x_t - \eta_{x,t} \cdot m_t / \|m_t\|$. The key is that the momentum parameter $\alpha_t = 1 - \beta_t$ is adaptively set as $\alpha_t = \alpha / \sqrt{\alpha^2 + \sum \|g_{x,k} - \tilde{g}_{x,k}\|^2}$, where the difference terms in the denominator provide an online estimate of the gradient variance.
    - **Design Motivation**: Gradient normalization decouples the step size from gradient magnitude; the momentum parameter is dynamically adjusted according to the estimated noise level—smaller $\alpha_t$ (more momentum smoothing) under high noise, and larger $\alpha_t$ (faster response) under low noise.

2. **Coordinated Adaptive Step Sizes for Upper and Lower Levels**:

    - **Function**: Simultaneously sets adaptive learning rates for both upper- and lower-level variables.
    - **Mechanism**: Upper-level step size $\eta_{x,t} = \eta_x \cdot \alpha'_t / t$, where $\alpha'_t = \alpha / \sqrt{\alpha^2 + \sum \|g_{x,k} - \tilde{g}_{x,k}\|^2 + \|g_{y,k}\|^2}$ incorporates both upper- and lower-level information; lower-level step size $\eta_{y,t} = \eta_y / \sqrt{\gamma^2 + \sum \|g_{y,k}\|^2}$ follows the standard AdaGrad-Norm form.
    - **Design Motivation**: $\alpha'_t$ jointly encodes gradient information from both levels to effectively control the ratio $\eta_{x,t}/\eta_{y,t}$, ensuring coordinated update speeds across levels (key to Lemma 5.7).

3. **Double-Sampling Noise Estimation**:

    - **Function**: Online estimation of the gradient noise level.
    - **Mechanism**: Two independent stochastic gradients $g_{x,t}$ and $\tilde{g}_{x,t}$ are drawn at each step; $\|g_{x,t} - \tilde{g}_{x,t}\|^2$ serves as an unbiased proxy for the variance. The cumulative sum $\sum \|g_{x,k} - \tilde{g}_{x,k}\|^2$ is shown to be bounded within $[\bar{\sigma}^2 t,\, 4\bar{\sigma}^2 t]$ with high probability (Lemma 5.5).
    - **Design Motivation**: Direct variance estimation requires knowledge of the true gradient, whereas the double-sampling difference requires only one additional gradient query to obtain an online variance estimate.

### Loss & Training
Ada-Minimax targets $\min_x \max_y f(x, y)$ (with $f$ strongly concave in $y$); Ada-BiO targets $\min_x f(x, y^*(x))$ s.t. $y^*(x) = \operatorname{argmin}_y g(x, y)$ (with $g$ strongly convex in $y$). Ada-BiO approximates the hypergradient $\nabla \Phi(x)$ via a Neumann series, requiring additional second-order information queries.

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | Ada-Minimax/BiO | TiAda/TFBO | SGDA/PDSM |
|------|---------|--------|-----------------|------------|-----------|
| Synthetic minimax ($\sigma=0$) | 1-D function | $\|\nabla\Phi\|$ | **Fastest convergence** | Comparable | Slower |
| Synthetic minimax ($\sigma=100$) | 1-D function | $\|\nabla\Phi\|$ | **Converges** | Diverges | Diverges |
| Deep AUC maximization | Sentiment140 | Train AUC | **+20% over PDSM** | Low | Suboptimal |
| Deep AUC maximization | Sentiment140 | Test AUC | **+2% over PDSM** | Low | Suboptimal |
| Hyperparameter optimization | TREC | Train/Test ACC | **Fast convergence** | TFBO diverges | — |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| $\alpha \in [0.1, 2.0]$ | AUC nearly unchanged | Robust to initial momentum parameter |
| $\eta_y \in [0.001, 0.05]$ | Early-stage variation; consistent final performance | Robust lower-level learning rate |
| $\eta_x \in [0.001, 0.01]$ | Affects early training dynamics | Final performance degrades at $\eta_x=0.05$ |
| $\gamma \in [0.01, 2.0]$ | High consistency | Robust initialization parameter |

### Key Findings
- Ada-Minimax converges under high noise ($\sigma=100$), whereas TiAda fails to converge even with extensive tuning.
- On Deep AUC maximization, Ada-Minimax achieves the fastest convergence both in terms of epochs and wall-clock time.
- Strong hyperparameter robustness: varying the four main hyperparameters ($\alpha$, $\eta_x$, $\eta_y$, $\gamma$) over wide ranges has minimal impact on final performance.
- Experiments validate the noise lower-bound assumption: empirical noise on the AUC task spans $[0.21, 210.71]$.

## Highlights & Insights
- **Theoretical novelty**: This is the first work to provide adaptive and sharp convergence guarantees for stochastic hierarchical optimization, achieving automatic rate interpolation without prior noise knowledge.
- **General analysis framework**: The paper first establishes an analysis framework for single-level adaptive NSGDM (Theorem 5.6), then extends it to minimax and bilevel settings—a clear and transferable proof strategy.
- **Hyperparameter robustness**: The four main hyperparameters exhibit minimal performance sensitivity over orders-of-magnitude variation, substantially reducing tuning burden.
- **Practical variant design**: Replacing the theoretical double-sampling difference with adjacent-iterate gradient differences (more computationally efficient) does not degrade practical performance.

## Limitations & Future Work
- Requires the assumption that stochastic gradient noise has a nonzero lower bound (Assumption 3.2(ii)); although the noiseless case is technically covered, this assumption is unnatural in certain settings.
- Only the nonconvex-strongly-concave/strongly-convex setting is considered; the more general nonconvex-nonconvex case remains to be explored.
- The contribution is primarily theoretical; deep learning experiments are limited in scale (single dataset with simple models).
- Each step requires two independent gradient samples (though the practical variant approximates this with adjacent-iterate gradients).

## Related Work & Insights
- **TiAda**: The only prior adaptive minimax method, but its convergence rate is not sharp with respect to noise (no explicit dependence on $\bar{\sigma}$).
- **AdaGrad-Norm**: The adaptive foundation for single-level optimization; this paper extends it to the lower level of hierarchical optimization.
- **Cutkosky & Mehta (2020)**: The original NSGDM method, but with a fixed momentum parameter; the key innovation in this paper is making the momentum parameter also adaptive.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The dual guarantee of adaptivity and sharpness in hierarchical optimization is a first; the combination of momentum normalization with adaptive parameters is original.
- **Experimental Thoroughness**: ⭐⭐⭐ — Synthetic experiments clearly validate the theory, but deep learning experiments cover only one dataset at limited scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and complete; the proof structure is well-organized and hierarchical.
- **Value**: ⭐⭐⭐⭐ — An important contribution to the optimization theory community; the hyperparameter robustness of the practical variant is a valuable reference for practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings](sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)

</div>

<!-- RELATED:END -->
