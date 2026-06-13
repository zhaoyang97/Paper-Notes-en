---
title: >-
  [Paper Note] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks
description: >-
  [ICLR 2026][Optimization][loss landscape] This paper reveals that systematic growth of curvature along low-loss paths generates entropic barriers, such that even when the energy path is flat…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "loss landscape"
  - "mode connectivity"
  - "entropic force"
  - "SGD dynamics"
  - "overparameterization"
date: 2026-05-08
content_hash: 7e1920ccca725182
---

# Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2512.06297](https://arxiv.org/abs/2512.06297)  
**Code**: None (authors commit to release upon de-anonymization)  
**Area**: Deep Learning Theory / Optimization
**Keywords**: loss landscape, mode connectivity, entropic force, SGD dynamics, overparameterization

## TL;DR
This paper reveals that systematic growth of curvature along low-loss paths generates entropic barriers, such that even when the energy path is flat, SGD noise confines optimization dynamics to flat regions near minima—resolving the paradox of "mode-connected but dynamically isolated" solutions.

## Background & Motivation

**Background**: Different minima of overparameterized neural networks can be connected via low-loss paths (mode connectivity), yet SGD training rarely explores intermediate points along these paths; once convergence to a minimum is achieved, the optimizer remains there.

**Limitations of Prior Work**: Mode connectivity implies the loss landscape is not rugged and that flat paths connect minima—yet optimizers exhibit "confined" behavior, constituting an apparent paradox that existing theory cannot adequately explain.

**Key Challenge**: Focusing solely on loss values (energy) neglects implicit forces arising from curvature variation—analogous to entropic forces in statistical physics—which bias noisy optimization dynamics toward flatter regions.

**Goal**: Why are energetically connected minima dynamically disconnected? How does curvature vary along low-loss paths? How do entropic and energy barriers evolve relative to each other during training?

**Key Insight**: Drawing on the effective potential of Brownian motion in statistical physics, the work treats SGD noise as an effective temperature and analyzes how curvature variation constrains optimization trajectories through entropic forces.

**Core Idea**: Systematic curvature increase along low-loss connecting paths produces entropic barriers that confine noisy optimization dynamics near minima, even when the energy path is entirely flat.

## Method

### Overall Architecture
Train multiple Wide ResNet / ResNet models to obtain distinct minima → apply AutoNEB to find the minimum energy path (MEP) between minima → measure curvature along the path (Hessian trace, maximum eigenvalue, Fisher matrix spectrum) → design projected SGD experiments to verify the existence and magnitude of entropic forces → analyze the persistence of entropic barriers during training via linear mode connectivity experiments.

### Key Designs

1. **Theoretical Framework for Entropic Forces**:

    - Function: Establish a mathematical model in which curvature variation produces an effective potential.
    - Mechanism: Consider a potential $V(x,y) = \frac{1}{2}g(y)x^2$, where $g(y)$ is the curvature function along the "soft" direction. After integrating out the fast variable $x$, the effective potential for the slow variable $y$ becomes $V_{\text{eff}}(y) = T \ln g(y)$, yielding a force proportional to $-\frac{d}{dy}\ln g(y)$ that drives the system toward regions of smaller $g(y)$ (i.e., flatter regions). The effective temperature is $T \propto \eta / B$ (learning rate / batch size).
    - Design Motivation: In neural networks, SGD noise acts as an effective temperature and curvature variation plays the role of $g(y)$, thereby explaining why optimizers prefer flat minima and remain confined near them.

2. **Curvature Measurement Along the MEP**:

    - Function: Systematically measure the Hessian spectrum along the minimum energy path using three complementary methods.
    - Mechanism: (a) Power iteration to estimate the maximum Hessian eigenvalue $\lambda_{\max}$, requiring only $\mathcal{O}(N)$ Hessian-vector products; (b) the Fisher information matrix $\mathcal{F}(\theta^*) = \mathbb{E}[s_\theta s_\theta^\top]$ to approximate the Hessian at minima and efficiently estimate the trace; (c) SVD of the score matrix on a subset of training data to estimate the leading eigenvalues. All three methods consistently show that curvature at the path midpoint is substantially higher than at the endpoints.
    - Design Motivation: A single measurement may be biased; the consistency of three independent methods strengthens the conclusions. In particular, the full-spectrum SVD analysis reveals that curvature increases across all directions, not merely in isolated ones.

3. **Projected SGD Experiments**:

    - Function: Constrain SGD updates to the MEP or linear path and directly observe entropic force effects.
    - Mechanism: Every $k$ SGD steps, parameters are projected onto the nearest segment of the path ($k=15$), where $k$ controls the trade-off between entropic force and path constraint. Experiments show that models initialized at the path midpoint are systematically pushed toward the endpoints, even when the loss increases in that direction; smaller batch sizes and larger learning rates accelerate relaxation, consistent with the prediction $T \propto \eta/B$.
    - Design Motivation: This eliminates the confound of the model leaving the path along other directions, isolating the entropic force effect induced by curvature variation.

### Loss & Training
Standard SGD with momentum 0.9 and weight decay $5 \times 10^{-4}$, learning rate 0.1, 200 training epochs, batch size 256, with learning rate divided by 5 at 30%/60%/80%/90% of training. AutoNEB uses 4 refinement cycles with learning rate annealed from 0.1 to $10^{-3}$. Projected SGD uses $\eta=0.02$ and $B=16$ as baseline.

## Key Experimental Results

### Main Results

| Experimental Setting | Metric | Result |
|---|---|---|
| WRN-16-4 MEP (multiple pairs of minima) | Hessian trace along path | Lowest at endpoints; systematic 2–3× increase at midpoint |
| WRN-16-4 MEP | $\lambda_{\max}$ along path | Midpoint approximately 2× higher than endpoints |
| WRN-16-4 MEP | Full SVD spectrum | Entire spectrum shifts upward as path interior is traversed |
| Projected SGD ($B=16$, $\eta=0.02$) | Relaxation time vs. initial position | Relaxation to endpoint is slower when initialized deeper into the path interior |

### Ablation Study

| Configuration | Relaxation Behavior | Notes |
|---|---|---|
| Vanilla SGD (baseline) | Standard relaxation | $B=16$, $\eta=0.02$ |
| Adam | Faster relaxation | Adaptive optimizer is more sensitive to curvature variation |
| SGD + Nesterov momentum | Faster relaxation | Momentum also amplifies entropic force effects |
| $B=16$ vs $B=256$ | ~10× difference in relaxation time | Validates that entropic force strength scales with effective temperature |
| $\eta=0.01$ vs $\eta=0.05$ | Larger learning rate yields faster relaxation | Higher temperature strengthens entropic forces |

### Key Findings
- Even when the loss along the path remains flat or decreases, curvature still increases systematically, ruling out the alternative explanation that "curvature increases only because loss decreases."
- Entropic barriers are more persistent than energy barriers: in linear mode connectivity experiments, as the branching epoch $k$ increases, loss instability disappears first, while curvature instability persists longer.
- Entropic forces can drive models to move against the gradient direction—free energy minimization rather than energy minimization.
- These phenomena are consistent across CIFAR-10/100, ResNet-20, ResNet-110, and WRN-16-4.

## Highlights & Insights
- Introducing the entropic force concept from statistical physics into deep learning optimization theory, this work explains a long-standing paradox with an elegant physical analogy: energetic connectivity does not imply dynamic connectivity. This framework elevates "SGD's implicit regularization toward flat minima" from an empirical observation to a mechanistic explanation with physical grounding.
- The experimental design is particularly elegant: projected SGD reduces the high-dimensional problem to a one-dimensional path, enabling direct measurement and quantification of entropic force effects without indirect inference.

## Limitations & Future Work
- The paths found by AutoNEB and linear interpolation introduce selection bias among all low-loss paths; more principled path sampling methods are needed.
- SGD noise is simplified to Gaussian white noise, whereas in practice it is neither fully white nor fully Gaussian, which may affect quantitative conclusions.
- Validation is limited to CIFAR-10/100 and relatively small-scale models; whether findings generalize to large-scale Transformers has not been investigated.

## Related Work & Insights
- **vs. Frankle et al. (2020)**: That work found that models sharing early training are linearly mode-connected; this paper further reveals that curvature barriers along such linear paths persist longer than loss barriers, complementing the understanding of what determines the final convergence region.
- **vs. Keskar et al. (2017)**: That work observed that small-batch SGD favors flat minima; this paper provides a more precise physical mechanism via entropic forces, and extends the analysis to curvature variation along connecting paths.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing entropic forces from statistical physics to explain the mode connectivity paradox represents a distinctive theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three curvature measurement methods provide cross-validation; projected SGD design is elegant; results are consistent across architectures and datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Physical intuition and mathematical derivations are seamlessly integrated; figures are clear; the logical chain is complete.
- Value: ⭐⭐⭐⭐ Significant theoretical value for understanding loss landscape structure and SGD behavior, with practical implications for methods such as weight-space ensembling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning](../../NeurIPS2025/optimization/neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/optimization/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
