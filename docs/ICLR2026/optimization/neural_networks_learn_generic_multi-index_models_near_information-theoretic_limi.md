---
title: >-
  [Paper Note] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit
description: >-
  [ICLR 2026][Optimization][Multi-Index Models] This paper proves that, under generic non-degeneracy assumptions, standard two-layer neural networks trained via hierarchical gradient descent can learn generic Gaussian Mult…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Multi-Index Models"
  - "Information-Theoretic Lower Bounds"
  - "Feature Learning"
  - "Power Iteration"
  - "Two-Layer Neural Networks"
date: 2026-05-08
content_hash: 19d4cc6ab401e5ac
---

# Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit

**Conference**: ICLR 2026
**arXiv**: [2511.15120](https://arxiv.org/abs/2511.15120)  
**Code**: None  
**Area**: Optimization
**Keywords**: Multi-Index Models, Information-Theoretic Lower Bounds, Feature Learning, Power Iteration, Two-Layer Neural Networks

## TL;DR
This paper proves that, under generic non-degeneracy assumptions, standard two-layer neural networks trained via hierarchical gradient descent can learn generic Gaussian Multi-Index models $f(\bm{x})=g(\bm{U}\bm{x})$ with $\tilde{O}(d)$ samples and $\tilde{O}(d^2)$ time, achieving information-theoretically optimal sample and time complexity. This constitutes the first proof that neural networks can efficiently learn hierarchical functions.

## Background & Motivation

**Background**: The Multi-Index model $f^*(\bm{x})=g^*(\bm{U}\bm{x})$ is a standard framework for studying representation learning, where the output depends on the input only through its projection onto a low-dimensional subspace. Information-theoretic lower bounds require $\Theta(d)$ samples, and polynomial-time spectral algorithms are known to attain this bound.

**Limitations of Prior Work**:
   - Single-index models ($r=1$) are well understood theoretically, but guarantees for neural network learning of Multi-Index models ($r>1$) remain extremely limited.
   - Damian et al. (2022) showed that neural networks can learn Multi-Index models, but require a suboptimal $\tilde{\Theta}(d^2)$ samples.
   - Arnaboldi et al. address only a special subclass with staircase structure, far from the general case.
   - The concurrent work of Mousavi-Hosseini & Wu (2026) achieves only weak recovery of partial feature directions and cannot guarantee full subspace recovery.

**Key Challenge**: Information theory guarantees that $\Theta(d)$ samples suffice, and spectral methods achieve this bound—yet whether **neural networks** trained via standard gradient descent can attain the same complexity remained unproven.

**Key Insight**: By analyzing gradient descent dynamics under small initialization, the paper identifies that the inner-layer weights implicitly perform a power iteration procedure, analogous to spectral initialization on the local Hessian. Critically, the number of training steps must be carefully calibrated: too few steps leave sample noise unreduced; too many cause all neurons to align with the leading eigenvector direction, losing coverage of non-dominant subspace directions.

**Core Idea**: Gradient descent on neural networks implicitly performs "power iteration with early stopping"—an appropriate number of training steps suffices to suppress finite-sample noise while preserving subspace diversity, thereby approaching the information-theoretic optimum.

## Method

### Overall Architecture
A two-layer neural network $f_{\Theta}(\bm{x}) = \sum_j a_j \sigma(\bm{w}_j^T \bm{x} + b_j)$ is trained via hierarchical gradient descent: **Phase 1**: Train inner-layer weights $\bm{W}$ on dataset $\mathcal{D}_1$ for $T_1$ steps; **Phase 2**: Re-initialize biases $\bm{b}$, then train outer-layer weights $\bm{a}$ on dataset $\mathcal{D}_2$ for $T_2$ steps.

### Key Designs

1. **Implicit Power Iteration Mechanism**:

    - Function: Proves that the first $T_1$ steps of gradient descent implicitly simulate power iteration on the local Hessian $\hat{\Sigma}_\ell = \frac{1}{n}\sum_i \ell_i \bm{x}_i\bm{x}_i^T$.
    - Mechanism: Under small initialization ($\epsilon_0 \to 0$), the activation approximates a quadratic ($\sigma''(0)=1$), neuron gradient updates are nearly independent, and each update $\bm{w}_j^{(t+1)} \approx \hat{\Sigma}_\ell \bm{w}_j^{(t)}$ resembles the power method. The large eigenvalues of $\hat{\Sigma}_\ell$ correspond to signal (the latent subspace), while small eigenvalues correspond to finite-sample noise.
    - Design Motivation: Power iteration naturally amplifies signal directions and attenuates noise—but must be stopped at the right step.

2. **Critical Stopping Time Condition**:

    - Function: Determines the optimal number of inner-layer training steps $T_1$.
    - Mechanism: $T_1$ too small → noise directions are insufficiently attenuated (especially when $n \sim d$); $T_1$ too large → all neurons align to the leading eigenvector direction, losing full coverage of the subspace. The optimal $T_1$ ensures that all $r$ signal directions are covered by sufficiently many neurons, while noise is reduced to a negligible level.
    - Design Motivation: This reveals a counterintuitive phenomenon—optimal learning requires a **diverging** (rather than $O(1)$) number of first-layer training steps.

3. **Generic Loss Function as Data Preprocessing**:

    - Function: Interprets the choice of loss function as a form of data preprocessing.
    - Mechanism: The choice of $\ell'(0, y)$ determines the spectral structure of $\Sigma_\ell$. For Multi-Index models with generative exponent 2, an appropriate loss function can make $\Sigma_\ell$ non-degenerate (Assumption 5). This extends the role of the loss function from an "optimization objective" to a "preprocessor for feature extraction."
    - Design Motivation: Broadens the applicable class of objective functions beyond squared loss to include $\ell^1$, Huber loss, and others.

### Loss & Training
- Phase 1: $\bm{W}^{(t)} \leftarrow \bm{W}^{(t-1)} - \eta_1[\nabla_W \hat{\mathcal{L}}_{\mathcal{D}_1} + \beta_1 \bm{W}^{(t-1)}]$ (GD with weight decay)
- Phase 2: Standard GD on outer weights $\bm{a}$ (convex optimization, converges to optimum)
- Symmetric initialization: $a_j = -a_{m-j}$, $\bm{w}_j^{(0)} = \bm{w}_{m-j}^{(0)}$ (eliminates mean bias)
- Learning rate selection is critical: excessively small learning rates cause the power iteration approximation to break down.

## Key Experimental Results

### Main Results (Key Quantities in Theorem 1)

| Quantity | Complexity | Information-Theoretically Optimal? |
|----|--------|-----------|
| Sample complexity | $\tilde{O}(d)$ | Yes (leading order) |
| Time complexity | $\tilde{O}(d^2)$ | Yes (leading order) |
| Test error | $o_d(1) \to 0$ | Yes |
| Applicable rank | Any constant $r$ | — |
| Applicable losses | Squared, $\ell^1$, Huber, etc. | — |

### Comparison with Prior Work

| Work | Sample Complexity | End-to-End Learning? | Generic Model? |
|------|----------|-----------|---------|
| Damian et al. 2022 | $\tilde{\Theta}(d^2)$ | Yes | Partial |
| Arnaboldi et al. 2023 | $\tilde{O}(d^L)$ ($L$=staircase steps) | Yes | Staircase structure |
| Mousavi-Hosseini & Wu 2026 | $O(d)$ | **No** (weak recovery only) | Yes |
| **Ours** | **$\tilde{O}(d)$** | **Yes** | **Yes** (non-staircase) |

### Key Findings
- **First end-to-end optimal guarantee**: Neural networks trained with GD fully learn Multi-Index models with information-theoretically optimal $\tilde{O}(d)$ samples.
- **Training steps cannot be too few**: Optimal results require $T_1 \to \infty$ first-layer training steps, ruling out all DMFT/state-evolution-based analyses that assume $O(1)$ steps.
- **Learning rate cannot be too small**: The power iteration approximation requires a moderately large learning rate; too small a rate causes the dynamics to degenerate.
- **No spectral initialization or warm-starting required**: Standard random initialization suffices—neural networks learn the correct features through gradient descent alone.
- **Loss function as preprocessing**: Choosing an appropriate loss function is equivalent to selecting a data preprocessing scheme, which can activate additional signal directions.

## Highlights & Insights
- The discovery of the **"implicit power iteration with early stopping" mechanism** is the paper's central theoretical contribution—gradient descent is not merely an optimizer; its dynamics inherently encode a spectral method. This provides a new perspective on understanding why deep learning works.
- The **counterintuitive finding that $O(1)$ steps are insufficient** challenges DMFT-based analytical frameworks, suggesting that the true mechanism of neural network learning may differ from the predictions of these simplified theories.
- **Treating the loss function as preprocessing** establishes a connection between loss selection and feature extraction—in practice, the choice of loss function can be informed by its effect on signal extraction, not solely by its optimization properties.

## Limitations & Future Work
- The **non-staircase assumption** is a key limitation—target functions with a generative staircase structure are not covered.
- Hierarchical training (W before a) rather than standard joint GD—theoretically cleaner but less aligned with practice.
- Requires sample splitting (two independent datasets)—a cost of theoretical tractability.
- Only two-layer networks are considered—analysis of deeper networks remains to be developed.
- Constant $r$ (subspace dimension) and $p$ (polynomial degree)—high-dimensional settings are not addressed.
- Leading-order optimal but constants are not precisely characterized—the gap in constants relative to optimal spectral methods is not quantified.

## Related Work & Insights
- **vs. Damian et al. 2022**: The first neural network learning result for Multi-Index models, but requiring $d^2$ samples; this paper reduces the requirement to $d$—a leading-order improvement.
- **vs. Mousavi-Hosseini & Wu 2026**: A concurrent work achieving $O(d)$ samples but only weak, incomplete subspace recovery; this paper proves full recovery with end-to-end learning.
- **vs. Spectral methods (DLB25)**: Information-theoretically optimal but not based on neural networks; this paper demonstrates that neural networks can match the efficiency of spectral methods.
- **vs. DMFT/state-evolution methods**: These require $O(1)$ training steps or warm-starting; this paper proves that optimal learning requires a diverging number of steps.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof that neural networks attain the information-theoretic optimum for Multi-Index learning; exceptionally deep theoretical contributions.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work with no empirical validation (though the theory is self-contained and complete).
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear problem positioning, though notation density is high.
- Value: ⭐⭐⭐⭐⭐ A milestone contribution to the theory of feature learning in neural networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[ICLR 2026\] Unifying Formal Explanations: A Complexity-Theoretic Perspective](unifying_formal_explanations_a_complexity-theoretic_perspective.md)
- [\[ICLR 2026\] Rethinking Consistent Multi-Label Classification Under Inexact Supervision](rethinking_consistent_multi-label_classification_under_inexact_supervision.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)
- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](rapid_training_of_hamiltonian_graph_networks_using_random_features.md)

</div>

<!-- RELATED:END -->
