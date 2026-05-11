---
title: >-
  [Paper Note] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study
description: >-
  [NeurIPS 2025][Optimization][generalization] This paper establishes, within the matrix factorization framework (a canonical theoretical testbed for neural networks), that Guess & Check (G&C…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "generalization"
  - "volume hypothesis"
  - "Guess & Check"
  - "matrix factorization"
  - "implicit bias"
date: 2026-05-08
content_hash: 0ff34ad3fd903469
---

# Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study

**Conference**: NeurIPS 2025
**arXiv**: [2506.03931](https://arxiv.org/abs/2506.03931)
**Code**: None
**Area**: Learning Theory / Generalization Theory
**Keywords**: generalization, volume hypothesis, Guess & Check, matrix factorization, implicit bias

## TL;DR
This paper establishes, within the matrix factorization framework (a canonical theoretical testbed for neural networks), that Guess & Check (G&C; randomly sampling parameters until the training set is fit) exhibits generalization that degrades with increasing width (the first provable demonstration of a canonical setting where G&C is strictly inferior to gradient descent) yet improves with increasing depth, revealing the fundamentally opposite roles that width and depth play in generalization.

## Background & Motivation
**Background**: Overparameterized neural networks trained via gradient descent typically generalize well. The conventional view attributes this to the implicit bias of GD, whereby GD tends to find solutions with specific properties (e.g., low rank, low norm).

**Limitations of Prior Work**: The recently proposed *volume hypothesis* challenges this view, arguing that good generalization arises not from any special preference of GD but because weight configurations that generalize well occupy a larger "volume" in parameter space. If the volume hypothesis holds, then Guess & Check (sampling from a prior distribution until the training data are fit) should achieve generalization comparable to GD.

**Key Challenge**: Experimental evidence is contradictory — Chiang et al. find that G&C matches GD and that width is beneficial, whereas Peleg et al. find that G&C is inferior to GD and that width is unhelpful. Both are purely empirical works subject to confounding factors and lacking theoretical analysis.

**Key Insight**: Matrix factorization is adopted as a theoretically tractable canonical testbed, enabling simultaneous study of the effects of width and depth while eliminating experimental confounders.

**Core Idea**: Increasing width causes the generalization of G&C to degrade to the level of not fitting the training data at all (Theorem 1), whereas increasing depth causes G&C generalization to approach perfection (Theorem 2) — the effects of width and depth are diametrically opposed.

## Method

### Overall Architecture
- **Problem Setting**: Low-rank matrix sensing, i.e., recovering a low-rank matrix $W^*$ from linear measurements.
- **Matrix Factorization Parameterization**: $W = W_d \sigma(W_{d-1} \sigma(\cdots \sigma(W_1)))$, with width $k$, depth $d$, and activation function $\sigma$.
- **Two Optimizers Compared**:
  - Gradient Descent (GD): iterates from a small random initialization.
  - Guess & Check (G&C): repeatedly samples from a prior distribution until the training loss falls below a threshold.

### Key Designs

1. **Theorem 1: Increasing Width → G&C Generalization Degrades**

   - **Function**: Proves that for antisymmetric activation functions (e.g., linear, tanh, sine), G&C generalization degrades to the level of random guessing as width tends to infinity.
   - **Mechanism**: Exploits the NNGP (Neural Network Gaussian Process) convergence of deep networks — as width $k \to \infty$, the output of the matrix factorization converges to a random matrix $W_{\text{iid}}$ whose entries are i.i.d. Training loss and generalization loss then become statistically independent, so conditioning on fitting the training data provides no information about generalization.
   - **Key Condition**: The activation function is antisymmetric, $\sigma(-\alpha) = -\sigma(\alpha)$, which ensures independence between columns.
   - **Finite-Width Bound**: When the prior is Gaussian, the difference between the posterior and prior probability is $O(1/\sqrt{k})$.

2. **Theorem 2: Increasing Depth → G&C Generalization Improves**

   - **Function**: Proves that for linear activations and a normalized Gaussian prior, G&C generalization approaches perfection as depth tends to infinity.
   - **Mechanism**: The normalized prior of deep matrix factorization $W = W_d W_{d-1} \cdots W_1$ induces a strong low-rank preference. As $d \to \infty$, the singular value distribution of $W$ concentrates on a single nonzero singular value (i.e., rank one), so low-rank solutions occupy exponentially greater volume in parameter space than high-rank solutions.
   - **Implicit Regularization Interpretation**: Depth itself achieves rank minimization through the geometric effect of the prior, without requiring GD's implicit bias.

3. **Experimental Validation**

   - **Function**: Implements G&C (via MCMC posterior sampling) and GD on matrix factorization to verify theoretical predictions.
   - **Observations**: G&C generalization degrades sharply with increasing width (consistent with Theorem 1) and improves significantly with increasing depth (consistent with Theorem 2); GD maintains good generalization across all widths and depths.

### Loss & Training
- Training loss: $\mathcal{L}_{\text{train}}(W) = \frac{1}{n} \sum_{i=1}^n (\langle A_i, W \rangle - y_i)^2$
- Generalization loss: $\mathcal{L}_{\text{gen}}(W) = \frac{1}{|\mathcal{B}|} \sum_{A \in \mathcal{B}} (\langle A, W \rangle - \langle A, W^* \rangle)^2$
- G&C is equivalent to sampling from the posterior $\mathcal{P}(\cdot | \mathcal{L}_{\text{train}} < \epsilon)$.
- Prior distribution: Kaiming scaling ($1/\sqrt{m_j}$) with optional normalization (preserving the product norm at 1).

## Key Experimental Results

### Main Results (Matrix Factorization, Varying Width/Depth)

| Setting | Width Variation | Depth Variation | GD Generalization | G&C Generalization |
|---------|----------------|----------------|-------------------|--------------------|
| Linear activation | Increase → G&C degrades | Increase → G&C improves | Consistently good | Width↑ poor / Depth↑ good |
| tanh activation | Increase → G&C degrades | Increase → G&C improves | Consistently good | Same as above |

### Ablation Study

| Configuration | Key Findings |
|--------------|-------------|
| Antisymmetric vs. non-antisymmetric activations | Antisymmetric (tanh, sine, linear) satisfy Theorem 1; ReLU (non-antisymmetric) may behave differently |
| With vs. without normalization | Normalization is critical for the depth improvement effect |
| Finite-width convergence rate | $O(1/\sqrt{k})$ convergence under Gaussian prior, consistent with experiments |

### Key Findings
- **First theoretical proof** of a canonical setting where G&C (under standard priors) is provably inferior to GD.
- Width and depth have diametrically opposite effects on G&C generalization — a finding that reconciles the contradictory experimental results in the literature.
- GD's implicit bias is indeed indispensable in wide networks; however, in deep networks, the geometric effect of the architecture itself may suffice.

## Highlights & Insights
- **Width–Depth Symmetry Breaking**: While overparameterization is intuitively assumed to be universally beneficial, this work reveals that width-based and depth-based overparameterization have completely opposite effects on G&C generalization. Increasing width renders the parameter space more "homogeneous" (the volume ratio of good to bad solutions approaches a constant), whereas increasing depth causes the volume of low-rank solutions to grow exponentially.
- **Novel Application of NNGP Convergence**: The Gaussian process convergence of infinite-width networks is repurposed from the analysis of prediction functions to proving the independence of training and test losses in generalization theory.
- **Practical Implications**: In G&C or Bayesian methods (which are essentially analogous), depth is more conducive to generalization than width, providing theoretical guidance for the "deep and narrow" vs. "wide and shallow" architectural design choice.

## Limitations & Future Work
- **Restricted to Matrix Factorization**: Although this is a canonical testbed, extending the results to fully connected or convolutional networks requires additional work.
- **Theorem 1 Requires Antisymmetric Activations**: ReLU does not satisfy this condition, yet it is the most widely used activation function in practice.
- **Theorem 2 Restricted to Linear Activations**: The depth effect under nonlinear activations remains an open problem.
- **Limited Practical Feasibility of G&C**: In high-dimensional parameter spaces, G&C is nearly impossible to execute in practice (the probability of sampling parameters that fit the training data is negligibly small), so the conclusions are primarily of theoretical significance.
- **Explicit Regularization Not Considered**: Techniques such as weight decay, commonly used in practical training, may alter the behavior of G&C.

## Related Work & Insights
- **vs. Chiang et al. (2023)**: They experimentally find that G&C ≈ GD and that width is beneficial, but their experimental design contains confounding factors; this paper theoretically proves that G&C degrades in wide networks.
- **vs. Peleg et al. (2024)**: They experimentally find that G&C < GD and that depth is unhelpful (possibly also confounded); this paper theoretically proves that depth in fact improves G&C.
- **vs. Implicit Bias Theory (Soudry, Gunasekar, et al.)**: Prior work proves that GD has a special preference; this paper confirms from a complementary angle that such preference is indeed indispensable in wide networks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First theoretical proof that G&C is provably inferior to GD, and revelation of the width/depth symmetry breaking.
- Experimental Thoroughness: ⭐⭐⭐ Experiments primarily validate theoretical predictions; scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorem statements are clear; proof sketches effectively convey the core ideas.
- Value: ⭐⭐⭐⭐ Significant contribution to the generalization theory community, addressing a fundamental open question.

## Additional Remarks
- The theoretical framework and technical tools developed in this paper offer insights for research in adjacent areas.
- The core contribution lies in a deep theoretical understanding that lays the groundwork for subsequent practical improvements.
- The paper is complementary in technique and methodology to other NeurIPS 2025 papers published concurrently.
- The exposition of problem motivation and technical approach is worth studying.
- Readers are encouraged to consult the appendix for more complete experimental details and proofs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

</div>

<!-- RELATED:END -->
