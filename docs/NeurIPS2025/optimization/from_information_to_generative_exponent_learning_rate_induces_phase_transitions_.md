---
title: >-
  [Paper Note] From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD
description: >-
  [NeurIPS 2025][Optimization][Single-index models] This work systematically characterizes how the learning rate induces a phase transition between two sample complexity regimes—one dominated by the "information exponent" and the other by the "generative exponent"—when learning Gaussian single-index models. It further proposes a novel layer-wise alternating SGD algorithm that circumvents the CSQ lower bounds without requiring sample reuse.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Single-index models"
  - "learning rate phase transitions"
  - "information exponent"
  - "generative exponent"
  - "SGD sample complexity"
date: 2026-05-08
content_hash: 11ba071affb32d87
---

# From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD

**Conference**: NeurIPS 2025  
**arXiv**: [2510.21020](https://arxiv.org/abs/2510.21020)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Single-index models, learning rate phase transitions, information exponent, generative exponent, SGD sample complexity

## TL;DR
This work systematically characterizes how the learning rate induces a phase transition between two sample complexity regimes—one dominated by the "information exponent" and the other by the "generative exponent"—when learning Gaussian single-index models. It further proposes a novel layer-wise alternating SGD algorithm that circumvents the CSQ lower bounds without requiring sample reuse.

## Background & Motivation

**Background**: The sample complexity of learning single-index models $y = \sigma_*(⟨\mathbf{x}, \boldsymbol{\theta}_*⟩) + \zeta$ with $\mathbf{x} \sim \mathcal{N}(0, I_d)$ is determined by the properties of the link function. The complexity of online SGD is dominated by the information exponent $p = \text{IE}(\sigma_*)$ (the order of the first non-zero Hermite coefficient), resulting in $\tilde{\Theta}(d^{(p-1)\vee 1})$.

**Limitations of Prior Work**:
   - While multiple gradient steps (batch reuse) can reduce the complexity to be dominated by the generative exponent $p_* = \text{GE}(\sigma_*) \leq p$, this landscape only holds when the learning rate is sufficiently large.
   - The best known bounds for full-batch gradient flow still depend on the information exponent, implying that the role of the learning rate has been overlooked.
   - Prior works have not systematically investigated how the choice of learning rate affects the transition between CSQ/SQ regimes.

**Key Challenge**: Intuitively, batch reuse should bypass the CSQ limitations, but the bounds for full-batch gradient flow (the extreme case of reuse) do not improve. This implies that the learning rate, rather than reuse alone, determines the query class.

**Key Insight**: A broad class of gradient algorithms can be unified under the form $\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} + \gamma \psi_\eta(y, ⟨\mathbf{x}, \mathbf{w}⟩) P_\mathbf{w}^\perp \mathbf{x}$, where $\eta$ controls the scale of the non-correlated term.

**Core Idea**: The sample complexity is determined by the competition among Hermite coefficients $\mu_i(\eta)$. The size of $\eta$ determines which term dominates, thereby eliciting a non-smooth phase transition between the information exponent regime and the generative exponent regime.

## Method

### Overall Architecture
Consider a two-layer network $f(\mathbf{x}; W, \mathbf{a}, \mathbf{b}) = \frac{1}{N}\sum_{j=1}^N a_j \sigma_j(⟨\mathbf{x}, \mathbf{w}_j⟩ + b_j)$ learning the target function. The unified update rule takes the form of spherical projected gradient: $\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} + \gamma \psi_\eta(y, ⟨\mathbf{x}, \mathbf{w}⟩) P_\mathbf{w}^\perp \mathbf{x}$, where $\gamma$ is the global learning rate, and $\eta$ controls the scaling of the non-correlated update term.

### Key Designs

1. **Unified Framework and Hermite Coefficient Analysis**:

    - Function: Unify online SGD, batch reuse SGD, and alternating SGD into a single framework.
    - Mechanism: Define the key quantity $\mu_i(\eta) = \mathbb{E}[\psi_\eta(\sigma_*(a), b) \mathsf{He}_i(a) \mathsf{He}_{i-1}(b)]$ where $(a,b) \sim \mathcal{N}(0, I_2)$. The alignment of the update direction with the target is $\mathbb{E}[⟨\boldsymbol{\theta}_*, \mathbf{g}⟩] = \sum_i i! \mu_i ⟨\boldsymbol{\theta}_*, \mathbf{w}⟩^{i-1}(1-⟨\boldsymbol{\theta}_*, \mathbf{w}⟩^2)$.
    - Design Motivation: The magnitude and non-zero status of $\mu_i(\eta)$ entirely dictate which Hermite order dominates the growth of alignment.

2. **Main Theorem (Theorem 3.3)**:

    - Function: Provide a general sample complexity formula.
    - Mechanism: Setting $\gamma \leq C\delta \max_i \mu_i d^{-(i/2 \vee 1)}$ yields the number of iterations required for weak recovery as
    $T(\eta) = \min_{\substack{1 \leq i \leq r \\ \mu_i > 0}} \tilde{\Theta}\big(\gamma^{-1} \mu_i(\eta)^{-1} d^{\frac{i-2}{2} \vee 0}\big)$
      After choosing the optimal $\gamma$, this simplifies to $T(\eta) = \min_i \tilde{\Theta}(\mu_i(\eta)^{-2} d^{(i-1)\vee 1})$.
    - Design Motivation: The $\min$ operation produces a non-smooth phase transition when the optimal $i$ varies with respect to $\eta$.

3. **Batch Reuse SGD Instantiation (Algorithm 1)**:

    - Function: Two-step update—execute the first gradient step with $\eta$, and run the second step with $\gamma$ on the same sample.
    - Mechanism: Taylor expansion yields $\psi_\eta(y,z) = y\sigma'(z) + \sum_{k=2}^r \frac{(\eta d)^{k-1}}{(k-1)!}\sigma^{(k)}(z)(\sigma'(z))^{k-1} y^k$. The phase transition condition indicates that the system remains in the information exponent regime when $\eta \leq d^{\frac{[(p_j-1)\vee 1]-[(p_i-1)\vee 1]}{2(j-i)}-1}$.
    - Design Motivation: Validate the correctness of the framework—when $\eta \lesssim d^{-(p+1)/2}$, the complexity degenerates to $\Theta(d^{(p-1)\vee 1})$ of online SGD, and when $\eta \gtrsim d^{-1}$, it recovers $\tilde{\Theta}(d)$.

4. **Alternating SGD (Algorithm 2, Ours)**:

    - Function: Layer-wise update—first update the second layer with $\eta$ via $\tilde{a} \leftarrow a + \eta y \sigma(⟨\mathbf{x}, \mathbf{w}⟩)$, and then update the first layer using the updated $\tilde{a}$ and $\gamma$.
    - Mechanism: The update function is $\psi_\eta(y,z) = ya\sigma'(z) + \eta y^2 \sigma(z)\sigma'(z)$, yielding the coefficient $\mu_i(\eta) = ai \cdot u_i(\sigma_*) u_i(\sigma) + \eta \cdot u_{i-1}(\sigma\sigma') u_i(\sigma_*^2)$. If $p_2 = \text{IE}(\sigma_*^2) < p$ and $\eta$ is sufficiently large, the second term dominates, leading to a complexity of $\tilde{\Theta}(\eta^{-2} d^{(p_2-1)\vee 1})$.
    - Design Motivation: Introduce the non-correlated term without sample reuse or modifying the standard squared loss—the timescale separation induced by the different learning rates across the two layers naturally generates a $y^2$ transformation.

### Loss & Training
- Alternating SGD: For $\sigma_* = \mathsf{He}_3$ ($p=3, p_2=2$), the phase transition threshold is $\eta \asymp d^{-1/2}$.
- Below $\eta$: $T = \Theta(d^2)$ (information exponent regime).
- Above $\eta$: $T = \tilde{\Theta}(d)$ (linear complexity).
- Generalizable to $D$-layer sparse networks: $T = \max_{1 \leq i \leq D} \tilde{\Theta}(\eta^{-2(i-1)} d^{(p_i-1)\vee 1})$.

## Key Experimental Results

### Main Results

| Algorithm | Complexity (small $\eta$) | Complexity (large $\eta$) | Phase Transition Threshold |
|------|-------------------|-------------------|---------|
| Online SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | N/A | None |
| Batch Reuse SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | $\tilde{\Theta}(d^{(p_*-1)\vee 1})$ | $\eta \asymp d^{-1}$ |
| Alternating SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | $\tilde{\Theta}(d^{(p_2-1)\vee 1})$ | $\eta \asymp d^{-\frac{p-p_2}{2}}$ |

### Numerical Experiments ($\sigma_* = \sigma = \mathsf{He}_3$, $d=50$)

| $\eta$ Range | Required Sample Size $n$ to Achieve $⟨\mathbf{w}, \boldsymbol{\theta}_*⟩ \geq 0.5$ | Regime |
|------------|--------------------------------------------------------------|------|
| $\eta \lesssim d^{-1/2}$ | $n \approx d^2 = 2500$ | information exponent |
| $\eta \gtrsim d^{-1/2}$ | $n$ decreases with $\eta^{-2}$ | transition zone |
| $\eta \approx 1$ | $n \approx d \cdot \text{polylog}$ | generative exponent |

### Key Findings
- Figure 1 clearly illustrates two regions with distinct slopes, where the phase transition point is consistent with the theoretical prediction $\eta \asymp d^{-1/2}$.
- Alternating SGD is the first algorithm to bypass the CSQ restriction without requiring batch reuse or changing the loss function.
- Deep $D$-layer networks can further exploit the information exponent of $\sigma_*^D$, achieving linear complexity in more scenarios.

## Highlights & Insights
- **Learning rate as a pivotal dimension of algorithm design**: It not only affects the convergence rate but also determines whether the algorithm falls into the CSQ or SQ category—a profound insight that was previously overlooked.
- **Simplicity of Alternating SGD**: Applying different learning rates (two timescales) to the two layers is sufficient to generate non-correlated updates, without resorting to batch reuse or altering the loss.
- **Expressiveness of the unified framework**: Analyzing the three algorithms under a unified manner via the $\mu_i(\eta)$ coefficients makes the phase transition conditions and complexity comparisons crystal clear.

## Limitations & Future Work
- Only single-index models are analyzed; generalization to multi-index models remains to be completed.
- The analysis assumes Gaussian input distributions; more general distributions warrant further investigation.
- The analysis of Alternating SGD only covers $p_2 = \text{IE}(\sigma_*^2)$, and verifying the conditions for deep networks is challenging.
- Non-polynomial activation functions are not covered.

## Related Work & Insights
- **vs Ben Arous et al. (BAGJ'21)**: They established the information exponent bound for online SGD; this work shows that the learning rate can change this bound.
- **vs Damian et al. / Lee et al.**: They achieved the generative exponent regime via batch reuse; this work reveals that this requires a sufficiently large $\eta$.
- **vs Joshi et al.**: They bypassed the CSQ limitations by modifying the loss function; our alternating SGD maintains the standard squared loss.
- **vs Cutkosky-Wei-Lin et al.**: They achieved SQ-optimality using generalized gradients with perturbation and averaging; our perspective is complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the learning rate inducing phase transitions is profound, and the alternating SGD concept is simple yet powerful.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical, with numerical verification restricted to a small scale of $d=50$.
- Writing Quality: ⭐⭐⭐⭐⭐ The unified framework is clearly elucidated, and the step-by-step unfolding of the three instances exhibits an excellent pace.
- Value: ⭐⭐⭐⭐⭐ It reveals the fundamental role of the learning rate in SGD complexity, carrying profound implications for understanding feature learning in neural networks.

## Related Papers

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
