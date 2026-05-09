---
title: >-
  [Paper Note] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias
description: >-
  [NeurIPS 2025][Optimization][SAM] Through a linear stability analysis framework, this paper demonstrates that "flat minima ⇒ better generalization" and "SGD prefers simple functions" are two sides of the same coin — data coherence simultaneously governs both phenomena, and SAM amplifies the simplicity bias further by imposing stricter stability conditions.
tags:
  - NeurIPS 2025
  - Optimization
  - SAM
  - SGD
  - linear stability
  - data coherence
  - simplicity bias
date: 2026-05-08
content_hash: 9ca343ffee2ccff0
---

# A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias

**Conference**: NeurIPS 2025
**arXiv**: [2511.17378](https://arxiv.org/abs/2511.17378)
**Code**: [https://github.com/changwk1001/Stability_Analysis_and_Simplicity-Bias.git](https://github.com/changwk1001/Stability_Analysis_and_Simplicity-Bias.git)
**Area**: Optimization Theory
**Keywords**: SAM, SGD, linear stability, data coherence, simplicity bias

## TL;DR
Through a linear stability analysis framework, this paper demonstrates that "flat minima ⇒ better generalization" and "SGD prefers simple functions" are two sides of the same coin — data coherence simultaneously governs both phenomena, and SAM amplifies the simplicity bias further by imposing stricter stability conditions.

## Background & Motivation
**Background**: Two central generalization hypotheses in deep learning: (1) the *flat minima hypothesis* — SGD/SAM preferentially converges to wide, shallow loss basins, and flatness correlates positively with generalization; (2) the *simplicity bias hypothesis* — overparameterized networks tend to learn solutions that rely on a small number of shared features.

**Limitations of Prior Work**: Both hypotheses are supported by extensive empirical evidence, yet a unified theoretical framework explaining their intrinsic connection is lacking. In particular: why does SGD prefer flat minima? Why does SAM select better solutions even among minima of equal flatness? How does data geometry influence optimization bias?

**Key Challenge**: SAM is designed to "seek flat minima," yet experiments show it exhibits selectivity beyond flatness — it still prefers certain solutions among minima of equal flatness. This suggests flatness alone is insufficient to explain generalization.

**Key Insight**: Linear stability analysis — linearizing the optimization dynamics near a minimum and characterizing stability via the spectral properties of the iteration matrix.

**Core Idea**: Data coherence (the degree of alignment among per-sample Hessians) simultaneously governs both the flatness preference and the simplicity bias; SAM amplifies this selectivity through an additional curvature penalty term.

## Core Problem
Can a unified theoretical framework be established to simultaneously explain the selective preferences of SGD and SAM over minima, and reveal how the geometric structure of training data — particularly the gradient alignment across training samples — determines which solutions serve as stable attractors? Specifically: (1) Does noise injection alter the set of stable minima? (2) What additional selectivity does SAM introduce relative to SGD? (3) How do these selectivities manifest as a preference for simple solutions in concrete network architectures?

## Method

### Overall Architecture
The analysis centers on **linear stability analysis**: the update dynamics are Taylor-expanded near a minimum $w^*$, and the spectral properties of the iteration matrix are examined to determine whether the minimum is an attractor. The core physical intuition is that if $\mathbb{E}[\|w_k\|^2]$ diverges during iteration, the solution is unstable (SGD will escape); if it converges, the solution is stable (SGD will remain).

### Key Designs
1. **Coherence Measure**: The coherence matrix is defined as $S_{ij} = \text{Tr}(H_i H_j)$, where $H_i$ is the per-example Hessian of sample $i$. The coherence measure is $\sigma = \lambda_{\max}(S)/\max_i \lambda_{\max}(H_i)$. High coherence implies that the curvature directions of different training samples are highly aligned — i.e., the model fits multiple samples using a small number of shared features. The paper proves that high-coherence solutions are more stable (permitting larger learning rates), because gradients along shared directions produce stronger restoring forces.

2. **Stability Conditions for SGD, Random Perturbations, and SAM**:

   - **SGD divergence condition** (known): $\lambda_{\max}(H) \geq \frac{\sigma}{\eta}\left(\frac{n}{B}-1\right)^{-1/2}$
   - **Random perturbations** (Theorem 3.1): The divergence condition is identical to that of SGD (noise injection does not alter which minima are stable), but the escape rate is faster by a constant factor; when stable, convergence is inexact and the iterates oscillate near the minimum.
   - **SAM divergence condition** (Theorem 3.2): $\lambda_{\max}(H) \geq \frac{\sigma}{\eta}\left(\frac{n}{B}-1\right)^{-1/2}\left(1+\frac{\rho}{\alpha}\lambda_{\min}(H)\right)^{-1}$. The additional curvature factor makes SAM strictly more demanding — minima that are marginally stable under SGD may become unstable under SAM. A matching lower bound (Theorem 3.3) establishes the tightness of this condition.

3. **Realization of Simplicity Bias in Two-Layer ReLU Networks**:

   - **Memorization vs. generalization solutions** (Theorem 3.4): Memorization solutions (each sample activates independent neurons) yield a diagonal coherence matrix ($S_{ij}=0,\ i\neq j$), hence minimal coherence and maximal instability; generalization solutions (shared neurons) yield nonzero off-diagonal entries, hence higher coherence and greater stability.
   - **$(C,r)$-generalization solutions** (Theorem 3.5): For fixed $r$ (equal flatness), $\lambda_{\max}(S) = O(n/2^C \cdot (d+1)^{1/2})$. Smaller $C$ (fewer features used) implies higher coherence and faster convergence, directly proving that SGD prefers simpler solutions even under equal-flatness constraints.
   - **SAM amplifies simplicity bias** (Theorem 3.6): SAM's effective coherence matrix contains additional $\rho/\alpha$-dependent terms that amplify the stability gap between solutions of different $C$ values.

### Loss & Training
The theoretical analysis is conducted under a quadratic loss approximation. Experiments employ two-layer ReLU networks with MSE loss, $d=100$ hidden units, batch size 10, $\eta=0.01$, and $\rho \in \{0.01, 0.05, 0.1, 0.2\}$. Additional validation is performed on CIFAR-10 with ResNet-18.

## Key Experimental Results

### Main Results: Coherence Measure vs. SAM Perturbation Radius

| Metric | SGD | SAM (ρ=0.05) | SAM (ρ=0.1) | SAM (ρ=0.2) |
|--------|-----|--------------|-------------|-------------|
| $\lambda_{\max}(S)$ | 133.9 | 121.5 | 90.3 | 65.7 |
| $\max_i \lambda_{\max}(H_i)$ | 12740 | 10103 | 6422 | 3446 |
| Hessian max eigenvalue | 6.776 | — | 3.834 | — |
| Effective rank (PCA 90%) | 94.39 | — | — | 29.14 |

Setup: two-layer ReLU network, $n=100$, variable $d$, $x \in \{-1,1\}^d$, $y = x[0] \cdot x[1]$ (ensuring both simple and complex solutions coexist).

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Stability boundary in $(B,\sigma)$ space | SGD and random perturbation boundaries largely coincide (validating Thm. 3.1); SAM boundary is strictly tighter |
| Varying $\rho/\alpha$ | Increasing SAM perturbation radius further shrinks the stable region |
| Fixed $r$, varying $C$ | SAM converges faster to low-$C$ solutions (validating Thm. 3.5, 3.6) |
| Training dynamics | Coherence is a dynamic quantity; SAM more aggressively reduces $\max_i \lambda_{\max}(H_i)$ |
| CIFAR-10/ResNet-18 | SAM reduces feature effective rank; approximate coherence measure decreases monotonically with SAM $\rho$ |

### Key Findings
- SAM's effect extends beyond "seeking flat minima" — it still selects high-coherence (simple) solutions among minima of equal flatness.
- Coherence is a dynamic quantity: $\lambda_{\max}(S)$ evolves throughout training, and SAM reduces it more effectively than SGD.
- Random perturbations (noise injection) do not alter which minima are stable; they only accelerate escape from unstable minima.

## Highlights & Insights
- **Theoretical elegance of unification**: The coherence measure $\sigma$ simultaneously explains both the flat-minima preference and the simplicity bias, revealing them as two sides of the same coin.
- **Theorem 3.4 (memorization solutions ⟺ diagonal coherence matrix) is particularly elegant**: It bridges the abstract notion of coherence with concrete neuron activation patterns.
- **SAM transcends flatness search**: SAM not only prefers flat solutions but also prefers solutions that exploit shared features — it retains discriminative power within equal-flatness solution manifolds.
- **Matching lower bound (Theorem 3.3)**: Establishes the tightness of the SAM divergence condition, with upper and lower bounds differing by at most a constant factor.
- **Practical implication**: The simplicity bias of SGD/SAM may partially explain why simple models or prompts outperform more complex alternatives in certain settings.

## Limitations & Future Work
- The core analysis relies on a linear approximation near minima (quadratic loss); non-local dynamics are not covered.
- Only SGD and SAM are analyzed; practical optimizers such as momentum SGD and Adam are not addressed.
- The construction of $(C,r)$-generalization solutions, while principled, is restrictive; solution structures in real networks are more complex.
- The coherence measure is computationally expensive (requiring per-sample Hessians) and is currently impractical for use during actual training.
- The CIFAR-10 experiments employ an approximate coherence measure, introducing a gap relative to the theoretical definition.
- Experiments are conducted primarily on synthetic binary data; behavior on real-world datasets (e.g., ImageNet) may differ.

## Related Work & Insights
- **vs. Dexter et al. (2024)**: That work analyzes the linear stability of SGD only; this paper extends the framework to random perturbations and SAM, and instantiates the theory on two-layer ReLU networks.
- **vs. Foret et al. (2021, original SAM)**: The original SAM paper interprets the method as seeking flat minima; this paper proves that SAM's bias extends beyond flatness — it additionally favors high-coherence (simple) solutions.
- **vs. Andriushchenko et al. (2023)**: That empirical work observes that SAM learns low-rank representations; this paper provides a theoretical explanation.
- The coherence measure may inspire new optimizer designs — e.g., adaptive learning rates based on gradient alignment across mini-batches.
- The framework connects to the Neural Collapse phenomenon: the inter-class feature alignment observed in late-stage training can be understood as a high-coherence state.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Unifying the flatness and simplicity hypotheses via a coherence measure constitutes a genuinely insightful theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — Synthetic data validation is thorough, but real-data validation is limited (only approximate CIFAR-10 experiments).
- **Writing Quality**: ⭐⭐⭐⭐ — The theoretical narrative is clear; the theorem → discussion → empirical evidence structure is well-organized, though notation is heavy.
- **Value**: ⭐⭐⭐⭐ — Contributes deep insight into generalization mechanisms in deep learning, though practical utility is limited by the computational intractability of the coherence measure.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)

<!-- RELATED:END -->
