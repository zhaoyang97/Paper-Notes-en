---
title: >-
  [Paper Note] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks
description: >-
  [NeurIPS 2025][Optimization][scaling laws] This paper provides a precise analysis of online SGD learning of additive models (sums of single-index functions) in shallow neural networks, proving that the learning of each teacher neuron exhibits a sharp phase transition (emergence), and that the superposition of many such transition curves across different timescales naturally produces a smooth power-law scaling law.
tags:
  - NeurIPS 2025
  - Optimization
  - scaling laws
  - emergence
  - SGD
  - shallow neural networks
  - multi-index model
date: 2026-05-08
content_hash: e9ae5be87931b9f9
---

# Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2504.19983](https://arxiv.org/abs/2504.19983)
**Code**: None
**Area**: Optimization Theory / Neural Network Learning Theory
**Keywords**: scaling laws, emergence, SGD, shallow neural networks, multi-index model

## TL;DR
This paper provides a precise analysis of online SGD learning of additive models (sums of single-index functions) in shallow neural networks, proving that the learning of each teacher neuron exhibits a sharp phase transition (emergence), and that the superposition of many such transition curves across different timescales naturally produces a smooth power-law scaling law.

## Background & Motivation
**State of the Field**: A growing body of theoretical work has studied gradient-based training of shallow networks on low-dimensional target functions, particularly the sample complexity of SGD for single-index and multi-index models. Empirically, large-scale model training exhibits predictable power-law scaling laws, where the loss decreases smoothly as a function of compute or data.

**Limitations of Prior Work**: (a) SGD learning of a single skill or direction exhibits emergent behavior — a "search phase" followed by a sudden drop — which appears to contradict smooth scaling laws; (b) existing analyses of multi-index models are largely restricted to narrow width $P = O(1)$ or uniform signal strengths, which are insufficient to produce the timescale separation needed to explain power-law decay; (c) prior work (e.g., OSSW24) analyzes hierarchical training (first optimizing directions, then weights), requiring student width $m \gtrsim P^{\Omega(1/a_{\min})}$, which is computationally infeasible.

**Root Cause**: Emergence (discrete jumps) and scaling laws (smooth power laws) appear contradictory. The key challenge is to analyze the difficult regime of large width, large condition number, and single-phase training within a unified framework.

**Starting Point**: The target function is modeled as $f_*(x) = \sum_{p=1}^P a_p \sigma(\langle x, v_p^* \rangle)$, where $a_p \asymp p^{-\beta}$ follows a power-law decay. The paper exploits an "automatic cancellation" mechanism arising from 2-homogeneous parameterization to show that learning different directions can be approximately decoupled.

**Core Idea**: In single-phase SGD training, each teacher direction undergoes a sharp phase transition (emergent transition) at time $T_p \propto a_p^{-1}$. The superposition of $P \gg 1$ emergence curves across distinct timescales naturally produces a power-law scaling law $\mathcal{L}(t) \sim t^{(1-2\beta)/\beta}$.

## Method

### Overall Architecture
- **Teacher model**: $f_*(x) = \sum_{p=1}^P a_p \sigma(v_p^* \cdot x)$, $x \sim \mathcal{N}(0, I_d)$, $\{v_p^*\}$ orthonormal, $\sigma$ an even function with information exponent $k_* > 2$
- **Student model**: $f(x) = \sum_{k=1}^m \|v_k\|^2 \sigma(\bar{v}_k \cdot x)$, using 2-homogeneous parameterization (second-layer weight = squared norm of first-layer weight)
- **Training algorithm**: Online SGD with a fresh sample at each step, updating both layers simultaneously
- **Objective**: Prove polynomial sample complexity and precisely characterize the recovery time for each teacher neuron

### Key Designs

1. **2-Homogeneous Parameterization and Automatic Cancellation**

    - **Function**: The second-layer weights of the student network are set to $\|v_k\|^2$, coupling directional recovery with norm growth.
    - **Mechanism**: Once $\bar{v}_p$ converges to $v_{\pi(p)}^*$, $\|v_p\|^2$ automatically grows to $a_{\pi(p)}$, effectively canceling the corresponding teacher direction from the loss — analogous to automatic deflation.
    - **Design Motivation**: This avoids the drawback of hierarchical training, in which optimizing directions via a correlation loss introduces exponential dependence on the condition number $\kappa = a_{\max}/a_{\min}$. Single-phase MSE training circumvents this issue through automatic cancellation.

2. **Greedy Maximum Selection**

    - **Function**: Establishes a mapping $\pi$ from student neurons to teacher neurons, determining the learning order.
    - **Mechanism**: Directions are ordered by $a_{\pi(p)} \cdot \bar{v}_{p,\pi(p)}^{2I-2}(0)$; directions with larger signal strength and larger initial overlap are learned first.
    - **Key Property**: Three gap conditions — row gap, column gap, and threshold gap — guarantee that irrelevant coordinates remain small throughout training.

3. **Approximate Decoupled Dynamics**

    - **Function**: Proves that the learning processes for different teacher directions can be analyzed approximately independently.
    - **Mechanism**: The evolution of the aligned coordinate $\bar{v}_{p,\pi(p)}^2$ is approximated by the ODE $\frac{d}{dt} \bar{v}^2 \approx 8a_{\pi(p)} \bar{v}^4$, with solution $\bar{v}^2(t) = (1/\bar{v}^2(0) - 8a t)^{-1}$, exhibiting a sharp phase transition at $T_p \simeq (8 a_{\pi(p)} \bar{v}_{p,\pi(p)}^2(0))^{-1}$.
    - **Control of Irrelevant Coordinates**: Using the information exponent condition $k_* > 2$ (i.e., $2I > 2$), irrelevant coordinates $\bar{v}_{p,\pi(q)}$ grow more slowly than aligned ones and remain at the $O(d^{-0.9})$ level throughout training.

4. **Discretization from Gradient Flow to Online SGD**

    - **Function**: Translates the continuous-time gradient flow analysis into a rigorous proof for discrete SGD.
    - **Mechanism**: A martingale-plus-drift argument is employed, controlling stochastic terms via Doob's inequality. The learning rate $\eta \propto a_{\min} \Delta^2 d^{-I}$ is chosen so that the SGD escape time deviates from the gradient flow by at most a factor of $(1 \pm \Delta)$.
    - **Unstable Discretization Trick**: When only the recovery of the top $P_*$ directions is of interest, the learning rate can be set to $\eta \propto a_{P_*}$ (rather than $a_{\min}$), yielding improved compute–sample scaling.

### Loss & Training
- MSE loss: $\ell(x) = \frac{1}{2}(f_*(x) - f(x))^2$
- Via Hermite expansion, the population MSE can be expressed as a tensor decomposition loss.
- Online SGD uses an independent fresh sample at each step; the learning rate $\eta$ must satisfy precise conditions to guarantee convergence.

## Key Experimental Results

### Main Results (Theory vs. Experiment)

| Setting | Theoretical Scaling | Observed | Notes |
|---|---|---|---|
| Fixed learning rate, $\beta=0.8$ | $\mathcal{L} \sim (mt)^{(1-2\beta)/(1+\beta)} = (mt)^{-1/3}$ | Slope $\approx -1/3$ | Compute-optimal frontier matches |
| Sample scaling | $\mathcal{L} \sim n^{(1-2\beta)/\beta}$ | Consistent | Matches minimax optimal rate |
| Width scaling | Approximation error $\sim m^{1-2\beta}$ | Consistent | Student width $m$ determines how many directions can be learned |

### Ablation Study

| Parameter | Effect | Notes |
|---|---|---|
| Information exponent $k_*$ | Sample complexity $\propto d^{k_*-1}$ | Larger $k_*$ requires more samples to enter the search phase |
| Power-law exponent $\beta$ | Scaling slope $= (2\beta-1)/\beta$ | Scaling law holds when $\beta > 1/2$ (loss is square-summable) |
| Condition number $\kappa$ | Ours: polynomial dependence vs. prior: exponential dependence | Core improvement |
| Student width $m$ | $m = \tilde{\Theta}(P_*)$ suffices | Only logarithmic overparameterization required |

### Key Findings
- **Individual learning curves are staircase-shaped** (emergence), but the superposition of $P \gg 1$ staircases yields a smooth power law.
- For $d = 2048$, $P = 1024$, $\sigma = h_4$, the theoretical and empirical compute-optimal frontier slopes are in close agreement.
- The unstable discretization scheme yields sample scaling exponents consistent with the minimax optimal rate for Gaussian sequence models.

## Highlights & Insights
- **A clean theoretical explanation of Emergence → Scaling Law**: Prior theories of scaling laws either assume linear models or fully decoupled tasks. This work is the first to rigorously establish this connection in a nonlinear feature-learning setting.
- **Automatic cancellation mechanism**: The 2-homogeneous parameterization causes already-learned directions to be automatically removed from the loss, elegantly circumventing the exponential dependence on the condition number that afflicts hierarchical training.
- **Single-phase training outperforms hierarchical training**: Counter-intuitively, simultaneously updating both layers is more efficient than first optimizing directions and then weights, reducing sample complexity from exponential to polynomial.
- **Unstable discretization**: Choosing a learning rate that is "too large" for weak-signal directions — sacrificing accurate tracking of those directions in exchange for faster convergence on strong-signal directions — is a transferable idea for practical adaptive learning rate design.

## Limitations & Future Work
- **Even activation functions**: The analysis assumes $\sigma$ is even, excluding common activations such as ReLU (whose information exponent is 1, violating $k_* > 2$).
- **Orthogonal teacher directions**: The framework requires $\{v_p^*\}$ to be orthogonal, whereas feature directions in practical models may be highly correlated.
- **Theory–practice gap**: The analysis holds asymptotically as $d \to \infty$; at finite dimension, the empirical scaling law slope may deviate from theoretical predictions.
- **Online SGD only**: Mini-batch SGD and practically used optimizers such as Adam are not analyzed.

## Related Work & Insights
- **vs. OSSW24 (hierarchical training)**: That work requires width $m \gtrsim P^{\Omega(1/a_{\min})}$, whereas this paper requires only $m = \tilde{O}(P)$. The key distinction is the automatic cancellation mechanism enabled by single-phase training.
- **vs. MLGT24, NFLL24 (additive model intuition)**: These works propose the intuition that "superposition of multiple skills produces scaling laws," but assume fully independent tasks. This paper is the first to rigorously prove approximate decoupling in a nonlinearly coupled setting.
- **vs. BAP24, LWK+24 (linear model scaling laws)**: Those works analyze scaling laws in linear models or the kernel regime; this paper extends the analysis to nonlinear feature learning. While the functional form of the scaling exponents is consistent, the underlying mechanisms differ.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First theoretical result establishing the emergence → scaling law connection in a nonlinear feature-learning setting.
- **Experimental Thoroughness**: ⭐⭐⭐ Primarily theoretical; numerical experiments only validate the scaling slope, with no large-scale empirical evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematically rigorous; proof sketches in the main text effectively convey the core ideas.
- **Value**: ⭐⭐⭐⭐⭐ Provides a new theoretical perspective on scaling laws with significant implications for the theory community.

## Additional Remarks
- The theoretical framework and technical tools developed in this paper offer insights for adjacent research areas.
- The core contribution lies in providing a deep theoretical understanding that lays the groundwork for subsequent practical optimization advances.
- The paper is methodologically complementary to other NeurIPS 2025 papers published concurrently.
- The exposition of problem motivation and technical approach is exemplary and worth studying.
- Readers are encouraged to consult the appendix for complete experimental details and full proofs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](../../ICLR2026/optimization/scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

<!-- RELATED:END -->
