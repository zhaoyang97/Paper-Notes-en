---
title: >-
  [Paper Note] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks
description: >-
  [ICML 2026][Optimization & Theory][RFLO] This paper derives analytical ODEs for the updates of BPTT, one-step tBPTT, and RFLO in student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability, and convergence rates, it is found that RFLO lacks the non-optimal saddle manifold of BPTT/tBPTT but at the cost of sign-dependent stabil
tags:
  - ICML 2026
  - Optimization & Theory
  - RFLO
  - tBPTT
date: 2026-05-08
content_hash: ff92969505c17649
---
# Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2606.00243](https://arxiv.org/abs/2606.00243)  
**Code**: To be confirmed  
**Area**: Optimization / Learning Theory / Neuroscience  
**Keywords**: Linear RNN, RFLO, tBPTT, Learning Dynamics, Low-rank constraints

## TL;DR
This paper derives analytical ODEs for the updates of BPTT, one-step tBPTT, and RFLO in student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability, and convergence rates, it is found that RFLO lacks the non-optimal saddle manifold of BPTT/tBPTT but at the cost of sign-dependent stability and slower convergence. Furthermore, RFLO is **intrinsically limited to low-rank perturbations of initial weights**, a constraint that generalizes to non-data-aligned settings.

## Background & Motivation

**Background**: The gold standard for training RNNs is Backpropagation Through Time (BPTT). However, BPTT is non-local in both space and time; updates depend on distant hidden states and errors from far earlier timesteps. For neuroscience, this non-locality makes BPTT biologically implausible. For neuromorphic hardware, non-local memory access hinders deployment. Consequently, the community has developed "local approximation" algorithms: one-step tBPTT (truncating BPTT to $\tau$ steps), RFLO (replacing the RTRL Jacobian product with diagonal matrices and random feedback), and e-prop (modifying RFLO for diagonal $W$).

**Limitations of Prior Work**: These local algorithms **are not true gradients of any objective function**. Thus, there are no theoretical guarantees that they descend the loss or converge to the same solutions as BPTT. The community relies on sporadic empirical comparisons and lacks a systematic analysis of basic learning dynamics such as fixed-point structures, stability, and convergence rates.

**Key Challenge**: Analyzing non-gradient, non-linear learning dynamics requires a sufficiently tractable setup that retains the core temporal complexity of RNNs while allowing for solvable ODE derivations.

**Goal**: To place BPTT, one-step tBPTT, and RFLO within a single mathematical framework to answer: (i) What do the fixed points look like? (ii) Which fixed points are stable? (iii) What are the relative convergence rates near the optimal manifold? (iv) What structural characteristics exist in the learned representation space?

**Key Insight**: The authors leverage the "data-aligned linear RNN" framework proposed by Proca et al. for BPTT. In this framework, the input, output, and recurrent matrices of both student and teacher are jointly diagonalized under the same orthogonal basis, reducing the learning dynamics of an $n$-dimensional RNN into $n$ decoupled 3D ODEs (where each mode involves only three scalar parameters: $(a,b,w)$).

**Core Idea**: The updates for tBPTT and RFLO are mapped into this diagonalization framework. By taking the dual limit of $T\to\infty$ and $\eta\to0$, three sets of comparable ODEs are obtained. These are then analyzed using standard dynamical systems tools (fixed points, Jacobian linearization, and numerical integration).

## Method

### Overall Architecture
The student–teacher linear RNN is driven by Gaussian white noise $x_t\sim\mathcal{N}(0,\mathbf{I})$. The student is defined by $h_{t+1}=Wh_t+Bx_t,\;y_{t+1}=Ah_{t+1}$, while teacher parameters are denoted with $\star$. The loss is the expectation of the final-step error $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$. Updates follow $\theta_{k+1}=\theta_k-\eta\Delta\theta_k$:

- **BPTT**: $\Delta W=\sum_{t=1}^{T}(W^{T-t})^\top A^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.
- **one-step tBPTT** ($\tau=1$): Retains only the $t=T$ term, i.e., $\Delta_\tau W=A^\top\mathbb{E}[\varepsilon_T h_{T-1}^\top]$.
- **RFLO**: Replaces $W^{T-t}$ with $\widehat{W}^{T-t}=\hat w^{T-t}\mathbf{I}$ and $A^\top$ with fixed random feedback $R^\top$, yielding $\Delta_{\mathrm{RFLO}} W=\sum_t(\widehat W^{T-t})^\top R^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.

Data alignment assumes the input-output correlation matrix $\Sigma_t^\star=\mathbb{E}[y_T^\star x_t^\top]$ can be decomposed as $\Sigma_t^\star=U S_t V^\top$ ($U,V$ orthogonal, $S_t$ diagonal), and teacher $(A_\star,W_\star,B_\star)$ and student $(A_0,W_0,B_0)$ are jointly diagonalized at initialization. RFLO random feedback $R=U\bar R P^\top$ is also incorporated.

Taking the $T\to\infty$ limit allows closed-form summation (using $\sum w^t = 1/(1-w)$), and $\eta\to0$ yields ODEs $\dot\theta=-\Delta\theta$. The common $a$-direction update is $\Delta a\to \tfrac{ab^2}{1-w^2}-\tfrac{a_\star b b_\star}{1-w w_\star}$. The $w$ and $b$ directions vary; for RFLO, $\Delta_{\mathrm{RFLO}} b\to \tfrac{\hat a a b}{1-\hat w w}-\tfrac{\hat a a_\star b_\star}{1-\hat w w_\star}$. Since $\hat a$ is in the numerator, $b=0$ does not automatically imply $\Delta b=0$, which is the fundamental reason RFLO lacks the non-optimal manifold.

### Key Designs

**1. Data alignment + dual limits diagonalize RNN learning into 3D ODEs: A unified base**
Direct analysis of high-dimensional, non-linear, temporally coupled learning dynamics is nearly impossible. Building on Proca et al. (2025), the authors prove that tBPTT and RFLO can also be diagonalized under the same orthogonal basis. This reduces the $n$-dimensional RNN into $n$ independent 3D systems of $(a,b,w)$. By taking the $T\to\infty$ limit, the updates become rational fraction ODEs such as:
$$\Delta a=\frac{ab^2}{1-w^2}-\frac{a_\star b b_\star}{1-w w_\star}$$
This 3D ODE retains the core temporal complexity (recurence $W^t$ remains in the denominator via the geometric series) while allowing for point-wise Jacobian linearization.

**2. Fixed-point structure comparison: RFLO lacks a non-optimal saddle manifold**
Fixed points define where learning stops. All three algorithms share an optimal manifold $\{ab=a_\star b_\star,\ w=w_\star\}$ where loss is minimized. However, BPTT and tBPTT possess an additional non-optimal manifold $\{a=b=0,\ w\text{ is arbitrary}\}$, where loss remains positive. RFLO lacks this because its $\Delta_{\mathrm{RFLO}} b$ is non-zero even at $a=b=0$ due to the term $\hat a a_\star b_\star/(1-\hat w w_\star)$. While this prevents sticking near saddle points, it introduces other trade-offs.

**3. Stability and convergence rate: RFLO sacrifices speed for low-rank solutions**
Linearizing the Jacobian on the optimal manifold reveals stability and speed. BPTT's Jacobian eigenvalues have negative real parts; its speed is determined by $\lambda_+$. RFLO's eigenvalues explicitly depend on $\mathrm{sgn}(\hat a s)$. It is stable when $\hat a s>0$ but may become unstable or oscillatory when $\hat a s < 0$. Furthermore, Proposition 3.1 proves RFLO updates are strictly low-rank: $W_K=W_0+\sum_{i=1}^o r_i q_i^\top$, with rank at most the output dimension $o$.

### Loss & Training
The theory utilizes $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$, with extensions to sequential loss $\mathcal{L}=\tfrac{1}{2T}\sum_{t=1}^T\|y_t-y_t^\star\|^2$ in Appendix H. Small-variance Gaussian initialization (Saxe et al. 2019 style) is used.

## Key Experimental Results

### Main Results
Comparison between experimental trajectories and theoretical ODE predictions in non-data-aligned students learning a teacher with 4 modes.

| Algorithm | Fixed-point Manifold | Stability | Conv. Rate (on Optimal Manifold) |
|-----------|----------------------|-----------|----------------------------------|
| BPTT      | Optimal (cyan) + Non-optimal (red, saddle) | Optimal: Stable; Non-optimal: Saddle | Fastest; slowest at $s=\pm\sqrt{|a_\star b_\star|}$ |
| tBPTT ($\tau=1$) | Same as BPTT | Same as BPTT | Close to BPTT when $\|w_\star\|$ is small |
| RFLO      | Optimal only | Sign-dependent: $\hat a s>0$ stable / $\hat a s<0$ unstable | Slowest for most $s$, except $s\to 0$ |

### Ablation Study (Rank)

| Algorithm | Spectral Shape of $W_K-W_0$ | Explanation |
|-----------|-----------------------------|-------------|
| BPTT      | High rank (multiple sig. singular values) | No locality constraints |
| e-prop    | Intermediate rank | Uses diagonal $W$ for backprop |
| tBPTT ($\tau=1$) | Low rank (close to RFLO) | Uses only the final-step error |
| RFLO      | Strictly rank $\le o$ (here $=1$) | Prop. 3.1: $W_K=W_0+\sum_{i=1}^o r_iq_i^\top$ |

### Key Findings
- **Theory extrapolates to non-aligned settings**: Alignment metrics increase during training; ODE predictions match numerical experiments for dominant modes, suggesting data alignment is an "emergent" property.
- **RFLO speed-stability trade-off**: Fewer fixed points seem advantageous, but stability becomes sign-dependent. RFLO often follows long trajectories to reach a stable branch, increasing wall-clock time.
- **Local rules $\implies$ low-rank solutions**: RFLO, tBPTT, and e-prop all produce lower-rank updates than BPTT. The rank is highly correlated with stability and performance.
- **Transferability to SSMs**: Since linear RNNs share expressive classes with modern State Space Models (Mamba, etc.), local rules may impose similar low-rank constraints on SSMs trained on neuromorphic hardware.

## Highlights & Insights
- **Unified Theoretical Framework**: Simultaneously explains fixed-point geometry, Jacobian stability, and solution rank within a single framework—offering deeper insights than empirical comparisons.
- **Mechanistic explanation for "impoverished" RFLO solutions**: Proposition 3.1 proves the rank limit depends only on the output dimension $o$ and the scalar feedback structure, independent of data alignment. This suggests neuromorphic systems need higher-rank feedback structures to learn complex solutions.
- **Legitimizing data alignment**: The discovery that alignment emerges during training validates the use of this framework, which was previously criticized as too restrictive.
- **Transferable design strategy**: The template of "mapping non-gradient rules to an ODE framework" is a powerful tool for studying surrogate gradients, such as those in spiking networks or quantized backprop.

## Limitations & Future Work
- Limited to linear RNNs; non-linear learning dynamics involve significantly more complex sum-of-product terms.
- The dual limit $T\to\infty$ and $\eta\to0$ ignores finite-batch and learning rate schedule effects.
- The "long-range detour" behavior of RFLO needs validation on larger-scale tasks and longer sequences.
- e-prop uses diagonal $W$ feedback; its theoretical rank bound remains to be derived formally.

## Related Work & Insights
- **vs. Proca et al. (2025)**: Extends their BPTT diagonalization to non-gradient algorithms (tBPTT, RFLO) and compares structural differences.
- **vs. Saxe et al. (2014, 2019)**: Moves from feedforward dynamics to the RNN domain by explicitly handling the geometric series arising from temporal recurrence.
- **vs. Murray (2019)**: Provides the analytical explanation (lack of non-optimal manifolds and rank bounds) for original RFLO empirical findings.
- **vs. Bellec et al. (2020)**: Shows e-prop's rank is higher than RFLO's but still constrained compared to BPTT, positioning it within a "local rule" hierarchy.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
