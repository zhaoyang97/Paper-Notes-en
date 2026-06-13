---
title: >-
  [Paper Note] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks
description: >-
  [ICML 2026][Optimization][Linear RNNs] This paper derives analytical ODEs for updates in BPTT, one-step tBPTT, and RFLO on student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Linear RNNs"
  - "RFLO"
  - "tBPTT"
  - "learning dynamics"
  - "low-rank constraints"
date: 2026-05-08
content_hash: cb9ccce60017f55c
---

# Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2606.00243](https://arxiv.org/abs/2606.00243)  
**Code**: To be confirmed  
**Area**: Optimization / Learning Theory / Neuroscience  
**Keywords**: Linear RNNs, RFLO, tBPTT, learning dynamics, low-rank constraints

## TL;DR
This paper derives analytical ODEs for updates in BPTT, one-step tBPTT, and RFLO on student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability, and convergence rates, it finds that RFLO lacks the non-optimal saddle manifold found in BPTT/tBPTT, but at the cost of stability dependency on signs and slower convergence. Crucially, RFLO is **restricted to low-rank perturbations of initial weights**—a limitation that generalizes to settings without data alignment.

## Background & Motivation

**Background**: The gold standard for training RNNs is BPTT (Backpropagation Through Time), but BPTT is non-local in both space and time—updates depend on distant hidden states and errors from much earlier moments. In neuroscience, this non-locality makes BPTT biologically implausible; for neuromorphic hardware, non-local memory access is a major bottleneck for deployment. Consequently, the community has developed "local approximation" algorithms: one-step tBPTT (truncating BPTT to $\tau$ steps), RFLO (replacing the RTRL Jacobian product with a diagonal matrix + random feedback), and e-prop (modifying RFLO with a diagonal $W$).

**Limitations of Prior Work**: These local algorithms **are not true gradients of any objective function**, meaning there is no theoretical guarantee they follow a loss descent path, nor that they converge to the same solutions as BPTT. The community relies on sporadic empirical comparisons and lacks a systematic analysis of basic learning dynamics properties like "fixed-point structures, stability, and convergence rates."

**Key Challenge**: Analyzing non-gradient, non-linear learning dynamics requires a sufficiently tractable setting that preserves the core difficulty of RNN temporal structures while allowing solvable ODE derivations.

**Goal**: To place BPTT, one-step tBPTT, and RFLO within a single mathematical framework to answer: (i) What do the fixed points look like? (ii) Which fixed points are stable? (iii) What are the convergence rates near the optimal manifold? (iv) What structural features do the learned solutions possess in the representation space?

**Key Insight**: The paper repurposes the "data-aligned linear RNN" framework designed by Proca et al. for BPTT, where the input/output/recurrent matrices of both student and teacher are jointly diagonalized under the same orthogonal basis. This reduces the learning dynamics of an $n$-dimensional RNN into $n$ uncoupled 3D ODEs (each mode having only three scalar parameters: $(a, b, w)$).

**Core Idea**: The updates for tBPTT and RFLO are mapped into this diagonalization framework. By taking the double limit of $T\to\infty$ and $\eta\to0$, three sets of comparable ODEs are obtained and analyzed using standard dynamical systems tools (fixed points + Jacobian linearization + numerical integration).

## Method

### Overall Architecture
Student–teacher linear RNNs are driven by the same Gaussian white noise $x_t\sim\mathcal{N}(0,\mathbf{I})$. The student updates are $h_{t+1}=Wh_t+Bx_t,\;y_{t+1}=Ah_{t+1}$, while teacher parameters are denoted with a $\star$ superscript. The loss is the expectation of the final-step error $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$. Updates follow the form $\theta_{k+1}=\theta_k-\eta\Delta\theta_k$:

- **BPTT**: $\Delta W=\sum_{t=1}^{T}(W^{T-t})^\top A^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.
- **one-step tBPTT** ($\tau=1$): Keeps only the $t=T$ term, i.e., $\Delta_\tau W=A^\top\mathbb{E}[\varepsilon_T h_{T-1}^\top]$.
- **RFLO**: Replaces $W^{T-t}$ with $\widehat{W}^{T-t}=\hat w^{T-t}\mathbf{I}$ (scalar times identity) and $A^\top$ with a fixed random feedback $R^\top$, yielding $\Delta_{\mathrm{RFLO}} W=\sum_t(\widehat W^{T-t})^\top R^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.

Data-alignment assumes the input–output correlation matrix $\Sigma_t^\star=\mathbb{E}[y_T^\star x_t^\top]$ can be decomposed as $\Sigma_t^\star=U S_t V^\top$, and both teacher $(A_\star,W_\star,B_\star)$ and student $(A_0,W_0,B_0)$ are initialized jointly diagonalized under $(U,V,P_\star)$. After alignment, each student mode $(a,b,w)$ evolves independently from others, decomposing the high-dimensional problem into $n$ uncoupled 3D ODEs. RFLO random feedback $R=U\bar R P^\top$ is also incorporated into this diagonalization.

By taking $T\to\infty$, the sum of updates is written in closed form (using geometric series $\sum w^t$), and $\eta\to0$ converts discrete updates into ODEs $\dot\theta=-\Delta\theta$ (Eqs. 19–22). The shared $a$-direction update is $\Delta a\to \tfrac{ab^2}{1-w^2}-\tfrac{a_\star b b_\star}{1-w w_\star}$. The $w,b$ directions vary; for instance, in RFLO, $\Delta_{\mathrm{RFLO}} b\to \tfrac{\hat a a b}{1-\hat w w}-\tfrac{\hat a a_\star b_\star}{1-\hat w w_\star}$. Since $\hat a$ is in the numerator, $b=0$ does not automatically lead to $\Delta b=0$, which is why RFLO lacks the non-optimal manifold.

### Key Designs

1. **Data-Alignment + Double Limit Diagonalizes RNN Learning to 3D ODEs**:
    - **Function**: Simplifies high-dimensional, non-linear, time-coupled learning dynamics into $n$ independent $(a,b,w)$ 3D ODEs, making fixed points, stability, and convergence rates analytical.
    - **Mechanism**: Building on the BPTT diagonalization of Proca et al. (2025), the key extension is proving tBPTT and RFLO can be diagonalized under the same orthogonal basis—provided RFLO's random feedback is $R=U\bar R P^\top$ and $\widehat W=\hat w\mathbf{I}$. Summing the geometric series $\sum_{t=0}^{\infty} w^t=1/(1-w)$ yields rational fractional ODEs like $\Delta a=\tfrac{ab^2}{1-w^2}-\tfrac{a_\star bb_\star}{1-ww_\star}$.
    - **Design Motivation**: Analyzing high-dimensional non-linear ODEs is nearly impossible. Diagonalization preserves the core temporal difficulty (recursion $W^t$ appears in the denominator via geometric series) while allowing pointwise Jacobian linearization.

2. **Fixed-Point Structure Comparison: RFLO Lacks a Saddle Manifold**:
    - **Function**: Identifies all fixed points for each algorithm and classifies them by geometric structure.
    - **Mechanism**: Solving $\dot w, \dot a, \dot b = 0$ as $T\to\infty$. All three algorithms share an **optimal manifold** $\{ab=a_\star b_\star,\;w=w_\star\}$ (zero loss curve). BPTT and tBPTT have an additional **non-optimal manifold** $\{a=b=0,\;w\text{ arbitrary}\}$, where loss persists at a strictly positive value. RFLO does not have this manifold because its $\Delta_{\mathrm{RFLO}} b$ is non-zero at $a=b=0$ due to the $\hat a a_\star b_\star/(1-\hat w w_\star)$ term.
    - **Design Motivation**: Fixed-point structures define where learning "stops." The extra saddle line in BPTT/tBPTT implies that initializations near it will experience a slow phase (traversing near the saddle point).

3. **Stability + Convergence Rate: RFLO Sacrifices Speed for Low-Rank Solutions**:
    - **Function**: Qualifies how fast and stable learning is near each optimal branch via Jacobian eigenvalues.
    - **Mechanism**: On the optimal manifold, BPTT's Jacobian eigenvalues are real and negative. RFLO's stability depends on $\mathrm{sgn}(\hat a s)$: if $\hat a s>0$, it is stable; if $\hat a s<0$, the leading eigenvalue $\lambda_+$ can be positive, causing instability or oscillation (Eqs. 28–29). Furthermore, Proposition 3.1 proves the RFLO update $W_K-W_0$ is restricted to a rank at most $o$ (output dimension).
    - **Design Motivation**: This explains the "detour" behavior of RFLO in Fig 4 and the lower-rank $W_K-W_0$ observed for local rules (RFLO, tBPTT, e-prop) in Fig 6—expressivity is constrained by the locality of the rules.

## Key Experimental Results

### Main Results
Student RNNs (non-data-aligned) learning a teacher (4 modes). Comparisons of experimental trajectories with theoretical ODE predictions.

| Algorithm | Fixed-Point Manifold | Stability | Convergence Rate (on Optimal Manifold) |
|-----------|----------------------|-----------|----------------------------------------|
| BPTT      | Optimal (cyan) + Non-optimal (red, saddle) | Optimal stable, Non-optimal saddle | Fastest; slowest at $s=\pm\sqrt{|a_\star b_\star|}$ |
| tBPTT ($\tau=1$) | Same as BPTT | Same as BPTT | Close to BPTT when $\|w_\star\|$ is small |
| RFLO      | Optimal only | Sign-dependent: $\hat a s>0$ stable / $\hat a s<0$ unstable | Slower for most $s$; faster only as $s\to 0$ |

### Ablation Study (Solution Rank)
| Algorithm | $W_K-W_0$ Spectral Shape | Explanation |
|-----------|--------------------------|-------------|
| BPTT      | High rank (multiple singular values) | No local constraints |
| e-prop    | Medium rank | Propagation using diagonal $W$ |
| tBPTT ($\tau=1$) | Low rank, close to RFLO | Uses only final-step error |
| RFLO      | Strictly rank $\le o$ (here $=1$) | Proposition 3.1: $W_K=W_0+\sum_{i=1}^o r_iq_i^\top$ |

### Key Findings
- **Theory extrapolates to non-aligned settings**: After a short transient, alignment (recurrent, input/output, feedback) increases; ODE predictions match numerical experiments for dominant modes, suggesting alignment is an emergent property.
- **RFLO sacrifices speed for stability**: Lacking non-optimal saddles is an advantage, but stability depends on $\mathrm{sgn}(\hat a s)$. Fig 4 shows RFLO often takes long "detours" to the opposite optimal branch, increasing wall-clock time.
- **Local Rules $\Rightarrow$ Low-Rank Solutions**: All local algorithms produce lower-rank $W_K-W_0$ than BPTT. In 20 simulations, RFLO converged only 13 times, while BPTT/tBPTT always converged.
- **Transfer to SSMs**: Since linear RNNs share expressivity with modern State Space Models (Mamba), RFLO-like local rules may impose similar low-rank constraints on SSMs, affecting neuromorphic training.

## Highlights & Insights
- **Unified theoretical framework**: Simultaneously covers three algorithms and three properties (geometry, stability, rank) in a single linear RNN study—a significant advance over purely empirical comparisons.
- **Mechanistic explanation for "impoverished" RFLO solutions**: Proposition 3.1 shows the rank is strictly bounded by output dimension $o$, advising hardware researchers to use higher-rank random feedback to learn complex solutions.
- **Emergent alignment**: The paper justifies the "data-alignment" assumption by showing it arises naturally during training ("the assumption is the result").
- **Transferable design**: Mapping non-gradient rules into an ODE framework to compare them with true gradients serves as a template for studying surrogate gradients (SNNs, synthetic gradients).

## Limitations & Future Work
- Restricted to linear RNNs; the mathematical tricks (geometric series, diagonalization) do not directly apply to non-linear dynamics.
- The double limit ($T\to\infty$, $\eta\to0$) ignores finite batch/step effects and noise which might alter stability.
- The "detour" behavior of RFLO is shown in small examples; its impact on large-scale SSMs needs verification.
- e-prop uses diagonal $W$ rather than scalar; its exact rank bounds remain to be derived.

## Related Work & Insights
- **vs. Proca et al. (2025)**: Extends their BPTT diagonalization to non-gradient algorithms tBPTT and RFLO.
- **vs. Saxe et al. (2014, 2019)**: Moves the "solvable dynamics" philosophy from feedforward networks to the RNN domain by handling the temporal geometric series.
- **vs. Murray (2019)**: Provides mechanistic explanations for RFLO's behavior where the original work provided only numerical evidence.

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
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)

</div>

<!-- RELATED:END -->
