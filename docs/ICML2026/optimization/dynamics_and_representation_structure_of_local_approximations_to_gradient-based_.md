---
title: >-
  [Paper Note] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks
description: >-
  [ICML 2026][Optimization & Theory][RFLO] This paper formulates the updates of BPTT, one-step tBPTT, and RFLO into analytical ODEs using student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability, and convergence rates, it is found that RFLO lacks the non-optimal saddle manifold present in BPTT/tBPTT. However, this comes at t
tags:
  - ICML 2026
  - Optimization & Theory
  - RFLO
  - tBPTT
date: 2026-05-08
content_hash: ef21a12c16ee1cc0
---
# Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2606.00243](https://arxiv.org/abs/2606.00243)  
**Code**: To be confirmed  
**Area**: Optimization / Learning Theory / Neuroscience  
**Keywords**: Linear RNN, RFLO, tBPTT, Learning Dynamics, Low-rank constraints

## TL;DR
This paper formulates the updates of BPTT, one-step tBPTT, and RFLO into analytical ODEs using student–teacher data-aligned linear RNNs. By comparing their fixed-point manifolds, stability, and convergence rates, it is found that RFLO lacks the non-optimal saddle manifold present in BPTT/tBPTT. However, this comes at the cost of stability being sign-dependent, slower convergence, and a **restriction to low-rank perturbations of initial weights**—a limitation that generalizes to non-data-aligned settings.

## Background & Motivation

**Background**: The gold standard for training RNNs is BPTT (Backpropagation Through Time), which is non-local in both space and time—updates depend on distant hidden states and errors from far earlier time steps. For neuroscience, this non-locality makes BPTT difficult to explain biological brain learning; for neuromorphic hardware, non-local memory access is a major bottleneck for deployment. Consequently, the community has developed several "local approximation" algorithms: one-step tBPTT (truncating BPTT to $\tau$ steps), RFLO (replacing the RTRL Jacobian product with diagonal matrices + random feedback), and e-prop (an adaptation of RFLO with diagonal $W$).

**Limitations of Prior Work**: These local algorithms **are not true gradients of any objective function**. Thus, there are no theoretical guarantees that they follow the loss descent or converge to the same solutions as BPTT. The community largely relies on scattered empirical comparisons, lacking a systematic analysis of fundamental learning dynamics properties such as fixed-point structures, stability, and convergence rates.

**Key Challenge**: Analyzing non-gradient, non-linear learning dynamics requires a sufficiently tractable setting that retains the core difficulties posed by the temporal structure of RNNs while remaining solvable via ODE derivations.

**Goal**: This work aims to place BPTT, one-step tBPTT, and RFLO within a single mathematical framework to answer: (i) What do the fixed points look like? (ii) Which fixed points are stable? (iii) What are the relative convergence speeds near the optimal manifold? (iv) What structural features do the learned solutions possess in the representation space?

**Key Insight**: The authors adapt the "data-aligned linear RNN" framework designed by Proca et al. for BPTT, where the input/output/recurrent matrices of student and teacher are jointly diagonalized under the same orthogonal basis. This reduces the learning dynamics of an $n$-dimensional RNN into $n$ uncoupled three-dimensional ODEs (each mode having only three scalar parameters: $(a,b,w)$).

**Core Idea**: By reformulating the update rules of tBPTT and RFLO into this diagonalization framework and taking the dual limits of $T\to\infty$ and $\eta\to0$, three sets of comparable ODEs are derived. These are then analyzed using standard dynamical system tools (fixed points, Jacobian linearization, and numerical integration).

## Method

### Overall Architecture
The student–teacher linear RNN is driven by the same Gaussian white noise $x_t\sim\mathcal{N}(0,\mathbf{I})$. The student updates as $h_{t+1}=Wh_t+Bx_t,\;y_{t+1}=Ah_{t+1}$, while teacher parameters are denoted with $\star$. The loss is the expectation of the final step error $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$. Update rules for all three algorithms follow the form $\theta_{k+1}=\theta_k-\eta\Delta\theta_k$:

- **BPTT**: $\Delta W=\sum_{t=1}^{T}(W^{T-t})^\top A^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.
- **one-step tBPTT** ($\tau=1$): Keeps only the $t=T$ term: $\Delta_\tau W=A^\top\mathbb{E}[\varepsilon_T h_{T-1}^\top]$.
- **RFLO**: Replaces $W^{T-t}$ with $\widehat{W}^{T-t}=\hat w^{T-t}\mathbf{I}$ (scalar times identity) and $A^\top$ with a fixed random feedback $R^\top$, yielding $\Delta_{\mathrm{RFLO}} W=\sum_t(\widehat W^{T-t})^\top R^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$.

The data alignment hypothesis requires that the input–output correlation matrix $\Sigma_t^\star=\mathbb{E}[y_T^\star x_t^\top]$ can be decomposed as $\Sigma_t^\star=U S_t V^\top$ ($U,V$ orthogonal, $S_t$ diagonal), and that both teacher $(A_\star,W_\star,B_\star)$ and student $(A_0,W_0,B_0)$ are jointly diagonalized at initialization. After alignment, each student mode $(a,b,w)$ evolves independently of its corresponding teacher mode $(a_\star,b_\star,w_\star)$.

Taking the $T\to\infty$ limit allows the summation to be written in closed form (using geometric series $\sum w^t$), and taking $\eta\to0$ converts discrete updates into ODEs $\dot\theta=-\Delta\theta$ (Equations 19–22). The common $a$-direction update is $\Delta a\to \tfrac{ab^2}{1-w^2}-\tfrac{a_\star b b_\star}{1-w w_\star}$, while $w$ and $b$ directions vary by algorithm. For instance, in RFLO, $\Delta_{\mathrm{RFLO}} b\to \tfrac{\hat a a b}{1-\hat w w}-\tfrac{\hat a a_\star b_\star}{1-\hat w w_\star}$. Since $\hat a$ is in the numerator, $b=0$ does not automatically lead to $\Delta b=0$, which is the fundamental reason RFLO lacks the non-optimal manifold.

### Key Designs

**1. Data Alignment + Dual Limits Diagonalize RNN Learning into 3D ODEs: A Unified Base**

Analyzing high-dimensional, non-linear, temporally coupled learning dynamics directly is intractable. The authors build upon the framework of Proca et al. (2025), proving that tBPTT and RFLO can also be diagonalized under the same orthogonal basis—provided the RFLO random feedback is defined as $R=U\bar R P^\top$ and $\widehat W=\hat w\mathbf{I}$. This diagonalization decomposes the $n$-dimensional RNN into $n$ independent $(a,b,w)$ 3D systems. Taking the $T\to\infty$ limit using $\sum_{t=0}^\infty w^t=1/(1-w)$ yields rational function ODEs such as:

$$\Delta a=\frac{ab^2}{1-w^2}-\frac{a_\star b b_\star}{1-w w_\star}$$

This allows for point-wise Jacobian linearization, serving as the mathematical base for the entire analysis.

**2. Fixed-point Structure Comparison: RFLO Lacks a Non-optimal Saddle Manifold**

Fixed points define where learning can stop. All three algorithms share an optimal manifold $\{ab=a_\star b_\star,\ w=w_\star\}$ where the loss is minimized. However, BPTT and tBPTT possess an additional non-optimal manifold $\{a=b=0,\ w\text{ is arbitrary}\}$, where the loss remains strictly positive at $\sum a_\star^2 b_\star^2/[2(1-w_\star^2)]$. RFLO lacks this manifold because its $\Delta_{\mathrm{RFLO}} b$ is non-zero even at $a=b=0$ due to the constant term $\hat a a_\star b_\star/(1-\hat w w_\star)$. While this avoids getting stuck in certain saddle points, it introduces stability costs.

**3. Stability and Convergence Rate: RFLO Sacrifices Speed for Low-rank Solutions**

Linearizing the Jacobian on the optimal manifold allows for the quantification of stability and speed. For BPTT, the Jacobian eigenvalues have negative real parts, with the largest eigenvalue $\lambda_+$ determining speed (slowest at $s=\pm\sqrt{|a_\star b_\star|}$). RFLO's stability explicitly depends on $\mathrm{sgn}(\hat a s)$; if $\hat a s<0$, $\lambda_+$ can become positive, leading to instability or oscillation (Equations 28–29). Furthermore, Proposition 3.1 proves that RFLO updates are necessarily low-rank: $W_K=W_0+\sum_{i=1}^o r_i q_i^\top$ and $B_K=B_0+\sum_{i=1}^o r_i q_i^{(b)\top}$, with rank at most the output dimension $o$.

### Loss & Training
The theory utilizes $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$, and Appendix H extends the results to sequence loss $\mathcal{L}=\tfrac{1}{2T}\sum_{t=1}^T\|y_t-y_t^\star\|^2$. Experiments use small-variance Gaussian initialization for the student.

## Key Experimental Results

### Main Results
A non-data-aligned student RNN learns a mode-aligned teacher (4 modes). Trajectories are compared against ODE predictions.

| Algorithm | Fixed-point Manifolds | Stability | Convergence Speed (on Optimal Manifold) |
| :--- | :--- | :--- | :--- |
| BPTT | Optimal (cyan) + Non-optimal (red, saddle) | Optimal stable, Non-optimal saddle | Fastest; slowest at $s=\pm\sqrt{|a_\star b_\star|}$ |
| tBPTT ($\tau=1$) | Same as BPTT | Same as BPTT | Close to BPTT when $\|w_\star\|$ is small |
| RFLO | Optimal only | Sign-dependent: Stable if $\hat a s>0$ / Unstable if $\hat a s<0$ | Slower at most $s$; faster only as $s\to 0$ |

### Ablation Study

| Algorithm | $W_K-W_0$ Spectral Shape | Explanation |
| :--- | :--- | :--- |
| BPTT | High rank (multiple significant singular values) | No locality constraints |
| e-prop | Medium rank | Backpropagation via diagonal $W$ |
| tBPTT ($\tau=1$) | Low rank, close to RFLO | Uses only the final step error |
| RFLO | Strictly rank $\le o$ (here $=1$) | Proposition 3.1: $W_K=W_0+\sum_{i=1}^o r_iq_i^\top$ |

### Key Findings
- **Theory Extrapolates to Non-aligned Settings**: After a short transient phase, the alignment (recurrent, input/output, random feedback) increases continuously. ODE predictions match numerical experiments well, suggesting data alignment is an "emergent property" rather than just a strict prerequisite.
- **RFLO Stability-Speed Trade-off**: Having fewer fixed points appears beneficial, but stability becomes sensitive to $\mathrm{sgn}(\hat a s)$. In Fig 4, RFLO fails to converge to the nearest optimal branch, instead taking a long detour, which increases wall-clock training time.
- **Local Rules Implicate Low-Rank Solutions**: All local algorithms (RFLO, tBPTT, e-prop) learn $W_K-W_0$ with lower rank than BPTT. In simulations, RFLO and e-prop converged in only ~60-65% of trials, whereas BPTT/tBPTT always converged.
- **Transferability to Non-linear/SSMs**: The authors note that linear RNNs share expressivity traits with modern State Space Models (SSMs like Mamba). The low-rank constraints of RFLO might suggest limitations for training linear SSMs on neuromorphic hardware.

## Highlights & Insights
- **Unified Mathematical Foundation**: Integrating fixed-point geometry, Jacobian stability, and solution rank into a single framework for three distinct algorithms provides a much deeper understanding than empirical comparisons alone.
- **Mechanistic Explanation for Poor RFLO Solutions**: The proof of Proposition 3.1 only requires $\widehat W=\hat w\mathbf{I}$, independent of data alignment. This strictly bounds the rank to the output dimension $o$, suggesting that neuromorphic systems require higher-rank random feedback to learn complex solutions.
- **Validation of the Data Alignment Hypothesis**: While often criticized as too strong, the observation that training naturally induces alignment re-legitimizes this theoretical lineage.
- **Template for Surrogate Gradients**: The methodology of mapping non-gradient rules to an ODE framework to compare them against true gradients provides a general template for studying other surrogate methods like synthetic gradients or quantized backprop.

## Limitations & Future Work
- Limited to Linear RNNs. Non-linear learning dynamics involve complexities that the current diagonalization techniques cannot yet bridge.
- The dual limits ($T\to\infty, \eta\to0$) ignore finite-batch and finite-step effects; mini-batch noise might alter RFLO stability results in practice.
- The "detour" behavior of RFLO was demonstrated on small examples; its impact on large-scale SSMs requires further validation.
- While e-prop was empirically compared, a closed-form rank bound for it was not derived, representing a natural next step.

## Related Work & Insights
- **vs Proca et al. (2025)**: Proca diagonalized BPTT; this work extends that framework to non-gradient algorithms (tBPTT and RFLO) and shifts the focus to the divergence between BPTT and its local approximations.
- **vs Saxe et al. (2014, 2019)**: Saxe studied learning dynamics in feedforward networks; this work brings that "non-linear dynamics of linear networks" philosophy to the recurrent domain, explicitly handling temporal geometric series.
- **vs Murray (2019)**: Murray provided numerical evidence for RFLO; this work provides the mechanistic explanation regarding the lack of non-optimal manifolds and the rank-limitations.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending the diagonalization framework to non-gradient rules and deriving the rank theorem are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across theoretical derivation, numerical ODE integration, and student-teacher training.
- Writing Quality: ⭐⭐⭐⭐ High mathematical density but maintains clear notation and logical flow.
- Value: ⭐⭐⭐⭐ Offers direct insights for both neuroscience (limits of biological plasticity) and neuromorphic hardware (designing feedback structures).

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
