---
title: >-
  [Paper Note] Learning Linear State-Space Models with Sparse System Matrices
description: >-
  [ICLR 2026][Time Series][Linear State-Space Models] This paper introduces sparsity-inducing priors (Student's t-distribution) to the system matrices $A, B, C, D$ of Linear State-Space Models (LSSM). By employing EM combined with block coordinate descent for MAP estimation, the method bypasses the unidentifiability caused by "similarity transformations," **learning sparse system matrices that are both accurate and capable of preserving the true topology between variables.**
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Linear State-Space Models"
  - "Sparse System Matrices"
  - "Sparse Bayesian Learning"
  - "EM Algorithm"
  - "Similarity Transformation"
  - "System Identification"
date: 2026-05-08
content_hash: c9f338fcdd8861ad
---

# Learning Linear State-Space Models with Sparse System Matrices

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0lct7PrPgS](https://openreview.net/forum?id=0lct7PrPgS)  
**Code**: [https://github.com/ArthinYS/Learning-Sparse-LSSM](https://github.com/ArthinYS/Learning-Sparse-LSSM)  
**Area**: Time Series / Dynamical System Identification  
**Keywords**: Linear State-Space Models, Sparse System Matrices, Sparse Bayesian Learning, EM Algorithm, Similarity Transformation, System Identification  

## TL;DR
This paper introduces sparsity-inducing priors (Student's t-distribution) to the system matrices $A, B, C, D$ of Linear State-Space Models (LSSM). By employing EM combined with block coordinate descent for MAP estimation, the method bypasses the unidentifiability caused by "similarity transformations," **learning sparse system matrices that are both accurate and capable of preserving the true topology between variables.**

## Background & Motivation
**Background**: Linear State-Space Models $x_t=Ax_{t-1}+Bu_t+\varepsilon_t,\ y_t=Cx_t+Du_t+\omega_t$ are fundamental tools for modeling time series in robotics, systems biology, industrial processes, and NLP. They provide a complete theory for controllability, observability, and stability, facilitating analysis and control. Real-world system coupling is often sparse—one gene regulates only a few others, or communication systems use sparse topologies for energy efficiency—meaning the true $A, B, C, D$ are usually sparse matrices.

**Limitations of Prior Work**: Classical LSSM learning algorithms (least squares PEM, Ho-Kalman, subspace 4SID, maximum likelihood EM) **cannot impose sparsity constraints on system matrices**. The root cause is the **similarity transformation**: for any non-singular matrix $\Phi$, the system $(\Phi A\Phi^{-1}, \Phi B, C\Phi^{-1}, D)$ generates the exact same input-output data as the original. Consequently, these algorithms can only learn solutions "up to a similarity transformation." This transformation not only changes numerical values but also destroys the topological structure of the system matrices, preventing the learned matrices from reflecting the true interaction mechanisms between variables.

**Key Challenge**: Sparsity constraints seek the "simplest explanation of data" (Occam's razor), but similarity transformations fill the solution space with equivalent yet dense realizations. Simultaneously, states $x_t$ are unobservable, and observations are contaminated by both process and measurement noise, making direct sparse estimation difficult.

**Goal**: Learn sparse system matrices that **preserve the inherent topological structure**, allowing the results to be used for investigating system interaction laws while surpassing classical algorithms in prediction accuracy.

**Key Insight**: **[Sparse Priors Lock Similarity Transformations]**—Applying sparsity-inducing priors to system matrices restricts the non-singular matrices $\Phi$ that can maintain sparsity ($\ell_0$ norm invariance) to generalized permutation matrices. This compresses the ambiguity of the similarity transformation down to mere "permutation + scaling," thereby preserving the topological structure.

## Method

### Overall Architecture
The hidden states $X=\{x_t\}$ are treated as latent variables. Student's t-distribution sparse priors are applied to the system matrices, and the joint posterior is formulated according to Bayesian rules. MAP estimation is performed via EM: the E-step uses an RTS smoother for closed-form state distribution updates, while the M-step employs block coordinate descent to analytically update system matrices, noise covariances, and prior hyperparameters block-by-block. Iterations continue until convergence, with the global convergence theorem ensuring arrival at a local maximum or saddle point of the posterior.

```mermaid
flowchart LR
    A[Noisy Observations y_t, u_t] --> B[Student's t Sparse Prior<br/>p Θ]
    B --> C[Joint Posterior p Θ|Y<br/>= Marginal Likelihood × Prior]
    C --> D[EM Iteration]
    D --> E[E-step: RTS Smoother<br/>Closed-form Latent State Update]
    E --> F[M-step: Block Coordinate Descent<br/>Row-wise Analytical Update A,B,C,D]
    F --> G[Update Noise R,Q<br/>+ Hyperparameters Γ]
    G -->|Not Converged| E
    G -->|Converged| H[Sparse System Matrices<br/>Preserving Topology]
```

### Key Designs

**1. Student's t Hierarchical Sparse Prior: Encoding Sparsity into the Probabilistic Model**  
The authors apply a Student's t-distribution prior to every element of $A, B, C, D$ to promote sparsity, as it is sharper at zero compared to other priors. Implementation follows a hierarchical approach: a zero-mean Gaussian prior $p(A_{ij}\mid\Gamma_{a,ij})=\frac{1}{\sqrt{2\pi\Gamma_{a,ij}}}\exp(-A_{ij}^2/2\Gamma_{a,ij})$ is assigned to each element, followed by an Inverse-Gamma hyper-prior on the unknown variance $\Gamma_{a,ij}$ (with hyperparameters $a_0, b_0$ set to minimal values like $10^{-6}$ for an uninformative prior). Marginalizing out $\Gamma$ yields the Student's t-distribution. This is a standard Sparse Bayesian Learning (SBL) construction—allowing the variance of irrelevant elements to shrink to zero, effectively forcing the corresponding parameters to zero. Noise $R, Q$ use uniform priors to obtain flat priors.

**2. EM + RTS Smoother: Closed-form Completion of Unobservable States**  
Since states $x_t$ are unobserved, directly maximizing the posterior $p(\Theta\mid Y)\propto p(Y\mid\Theta)p(\Theta)$ is infeasible. The authors introduce $x_t$ as latent variables in an EM framework, iteratively maximizing the expected log-posterior $H(\Theta\mid\Theta^k)=\mathbb{E}_{X^k}[\log(p(Y,X\mid\Theta)p(\Theta))]$. The E-step uses the **Rauch–Tung–Striebel (RTS) smoother** for a closed-form solution of $p(x_t\mid Y,\Theta^k)=\mathcal{N}(m_t^k,P_t^k)$: first a forward Kalman filter for $\mu_t,\Sigma_t$, then a backward recursion for $m_t^k=\mu_t^k+G_t^k(m_{t+1}^k-\mu_{t+1}^k)$. Additionally, a lag-one covariance smoother $P_{t,t-1}^k$ is required to assemble the loss function $H=H_1(A,B,R)+H_2(C,D,Q)+H_3(\cdot,\Gamma)$.

**3. Row-wise Analytical Update via Block Coordinate Descent: Handling High-Coupling Non-convex Objectives**  
$H(\Theta\mid\Theta^k)$ is non-convex and highly coupled, preventing a simultaneous closed-form solution. The authors use block coordinate descent to update system matrices **row by row**. Taking the $r$-th row of $A$ as an example, setting the derivative to zero yields $A_r^{k+1}=\big(\sum_t (m_{t,r}^k-B_r^k u_t)(m_{t-1}^k)'+P_{t,t-1,r}^k\big)\big(\sum_t(P_{t-1}^k+m_{t-1}^k(m_{t-1}^k)')+R_{rr}^k\Gamma_{a,r}^{kd}\big)^{-1}$. The term $\Gamma_{a,r}^{kd}$ is the diagonal regularization injected by the sparse prior, facilitating automatic shrinkage. $B, C, D$ are updated similarly. Noise covariances $R_{rr}, Q_{rr}$ and hyperparameters $\Gamma$ also have analytical update formulas, such as $\Gamma_{a,ij}^{k+1}=\frac{(A_{ij}^{k+1})^2+2b_0}{2a_0+3}$—smaller parameters lead to smaller variance updates, further pushing them toward zero in a positive feedback loop of sparsity.

**4. Locking Similarity Transformations to Generalized Permutation Matrices: Theoretical Explanation of Identifiability**  
This is the theoretical pillar of the paper. The authors define "essential identifiability": if any non-singular $\Phi$ satisfying sparsity invariance (e.g., $\|\Phi A\Phi^{-1}\|_0=\|A\|_0$) must be a generalized permutation matrix, the system is identifiable. The intuition: classical solutions differ by an arbitrary $\Phi$, which disrupts topology; however, requiring sparsity preservation severely restricts the degrees of freedom of $\Phi$—to only "scaling + permutation" for identifiable systems. A 3D example explicitly verifies that for a given sparse $A, B, C, D$, valid $\Phi$ matrices can only be one of six generalized permutation matrices. Thus, the difference between the prior-learned matrices and ground truth **stems only from the scaling definition of state variables**, leaving the topology intact. Based on the global convergence theorem (Luenberger 1984), the authors prove that the sequence $\{\Theta^k\}$ converges to a local maximum or saddle point of the posterior.

## Key Experimental Results
Baseline: LSM PEM, LSM HK, 4SID, MLE. Metric: Mean Relative Error MRE $=\frac{1}{T}\sum_t\frac{\|y_t-\hat y_t\|_2^2}{\|y_t\|_2^2}$ (lower is better), 2:1 train/test split.

### Main Results (MRE on Real Industrial Systems, runtime in parentheses)

| Dataset | Ours | LSM PEM | LSM HK | 4SID | MLE |
|---|---|---|---|---|---|
| Evaporation | **14.93%** (249.14 s) | 17.90% (9.35 s) | Inf (18.45 s) | 43.77% (3.92 s) | 20.14% (152.34 s) |
| Glass furnace | **23.63%** (45.62 s) | 62.62% (0.52 s) | 31.21% (6.20 s) | 24.32% (0.29 s) | 30.21% (33.36 s) |
| Steam generator | **20.70%** (441.88 s) | 22.70% (4.94 s) | 39.80% (84.52 s) | 29.26% (0.58 s) | 22.23% (299.10 s) |

Ours achieves the lowest MRE on all three real datasets, though with the longest runtime (due to matrix inversions in closed-form updates). Note that LSM HK has an MRE of Inf on the Evaporation system (learned model is unstable, prediction diverges), indicating classical algorithms can fail completely on highly sparse systems, while Ours remains robust.

### Synthetic System Results (Topology Recovery)
On a 10D synthetic system ($A$ is anti-diagonal with non-zero elements 0.8, $B=C=D=2I_{10}$, 100 parameters per matrix, extremely sparse): classical algorithms learn matrices that are completely different from ground truth in both value and topology due to similarity transformations. **Only Ours preserves the topological structure between variables.** By comparing learned $B, C$ with ground truth, one can derive $\Phi \approx 2I_{10}$, consistent with the theoretical analysis in Section 4.1, while achieving the lowest MRE among all algorithms.

### Ablation Study
Supplementary validations confirm the results:
- **Robustness to Initializations (Appendix H.1)**: Under different initial $A, B, C, D$, the algorithm consistently converges to sparse solutions that preserve topology.
- **20 Independent Trials (Appendix H.2)**: Reports success rates for topology recovery and average MRE. Ours consistently leads in both.
- **Complex Structures (Appendix I)**: Remains effective for non-diagonal and non-invertible system matrices.
- **Threshold Truncation (Appendix G)**: Numerical optimization rarely yields exact zeros; a preset threshold truncates small parameters to accelerate convergence, applied to all baselines for fairness.

### Key Findings
- **Topology Fidelity is the Unique Selling Point**: Classical algorithms learn dense equivalent solutions. Ours learns sparse matrices where the difference from ground truth is only a generalized permutation matrix (scaling + permutation).
- **Win-Win for Accuracy and Interpretability, at the Cost of Computation**: Accuracy leads while runtime is significantly higher than classical methods.
- Robust to initializations and effective for complex cases like non-diagonal/non-invertible matrices (Appendix I).

## Highlights & Insights
- **Clear Explanation of Sparsity vs. Identifiability**: The ambiguity of similarity transformations stems from the excessive degrees of freedom in $\Phi$. Sparse constraints lock $\Phi$ into a generalized permutation matrix—the most elegant part of the paper.
- **Fully Closed-form + Convergence Guarantees**: RTS smoothing in the E-step and block coordinate descent in the M-step provide analytical updates, coupled with global convergence theorems for high theoretical integrity and reproducibility.
- **Migrating SBL to LSSM with Latent States and Dual Noise**: While SBL often assumes observable states, this work completes sparse estimation under unobservable states and double noise contamination (process and measurement), filling a practical gap.
- **Occam's Razor in Practice**: Instead of saying "fewer parameters are better," the paper establishes a causal chain: "minimum parameters $\Leftrightarrow$ topological identifiability $\Leftrightarrow$ investigation of interaction laws," elevating sparsity from an engineering trick to a source of interpretability.

## Limitations & Future Work
- **High Computational Overhead**: Closed-form updates involve numerous matrix inversions, resulting in the longest runtime, making it difficult to scale to massive problems. Stochastic EM (mini-batch) is suggested for future cost reduction.
- **Numerical Shrinkage Risk**: Similarity transformations might push many parameters to extremely small values, leading to numerical errors.
- **Scaling Ambiguity Remains**: Currently guarantees topological consistency, but numerical values still differ by scaling; future work aims for additional constraints to achieve exact value matching.
- **Identifiable Systems Only**: Topology fidelity guarantees rely on the system being "essentially identifiable."
- Restricted to linear systems; only a local linear approximation for strongly nonlinear systems.

## Related Work & Insights
This work sits at the intersection of: (1) Least squares/PEM—sensitive to noise and unable to characterize sparsity; (2) Subspace identification 4SID—reliant on Hankel matrix projection, difficult to obtain exact matrices; (3) Maximum Likelihood EM (Shumway-Stoffer, Ghahramani-Hinton)—unable to handle sparsity; (4) Sparsity-inducing methods (Lasso, reweighted $\ell_1$, SBL)—previously assuming observable states. The contribution is grafting SBL's sparse priors into the EM-LSSM framework and explaining why sparsity breaks similarity transformation ambiguity. **Insight**: When an identification problem has unidentifiability due to an equivalence class, introducing a structural constraint invariant to that class (sparsity/$\ell_0$) can compress the equivalence class to trivial ambiguities—a paradigm applicable to other system identification, causal discovery, and network reconstruction problems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Highly novel insight on locking similarity transformations via sparse priors.
- **Experimental Thoroughness**: ⭐⭐⭐ — Synthetic + 3 real datasets with robustness tests, but small scale and lacking comparison with modern deep SSMs (e.g., S4/Mamba).
- **Writing Quality**: ⭐⭐⭐⭐ — Complete derivations and clear theoretical arguments.
- **Value**: ⭐⭐⭐⭐ — High utility for industrial processes and systems biology needing interpretable matrices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICLR 2026\] ConvT3: Structured State Kernels for Convolutional State Space Models](convt3_structured_state_kernels_for_convolutional_state_space_models.md)
- [\[ICLR 2026\] Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)
- [\[ICLR 2026\] Learning Mixtures of Linear Dynamical Systems via Hybrid Tensor-EM Method](learning_mixtures_of_linear_dynamical_systems_via_hybrid_tensor-em_method.md)
- [\[ICLR 2026\] Learning Koopman Representations with Controllability Guarantees](learning_koopman_representations_with_controllability_guarantees.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICLR 2026\] ConvT3: Structured State Kernels for Convolutional State Space Models](convt3_structured_state_kernels_for_convolutional_state_space_models.md)
- [\[ICLR 2026\] Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)
- [\[ICLR 2026\] Learning Mixtures of Linear Dynamical Systems via Hybrid Tensor-EM Method](learning_mixtures_of_linear_dynamical_systems_via_hybrid_tensor-em_method.md)
- [\[ICLR 2026\] Learning Koopman Representations with Controllability Guarantees](learning_koopman_representations_with_controllability_guarantees.md)

</div>

<!-- RELATED:END -->
