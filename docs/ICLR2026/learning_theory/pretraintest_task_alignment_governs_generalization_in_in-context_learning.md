---
title: >-
  [Paper Note] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning
description: >-
  [ICLR 2026][Learning Theory][Generalization Error] This paper uses a solvable linear attention model for in-context linear regression to derive an exact high-dimensional formula for ICL generalization error under arbitrary mismatch between the pretraining task covariance $C_{\text{train}}$ and the test task covariance $C_{\text{test}}$. From this, a "task alignment metric" is extracted, which accurately predicts ICL performance not only in solvable models but also in nonlinea…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "In-Context Learning"
  - "Generalization Error"
  - "Task Alignment"
  - "Linear Attention"
  - "High-Dimensional Theory"
date: 2026-05-08
content_hash: 407eef93bb0d6209
---

# Pretrain–Test Task Alignment Governs Generalization in In-Context Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KZLeg0MQ2r](https://openreview.net/forum?id=KZLeg0MQ2r)  
**Code**: TBD  
**Area**: Learning Theory / In-Context Learning  
**Keywords**: In-Context Learning, Generalization Error, Task Alignment, Linear Attention, High-Dimensional Theory

## TL;DR
This paper uses a solvable linear attention model for in-context linear regression to derive an exact high-dimensional formula for ICL generalization error under arbitrary mismatch between the pretraining task covariance $C_{\text{train}}$ and the test task covariance $C_{\text{test}}$. From this, a "task alignment metric" is extracted, which accurately predicts ICL performance not only in solvable models but also in nonlinear Transformers, revealing a specialization-generalization tradeoff where "increased pretraining task diversity is not always better."

## Background & Motivation
**Background**: In-context learning (ICL) is a core capability of Transformers—the model "meta-learns" a learning algorithm during pretraining and performs new tasks at test time using only a few examples in the prompt, without further training. Extensive theoretical work exists around ICL, with the mainstream approach using linear/kernelized attention for in-context linear regression to prove these structures implicitly implement ridge regression, gradient descent, or Bayesian inference.

**Limitations of Prior Work**: Almost all these theoretical analyses make strong simplifying assumptions—data coming from isotropic Gaussians, **identical task distributions for pretraining and testing**, and generalization studied only in infinite sample or population limits. In reality, the essence of ICL is that pretraining tasks and test tasks **will not be identical** (there is no free lunch). The core question of "how to choose the pretraining task distribution to ensure ICL generalizes in the real test world" remains theoretically unanswered.

**Key Challenge**: A mismatch exists between the pretraining and test task structures. The impact of this mismatch on generalization is non-trivial—it is influenced by both the spectral structure of task covariances and "finite-size effects" such as finite context length, finite task diversity, and label noise. These factors are entangled, and no existing metric can characterize them.

**Goal**: Within the simplest yet solvable setting of "linear attention for in-context linear regression," allow **arbitrary covariance structures** for both pretraining and test task distributions. The goal is to precisely characterize how task mismatch determines ICL generalization error and identify the alignment metric that truly drives this error.

**Key Insight**: The authors draw on established results for ridge regression under covariate shift, where "train-test feature covariance alignment" determines out-of-distribution generalization. They systematically bring this "spectral alignment + finite sample resolution" perspective into the ICL setting for the first time.

**Core Idea**: ICL generalization error can be decomposed into a scalar term independent of task structure + a mismatch term $e_{\text{misalign}} = \langle C_{\text{test}}, K\rangle$, where $K$ is a matrix "filtered" from $C_{\text{train}}$ by finite samples/noise. This alignment metric is a precise characterization of pretrain-test task alignment and can predict performance across architectures.

## Method

### Overall Architecture
This paper does not propose a new model but performs a thorough high-dimensional asymptotic analysis of an existing "solvable ICL model," distilling the resulting analytical formulas into an interpretable, transferable alignment metric. The logical chain is: **Build solvable model → Find optimal predictor → Derive high-dimensional error formula → Extract alignment metric → Validate across architectures + Deduce tradeoff conclusions**.

The model setup is as follows: The context is a sequence $\{x_1, y_1, \dots, x_\ell, y_\ell, x_{\ell+1}\}$, where $y_i = \langle x_i, w\rangle + \epsilon_i$ is an approximate linear mapping with noise $\epsilon_i\sim N(0,\rho)$. The model estimates the task vector $w$ from the first $\ell$ examples to predict $y_{\ell+1}$. During pretraining, the task vector $w^\mu$ for each context is sampled uniformly from a **finite set of $k$ tasks** $\{t_1,\dots,t_k\}$, where $t_j\sim N(0, C_{\text{train}})$; here $k$ is called **task diversity**, and $C_{\text{train}}$ controls the structure. At test time, $w^{\text{test}}\sim N(0, C_{\text{test}})$, where $C_{\text{test}}$ can be entirely different from $C_{\text{train}}$—the source of "arbitrary mismatch."

The model uses single-layer linear self-attention $A = Z + VZ(KZ)^\top(QZ)/\ell$, and the prediction is the bottom-right element $\hat y = A_{d+1,\ell+1}$. Following prior work, terms with negligible contribution to estimating $w$ are dropped ($v_{21}=0$), simplifying the output to $\hat y_{\ell+1} = \mathrm{tr}(\Gamma H_Z^\top)$, where $\Gamma$ is the parameter matrix and $H_Z$ is the data matrix. This reduces the model to a **ridge regression problem over parameter matrix $\Gamma$**, solvable analytically in the minimum-norm limit $\lambda\to 0$.

Analysis is performed in the high-dimensional scaling limit: token dimension $d$, context length $\ell$, batch size $n$, and task diversity $k$ all tend to infinity, maintaining three constant ratios: $\alpha = \ell/d$ (context length parameter), $\tau = n/d^2$ (batch size parameter), and $\kappa = k/d$ (task diversity parameter). This limit makes the model solvable while preserving interesting finite-size phenomena.

### Key Designs

**1. Solvable ICL Model with Arbitrary Mismatch: Incorporating "Train ≠ Test Task Distribution"**

Previous solvable ICL models almost always assumed $C_{\text{train}} = C_{\text{test}}$ and isotropy, which erases the "task mismatch" the authors aim to study. The key innovation here is introducing two **independent arbitrary covariances** $C_{\text{train}}$ and $C_{\text{test}}$, and explicitly parameterizing task diversity $k$. When $k<n$, tasks repeat within a pretraining batch, decoupling "how many truly distinct tasks are seen" from "the spectral structure of those tasks." This allows for a deeper exploration of task generalization and spectral alignment.

**2. Exact High-Dimensional ICL Generalization Formula and Alignment Metric $K$**

This is the theoretical core. The authors prove that in the high-dimensional limit, the ICL test error can be written as:

$$E_{\text{ICL}}(\Gamma^*) \simeq e_{\text{scalar}}(\lambda_{\text{train}}, c_{\text{test}}) + e_{\text{misalign}}(C_{\text{train}}, C_{\text{test}}),$$

where $e_{\text{scalar}}$ depends on the task structure only through the trace $c_{\text{test}}=\mathrm{tr}[C_{\text{test}}]$ and the spectrum of $C_{\text{train}}$; the term characterizing "mismatch" is:

$$e_{\text{misalign}}(C_{\text{train}}, C_{\text{test}}) = \langle C_{\text{test}}, K\rangle, \qquad K \equiv q\,F_\kappa(\sigma) + (q\tilde\lambda - \sigma^2)\,F'_\kappa(\sigma).$$

Here, $F_\kappa(z)$ and $M_\kappa(z)$ are a pair of **resolvent quantities** defined by self-consistent implicit equations, satisfying $(R_k + zI_d)^{-1}\simeq F_\kappa(z)$, where $R_k = \frac1k\sum_j t_jt_j^\top$ is the empirical $k$-sample task covariance. Intuitively, $F_\kappa$ and $M_\kappa$ characterize how much signal from $C_{\text{train}}$ can be recovered given $k$ samples and a noise threshold $z$. The **effective noise** $\sigma = (\rho + c_{\text{train}})/\alpha + \tilde\lambda$ aggregates label noise $\rho$, context length $\alpha$, and the effective ridge $\tilde\lambda$ (determined by $\tilde\lambda M_\kappa(\sigma)=1-\tau$), representing the cost the ICL model pays to decouple token statistics from task information.

Why does $\langle C_{\text{test}}, K\rangle$ work as an alignment metric? In a simplified analogy like $\langle C_{\text{test}} C_{\text{train}}^{-1}\rangle$, alignment is greatest when the strong directions of the test and train tasks align. $K$ shares eigenvectors with $C_{\text{train}}$ and its eigenvalues are inversely ordered relative to $C_{\text{train}}$, inheriting this "relative strength" property. Crucially, $K$ accounts for the fact that **finite samples can only partially resolve $C_{\text{train}}$**, something simple population metrics cannot do.

**3. Cross-Architecture Metric Transfer: Predicting Nonlinear Transformer Performance**

While derived for linear attention, the authors use a two-layer nonlinear Transformer (with softmax attention and MLP) to perform ICL. They use $e_{\text{misalign}}$ and competitor metrics (e.g., population $\langle C_{\text{test}} C_{\text{train}}^{-1}\rangle$, CKA) to predict actual ICL error. The results show the proposed metric achieves a Spearman correlation of **0.99**, significantly outperforming others. This suggests the metric captures an **architecture-independent essence** of ICL generalization.

**4. Specialization-Generalization Tradeoff: Mismatched Distributions are Often Optimal**

The authors ask if "pretraining on the test distribution" ($C_{\text{train}}=C_{\text{test}}$) is optimal. The answer is **no**. Corollary 4.1 shows that when $C_{\text{train}}$ and $C_{\text{test}}$ are simultaneously diagonalizable, mismatch error is extremized. Corollary 4.2 shows that for a fixed $C_{\text{train}}$ and constrained $c_{\text{test}}$, the optimal test covariance is a rank-one spike aligned with the **maximum eigenvalue direction** of $C_{\text{train}}$. Furthermore, for power-law spectra, Figure 4 shows that when task diversity $\kappa$ is small (data scarcity), a pretraining spectrum **steeper** than the test spectrum ($p_{\text{train}}>p_{\text{test}}$) significantly reduces error. Focusing pretraining on a low-dimensional subspace creates a strong inductive bias; "overfitting on few directions" surpasses "weak learning on many directions" for generalization.

### Loss & Training
The optimal parameters are obtained by minimizing the ridge-regularized MSE on the next-output prediction:

$$\Gamma^* = \arg\min_\Gamma \sum_{\mu=1}^n\big(y^\mu_{\ell+1} - \mathrm{tr}(\Gamma (H^\mu)^\top)\big)^2 + \frac{n}{d}\lambda\,\mathrm{tr}(\Gamma\Gamma^\top),$$

focusing on the minimum-norm predictor ($\lambda\to 0$). Nonlinear Transformer experiments use standard two-layer softmax attention + MLP architectures trained on the same task distributions (details in Appendix H).

## Key Experimental Results

### Main Results

| Validation Content | Setting | Result |
|--------|------|------|
| Theoretical $e_{\text{ICL}}$ vs. Simulation | $d=120,\alpha=2,\tau=4,\rho=0.01$ | Theoretical curves perfectly match sampled MSE (Figure 1) |
| $e_{\text{ICL}}$ vs. Task Diversity $\kappa$ | Aligned vs. Mismatched $C_{\text{test}}$ | Monotonic decrease when aligned; non-monotonic or increasing when mismatched |
| Metric for Linear Model Error | Power-law / Low-rank $C_{\text{test}}$ | $\langle C_{\text{test}}K\rangle$ perfectly correlates with $e_{\text{ICL}}$ (Figure 2) |

### Alignment Metric Comparison

| Alignment Metric | Spearman Correlation with Nonlinear Transformer ICL Error | Description |
|------|---------|------|
| $e_{\text{misalign}}=\langle C_{\text{test}}K\rangle$ (Ours) | **0.99** | Includes finite sample resolution + effective noise (Optimal) |
| $\langle C_{\text{test}}F_\kappa(\sigma)\rangle$ | 0.98 | Based on resolvent; less precise than $K$ |
| $\langle C_{\text{test}}C_{\text{train}}^{-1}\rangle$ | 0.96 | Population quantity; lacks finite sample effects |
| $1/\mathrm{CKA}(C_{\text{train}}, C_{\text{test}})$ | 0.39 | Designed for representation similarity; fails for ICL error |

### Key Findings
- **"More task diversity is always better" is false**: Whether extra tasks help depends on pretrain-test alignment. Under mismatch, increasing $\kappa$ can harm performance.
- **Finite sample resolution is the differentiator**: The proposed metric succeeds where population metrics and CKA fail because it encodes that finite $k$ samples only partially recover $C_{\text{train}}$.
- **Strong inductive bias is better for scarce data**: At low $\kappa$, focusing pretraining on low-dimensional subspaces (steeper spectra) reduces error—"overfitting on few directions > weak learning on many."
- **Anisotropy is a prerequisite**: If $C_{\text{train}}=I_d$ (isotropic), all test covariances with fixed trace perform identically; leveraging anisotropy is key.

## Highlights & Insights
- **Systematically brings "spectral alignment + finite sample resolution" to ICL**: The effective noise $\sigma = (\rho + c_{\text{train}})/\alpha + \tilde\lambda$ mirrors the optimal regularization structure in standard ridge regression, making ICL error interpretable.
- **Architecture-agnostic metric derived from simple models**: The 0.99 correlation suggests ICL generalization is dominated by task spectral alignment, a significant scientific signal.
- **Counter-intuitive "optimal task misalignment"**: Instead of matching the test distribution, it is often better to provide a "curriculum" that compresses signals into shared directions, helping the Transformer learn a generalizable algorithm rather than memorizing the test structure.
- **Reusable Methodology**: Using the resolvent $F_\kappa(z)$ to characterize "how much covariance structure can be recovered from finite samples" can be transferred to other meta-learning/transfer learning error analyses.

## Limitations & Future Work
- **Setting restricted to linear regression + single-layer linear attention**: While the metric works for two-layer nonlinear Transformers, it is far from real-world LLMs with multi-layer structures and language tokens.
- **Task relations limited to linear, noise is Gaussian i.i.d.**: Real task structures are more complex (nonlinear, heavy-tailed, correlated noise).
- **Interaction between $e_{\text{scalar}}$ and $e_{\text{misalign}}$ not fully explored**: Deeper analysis of their interaction is needed to derive a general heuristic for "optimal pretraining distributions."
- **Dependencies on conjectures**: Some conclusions (e.g., $K$ eigenvalue ordering) are proved for $\tau>1$ but rely on numerical evidence for $\tau<1$.

## Related Work & Insights
- **vs. Lu et al. (2025) / Zhang et al. (2024a)**: They use the same solvable model but assume pretraining = test distributions. This work generalizes to **arbitrary covariance mismatch** and introduces the mismatch term $e_{\text{misalign}}$.
- **vs. Ridge Regression Covariate Shift (Atanasov 2025 / Canatar / Patil / Tripuraneni)**: These works study how feature covariance alignment determines OOD generalization; this paper moves the alignment focus from features to the **task covariance** level in ICL.
- **vs. Raventós et al. (2023)**: They empirically found a transition from memorization to generalization via task diversity; this paper provides the theoretical counterpart and conditions under which diversity is not beneficial.
- **vs. Goddard et al. (2025)**: They study generalization on spherical task manifolds; this paper uses an arbitrary covariance + finite sample framework to characterize task structure mismatch more generally.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide exact ICL generalization formulas and transferable alignment metrics under arbitrary task mismatch.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid numerical simulations and cross-architecture validation, though limited to linear regression tasks and non-language ICL.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and helpful analogies to ridge regression; high formula density may be challenging for non-theorists.
- Value: ⭐⭐⭐⭐⭐ Highlights that task alignment, not just diversity, governs ICL; insights on optimal pretraining distributions are practically valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] Neural Collapse in Multi-Task Learning](neural_collapse_in_multi-task_learning.md)

</div>

<!-- RELATED:END -->
