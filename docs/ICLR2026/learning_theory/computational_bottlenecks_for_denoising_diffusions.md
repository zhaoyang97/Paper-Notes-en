---
title: >-
  [Paper Note] Computational Bottlenecks for Denoising Diffusions
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper proves that if the denoising problem corresponding to a distribution $\mu$ exhibits an "information-computation gap," denoising diffusion sampling will inevitably fail even if direct sampling from $\mu$ is easy. Specifically, there exist drifts that are nearly optimal for score matching yet lead to completel
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 63b842d0d39a8921
---
# Computational Bottlenecks for Denoising Diffusions

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=rAjHUNXybH](https://openreview.net/forum?id=rAjHUNXybH)  
**Code**: None  
**Area**: Learning Theory / Diffusion Models  
**Keywords**: Diffusion sampling, information-computation gap, score matching, computational lower bounds, sparse low-rank matrices

## TL;DR
This paper proves that if the denoising problem corresponding to a distribution $\mu$ exhibits an "information-computation gap," denoising diffusion sampling will inevitably fail even if direct sampling from $\mu$ is easy. Specifically, there exist drifts that are nearly optimal for score matching yet lead to completely erroneous sampling trajectories, and all Lipschitz polynomial-time drifts with near-optimal score matching also fail. Theoretical and numerical evidence is provided using the toy example of sparse low-rank matrices.

## Background & Motivation
**Background**: Denoising diffusion (diffusion sampling, DS) has become a core paradigm in generative AI. To sample from a target distribution $\mu$, DS constructs a stochastic process $(\hat x_t)$ such that $\hat x_T$ approximately follows $\mu$ at a large time $T$. It reduces the "sampling" problem to "approximating a drift function $m(y,t)$," where this drift is exactly the Bayes-optimal denoiser under Gaussian noise $m(y,t)=\mathbb{E}\{x\mid tx+\sqrt{t}\,g=y\}$, learned by minimizing the score-matching objective. Here, time $t$ can be interpreted as the signal-to-noise ratio (SNR) of the denoising problem.

**Limitations of Prior Work**: Most theoretical literature on DS focuses on sampling quality degradation caused by "finite samples $N$" or "non-zero step size $\Delta$." However, a more fundamental but often avoided limitation exists: $\hat m$ must be **polynomial-time computable**. This constraint exists even with full information about $\mu$ ($N=\infty$)—it is purely computational.

**Key Challenge**: The field of statistical estimation has long identified "information-computation gaps": in certain problems, the optimal estimator achieves small error, but all polynomial-time algorithms are blocked by a "wall" and cannot reach that error (valid under conjectures like P$\neq$NP). Since the DS drift is a denoiser, a natural conjecture is: a denoising gap $\Rightarrow$ DS failure. Koehler & Vuong (2024) mentioned this informally, but **strictly speaking, this statement is incorrect**: if sampling from $\mu$ is easy, one could construct a drift that directly outputs a fixed sample $x\sim\mu$ for $t\ge t_0$, making diffusion sampling correct—though this drift would be far from the optimal denoiser (this paper provides a formal counterexample in Proposition 2.1). Thus, the link "gap $\Rightarrow$ DS failure" is less obvious than it seems.

**Goal**: To complete the theoretical rigorousness of this phenomenon observed in multiple physics/Bayesian statistics papers: in what sense does a "denoising gap" truly cause "DS failure"?

**Key Insight**: Instead of focusing on a specific drift, characterize the blind spots of the "near-optimal score matching" learning objective itself: do drifts exist that are nearly optimal for score matching but catastrophic for sampling? Would the optimal drift necessarily succeed in sampling?

**Core Idea**: Strictly attribute DS failure to the computational requirement of approximating the Bayes-optimal denoiser with polynomial-time functions. As long as the denoising problem has an information-computation gap, the score-matching path cannot find a drift that enables correct sampling.

## Method

### Overall Architecture
This is a theoretical paper centered on a set of impossibility theorems regarding when diffusion sampling must fail. The logic chain is as follows: first, rewrite DS as a denoising problem—drift $m(y,t)$ is the Bayes-optimal estimator of $x$ from $y_t=tx+W_t$, with $t$ as SNR. Then, introduce an algorithmic threshold $t_{\mathrm{alg}}$, assuming that below it, all polynomial-time denoisers perform no better than constant estimators, and above it, reliable polynomial-time detection exists. Under these assumptions, two classes of failure conclusions are derived, followed by a reverse reduction to close the "estimation $\leftrightarrow$ sampling" equivalence loop.

Specifically, the conclusions consist of three parts: (1) **Theorem 1 / Corollary 3.2**—there exist drifts that are near-optimal for score matching (near-optimal among polynomial-time algorithms with super-polynomially small gaps) yet yield completely incorrect sampling; (2) **Theorem 3 / Corollary 5.1**—with an added Lipschitz condition, all polynomial-time drifts that are near-optimal for score matching yield incorrect sampling, closing the loophole that "the optimal drift might sample well"; (3) **Theorem 2 / Theorem 5**—reverse reduction: if precise DS is possible in polynomial time, then $x$ can also be estimated near the Bayes-optimality in polynomial time. The contrapositive is exactly "gap $\Rightarrow$ inability to perform correct DS in polynomial time." The entire paper uses sparse low-rank matrices $\mu_{n,k}$ as a recurring instantiation example, supported by numerical experiments using GNN denoisers.

> These conclusions do not depend on specific network architectures or training algorithms but on the inherent defects of the "score-matching objective + polynomial-time constraint"—essentially a computational complexity conclusion rather than one of optimization or generalization.

### Key Designs

**1. Mechanism: Translating "Diffusion Sampling Success" to "Denoising Computational Gap"**

The diffusion process satisfies $y_t = tx + W_t$ (where $W_t$ is Brownian motion). The ideal drift $m(y,t)=\mathbb{E}\{x\mid y_t=y\}$ is the MSE-optimal solution for estimating $x$ from Gaussian observations $y_t\sim\mathcal{N}(tx, tI_d)$, so learning the drift is equivalent to denoising, with $t$ as SNR. Building on this, the paper adopts the definition of information-computation gaps from statistical estimation: there exists a constant $\mathrm{gap}(t)>0$ such that for all polynomial-time algorithms $\hat m$, when $d$ is sufficiently large:
$$\mathbb{E}\big[\|\hat m(y_t)-x\|^2\big]\ \ge\ \inf_\phi \mathbb{E}\big[\|\phi(y_t)-x\|^2\big]+\mathrm{gap}(t).$$
The value of this perspective is that it anchors the seemingly dynamical problem of "diffusion sampling failure" to a pure computational complexity fact. However, making "gap $\Rightarrow$ failure" hold is non-trivial—Proposition 2.1 constructs a counterexample: without near-optimality in score matching (M2), a polynomial-time drift can have $\mathbb{E}[\|\hat m(y_t,t)-x\|^2]=2(1-o(1))$ (poor score matching) yet $W_1(\hat m(\hat y_{\ell\Delta},\ell\Delta), x)=0$ (perfect sampling). This shows: **failure is not that sampling itself is impossible, but that the path of "searching for drifts via score matching" is blocked by the gap**. The "near-optimal score matching" condition must be strictly tied to the conclusion.

**2. Theorem 1: Constructing Drifts with Near-Optimal Score Matching but Catastrophic Sampling**

This is the first type of impossibility. Given any polynomial-time drift $\hat m_0$, the paper explicitly constructs another polynomial-time drift $\hat m$ such that its score-matching value is nearly identical to $\hat m_0$, but its sampling is as incorrect as possible. The construction relies on two assumptions: **Assumption 1 (Small norm below threshold)**—for $t<t_{\mathrm{alg}}$, any "good" polynomial-time drift has a very small norm (because in the gap region, it cannot outperform a constant estimator $\mathbb{E}[x]\approx 0$, and the optimal strategy is to shrink the output close to zero); **Assumption 2 (Reliable detection above threshold)**—there exists a polynomial-time binary test $\phi$ that can reliably distinguish $y_t=tx+W_t$ containing a signal from pure noise $\mathcal{N}(a,tI_d)$. The construction applies a gating mechanism: $\hat m(y,t)\mathbf{1}_{\|\hat m_0(y,t)\|\le\varepsilon}$ when $t\le(1-\gamma)t_{\mathrm{alg}}$, and $\hat m_0(y,t)\phi(y,t)$ when $t\ge(1+\delta)t_{\mathrm{alg}}$.

Why can it satisfy both conditions? On the **true coupling** $(x,y_t)$, the gating barely changes the value of $\hat m_0$, so the score-matching gap is suppressed to super-polynomially small: $\int_0^\infty \mathbb{E}[\|\hat m(y_t,t)-\hat m_0(y_t,t)\|^2]\,dt = O(\eta_1\vee\eta_2)$ (in Corollary 3.2, this is $O(n^{-D})$ for any $D$, i.e., M2). However, DS actually runs a **self-generated process** $\hat y_t$, starting from noise and relying on the drift to climb. The small-norm truncation below the threshold prevents the signal from ever emerging; by the time $t$ reaches $t_{\mathrm{alg}}$, there is no signal in the trajectory to amplify, and the test $\phi$ consistently classifies it as noise and gates it out. Consequently, generated samples are stuck near $0/\mathbb{E}[x]$, leading to $W_1(\hat m(\hat y_t,t),x)\ge\alpha-o_d(1)$ (i.e., maximized $W_1$, M3).

**3. Theorem 3: Closing the Lipschitz Loophole and Mixture Distribution Trick**

Theorem 1 only states that "there exists" a bad near-optimal drift, without ruling out that the "optimal drift" might sample well. Theorem 3 closes this path: as long as the drift $\hat m(\cdot,t)$ is $C/t$-Lipschitz on $t\ge(1+\delta)t_{\mathrm{alg}}$ (where $C/t$ is chosen because the two $t$ factors in $y_t=tx+W_t$ cancel out), any polynomial-time drift near-optimal for score matching results in incorrect sampling, with the lower bound $W_1\ge\alpha-o(1)$. This condition is natural—neural networks with bounded layers and controlled weight operator norms fall into this category.

To satisfy the technical condition "small on pure noise" (Condition 2 of Theorem 3: $\Delta\sum_t\mathbb{E}[\|\hat m(W_t,t)\|^2]=o(1)$), a mixture distribution trick is introduced: $\bar\mu_{n,k}=\tfrac12\delta_0+\tfrac12\mu^0_{n,k}$, where with probability 1/2, $x=0$, and with 1/2, $x$ is a centered sparse rank-one matrix. The MSE then decomposes as:
$$\mathbb{E}_{x\sim\bar\mu_{n,k}}[\|\hat m(y_t,t)-x\|^2]=\tfrac12\mathbb{E}_{x\sim\mu^0_{n,k}}[\|\hat m(y_t,t)-x\|^2]+\tfrac12\mathbb{E}[\|\hat m(W_t,t)\|^2].$$.
To minimize the first term (fitting the signal), the second term must also be minimized (tending to zero on pure noise), thereby forcing the "tend to zero on pure noise" property. The cost is that the error baseline changes from 1 to $1/2$ (the error of the optimal constant denoiser on the mixture), thus the lower bound in Corollary 5.1 is $W_1\ge\tfrac12-o_n(1)$.

**4. Theorem 2: Reverse Reduction from Diffusion Sampling to Denoising**

Complementary to "denoising difficulty $\Rightarrow$ sampling difficulty," Theorem 2 provides the reverse direction: if one can perform DS in polynomial time with sufficient precision (i.e., $D_{\mathrm{KL}}(\bar P^{T,\Delta}_{\hat y}\|P^T_y)\le\varepsilon$), then a randomized algorithm $\hat m_+$ can be constructed to estimate the posterior expectation: $\mathbb{E}[\|\hat m_+(y)-m(y)\|^2]\le 2\bar\varepsilon+2N^{-1}$. Its contrapositive is exactly "existence of info-computation gap $\Rightarrow$ inability to perform DS correctly in polynomial time," which aligns with the impossibility conclusions. This implies that for any problem with a known denoising gap, diffusion is inherently a dead end.

### A Concrete Example
The abstract thresholds are instantiated using sparse low-rank matrices $\mu_{n,k}$. Let $u\sim\mathrm{Unif}(B_{n,k})$ be a unit vector with $k$ non-zero entries, each $\pm 1/\sqrt{k}$, and the target distribution is $x=uu^T$ ($d=n^2$). **Direct sampling is trivial**: simply draw a $u$ satisfying the sparsity constraint and set $x=uu^T$. However, the corresponding denoising problem (recovering $uu^T$ from $y_t=tx+W_t$, essentially submatrix localization / sparse PCA) has a wall: the Bayesian threshold is $t_{\mathrm{Bayes}}=2k\log(n/k)$, while the algorithmic threshold is $t_{\mathrm{alg}}=n/2$ in the moderately sparse region $\sqrt n\ll k\ll n$, and $t_{\mathrm{alg}}=k^2\log(n/k^2)$ in the extremely sparse region $k\ll\sqrt n$. In the interval $t_{\mathrm{Bayes}}\ll t\ll n$, the optimal estimator can accurately recover $x$, but (under the widely accepted Conjecture 3.1) no polynomial-time algorithm can. Feeding this gap into Theorem 1 / Theorem 3 yields the conclusion: even if sampling itself takes a second, polynomial-time diffusion trained via score matching will definitely fail—it will push all samples toward zero. The same theorems apply to tensor PCA, decoding random linear codes, low-rank + noise posteriors, and random constraint satisfaction problems.

## Key Experimental Results

### Main Results
Numerical experiments corroborate the theoretical predictions of failure modes, with $n=350$ and $k=20$ (falling in the moderately sparse region, $t_{\mathrm{alg}}=n/2$). Three polynomial-time denoisers are compared: spectral + projection (Algorithm 2), 25-step power iteration for eigenvector approximation followed by projection, and a learned Graph Neural Network (GNN, since power iteration can be approximated by a 25-layer GNN, making (b) a baseline for GNN).

| Denoiser | Behavior for $t<t_{\mathrm{alg}}$ | Behavior for $t>t_{\mathrm{alg}}$ | Crosses the Threshold Wall? |
|--------|--------|--------|--------|
| Spectral + Projection (Alg. 2) | High MSE | Fair performance | No |
| 25-step Power Iteration + Projection | High MSE | Fair performance | No |
| Learned GNN Denoiser | High MSE (lowest but still stuck) | Best performance | No |

All three hit the wall at $t_{\mathrm{alg}}=n/2$: while performance is decent above the threshold, none can cross the $t_{\mathrm{alg}}$ barrier—verifying Theoretical Prediction 1 (the learned denoiser has MSE near $1/2$ for $t<(1-\delta)t_{\mathrm{alg}}$ and near $0$ for $t>(1+\delta)t_{\mathrm{alg}}$). Notably, the GNN **outperforms** the spectral algorithm and its power iteration approximation, but no amount of architectural advantage can overcome the computational wall.

### Key Findings
- The root cause of failure is not insufficient model capacity or training—the GNN learns well above the threshold and is the best among the three below it, yet all are blocked by the same $t_{\mathrm{alg}}$ wall, which is the fingerprint of a computational lower bound.
- The failure mode is "collapse to the origin": the Frobenius norm of generated samples vanishes, meaning diffusion fails to establish any signal structure, consistent with the mechanism in Theorem 1/3 where the drift is suppressed to small norms in the low SNR regime.

## Highlights & Insights
- Strictly anchoring "diffusion sampling failure" to "information-computation gaps" creates a bridge between generative feasibility and statistical computational lower bounds—turning "folk wisdom" into a formal framework.
- The contrast between "near-optimal score matching with wrong sampling" and the "counterexample of poor score matching with correct sampling" (Proposition 2.1) is insightful: it shows the issue isn't sampling capability but the use of "score matching as a compass"—when the difference between a good and bad drift in score matching is super-polynomially small, the objective cannot distinguish them. This suggests that for distributions with gaps, minimizing score matching might not be the correct path to find a good drift.
- The Lipschitz condition is realistic: it isn't just a mathematical convenience but covers real neural networks with bounded layers and controlled weight norms, ensuring that the "all Lipschitz optimizers fail" conclusion hits practice.
- The mixture distribution $\tfrac12\delta_0+\tfrac12\mu^0$ is a clever trick: via MSE decomposability, it transforms "tending to zero on pure noise" from a hard-to-verify technical condition into a natural byproduct of the training objective.
- A transferable criterion: before using diffusion for a new distribution, ask "does its denoising problem have a computational gap?" If it does, one should expect diffusion to collapse and consider alternative sampling paths.

## Limitations & Future Work
- The conclusions are **conditional**: they rely on conjectures about information-computation gaps (e.g., Conjecture 3.1), which are P$\neq$NP-style and currently unprovable unconditionally—an inherent limitation of the field acknowledged by the authors.
- Theorem 3 only excludes **Lipschitz** polynomial-time near-optimal drifts; in principle, a **non-Lipschitz** near-optimal drift could still sample well. However, the authors argue that since it would only differ from a bad drift by a super-polynomially small score-matching value, score matching would fail to locate it anyway.
- The forms of Assumption 1 and 2 do not directly match the gap definition; the authors argue they hold for a wide class of problems unless the gap vanishes, but the strict equivalence "Assumption $\Leftrightarrow$ gap" is left for future work.
- Instantiation is currently concentrated on sparse low-rank matrices and related PCA/coding examples; Bayesian posteriors and random CSPs require additional technical handling.

## Related Work & Insights
- **vs. Positive Guarantees for DS (Chen et al. 2023, Montanari & Wu 2023, etc.)**: These works provide forward bounds reducing diffusion sampling to score estimation for certain distribution classes; this paper provides the necessary converse—these guarantees cannot hold when a denoising gap exists.
- **vs. Koehler & Vuong (2024)**: While they informally suggested "gaps lead to DS failure" using spiked Wigner models, this paper identifies the rigor gap (Proposition 2.1) and provides the necessary "near-optimal score matching" constraint.
- **vs. Empirical Observations in Physics/Bayesian Stats**: While earlier works noted that Gibbs sampling works while diffusion fails on certain measures, they lacked formal assertions; this paper provides the methodology to "rigorize" those lines of thought.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally bridges DS failure to info-computation gaps and corrects previous informal assertions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Numerical experiments clearly support the theoretical predictions, though the scale is small and serves an illustrative rather than systemic evaluation role.
- Writing Quality: ⭐⭐⭐⭐ Logical flow is clear with honest disclosure of counterexamples, though the density of theorems and assumptions is high for non-specialists.
- Value: ⭐⭐⭐⭐⭐ Provides a principled criterion for when not to use diffusion sampling, offering significant guidance for generative model theory and practice.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Language Identification in the Limit with Computational Trace](language_identification_in_the_limit_with_computational_trace.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] Gradient Descent Dynamics of Rank-One Matrix Denoising](gradient_descent_dynamics_of_rank-one_matrix_denoising.md)
- [\[ICLR 2026\] Score-Based Density Estimation from Pairwise Comparisons](score-based_density_estimation_from_pairwise_comparisons.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
