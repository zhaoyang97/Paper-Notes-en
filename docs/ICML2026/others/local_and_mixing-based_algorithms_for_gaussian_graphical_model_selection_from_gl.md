---
title: >-
  [Paper Note] Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics
description: >-
  [ICML 2026][Gaussian Graphical Model] This work is the first to study the problem of learning Gaussian graphical model structure from a single trajectory of Gaussian Glauber dynamics. Two complementary algorithms are pro…
tags:
  - "ICML 2026"
  - "Gaussian Graphical Model"
  - "Glauber Dynamics"
  - "Dobrushin Condition"
  - "Burn-in/Thinning"
  - "Total Variation Mixing"
date: 2026-05-08
content_hash: b88f77cd98498e8c
---

# Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics

**Conference**: ICML 2026  
**arXiv**: [2412.18594](https://arxiv.org/abs/2412.18594)  
**Code**: Not yet released  
**Area**: Probabilistic Graphical Models / Structure Learning / Glauber Dynamics  
**Keywords**: Gaussian Graphical Model, Glauber Dynamics, Dobrushin Condition, Burn-in/Thinning, Total Variation Mixing

## TL;DR
This work is the first to study the problem of learning Gaussian graphical model structure from a single trajectory of Gaussian Glauber dynamics. Two complementary algorithms are proposed: LET-GL (local edge testing based on i,i,j,i windows, perfectly parallelizable) and BTR-GL (under the Dobrushin condition, uses burn-in/thinning to "decorrelate" the trajectory into approximately i.i.d. samples, which are then fed to existing i.i.d. learners). The paper provides finite-sample recovery guarantees, information-theoretic lower bounds, and an independently valuable total variation mixing bound for the random-scan Gaussian Gibbs sampler.

## Background & Motivation

**Background**: Structure learning for Gaussian graphical models (GGMs) has long assumed i.i.d. samples—penalized likelihood (GLASSO), neighborhood regression (Meinshausen-Bühlmann), and information-theoretically optimal DICE all rely on this assumption. However, many real-world datasets arise from dynamical processes: epidemics, coordination games, or intermediate results from MCMC sampling—i.i.d. samples may not exist at all.

**Limitations of Prior Work**: (1) Bresler 2014 pioneered "learning Ising from Glauber," but Ising variables are bounded ($\{-1,+1\}$), so the approach does not transfer directly to Gaussians; (2) An earlier version of this work (TRD25) used an "i,j,i three-step window + ratio statistic," but Mahbod Majid pointed out that the numerator and denominator of the ratio are not independent—the update of i is not decoupled from previous noise, requiring strong assumptions; (3) There is a lack of tools to "decorrelate" the trajectory into i.i.d. samples for use with existing i.i.d. learners (like DICE).

**Key Challenge**: Gaussian variables are unbounded over the real line, so traditional Ising-style "sign flip probability" estimation fails completely; Glauber trajectories are strongly correlated, and there is no high-dimensional TV bound to convert them to i.i.d. samples (Wasserstein bounds via Kantorovich-Rubinstein do not yield TV bounds, as the Gaussian Gibbs transition kernel is not globally Lipschitz).

**Goal**: (i) Fix the dependency flaw in the ratio estimator and provide a truly usable local edge detection algorithm; (ii) Under the Dobrushin condition, prove that after burn-in/thinning, the trajectory is close in joint TV to i.i.d. Gaussian samples, thus reducing "dynamical structure learning" to i.i.d. structure learning; (iii) Prove information-theoretic lower bounds to locate the minimax position of the algorithms.

**Key Insight**: The authors replace "i,j,i" with "i,i,j,i"—inserting an extra i update as a buffer, so that the subsequent change in i is decoupled from previous i noise; they also use a "product" rather than a "ratio" statistic, yielding a controlled estimator with conditional expectation $\beta_{ij}\theta_{jj}^{-1}$. On the other hand, they leverage Wasserstein contraction of Gaussian Gibbs under the Dobrushin condition, plus a novel "thresholded Lipschitz" technique to upgrade the Wasserstein bound to a TV bound.

**Core Idea**: Both approaches are equally important—"test without waiting for mixing" (LET-GL) and "wait for mixing, then reduce to i.i.d." (BTR-GL); the trade-offs are distinct (the former requires no Dobrushin assumption but has higher sample complexity and perfect parallelism; the latter requires Dobrushin but achieves near-minimax sample complexity).

## Method

### Overall Architecture
Observe a continuous-time Glauber trajectory $\{\mathbf{Y}^{(t)}\}_{t=0}^T$, where at each time $S_n$, a coordinate $I^{(n)} \in [p]$ is randomly selected and updated according to the conditional Gaussian $\mathcal{N}(\sum_{j\in N(i)} \beta_{ij}X_j^{(n-1)}, \sigma_{X_i|N(i)}^2)$. The dataset is $\mathcal{D} = \{(\mathbf{X}^{(n)}, S_n, I^{(n)})\}_{n=0}^N$. The algorithm aims to recover the edge set $E$ of the underlying graph $G$. LET-GL performs edge-by-edge testing directly on the trajectory; BTR-GL discards the first $\mathfrak{b}$ steps (burn-in), then retains one sample every $\mathfrak{t}$ steps, $\mathbf{Y}^{(s)} := \mathbf{X}^{(\mathfrak{b}+s\mathfrak{t})}$, and treats $\{\mathbf{Y}^{(s)}\}$ as approximately i.i.d. Gaussian samples for i.i.d. structure learners like DICE.

### Key Designs

1. **LET-GL: i,i,j,i window + product statistic (local edge detection without mixing)**:

    - **Function**: For each candidate edge $\{i,j\}$, constructs a separate hypothesis test; time complexity is perfectly parallelizable to per-core $\tilde{\mathcal{O}}(d^2 p)$, matching Meinshausen-Bühlmann neighborhood regression.
    - **Mechanism**: The trajectory is divided into $k_\max = \lfloor T/\tau\rfloor$ blocks of length $\tau$, each further split into four equal parts $W_1, W_2, W_3, W_4$. Define the update event $U_{ij}^k$: $W_1, W_2$ each have at least one i update and no j; $W_3$ has j and no i; $W_4$ has i and no j. When $U_{ij}^k \cap Q_{ij}^k$ ("neighbor quiet" event) occurs, record the product of two increments $\Delta Y_i^k \cdot \Delta Y_j^k$, whose conditional expectation (Lemma 1) satisfies $|E[\cdot]| \geq |\beta_{ij}|\theta_{jj}^{-1}$, and is zero if there is no edge. The final statistic is $T_{ij} = |\frac{1}{k_\max}\sum_k \mathbb{1}_{U_{ij}^k}\Delta Y_i^k \Delta Y_j^k|$, compared to threshold $\rho$ for edge detection.
    - **Design Motivation**: The extra i update in "i,i,j,i" acts as a buffer, making the final i change conditionally independent of previous i noise—this is key to fixing the dependency flaw in the earlier ratio estimator. The product form is more stable than the ratio, as ratios can explode in unbounded Gaussians. By tuning $\tau$, the method balances two conflicting events: large $\tau$ increases $U_{ij}^k$ frequency but makes $Q_{ij}^k$ less likely; small $\tau$ has the opposite effect.

2. **TV mixing bound for high-dimensional random-scan Gaussian Gibbs (Lemma 10, technical core)**:

    - **Function**: Proves that under Dobrushin radius $r = \max_i \sum_{j\neq i}|\beta_{ij}| < 1$, $\|K^t(x,\cdot) - \pi\|_\mathrm{TV} \leq \varepsilon$ requires only $t \geq C\cdot \frac{p}{1-r}\log(p^{3/2}/\varepsilon)$ steps—after normalization, the mixing time is only polylogarithmic in dimension $p$, independent of the spectral gap or the condition number of $\Theta$.
    - **Mechanism**: The Gaussian Gibbs transition kernel is not globally Lipschitz, so the standard Kantorovich-Rubinstein route from Wasserstein to TV fails. The authors introduce a "thresholded (approximate) Lipschitz" property—the smoothed kernel is Lipschitz on a high-probability set, with the failure event appearing as an explicit additive defect. Combining burn-in/thinning decomposition and Wasserstein contraction results for random-scan Gibbs (Wang et al.), they show that the subsampled trajectory is close in joint TV to $\pi^{\otimes m}$.
    - **Design Motivation**: The entire reduction in BTR-GL relies on this TV bound—only with joint TV closeness can the guarantees of i.i.d. learners (like DICE) be transferred. This high-dimensional TV bound is itself an independent contribution to the MCMC community.

3. **BTR-GL: burn-in + thinning + black-box i.i.d. learner**:

    - **Function**: "Decorrelates" the Glauber trajectory into approximately i.i.d. samples, then uses an i.i.d. GGM learner as a black box. When DICE is used as the base learner, the total observation time is $\mathcal{O}(dp\,\mathrm{polylog}(p/\delta)/(\kappa^2(1-r)))$, where $\kappa = \min_{\{i,j\}\in E}|\beta_{ij}\beta_{ji}|^{1/2}$ is the minimum normalized edge strength.
    - **Mechanism**: Discards the first $\mathfrak{b}$ steps to eliminate initialization bias, then retains one sample every $\mathfrak{t}$ steps. Lemma 10 guarantees that $(\mathbf{Y}^{(0)}, \dots, \mathbf{Y}^{(m-1)})$ is $\varepsilon$-close in TV to $\pi^{\otimes m}$. The $1-\delta$ high-probability guarantees of DICE and similar learners thus transfer to this trajectory, with the TV closeness $\varepsilon$ entering the union bound.
    - **Design Motivation**: In practice, many GGM systems satisfy the Dobrushin condition (locally weak coupling). BTR-GL "amortizes" the mixing time into the observation time, achieving near-minimax statistical complexity—much more efficient than LET-GL under strong dependence, but at the cost of losing LET-GL's "no mixing assumption" and "per-edge parallelism" advantages.

### Loss & Training
This is a theory + algorithm paper, with no traditional training. The key hyperparameters for LET-GL are $\tau$ (window length, balancing $\mathbb{P}[U_{ij}^k] = [(1-e^{-\tau/4})e^{-\tau/4}]^4$ and $\mathbb{P}[Q_{ij}^k] \geq e^{-\tau d}$) and $\rho$ (edge threshold). The algorithm also introduces a high-probability bounded event $B_\delta = \{\max |Y_i^{(t)}| \leq y_\max\}$ (where $y_\max = C_1 \sigma_\max\sqrt{\log(p/\delta)}$), ensuring that the test statistic is almost surely bounded $|T_{ij}^k| \leq 4 y_\max^2$ under the conditional probability, enabling the use of martingale concentration inequalities.

## Key Experimental Results

### Main Results (Synthetic d-regular GGM, fixed $\delta$)

| Algorithm | Observation Time Complexity | Needs Dobrushin? | Computational Parallelism |
|-----------|----------------------------|------------------|--------------------------|
| **LET-GL** | $\mathcal{O}(d^3 p\,\mathrm{polylog}\,p / \beta_\min^5)$ | No | Perfect parallelism, per-edge $\tilde{\mathcal{O}}(d^2 p)$ |
| **BTR-GL + DICE** | $\mathcal{O}(d p\,\mathrm{polylog}(p/\delta) / (\kappa^2(1-r)))$ | Yes | Inherits DICE complexity $\mathcal{O}(p^{2d+1})$ |
| GLASSO | — | — | $\mathcal{O}(p^3)$ |
| PC algorithm | — | — | $\mathcal{O}(p^{d+2})$ |
| Meinshausen neighborhood | — | — | Per-node parallelism |
| **Lower Bound (Ours)** | $\Omega(\log(p-d)/\beta_\min^2)$; when $\beta_\min = \Theta(1/d)$, $\Omega(d^2 \log p)$ | — | — |

### Theoretical Results Comparison

| Key Theorem | Content | Significance |
|-------------|---------|--------------|
| Theorem 1 | LET-GL recovers $E$ with high probability for $T = \mathcal{O}(d^3 p\,\mathrm{polylog}\,p / \beta_\min^5)$ | Provable guarantee without mixing assumption |
| Lemma 10 | High-dimensional random-scan Gaussian Gibbs TV mixing time $\tilde{\mathcal{O}}(p/(1-r))$ under Dobrushin | Independent MCMC value, condition-number-free |
| BTR-GL Main Theorem | Under Dobrushin, $T = \mathcal{O}(dp/(\kappa^2(1-r)))$ + polylog suffices | Near minimax optimal when $\kappa \asymp \beta_\min$, $d$ bounded |
| Information-theoretic Lower Bound | Any algorithm $T \geq \Omega(\log(p-d)/\beta_\min^2)$ | Sets the fundamental limit for the problem class |

### Key Findings
- **Product statistic outperforms ratio statistic**: The earlier TRD25 i,j,i ratio failed due to dependency flaws; the i,i,j,i product form in this work both fixes the dependency and adapts to the unbounded Gaussian domain.
- **BTR-GL is near minimax under Dobrushin**: When $\kappa \asymp \beta_\min$, $d$ bounded, and $1-r$ constant, BTR-GL's sample complexity matches the $\Omega(d^2\log p)$ lower bound (up to polylog factors).
- **LET-GL's parallelism is unique**: Each candidate edge is completely independent, so all $\binom{p}{2}$ edges can be tested in parallel, with per-core cost $\tilde{\mathcal{O}}(d^2 p)$—unmatched by GLASSO/PC, etc.
- **TV mixing bound is condition-number-free**: Traditional spectral gap mixing times are hampered by $\kappa(\Theta)$; the transport-side approach here bypasses this, especially useful for high-dimensional sparse GGMs.
- **Complementary to concurrent work SWMM26**: SWMM26 uses i,i,j,i window + ratio + robust aggregation; this work uses i,i,j,i window + product + martingale concentration. Both independently validate the window choice, and BTR-GL provides a Dobrushin-based route absent in SWMM26.

## Highlights & Insights
- **The "buffer" idea in i,i,j,i windows is elegant**: Inserting an extra update as a conditional independence "insulator" both removes dependency contamination and enables the ratio-to-product transition. This "purifying estimators via update schedule design" can be generalized to other dynamical models.
- **Thresholded Lipschitz is a minor innovation in transport theory**: Formalizing "Lipschitz on a high-probability set + additive defect for failure events" may benefit TV mixing analysis for other non-globally Lipschitz kernels.
- **The duality of the two algorithms is instructive**: One side "tests without waiting for mixing" (local, assumption-free, high complexity, strong parallelism), the other "waits for mixing then reduces" (global, needs Dobrushin, low complexity, serial). This "local/global duality" recurs in dynamical structure learning and is worth further exploration.
- **Information-theoretic lower bound provides clarity**: $\Omega(d^2\log p)$ is not just a result but a "signpost"—indicating that under Dobrushin, the ceiling is near, and future improvements can only seek small constant factors in the polylog.

## Limitations & Future Work
- LET-GL's sample complexity has a high polynomial dependence (fifth power) on $\beta_\min$; observation time inflates when edge strengths are weak.
- BTR-GL assumes Dobrushin $r < 1$, which may not hold for dense or strongly coupled GGMs; mixing analysis under weaker conditions is needed.
- The constant $C$ in the TV mixing bound is not explicit; practical implementation requires empirical tuning; guidelines for burn-in length $\mathfrak{b}$ and thinning $\mathfrak{t}$ are lacking.
- No real data experiments—only synthetic d-regular GGMs; performance in applications like finance or neuroscience remains to be validated.
- No treatment of mis-specification—if the data are not Gaussian or not generated by Glauber dynamics, algorithm behavior is unknown.
- BTR-GL's i.i.d. black box relies on DICE, which itself has $\mathcal{O}(p^{2d+1})$ combinatorial complexity; computational cost remains high. The trade-off between complexity and accuracy when replacing the base learner with GLASSO or neighborhood regression is not fully discussed.

## Related Work & Insights
- **vs Bresler 2014 (Ising from Glauber)**: This work generalizes to Gaussians; moving from bounded discrete to unbounded continuous introduces new challenges (unbounded statistics, continuous noise), addressed via "event $B_\delta$ + conditional expectation" techniques.
- **vs SWMM26 (Shen-Wu-Majid-Moitra concurrent work)**: Both identify the dependency issue in i,j,i and use i,i,j,i windows; SWMM26 uses ratio + robust aggregation without Dobrushin, this work uses product + Dobrushin, providing BTR-GL as an extra route. The methods are different but complementary.
- **vs DICE (Misra 2020)**: DICE is the information-theoretically optimal GGM learner under i.i.d. settings; this work uses it as a black box in BTR-GL, extending DICE's capabilities to dynamical data.
- **vs Meinshausen-Bühlmann neighborhood selection**: MB uses Lasso for per-node regression; LET-GL does per-edge testing. Both are per-X parallel, but LET-GL does not require the i.i.d. assumption.
- **vs Wang 2014/2017 (Wasserstein contraction for Gibbs)**: This work leverages their Wasserstein contraction, with the key contribution being the successful reduction from Wasserstein to TV, enabling connection to i.i.d. learners.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide GGM structure learning algorithms, information-theoretic lower bounds, and an independently valuable high-dimensional TV mixing bound for Gaussian Glauber dynamics—all three are new.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments validate theory, but no real data; theory is solid but experimental depth is average.
- Writing Quality: ⭐⭐⭐⭐ The introduction candidly discusses "flaws in earlier versions + fixes + relation to SWMM26"; theorem statements are clear, with proof sketches outlining key ideas.
- Value: ⭐⭐⭐⭐ Important progress for the graphical model theory community, with potential applications in MCMC diagnostics and epidemic modeling; the TV mixing bound is independently useful for the MCMC community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](../../NeurIPS2025/others/a_unified_framework_for_variable_selection_in_modelbased_clu.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)
- [\[ICML 2026\] Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)
- [\[ICML 2026\] DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](../../AAAI2026/others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)

</div>

<!-- RELATED:END -->
