---
title: >-
  [Paper Note] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as the regularization term. It replaces the rigid "Lower Boundedness of Second Moments (LBSM) $\Omega(1)$" assumption found in prior work with a relative "balance constant" $B$. This provi
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 46623707e19eff58
---
# Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety

**Conference**: ICML 2026  
**arXiv**: [2605.17126](https://arxiv.org/abs/2605.17126)  
**Code**: https://github.com/seokjinkim0428/Multi-task-Linear-Regression  
**Area**: Statistical Learning Theory / Multi-task Learning / Robust Regression  
**Keywords**: Multi-task linear regression, matrix-weighted regularization, minimum eigenvalue, balance constant, outlier tasks, safety guarantees

## TL;DR
This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as the regularization term. It replaces the rigid "Lower Boundedness of Second Moments (LBSM) $\Omega(1)$" assumption found in prior work with a relative "balance constant" $B$. This provides minimax rates, adaptivity, and safety guarantees (fallback to independent task learning, ITL) simultaneously in ill-conditioned, low-rank, or high-dimensional scenarios with outlier tasks.

## Background & Motivation

**Background**: The framework for robust linear regression in multi-task learning with a "minority of outlier tasks + majority of related tasks" is represented by ARMUL (Duan & Wang 2023). It jointly estimates parameters for $m$ tasks by sharing a center parameter $\beta$ and using $\ell_2$ distance regularization $\lambda\|\theta_j-\beta\|_2$, adapting to unknown outlier proportions $\varepsilon$ and similarity radii $\delta$.

**Limitations of Prior Work**: All theoretical guarantees in this line of work (Duan & Wang 2023, Tian et al. 2025/2026) depend on the Lower Boundedness of Second Moments (LBSM), which requires the empirical second moments of each task to satisfy $\rho \mathbf{I}_d \preceq \bm\Sigma_j \preceq L\mathbf{I}_d$ with $\rho=\Omega(1)$. The upper bound $\rho^{-2}\cdot(d/(mn)+\min(L^4\delta^2/\rho^2,d/n)+\varepsilon^2 d/n)$ becomes vacuous in realistic scenarios where $\rho$ is very small, such as high-dimensional uniform distributions on spheres ($\rho\asymp 1/d$), fast spectral decay, or adaptive sampling in linear bandits.

**Key Challenge**: LBSM is required to ensure identifiability under the "Euclidean parameter error" metric. However, for prediction MSE, directions with the weakest observations should not be heavily penalized—the prediction rate of single-task least squares $\tilde{\mathcal O}(d/n)$ does not require a bounded condition number. In other words, **LBSM is a requirement from the perspective of parameter error rather than prediction error**.

**Goal**: The paper addresses three sub-problems: (i) whether multi-task transfer gains can be proven without $\rho=\Omega(1)$; (ii) whether a safety rate back to ITL can be guaranteed when similarity structure is absent; and (iii) what the "minimal condition" replacing LBSM should look like.

**Key Insight**: The authors observe that $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt{n_j}$ measures inconsistency in the **prediction space** rather than the parameter space. By placing the regularization in "whitened coordinates" $\bm\Sigma_j^{1/2}\theta$, unobserved directions are naturally not penalized. The remaining problem is ensuring information sharing after whitening, which requires the second moments of different tasks to be comparable in some average sense.

**Core Idea**: Replace the $\ell_2$ regularization with the matrix-weighted norm $\|\theta_j-\beta\|_{\bm\Sigma_j}$ and introduce a **one-sided, average-type** "balance" assumption $\bm\Sigma_j\preceq B\cdot\bm\Sigma_{\mathbf S}$ (the second moment of each task is controlled by the average of inlier tasks), where $B$ replaces the role of $\rho^{-1}$.

## Method

### Overall Architecture
Consider $m$ tasks, each with $n$ samples $(x_{ji},y_{ji})$ following a linear model $y_{ji}=x_{ji}^\top\theta_j^\star+\varepsilon_{ji}$. Among the unknown parameters, $|\mathbf S|/m\ge 1-\varepsilon$ inliers lie within an $\ell_2$ ball of radius $\delta$ centered at $\theta^\star$, while the rest are arbitrary outliers; $\varepsilon, \delta, \mathbf S$ are all unknown. The estimator MTLR formulates the loss across all tasks plus the center $\beta$ as a joint convex optimization, where the regularization term for each task is weighted by its own empirical second moment $\bm\Sigma_j = \mathbf X_j^\top \mathbf X_j / n$. The performance is evaluated using prediction MSE $\mathcal E^{\mathrm{in}}_j=\|\hat\theta_j-\theta_j^\star\|_{\bm\Sigma_j}^2$.

### Key Designs

**1. Joint Convex Loss with Matrix-Weighted Regularization: Removing Dependency on Minimum Eigenvalues**

ARMUL uses $\ell_2$ regularization $\lambda\|\theta_j-\beta\|_2$ to treat strong and weak directions equally, forcibly pulling directions that single tasks cannot identify toward the center $\beta$, leading to the factor $\rho^{-2}$ in the guarantees. This paper switches to locally whitened matrix-weighted regularization: the loss $\mathcal L(\Theta)=\sum_{j=1}^{m} w_j\big(f_j(\theta_j)+\lambda_j\|\theta_j-\beta\|_{\bm\Sigma_j}\big)$, where $f_j(\theta)=\|\mathbf Y_j-\mathbf X_j\theta\|_2^2/(2n_j)$. The term $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt n$ measures the "prediction difference between $\theta_j$ and $\beta$ under task $j$'s own design matrix." This is equivalent to $\ell_2$ regularization after whitening by $\bm\Sigma_j^{1/2}$—if a direction is barely observed by $\mathbf X_j$, the bias in that direction is not penalized, effectively "muting" weak directions. With $\lambda_j\asymp\sqrt{d/n_j}$ and reparameterization $v_j=\theta_j-\beta$, the objective is jointly convex and solved via L-BFGS-B. By aggregating only in observed directions, the dependency on $\rho$ is fundamentally eliminated.

**2. Balance Constant $B$ Replacing LBSM: Characterizing Geometric Compatibility via Relative Spectral Conditions**

LBSM requires absolute spectral lower bounds $\rho\mathbf I\preceq\bm\Sigma_j\preceq L\mathbf I$ with $\rho=\Omega(1)$, which fails in high dimensions or adaptive sampling. The authors adopt a relative, average-type condition—Assumption 1 requires the existence of $B\in[1,\infty]$ such that $\bm\Sigma_j\preceq B\cdot \bm\Sigma_{\mathbf S}$ for all $j$, where $\bm\Sigma_{\mathbf S}=|\mathbf S|^{-1}\sum_{j\in\mathbf S}\bm\Sigma_j$ is the average of inliers. This is a one-sided upper bound compared against the "average" rather than "any pair," explicitly allowing rank deficiency or spectral decay in individual $\bm\Sigma_j$. LBSM is a special case: if $\rho\mathbf I\preceq\bm\Sigma_j\preceq L\mathbf I$, then $B=L/\rho$. For pairwise comparable tasks, $B=B'$. For low-rank dual populations (one group $\bm\Sigma=\mathbf I$, the other $\bm\Sigma=\mathbf 0$), $B=|\mathbf S|/|\mathbf S\cap\mathbf I|$. Combined with covariate concentration $\nu_j$, the empirical $B$ transitions smoothly to its population version $\bar B$. The intuition is that if inlier second moments align in similar directions, the average $\bm\Sigma_{\mathbf S}$ serves as a good "common skeleton." Being controlled by it implies sharable information, whereas if $B=\infty$ (disjoint information directions), coordination is unnecessary and the algorithm should safely revert to ITL.

**3. Two-layer "Safety + Adaptivity" Rate Structure: Automatic Fallback without Switching**

In practice, users do not know $B, \varepsilon, \delta$, and a guarantee requiring correct hyperparameter selection is of little use. Theorem 2 provides two bounds that hold simultaneously with high probability. **Safety** holds for **any** $B, \varepsilon, \delta$: $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim q^2(d/n)\zeta$, matching the minimax rate of independent tasks ($\zeta=\log(16m/\kappa)$)—the algorithm "does no harm" at worst. **Adaptivity** holds for inliers $j\in\mathbf S$ when Assumption 1 is satisfied and $B\lesssim\min(1/\varepsilon,m)$, yielding $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim (Bd/(mn)+\min(B\delta^2,q^2d/n)+q^2B^2\varepsilon^2 d/n)\zeta$, achieving minimax optimality when conditions are favorable. Theorem 3 extends in-sample bounds to population MSE using the empirical-population comparability constant $\nu_j$ and provides a second layer of safety via domain projection $\mathcal E_j(\hat\theta_j^\xi)\lesssim \mathcal E^{\mathrm{in}}_j(\hat\theta_j)+\xi^2 U_j^2/n$. Theorem 4 mirrors these results for GLMs with bounded link function curvature. The fact that neither rate requires prior information is key for deployment.

### Loss & Training
The linear model uses squared loss, and GLMs use negative log-likelihood $f_j(\theta)=\frac{1}{n}\sum_i(\psi(x_{ji}^\top\theta)-y_{ji}x_{ji}^\top\theta)$. $\lambda_j$ is set to $q\sqrt{d\zeta/n}$, with $q$ tuned via 5-fold CV in $\{0.05, 0.10, \dots, 0.50\}$. Parameters for GLMs must be constrained within $\mathbf B(0, \xi)$ to ensure the link function curvature lies in $[\alpha_\ell, \alpha_u]$.

## Key Experimental Results

### Main Results
Synthetic benchmark with $n=100, m=30, d=30$, with covariance shape $\mathbf x$ on a unit ball and $k^{-\alpha}$ coordinate scaling (30 Monte Carlo trials). Comparisons include DP (pooling), ITL (independent), and ARMUL (Duan-Wang 2023). The table below shows total task MSE under correlation scanning ($B\equiv 1, \varepsilon=0.1, \alpha=1$).

| $\delta$ | Ours | ARMUL | DP | ITL |
|---|---|---|---|---|
| 0.2 | **0.0138** | 0.041+ | >0.04 | >0.04 |
| 0.8 | **~0.020** | 0.041+ | >0.04 | >0.04 |
| 3.2 | **0.0259** | 0.041+ | >0.04 | >0.04 |

HAR real-world data (30-subject multi-task logistic regression, binary classification "standing vs. others," $d=561$, sub-tasks $n\approx 343$, no PCA), 20% test set, 30 random splits.

| Method | Mean Error (%) | SD |
|------|--------------|-----|
| **Ours** | **1.25** | 0.32 |
| ITL | 4.67 | 0.51 |
| DP | 7.61 | 0.46 |
| ARMUL | (Table 1 cut off, full digits not provided) | — |

### Ablation Study
The paper uses four single-variable scans instead of traditional ablation, varying one factor to correspond with theoretical variables.

| Variable | Range | Key Finding |
|---------|---------|---------|
| Similarity $\delta$ | 0.2-3.2 | Ours leads ITL/ARMUL throughout, largest advantage at small $\delta$. |
| Outlier Proportion $\varepsilon$ | 0.05-0.4 | Total task MSE rises smoothly from 0.006 to 0.046 with no jumps; ITL strongest on outlier tasks. |
| Spectral Decay $\alpha$ | 0-2.0 | Ours remains dominant at $\alpha=1.5, 2.0$ (highly ill-conditioned), proving LBSM is no longer necessary. |
| Balance $\bar B$ | 5, 10, 15, 20 | Ours is optimal at $\bar B=5$; as $\bar B$ increases, ITL catches up and Ours converges to ITL without catastrophic negative transfer. |

### Key Findings
- Under strongly ill-conditioned settings ($\alpha=2$), ARMUL's bound becomes vacuous due to $\rho\asymp d^{-2}$, while the proposed method remains robust, empirically validating the removal of the $\rho^{-2}$ factor.
- A clear "dual-regime switch" is observed in the balance scan: at small $\bar B$, the method benefits from multi-task gains; at large $\bar B$, it nearly overlaps with ITL, fulfilling the Safety results of Theorem 2.
- On HAR using logistic regression without PCA (removing ARMUL's hallmark preprocessing), our error rate of 1.25% is nearly 4x lower than ITL's 4.67%, showing the effectiveness of matrix-weighted regularization in real high-dimensional low-SNR scenarios.

## Highlights & Insights
- Switching "parameter space regularization" to "prediction space regularization" is a minimal modification—essentially one line of code—but theoretically resolves the $\rho$ factor that plagued this field for years.
- The "balance constant $B$" is strikingly similar to the coverage conditions in covariate shift analysis for transfer learning. This aligns robust MTL with single-source transfer analysis, allowing for the use of covariate shift tools in the future.
- The "Safety + Adaptivity" structure (dual rates for a single estimator without knowing when to transfer) is a highly reusable paradigm applicable to federated learning or personalized recommendation.

## Limitations & Future Work
- Theoretical results still assume an upper bound $\|\bm\Sigma_j\|_{\mathrm{op}}\le 1$, which might require additional conditions under heavy-tailed or adversarial designs.
- Experimental scale is relatively small ($d\le 561$); the "balance constant" and estimation stability in over-parameterized deep learning settings have not been fully tested.
- Diagnostic estimation of $B$ ($B_{\mathrm{emp}}$) depends on pseudo-inverses and generalized square roots, which can be numerically unstable for small $m$; the paper admits this is an intuitive diagnostic rather than an estimator input.
- Requirements for GLM link function curvature ($\alpha_\ell\le\psi''\le\alpha_u$) mean non-smooth cases like softmax or hinge loss still require separate analysis.

## Related Work & Insights
- **vs Duan & Wang (2023, ARMUL)**: Uses the same $\ell_2$-closeness + outlier model, but ARMUL's $\ell_2$ regularization leads to a $\rho^{-2}$ dependency; this paper eliminates $\rho$ via matrix weighting and explicitly includes "safety rates."
- **vs Tian et al. (2025/2026, shared low-rank representation)**: They pursue "shared low-rank subspaces," requiring different identifiability conditions; this paper pursues "$\ell_2$-closeness," providing complementary optimal transfers under different structural assumptions.
- **vs Bhattacharya et al. (2025, semi-parametric MTL inference)**: Extends ARMUL to semi-parametric settings but still relies on LBSM; the matrix-weighting technique here could potentially be integrated to remove their spectral lower bound assumptions.
- **vs Soare et al. (2014) / Wang et al. (2021) / Sessa et al. (2023)**: Traditional $\ell_2$-closeness models without outliers are covered here as special cases ($\varepsilon=0$), with safety rates guaranteed even under ill-conditioned designs.

## Rating
- Novelty: [To be evaluated]
- Experimental Thoroughness: [To be evaluated]
- Writing Quality: [To be evaluated]
- Value: [To be evaluated]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/learning_theory/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](../../ICML2025/learning_theory/heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](../../NeurIPS2025/learning_theory/sample-adaptivity_tradeoff_in_on-demand_sampling.md)

</div>

<!-- RELATED:END -->
