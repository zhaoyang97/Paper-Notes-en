---
title: >-
  [Paper Note] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as a regularization term. It replaces the rigid "minimum eigenvalue $\Omega(1)$ of the second moment for each task" assumption found in prior work with a relative "balance constant" $B$. T
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: f28ac5c00c0efed6
---
# Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety

**Conference**: ICML 2026  
**arXiv**: [2605.17126](https://arxiv.org/abs/2605.17126)  
**Code**: https://github.com/seokjinkim0428/Multi-task-Linear-Regression  
**Area**: Statistical Learning Theory / Multi-task Learning / Robust Regression  
**Keywords**: Multi-task Linear Regression, Matrix-weighted Regularization, Minimum Eigenvalue, Balance Constant, Outlier Tasks, Safety Guarantee

## TL;DR
This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as a regularization term. It replaces the rigid "minimum eigenvalue $\Omega(1)$ of the second moment for each task" assumption found in prior work with a relative "balance constant" $B$. This provides minimax rates, adaptivity, and safety guarantees that fall back to Independent Task Learning (ITL) in high-dimensional scenarios involving ill-conditioned, low-rank, or outlier-contaminated tasks.

## Background & Motivation

**Background**: The robust linear regression framework for multi-task learning with "few outlier tasks + majority related tasks" is epitomized by ARMUL (Duan & Wang, 2023). It jointly estimates parameters for $m$ tasks by sharing a central parameter $\beta$ with $\ell_2$ distance regularization $\lambda\|\theta_j-\beta\|_2$, adaptively aggregating without knowing the outlier proportion $\varepsilon$ and similarity radius $\delta$.

**Limitations of Prior Work**: All theoretical guarantees in this line of work (Duan & Wang 2023, Tian et al. 2025/2026) rely on the Lower Boundedness of Second Moments (LBSM): requiring the empirical second moment of each task to satisfy $\rho \mathbf{I}_d \preceq \bm\Sigma_j \preceq L\mathbf{I}_d$ with $\rho=\Omega(1)$. Their upper bound $\rho^{-2}\cdot(d/(mn)+\min(L^4\delta^2/\rho^2,d/n)+\varepsilon^2 d/n)$ directly becomes vacuous in realistic scenarios where $\rho$ is very small (e.g., uniform distribution on a high-dimensional sphere $\rho\asymp 1/d$, fast spectral decay, or adaptive sampling in linear bandits).

**Key Challenge**: LBSM is required to ensure identifiability under the "Euclidean parameter error" metric. However, for prediction MSE, directions with the weakest observations should not be heavily penalized—the prediction rate of single-task least squares $\tilde{\mathcal O}(d/n)$ does not require a bounded condition number. In other words, **LBSM is a requirement from the perspective of parameter error, rather than prediction error.**

**Goal**: To decompose this into three sub-problems: (i) Can multi-task transfer benefits be proven without $\rho=\Omega(1)$? (ii) Can a safety rate that automatically falls back to ITL be guaranteed when no similarity structure exists? (iii) What should the "minimal condition" replacing LBSM look like?

**Key Insight**: The authors observe that $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt{n_j}$ measures inconsistency in the **prediction space**, not the parameter space. By placing regularization in the "whitened coordinates" $\bm\Sigma_j^{1/2}\theta$, unobserved directions are naturally not penalized. The remaining problem is how to maintain information sharing after whitening—this requires the second moments of various tasks to be comparable in some average sense.

**Core Idea**: Replace $\ell_2$ regularization with the matrix-weighted norm $\|\theta_j-\beta\|_{\bm\Sigma_j}$ and introduce a **one-sided, average-type** "balance" assumption $\bm\Sigma_j\preceq B\cdot\bm\Sigma_{\mathbf S}$ (the second moment of each task is controlled by the average second moment of inlier tasks), where $B$ replaces the role of $\rho^{-1}$.

## Method

### Overall Architecture
Consider $m$ tasks, each with $n$ samples $(x_{ji},y_{ji})$ following a linear model $y_{ji}=x_{ji}^\top\theta_j^\star+\varepsilon_{ji}$. Among the unknown parameters, $|\mathbf S|/m\ge 1-\varepsilon$ inlier parameters fall within an $\ell_2$ ball of radius $\delta$ centered at $\theta^\star$, while the rest are arbitrary outliers; $\varepsilon,\delta,\mathbf S$ are all unknown. The estimator MTLR formulates the loss for all tasks plus the center $\beta$ as a joint convex optimization, where the regularization term is weighted by each task's empirical second moment $\bm\Sigma_j=\mathbf X_j^\top\mathbf X_j/n$. The evaluation focuses on the prediction MSE $\mathcal E^{\mathrm{in}}_j=\|\hat\theta_j-\theta_j^\star\|_{\bm\Sigma_j}^2$.

### Key Designs

**1. Joint Convex Loss with Matrix-Weighted Regularization: Removing Dependency on Minimum Eigenvalues**

ARMUL treats strong and weak directions equally using $\ell_2$ regularization $\lambda\|\theta_j-\beta\|_2$, which forcibly pulls directions that independent tasks cannot identify toward the center $\beta$, thus embedding the factor $\rho^{-2}$ into the guarantees. Ours switches to a locally whitened matrix-weighted regularization: the loss $\mathcal L(\Theta)=\sum_{j=1}^{m} w_j\big(f_j(\theta_j)+\lambda_j\|\theta_j-\beta\|_{\bm\Sigma_j}\big)$, where $f_j(\theta)=\|\mathbf Y_j-\mathbf X_j\theta\|_2^2/(2n_j)$. The term $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt n$ measures the "prediction difference between $\theta_j$ and $\beta$ under task $j$'s own design matrix." Equivalently, this is $\ell_2$ regularization after whitening by $\bm\Sigma_j^{1/2}$—if a direction is barely observed by $\mathbf X_j$, the deviation in that direction is not penalized, effectively "muting" weak observation directions. With $\lambda_j\asymp\sqrt{d/n_j}$ and reparameterization $v_j=\theta_j-\beta$, the objective is jointly convex and solved via L-BFGS-B. In short, aggregation occurs only in observed directions, fundamentally severing the dependency on $\rho$.

**2. Balance Constant $B$ Replacing LBSM: Characterizing Geometric Compatibility via Relative Spectral Conditions**

LBSM requires an absolute spectral lower bound $\rho\mathbf I\preceq\bm\Sigma_j\preceq L\mathbf I$ with $\rho=\Omega(1)$ for every task, which is often violated in high dimensions. The authors propose a relative, average-type condition—Assumption 1 requires the existence of $B\in[1,\infty]$ such that $\bm\Sigma_j\preceq B\cdot \bm\Sigma_{\mathbf S}$ holds for all $j$, where $\bm\Sigma_{\mathbf S}=|\mathbf S|^{-1}\sum_{j\in\mathbf S}\bm\Sigma_j$ is the inlier average. This is merely a one-sided upper bound compared against the "average" rather than "any pair," explicitly allowing individual $\bm\Sigma_j$ to be rank-deficient or have spectral decay. LBSM is a special case: if $\rho\mathbf I\preceq\bm\Sigma_j\preceq L\mathbf I$, then $B=L/\rho$. For two low-rank groups (one $\bm\Sigma=\mathbf I$, another $=\mathbf 0$), $B=|\mathbf S|/|\mathbf S\cap\mathbf I|$. Combined with covariate concentration $\nu_j$, the empirical $B$ transitions smoothly to its population version $\bar B$. The intuition: if the second moments of inlier tasks lie in similar directions, the average moment $\bm\Sigma_{\mathbf S}$ acts as a good "shared skeleton." Each $\bm\Sigma_j$ being controlled by it implies sharable information; if $B=\infty$ (disjoint information directions), coordination is unnecessary, and the algorithm should revert to ITL.

**3. Two-layer "Safety + Adaptivity" Rates: Automatic Fallback for a Single Estimator**

In practice, users do not know $B,\varepsilon,\delta$. A guarantee that requires choosing the correct hyperparameters to be effective has little utility. Theorem 2 provides two bounds that hold simultaneously with high probability. **Safety** holds for **arbitrary** $B,\varepsilon,\delta$: $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim q^2(d/n)\zeta$, matching the minimax rate of independent tasks ($\zeta=\log(16m/\kappa)$)—the algorithm "loses nothing in the worst case." **Adaptivity** gives $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim (Bd/(mn)+\min(B\delta^2,q^2d/n)+q^2B^2\varepsilon^2 d/n)\zeta$ for inliers $j\in\mathbf S$ when Assumption 1 holds and $B \lesssim \min(1/\varepsilon, m)$, achieving minimax optimality. Theorem 3 extends in-sample bounds to population MSE using the empirical-population comparability constant $\nu_j$, and Theorem 4 mirrors these conclusions for GLMs with bounded link function curvature. Neither rate requires prior information, which is key to its practical deployment.

### Loss & Training
Square loss is used for linear models, and negative log-likelihood $f_j(\theta)=\frac{1}{n}\sum_i(\psi(x_{ji}^\top\theta)-y_{ji}x_{ji}^\top\theta)$ for GLMs. $\lambda_j$ is set to $q\sqrt{d\zeta/n}$, where $q$ is tuned via 5-fold CV in $\{0.05,0.10,\dots,0.50\}$. For GLMs, parameters are restricted to $\mathbf B(0,\xi)$ to ensure the link function curvature stays within $[\alpha_\ell,\alpha_u]$.

## Key Experimental Results

### Main Results
Synthetic benchmark with $n=100, m=30, d=30$. Covariance shape for $\mathbf x$ involves unit ball plus $k^{-\alpha}$ coordinate scaling, with 30 Monte Carlo trials. Comparison includes DP (pooling), ITL (independent), and ARMUL (Duan-Wang 2023). The table below shows all-task MSE across correlation scans ($B\equiv 1, \varepsilon=0.1, \alpha=1$).

| $\delta$ | Ours | ARMUL | DP | ITL |
|---|---|---|---|---|
| 0.2 | **0.0138** | 0.041+ | >0.04 | >0.04 |
| 0.8 | **~0.020** | 0.041+ | >0.04 | >0.04 |
| 3.2 | **0.0259** | 0.041+ | >0.04 | >0.04 |

HAR real-world data (Multi-task logistic regression for 30 subjects, binary classification of standing vs. others, $d=561$, sub-task $n\approx 343$, no PCA), 20% test, 30 random splits.

| Method | Mean Error (%) | SD |
|------|--------------|-----|
| **Ours** | **1.25** | 0.32 |
| ITL | 4.67 | 0.51 |
| DP | 7.61 | 0.46 |
| ARMUL | (Digits missing from Table 1) | — |

### Ablation Study
The paper uses four univariate scans instead of traditional ablation, varying one variable at a time to correspond to a factor in the theory.

| Scanned Variable | Range | Key Findings |
|---------|---------|---------|
| Similarity $\delta$ | 0.2-3.2 | Ours consistently leads ITL/ARMUL, with largest advantage at small $\delta$. |
| Outlier Proportion $\varepsilon$ | 0.05-0.4 | All-task MSE smoothly rises from 0.006 to 0.046 without jumps. ITL is strongest on outlier tasks (expected). |
| Spectral Decay $\alpha$ | 0-2.0 | Ours remains dominant at $\alpha=1.5, 2.0$ (highly ill-conditioned), validating that LBSM is no longer necessary. |
| Balance $\bar B$ | 5, 10, 15, 20 | Ours is best at $\bar B=5$; as $\bar B$ increases, ITL catches up, and Ours converges to ITL without catastrophic negative transfer. |

### Key Findings
- In strongly ill-conditioned cases ($\alpha=2$), ARMUL's bound becomes vacuous due to $\rho\asymp d^{-2}$, whereas Ours remains robust—empirical evidence of "removing $\rho^{-2}$."
- The balance scan reveals a clear "dual-regime switch": when $\bar B$ is small, Ours benefits from multi-task gains; when $\bar B$ is large, Ours coincides with ITL—numerically realizing the safety part of Theorem 2.
- On HAR, where PCA is skipped for logistic regression (removing a signature ARMUL preprocessing step), Ours achieves a 1.25% error rate, nearly 4x lower than ITL's 4.67%, showing that matrix-weighted regularization remains effective in real high-dimensional, low SNR contexts.

## Highlights & Insights
- Switching from "parameter space regularization" to "prediction space regularization" is a minimal modification—just one line of formula change in engineering—but theoretically resolves the $\rho$ factor that has plagued this line of research for years.
- The "balance constant $B$" is strikingly similar to the covariate shift coverage conditions in transfer learning. This aligns robust MTL with covariate shift analysis, allowing for the potential use of various covariate shift tools in the future.
- The **single-estimator dual-rate** structure of "Safety + Adaptivity" (without needing to know beforehand whether to transfer) is a highly reusable paradigm applicable to federated learning, personalized recommendations, and other statistical estimation tasks with "individual heterogeneity + group sharability."

## Limitations & Future Work
- Theoretical results still assume an upper bound on $\|\bm\Sigma_j\|_{\mathrm{op}}\le 1$, which might require additional conditions under heavy-tailed or adversarial designs.
- Experimental scale is relatively small ($d\le 561$). The measured values of "Balance" and estimation stability in over-parameterized deep learning scenarios have not been fully tested.
- The diagnostic estimation of $B$ ($B_{\mathrm{emp}}$) depends on pseudo-inverses and generalized square roots, which can be numerically unstable when $m$ is small; the paper admits this is more of an intuitive diagnostic than an estimator input.
- GLM link functions require curvature bounds $\alpha_\ell\le\psi''\le\alpha_u$, meaning non-smooth cases like softmax or hinge loss still require separate analysis.

## Related Work & Insights
- **vs. Duan & Wang (2023, ARMUL)**: Same $\ell_2$-closeness + outlier model, but they use $\ell_2$ regularization resulting in $\rho^{-2}$ dependency. Ours uses matrix-weighted regularization to eliminate $\rho$ and explicitly derives safety rates; our rate structure matches theirs in favorable regimes.
- **vs. Tian et al. (2025/2026, Low-rank Shared Representation)**: They pursue "shared low-rank subspaces," requiring different identifiability conditions. Ours follows the "$\ell_2$-closeness" route; the two are complementary, providing optimal transfer under different structural assumptions.
- **vs. Bhattacharya et al. (2025, Semi-parametric Multi-task Inference)**: An extension of the Duan-Wang framework to semi-parametric settings, but still relying on LBSM. Our matrix-weighting technique could potentially be integrated to remove their spectral lower bound assumptions.
- **vs. Soare et al. (2014) / Wang et al. (2021) / Sessa et al. (2023)**: Classic $\ell_2$-closeness traditions without outliers. Ours provides unified coverage ($\varepsilon=0$ as a special case) and ensures safety rates even under ill-conditioned designs.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond](../../ICLR2026/learning_theory/bounds_of_chain-of-thought_robustness_reasoning_steps_embed_norms_and_beyond.md)
- [\[ICML 2026\] Task-Restricted Symmetries in Recurrent Weight Space](task-restricted_symmetries_in_recurrent_weight_space.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/learning_theory/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](../../ICML2025/learning_theory/heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)

</div>

<!-- RELATED:END -->
