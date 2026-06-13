---
title: >-
  [Paper Note] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety
description: >-
  [ICML 2026][Statistical Learning Theory / Multi-task Learning / Robust Regression][Multi-task linear regression] This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}…
tags:
  - "ICML 2026"
  - "Statistical Learning Theory / Multi-task Learning / Robust Regression"
  - "Multi-task linear regression"
  - "matrix-weighted regularization"
  - "minimum eigenvalue"
  - "balance"
  - "outlier tasks"
  - "safety guarantees"
date: 2026-05-08
content_hash: 0b208001486c8f21
---

# Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety

**Conference**: ICML 2026  
**arXiv**: [2605.17126](https://arxiv.org/abs/2605.17126)  
**Code**: https://github.com/seokjinkim0428/Multi-task-Linear-Regression  
**Area**: Statistical Learning Theory / Multi-task Learning / Robust Regression  
**Keywords**: Multi-task linear regression, matrix-weighted regularization, minimum eigenvalue, balance, outlier tasks, safety guarantees

## TL;DR
This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as a regularization term. It replaces the rigid "minimum eigenvalue of the second moment $\Omega(1)$ for every task" assumption in prior work with a relative "balance constant" $B$. This provides minimax, adaptive, and fallback safety guarantees to Independent Task Learning (ITL) in high-dimensional scenarios with ill-conditioned, low-rank, or outlier tasks.

## Background & Motivation

**Background**: The robust linear regression framework for "a few outlier tasks + many related tasks" in multi-task learning is represented by ARMUL (Duan & Wang 2023). It jointly estimates parameters for $m$ tasks by sharing a central parameter $\beta$ and an $\ell_2$ distance regularization $\lambda\|\theta_j-\beta\|_2$, adapting to unknown outlier ratios $\varepsilon$ and similarity radii $\delta$.

**Limitations of Prior Work**: All theoretical guarantees in this line of work (Duan & Wang 2023, Tian et al. 2025/2026) depend on Lower Boundedness of Second Moments (LBSM), requiring the empirical second moment of each task to satisfy $\rho \mathbf{I}_d \preceq \bm\Sigma_j \preceq L\mathbf{I}_d$ with $\rho=\Omega(1)$. The upper bound $\rho^{-2}\cdot(d/(mn)+\min(L^4\delta^2/\rho^2,d/n)+\varepsilon^2 d/n)$ becomes vacuous in realistic scenarios where $\rho$ is very small (e.g., uniform distributions on high-dimensional spheres $\rho\asymp 1/d$, rapidly decaying spectral features, or adaptive sampling in linear bandits).

**Key Challenge**: LBSM is necessary to ensure identifiability under the "Euclidean parameter error" metric. However, for prediction MSE, directions with the weakest observations should not be heavily penalized—the prediction rate $\tilde{\mathcal O}(d/n)$ for single-task least squares does not require a bounded condition number. In other words, **LBSM is a requirement of the parameter error perspective, not the prediction error perspective**.

**Goal**: Decomposition into three sub-problems: (i) Can multi-task transfer gains be proven without $\rho=\Omega(1)$? (ii) Can a safety rate that automatically falls back to ITL be guaranteed in the absence of similarity structures? (iii) What should the "minimum condition" replacing LBSM look like?

**Key Insight**: The authors observe that $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt{n_j}$ measures inconsistency in the **prediction space** rather than the parameter space. By placing regularization in "whitened coordinates" $\bm\Sigma_j^{1/2}\theta$, unobserved directions are naturally not penalized. The remaining problem is "how to maintain information sharing for multi-task aggregation after whitening"—this requires that the second moments of various tasks be comparable in some average sense.

**Core Idea**: Replace $\ell_2$ regularization with the matrix-weighted norm $\|\theta_j-\beta\|_{\bm\Sigma_j}$ and introduce a **one-sided, average-type** "balance" assumption $\bm\Sigma_j\preceq B\cdot\bm\Sigma_{\mathbf S}$ (where the second moment of each task is controlled by the average second moment of inlier tasks), letting $B$ replace the role of $\rho^{-1}$.

## Method

### Overall Architecture
Given $m$ tasks, each with $n$ samples $(x_{ji},y_{ji})$ following the linear model $y_{ji}=x_{ji}^\top\theta_j^\star+\varepsilon_{ji}$, a fraction $|\mathbf S|/m\ge 1-\varepsilon$ of the unknown inlier parameters fall within an $\ell_2$ ball of radius $\delta$ centered at $\theta^\star$, while the rest are arbitrary outliers; $\varepsilon, \delta$, and $\mathbf S$ are all unknown. The estimator, MTLR, formulates the loss for all tasks plus the center $\beta$ as a joint convex optimization, where the regularization term is weighted by each task's empirical second moment $\bm\Sigma_j=\mathbf X_j^\top\mathbf X_j/n$. The evaluation focuses on the predictive MSE $\mathcal E^{\mathrm{in}}_j=\|\hat\theta_j-\theta_j^\star\|_{\bm\Sigma_j}^2$.

### Key Designs

1. **Joint Convex Loss with Matrix-Weighted Regularization**:
    - **Function**: Replaces the $\ell_2$ regularization of ARMUL with a "locally whitened" regularization for each task, removing the dependency on minimum eigenvalues from the analysis.
    - **Mechanism**: The loss is $\mathcal L(\Theta)=\sum_{j=1}^{m} w_j\big(f_j(\theta_j)+\lambda_j\|\theta_j-\beta\|_{\bm\Sigma_j}\big)$, where $f_j(\theta)=\|\mathbf Y_j-\mathbf X_j\theta\|_2^2/(2n_j)$ and $\|\theta_j-\beta\|_{\bm\Sigma_j}=\|\mathbf X_j(\theta_j-\beta)\|_2/\sqrt n$. This measures the "prediction difference between $\theta_j$ and $\beta$ under task $j$'s own design matrix"—if a direction is barely observed by $\mathbf X_j$, the deviation in that direction is not penalized. This is equivalently an $\ell_2$ regularization after whitening by $\bm\Sigma_j^{1/2}$. With $\lambda_j\asymp\sqrt{d/n_j}$ and reparameterization $v_j=\theta_j-\beta$, the objective is jointly convex and solved via L-BFGS-B.
    - **Design Motivation**: To share information among good tasks while allowing weak observation directions to "auto-mute." Euclidean regularization treats strong and weak directions equally, forcing directions that independent tasks cannot identify toward $\beta$, which introduces $\rho^{-2}$ factors into the guarantees. Matrix-weighted regularization only aggregates in directions actually observed, fundamentally breaking the $\rho$ dependence.

2. **Balance Constant $B$ Replacing LBSM**:
    - **Function**: Uses a "relative rather than absolute" spectral condition to characterize "inter-task geometric compatibility," serving as the weakest condition for transfer gains.
    - **Mechanism**: Assumption 1 requires the existence of $B\in[1,\infty]$ such that $\bm\Sigma_j\preceq B\cdot \bm\Sigma_{\mathbf S}$ holds for all $j$, where $\bm\Sigma_{\mathbf S}=|\mathbf S|^{-1}\sum_{j\in\mathbf S}\bm\Sigma_j$ is the average of inliers. This is a one-sided upper bound compared against the "average" rather than "any pair," specifically allowing each $\bm\Sigma_j$ to be rank-deficient or exhibit spectral decay. LBSM is a special case: if $\rho\mathbf I\preceq\bm\Sigma_j\preceq L\mathbf I$, then $B=L/\rho$. Using covariate concentration $\nu_j$, the empirical $B$ can be transitioned to the population version $\bar B$.
    - **Design Motivation**: LBSM is an "absolute spectral lower bound" often violated in high dimensions or adaptive sampling. Relative "inter-task geometric compatibility" is the essence of multi-task transferability. If the second moments of all inlier tasks fall in similar directions, the average moment $\bm\Sigma_{\mathbf S}$ acts as a good "common skeleton," and control by it implies shareable information. If $B=\infty$ (e.g., information directions of inlier tasks do not overlap), coordination is unnecessary, and the algorithm should revert to ITL.

3. **Two-tiered "Safety + Adaptivity" Rate Structure**:
    - **Function**: Allows the estimator to enjoy multi-task gains when tasks are related and geometrically compatible, while automatically falling back to ITL rates without manual switching when conditions are unfavorable.
    - **Mechanism**: Theorem 2 provides two bounds holding with the same probability. **Safety**: For any $B, \varepsilon, \delta$, $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim q^2(d/n)\zeta$, matching the minimax rate of independent tasks, where $\zeta=\log(16m/\kappa)$. **Adaptivity**: If Assumption 1 holds and $B\lesssim\min(1/\varepsilon,m)$, then for inliers $j\in\mathbf S$, $\mathcal E^{\mathrm{in}}_j(\hat\theta_j)\lesssim (Bd/(mn)+\min(B\delta^2,q^2d/n)+q^2B^2\varepsilon^2 d/n)\zeta$. The average MSE reaches minimax optimality in good regimes. Theorem 3 extends sample bounds to population MSE, and Theorem 4 extends the conclusions to GLMs (given bounded link function curvature).
    - **Design Motivation**: In deployment, $B, \varepsilon, \delta$ are unknown. A theoretical guarantee that requires the user to select the correct hyperparameters/algorithm beforehand has limited practical value. The safety rate ensures "worst-case parity," while the adaptivity rate allows "gains when favorable," neither requiring prior information.

### Loss & Training
Square loss is used for linear models, and negative log-likelihood $f_j(\theta)=\frac{1}{n}\sum_i(\psi(x_{ji}^\top\theta)-y_{ji}x_{ji}^\top\theta)$ for GLMs; $\lambda_j$ is set to $q\sqrt{d\zeta/n}$, with $q$ tuned via 5-fold CV in $\{0.05, 0.10, \dots, 0.50\}$. For GLMs, parameters must be constrained within $\mathbf B(0,\xi)$ to ensure the link function curvature remains in $[\alpha_\ell, \alpha_u]$.

## Key Experimental Results

### Main Results
Synthetic benchmark with $n=100, m=30, d=30$, covariance shape determined by unit ball $\mathbf x$ with $k^{-\alpha}$ coordinate scaling, 30 Monte Carlo trials. Comparison targets include DP (pooling), ITL (independent), and ARMUL (Duan-Wang 2023). The table below shows total task MSE under correlation scanning ($B\equiv 1$, $\varepsilon=0.1$, $\alpha=1$).

| $\delta$ | Ours | ARMUL | DP | ITL |
|---|---|---|---|---|
| 0.2 | **0.0138** | 0.041+ | >0.04 | >0.04 |
| 0.8 | **~0.020** | 0.041+ | >0.04 | >0.04 |
| 3.2 | **0.0259** | 0.041+ | >0.04 | >0.04 |

HAR real data (multi-task logistic regression for 30 subjects, binary classification of standing vs. others, $d=561$, sub-task $n\approx 343$, no PCA), 20% test set, 30 random splits.

| Method | Avg Error Rate (%) | SD |
|------|--------------|-----|
| **Ours** | **1.25** | 0.32 |
| ITL | 4.67 | 0.51 |
| DP | 7.61 | 0.46 |
| ARMUL | (Not fully reported in Table 1) | — |

### Ablation Study
The paper uses four sets of univariate scans in place of traditional ablation, varying one factor to correspond with theoretical variables.

| Scan Variable | Range | Key Findings |
|---------|---------|---------|
| Similarity $\delta$ | 0.2-3.2 | Ours consistently outperforms ITL/ARMUL, with largest gains at small $\delta$. |
| Outlier ratio $\varepsilon$ | 0.05-0.4 | Total task MSE increases smoothly from 0.006 to 0.046; ITL performs best on outliers. |
| Spectral decay $\alpha$ | 0-2.0 | Ours dominates even at $\alpha=1.5, 2.0$ (highly ill-conditioned), validating that LBSM is no longer necessary. |
| Balance $\bar B$ | 5, 10, 15, 20 | Ours is optimal at $\bar B=5$; as $\bar B$ increases, ITL catches up, and Ours automatically aligns with ITL without catastrophic negative transfer. |

### Key Findings
- Under strong ill-conditioning ($\alpha=2$), ARMUL's bound becomes vacuous due to $\rho\asymp d^{-2}$, while the proposed method remains robust—an empirical realization of "eliminating the $\rho^{-2}$ factor."
- Balance scans show a clear "dual-regime switch": at small $\bar B$, Ours benefits from multi-task gains; at large $\bar B$, Ours nearly overlaps with ITL, numerically confirming the safety portion of Theorem 2.
- Logistic regression on HAR without PCA removes ARMUL's signature preprocessing. The proposed method's error rate of 1.25% is nearly 4x lower than ITL's 4.67%, demonstrating the efficacy of matrix-weighted regularization in real high-dimensional low-SNR scenarios.

## Highlights & Insights
- Changing "parameter space regularization" to "prediction space regularization" is a minimal modification—essentially one line in the code—but theoretically solves the $\rho$ factor that plagued this field for years.
- The "balance constant $B$" is strikingly similar to covariate shift coverage conditions in transfer learning. This aligns robust MTL analysis with covariate shift tools.
- The **single-estimator dual-rate** structure of "Safety + Adaptivity" (not requiring prior knowledge of when to transfer) is a highly reusable paradigm for personalized recommendation, federated learning, and other statistical estimation tasks with "individual heterogeneity + group shareability."

## Limitations & Future Work
- Theoretical results still assume an upper bound $\|\bm\Sigma_j\|_{\mathrm{op}}\le 1$, which may require extra conditions under heavy-tailed or adversarial designs.
- Experimental scale is relatively small ($d\le 561$); the "balance" values and estimation stability in over-parameterized deep learning scenarios are not yet fully tested.
- Diagnostic estimation of $B$ via $B_{\mathrm{emp}}$ depends on pseudoinverses and generalized square roots, which can be numerically unstable for small $m$.
- Curvature constraints for GLM link functions ($\alpha_\ell\le\psi''\le\alpha_u$) mean non-smooth cases like softmax or hinge loss still require separate analysis.

## Related Work & Insights
- **vs Duan & Wang (2023, ARMUL)**: Uses the same $\ell_2$-closeness + outlier model, but their $\ell_2$ regularization leads to a $\rho^{-2}$ dependency. This work uses matrix-weighted regularization to eliminate $\rho$ and explicitly includes safety rates.
- **vs Tian et al. (2025/2026, shared low-rank representation)**: They pursue "shared low-rank subspaces" which require different identifiability conditions. This work uses "$\ell_2$-closeness," providing complementary perspectives.
- **vs Bhattacharya et al. (2025, semi-parametric multi-task inference)**: An extension of the ARMUL framework that still relies on LBSM; the matrix-weighting technique here could potentially remove their spectral lower bound assumptions.
- **vs Soare et al. (2014) / Wang et al. (2021)**: Traditional $\ell_2$-closeness without outliers. This work covers those as special cases ($\varepsilon=0$) while ensuring safety under ill-conditioned designs.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/learning_theory/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/learning_theory/the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](../../NeurIPS2025/learning_theory/sample-adaptivity_tradeoff_in_on-demand_sampling.md)

</div>

<!-- RELATED:END -->
