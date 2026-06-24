---
title: >-
  [Paper Note] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression
description: >-
  [ICML 2026][Causal Inference][Instrumental Variable Regression] Addressing the blind spot in Nonparametric Instrumental Variable (NPIV) regression where SpecIV learns spectral features focusing solely on the $X-Z$ relationship while ignoring the outcome $Y$, this paper proposes Augmented Spectral Feature Learning. By adding a regression loss of $Y$ projected onto $Z$ features to the contrastive loss of SpecIV, the method is equivalent to performing a truncated SVD on an "augm…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Instrumental Variable Regression"
  - "Spectral Feature Learning"
  - "Augmented Operator"
  - "Contrastive Loss"
  - "Causal Effect Estimation"
date: 2026-05-08
content_hash: 782616aa7091d8fc
---

# Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression

**Conference**: ICML 2026  
**arXiv**: [2512.00919](https://arxiv.org/abs/2512.00919)  
**Code**: None  
**Area**: Causal Inference / Instrumental Variable Regression / Nonparametric IV / Spectral Methods  
**Keywords**: Instrumental Variable Regression, Spectral Feature Learning, Augmented Operator, Contrastive Loss, Causal Effect Estimation

## TL;DR
Addressing the blind spot in Nonparametric Instrumental Variable (NPIV) regression where SpecIV learns spectral features focusing solely on the $X-Z$ relationship while ignoring the outcome $Y$, this paper proposes Augmented Spectral Feature Learning. By adding a regression loss of $Y$ projected onto $Z$ features to the contrastive loss of SpecIV, the method is equivalent to performing a truncated SVD on an "augmented operator" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$ that incorporates $Y$ information. This allows for causal effect recovery using extremely low-rank features even in "bad" cases where the structural function $h_0$ is poorly aligned with the top singular functions of $\mathcal{T}$.

## Background & Motivation

**Background**: NPIV is a core tool for causal effect estimation in the presence of latent confounders, formulated as $Y=h_0(X)+U, \mathbb{E}[U|Z]=0$, which is equivalent to solving the linear inverse problem $\mathcal{T}h_0=r_0$, where $\mathcal{T}$ is the conditional expectation operator from $L_2(X)$ to $L_2(Z)$. While Two-Stage Least Squares (2SLS) is the classical approach, recent studies have replaced features $\varphi(X)$ and $\psi(Z)$ with adaptively learned neural networks. Representative methods include DFIV, SpecIV, and minimax saddle-point methods. SpecIV (Sun et al., 2025) approximates the top $d$-dimensional singular subspace of $\mathcal{T}$ via contrastive loss minimization, effectively performing a low-rank approximation $\mathcal{T}_d$.

**Limitations of Prior Work**: Meunier et al. (2025) demonstrated that SpecIV is optimal only when $h_0$ is primarily expanded on the first $d$ right singular functions $v_1, \ldots, v_d$ of $\mathcal{T}$. If $h_0$ lies on a $v_k$ corresponding to a small singular value $\lambda_k$ (spectral misalignment), SpecIV either requires increasing the rank $d$ beyond $k$ (introducing $k-1$ uninformative dimensions) or fails entirely.

**Key Challenge**: The objective of SpecIV for learning features **completely ignores Y**. Consequently, it only selects directions with the strongest $X-Z$ relationship rather than those most useful for predicting $Y$. While this is acceptable when the two coincide, the algorithm wastes all capacity on $Y$-irrelevant directions when they do not (e.g., when $h_0$ resides in the tail components of $\mathcal{T}$).

**Goal**: Enable spectral feature learning to "see" $Y$—preserving the property that features must explain $\mathcal{T}$ while biasing them toward directions with predictive power for $Y$, thereby recovering $h_0$ with smaller feature dimensions $d$ even under spectral misalignment.

**Key Insight**: The property of "predictive power for $Y$" is formalized as minimizing the MSE of linear prediction of $Y$ using $\psi(Z)$, which is added to the SpecIV contrastive loss. The authors discovered that the entire optimization objective is equivalent to performing truncated SVD on an "augmented operator" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$, where $r_0 = \mathbb{E}[Y|Z]$ is appended as an additional column. This provides a clean operator perspective and a basis for theoretical analysis.

**Core Idea**: Replace the bare operator with $\mathcal{T} \to \mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$, allowing the spectral decomposition to naturally pull $r_0$ (and thus the signal direction of $h_0$) into the principal singular subspace. A hyperparameter $\delta$ controls the intensity of the bias toward $Y$, with $\delta=0$ reducing to SpecIV.

## Method

### Overall Architecture

The method addresses the NPIV inverse problem $\mathcal{T}h_0=r_0$ ($\mathcal{T}:L_2(X)\to L_2(Z)$ is the conditional expectation operator of $X|Z$, $r_0=\mathbb{E}[Y|Z]$). The core issue is that SpecIV ignores $Y$ during feature learning. Thus, a regularization term for $Y$-predictive power is added to the contrastive loss. The workflow remains two-stage: first, learn a pair of neural network features $\varphi_\theta:\mathcal{X}\to\mathbb{R}^d$ and $\psi_\theta:\mathcal{Z}\to\mathbb{R}^d$ (plus an auxiliary vector $\omega\in\mathbb{R}^d$) on dataset $\tilde{\mathcal{D}}_m$ by minimizing the augmented contrastive loss $\mathcal{L}_\delta^{(d)}(\theta,\omega)$. Second, substitute the learned features into a closed-form 2SLS estimator $\widehat{h}_\theta(x)=\varphi_\theta(x)^\top \widehat{C}_{\psi\varphi}^{-1}\widehat{\mathbb{E}}_n[Y\psi_\theta(Z)]$ on an independent dataset $\mathcal{D}_n$. Compared to SpecIV, only the objective function in the feature learning step changes; the downstream 2SLS is reused as is.

### Key Designs

**1. Augmented Operator $\mathcal{T}_\delta$ and Regularization $\mathcal{R}_\delta$: Injecting $Y$ Information into Spectral Decomposition**

SpecIV's feature learning objective lacks $Y$, missing $h_0$ if it falls on small singular directions of $\mathcal{T}$. This paper adds a regularization term $\mathcal{R}_\delta^{(d)}(\theta) = -\delta^2 \mathbb{E}[Y\psi_\theta(Z)]^\top C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$ to the SpecIV contrastive loss $\mathcal{L}_0^{(d)}$, resulting in $\mathcal{L}_\delta^{(d)}=\mathcal{L}_0^{(d)}+\mathcal{R}_\delta^{(d)}$. Intuitively, $-\delta^{-2}\mathcal{R}_\delta^{(d)}$ is exactly the MSE of predicting $Y$ via linear regression on $\psi_\theta(Z)$ (up to an irrelevant constant), forcing $\psi$ features to span directions that explain $Y$.

The elegant part is its operator interpretation. Defining the augmented operator $\mathcal{T}_\delta:L_2(X)\times\mathbb{R}\to L_2(Z)$ as $\mathcal{T}_\delta(h,a)=\mathcal{T}h+a\cdot\delta\cdot r_0$, which is equivalent to appending $\delta r_0$ as an extra column to $\mathcal{T}$, denoted $\mathcal{T}_\delta=[\mathcal{T}\mid\delta r_0]$. Proposition 4.1 proves $\mathcal{L}_\delta^{(d)}\ge -\|\mathcal{T}_\delta^{(d)}\|_{HS}^2$, where the lower bound is reached when the learned operator equals the best rank-$d$ truncation $\mathcal{T}_\delta^{(d)}$. This works because appending $\delta r_0$ amplifies signal components of $h_0$ originally suppressed by small singular values into the top singular subspace, fundamentally resolving spectral misalignment.

**2. Introducing Auxiliary Variable $\omega$ for Differentiable Joint Optimization**

The regularization term $\mathcal{R}_\delta$ contains $C_{\psi_\theta}^{-1}$, and backpropagating through matrix inversion is numerically unstable. To address this, the "inner minimization" implicit in $\mathcal{R}_\delta$ is made explicit by introducing an auxiliary vector $\omega\in\mathbb{R}^d$ and jointly minimizing $\mathcal{L}_\delta^{(d)}(\theta,\omega) = \mathcal{L}_0^{(d)}(\theta) - 2\delta\mathbb{E}[Y\psi_\theta(Z)]^\top \omega + \omega^\top C_{\psi_\theta}\omega$. For a fixed $\theta$, this is convex quadratic in $\omega$, with the optimal $\omega_\theta^* = \delta C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$. Substituting this back recovers the original $\mathcal{L}_\delta^{(d)}$. This avoids matrix inversion in backpropagation. Theoretically, $\omega$ corresponds to the coordinates of the "extra row" in the SVD of $\mathcal{T}_\delta$.

**3. Two Heuristics for Selecting $\delta$ in Unsupervised Feature Learning**

Since feature learning lacks a validation set, $\delta$ cannot be chosen via cross-validation. Two observable signals are used. The first is **Alignment Estimation**: Proposition 6.1 gives the projection length of $h_0$ on the learned subspace $\|\Pi_{\varphi_\star}h_0\|^2 = \alpha^\top(I_d-\omega_\star\omega_\star^\top)^{-1}\alpha$, which can be estimated using learned values. $\delta$ is increased as long as alignment significantly improves. The second is **Loss Balancing**: Treating $\mathcal{R}_\delta$ as regularization for $\mathcal{L}_0$, $\delta$ is increased until $\mathcal{R}_\delta$ drops sharply while $\mathcal{L}_0$ rises significantly—indicating the features are overfitting the direction of $r_0$ and losing the overall approximation of $\mathcal{T}$.

### Loss & Training

The final training objective (empirical form using $\tilde{\mathcal{D}}_m$):

$\widehat{\mathcal{L}}_\delta^{(d)}(\theta,\omega) = \widehat{\mathbb{E}}_X\widehat{\mathbb{E}}_Z[(\varphi_\theta(X)^\top\psi_\theta(Z))^2] - 2\widehat{\mathbb{E}}[\varphi_\theta(X)^\top\psi_\theta(Z)] - 2\delta \widehat{\mathbb{E}}[Y\psi_\theta(Z)]^\top\omega + \omega^\top \widehat{C}_{\psi_\theta}\omega$.

The first term $\widehat{\mathbb{E}}_X\widehat{\mathbb{E}}_Z$ is approximated using in-batch marginal sampling (shuffling $X$ and $Z$ within a batch), following standard SpecIV practice. Adam optimizer is used for joint updates of $(\theta,\omega)$. Features are frozen after training for the closed-form 2SLS on $\mathcal{D}_n$.

## Key Experimental Results

### Main Results

**Synthetic Data**: A controllable operator $\mathcal{T}=\mathbf{1}_Z\otimes\mathbf{1}_X+\sum_{i=1}^{d-1}\sigma_i u_i\otimes v_i$ is constructed where $\sigma_i$ decays linearly. $c_\alpha = \alpha_{d-1}/\alpha_1$ controls the alignment of $h_0$ with the spectrum of $\mathcal{T}$. Normalized IV loss $\|\widehat{h}_\theta-h_0\|^2$ is reported (relative to $\delta=0$).

| Scenario ($c_\sigma$, $c_\alpha$) | $\delta=0$ (SpecIV) | $\delta=0.5$ | $\delta=1.0$ | $\delta=3.0$ | $\delta=5.0$ |
|---|---|---|---|---|---|
| $c_\sigma=0.2,\;c_\alpha=5.0$ (Severe misalignment) | 1.00 | Significant drop | Further drop | Optimal zone | Slight increase |
| $c_\sigma=0.8,\;c_\alpha=0.2$ (High alignment) | 1.00 | Improvement | Improvement | Stable | Slightly worse |

Qualitative Conclusion: In the most difficult scenarios (poor alignment + fast singular value decay), the augmented method reduces IV loss to a fraction of SpecIV's. For well-aligned scenarios, a small $\delta > 0$ still yields consistent improvements.

**dSprites IV Benchmark**: In addition to the original $h_{\text{old}}$ (heart shape, "good" scenario), the authors propose $h_{\text{new}}$ (ellipse shape, where $h_0$ requires features from small singular directions). Comparison with DFIV and KIV:

| Setting | SpecIV ($\delta=0$) | AugSpecIV (Small positive $\delta$) | DFIV |
|---|---|---|---|
| $h_{\text{old}}$ ("Good") | Baseline | **~20% average improvement** | Strong baseline |
| $h_{\text{new}}$ ("Bad") | **Severely underperforms** DFIV | Matches or exceeds DFIV | Strongest baseline |

**Off-Policy Evaluation (OPE)**: In BSuite environments, OPE is reformulated as an iterative NPIV problem. AugSpecIV significantly outperforms SpecIV and DFIV in Cartpole and matches SpecIV in Catch. The automatically selected $\delta$ values vary significantly ($1 \sim 10^{-3}$), showing the method adapts to different misalignment levels.

### Ablation Study

| Configuration | Performance | Description |
|---|---|---|
| Full ($\mathcal{L}_0+\mathcal{R}_\delta$, joint $(\theta,\omega)$) | Optimal | Complete method |
| $\delta=0$ (No $\mathcal{R}_\delta$) | Reduces to SpecIV | Fails under misalignment |
| $\delta\to\infty$ ($\mathcal{R}_\delta$ dominant) | IV loss rebounds | Features overfit to $r_0$ direction |

### Key Findings

- **Small positive $\delta$ is a "free lunch"**: Improvement is consistently observed for $\delta > 0$ across synthetic and dSprites tasks, reducing parameter tuning burden.
- **Larger $\delta$ is not always better**: When $\mathcal{R}_\delta$ dominates $\mathcal{L}_0$, features lose the ability to approximate $\mathcal{T}$, causing IV loss to rise.
- **$\delta$ values in OPE span 3 orders of magnitude**, suggesting that "outcome-awareness" is essential as misalignment degrees vary greatly between tasks.

## Highlights & Insights

- **Equivalence of Regularization and Operator Augmentation**: Adding $\mathcal{R}_ \delta$ is not just an empirical trick; it is mathematically equivalent to truncated SVD on $[\mathcal{T}\mid\delta r_0]$. This allows the method to inherit the theoretical framework of SpecIV (Wedin sin-Θ, Weyl’s inequality).
- **Dual Function of Auxiliary Variable $\omega$**: It serves both as an optimization tactic to avoid matrix inversion and as a signal for model selection by estimating subspace alignment.
- **Benchmark Diagnosis**: By constructing $h_{\text{new}}$, the authors exposed the blind spot of SpecIV that was hidden by the "good" alignment of previous benchmarks.

## Limitations & Future Work

- Currently handles only **Rank-1 augmentation** ($\delta r_0$). Generalizing to Rank-$K$ augmentation for multiple outputs or higher moments is left for future work.
- Downstream 2SLS bounds rely on successful neural network training; proving convergence for DNNs on this objective remains an open problem.
- **$\delta$ selection is still heuristic**: The signals are based on curve trends rather than a closed-form automatic method like cross-validation.
- In some OPE tasks, it still underperforms DFIV, suggesting spectral misalignment is not the only challenge in iterative value estimation.

## Related Work & Insights

- **vs SpecIV (Sun et al., 2025)**: This paper is a direct superset. SpecIV only considers $X-Z$ relations; this work proves it fails under misalignment and fixes it at low cost.
- **vs DFIV (Xu et al., 2020)**: DFIV avoids explicit low-rank decomposition. This work shows that "explicit spectral + Y-aware" paths can match or exceed end-to-end black-box methods while providing better theoretical control.
- **vs Minimax methods**: Minimax methods use a GMM-adversarial framework. This work focuses on the SVD of the conditional expectation operator, which is computationally cheaper (closed-form 2SLS) and easier to tune.

## Rating
- Novelty: ⭐⭐⭐⭐ (Elegant equivalence between regularization and operator augmentation).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers synthetic, dSprites, and OPE).
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous notation and logical progression).
- Value: ⭐⭐⭐⭐ (Directly addresses a known weakness of SpecIV with low implementation cost).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](../../NeurIPS2025/causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ICML 2026\] ECSEL: Explainable Classification via Signomial Equation Learning](ecsel_explainable_classification_via_signomial_equation_learning.md)
- [\[ICML 2026\] Toward Scalable and Valid Conditional Independence Testing with Spectral Representations](toward_scalable_and_valid_conditional_independence_testing_with_spectral_represe.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
