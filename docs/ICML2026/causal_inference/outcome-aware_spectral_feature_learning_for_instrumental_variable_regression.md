---
title: >-
  [Paper Note] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression
description: >-
  [ICML 2026][Causal Inference][Instrumental variable regression] Addressing the "blind spot" in non-parametric instrumental variable (NPIV) regression where SpecIV learns spectral features that only consider the $X-Z$ rel…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Instrumental variable regression"
  - "spectral feature learning"
  - "augmented operator"
  - "contrastive loss"
  - "causal effect estimation"
date: 2026-05-08
content_hash: 5809040da2433c46
---

# Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression

**Conference**: ICML 2026  
**arXiv**: [2512.00919](https://arxiv.org/abs/2512.00919)  
**Code**: None  
**Area**: Causal Inference / Instrumental Variable Regression / Non-parametric IV / Spectral Methods  
**Keywords**: Instrumental variable regression, spectral feature learning, augmented operator, contrastive loss, causal effect estimation

## TL;DR
Addressing the "blind spot" in non-parametric instrumental variable (NPIV) regression where SpecIV learns spectral features that only consider the $X-Z$ relationship without looking at the outcome $Y$, this paper proposes Augmented Spectral Feature Learning. By adding a regression loss—projecting $Y$ onto $Z$ features—to the contrastive loss of SpecIV, the method is equivalent to performing truncated SVD on an "augmented operator" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$ that incorporates $Y$ information. This enables the recovery of causal effects with low-rank features even in "bad" cases where the structural function $h_0$ is poorly aligned with the top singular functions of $\mathcal{T}$.

## Background & Motivation

**Background**: NPIV is a core tool for estimating causal effects in the presence of latent variables, formulated as $Y=h_0(X)+U, \mathbb{E}[U|Z]=0$. This is equivalent to solving the linear inverse problem $\mathcal{T}h_0=r_0$, where $\mathcal{T}$ is the conditional expectation operator from $L_2(X)$ to $L_2(Z)$. Two-Stage Least Squares (2SLS) is a classic approach; recent research has replaced fixed bases with neural network features $\varphi(X)$ and $\psi(Z)$. A representative method, SpecIV (Sun et al., 2025), minimizes a contrastive loss to make learned features approximate the top $d$-dimensional singular subspace of $\mathcal{T}$, essentially performing a low-rank approximation $\mathcal{T}_d$.

**Limitations of Prior Work**: Meunier et al. (2025) proved that SpecIV is optimal only when $h_0$ primarily expands onto the first $d$ right singular functions $v_1, \ldots, v_d$ of $\mathcal{T}$. Once $h_0$ primarily falls on a $v_k$ corresponding to a small singular value $\lambda_k$ (spectral misalignment), SpecIV either fails or requires the rank $d$ to be increased beyond $k$ (introducing $k-1$ uninformative dimensions).

**Key Challenge**: The objective for learning features in SpecIV **completely ignores Y**. Consequently, it can only select "directions with the strongest relationship between $X$ and $Z$," rather than "directions most useful for predicting $Y$." When these coincide, there is no issue; when they do not (e.g., $h_0$ lies in the tail components of $\mathcal{T}$), the algorithm wastes capacity on $Y$-irrelevant directions.

**Goal**: To make spectral feature learning "see" $Y$—retaining the property that features explain $\mathcal{T}$ while additionally biasing them toward directions with predictive power for $Y$. This allows for the recovery of $h_0$ using a smaller feature dimension $d$ under spectral misalignment.

**Key Insight**: Formalize "predictive power for $Y$" as minimizing the MSE of linearly predicting $Y$ using $\psi(Z)$, and incorporate this term into the SpecIV contrastive loss. The authors discovered that the resulting optimization objective is exactly equivalent to performing truncated SVD on an "augmented operator" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$, which treats $r_0=\mathbb{E}[Y|Z]$ as an additional column. This provides a clean operator perspective and a starting point for theoretical analysis.

**Core Idea**: Replace the bare operator with $\mathcal{T} \to \mathcal{T}_\delta = [\mathcal{T}\mid\delta r_0]$ to ensure the spectral decomposition naturally pulls $r_0$ (and thus the "signal direction" of $h_0$) into the principal singular subspace. A hyperparameter $\delta$ controls the strength of the bias toward $Y$, with $\delta=0$ reducing to SpecIV.

## Method

### Overall Architecture

Let $\mathcal{T}:L_2(X)\to L_2(Z)$ be the conditional expectation operator of $X|Z$ with singular value decomposition $\mathcal{T}=\sum_i \lambda_i u_i\otimes v_i$. The target structural function $h_0$ satisfies $\mathcal{T}h_0 = r_0$, where $r_0=\mathbb{E}[Y|Z]$. The method consists of two stages:

1.  **Spectral Feature Learning Stage** (on dataset $\tilde{\mathcal{D}}_m$): Learn a pair of neural network features $\varphi_\theta:\mathcal{X}\to\mathbb{R}^d$ and $\psi_\theta:\mathcal{Z}\to\mathbb{R}^d$, along with an auxiliary vector $\omega\in\mathbb{R}^d$, by minimizing the augmented contrastive loss $\mathcal{L}_\delta^{(d)}(\theta,\omega)$.
2.  **2SLS Estimation Stage** (on dataset $\mathcal{D}_n$): Plug the learned $\varphi_{\hat\theta}, \psi_{\hat\theta}$ directly into the closed-form 2SLS estimator:
    
    $\widehat{h}_\theta(x)=\varphi_\theta(x)^\top \widehat{C}_{\psi\varphi}^{-1}\widehat{\mathbb{E}}_n[Y\psi_\theta(Z)]$,
    
    where $\widehat{C}_{\psi\varphi}$ is the sample mean of $\psi(Z)\varphi(X)^\top$.

Compared to SpecIV, this method only modifies the objective function in the feature learning step; the downstream 2SLS remains identical.

### Key Designs

1.  **Augmented Operator $\mathcal{T}_\delta$ and Additional Regularization $\mathcal{R}_\delta$ (Core)**:
    - **Function**: Explicitly injects $Y$ information into the spectral feature learning objective, causing the "feature-spanning subspace" to automatically bias toward the actual direction of $h_0$.
    - **Mechanism**: Define $\mathcal{T}_\delta:L_2(X)\times\mathbb{R}\to L_2(Z)$ as $\mathcal{T}_\delta(h,a)=\mathcal{T}h+a\cdot\delta\cdot r_0$, which is equivalent to appending $\delta r_0$ as a column to $\mathcal{T}$. The feature learning loss is modified from the SpecIV contrastive loss $\mathcal{L}_0^{(d)}$ to $\mathcal{L}_\delta^{(d)}=\mathcal{L}_0^{(d)}+\mathcal{R}_\delta^{(d)}$, where the regularization term $\mathcal{R}_\delta^{(d)}(\theta) = -\delta^2 \mathbb{E}[Y\psi_\theta(Z)]^\top C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$. Proposition 4.1 proves that $\mathcal{L}_\delta^{(d)}\ge -\|\mathcal{T}_\delta^{(d)}\|_{HS}^2$, with the lower bound reached if and only if the learned operator equals the best rank-$d$ truncation $\mathcal{T}_\delta^{(d)}$ of $\mathcal{T}_\delta$. Thus, minimizing this loss is equivalent to learning the truncated SVD of $\mathcal{T}_\delta$.
    - **Design Motivation**: Intuitively, $-\delta^{-2}\mathcal{R}_\delta^{(d)}$ is the MSE of predicting $Y$ using a linear regression on $\psi_\theta(Z)$ (up to an irrelevant constant). This term forces $\psi$ features to span directions that "explain $Y$." The operator perspective reveals that appending $\delta r_0$ amplifies components of $h_0$ that were originally suppressed by small singular values in $\mathcal{T}$, fundamentally resolving spectral misalignment.

2.  **Differentiable Joint Optimization**:
    - **Function**: Avoids backpropagating through $C_{\psi_\theta}^{-1}$ during network training, which is numerically unstable.
    - **Mechanism**: Make the "inner minimization" in $\mathcal{R}_\delta$ explicit by introducing an auxiliary vector $\omega \in \mathbb{R}^d$ and jointly minimizing $\mathcal{L}_\delta^{(d)}(\theta,\omega) = \mathcal{L}_0^{(d)}(\theta) - 2\delta\mathbb{E}[Y\psi_\theta(Z)]^\top \omega + \omega^\top C_{\psi_\theta}\omega$. For any fixed $\theta$, this is a convex quadratic in $\omega$, with the optimal $\omega_\theta^* = \delta C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$.
    - **Design Motivation**: Backpropagation only requires standard matrix multiplication of $C_{\psi_\theta}$ without inversion. Furthermore, $\omega$ theoretically corresponds to the coordinates of the "extra column" in the SVD of $\mathcal{T}_\delta$ (see $\omega_{*,i}$ in Eq. 6). The trained $\hat\omega$ can also be reused for selecting $\delta$ (Proposition 6.1).

3.  **Two Heuristics for Selecting $\delta$**:
    - **Function**: Allows for selecting an appropriate $\delta$ even without a validation set in unsupervised feature learning.
    - **Mechanism**: ① **Alignment Estimation**: Proposition 6.1 provides the projection length of $h_0$ onto the learned $\{\varphi_{\star,i}\}$ space as $\|\Pi_{\varphi_\star}h_0\|^2 = \alpha^\top(I_d-\omega_\star\omega_\star^\top)^{-1}\alpha$, where $\alpha_i=\mathbb{E}[Y\psi_{\star,i}(Z)]\sigma_{\star,i}^{-1}$. This can be estimated using learned $\hat\sigma, \hat\psi, \hat\omega$. The heuristic is to gradually increase $\delta$ as long as the estimated alignment rises significantly. ② **Loss Balancing**: Treat $\mathcal{R}_\delta$ as a regularizer for $\mathcal{L}_0$. Increase $\delta$ until $\mathcal{R}_\delta$ drops sharply while $\mathcal{L}_0$ rises significantly—excessive $\delta$ causes features to overfit the $r_0$ direction, losing the overall approximation of $\mathcal{T}$.
    - **Design Motivation**: Small positive $\delta$ almost always helps, but excessively large $\delta$ causes the norm of $\omega_*$ to diverge and $(I-\omega\omega^\top)^{-1}$ to become unstable. These heuristics provide observable stopping signals.

### Loss & Training

Final training objective (empirical form using $\tilde{\mathcal{D}}_m$):

$\widehat{\mathcal{L}}_\delta^{(d)}(\theta,\omega) = \widehat{\mathbb{E}}_X\widehat{\mathbb{E}}_Z[(\varphi_\theta(X)^\top\psi_\theta(Z))^2] - 2\widehat{\mathbb{E}}[\varphi_\theta(X)^\top\psi_\theta(Z)] - 2\delta \widehat{\mathbb{E}}[Y\psi_\theta(Z)]^\top\omega + \omega^\top \widehat{C}_{\psi_\theta}\omega$.

The first term uses in-batch marginal sampling (re-pairing X and Z within a batch) to approximate the expectation over the product of marginal distributions. Adam is used to jointly update $(\theta, \omega)$. After feature learning, features are frozen for closed-form 2SLS on the independent dataset $\mathcal{D}_n$. This sample splitting ensures that estimation errors are decoupled from feature learning errors, corresponding to the $\sqrt{d/n}$ rate in Theorem 5.1.

## Key Experimental Results

### Main Results

**Synthetic Data**: A controlled operator $\mathcal{T}=\mathbf{1}_Z\otimes\mathbf{1}_X+\sum_{i=1}^{d-1}\sigma_i u_i\otimes v_i$ is constructed, where $\sigma_i$ decays linearly to $\sigma_{d-1}=c_\sigma\sigma_1$. The structural function $h_0=\sum\alpha_i v_i$ is controlled by $c_\alpha=\alpha_{d-1}/\alpha_1$ to determine the degree of spectral misalignment. The table reports normalized IV loss $\|\widehat{h}_\theta-h_0\|^2$ (relative to the mean at $\delta=0$).

| Scenario ($c_\sigma$, $c_\alpha$) | $\delta=0$ (SpecIV) | $\delta=0.5$ | $\delta=1.0$ | $\delta=3.0$ | $\delta=5.0$ |
|---|---|---|---|---|---|
| $c_\sigma=0.2, c_\alpha=5.0$ (Severe misalignment) | 1.00 | Significant drop | Further drop | Optimal zone | Slight rebound |
| $c_\sigma=0.8, c_\alpha=0.2$ (High alignment) | 1.00 | Still improved | Improved | Stable | Slightly worse |

Qualitative Conclusion: In difficult scenarios where misalignment is severe and singular values decay rapidly, the augmented method reaches an IV loss multiple times lower than SpecIV. In well-aligned scenarios, a small $\delta > 0$ still yields consistent improvements.

**dSprites IV Benchmark**: In addition to the original $h_{\text{old}}$ (heart-shaped sprite, where $h_0$ aligns with top singular functions), the authors propose $h_{\text{new}}$ (ellipsoid orientation, requiring features from small singular value directions). Comparison with DFIV and KIV:

| Setting | SpecIV ($\delta=0$) | AugSpecIV (Small $\delta$) | DFIV |
|---|---|---|---|
| $h_{\text{old}}$ ("Good") | Baseline | **~20% Improvement** over SpecIV | Strong baseline |
| $h_{\text{new}}$ ("Bad") | **Seriously trails** DFIV | Matches or exceeds DFIV | Strongest baseline |

Key Finding: The failure of bare SpecIV on $h_{\text{new}}$ validates the theory—the top singular subspace is not the optimal basis for $h_0$. Increasing $\delta$ significantly increases the projection of features onto $h_{\text{new}}$, drastically reducing loss.

**Off-Policy Evaluation (OPE)**: In BSuite Cartpole, Mountain Car, and Catch, OPE is cast as an iterative NPIV problem. Results show AugSpecIV significantly outperforms SpecIV and DFIV in Cartpole. In Catch, it is as strong as SpecIV; in Mountain Car, it trails DFIV. No single method is consistently optimal across all tasks, but the automatically selected $\delta$ values ($1, 10^{-3}, 10^{-2}$) demonstrate the necessity of being "outcome-aware."

### Ablation Study

| Configuration | Performance | Description |
|---|---|---|
| Full ($\mathcal{L}_0+\mathcal{R}_ \delta$, joint $(\theta,\omega)$) | Optimal | Complete method |
| $\delta=0$ (No $\mathcal{R}_\delta$) | Degenerates to SpecIV | Collapses under misalignment |
| $\delta\to\infty$ zone ($\mathcal{R}_\delta$ dominant) | IV loss rebounds | Features only learn the $r_0$ direction |
| $\mathcal{R}_\delta$ only (No $\mathcal{L}_0$) | Suboptimal | Cannot guarantee the overall structure of $\mathcal{T}$ |

### Key Findings

- **Small positive $\delta$ is nearly a "free lunch"**: Both synthetic and dSprites data show that $\delta > 0$ almost always improves performance and is insensitive to the specific value, reducing the hyperparameter tuning burden.
- **$\delta$ is not "the larger the better"**: When $\mathcal{R}_\delta$ dominates $\mathcal{L}_0$, features "escape" to only explain $r_0$, losing the ability to approximate $\mathcal{T}$ and causing IV loss to rise.
- **OPE $\delta$ values vary by 3 orders of magnitude**: This proves that the degree of dynamic misalignment varies greatly by task, strongly supporting the necessity of "outcome-aware" learning.

## Highlights & Insights

- **Equivalence between regularization and operator augmentation**: The regularization $\mathcal{R}_\delta$ might seem like a simple heuristic, but its exact equivalence to the truncated SVD of $[\mathcal{T}\mid\delta r_0]$ allows AugSpecIV to inherit the operator theory framework (Wedin sin-Θ, Weyl inequalities). This yields high-probability error bounds rather than just empirical observations.
- **Dual-use of auxiliary variable $\omega$**: It serves as both an optimization trick to avoid matrix inversion and as a physical coordinate in the augmented SVD space for model selection ($\delta$ tuning).
- **New dSprites benchmark $h_{\text{new}}$**: The original benchmark was found to be a "good" scenario, masking SpecIV's weaknesses. By constructing $h_{\text{new}}$ with ellipsoid orientations assigned to tail singular directions, the authors exposed the limitations of bare SpecIV.
- **Transferable Paradigm**: This "augmentation" approach can be applied to any scenario using spectral methods for conditional operator decomposition (e.g., value function estimation in OPE, transition operators in molecular dynamics).

## Limitations & Future Work

- Current theory only handles **Rank-1 augmentation** ($\delta r_0$). Generalizing to Rank-$K$ augmentation (e.g., adding higher-order moments) requires more complex perturbation analysis.
- **Downstream 2SLS error bounds depend on "successful representation learning"**: The theory assumes the DNN can optimize the augmented loss near its minimum, but proving convergence for DNN training on this objective remains an open problem.
- **Heuristic $\delta$ selection**: The heuristics depend on observing curve trends rather than a closed-form cross-validation approach.
- **No method is consistently optimal in OPE**: AugSpecIV still trails DFIV in Mountain Car, suggesting spectral misalignment is not the only challenge in OPE.

## Related Work & Insights

- **vs SpecIV (Sun et al., 2025)**: This paper is a direct superset of SpecIV. While SpecIV only considers $X,Z$ relationships, this work proves it fails under misalignment and fixes it at a low cost.
- **vs DFIV (Xu et al., 2020)**: DFIV learns the conditional expectation $\mathbb{E}[\varphi(X)|Z]$ directly without explicit low-rank decomposition. AugSpecIV matches DFIV on $h_{\text{new}}$, proving that a "spectral + outcome-aware" path can catch up to black-box end-to-end methods while offering more theoretical control.
- **vs Bruns-Smith (2024)**: That work learns $Z$-features predictive of $Y$ first. In contrast, AugSpecIV binds "useful for $Y$" and "useful for $\mathcal{T}$" into a single operator objective, requiring weaker theoretical conditions.
- **Insight**: Appending task-relevant signals as extra columns for spectral decomposition is a highly transferable paradigm for "operator learning + downstream tasks."

## Rating
- Novelty: ⭐⭐⭐ Handing the "Y-MSE regularization" and "operator augmentation + truncated SVD" as an exact equivalence is a clean and new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering synthetic, dSprites, and OPE tasks with failure-repair contrasts is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from motivation to operator theory to experiments is excellent.
- Value: ⭐⭐⭐⭐ Directly fixes a known blind spot in SpecIV with minimal implementation overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](../../NeurIPS2025/causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ICML 2026\] ECSEL: Explainable Classification via Signomial Equation Learning](ecsel_explainable_classification_via_signomial_equation_learning.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Rank-Learner: Orthogonal Ranking of Treatment Effects](rank-learner_orthogonal_ranking_of_treatment_effects.md)

</div>

<!-- RELATED:END -->
