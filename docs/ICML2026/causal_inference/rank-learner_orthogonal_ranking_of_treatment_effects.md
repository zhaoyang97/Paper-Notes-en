---
title: >-
  [Paper Note] Rank-Learner: Orthogonal Ranking of Treatment Effects
description: >-
  [ICML 2026][Causal Inference][Treatment Effect Ranking] Ours proposes Rank-Learner—the first Neyman-orthogonal two-stage treatment effect **ranking** learner for observational data. By replacing the indirect "CATE estima…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Treatment Effect Ranking"
  - "Neyman Orthogonality"
  - "Two-stage Learner"
  - "Doubly Robust"
  - "Pairwise Loss"
date: 2026-05-08
content_hash: 3ad310de4f81154b
---

# Rank-Learner: Orthogonal Ranking of Treatment Effects

**Conference**: ICML 2026  
**arXiv**: [2602.03517](https://arxiv.org/abs/2602.03517)  
**Code**: https://github.com/henriarnoUG/rank-learner (Available)  
**Area**: Causal Inference / Treatment Effect Ranking / Orthogonal Learning  
**Keywords**: Treatment Effect Ranking, Neyman Orthogonality, Two-stage Learner, Doubly Robust, Pairwise Loss

## TL;DR
Ours proposes Rank-Learner—the first Neyman-orthogonal two-stage treatment effect **ranking** learner for observational data. By replacing the indirect "CATE estimation then ranking" approach with pairwise soft labels and doubly robust correction terms, it consistently outperforms T/DR-learners and non-orthogonal plug-in rankers on synthetic, semi-synthetic, and Criteo uplift datasets.

## Background & Motivation

**Background**: In scenarios such as medical triage, targeted marketing, and public policy, decision-makers do not necessarily require precise CATE values; instead, they need to **rank individuals by treatment effects** to allocate limited resources to those who benefit most. However, existing literature focuses almost exclusively on CATE estimation, with very few works dedicated to treatment effect ranking.

**Limitations of Prior Work**: Standard learning-to-rank (LTR) methods (pointwise / pairwise / listwise) require "ranking labels" as supervision, but the counterfactual $Y(1)-Y(0)$ is never observable. Current approaches take two indirect paths: (i) first estimating $\hat\tau(x)$ using T-learner / DR-learner and then ranking by these estimates; (ii) directly applying LTR as seen in the tree ranker by Kamran et al. 2024 or the plug-in ranker by Vanderschueren et al. 2024, both of which are **not Neyman-orthogonal** and are highly sensitive to errors in first-stage nuisance estimation.

**Key Challenge**: CATE estimation solves a **harder problem than ranking**—it requires identifying the complete function $g(x)=\tau(x)$, whereas ranking only requires identifying an order-preserving transformation $g(x)=h(\tau(x))$. Directing the errors of a "hard problem" as supervision for an "easy problem" is sample-inefficient. Furthermore, non-orthogonal objectives allow nuisance bias to propagate first-order to the scorer, which is particularly fatal in small samples.

**Goal**: Construct a two-stage learner that **directly targets ranking and is robust to nuisance errors**, satisfying: (i) model-agnosticism, (ii) Neyman orthogonality, and (iii) a population minimizer that yields the correct ranking.

**Key Insight**: Starting from the pairwise binary cross-entropy loss $\mathcal L^{\text{bin}}$, its population minimizer is any order-preserving transformation $h\circ\tau$, which is exactly what is needed for ranking. However, $\mathcal L^{\text{bin}}$ uses non-smooth indicator labels $\mathbf 1\{\tau(X)>\tau(X')\}$, preventing influence function orthogonalization. By replacing the indicator with a smooth sigmoid soft label $t_\tau$, the ranking semantics are preserved while enabling the path to orthogonalization.

**Core Idea**: Replace hard indicators with sigmoid soft labels $t_\tau(X,X')=\sigma((\tau(X)-\tau(X'))/\kappa)$, then derive a DR-score correction term $\omega_\tau\cdot\Delta_\eta$ via influence functions. This results in the Neyman-orthogonal pseudo-label $\tilde t_\eta = t_\tau + \omega_\tau\cdot\Delta_\eta$, which is then fed into a standard pairwise BCE.

## Method

### Overall Architecture
Setup: Observational data $W=(X,T,Y)$ with $T\in\{0,1\}$. Under standard causal assumptions (consistency, positivity, unconfoundedness), the goal is to learn a scoring function $g:\mathcal X\to\mathbb R$ that shares the same order as $\tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x]$. Rank-Learner follows a classic two-stage process:

```mermaid
graph TD
    A[Observational Data W] --> B[Stage 1: Nuisance Estimation]
    B --> C["Nuisance Parameters: \hat{\mu}_1, \hat{\mu}_0, \hat{e}"]
    C --> D[Stage 2: Orthogonal Ranking]
    D --> E[Generate Pseudo-labels t_tilde via DR Correction]
    E --> F[Minimize Pairwise BCE with Sub-sampling]
    F --> G[Inference: Scalar Scoring g(x)]
```

- **Stage 1 (Nuisance Estimation)**: Use cross-fitting to estimate three nuisance components—the two response surfaces $\mu_t(x)=\mathbb E[Y\mid T=t,X=x]$ and the propensity score $e(x)=P(T=1\mid X=x)$, yielding $\hat\eta=(\hat\mu_1,\hat\mu_0,\hat e)$.
- **Stage 2 (Orthogonal Ranking)**: Randomly sample a set of pairs $\mathcal P$ from the training set, construct pseudo-labels $\tilde t_{\hat\eta}(w_i,w_j)$ according to the derived formula, and minimize the empirical loss $\frac{1}{|\mathcal P|}\sum_{(i,j)}\ell(p_g(x_i,x_j),\tilde t_{\hat\eta}(w_i,w_j))$, where $p_g(x,x')=\sigma(g(x)-g(x'))$ and $\ell$ is the binary cross-entropy.
- **Inference**: Perform point-wise evaluation of $\hat g(x)$ for ranking; **pairwise comparisons are not required during the inference stage**.

The difference from the "plug-in CATE → ranking" framework lies solely in the **objective construction** of the second stage. The loss form remains standard BCE, while orthogonality is encapsulated within the pseudo-label.

### Key Designs

1. **Soft Pairwise Ranking Loss (Downgrading "Hard Problems" to "Easy Problems")**:
    - **Function**: Converts the regression problem of "identifying CATE values" into a pairwise classification problem of "identifying $\tau$ order," aligning with the final task.
    - **Mechanism**: Define soft targets $t_\tau(X,X')=\sigma((\tau(X)-\tau(X'))/\kappa)$ and model preferences $p_g(X,X')=\sigma(g(X)-g(X'))$, then minimize $\mathcal L^{\text{soft}}=\mathbb E_{X,X'}[\ell(p_g,t_\tau)]$. The population minimizer is $g(x)=\tau(x)/\kappa + c$, which recovers the rank. $\kappa>0$ controls smoothness: as $\kappa\to 0$, it approaches the hard indicator loss $\mathcal L^{\text{bin}}$.
    - **Design Motivation**: $\mathcal L^{\text{cate}}$ is uniquely minimized at $g(x)=\tau(x)$, requiring full recovery of the CATE function; $\mathcal L^{\text{soft}}$ imposes only order-preserving constraints, making it easier to solve. Crucially, soft labels are **differentiable with respect to $\tau$**, enabling influence function orthogonalization—whereas hard indicators are not.

2. **Neyman-Orthogonal Doubly Robust Pseudo-labels**:
    - **Function**: Use first-stage $\hat\eta$ to construct $\tilde t_{\hat\eta}$ such that the second-stage objective is insensitive to first-order perturbations in $\hat\eta$, suppressing plug-in bias.
    - **Mechanism**: Correct $\mathcal L^{\text{soft}}$ along the influence function to obtain $\tilde t_\eta(W,W')=t_\tau(X,X')+\omega_\tau(X,X')\cdot\Delta_\eta(W,W')$, where weights $\omega_\tau=\tfrac{1}{\kappa}t_\tau(1-t_\tau)$ and the DR-score difference $\Delta_\eta=(\phi_\eta(W)-\tau(X))-(\phi_\eta(W')-\tau(X'))$. The point-wise DR score is $\phi_\eta(W)=\tfrac{T}{e(X)}(Y-\mu_1(X))-\tfrac{1-T}{1-e(X)}(Y-\mu_0(X))+\mu_1(X)-\mu_0(X)$. Theorem 5.1 proves that $\mathcal L^{\text{orth}}=\mathbb E_{W,W'}[\ell(p_g,\tilde t_\eta)]$ is Neyman-orthogonal to $\eta=(\mu_1,\mu_0,e)$.
    - **Design Motivation**: The weight $\omega_\tau$ acts as a natural "uncertainty gate." When the plug-in soft target is near $0$ or $1$ (rank is clear), the weight is low and $\tilde{t} \approx t_\tau$. When the target is near $0.5$ (rank is ambiguous), the weight increases, activating the DR correction to push "uncertain" pairs toward more credible pseudo-labels.

3. **Cross-fitting + Pairwise Sub-sampling**:
    - **Function**: Industrializes "$n^2$ training pairs" into scalable training while preserving theoretical properties.
    - **Mechanism**: Stage 1 uses cross-fitting to avoid overfitting bias. Stage 2 **uniformly samples sub-sets** $\mathcal P\subset\mathcal P_{\text{all}}$ of pairs in each epoch. Hyperparameter $\kappa$ is selected via an approximation of AUTOC (Chernozhukov et al. 2025).
    - **Design Motivation**: Cross-fitting is standard for orthogonal learning. Sub-sampling reduces complexity from $O(n^2)$ to $O(|\mathcal P|)$. Experiments show that performance saturates at small sampling ratios, indicating that quality depends on nuisances rather than the number of pairs.

### Loss & Training
The final empirical objective is:
$$\mathcal L^{\text{orth}}(g,\hat\eta)=\tfrac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}\ell(p_g(x_i,x_j),\tilde t_{\hat\eta}(w_i,w_j))$$
Model selection for $\kappa$ and other hyperparameters follows the approximate AUTOC. Points-wise forward passes are used for inference.

## Key Experimental Results

### Main Results

Synthetic benchmark (Test AUTOC, 5 seeds, higher is better, oracle = 1.40):

| Method | $n=100$ | $n=500$ | $n=1{,}000$ | $n=2{,}000$ |
|------|---------|---------|---------|---------|
| T-learner | 0.88 ± 0.17 | 1.24 ± 0.05 | 1.32 ± 0.02 | 1.36 ± 0.00 |
| DR-learner | 0.80 ± 0.18 | 1.28 ± 0.05 | 1.33 ± 0.02 | 1.36 ± 0.02 |
| Plug-in ranker | 0.69 ± 0.32 | 1.24 ± 0.06 | 1.31 ± 0.02 | 1.36 ± 0.00 |
| **Rank-Learner (Ours)** | **1.00 ± 0.19** | **1.31 ± 0.01** | **1.34 ± 0.01** | **1.37 ± 0.00** |

Semi-synthetic and Criteo real-world data (Test AUTOC / AUUC×10³):

| Dataset | Metric | T-learner | DR-learner | Plug-in ranker | Rank-Learner |
|--------|------|-----------|------------|----------------|--------------|
| MovieLens (n=1k) | AUTOC | 1.31 ± 0.03 | 1.34 ± 0.02 | 1.30 ± 0.03 | **1.35 ± 0.01** |
| MIMIC-III (n=1k) | AUTOC | 1.12 ± 0.05 | 1.17 ± 0.02 | 1.11 ± 0.05 | **1.18 ± 0.02** |
| CPS (n=1k) | AUTOC | 0.87 ± 0.08 | 0.92 ± 0.02 | 0.87 ± 0.08 | **0.95 ± 0.01** |
| Criteo (test 1M) | AUUC×10³ | 5.08 ± 1.62 | 5.17 ± 1.13 | 5.04 ± 1.65 | **5.90 ± 0.40** |

### Ablation Study

| Configuration | Key Metric (Synthetic n=500) | Description |
|------|------------------------|------|
| Full Rank-Learner | AUTOC 1.31 | Complete orthogonal pseudo-label |
| w/o Orthogonal Correction (= Plug-in ranker) | AUTOC 1.24 | Remove $\omega_\tau\cdot\Delta_\eta$; Loss of 0.07 |
| w/o Direct Ranking (= DR-learner) | AUTOC 1.28 | Rank via CATE estimates; Loss of 0.03 |
| Pair sub-sampling ratio scan | Saturated quickly | Small fraction of $n^2$ pairs is sufficient |

### Key Findings
- **Orthogonalization is most valuable in small samples**: At $n=100$, Rank-Learner is 0.31 AUTOC higher than the plug-in ranker; at $n=2{,}000$, the gap is only 0.01. Smaller samples yield noisier nuisances, increasing the marginal utility of the DR correction.
- **Direct Ranking > CATE then Rank**: DR-learner is already orthogonal, yet Rank-Learner remains superior at all training scales. This confirms that "solving an easier problem" provides independent gains alongside orthogonalization.
- **Real-world gains are significant**: On Criteo, Rank-Learner achieves a 14% higher AUUC than the runner-up DR-learner with significantly lower variance (0.40 vs 1.13).

## Highlights & Insights
- **"Problem Dimension Reduction" Philosophy**: Ours explicitly highlights that CATE estimation is a harder statistical problem than ranking. Using $\mathcal L^{\text{soft}}$ aligns the training objective with the decision goal. This philosophy is transferable to uplift, policy learning, and ranking-aware fairness.
- **Orthogonalization in Labels, Not Loss**: While orthogonal learning typically requires rewriting the loss function, this work packages nuisance corrections into the pseudo-label $\tilde t_\eta$. This allows any pairwise ranking architecture (RankNet, LambdaMART) to benefit via a drop-in label replacement.
- **$\kappa$ as a Bridge**: The hyperparameter $\kappa$ allows continuous interpolation between "scaled CATE estimation" and "pure ranking," providing a knob to tune the bias–variance trade-off.

## Limitations & Future Work
- **Binary Treatment & Point Outcomes**: The current derivation covers $T\in\{0,1\}$ and continuous/binary $Y$. Multi-arm or time-varying confounding would require new influence functions.
- **Causal Assumptions**: Ours depends on consistency, positivity, and unconfoundedness; propensity estimation remains a challenge in heavily imbalanced settings.
- **Expansion Directions**: Future work could include listwise/top-k orthogonal objectives or integrating Rank-Learner as a policy module in sequential decision-making.

## Related Work & Insights
- **vs DR-learner (Kennedy 2023)**: Both use DR scores and cross-fitting, but DR-learner's second stage is CATE regression $\mathbb E[(g(X)-\phi_\eta(W))^2]$, while Rank-Learner embeds $\phi_\eta$ into pairwise soft labels for ranking.
- **vs Tree ranker (Kamran et al. 2024)**: The tree ranker directly optimizes non-differentiable criteria based on DR pseudo-outcomes but is model-specific and lacks Neyman orthogonality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to extend Neyman orthogonality to treatment effect ranking via soft pairwise labels.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic, semi-synthetic, and real-world data with extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression of motivation and excellent alignment of intuition with theory.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically grounded and easy-to-implement baseline for ranking-centric causal tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](../../ICLR2026/causal_inference/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/causal_inference/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[NeurIPS 2025\] Transferring Causal Effects using Proxies](../../NeurIPS2025/causal_inference/transferring_causal_effects_using_proxies.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](../../ICML2025/causal_inference/isolated_causal_effects_of_natural_language.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
