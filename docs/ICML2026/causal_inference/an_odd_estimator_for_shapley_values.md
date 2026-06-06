---
title: >-
  [Paper Note] An Odd Estimator for Shapley Values
description: >-
  [ICML2026][Causal Inference][Shapley values] This paper proves that Shapley values depend only on the odd component of the set function. Based on this, it proposes OddSHAP: it uses paired sampling to isolate odd signals…
tags:
  - "ICML2026"
  - "Causal Inference"
  - "Shapley values"
  - "feature attribution"
  - "OddSHAP"
  - "paired sampling"
  - "Fourier regression"
date: 2026-05-08
content_hash: 6d7c83d06ab56f15
---

# An Odd Estimator for Shapley Values

**Conference**: ICML2026  
**arXiv**: [2602.01399](https://arxiv.org/abs/2602.01399)  
**Code**: https://github.com/FFmgll/oddshap  
**Area**: Interpretability  
**Keywords**: Shapley values, feature attribution, OddSHAP, paired sampling, Fourier regression  

## TL;DR
This paper proves that Shapley values depend only on the odd component of the set function. Based on this, it proposes OddSHAP: it uses paired sampling to isolate odd signals, GBT to filter high-order odd Fourier interactions, and performs sparse odd regression, significantly outperforming flexible-budget Shapley estimators in mid-to-high dimensional explanation tasks.

## Background & Motivation
**Background**: The Shapley value is one of the most widely used feature attribution frameworks in machine learning interpretability. It treats model predictions as a set function $f:2^{[d]}\to\mathbb{R}$ and assigns the average marginal contribution to each feature. Since exact calculation requires traversing an exponential number of coalitions, practical methods typically use sampling or surrogate regression approximations, such as KernelSHAP, LeverageSHAP, Permutation Sampling, SVARM, MSR, PolySHAP, and various proxy-based estimators.

**Limitations of Prior Work**: Many advanced estimators employ paired sampling, where for every sampled coalition $S$, the complement $S^c$ is also sampled. While this technique is empirically effective, the reasons for its success are not fully understood. Simultaneously, high-order polynomial or surrogate estimators offer greater expressivity but face combinatorial explosion: the number of candidate interaction terms grows rapidly with the order, making it difficult to maintain both accuracy and stability under limited budgets.

**Key Challenge**: Shapley values are only concerned with the functional components that affect marginal contributions, but traditional regression-based estimators often attempt to fit components irrelevant to the Shapley value. If an estimator wastes its sampling budget on the irrelevant even component or a large number of low-impact interactions, variance and computational costs increase.

**Goal**: The authors aim to provide a rigorous theoretical explanation for paired sampling and design a budget-flexible Shapley estimator based on this explanation: one that can leverage high-order interactions to improve accuracy without needing to fit all high-order terms.

**Key Insight**: The paper starts from the odd/even decomposition of set functions. If defined as $f_{odd}(S)=\frac12(f(S)-f(S^c))$, then the Shapley value satisfies $\phi_i(f)=\phi_i(f_{odd})$. Consequently, the estimator can fit only the odd component and entirely discard the even component.

**Core Idea**: Transform Shapley estimation from "fitting the entire value function" to "fitting only the sparse interactions in the odd Fourier subspace that contribute to the Shapley value."

## Method
The theoretical foundation of OddSHAP involves two steps. First, it is proved that paired sampling decomposes the weighted least squares objective into two non-interfering parts: odd and even. As long as the target is the Shapley value, the even part can be ignored. Second, the Fourier basis is adopted because the Fourier basis functions $\chi_T(S)=(-1)^{|S\cap T|}$ are naturally divided into odd/even categories based on the parity of $|T|$: if $|T|$ is odd, it is an odd term; if it is even, it is an even term. This makes "fitting only odd interactions" algorithmically straightforward.

### Overall Architecture
The input consists of a black-box value function $f$, a sampling budget $m$, and a regression variable factor $\eta$. The algorithm first obtains a set of coalitions using paired sampling; then, it fits a gradient boosted tree (GBT) proxy model to filter out odd Fourier interactions with the largest magnitudes; finally, it solves a weighted least squares problem with boundary constraints on the support $T_{\leq1}\cup T_{odd}$ to obtain the odd polynomial approximation $\hat f_{odd}$, from which Shapley values are directly calculated using Fourier coefficients.

If the budget is too low to stably regress even the linear terms, specifically $m<d\eta$, the algorithm retreats to the TreeSHAP output of the GBT. Otherwise, the number of candidate high-order odd interactions is set to $|T_{odd}|=\lceil m/\eta\rceil-d$, ensuring that the number of regression variables grows linearly with the budget rather than combinatorially with the number of features and the order.

### Key Designs
1.  **Odd component theoretical criterion**:
    - **Function**: Proves that the effective signal for the Shapley value resides only in the odd part of the set function.
    - **Mechanism**: Decomposes any set function as $f=f_{odd}+f_{even}$, where the odd part satisfies $f_{odd}(S)=-f_{odd}(S^c)$ and the even part satisfies $f_{even}(S)=f_{even}(S^c)$. The paper provides the observation $\phi_i(f)=\phi_i(f_{odd})$, meaning the even component contributes zero to the Shapley values of all features.
    - **Design Motivation**: This directly explains why paired sampling is useful: it is not merely a variance reduction trick, but rather orthogonally projects out the Shapley-irrelevant even components from the estimation target.

2.  **Fourier odd regression**:
    - **Function**: Makes fitting "only odd terms" executable at the basis function level.
    - **Mechanism**: In the Fourier basis, whether $\chi_T$ is odd is determined solely by whether $|T|$ is odd. OddSHAP retains only linear terms and the filtered odd high-order interactions, ensuring through rigorous boundary constraints that the estimated Shapley values satisfy efficiency (i.e., the sum equals $f([d])-f(\emptyset)$). The formula for calculating attribution from coefficients is $\phi_i(\hat f_{odd})=-2\sum_{T\ni i, |T|\ odd}\beta_T/|T|$.
    - **Design Motivation**: The unanimity basis commonly used in KernelSHAP/LeverageSHAP does not cleanly separate odd and even components; the Fourier basis provides a natural structural separation.

3.  **GBT interaction screening + Budget-adaptive support set**:
    - **Function**: Retains the most impactful odd interactions for the value function without enumerating all high-order interactions.
    - **Mechanism**: The algorithm first fits a GBT proxy using the same batch of samples, then extracts odd Fourier coefficients with the largest absolute magnitudes using a ProxySPEX-style method. The regression support size is controlled by $\eta$; a larger budget allows for more interactions. If the budget is insufficient, it falls back to TreeSHAP to avoid underdetermined regression.
    - **Design Motivation**: Machine learning value functions often have only a few important Fourier interactions. Screening before regression provides a trade-off between expressivity and statistical stability.

### Loss & Training
The core optimization of OddSHAP is weighted least squares regression with Shapley kernel weights, but the objective is solved only on the odd Fourier support. After pre-calculating $f_{odd}(S)=\frac12(f(S)-f(S^c))$ via paired samples, the complement rows can be discarded, using only $m/2$ representative samples to fit the odd target. The regression also explicitly handles boundary constraints, rather than approximating them with pseudo-infinite weights as done in some KernelSHAP implementations.

## Key Experimental Results

### Main Results
The experiments evaluate Shapley approximations across 30 random prediction instances on 8 value functions, covering language, image, tabular, and synthetic functions. The evaluation metrics are the median and IQR of the MSE relative to ground-truth Shapley values.

| Dataset / Function | Dim | Area | Ours | Prev. SOTA / Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | 14 | language | Comparable to best flexible-budget methods like RegressionMSR | RegressionMSR / LeverageSHAP | No significant disadvantage in low dimensions |
| ViT16 | 16 | image | Comparable to best flexible-budget methods; outperforms most FFD corrected settings | RegressionMSR / FFD variants | High-order interactions more active in deep models |
| Cancer | 30 | tabular | Outperforms all flexible-budget baselines at mid-to-high budgets | LeverageSHAP / MSR / SVARM / FourierSHAP | Up to 62x MSE reduction due to interaction modeling |
| CG60 / IL60 | 60 | synthetic | Clearly leads flexible-budget baselines when budget is sufficient | MSR / FourierSHAP / RegressionMSR | Advantage more pronounced in high-dimensional interaction functions |
| NHANES | 79 | tabular | Outperforms flexible-budget baselines at mid-to-high budgets | TreeSHAP ground truth comparison | Remains usable as dimension increases |
| Crime | 101 | tabular | Maintains competitiveness on the runtime-MSE curve | LeverageSHAP / FFD-RD / Proxy | More scalable than fixed $O(d^2)$ designs |

### Ablation Study
The ablation study directly validates OddSHAP's three core choices: the number of interactions, paired sampling, and retaining only odd interactions.

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| $\eta=10$, approx. 1000 interactions, 10000 samples | At least 6x MSE reduction across all functions, up to 62x on Cancer | A moderate number of odd high-order interactions significantly outperforms interaction-free LeverageSHAP |
| $\eta\in\{2,5,10,50\}$ | MSE rebounds after too many interactions | Increased expressivity leads to overfitting; support set should not expand infinitely with budget |
| Paired + Odd interactions | Normalized best configuration | Directly isolates the odd component, focusing budget on terms that contribute to Shapley values |
| Paired + All interactions | Slightly worse MSE and slower | Even terms mathematically cancel out but consume interaction budget and computation time |
| Non-paired sampling | Overall weaker than paired sampling | Without paired structure, odd/even separation is not clean, leading to more unstable estimates |
| FFD-RD fixed-budget | Strong on tree models, degrades on deep models | Relies on high-order interaction truncation assumptions; $O(d^2)$ sample requirement is inflexible in high dimensions |

### Key Findings
- The value of paired sampling is rigorously explained as even-odd separation rather than a simple empirical variance reduction technique.
- OddSHAP does not sacrifice performance in low-dimensional tasks and clearly outperforms flexible-budget baselines in mid-to-high dimensional tasks by modeling sparse odd interactions.
- Even interactions contribute nothing to the Shapley value; continuing to fit even terms under paired sampling only diverts budget and increases runtime.

## Highlights & Insights
- The paper elevates a common engineering trick into a clear theory: paired sampling is precisely estimating the odd component. This explanation is elegant and guides the design of new estimators.
- The choice of the Fourier basis is excellent. It is not chosen for mathematical aesthetics, but because odd/even properties can be directly determined by interaction order, allowing the algorithm to precisely discard irrelevant subspaces.
- The role of the GBT proxy is well-positioned: it does not serve directly as the final explainer but helps identify sparse high-impact interactions, while constrained regression ensures Shapley consistency.
- This paper provides an important reminder for interpretability methods: estimating the value function itself is not identical to estimating all information required for attribution. Fitting only the attribution-relevant subspace can be more efficient than fitting the complete function.

## Limitations & Future Work
- The regression phase of OddSHAP grows quadratically with the number of selected interactions; if the number of interactions remains tied to the sampling budget, overall overhead can grow approximately cubically with $m$. The authors suggest capping the number of interactions and decoupling it from $m$ at large budgets.
- Paired sampling compresses $m$ queries into $m/2$ independent rows, which reduces independent subset coverage and may increase variance or mutual coherence; thus, it is not guaranteed to outperform non-paired sampling on all functions.
- Interaction screening relies on a GBT proxy. If the proxy fails to capture important Fourier interactions of the true value function, OddSHAP may miss critical high-order terms.
- Fixing $\eta=10$ was robust in experiments, but how to adaptively select $\eta$ across different domains, dimensions, and value function evaluation costs warrants further research.

## Related Work & Insights
- **vs KernelSHAP / LeverageSHAP**: These essentially perform low-order/linear surrogate regression; OddSHAP adds filtered odd high-order interactions while maintaining consistency, thereby reducing bias on complex value functions.
- **vs PolySHAP**: PolySHAP extends to polynomial regression but faces combinatorial explosion in candidate terms; OddSHAP controls support size using the Fourier odd subspace and GBT screening.
- **vs RegressionMSR / ProxySPEX**: Proxy methods use a learner to approximate the value function; OddSHAP uses the proxy as an interaction screener while still calculating Shapley via rigorous regression and boundary constraints.
- **vs FFD-RD**: FFD utilizes fixed combinatorial designs and high-order truncation assumptions, which is strong for tree models; OddSHAP is more flexible, especially for deep models or functions with active high-order interactions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The odd component criterion and OddSHAP design are highly insightful, tightly integrating theory and algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 8 value functions, multiple baseline categories, runtime, interaction sparsity, and paired sampling ablations with sufficient support.
- Writing Quality: ⭐⭐⭐⭐☆ The structure is clear, though the Fourier/Shapley theoretical density is high, requiring background knowledge from some readers.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for Shapley estimation, sampling design, and high-order interaction attribution in interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICML 2026\] Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal-jepa_learning_world_models_through_object-level_latent_masking.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](investigating_memory_in_model-free_rl_with_popgym_arcade.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)

</div>

<!-- RELATED:END -->
