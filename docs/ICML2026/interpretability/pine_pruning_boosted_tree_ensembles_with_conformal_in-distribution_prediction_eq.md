---
title: >-
  [Paper Note] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence
description: >-
  [ICML 2026][Interpretability][Tree Ensemble Pruning] PINE contracts the "faithful pruning" equivalence constraint for boosted tree ensembles from the entire input space to an "in-distribution region" $\mathcal{X}_{\text{…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Tree Ensemble Pruning"
  - "Faithful Pruning"
  - "Conformal Prediction"
  - "In-Distribution Equivalence"
  - "Chow-Liu Tree"
date: 2026-05-08
content_hash: a22598264168472e
---

# PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence

**Conference**: ICML 2026  
**arXiv**: [2605.28068](https://arxiv.org/abs/2605.28068)  
**Code**: TBD  
**Area**: Model Compression / Tree Ensemble Pruning / Conformal Prediction  
**Keywords**: Tree Ensemble Pruning, Faithful Pruning, Conformal Prediction, In-Distribution Equivalence, Chow-Liu Tree

## TL;DR
PINE contracts the "faithful pruning" equivalence constraint for boosted tree ensembles from the entire input space to an "in-distribution region" $\mathcal{X}_{\text{ID}}(\alpha)$ defined by Chow-Liu tree likelihood and split conformal calibration. By smoothly controlling the compression-fidelity trade-off with a single parameter $\alpha$, it improves the compression rate by up to 30% relative to FIPE across 12 public tabular datasets, while providing a provable guarantee that the probability of "prediction consistency before and after pruning" is $\geq 1-\alpha$.

## Background & Motivation
**Background**: For tabular data, boosted decision tree ensembles like XGBoost remain SOTA. However, large ensembles suffer from slow inference and difficult verification (robustness/fairness), making post-training ensemble pruning a common practice. Existing approaches follow two paths: (a) accuracy-oriented pruning (IC/DREP/MDEP/ForestPrune, etc.), which only requires that accuracy does not drop significantly, allowing individual predictions to change; (b) faithful pruning (Born-Again Trees, FIPE), which requires exact prediction equivalence for any input before and after pruning.

**Limitations of Prior Work**: While accuracy-oriented pruning achieves high compression, many predictions change—in high-stakes scenarios like healthcare or finance, where downstream workflows or robustness/fairness checks are built around specific model outputs, any change in prediction necessitates re-doing the entire downstream pipeline. Faithful pruning like FIPE treats "prediction invariance" as a hard constraint. Unfortunately, it requires this to hold over the entire input space $\mathcal{X}$, including OOD "ghost points" that rarely appear in reality (e.g., logically impossible points like "Preschool education + 13.5 years of schooling" in *Adult* or "prior offenses=0 AND prior offenses>3" in *COMPAS*). By preserving fine-grained boundaries for these points, FIPE can only prune a 30-tree toy example down to 11 trees.

**Key Challenge**: A structural trade-off exists between fidelity and compression. Achieving 100% fidelity requires accounting for all OOD points, limiting compression; abandoning fidelity destroys decision consistency.

**Goal**: To find a mechanism that guarantees prediction equivalence on "inputs likely to occur" without making promises for OOD regions, where the size of this "in-distribution region" can be smoothly adjusted.

**Key Insight**: The authors observe that OOD regions are insignificant for decision-making but consume many equivalence constraints. By requiring equivalence only within an **in-distribution region** $\mathcal{X}_{\text{ID}}$, the feasible region for pruning expands immediately. Furthermore, if the coverage of $\mathcal{X}_{\text{ID}}$ is calibrated via conformal prediction, the probability that "future inputs fall into $\mathcal{X}_{\text{ID}}$" can be bounded at $\geq 1-\alpha$, translating "prediction equivalence" into a $\geq 1-\alpha$ probabilistic guarantee.

**Core Idea**: Use the negative log-likelihood of a Chow-Liu tree as a "plausible score" $s(\bm{x})$ and split conformal calibration to obtain a threshold $\tau(\alpha)$, defining $\mathcal{X}_{\text{ID}}(\alpha)=\{\bm{x}:s(\bm{x})\leq\tau(\alpha)\}$. This restricts the FIPE Oracle's search for counterexamples from $\mathcal{X}$ to $\mathcal{X}_{\text{ID}}(\alpha)$. The tree-like decomposition of Chow-Liu trees allows them to be integrated cleanly into a Mixed-Integer Linear Programming (MILP) formulation.

## Method

### Overall Architecture
The input consists of a pre-trained boosted tree ensemble $\mathcal{T}=\{T_m\}_{m=1}^M$, initial weights $\bm{w}^{(0)}$, a miscoverage level $\alpha$, a fitting set $\mathcal{D}_{\text{fit}}$ for the score function, and a calibration set $\mathcal{D}_{\text{cal}}$ for the threshold. The output is a sparse set of weights $\bm{w}$ (trees with zero weight are discarded) such that $\hat{y}(\bm{x};\bm{w})=\hat{y}(\bm{x};\bm{w}^{(0)})$ for any $\bm{x}\in\mathcal{X}_{\text{ID}}(\alpha)$.

The pipeline follows the iterative "Pruner + Oracle" framework of FIPE: (1) Fit the Chow-Liu score $s_{\text{CL}}(\cdot)$ on $\mathcal{D}_{\text{fit}}$; (2) Compute $\tau(\alpha)$ using the order statistics of $s_{\text{CL}}$ on $\mathcal{D}_{\text{cal}}$; (3) Initialize the equivalence constraint set $\mathcal{S}^{(0)}\leftarrow\mathcal{D}_{\text{fit}}$; (4) Repeat [Pruner solves for the sparsest weights satisfying equivalence on $\mathcal{S}^{(t)}$ → Oracle uses MILP to find a new counterexample within $\mathcal{X}_{\text{ID}}(\alpha)$ and merges it into $\mathcal{S}^{(t+1)}$] until the Oracle returns an empty set, yielding a **certified** equivalence guarantee on $\mathcal{X}_{\text{ID}}(\alpha)$.

Compared to FIPE, the primary modification is adding the linear constraint "$s_{\text{CL}}(\bm{x})\leq\tau(\alpha)$" to the Oracle. To ensure this is encodable in MILP, the choice of the plausible score is restricted, leading to the use of Chow-Liu trees.

### Key Designs

1. **Chow-Liu NLL as an MILP-encodable plausible score**:
    - **Function**: Characterizes whether an input is in-distribution as a scalar score $s(\bm{x})=-\log p_{\text{CL}}(\tilde{\bm{x}})$ and ensures this score can be embedded into an MILP.
    - **Mechanism**: Each continuous feature is discretized into $B$ bins to obtain $\tilde{\bm{x}}\in\{1,\dots,B\}^p$. A Maximum Mutual Information spanning tree (Chow-Liu tree) is fitted on the feature graph to approximate the joint distribution $p_{\text{CL}}(\tilde{\bm{x}})=p(\tilde{x}_r)\prod_{j\neq r}p(\tilde{x}_j\mid\tilde{x}_{\text{pa}(j)})$. The negative log-likelihood decomposes into "root marginal + tree-edge conditionals," each depending on a single bin or a parent-child bin pair. In MILP, indicator variables $q_{i,b}\in\{0,1\}$ mark "feature $i$ falls in bin $b$" and $u_{i,j,b,b'}\in\{0,1\}$ mark "parent-child bin combinations," with AND logic encoded via three linear constraints ($u_{i,j,b,b'}\leq q_{i,b}$, $u_{i,j,b,b'}\leq q_{j,b'}$, $u_{i,j,b,b'}\geq q_{i,b}+q_{j,b'}-1$). Thus, $s_{\text{CL}}$ is a linear sum of $q$ and $u$, with a constraint scale of $\mathcal{O}(pB^2)$, far smaller than the $\mathcal{O}(B^p)$ of a discrete input space. Furthermore, bin boundaries are rounded to the split thresholds $\Theta_j$ present in the ensemble $\mathcal{T}$ to prevent geometric misalignment between the MILP feasible region and $\{s\leq\tau\}$.
    - **Design Motivation**: For the Oracle's search for counterexamples in $\mathcal{X}_{\text{ID}}$ to be solvable, the score function must be MILP-friendly. Chow-Liu trees satisfy both "good joint distribution capture" and "additivity/linearizability," making them more suitable than isolation forests or KDE for embedding into faithful pruning frameworks.

2. **Split conformal calibration for $\tau(\alpha)$ with probabilistic guarantees**:
    - **Function**: Transforms the selection of the in-distribution region size from an empirical heuristic into a knob $\alpha$ with finite-sample, distribution-free guarantees.
    - **Mechanism**: Compute the order statistics $s_{(1)}\leq\cdots\leq s_{(n)}$ of scores $\{s(\bm{x}_i)\}$ on $\mathcal{D}_{\text{cal}}$ and set $\tau(\alpha)=s_{(k)}$ where $k=\lceil(n+1)(1-\alpha)\rceil$. Under the exchangeability assumption, $\mathbb{P}[s(\bm{X}_{\text{new}})\leq\tau(\alpha)]\geq 1-\alpha$. Combined with the Oracle proving no counterexamples exist in $\mathcal{X}_{\text{ID}}(\alpha)$, this yields Proposition 4.2: "$\mathbb{P}[\hat{y}(\bm{X}_{\text{new}};\bm{w})=\hat{y}(\bm{X}_{\text{new}};\bm{w}^{(0)})]\geq 1-\alpha$". This is a marginal probability guarantee.
    - **Design Motivation**: Existing faithful pruning lacks a tuning knob. Conformal prediction provides a way to define "almost all future inputs" with rigorous bounds without depending on distribution shape, turning the fidelity-compression trade-off into a user-controllable axis.

3. **Double exponential contraction of compressibility and search complexity**:
    - **Function**: Theoretically explains why contracting the guarantee region slightly leads to significant gains in compression and search speed.
    - **Mechanism**: The original space $\mathcal{X}$ is partitioned into up to $\prod_j(|\Theta_j|+1)$ cells, which explodes in high dimensions. With the Chow-Liu constraint, the Oracle only needs to search the discrete state set $A_\tau=\{\tilde{\bm{x}}:-\log p_{\text{CL}}(\tilde{\bm{x}})\leq\tau\}$. Proposition 4.3 shows $|A_\tau|\leq e^\tau$ (Proof: components $\tilde{\bm{x}}\in A_\tau$ have $p_{\text{CL}}(\tilde{\bm{x}})\geq e^{-\tau}$; by probability normalization $1\geq|A_\tau|e^{-\tau}$). Since $\tau(\alpha)$ is non-increasing with $\alpha$, increasing $\alpha$ leads to an exponential decrease in the search state upper bound $e^{\tau(\alpha)}$. Additionally, $\mathcal{X}_{\text{ID}}(\alpha)\subseteq\mathcal{X}$ implies relaxed constraints and a larger feasible region for pruning.
    - **Design Motivation**: While "relaxed constraints lead to more compression" is intuitive, the authors link "search affordability" to the same knob $\tau(\alpha)$, showing that PINE's effectiveness stems from the exponential reduction in search difficulty.

### Loss & Training
Objective: $\arg\min_{\bm{w}\geq 0}\|\bm{w}\|_0$, s.t. $\hat{y}(\bm{x};\bm{w})=\hat{y}(\bm{x};\bm{w}^{(0)}),\forall\bm{x}\in\mathcal{X}_{\text{ID}}(\alpha)$. Solved via iterative Pruner (sparsest feasible weights on $\mathcal{S}^{(t)}$) and Oracle (MILP search for counterexamples restricted by $s\leq\tau(\alpha)$, using the leaf-selection MILP encoding from OCEAN). Main experiments use $\ell_0$, with $\ell_1$ approximations in Appendix B.2 to save time. Solver: Gurobi v11.0.3. XGBoost hyperparams: $D=2$, $M=30$; $\alpha \in \{0.05, 0.1, 0.2, 0.4, 0.6, 0.8\}$; bins $B=4$.

## Key Experimental Results

### Main Results
12 UCI/OpenML tabular classification datasets, 5 random seeds, comparing PINE-CL with FIPE (faithful baseline) and IC/DREP/MDEP (accuracy-oriented baselines). Results for *Pima-Diabetes*:

| Method | $\alpha$ | Pruning Rate (%) ↑ | Fidelity (%) ↑ | Time (s) ↓ | Iterations ↓ |
|--------|----------|-------------------|----------------|-------------|--------------|
| FIPE | – | 17.3 | 100.0 | 42.5 | 24.6 |
| PINE-CL | 0.05 | 22.7 | 100.0 | 48.1 | 19.2 |
| PINE-CL | 0.1 | 26.7 | 100.0 | 48.0 | 19.6 |
| PINE-CL | 0.2 | 30.0 | 100.0 | 47.0 | 19.6 |
| PINE-CL | 0.4 | 34.7 | 99.9 | 33.8 | 15.2 |
| PINE-CL | 0.6 | 45.3 | 98.6 | 19.4 | 11.0 |
| PINE-CL | 0.8 | 55.3 | 98.3 | 12.0 | 7.8 |

Averaged across 12 datasets: As $\alpha$ increases from 0.05 to 0.8, average pruning rate rises from 44.6% to 67.8%, while average fidelity only drops from 99.96% to 99.15%. Compression rate gains over FIPE reach up to 30%.

### Ablation Study

| Dimension | Config | Observation | Explanation |
|-----------|--------|-------------|-------------|
| Depth $D$ ($M=30,\alpha=0.8$) | $D=2\to 5$ | Pruning rate 66.94% → 34.44%, Time 3.82s → 932.75s, Iters 2.17 → 13.17 | Deeper trees create more local decision regions; more trees are "partially useful." |
| Tree count $M$ ($D=3,\alpha=0.8$) | $M=10\to 50$ | Pruning rate remains 39.17% → 51.11%, fidelity ≈ 99% | $M$ primarily affects optimization overhead, not compression rate. |
| Bin count $B$ | Appendix B.6 | Robust trend | Binning granularity is not a bottleneck. |
| Objective $\ell_0$ vs $\ell_1$ | Appendix B.2 | $\ell_1$ is faster, results similar | Suitable for large-scale scenarios. |

### Key Findings
- **RQ1**: Accuracy-oriented methods (IC/DREP/MDEP) maintain test accuracy but fidelity drops monotonically with pruning rate, showing that "similar accuracy" $\neq$ "consistent decisions." PINE maintains fidelity near 1 even at high compression.
- **RQ2**: Empirical test coverage $\hat{\pi}_{\text{ID}}$ closely tracks the target $1-\alpha$, proving split conformal calibration is valid on real data.
- **RQ3**: Evaluating conditional fidelity $\hat{\rho}_{\text{ID}}$ on $\mathcal{X}_{\text{ID}}(\alpha)$ shows that IC/DREP/MDEP still fail ($<1$)—they change predictions even for in-distribution inputs. PINE achieves exactly $\hat{\rho}_{\text{ID}}=1$ by design.
- **Case study**: Counterexamples ignored by PINE often involve logically impossible attribute combinations. FIPE must preserve extra trees to handle these, whereas PINE ignores them to gain compression.
- **Practical $\alpha$ selection**: Post-hoc selection on a held-out $\mathcal{D}_{\text{sel}}$ with a 95% fidelity goal achieved 70.8% average compression; a more conservative Bonferroni-Clopper-Pearson upper bound selector achieved 57.2% compression and 99.72% fidelity.

## Highlights & Insights
- **Natural integration of "Faithful Pruning + Conformal"**: Faithful pruning was previously a binary "entire space" problem. Conformal prediction provides a distribution-free probability for coverage. Combining them transforms a hard constraint into a $1-\alpha$ probabilistic one. This paradigm of "relaxing symbolic constraints using probabilistic tools" is generalizable to other tasks like rule extraction or auditing.
- **Plausible score selection is about MILP-friendliness, not just statistics**: The authors emphasize compatibility with any "tree-structured distribution constraint." This is a hidden extension point—integrating stronger density models (like normalizing flows) would first require solving their linear encoding.
- **Theoretical bound $|A_\tau|\leq e^\tau$ links compression and efficiency**: A seemingly loose information-theoretic bound elegantly connects "in-distribution state count" to search complexity, providing monotone controllability via $\alpha\uparrow\Rightarrow\tau\downarrow\Rightarrow|A_\tau|\downarrow$.
- **Using OOD concepts to evaluate baselines (RQ3)**: The definition of $\hat{\rho}_{\text{ID}}$ exposes that accuracy-oriented methods change predictions even for high-density inputs, which is more convincing than comparing raw fidelity numbers.

## Limitations & Future Work
- **Dependency on MILP optimality**: Probabilistic guarantees rely on the Oracle reaching certified optimality; time-limited searches degrade into empirical guarantees. Large ensembles ($D=5$) already show high overhead.
- **Exchangeability assumption**: Relies on $\mathcal{D}_{\text{cal}}$ being i.i.d. with future inputs; distribution shift would require extensions like conditional or weighted conformal prediction.
- **Discretization and Rounding overhead**: As the number of features and split thresholds $|\Theta_j|$ grows, the engineering cost of bin rounding increases.
- **Hard decision consistency only**: Does not handle equivalence for regression or soft probability logits. Mismatches in confidence scores may still occur.
- **Lack of task-aware $\alpha$ selection theory**: Selecting $\alpha$ currently relies on empirical held-out sets rather than a bridge directly translating "business-acceptable fidelity" to $\alpha$.

## Related Work & Insights
- **vs FIPE (Emine et al., 2025)**: FIPE requires $\hat{y}$ equivalence on the entire $\mathcal{X}$. PINE uses conformal prediction to restrict this to $\mathcal{X}_{\text{ID}}(\alpha)$, replacing "100% fidelity" with a "$\geq 1-\alpha$ probabilistic fidelity" and introducing a tunable knob.
- **vs Born-Again Tree Ensembles (Vidal & Schiffer, 2020)**: BATE mashes the ensemble into one large equivalent tree, changing the model class; PINE preserves the ensemble structure, which is better for downstream boosting-based verification.
- **vs ForestPrune (Liu & Mazumder, 2023) / LOP (Devos et al., 2025)**: These methods delete nodes or subtrees for higher compression but abandon fidelity; PINE differentiates itself with well-defined probabilistic fidelity.
- **vs accuracy-oriented IC/DREP/MDEP**: These "contribution + diversity" subset selection methods (from PyPruning) offer no equivalence guarantees and change predictions even in-distribution (RQ3).
- **Transferable Insight**: The "Faithful Pruning = Symbolic Verification (MILP) + Global Equivalence" paradigm can be applied to any extraction/distillation task. By finding an MILP-encodable and calibrated plausible score, one can transform "too-strict" global space requirements into "probabilistically guaranteed" in-distribution ones.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining conformal coverage with faithful pruning constraints is highly natural, though the components themselves are mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 datasets, 5 seeds, 6 $\alpha$ values, plus sensitivities for depth, tree count, bins, and objectives.
- Writing Quality: ⭐⭐⭐⭐ Clear trajectory from motivation to theory and case studies; high readability of algorithms and formulas.
- Value: ⭐⭐⭐⭐ Provides a deployable "controllable fidelity" solution for high-stakes tabular compression, with a reusable methodology for auditing and extraction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)
- [\[ICML 2026\] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions](bridging_the_knowledge-prediction_gap_in_llms_on_multiple-choice_questions.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/interpretability/conformal_prediction_as_bayesian_quadrature.md)
- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)
- [\[ICML 2026\] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions](bridging_the_knowledge-prediction_gap_in_llms_on_multiple-choice_questions.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)

</div>

<!-- RELATED:END -->
