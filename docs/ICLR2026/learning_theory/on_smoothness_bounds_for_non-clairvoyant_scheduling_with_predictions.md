---
title: >-
  [Paper Note] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions
description: >-
  [ICLR 2026][Theory of Learning-Augmented Algorithms][Algorithms with Predictions] This paper redefines the "smoothness" metric in algorithms with predictions by measuring the competitive ratio only on subsets of instances where the "prediction actually provides additional information," thereby preventing the old definition from being contaminated by uninformative instances. Under this new metric, the paper provides tighter lower bounds and matching algorithms for three types…
tags:
  - "ICLR 2026"
  - "Theory of Learning-Augmented Algorithms"
  - "Online Scheduling"
  - "Algorithms with Predictions"
  - "Smoothness"
  - "Non-Clairvoyant Scheduling"
  - "Competitive Ratio"
  - "Adversarial Lower Bounds"
date: 2026-05-08
content_hash: 76ed46b7fe6f75cc
---

# On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Zcn4n57lHg](https://openreview.net/forum?id=Zcn4n57lHg)  
**Code**: None  
**Area**: Theory of Learning-Augmented Algorithms / Online Scheduling  
**Keywords**: Algorithms with Predictions, Smoothness, Non-Clairvoyant Scheduling, Competitive Ratio, Adversarial Lower Bounds

## TL;DR
This paper redefines the "smoothness" metric in algorithms with predictions by measuring the competitive ratio only on subsets of instances where the "prediction actually provides additional information," thereby preventing the old definition from being contaminated by uninformative instances. Under this new metric, the paper provides tighter lower bounds and matching algorithms for three types of non-clairvoyant scheduling problems (single-machine total completion time: $\eta$ lower bound + $\eta^2$ algorithm; identical parallel machine makespan: $2-O(\eta^{-2})$ lower bound + $O(\eta^2)$ algorithm; related machine makespan: tight $\lceil\log\eta\rceil$ bound).

## Background & Motivation
**Background**: "Algorithms with predictions" (learning-augmented algorithms) is an active direction in online decision-making. These give an online algorithm a prediction of the unknown input and evaluate it using two primary metrics: **consistency** (competitive ratio with perfect predictions) and **robustness** (competitive ratio with worst-case predictions). Beyond these extremes, it is desired that algorithms are **smooth**: performance degrades gracefully as prediction error increases, rather than dropping off a cliff. The common practice is to express the competitive ratio as a function of prediction error $\eta$, known as the smoothness curve.

**Limitations of Prior Work**: The authors point out that the traditional definition of smoothness ("competitive ratio as a function of error") is often **meaningless**. For a given error $\eta$, one can often construct a set of instances where the prediction provides no additional information—especially when error is defined in "absolute terms" (e.g., $L_1$ distance $\sum_j|p_j^*-p_j|$). The paper provides a straightforward counterexample: take any instance $I$, scale all job sizes such that $\sum_j p_j'^* = \eta$, and set all predictions to 0. The error is exactly $\eta$, yet the prediction is completely uninformative, and the algorithm performs as poorly as the non-clairvoyant worst case. Since this holds for **any** $\eta$, the old smoothness curve can be driven arbitrarily high by these uninformative instances, failing to reflect the marginal benefit of "reasonable predictions."

**Key Challenge**: Smoothness aims to measure the "marginal value brought by predictions," but the old definition mixes "useful" and "useless" prediction instances into a single supremum, resulting in a metric dominated by useless instances that loses diagnostic meaning.

**Goal**: Redefine a smoothness metric measured **only on instances where the prediction is guaranteed to provide additional information**. This allows the lower bound to be interpreted as "even if predictions always provide information, no algorithm can do better than this ratio," characterizing both the limits and potential of predictions.

**Key Insight**: Partition the set of all instances $\mathcal{I}$ into three parts based on prediction error: perfect predictions, errors so large that predictions cannot distinguish anything in the worst case, and errors small enough that predictions certaintly provide information—corresponding to consistency, robustness, and smoothness, respectively. The key is a predicate that precisely characterizes "uninformative predictions."

**Core Idea**: Use a predicate $R(I)$ to determine "whether there exists a constant prediction vector with error exactly $\eta(I)$" to define "uninformative" instances and exclude them from the smoothness measurement domain. The paper re-evaluates lower bounds and algorithm analysis for classic scheduling problems within this "reasonable prediction" subset.

## Method
This paper is purely theoretical. The core is not an algorithmic pipeline but **the definition of a new metric + proofs of lower bounds and matching algorithms for three scheduling problems under this metric**. Under the Overall Architecture: the instance universe is partitioned into three blocks using the predicate $R(I)$, smoothness is redefined, and then adversarial instances are constructed for lower bounds and algorithms are designed for upper bounds across three classic non-clairvoyant scheduling problems to verify that the new metric provides tighter bounds and reveals the marginal value of predictions.

### Overall Architecture
**Problem Setting**: $m$ machines, $n$ independent jobs. The true size $p_j^*$ of job $J_j$ is unknown until completion (non-clairvoyant), and machine $M_i$ has speed $s_i$. A predicted size $p_j$ is given upon arrival. Error is defined **multiplicatively**: for a single job $\eta_j=\max\{p_j/p_j^*,\,p_j^*-p_j\}$, and for an instance $\eta(I)=\max_j\eta_j\ge 1$, where $\eta=1$ iff the prediction is perfect. Multiplicative error is **scale-invariant** compared to $L_1$; scaling all job sizes by 100 multiplies $L_1$ error by 100 but leaves multiplicative error unchanged, which better fits actual algorithmic behavior.

**Partitioning Instances + Redefining Metric**: Given an error function $\eta(\mathbf{p},\mathbf{p}^*)$, the predicate $R(I)$ determines if "there exists a constant vector $\mathbf{p}$ such that $\eta(\mathbf{p},\mathbf{p}^*(I))=\eta(I)$." If true, it means at this error level, the prediction could be manifested as "all job predictions are identical," failing to provide any ordering or grouping and thus equaling no prediction. The partition is:
$$\mathcal{I}_c=\{I\mid\eta(I)=\eta(\mathbf{p}^*,\mathbf{p}^*)\},\quad \mathcal{I}_r=\{I\mid R(I),\,I\notin\mathcal{I}_c\},\quad \mathcal{I}_s=\{I\mid\neg R(I),\,I\notin\mathcal{I}_c\}$$
These three sets are exhaustive and mutually exclusive. Consistency, robustness, and smoothness are the suprema of the competitive ratio $A(I)/\mathrm{Opt}(I)$ over $\mathcal{I}_c, \mathcal{I}_r, \mathcal{I}_s$ respectively, where smoothness is $s_A(\eta)=\sup_{I\in\mathcal{I}_s,\,\eta(I)\le\eta}A(I)/\mathrm{Opt}(I)$. Under multiplicative error, one can derive $\mathcal{I}_s=\{I\mid 1<\eta(I)<\sqrt{P(I)}\}$, where $P(I)$ is the ratio of the largest to smallest job size—intuitively meaning "the error is not yet large enough to be disguised by a constant vector."

**Solving Three Problems**: The framework is applied to three non-clairvoyant scheduling problems under Graham notation, providing lower bounds (adversarial construction) and upper bounds (algorithms).

### Key Designs

**1. Refined Smoothness Definition: Using Predicate $R(I)$ to Exclude Uninformative Instances**

**Design Motivation**: To address the contamination of smoothness by uninformative instances. **Mechanism**: Instead of taking the supremum of the competitive ratio over all imperfect instances, $R(I)$ is used to set aside instances where a constant prediction vector with the same error exists ($\mathcal{I}_r$). Measurement is restricted to the complement $\mathcal{I}_s$. Formally, $s_A(\eta)=\sup_{I\in\mathcal{I}_s,\,\eta(I)\le\eta}A(I)/\mathrm{Opt}(I)$. The resulting lower bound has a clean interpretation: **even when predictions always provide extra information, no deterministic algorithm can perform better than $s_A(\eta)$**.

**2. Single Machine Total Completion Time: Min-Max Adversary Yields $\eta$ Lower Bound and $\eta^2$ Algorithm**

For the $1\,|\,\text{online-time-nclv, pmtn}\,|\,\sum C_j$ problem. **Limitations of Prior Work**: Previous works used scale-sensitive $L_1$ error and only yielded $O(\eta^2\log\eta)$ upper bounds (e.g., from Azar et al. 2022). With multiplicative error, the lower bound proof involves a **scheduler vs. adversary** game. An adaptive adversary forces any algorithm to schedule in non-increasing order while the optimum is non-decreasing. Theorem 4.1 provides a piecewise lower bound ($\eta$ when $\eta\le\eta^*\approx1.835$), and Theorem 4.2 proves that **Shortest Predicted Job Size First (SPJF)** achieves an $\eta^2$ competitive ratio.

**3. Identical Parallel Machines Makespan: Dynamic Adversary + List-Adjusting Technique**

For $Pm\,|\,\text{online-time-nclv, pmtn-restart}\,|\,C_{\max}$. Bampis et al. (2023) gave a lower bound of $2-1/\lfloor\eta^2\rfloor$ and an upper bound of $\min\{2(\eta^2+1)/3,\,2\}$. **Ours** improves both. The **lower bound** uses a dynamic adversary that adapts with error, achieving $2-O(\eta^{-2})$. The **upper bound** uses a "list-adjusting" technique: running a $(1+\epsilon)$-approximate offline algorithm treating predictions as truth (named SIMPLE) yields $1+\epsilon$ consistency and $O(\eta^2)$ smoothness, but poor robustness ($\Theta(m)$). The logic is: if machine $M_i$ finishes jobs assigned by SIMPLE while others are waiting, it processes a waiting job instead of idling. This ensures the schedule can be replicated by list scheduling, inheriting $2-\frac{1}{m}$ robustness. Theorem 5.4 gives a $\min\{\eta^2+\epsilon,\,2-\frac{1}{m}\}$ competitive ratio.

**4. Related Machines Makespan: Tight $\lceil\log\eta\rceil$ Bound Filling the Gap**

For $Qm\,|\,\text{online-time-nclv, pmtn-restart}\,|\,C_{\max}$. While Zhao et al. (2024) provided an $O(\log \eta)$ algorithm, a lower bound was missing. Theorem 6.1 fills this gap: $s_A(\eta)\ge\lceil\log\eta\rceil$. The construction uses an adversarial instance family where groups of machines and jobs have exponentially increasing speeds and sizes, but all predictions are identical ($\eta$). Since an $O(\log \eta)$ algorithm exists, this lower bound is **asymptotically tight**.

## Key Experimental Results

### Main Results: Smoothness Bounds for Three Scheduling Problems

| Problem (Graham notation) | Error Metric | Prev. Bounds | Lower Bound (Ours) | Upper Bound (Ours) |
|---|---|---|---|---|
| Single Machine $\sum C_j$ | Multiplicative $\eta$ | $O(\eta^2\log\eta)$ | $\eta$ ($\eta\le1.835$), $\lambda(\eta) \to 2$ | $\eta^2$ (SPJF) |
| Identical Parallel $C_{\max}$ | Multiplicative $\eta$ | LB $2-1/\lfloor\eta^2\rfloor$; UB $\min\{\tfrac{2(\eta^2+1)}{3},2\}$ | $2-O(\eta^{-2})$ | $\min\{\eta^2+\epsilon,\,2-\tfrac1m\}$ |
| Related Machines $C_{\max}$ | Multiplicative $\eta$ | $O(\log\eta)$ algorithm, **no LB** | $\lceil\log\eta\rceil$ (Tight) | $O(\log\eta)$ (Existing) |

### Key Findings
- **Lower bounds do not exceed robustness constants**: For identical machines $(\le 2)$ and single machine $(\to 2)$, the bounds are consistent with the existence of constant competitive non-clairvoyant algorithms, meaning the new metric does not artificially inflate the bounds.
- **Gap remains in single machine problems**: The gap between LB $\eta$ and UB $\eta^2$ persists; closing this is an open problem.
- **Related machines is the only problem with tight bounds across the board**: With the non-clairvoyant lower bound of $\Omega(\log m)$, existing algorithms are simultaneously asymptotically optimal across consistency, smoothness, and robustness.
- **New metric sharpens lower bounds without weakening upper bounds**: All lower bounds also hold for the alternative definition $s_A'$ that does not exclude $\mathcal{I}_r$.

## Highlights & Insights
- **The "Adversary Constrained by Prediction" perspective**: In traditional competitive analysis, the adversary can create any instance. With predictions, the adversary is restricted to instances consistent with the prediction within error $\eta$. The smoothness lower bound quantifies the advantage this constraint gives the scheduler.
- **Predicate $R(I)$ is a reusable detector**: This tripartite partition can be generalized to other online problems (e.g., one-way trading) to define meaningful smoothness.
- **List-adjusting as a robustification patch**: Transforming a high-consistency but non-robust offline approximation (SIMPLE) into a robust one by simply preventing idling is a powerful and transferable strategy.
- **Business interpretation of bounds**: Translating an $\Omega(\eta)$ bound into "halving prediction error at most doubles performance gain" provides a quantitative basis for deciding whether to invest in more accurate models.

## Limitations & Future Work
- **Unclosed gaps**: Significant gaps remain for single-machine and identical parallel machine problems.
- **Scope**: The study is limited to deterministic algorithms and three specific scheduling objectives. Randomized algorithms or metrics like weighted completion time are not yet covered.
- **Upper bound improvement**: It remains unknown if there exists a problem where $s_A(\eta) < s_A'(\eta)$, i.e., whether the refined definition allows for strictly better algorithms.
- **Dependence on error metric**: The partition of $\mathcal{I}_s$ depends on the multiplicative error. Applying this to new error metrics requires re-deriving $R(I)$.

## Related Work & Insights
- **vs. Azar et al. (2023)**: Their discrete-smoothness requires predictions to be isomorphic to the solution; this paper's smoothness does not restrict prediction form and uses an information-theoretic approach.
- **vs. Bampis et al. (2023)**: They used a static adversary; this paper uses an adaptive dynamic adversary to sharpen the lower bound and introduces list-adjusting to improve upper bounds.
- **vs. Zhao et al. (2024)**: This paper completes their work by providing the matching lower bound $\lceil\log\eta\rceil$ for related machines.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefines a core metric (smoothness) with a formal, generalizable predicate $R(I)$.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid theoretical coverage of three major problems, though some gaps remain.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear motivation using counterexamples; excellent intuitive explanations of adversarial games.
- **Value**: ⭐⭐⭐⭐ Provides a more diagnostic framework for the learning-augmented algorithm community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Non-Clairvoyant Scheduling with Progress Bars](../../NeurIPS2025/learning_theory/non-clairvoyant_scheduling_with_progress_bars.md)
- [\[ICLR 2026\] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling](atlas_alibaba_dataset_and_benchmark_for_learning-augmented_scheduling.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICLR 2026\] Semi-Parametric Contextual Pricing with General Smoothness](semi-parametric_contextual_pricing_with_general_smoothness.md)
- [\[ICLR 2026\] Strong Correlations Induce Cause Only Predictions in Transformer Training](strong_correlations_induce_cause_only_predictions_in_transformer_training.md)

</div>

<!-- RELATED:END -->
