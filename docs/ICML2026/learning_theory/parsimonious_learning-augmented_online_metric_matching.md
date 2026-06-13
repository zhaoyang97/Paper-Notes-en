---
title: >-
  [Paper Note] Parsimonious Learning-Augmented Online Metric Matching
description: >-
  [ICML 2026][Online Optimization / Learning-Augmented Algorithms / Online Metric Matching][Online Metric Matching] This paper resolves an open question posed by Im et al. (2022) by extending "action-based prediction" for…
tags:
  - "ICML 2026"
  - "Online Optimization / Learning-Augmented Algorithms / Online Metric Matching"
  - "Online Metric Matching"
  - "Learning-augmented algorithms"
  - "parsimonious prediction"
  - "Follow-the-Prediction"
  - "competitive ratio"
date: 2026-05-08
content_hash: 549869dde30e6d4d
---

# Parsimonious Learning-Augmented Online Metric Matching

**Conference**: ICML 2026  
**arXiv**: [2605.26886](https://arxiv.org/abs/2605.26886)  
**Code**: None (Theoretical paper with numerical experiments)  
**Area**: Online Optimization / Learning-Augmented Algorithms / Online Metric Matching  
**Keywords**: Online Metric Matching, Learning-augmented algorithms, parsimonious prediction, Follow-the-Prediction, competitive ratio

## TL;DR
This paper resolves an open question posed by Im et al. (2022) by extending "action-based prediction" for Online Metric Matching (OMM) into a "parsimonious prediction" framework—where predictions are issued expensively every $k$ steps. Using a Follow-the-Prediction (FtP) framework combined with a meta-algorithm that automatically generates "virtual predictions," the authors provide deterministic and randomized competitive ratio upper bounds that essentially match known lower bounds.

## Background & Motivation

**Background**: Online Metric Matching (OMM) is a classic problem in online optimization: $n$ server locations are known upfront, and $n$ requests arrive sequentially. Each request must be immediately and irrevocably matched to an unoccupied server to minimize the total matching distance. Thirty years ago, Kalyanasundaram–Pruhs and Khuller established a $(2n-1)$ deterministic competitive ratio. Bansal et al. (2014) pushed randomized algorithms to $O(\log^2 n)$, both of which are tight within constant factors.

**Limitations of Prior Work**: The gap between classic upper and lower bounds primarily stems from having "zero knowledge of the future." The learning-augmented framework seeks to break worst-case barriers using predictions. The Follow-the-Prediction (FtP) algorithm by Antoniadis et al. (2023b) requests an action prediction $P_t$ from an oracle in every round, guaranteeing $9 \cdot \min\{\text{cost}(\text{OPT}) + 2\eta, (2n-1)\text{cost}(\text{OPT})\}$. However, generating predictions often requires running a large model, making round-by-round calls prohibitively expensive.

**Key Challenge**: While high-quality predictions can significantly narrow the gap between bounds, each query incurs inference costs. The challenge lies in extracting maximum value from sparse or restricted predictions. This concept of "parsimonious" prediction was introduced by Im et al. (2022) for caching; this paper generalizes it to OMM.

**Goal**: Design OMM algorithms and establish competitive ratio bounds under two "parsimonious" mechanisms: (i) well-separated queries (querying every $k$ rounds) and (ii) bounded budget (total queries limited to $B$).

**Key Insight**: Since FtP requires a prediction $P_t$ for every round, can the algorithm synthesize "virtual predictions" between two real queries? If this synthesizer ensures the quality of intermediate matches, the FtP analysis can be extended.

**Core Idea**: Define two new algorithmic properties—*adherence* (the "set distance" of intermediate matches is close to the optimal maximum matching) and *strong competitiveness* (the intermediate matching cost is close to the optimal matching)—and prove that any subroutine satisfying both can "interpolate" usable virtual predictions.

## Method

### Overall Architecture

The overall algorithm is a "parsimonious" wrapper for FtP. The timeline is divided into phases of length $k$. In the first round of each phase, the oracle provides a real prediction $\widehat P = P_{ik}$. For the remaining $k-1$ rounds, an auxiliary subroutine $\mathcal A$ runs on the "remaining servers $S \setminus \widehat P$ and requests arriving within the phase." The set of servers $\widehat S$ currently matched by $\mathcal A$ is combined with $\widehat P$ to form a "virtual prediction" $P_t = \widehat P \cup \widehat S$. This sequence $\{P_t\}_t$ is then fed into the standard FtP. Intuitively, $\widehat P$ provides the "global direction" while $\mathcal A$ refines local details using real-time arrival data.

### Key Designs

1.  **Adherence + Strong Competitiveness: Characterizing Subroutines for Virtual Predictions**:
    *   **Function**: Defines two structural properties required for intermediate matches so that the resulting virtual predictions can be absorbed by FtP analysis.
    *   **Mechanism**: For an algorithm $\mathcal{A}$ and its instantaneous match $\mathcal{A}_t$ (matching servers $S_t$ to requests $R_t$). $\mathcal{A}$ is $\gamma$-adherent if for any $t$, $\mathsf{dist}(S_t, R_t) \le \gamma \cdot \min_{M \in \mathcal M_t} \mathsf{cost}(M)$, where $\mathcal M_t$ is the set of maximum matchings for $R_t$. $\mathcal{A}$ is strongly $\rho$-competitive if $\mathbb E[\mathsf{cost}(\mathcal A_t)] \le \rho(t) \cdot \min_{M \in \mathcal M_t} \mathsf{cost}(M)$. Adherence ensures the "state" is close to optimal set-wise, while strong competitiveness ensures the cumulative cost is close to optimal.
    *   **Design Motivation**: Conventional OMM only evaluates the final output ($t=n$). Many classic algorithms (Kalyanasundaram–Pruhs, Nayyar–Raghvendra, Bansal, etc.) happen to maintain good partial matches; these properties formalize this "process-friendliness" for plug-and-play use.

2.  **Parsimonious Meta-Algorithm for FtP (Algorithm 2 + Algorithm 1)**:
    *   **Function**: Transforms "prediction per round" into "one real prediction every $k$ rounds + phase-wise subroutine synthesis" with a unified cost bound.
    *   **Mechanism**: For each phase, a new instance of $\mathcal{A}$ is reset on $S \setminus \widehat P$. In non-query rounds, requests are fed to $\mathcal{A}$ to obtain $\widehat S$, and $P_t = \widehat P \cup \widehat S$ is constructed for FtP. In query rounds, $\widehat P$ is updated and $\mathcal{A}$ is reset. This ensures $|P_t| = t$ and $P_t \subseteq S$. The analysis uses Lemma 3.2 (FtP cost bounded by $\sum_t \mathsf{dist}(P_t, P_{t-1} \cup \{r_t\})$). The inter-phase gaps are controlled via adherence and triangle inequality (Lemma 3.5), while intra-phase costs are controlled by strong competitiveness (Lemma 3.4), resulting in the unified bound in Theorem 3.1: $(1+\gamma+\rho(k-1))\,\text{cost}(\text{OPT}) + (2+\gamma+\rho(k-1))\,\eta(Q)$.

3.  **Lower Bounds: Hard Instances on Star Metrics**:
    *   **Function**: Proves that the upper bounds in Theorem 1.1 are essentially optimal for deterministic algorithms, even with perfect predictions.
    *   **Mechanism**: An adversarial sequence is constructed on a star metric with $n$ leaves. Any deterministic algorithm with at most $B$ queries suffers a loss of at least $\frac{2n}{B+1}-1$ (Theorem 4.1). Under the well-separated mechanism, the loss is at least $2k-1$ (Theorem 4.2). For randomized algorithms, the lower bounds are $\Omega(\log k)$ (well-separated) and $1 + o\!\left(\frac{\log(n/B)}{B}\right)$ (bounded budget) (Theorems 4.3–4.4).

### Loss & Training

To handle adversarial or erroneous predictions, the authors utilize the combination trick from Fiat–Rabani–Ravid: combining "Ours" (which relies on predictions) with a "no-prediction" baseline (a deterministic $(2n-1)$-competitive algorithm or Greedy) using a 9-fold multiplicative factor (Theorem 2.3). This ensures global robustness and forms the basis for the `Comb-Comp` and `Comb-Greedy` variants in the experiments.

## Key Experimental Results

### Main Results: Parsimonious Gains on Synthetic & Real Data (Perfect Prediction, $k \in [1, 20]$)

| Instance Class | Metric | Evaluation Goal | Ours Performance |
| :--- | :--- | :--- | :--- |
| Line | 1D Absolute Diff | Competitive ratio vs. $k$ | Increases monotonically with $k$ but stays well below Comp/Greedy |
| Plane | 2D Euclidean | Competitive ratio vs. $k$ | Superior to other baselines; Comb series degrades slower |
| Taxi (Chicago 2013–2023) | Manhattan | Real car-hailing data | Ours leads throughout; Comb-Greedy occasionally surpassed by Greedy |

*Note: All instances used $n=100$ servers and $n=100$ requests. Results are averages over 100 independent trials. The case $k=1$ is verified as equivalent to Antoniadis et al. (2023b).*

### Ablation Study: Robustness under Noise Radius $r$

| Configuration | Near Perfect Prediction | High Noise | Explanation |
| :--- | :--- | :--- | :--- |
| Ours ($k=1$, FtP equiv.) | Near optimal | Sharpest degradation | Frequent prediction use amplifies noise |
| Ours (large $k$) | Slightly worse than $k=1$ | Slower degradation slope | Lower frequency reduces sensitivity to single errors |
| Comb-Comp / Comb-Greedy | Close to Ours | Most robust | Baseline algorithms provide a safety net |
| Comp / Greedy | Equal or worse than Ours | Noise-independent | Does not utilize predictions |

### Key Findings

*   The true cost of parsimony under perfect prediction is an increase in the competitive ratio from $O(1)$ to $\Theta(k)$. However, this reduces prediction calls from $n$ to $\lceil n/k \rceil$, which is highly advantageous for expensive models.
*   The randomized upper bound of $O(\log n \cdot \log k)$ reveals a decomposition: $\log n$ comes from HST embedding distortion, and $\log k$ comes from the randomized matching of the phase subroutine on $k-1$ requests.
*   The combination algorithm occasionally underperforms the standalone baseline (e.g., Comb-Greedy vs. Greedy on Plane/Taxi), suggesting the factor of 9 is not tight and switching overhead might exceed gains when the gap is small.

## Highlights & Insights

*   Explicitly defining *adherence* and *strong competitiveness* provides both a library of usable subroutines and a template for porting the parsimonious framework to other online problems like $k$-server or metrical task systems.
*   The concept of "using an algorithm as a virtual predictor" is elegant: a classic online algorithm is essentially the best guess in the absence of future information. Combining it with real predictions creates a monotonically interpretable behavior.
*   The alignment of upper and lower bounds is notable; many learning-augmented papers only provide upper bounds. By generalizing the star metric adversarial construction, the $2k-1$ factor is shown to be nearly tight for deterministic cases.

## Limitations & Future Work

*   The randomized upper bound still contains a $\log n$ factor, whereas the lower bound is only $\Omega(\log k)$. Determining if $\log n$ is avoidable is an open problem.
*   The constant factor of 9 in the combination algorithm is large. Empirical evidence shows robust-consistent trade-offs are not yet tight.
*   The current prediction model is action-based. Whether weaker, cheaper prediction semantics (e.g., predicting only the next step or priority) can be utilized remains to be explored.
*   Evaluation is limited to synthetic and one real-world dataset (Chicago Taxi). Complex resource allocation scenarios like CDNs or ad auctions need further validation.

## Related Work & Insights

*   **vs Antoniadis et al. (2023b) (FtP)**: This work is a strict extension; it recovers FtP at $k=1$ and provides a form usable for non-dense prediction scenarios.
*   **vs Im et al. (2022) (Parsimonious Caching)**: Applies the parsimonious framework to a different domain. While caching states are "sets," OMM states are "matchings," requiring new process-based metrics.
*   **vs Sadek & Eliás (2024) (Parsimonious MTS/Caching)**: Follows the well-separated query format, but OMM cannot reuse predictions as directly as MTS, necessitating the explicit construction of virtual predictions.
*   **vs Bansal et al. (2014) (Randomized OMM)**: This work leverages Bansal’s 2-HST algorithm as a strongly $O(\log t)$-competitive subroutine, demonstrating how to transform existing online algorithms into parsimonious components.

## Rating
- **Novelty**: Pending
- **Experimental Thoroughness**: Pending
- **Writing Quality**: Pending
- **Value**: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/learning_theory/learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICML 2026\] Realizable Bayes-Consistency for General Metric Losses](realizable_bayes-consistency_for_general_metric_losses.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)

</div>

<!-- RELATED:END -->
