---
title: >-
  [Paper Note] Towards Optimal Robustness in Learning-Augmented Paging
description: >-
  [ICML 2026][Online Algorithms / Learning-Augmented Algorithms / Cache Paging][learning-augmented paging] This paper proposes a unified "Relative Prediction Budget" (RPB) perspective for randomized online paging with pred…
tags:
  - "ICML 2026"
  - "Online Algorithms / Learning-Augmented Algorithms / Cache Paging"
  - "learning-augmented paging"
  - "competitive ratio"
  - "OnlineMin"
  - "relative prediction budget"
  - "robustness"
date: 2026-05-08
content_hash: 291a3ba7d291c99a
---

# Towards Optimal Robustness in Learning-Augmented Paging

**Conference**: ICML 2026  
**arXiv**: [2606.01342](https://arxiv.org/abs/2606.01342)  
**Code**: None  
**Area**: Online Algorithms / Learning-Augmented Algorithms / Cache Paging  
**Keywords**: learning-augmented paging, competitive ratio, OnlineMin, relative prediction budget, robustness

## TL;DR
This paper proposes a unified "Relative Prediction Budget" (RPB) perspective for randomized online paging with predictions. Based on OnlineMin, the RPB-OnOPT framework is designed, pushing the provable robust competitive ratio from the existing $2H_k+O(1)$ to $H_k+O(1)$, which is near the information-theoretic lower bound, while maintaining 1-consistency.

## Background & Motivation

**Background**: Online paging is a classic prototype for online decision-making—given a cache of capacity $k$ and a sequence of page requests, hits incur no cost, while misses require eviction and fetching. The lower bound for the competitive ratio of any deterministic algorithm is $k$ (achieved by LRU), while the randomized lower bound is $H_k = \sum_{i=1}^{k}1/i \approx \ln k$, approximated by work-function-based algorithms like Equitable, K_Equitable, and OnlineMin. The Learning-augmented Algorithms (ALPS) approach superimposes a potentially inaccurate ML prediction (typically the next-arrival time) onto the sequence. The goal is to simultaneously achieve 1-consistency (approaching OPT when predictions are perfect) and bounded robustness (remaining close to the classic competitive ratio when predictions are arbitrarily poor).

**Limitations of Prior Work**: Robust upper bounds for algorithms based on the Marker framework, such as PredictiveMarker and LMarker, are stalled around $2H_k+O(1)$. While the Trust&Doubt and F&R series can achieve 1-consistency, their robustness constants deviate from the optimal $H_k$. In other words, existing solutions are twice as costly as OPT in the "worst case"—this is due to the inherent hard upper bound of the Marker mechanism and the decoupling of prediction budget from prediction quality.

**Key Challenge**: To achieve 1-consistency, one must "dare to use predictions," but to achieve $H_k$-level robustness, one must "dare to ignore predictions." Current algorithms either "over-trust" predictions at fixed thresholds (e.g., PredictiveMarker uses an $H_k$ threshold, magnifying errors to $4H_k$) or are "over-conservative" (e.g., LMarker uses a threshold of 1, missing many accurate predictions). Both lack a fine-grained scheduler that incorporates both prediction quality and the baseline robustness cost.

**Goal**: (i) Explicitly formalize the design essence of all current "robust-consistent" algorithms into a unified primitive; (ii) construct a framework on this primitive that provably reaches $H_k+O(1)$ robustness; (iii) maintain 1-consistency and demonstrate effectiveness on real-world workloads.

**Key Insight**: The authors observe that all robust learning-augmented paging algorithms essentially maintain a "non-negative budget $B_t$ relative to a robust baseline $\mathcal{A}$"—deviating from $\mathcal{A}$ to follow prediction suggestions is only allowed when $B_t > 0$. The difference lies in the granularity of "earning" and "spending" the budget. Simultaneously, online optimal algorithms like OnlineMin naturally maintain a work function layer structure of "valid configurations," providing a superior robust foundation for grafting predictions compared to Marker.

**Core Idea**: Attach the RPB bookkeeping to OnlineMin, linking budget earning directly to "past prediction effectiveness vs. the worst-case expected cost of online optimal." This achieves optimal consistency and near-optimal robustness for the first time.

## Method

### Overall Architecture
Input: An online request sequence $\sigma = \langle r_1, \dots, r_n \rangle$ and next-arrival time predictions for each step; Output: Cache eviction decisions. The algorithm maintains four components at each step: (1) current work function $\omega$ (represented compactly by $k+1$ layers $L_0, \dots, L_k$, where $L_0$ refers to outer pages and $L_i, i>0$ refers to support layers); (2) current cache configuration $C$ (always a valid configuration); (3) non-negative budget $B$; (4) evaluator variable $Y = U(\omega) = k - |R(\omega)|$, representing the number of unrevealed layers.

When a request arrives, $\omega$ is updated according to work function rules, followed by branching:

- **Hit**: No action taken.
- **Miss and $p \in L_0$ (outer support, maximum uncertainty)**: Evict according to prediction and reset the budget to a constant $\tau = O(1)$.
- **Miss and $p \in L_i, i > 0$ (lazy request)**: Identify a candidate set $V$ based on OM rules, then use a decision function $\mathcal{J}_{RPB}(Y, \omega)$ to decide whether to increment the budget by 1. If $B > 0$, spend 1 unit and evict from $V$ according to prediction; otherwise, fall back to OM's priority-based eviction.

After processing the request, update $Y \leftarrow \mathcal{U}_{RPB}(\omega)$ for the next step. This process is unified in the RPB-OnOPT framework, where instantiating different $\mathcal{J}_{RPB}$ and $\mathcal{U}_{RPB}$ yields different robust-consistency trade-offs.

### Key Designs

1. **Relative Prediction Budget (RPB)**:
    - **Function**: Uses "spendable budget $B_t \ge 0$ relative to a robust baseline $\mathcal{A}$" as a unified primitive to characterize and improve prediction usage intensity in existing algorithms.
    - **Mechanism**: Reinterprets BlindOracle&LRU, PredictiveMarker, LMarker, and F&R as "RPB algorithms with different earning rules." For example, PredictiveMarker earns $H_k$ units at once on each clean request, while LMarker earns only 1 unit; F&R earns 1 unit when its cost matches Belady. This paper points out these rules lack fine-grained coupling with prediction validity, locking the robustness constant. RPB requires budget earning to be tied to both "past algorithm performance" and the "baseline's expected cost under worst-case lazy adversaries."
    - **Design Motivation**: To provide a unified metric that explains why previous work stalled at $2H_k$ and points towards achieving $H_k+O(1)$.

2. **RPB-OnOPT Framework (Online Optimal Foundation + Adaptive Budget)**:
    - **Function**: Implements RPB on online optimal algorithms like OnOPT (e.g., OnlineMin) instead of Marker to unlock smaller robustness constants.
    - **Mechanism**: (i) For requests in $L_0$, evict via prediction and reset budget to $\tau=O(1)$. Theorem 5.1 proves this is the minimum prediction usage to achieve 1-consistency. (ii) For requests in $L_i, i>0$, use OM's candidate set rule $V = C \cap \bigcup_{l=1}^{z} L_l$ to ensure evictions stay within a valid configuration before deciding whether to follow predictions. OnOPT more effectively algorithmizes the maintenance of valid configurations, providing a stable foundation for $H_k$-level potential function analysis.
    - **Design Motivation**: Marker's mechanism itself limits the lower bound to $2H_k+O(1)$. By switching to OnOPT, the baseline competitive ratio drops from $2H_k-1$ to $H_k$. Combining this with an $O(1)$ additive term approaches the information-theoretic lower bound.

3. **RPB-OM Instantiation and Decision Function $\mathcal{J}_{RPB}$**:
    - **Function**: Provides RPB-OM, a specific instance of RPB-OnOPT, proving it is 1-consistent and $(H_k+O(1))$-robust.
    - **Mechanism**: Replaces custom priorities in OnOPT with OM's randomized priorities. The evaluator $Y = U(\omega) = k-|R(\omega)|$ measures uncertainty; the decision function $\mathcal{J}_{RPB}(Y, \omega) = (U(\omega) \le (Y+2)/e - 2)$ increments the budget only when uncertainty decreases significantly, preventing the algorithm from being misled by adversarial sequences. Robustness analysis relies on new properties: potential function increases by at most $H_k-1$ on $L_0$ requests and decreases by at least $1/(U(\omega)+1)$ on lazy adversary requests. These together keep RPB-OM's extra cost within an additive constant.
    - **Design Motivation**: Aligning the budget earning trigger with the exponential decay structure of OM's potential function ensures that even with poor predictions, the cumulative deviation from OM is only an $O(1)$ additive term.

### Loss & Training
This paper does not involve ML training; predictions are treated as a black box. All "training" costs are explicitly accounted for in RPB's constant $\tau$, decision thresholds, and potential function proofs.

## Key Experimental Results

### Main Results
The authors compared major learning-augmented paging algorithms on LIRS and SPC1 traces, using the ratio of cost to OPT (cost/OPT) as the metric.

| Algorithm | Perfect Pred. | Med. Error | Adv. Pred. | Robustness Bound |
| :--- | :--- | :--- | :--- | :--- |
| BlindOracle | 1.00 | Spikes | Unbounded | $\infty$ |
| BlindOracle & LRU | 1.05 | 1.3-1.5 | $\le 2k$ | $2k$ |
| PredictiveMarker | $\sim$2 | 1.4-1.8 | $\le 4H_k$ | $4H_k$ |
| LMarker | $\sim$2 | 1.4-1.6 | $\le 2H_k+4$ | $2H_k+4$ |
| **RPB-OM (Ours)** | **1.00** | **1.1-1.3** | **$\le H_k+O(1)$** | **$H_k+O(1)$** |

### Ablation Study
| Configuration | Avg. cost/OPT | Description |
| :--- | :--- | :--- |
| RPB-OM Full | 1.10-1.30 | Budget linked to $U(\omega)$ |
| RPB-OM, $\mathcal{J}_{RPB} \equiv \text{true}$ | $\sim$1.5 | Aggressive strategy, worse robustness |
| RPB-OM, $\mathcal{J}_{RPB} \equiv \text{false}$ | $\sim$1.4 | Pure OM, 1-consistency lost |
| RPB on Marker Base | $\sim$1.6 | Reverts to $2H_k$ bound |

### Key Findings
- Switching the robust base from Marker to OnOPT is the fundamental reason for pushing the bound from $2H_k$ to $H_k$.
- Using $U(\omega)$ for $\mathcal{J}_{RPB}$ instead of "total past cost" is critical: the former aligns with the potential function decay rate $1/(U+1)$ to yield an additive constant, while the latter introduces a multiplicative factor of 2.
- On real traces, RPB-OM reduces page faults by 10%-20% compared to LMarker/PredictiveMarker even with imperfect predictions, showing that "earning budget relative to a robust baseline" utilizes medium-accuracy predictions more effectively.

## Highlights & Insights
- **Leverage of a Unified View**: RPB serves as a "mirror," mapping diverse algorithms onto a single "budget bookkeeping" logic to reveal who is over-trusting or over-conservative.
- **"Base-Swapping" > "Threshold Tuning"**: While previous works tuned thresholds on Marker, they were capped at $2H_k+O(1)$. This paper proves that approaching $H_k$ requires switching to an online optimal foundation like OnOPT. This observation is valuable for the broader ALPS design philosophy (e.g., $k$-server, metrical task systems).
- **Potential Function Properties**: Corollary 5.5 and Lemma 5.6 provide precise bounds on OM potential changes, serving as general tools for analyzing any learning-augmented variant on OM.

## Limitations & Future Work
- The robust bound is $H_k+O(1)$ rather than $H_k$; removing the constant is equivalent to using no prediction.
- The form of $\mathcal{J}_{RPB}$ remains relatively simple; aligning it with OM's exact step-wise expected cost would increase complexity.
- Experiments focused on next-arrival time; more realistic noisy or streaming predictions were not explored.
- The framework is limited to "randomized paging" and does not address deterministic settings or hierarchical caching.

## Related Work & Insights
- **vs. BlindOracle&LRU (Wei, 2020)**: They switch globally between two algorithms with coarse granularity and $2k$ worst-case; RPB switches fine-gradually within OnOPT, tightening robustness to $H_k+O(1)$.
- **vs. PredictiveMarker / LMarker (Lykouris-Vassilvtiskii 2018; Rohatgi 2020)**: They use fixed thresholds ( $H_k$ and 1) on Marker, hitting limits of $4H_k$ and $2H_k+4$. This work proves the Marker base itself is the bottleneck.
- **vs. F&R (Sadek & Elias, 2024)**: They achieved $\mathcal{O}(\log k)$ robustness, but with sub-optimal leading constants; this work reduces the constant to 1.
- **Insight**: For other ALPS problems, one should ask: "Is our robust base already the online optimal algorithm for this problem?" If not, swapping the base often yields higher returns than tuning thresholds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve 1-consistency and near-optimal $H_k+O(1)$-robustness with the RPB primitive.
- Experimental Thoroughness: ⭐⭐⭐ Solid on traces, but prediction noise types are narrow.
- Writing Quality: ⭐⭐⭐⭐ High readability, framing complex work function analysis as a clear narrative.
- Value: ⭐⭐⭐⭐⭐ Provides optimal robust results and a guiding perspective on "base-swapping" for the ALPS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/learning_theory/learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/learning_theory/the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2026\] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](../../ICML2025/others/near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](optimal_regularization_for_performative_learning.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[ICML 2026\] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)

</div>

<!-- RELATED:END -->
