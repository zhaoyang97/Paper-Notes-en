---
title: >-
  [Paper Note] LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams
description: >-
  [ICML 2026][AI Safety][Differential Privacy] LAPRAS uses a predictor for "which queries will arrive" to split an online DP query stream into in-prediction and out-of-prediction categories. For in-prediction queries…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Linear Query"
  - "Matrix Mechanism"
  - "Prediction-Augmented"
  - "Budget Allocation"
date: 2026-05-08
content_hash: 1dea79c4abbce48a
---

# LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams

**Conference**: ICML 2026  
**arXiv**: [2605.01960](https://arxiv.org/abs/2605.01960)  
**Code**: None  
**Area**: AI Safety / Differential Privacy / Learning-Augmented Algorithms  
**Keywords**: Differential Privacy, Linear Query, Matrix Mechanism, Prediction-Augmented, Budget Allocation

## TL;DR
LAPRAS uses a predictor for "which queries will arrive" to split an online DP query stream into in-prediction and out-of-prediction categories. For in-prediction queries, it releases answers with low noise in one shot using the offline-optimal Matrix Mechanism. For out-of-prediction queries, it applies Smooth Allocation, estimating the total number of "unpredicted queries" online based on observed positions and allocating budget smoothly. When predictions are accurate, utility nearly matches the offline optimum; when predictions are poor, performance degrades gracefully to the online baseline.

## Background & Motivation
**Background**: Differential Privacy (DP) is the de facto standard for industrial-scale data analysis. In the **offline** setting, given a fixed workload $\mathrm{W}$, the Matrix Mechanism (MM) leverages query correlations to design an optimal strategy matrix, minimizing total error. In the **online** setting, queries $q_1, \dots, q_S$ arrive sequentially and must be answered immediately, with a fixed global budget $(\varepsilon, \delta)$.

**Limitations of Prior Work**: Theory shows that online DP can be **exponentially worse** than offline DP—since future queries are unknown, mechanisms must conservatively allocate less budget per query, increasing noise and rendering data nearly unusable. Existing solutions like Private Multiplicative Weights, Privacy Odometers/Filters, and CacheDP are either computationally expensive or "passively" cache historical results, failing to proactively exploit future workload structure.

**Key Challenge**: Online mechanisms only see the past, but in real-world systems (SCOPE, SQL Server, Azure SQL), over 60% of query streams are periodic, and over 90% of resources come from a few templates. This **predictability** is a free prior, but traditional DP algorithms cannot translate "I guess these queries are coming" into "I precompute their low-noise answers."

**Goal**: Design a **learning-augmented** online DP mechanism, given a prediction set $\mathrm{P}$, that satisfies: (i) near-offline MM utility when predictions are accurate (high overlap); (ii) no worse than "independent Gaussian noise" when predictions are wrong; (iii) always satisfies $(\varepsilon, \delta)$-DP; (iv) solves the core budget allocation problem—how to allocate budget without knowing the total number of bad queries.

**Key Insight**: The authors make a seemingly obvious but crucial assumption—**query arrival order is uniformly random**. This transforms the "unknown total $B$" problem into a negative hypergeometric distribution problem, enabling unbiased estimation of $B$ from the arrival positions of the first few bad queries.

**Core Idea**: Use the prediction set to split the stream; in-prediction queries use offline MM, out-of-prediction queries use "stopping-time unbiased estimation + smooth allocation" for online budget allocation.

## Method

### Overall Architecture
The global budget $\varepsilon$ is divided into four parts: $\varepsilon_{\text{MM}}$ for the Matrix Mechanism on the prediction set; $\varepsilon_{\text{badInit}}$ for the first $T = \lceil \log^2 S \rceil$ bad queries as "warm-up"; $\varepsilon_{\text{remBad}}$ for subsequent bad queries; $\varepsilon_{\text{reserve}}$ as a safety buffer. As each query $q_t$ arrives, it is classified: if $q_t \in \mathrm{P}$, the precomputed MM result is used (**zero additional privacy cost** due to post-processing immunity); otherwise, it is treated as a bad query and processed online. The Smooth Allocation computes $\varepsilon_b$, and the Analytic Gaussian Mechanism (AGM) adds noise to the output. If the reserve drops below threshold $\varepsilon_{\min}$, the mechanism terminates early to avoid DP violation.

### Key Designs

1. **Stopping-Time Unbiased Estimator $\widehat{B}$**:

    - **Function**: Infers the total number $B$ of bad queries from observed arrival positions, without prior knowledge.
    - **Mechanism**: Under the random order assumption, if the first $b$ bad queries appear at position $n$, estimate $\widehat{B}(b) = S \cdot \frac{b-1}{n-1}$. When $b = T = \lceil \log^2 S \rceil$, $B_{\text{est}}$ is fixed. The paper proves $\mathbb{E}[\widehat{B}] = B$ (based on negative hypergeometric distribution $Y \sim \mathrm{NHG}(S, G, T)$), with variance bounded by $O(B^2 / \log^2 S)$, and concentration fast enough for budget allocation.
    - **Design Motivation**: The main pain point in online DP is having to allocate budget for the worst case $S$ (every query could be bad); with $\widehat{B}$, budget can be allocated according to the actual number of bad queries, reducing expected noise from $O(S^2)$ to $O(B^2)$.

2. **Smooth Allocation**:

    - **Function**: Treats $\varepsilon_{\text{pool}} = \varepsilon_{\text{badInit}} + \varepsilon_{\text{remBad}}$ as a dynamic pool, recalculating the budget for each arriving bad query.
    - **Mechanism**: For the $b$-th bad query, estimate remaining bad queries as $\widehat{B}_{\text{rem},b} = \max(1, \widehat{B}(b) - b)$, allocate $\varepsilon_b = \frac{\varepsilon_{\text{rem},b-1}}{\widehat{B}_{\text{rem},b} + 1}$ ($+1$ prevents early overspending), and update the pool $\varepsilon_{\text{rem},b} = \varepsilon_{\text{rem},b-1} - \varepsilon_b$. When bad queries are sparse, $\varepsilon_b$ increases (spend more per query); when dense, $\varepsilon_b$ decreases (save to avoid exhaustion). The paper proves $\sum_b \varepsilon_b < \varepsilon_{\text{pool}}$ (Lemma 4.5), ensuring DP budget conservation.
    - **Design Motivation**: Static Allocation locks $B_{\text{est}}$ at the $T$-th bad query, and if early bad query density is abnormal, it remains inaccurate; Smooth Allocation continuously adjusts, making it more robust to prediction errors.

3. **Offline-Online Budget Split + Reserve Protection**:

    - **Function**: Uses MM for low-noise release on the prediction set and geometric decay on the reserve for worst-case protection.
    - **Mechanism**: The prediction set $\mathrm{P}$ uses Matrix Mechanism with $(\varepsilon_{\text{MM}}, \delta_i)$ to solve for the optimal strategy matrix $\mathbf{A}$ offline; results $W \mathbf{A}^+ \mathcal{K}(\mathbf{A}, x)$ are reused at query time with zero privacy cost. If total bad queries exceed $B_{\text{est}}$, each excess query spends $\varepsilon_{\text{reserve}} / 2$, then reserve is halved; if it drops below $\varepsilon_{\min}$, the mechanism stops. Combined with basic composition and post-processing immunity, the overall mechanism remains $(\varepsilon, \delta)$-DP (Theorem 4.6).
    - **Design Motivation**: The core advantage of MM is leveraging workload correlation to reduce variance; only by concentrating "likely queries" in MM can high utility be achieved online. Geometric decay in reserve ensures that even if all predictions are wrong, the mechanism never overspends the budget.

### Loss & Training
LAPRAS is an algorithm, not a learning model, so there is no training loss. Theoretical utility bound (Section 4): $\sum_{q \in \mathcal{S}} \mathbb{E}[U_{\text{LAPRAS}}(q)^2] = O(\frac{B^2 \ln(1/\delta)}{\varepsilon^2}) + O(\sum_q \mathbb{E}[U_{\text{MM}}(q)^2])$, and guarantees $\le c \cdot \sum_q \mathbb{E}[U_{\text{Online}}(q)^2]$ (robustness).

## Key Experimental Results

### Main Results
On two real datasets, Adult and Gowalla, with $\varepsilon = 1.0$, compared to OfflineMM and independent Gaussian noise Online baseline:

| Dataset | Scenario | OfflineMM (MAE) | Online | LAPRAS (Ours) |
|---------|----------|-----------------|--------|--------------|
| Adult | High overlap ($\rho \approx 1$) | ~14 | 193.4 | 14.3 |
| Adult | Low overlap ($\rho \approx 0$) | — | 186.5 | 201.8 |
| Gowalla | High overlap | ~17 | 181.2 | 17.1 |
| Gowalla | Low overlap | — | 204.1 | 213.9 |

With high overlap, MAE **drops by an order of magnitude**; with low overlap, performance is on par with Online, empirically supporting the consistency-robustness tradeoff.

### Ablation Study
Four budget allocation strategies (Table 1): equal / matrix-heavy / query-heavy / reserve-heavy.

| Strategy | $\varepsilon_{\text{MM}}$ | $\varepsilon_{\text{badInit}}$ | $\varepsilon_{\text{reserve}}$ | Applicable Scenario |
|----------|---------------------------|-------------------------------|-------------------------------|--------------------|
| equal | $\varepsilon/4$ | $\varepsilon/4$ | $\varepsilon/4$ | General |
| matrix-heavy | $\varepsilon/2$ | $\varepsilon/6$ | $\varepsilon/6$ | Accurate prediction |
| query-heavy | $\varepsilon/6$ | $\varepsilon/3$ | $\varepsilon/6$ | Poor prediction |
| reserve-heavy | $\varepsilon/6$ | $\varepsilon/6$ | $\varepsilon/2$ | Extreme uncertainty |

Experiments show matrix-heavy yields best utility under high overlap, but degrades severely under low overlap; query-heavy and reserve-heavy provide better protection under low overlap—implicitly acknowledging that real systems should tune configuration based on overlap prior.

### Key Findings
- Smooth Allocation is more robust than Static Allocation, especially when early bad query density is abnormal or $B < T$; Static wastes $\varepsilon_{\text{badInit}}$ directly.
- $T = \lceil \log^2 S \rceil$ (log-squared calibration window) is the sweet spot for balancing estimation variance and budget waste; smaller $T$ leads to explosive variance, larger $T$ wastes warm-up budget.
- Estimating $\widehat{B}$ itself consumes no extra budget—it only uses the **arrival positions** of bad queries, not the data.

## Highlights & Insights
- Formally translates the empirical fact that "industrial query streams are predictable" into provable DP algorithmic acceleration—a clean application of learning-augmented algorithms in privacy.
- Combines stopping-time estimators with the random order assumption to circumvent the core online DP challenge of "unknown total $B$"; this technique can transfer to other online budget allocation problems (e.g., bandwidth allocation, privacy in ad auctions).
- The dual guarantee of consistency and robustness is elegant: achieves offline optimality when predictions are accurate, and does not degrade when predictions are wrong. This "graceful improvement, no loss otherwise" property is key for ML4Sys deployment.

## Limitations & Future Work
- The random order assumption does not always hold in real workloads—periodic cron jobs can cause strong non-uniform arrivals, potentially breaking the unbiasedness of $\widehat{B}$.
- How to construct the prediction set $\mathrm{P}$ is out of scope, but actual performance is directly determined by $\rho$; how to update $\mathrm{P}$ online and combine multiple predictors are promising directions for future research.
- Evaluation is currently limited to linear counting queries and two datasets; effectiveness for more complex queries (joins, selectivity estimation) remains unknown.

## Related Work & Insights
- **vs Private Multiplicative Weights (PMW)**: PMW maintains a synthetic database to answer arbitrary queries but update cost explodes with domain size; LAPRAS retains MM's low computational cost and focuses on the budget allocation challenge.
- **vs CacheDP**: CacheDP is reactive—caches answers to historical queries, relying on redundancy and incurring cold-start costs; LAPRAS is proactive—uses predictions to pre-release low-noise answers for future queries. The two are fundamentally orthogonal.
- **vs Privacy Odometers/Filters**: Odometers are descriptive privacy accounting tools; LAPRAS provides a **prescriptive** allocation strategy, telling you how to spend and when to stop.

## Rating
- Novelty: ⭐⭐⭐⭐ Connecting learning-augmented ideas to online DP is a new direction; Smooth Allocation is cleanly designed.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets + four strategies; cross-dataset generalization and real workload validation could be stronger.
- Writing Quality: ⭐⭐⭐⭐ Theory and algorithms are clear, with complete proofs.
- Value: ⭐⭐⭐⭐ Directly valuable for industrial DP system deployment, as repetitive query workloads are the norm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](../../NeurIPS2025/ai_safety/private_continual_counting_of_unbounded_streams.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](../../NeurIPS2025/ai_safety/nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Learning-Augmented Facility Location Mechanisms for Envy Ratio](../../NeurIPS2025/ai_safety/learning-augmented_facility_location_mechanisms_for_envy_ratio.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)

</div>

<!-- RELATED:END -->
