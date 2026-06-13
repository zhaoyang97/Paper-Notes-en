---
title: >-
  [Paper Note] LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams
description: >-
  [ICML 2026][AI Safety][Differential Privacy] LAPRAS utilizes a predictor to determine "which queries will arrive," splitting the online DP query stream into two categories: predicted and unpredicted. Predicted queries ar…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Linear Queries"
  - "Matrix Mechanism"
  - "Prediction Augmentation"
  - "Budget Allocation"
date: 2026-05-08
content_hash: d80b64b38a924a52
---

# LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams

**Conference**: ICML 2026  
**arXiv**: [2605.01960](https://arxiv.org/abs/2605.01960)  
**Code**: None  
**Area**: AI Security / Differential Privacy / Learning-Augmented Algorithms  
**Keywords**: Differential Privacy, Linear Queries, Matrix Mechanism, Prediction Augmentation, Budget Allocation

## TL;DR
LAPRAS utilizes a predictor to determine "which queries will arrive," splitting the online DP query stream into two categories: predicted and unpredicted. Predicted queries are released with low noise in one go using the offline optimal Matrix Mechanism, while unpredicted queries use Smooth Allocation to online-estimate the total count and smoothly distribute the budget based on observed "unpredicted query" positions. This approach nearly matches the offline optimal when predictions are accurate and degrades to the online baseline when they are poor.

## Background & Motivation
**Background**: Differential Privacy (DP) is the de facto standard for industrial-grade data analysis. In **offline** scenarios with a fixed workload $\mathrm{W}$, the Matrix Mechanism (MM) can design an optimal strategy matrix by exploiting correlations between queries to minimize total error. However, in **online** scenarios, queries $q_1, \dots, q_S$ arrive sequentially and must be answered immediately while adhering to a globally fixed total budget $(\varepsilon, \delta)$.

**Limitations of Prior Work**: It has been theoretically proven that online DP can be **exponentially** worse than offline DP—because the future is unknown, mechanisms must conservatively allocate a small budget to each query, resulting in high noise and low data utility. Existing solutions like Private Multiplicative Weights, Privacy Odometers/Filters, or CacheDP are either computationally expensive or "passively" cache historical results, failing to actively exploit the structure of future workloads.

**Key Challenge**: Online mechanisms only see the past, yet in actual industrial systems (e.g., SCOPE, SQL Server, Azure SQL), over 60% of query streams are periodic or repetitive, and over 90% of resources are consumed by a few templates. This **predictability** is a free prior, yet traditional DP algorithms cannot translate the guess "I expect these queries to arrive" into "I will pre-calculate their low-noise answers."

**Goal**: Design a **learning-augmented** online DP mechanism that satisfies: (i) approaching the utility of offline MM when predictions are accurate (high overlap); (ii) performing no worse than "independent Gaussian noise" when predictions are entirely wrong; (iii) strictly satisfying $(\varepsilon, \delta)$-DP; (iv) solving the core budget allocation problem of not knowing how many "bad queries" will aggregate in total.

**Key Insight**: The authors employ a seemingly simple but critical assumption—**query arrival order is uniformly random**. This transforms the "unknown total $B$" problem into a negative hypergeometric distribution problem, allowing for an unbiased estimation of $B$ based on the arrival positions of the first few bad queries.

**Core Idea**: Divide the stream using a prediction set $\mathrm{P}$; use offline MM for predicted queries and employ "stopping time unbiased estimation + smooth allocation" for online budget distribution for unpredicted ones.

## Method

### Overall Architecture
The global budget $\varepsilon$ is partitioned into four parts: $\varepsilon_{\text{MM}}$ for the Matrix Mechanism on the prediction set; $\varepsilon_{\text{badInit}}$ for "warming up" the first $T = \lceil \log^2 S \rceil$ bad queries; $\varepsilon_{\text{remBad}}$ for subsequent bad queries; and $\varepsilon_{\text{reserve}}$ as a safety buffer. Upon arrival, each query $q_t$ is classified: if $q_t \in \mathrm{P}$, the result is retrieved from the pre-computed MM output (**zero additional privacy cost** due to post-processing immunity); otherwise, it is treated as a bad query and processed online using the Analytic Gaussian Mechanism (AGM) with a budget $\varepsilon_b$ calculated via Smooth Allocation. If the reserve falls below a threshold $\varepsilon_{\min}$ at any point, the mechanism terminates early to prevent DP violations.

### Key Designs

1. **Unbiased Estimator $\widehat{B}$**:
    - **Function**: Infers the total count $B$ from observed arrival positions of bad queries without requiring prior knowledge.
    - **Mechanism**: Under the random order assumption, if the $b$-th bad query appears at position $n$, the estimate is $\widehat{B}(b) = S \cdot \frac{b-1}{n-1}$. The estimate $B_{\text{est}}$ is locked when $b = T = \lceil \log^2 S \rceil$. The paper proves $\mathbb{E}[\widehat{B}] = B$ (based on the negative hypergeometric distribution $Y \sim \mathrm{NHG}(S, G, T)$), with a variance upper bound of $O(B^2 / \log^2 S)$, converging fast enough to drive privacy budget allocation.
    - **Design Motivation**: The greatest pain point of online DP is the necessity to allocate budget based on the worst-case $S$ (where every query might be bad). With $\widehat{B}$, the budget can be allocated according to the actual number of bad queries, reducing expected noise from $O(S^2)$ to $O(B^2)$.

2. **Smooth Allocation**:
    - **Function**: Treats $\varepsilon_{\text{pool}} = \varepsilon_{\text{badInit}} + \varepsilon_{\text{remBad}}$ as a dynamic pool, recalculating the budget to spend for each arriving bad query.
    - **Mechanism**: When the $b$-th bad query arrives, the remaining bad queries are estimated as $\widehat{B}_{\text{rem},b} = \max(1, \widehat{B}(b) - b)$. The budget spent is $\varepsilon_b = \frac{\varepsilon_{\text{rem},b-1}}{\widehat{B}_{\text{rem},b} + 1}$ (the $+1$ prevents early overspending), and the pool is updated as $\varepsilon_{\text{rem},b} = \varepsilon_{\text{rem},b-1} - \varepsilon_b$. If bad queries are sparse, $\varepsilon_b$ increases; if dense, $\varepsilon_b$ decreases to prevent exhaustion. The paper proves $\sum_b \varepsilon_b < \varepsilon_{\text{pool}}$ (Lemma 4.5), ensuring DP budget conservation.
    - **Design Motivation**: Static Allocation locks $B_{\text{est}}$ once at the $T$-th bad query, remaining inaccurate if early density is anomalous. Smooth Allocation identifies and calibrates as it goes, making it more robust to prediction bias.

3. **Offline-Online Budget Splitting + Reserve Protection**:
    - **Function**: Provides low-noise release via MM for the prediction set and worst-case protection via geometric decay for the reserve.
    - **Mechanism**: For the prediction set $\mathrm{P}$, the offline optimal strategy matrix $\mathbf{A}$ is solved using $(\varepsilon_{\text{MM}}, \delta_i)$. The results $W \mathbf{A}^+ \mathcal{K}(\mathbf{A}, x)$ are reused at zero privacy cost when queries arrive. If the total number of bad queries exceeds $B_{\text{est}}$, each excess query spends $\varepsilon_{\text{reserve}} / 2$, and the reserve is halved. Termination occurs if it drops below $\varepsilon_{\min}$. Combined with basic composition and post-processing immunity, the system remains $(\varepsilon, \delta)$-DP (Theorem 4.6).
    - **Design Motivation**: MM's core advantage is utilizing workload correlation to thin out variance; high online utility is achieved by concentrating "likely queries" within it. Geometrically decaying the reserve ensures the mechanism never exceeds the budget even if the predictor fails entirely.

### Loss & Training
LAPRAS is an algorithm rather than a learned model, thus it has no training loss. Theoretical utility bounds (Section 4) are given as: $\sum_{q \in \mathcal{S}} \mathbb{E}[U_{\text{LAPRAS}}(q)^2] = O(\frac{B^2 \ln(1/\delta)}{\varepsilon^2}) + O(\sum_q \mathbb{E}[U_{\text{MM}}(q)^2])$, while ensuring $\le c \cdot \sum_q \mathbb{E}[U_{\text{Online}}(q)^2]$ (robustness).

## Key Experimental Results

### Main Results
On two real-world datasets, Adult and Gowalla, with $\varepsilon = 1.0$, LAPRAS was compared against OfflineMM and an independent Gaussian noise Online baseline:

| Dataset | Scenario | OfflineMM (MAE) | Online | LAPRAS (Ours) |
|--------|------|------|--------|------|
| Adult | High overlap ($\rho \approx 1$) | ~14 | 193.4 | 14.3 |
| Adult | Low overlap ($\rho \approx 0$) | — | 186.5 | 201.8 |
| Gowalla | High overlap | ~17 | 181.2 | 17.1 |
| Gowalla | Low overlap | — | 204.1 | 213.9 |

Under high overlap, MAE **drops by an order of magnitude**. Under low overlap, performance remains within the same magnitude as the Online baseline, empirically supporting the consistency-robustness trade-off.

### Ablation Study
Four budget allocation strategies (Table 1): equal / matrix-heavy / query-heavy / reserve-heavy.

| Strategy | $\varepsilon_{\text{MM}}$ | $\varepsilon_{\text{badInit}}$ | $\varepsilon_{\text{reserve}}$ | Use Case |
|------|----|----|----|----|
| equal | $\varepsilon/4$ | $\varepsilon/4$ | $\varepsilon/4$ | General |
| matrix-heavy | $\varepsilon/2$ | $\varepsilon/6$ | $\varepsilon/6$ | Accurate predictions |
| query-heavy | $\varepsilon/6$ | $\varepsilon/3$ | $\varepsilon/6$ | Poor predictions |
| reserve-heavy | $\varepsilon/6$ | $\varepsilon/6$ | $\varepsilon/2$ | Extreme uncertainty |

Experiments show matrix-heavy yields best utility under high overlap but degrades significantly under low overlap; query-heavy and reserve-heavy provide better protection under low overlap—acknowledging that real systems should be configured based on overlap priors.

### Key Findings
- Smooth Allocation is more robust than Static Allocation, especially when the early density of bad queries is anomalous or when $B < T$; Static simply wastes $\varepsilon_{\text{badInit}}$.
- The log-squared calibration window $T = \lceil \log^2 S \rceil$ is the sweet spot between estimation variance and budget waste; smaller $T$ causes variance to explode, while larger $T$ wastes the warm-up budget.
- Estimating $\widehat{B}$ itself consumes no extra budget—it only observes bad query **arrival positions** without querying the data.

## Highlights & Insights
- Formally translating the empirical fact that "industrial system query streams are predictable" into a provable acceleration for DP algorithms is a clean application of learning-augmented algorithm principles to the privacy domain.
- The use of a stopping time estimator combined with the random order assumption bypasses "unknown total $B$," a fundamental hurdle in online DP. This technique could be transferred to other online budget allocation problems (e.g., bandwidth allocation or privacy-preserving ad bidding).
- The consistency-robustness guarantee is elegant: achieving offline-optimal performance when predictions are correct without losing significant ground when they are wrong. This "icing on the cake if available, no loss if not" property is a critical deployment characteristic for ML4Sys work.

## Limitations & Future Work
- The random order assumption does not always hold in real workloads—periodic cron jobs introduce strong non-uniform arrivals, which may break the unbiasedness of $\widehat{B}$.
- The origin of the prediction set $\mathrm{P}$ is outside the scope of the paper, though the actual effect is determined by $\rho$; how to update $\mathrm{P}$ online or integrate multiple predictors warrants further study.
- Evaluation is currently limited to linear counting queries on two datasets; its effectiveness for complex queries like joins or selectivity estimation remains unknown.

## Related Work & Insights
- **vs Private Multiplicative Weights (PMW)**: PMW maintains a synthetic database to solve arbitrary queries, but update costs explode with domain size. LAPRAS maintains the low computational overhead of MM and focuses on the budget allocation challenge.
- **vs CacheDP**: CacheDP is reactive—it caches historical query answers and depends on redundancy with cold-start costs. LAPRAS is proactive—using predictions to pre-release future queries with low noise; the two are essentially orthogonal.
- **vs Privacy Odometers/Filters**: Odometers are descriptive accounting tools; LAPRAS provides a **prescriptive** allocation strategy detailing how to spend and when to stop.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying learning-augmented ideas to online DP is a fresh direction; the Smooth Allocation algorithm is clean.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets plus four strategies; cross-dataset generalization and validation on real-world workloads could be stronger.
- Writing Quality: ⭐⭐⭐⭐ Theory and algorithms are clear, and theorem proofs are complete.
- Value: ⭐⭐⭐⭐ Directly valuable for industrial DP system deployment, where repetitive query workloads are the norm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](../../NeurIPS2025/ai_safety/private_continual_counting_of_unbounded_streams.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](../../NeurIPS2025/ai_safety/nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[NeurIPS 2025\] Learning-Augmented Facility Location Mechanisms for Envy Ratio](../../NeurIPS2025/ai_safety/learning-augmented_facility_location_mechanisms_for_envy_ratio.md)

</div>

<!-- RELATED:END -->
