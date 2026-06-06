---
title: >-
  [Paper Note] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation
description: >-
  [ICML 2026][Optimization][Federated Learning] Addressing the issue where "Byzantine clients temporarily form a majority in sampled clients" under partial participation collapses existing robust aggregators…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Federated Learning"
  - "Byzantine Robustness"
  - "Partial Participation"
  - "Delayed Momentum"
  - "Robust Aggregation"
date: 2026-05-08
content_hash: 3481d3fbfb598493
---

# Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation

**Conference**: ICML 2026  
**arXiv**: [2509.02970](https://arxiv.org/abs/2509.02970)  
**Code**: Not yet public  
**Area**: Optimization / Federated Learning  
**Keywords**: Federated Learning, Byzantine Robustness, Partial Participation, Delayed Momentum, Robust Aggregation

## TL;DR
Addressing the issue where "Byzantine clients temporarily form a majority in sampled clients" under partial participation collapses existing robust aggregators, this paper proposes the Delayed Momentum Aggregation principle. The server feeds the current round's new momentum along with the most recent cached momentum from unsampled clients into a robust aggregator. This extends the global Byzantine ratio $\delta < 1/2$ to every aggregation round. Based on this, the DeMoA optimizer is designed, achieving stable training for ResNet-18/CIFAR-10 even under the extreme setting of $p=0.1$ and $\delta=0.2$.

## Background & Motivation

**Background**: The "standard configuration" for Byzantine-robust Federated Learning (FL) involves robust aggregators (Krum, Coordinate Median, RFA, CCLIP) combined with client-side local momentum for variance reduction. The former isolates malicious updates, while the latter distinguishes time-accumulated attacks (e.g., ALIE) from benign stochastic noise. Existing theories almost exclusively assume full client participation.

**Limitations of Prior Work**: Practical systems require partial participation due to bandwidth, battery, and availability constraints. However, the naive combination of "partial participation + robust aggregation" fails. The reason is critical: even if the global Byzantine ratio is $\delta < 1/2$, independent sampling leads to certain rounds where the sampled set itself contains a Byzantine majority ("Byzantine majority round"). Any robust aggregator relying solely on current-round inputs cannot distinguish good from bad. At $p=0.1, \delta=0.2$, such catastrophic rounds occur within the first epoch, leading to the collapse of FedAvg/FedCM.

**Key Challenge**: The fundamental conflict between communication efficiency (small $p$) and robustness—reducing the participation rate exponentially increases the probability of Byzantine majority rounds. Allouah et al. (2024) characterized partial participation but required $p$ to be too large and did not solve the majority round issue; Malinovsky et al. (2024) used variance reduction with clipping to resist majority rounds but relied on large minibatches or full gradients that are impractical in deep learning.

**Goal**: To find a solution that tolerates Byzantine majority rounds, adapts to standard deep learning minibatches, and introduces no additional communication overhead.

**Key Insight**: The server actually retains a cache of the last momentum sent by each client. If these caches are treated as "virtual current-round updates" and fed into the robust aggregator together, the aggregator effectively views all $n$ clients. Consequently, the Byzantine ratio always equals the global $\delta$, eliminating majority rounds.

**Core Idea**: Use "Delayed Momentum Aggregation" to restore the robust aggregator's perspective from a "sampled subset" to the "global set," and use carefully selected momentum coefficients and delay corrections to ensure theoretical convergence of this stitching approach.

## Method

### Overall Architecture
DeMoA maintains the standard shell of synchronous FL: in each round $t$, the server independently samples each client with probability $p_t$ to form $\mathcal{S}_t$. Sampled clients calculate a stochastic gradient using local data, update their momentum, and transmit it back. The momentum of unsampled clients is "re-weighted" on the server side to simulate the same decay rhythm. The server then inputs the momentum of all $n$ clients (whether fresh or cached) into a $(\delta,c)$-robust aggregator $\mathrm{Agg}$. The output $\bm{m}^t$ serves as the direction for the parameter update: $\bm{x}^t \leftarrow \bm{x}^{t-1} - \eta\,\bm{m}^t$. The only added state is "the server caches one momentum vector per client," keeping communication volume identical to FedAvg.

### Key Designs

1.  **Delayed Momentum Aggregation Principle**:
    -   **Function**: Expands the input set of the robust aggregator from the sampled subset $\mathcal{S}_t$ to the full set $[n]$, ensuring the Byzantine ratio faced by the aggregator in every round equals the global $\delta < 1/2$.
    -   **Mechanism**: For unsampled clients $i \notin \mathcal{S}_t$, their momentum $\bm{m}_i^{t-\tau(i,t)}$ from the last sampled round $t-\tau(i,t)$ is passed through a lightweight preprocessing function $\mathcal{P}$ before aggregation. The set $\{\bm{m}_i^t\}_{i\in\mathcal{S}_t}\cup\{\mathcal{P}(\bm{m}_i^{t-\tau(i,t)},i,t)\}_{i\notin\mathcal{S}_t}$ is aggregated. Under small step sizes, delayed momentum is a good approximation of $\nabla f_i(\bm{x}^t)$, allowing the aggregator to see signals from all honest clients while mitigating non-IID drift.
    -   **Design Motivation**: To fundamentally eliminate the failure path of "Byzantine majority in sampled subsets" rather than remedying it post-hoc with clipping or large batches, without increasing communication since cached momentum already exists at the server.

2.  **Special Momentum Coefficient $(1-\alpha_t p_t)$ to Decouple Sampling and Momentum Noise**:
    -   **Function**: Restores "standard" momentum recurrence in expectation for local momentum updates while stripping additional variance caused by sampling randomness.
    -   **Mechanism**: Sampled clients use $\bm{m}_i^t = (1-\alpha_t p_t)\bm{m}_i^{t-1} + \alpha_t \nabla f_i(\bm{x}^{t-1};\xi_i^t)$, while unsampled clients use $\bm{m}_i^t = (1-\alpha_t p_t)\bm{m}_i^{t-1}$. With an indicator variable $r_i^t \sim \mathrm{Ber}(p_t)$, the expected recurrence equals standard momentum $(1-\alpha_t p_t)\bm{m}_i^{t-1} + \alpha_t p_t \nabla f_i$. Compared to naive modifications (using $1-\alpha_t$ for sampled and keeping unsampled unchanged), this design has a variance of only $\alpha_t^2 p_t(1-p_t)\|\nabla f_i\|^2$, avoiding an explosive term dependent on $\|\bm{m}_i^{t-1}\|^2$.
    -   **Design Motivation**: Directly adopting FedCM's coefficient $(1-\alpha_t)$ randomizes the momentum coefficient itself, introducing variance linked to historical momentum norms which may cause divergence. Merging "explicit momentum" and "sampling-induced implicit momentum" into a single effective parameter $\alpha_t p_t$ in expectation is the core mathematical technique for controlling delayed variance.

3.  **Preprocessing Function $\mathcal{P}$ to Remove Implicit Momentum Effects**:
    -   **Function**: Rescales cached momentum delayed by $\tau(i,t)$ rounds according to the cumulative decay factor, preventing delayed momentum from being double-counted during aggregation.
    -   **Mechanism**: Defines $\mathcal{P}(\bm{m}_i^{t-\tau(i,t)},i,t) = \left[\prod_{s=t-\tau(i,t)+1}^{t}(1-\alpha_s p_s)\right]\bm{m}_i^{t-\tau(i,t)}$, which effectively decays the cached momentum along the trajectory it would have followed if it had remained unsampled until round $t$.
    -   **Design Motivation**: Asynchronous/delayed momentum methods (e.g., OrMo, MIFA) previously found that using old momentum directly introduces "implicit momentum" terms tied to strong bounded gradient assumptions. This paper provides a closed-form correction for the synchronous partial participation setting, removing the reliance on bounded gradients and treating MIFA as a special degenerate case where $\alpha=1$, $\mathrm{Agg}=\text{mean}$, and $\mathcal{P}=\text{id}$.

### Loss & Training
DeMoA does not change the training loss but replaces the optimizer. The step size $\eta$, momentum $\alpha_t$, and sampling probability $p_t$ are selected according to the coupling rules in Theorem 3.1. Implementation only requires replacing the "averaging" step in the FedAvg framework with robust aggregation using cached momentum.

## Key Experimental Results

### Main Results

Setup: $n=25$ clients, $\delta=0.2$, CCLIP robust aggregation. Comparison of FedAvg, FedCM, Byz-VR-MARINA-PP, and DeMoA on IID and non-IID data.

| Dataset | Participation rate $p$ | Metric | FedAvg / FedCM | Byz-VR-MARINA-PP | DeMoA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MNIST (ConvNet) | 0.5 | First Byzantine Majority | Collapses after epoch 3 | Stable but lower accuracy | Highest throughout |
| CIFAR-10 (ResNet-18) | 0.1 | First Byzantine Majority | Collapses at epoch 1 | High variance, occasional non-IID failure | Stable convergence, highest accuracy |

DeMoA achieves the highest final accuracy and lowest variance across almost all combinations of five attacks (ALIE, Bit-Flipping, IPM, Label-Flipping, Mimic) and four aggregators (CM, Krum, RFA, CCLIP).

### Ablation Study

| Configuration | Phenomenon | Interpretation |
| :--- | :--- | :--- |
| No Byzantine ($\delta=0$), $p=0.5$, naive avg | DeMoA still outperforms FedCM, Byz-VR-MARINA-PP | Delayed momentum acts as implicit regularization, mitigating non-IID drift |
| Replace coefficient $(1-\alpha_t p_t) \to (1-\alpha_t)$ | Variance includes $\alpha_t^2 p_t(1-p_t)\|\bm{m}_i^{t-1}\|^2$ | Explains why naive momentum is amplified by sampling noise |
| Remove $\mathcal{P}$ (Degenerate to MIFA + naive avg) | Robust constant $c=\infty$, theory becomes void | Shows $\mathcal{P}$ is the bridge connecting delayed momentum and robust aggregation |
| $\delta$ approaching $\min(1/2, 1/(60c(B^2+\alpha(1-p))))$ | Performance drops under some aggregators | Crosses the breakdown point for the aggregator; theory consistent with phenomenon |

### Key Findings
-   **Failure mode accurately located**: The collapse of FedAvg/FedCM occurs strictly after the "first Byzantine majority round," validating that delayed momentum aggregation solves this specific failure point.
-   **Convergence rate**: The non-vanishing term $\mathcal{O}(c\delta\zeta^2)$ in $\frac{1}{T}\sum\mathbb{E}\|\nabla f(\bm{x}^t)\|^2 = \mathcal{O}(c\delta\zeta^2 + \cdots)$ is of the same order as the lower bound for full participation, rather than being amplified by $1/\gamma^2$ due to communication sparsity as seen in decentralized gossip.
-   **Over-parameterization**: Under $(\zeta=0,B)$-heterogeneity assumptions (Corollary 3.2), the non-vanishing term disappears, and the rate returns to the i.i.d. optimal rate.

## Highlights & Insights
-   Uses the simple operation of "expanding the aggregation input set" to reduce the "Byzantine majority round" problem to a full-participation problem, decoupling the conflict between communication efficiency and robustness with zero communication overhead.
-   The momentum coefficient $(1-\alpha_t p_t)$ is a textbook-level minor modification—changing one scalar allows two sources of randomness (sampling and momentum updates) to decouple in terms of variance, making it directly transferable to any "partial participation + momentum" optimizer.
-   The preprocessing function $\mathcal{P}$ provides a closed-form mapping that makes "delayed momentum equivalent to non-delayed," bringing conclusions from asynchronous/delayed optimization that require "bounded gradients" into the synchronous partial participation setting without needing that assumption.

## Limitations & Future Work
-   The server must store one momentum vector per client, which requires significant memory when $n$ is very large. The paper suggests embedding communication compression into $\mathcal{P}$, but systemic analysis is pending.
-   The convergence rate still contains the term $\Gamma = (1-p)\cdot\Theta(1+B^2+c\delta G)/(G(1-60c\delta B^2))$, where constants increase with very small $p$ and high heterogeneity. Over-parameterization eliminates non-vanishing terms but theoretical influence from $\Gamma$ remains.
-   Experiments only covered ResNet-18/CIFAR-10 and ConvNet/MNIST, without verification on large models or large-scale FL. Only independent Bernoulli sampling was studied; complex selection strategies (clustered sampling, power-of-choice) are left for future work.
-   The upper bound for the Byzantine ratio $\delta$ may be tight for highly heterogeneous data. Adaptive attacks specifically targeting delayed momentum have not been systematically evaluated.

## Related Work & Insights
-   **vs Allouah et al. 2024 (First on PP)**: They characterize required participation rates but require large $p$, do not handle Byzantine majority rounds, and lack momentum, making them vulnerable to time-accumulated attacks. Ours uses cached momentum to eliminate majority rounds while retaining robustness against time-accumulated attacks.
-   **vs Malinovsky et al. 2024 (Byz-VR-MARINA-PP)**: They rely on MARINA-style variance reduction with clipping to resist majority rounds but require large batches or full gradients, and clipping introduces bias. DeMoA is more stable and accurate under standard minibatches.
-   **vs MIFA / Fedvarp / CA2FL (Cache-based methods)**: These are motivated by client unavailability rather than Byzantine robustness, analyze only SGD, and naive momentum application triggers implicit momentum effects. DeMoA treats them as degenerate cases and provides corrections.
-   **vs OrMo (Asynchronous Momentum SGD)**: Conceptually inspired the preprocessing function $\mathcal{P}$, but OrMo relies on bounded gradient assumptions; this paper removes that strong assumption in the synchronous partial participation setting.

## Rating
-   Novelty: TBD
-   Experimental Thoroughness: TBD
-   Writing Quality: TBD
-   Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](../../NeurIPS2025/optimization/mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)

</div>

<!-- RELATED:END -->
