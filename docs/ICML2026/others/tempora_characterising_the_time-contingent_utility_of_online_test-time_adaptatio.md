---
title: >-
  [Paper Note] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation
description: >-
  [ICML 2026][TTA] TEMPORA reframes TTA evaluation from "offline accuracy without delay limits" to "serviceable utility under latency constraints." By utilizing three categories of time constraints (discrete, continuous…
tags:
  - "ICML 2026"
  - "TTA"
  - "Latency constraints"
  - "Ranking instability"
  - "Utility decomposition"
  - "Edge deployment"
date: 2026-05-08
content_hash: 49ed1e6c10a0c431
---

# TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation

**Conference**: ICML 2026  
**arXiv**: [2602.06136](https://arxiv.org/abs/2602.06136)  
**Code**: https://github.com/sudotensor/tempora (Available)  
**Area**: Self-Supervised / Test-Time Adaptation / Evaluation Framework  
**Keywords**: TTA, Latency constraints, Ranking instability, Utility decomposition, Edge deployment

## TL;DR
TEMPORA reframes TTA evaluation from "offline accuracy without delay limits" to "serviceable utility under latency constraints." By utilizing three categories of time constraints (discrete, continuous, and amortised) alongside decomposable utility metrics, 750+ experiments on ImageNet-C × ResNet-50 demonstrate that offline leaderboards fail to represent deployment reality: the SOTA offline method loses its title in 87.9% of latency-constrained scenarios, and methods become increasingly unpredictable as they approach real-world deployment.

## Background & Motivation

**Background**: Fully Test-Time Adaptation (Fully TTA) assumes the model can only use pre-trained parameters $\theta$ and incoming unlabeled samples $x_i \sim \mathcal{T}$ to correct distribution shifts. The mainstream approach involves freezing the backbone and updating only BN affine parameters (e.g., Tent, ETA, SAR, CMF, DeYO). Based on ImageNet-C offline average accuracy rankings, CMF is generally considered the SOTA.

**Limitations of Prior Work**: Offline evaluation treats TTA as a "static mapping $f_\theta: \mathcal{X} \to \mathcal{Y}$," defaulting to the assumption that the next data batch waits for the current batch's adaptation to complete. However, in real systems, a delayed prediction is a useless prediction—a mobile photo classification delayed by 200 ms results in a missed user interaction, a drone delayed by 1 s crashes, and a surveillance video delayed by one frame loses that frame. This paper measures that a common method (diffusion-based input adaptation) is $810\times$ slower than standard inference, a fact completely obscured by offline leaderboards.

**Key Challenge**: There is a trade-off between "accuracy gain from adaptation" and the "wall-clock time consumed by adaptation," yet existing evaluations measure only the former and relegate the latter to a footnote. Alfarra et al. (2024) first attempted to model latency, but their unit of speed was "inference FLOPs relative to the model" rather than physical milliseconds, causing semantic drift across different hardware/models. Furthermore, their ceiling operator mapped any $\delta \in (k\gamma, (k+1)\gamma]$ to the same miss rate, smoothing over internal optimizations within methods.

**Goal**: To transform TTA evaluation into measuring "how many usable predictions a method actually delivers under given physical latency constraints (ms)" and to decompose this new dimension into diagnostic sub-items to answer "why the ranking changed" rather than just "if it changed."

**Key Insight**: The authors categorize deployment scenarios into three prototypes based on the source of time pressure: environment-driven (sensors pushing data at a fixed frame rate), user-driven (users waiting for a response), and resource-driven (total battery or compute budget). These correspond to discrete, continuous, and amortised utility measures, respectively. Each measure is designed as an analytically decomposable product or weighted sum, separating "accuracy per se" from "time penalty."

**Core Idea**: By using "physical millisecond anchoring + three-tier time prototypes + decomposable utility," TTA is redefined from "running fast on an offline leaderboard" to "performing within the hardware constraints and latency tolerance of the customer."

## Method

TEMPORA is not a new TTA method but an evaluation framework consisting of three parts: (i) temporal scenario characterization of deployment constraints, (ii) an evaluation protocol for wall-clock simulation, and (iii) latency-aware utility metrics that couple accuracy and latency into a single scalar.

### Overall Architecture
The data stream is formalized as $\{(\mathbf{x}_i, t_i)\}_{i=1}^N$, where $t_i$ is the actual arrival time of the $i$-th batch. Each batch follows a pipeline of "pickup → prediction → subsequent adaptation." The authors decompose single-batch latency based on prediction time as $\delta_i = e_i + \ell_i$: $e_i$ is the intrinsic part (pickup to prediction), and $\ell_i$ is the extrinsic part (backpropagation updates after prediction that block the next batch). The framework runs three different "arrival-queueing-utility settlement" rules on this common timeline, requiring all latencies to be measured in milliseconds on the same GPU rather than using FLOPs proxies.

### Key Designs

1.  **Discrete Utility (Environment-driven)**:
    - **Function**: Models scenarios where sensors force data at fixed intervals $\gamma$ (e.g., cameras, LiDAR). Batches that the pipeline cannot keep up with are permanently dropped.
    - **Mechanism**: Mimics pipeline start/end times using recursion $s_{j+1} = \max(f_j, t_{p_{j+1}})$ and $f_{j+1} = s_{j+1} + \delta_{p_{j+1}}$. The next batch pointer $p_{j+1} = \max(p_j+1, \lfloor f_j/\gamma\rfloor+1)$ allows "batch skipping" to emerge naturally from wall-clock simulation. Utility is $U_{\text{discrete}} = \alpha \cdot \bar{a}_{\text{served}}$, where $\alpha = |\mathcal{Q}|/N$ is availability and $\bar{a}_{\text{served}}$ is the average accuracy of served batches. These factors independently diagnose "system errors (skipped batches)" and "model errors (misclassifications)."
    - **Design Motivation**: Addresses three flaws in Alfarra et al.—adds a batch-sized buffer to eliminate forced idling, uses real milliseconds to eliminate stepped penalties, and removes zero-cost fallback models to avoid masking performance with optimistic predictions. It explicitly exposes "computational insolvency," such as when ETA cannot match AdaBN even with a $1.5\times$ higher served accuracy due to availability ceilings.

2.  **Continuous Utility (User-driven)**:
    - **Function**: Models a "greedy user" in interactive systems who sends $x_{i+1}$ only after receiving $y_i$, applying a soft penalty for slow responses rather than dropping them.
    - **Mechanism**: The user-perceived wait time is $w_i = \ell_{i-1} + e_i$. With effective response latency $d_i = \max(0, w_i - \lambda)$, accuracy is weighted by a hyperbolic discount $\kappa_i = (1 + d_i / (T - \lambda))^{-1}$ within $[0, 1]$, where $T$ is the "half-life" threshold. Utility is $U_{\text{continuous}} = \bar{a} \cdot \bar{\kappa} + \mathrm{Cov}(a, \kappa)$, decomposed into "average accuracy × average responsiveness" plus an "alignment term." A negative alignment term indicates that high accuracy occurs specifically on slow-response batches.
    - **Design Motivation**: Hyperbolic decay aligns better with subjective time perception in HCI (100 ms delay perception $\gg$ 1 s delay perception) and remains monotonic and bounded. Distinguishing intrinsic and extrinsic overhead reveals that gradient-based methods' extrinsic overhead $\bar{\ell}$ (56–154 ms) is the root cause of their failure in user-driven scenarios.

3.  **Amortised Utility (Resource-driven)**:
    - **Function**: Models devices with total budget limits (e.g., drone battery, overnight inference), where individual batch latency is less critical than cumulative overhead $\sum c_i \le B$.
    - **Mechanism**: Defines a cut-off point $m = \max\{j : \sum_{i=1}^{j} c_i \le B\}$. The first $m$ batches undergo normal adaptation, after which parameters $\theta_m$ are frozen for remaining forward passes. Utility is $U_{\text{amortised}} = \beta \cdot \bar{a}_{\text{adapt}} + (1 - \beta) \cdot \bar{a}_{\text{frozen}}$, where $\beta = m/N$ is the adaptation ratio. A result where $\bar{a}_{\text{frozen}} < a_0$ exposes "harmful adaptation."
    - **Design Motivation**: Quantifies whether adaptation was "worth the cost." Methods are compared by sweeping the Pareto frontier of various budgets. This reveals that methods like SHOT-IM, which retain source running statistics, maintain accuracy when frozen, while Tent, ETA, CMF, SAR, and DeYO often collapse to 0.1%.

### Loss & Training
TEMPORA does not train models but runs evaluations. The framework tests 9 Fully TTA methods (AdaBN, LAME, NEO, Tent, ETA, SHOT-IM, CMF, DeYO, SAR) plus a standard inference baseline on an RTX 4080 using ImageNet-C severity 5 (15 corruptions). Each method uses default hyperparameters with a single seed (2025). Standard inference latency $\lambda$ is set to 39.9 ms.

## Key Experimental Results

### Main Results
Decomposition of methods under three time metrics (average of 15 corruptions, 11,715 batches):

| Method | Discrete $U(\rho{=}100\%)$ | Continuous $U(T{=}50\,\text{ms})$ | Amortised $U(B{=}1\,\text{s})$ | Offline Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Standard | 18.2 | 18.16 | 18.16 | 18.2 |
| AdaBN | 30.8 | 28.42 | 31.72 | ≈26 |
| NEO | 22.1 | 22.14 | 22.14 | ≈22 |
| ETA | 18.7 | 7.22 | 0.87 | High (offline #3) |
| CMF | 11.5 | 3.84 | 0.48 | Offline SOTA (10/15) |
| SAR | 7.8 | 2.73 | 0.40 | ≈32 |
| SHOT-IM | 13.5 | 4.73 | **32.22** | 13.15-67.62 |

Per-batch latency decomposition ($\lambda=39.9$ ms):

| Method | $\bar{\delta}$ (ms) | $\bar{e}-\lambda$ (Intrinsic) | $\bar{\ell}$ (Extrinsic) | Slowdown |
| :--- | :--- | :--- | :--- | :--- |
| AdaBN | 41.1 | 1.1 | 0.0 | 1.03× |
| Tent | 97.1 | 1.2 | 56.1 | 2.43× |
| ETA | 97.7 | 1.2 | 56.6 | 2.45× |
| SHOT-IM | 121.0 | 1.2 | 79.8 | 3.03× |
| CMF | 160.1 | 1.2 | 119.0 | 4.01× |
| SAR | 195.2 | 1.2 | 154.1 | 4.89× |

### Ablation Study

| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| Offline vs. $\rho=100\%$ | CMF #1 in 10/15 offline → Wins only 29/240 (12.1%) under latency | Direct evidence of ranking instability. |
| Different $T$ (50→1000 ms) | Spearman $r_s$ rises from -0.19 to 0.97 | Evaluation approaches offline ranking as latency constraints loosen. |
| ETA at $\rho=100\%$ | Served acc 45.6%, but $\alpha=0.41$ → 18.7% | Availability ceiling; needs 75.1% served acc to match AdaBN. |
| SAR at $\rho=100\%$ | Requires 149.5% accuracy to match AdaBN | Mathematically impossible; computational insolvency. |
| Amortised $B=1$ s | Gradient methods exhaust budget in ~20 batches | "Harmful adaptation" failure mode; accuracy drops to 0.1%. |
| SHOT-IM Anomaly | Maintains 32.2% after freeze due to source stats | Robustness invisible in offline evaluations. |

### Key Findings
- **Extrinsic overhead is the killer**: While intrinsic overhead is only 0–1.2 ms across methods, the 56–154 ms backpropagation overhead of gradient methods ($\ell_i$) crushes availability and responsiveness in environment and user-driven scenarios.
- **Offline leaders are the biggest losers**: CMF lost 211 out of 240 latency trials (87.9%), with 15% lower utility than the winners. ETA, though never an offline champion, won 42.9% of latency trials, primarily in medium-pressure settings ($35\% \le \rho \le 70\%$).
- **Geometric interpretations of failure modes**: In discrete settings, availability $\alpha$ acts as a utility ceiling (additive displacement); in continuous settings, $\bar{\kappa}$ discounts accuracy (multiplicative discounting); in amortised settings, parameter drift after budget depletion causes negative contribution (cumulative reversal).

## Highlights & Insights
- **Physical millisecond anchoring** ensures that a threshold (e.g., 50 ms) maintains consistent semantics across different hardware and models, enabling cross-paper and cross-hardware comparisons—a significant advancement over prior FLOPs-based proxies.
- **Utility decomposition** ($U_{\text{discrete}} = \alpha \cdot \bar{a}_{\text{served}}$, etc.) makes the reason for failure explicit. The paper identifies three requirements for "deployable adaptation" (compute adjustment per corruption, time-aware scaling, anytime performance) that serves as a specification for future methods.
- **Batch-sized buffer + wall-clock simulation** ensures missed batches are an emergent result rather than a pre-calculated input. This "simulation over analytical scoring" approach is transferable to streaming continual learning and online RL.

## Limitations & Future Work
- **Scope Restrictions**: Main experiments were limited to ImageNet-C × ResNet-50. While the appendix includes ViT-B/16 and Pi 5 CPU tests, other modalities (segmentation, translation) require recalibration of $\lambda$, $T$, and $B$.
- **Method Coverage**: Extremely slow methods (e.g., DDA, MEMO) and high-complexity wrapper methods were excluded. The findings currently apply best to mid-overhead Fully TTA methods.
- **Single Seed & Default Hyperparameters**: While hyperparameter sweeps might shift some rankings, methods near computational insolvency are unlikely to be saved by tuning.
- **Future Directions**: Combining the three utility types into hybrid metrics and integrating TEMPORA with benchmarks for distribution authenticity (like BoTTA) to simulate non-i.i.d. real-world streams.

## Related Work & Insights
- **vs. Alfarra et al. (ICML 2024)**: TEMPORA replaces their FLOPs-proxy and ceiling discretization with physical milliseconds and wall-clock simulation, allowing cross-hardware reuse.
- **vs. BoTTA / UniTTA / TTAB (2023-2025)**: While those focus on distribution authenticity (non-i.i.d.), TEMPORA focuses on temporal authenticity. The two are orthogonal and can be combined.
- **vs. Ghunaim et al. (2023)**: Similar observations regarding sample starvation in online CL, but TEMPORA physicalizes and decomposes these for the TTA domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Not a new method, but the "physical ms + three prototypes + decomposable utility" design is a missing engineering paradigm for the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High density: 9 methods × 15 corruptions × 16 time scenarios = 750+ evaluations, plus cross-hardware validation on Pi 5.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Formalization and failure mode diagnostics are exceptionally clear, providing analytical conditions for when methods beat the baseline.
- **Value**: ⭐⭐⭐⭐⭐ Challenges the community's "offline accuracy is king" inertia and provides clear design targets for future TTA methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](private_and_stable_test-time_adaptation_with_differential_privacy.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)

</div>

<!-- RELATED:END -->
