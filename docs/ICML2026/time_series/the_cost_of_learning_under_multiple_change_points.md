---
title: >-
  [Paper Note] The Cost of Learning Under Multiple Change Points
description: >-
  [ICML 2026][Time Series][Online learning] This paper proposes the Anytime Tracking CUSUM (ATC) algorithm. By employing a time-varying adaptive threshold and the selective detection principle…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Online learning"
  - "Change point detection"
  - "Dynamic regret"
  - "Non-stationary environments"
  - "Endogenous confounding"
date: 2026-05-08
content_hash: 4921b6e2bcf00423
---

# The Cost of Learning Under Multiple Change Points

**Conference**: ICML 2026  
**arXiv**: [2602.11406](https://arxiv.org/abs/2602.11406)  
**Code**: To be confirmed  
**Area**: Time Series / Online Learning Theory  
**Keywords**: Online learning, Change point detection, Dynamic regret, Non-stationary environments, Endogenous confounding

## TL;DR
This paper proposes the Anytime Tracking CUSUM (ATC) algorithm. By employing a time-varying adaptive threshold and the selective detection principle, it achieves near-minimax optimal dynamic regret $O(\sigma^2 (S+1) \log T)$ without requiring any **detectability assumptions** (e.g., minimum spacing or minimum jump size). Furthermore, it formally quantifies the logarithmic degradation bound of "endogenous confounding caused by missed detections" in multi-change point scenarios for the first time.

## Background & Motivation

**Background**: Non-stationary environments in online learning have been studied for years. While high-confidence ($\delta$-PAC) theory for single change point detection (e.g., CUSUM) is mature, practical applications typically involve an **unknown number and unknown locations** of multiple change points and require anytime algorithms (without prior knowledge of the horizon $T$).

**Limitations of Prior Work**: Existing methods often assume "detectability"—requiring minimum spacing and minimum jump sizes between change points. If these assumptions fail, two problematic phenomena occur:
- **Endogenous confounding**: When a change point is missed, legacy data remains in the reference statistics, polluting the detection baseline for subsequent change points; the learner's failure exacerbates future detection tasks.
- **Cascading collapse**: Confounding accumulates gradually, causing the detection power for subsequent change points to decline continuously, eventually leading to a complete breakdown in algorithm performance. This is particularly severe in non-parametric settings where reference distributions must be estimated from historical samples.

**Key Challenge**: How to design an algorithm that can quickly adapt to large changes while remaining stable against small or transient changes—avoiding increased variance from frequent restarts—all without relying on detectability assumptions?

**Goal**: To establish the learning-theoretic foundations for multi-change point online learning by providing lower bounds for dynamic regret and designing algorithms that achieve these bounds.

**Key Insight**: The authors abandon the "high-confidence detection" framework in favor of a **regression regret** approach. By using dynamic regret (the cumulative squared error between predicted values and the time-varying true mean) as a unified metric, the costs of detection delay, false alarms, and endogenous confounding are naturally encoded. The crucial insight is that it is **not necessary to detect every change point**—the regret cost of missing small or brief changes is controllable. The key is to use adaptive thresholds to distinguish between detectable and undetectable changes.

**Core Idea**: The combination of a time-varying adaptive threshold, the selective detection principle, and a logarithmic upper bound on SNR degradation for missed detections allows the algorithm to achieve near-minimax optimal dynamic regret of $O(\sigma^2 (S+1) \log T)$.

## Method

### Overall Architecture
ATC extends the classical CUSUM to multi-point tracking. At each time step $t$, it maintains:
- The last restart time $r$ (initially $r = 1$);
- The cumulative sum $G_t = \sum_{i=1}^t X_i$ for fast calculation of segment means.

The process consists of two stages:
- **Detection**: Calculate the CUSUM statistic $C_t^r = \max_{r < k < t} \hat{D}_{k,t}^r$ and compare it against a time-varying threshold $\gamma_t^r$. If the threshold is exceeded, a restart is triggered: $r \leftarrow t$.
- **Prediction**: Output the mean of the most recent complete segment $\hat{\mu}_t = \frac{1}{t-r} \sum_{i=r}^{t-1} X_i$.

### Key Designs

1. **CUSUM Detection Statistic**:
    - **Function**: Scans all split points $k$ within the current restart interval $[r, t)$ to find the strongest evidence of a change point.
    - **Mechanism**: $\hat{D}_{k,t}^r = \frac{1}{\sigma} \sqrt{\frac{(k-r)(t-k)}{t-r}} \left| \bar{X}_{r:k-1} - \bar{X}_{k:t-1} \right|$. This is a normalized statistic of the difference between two segment means, similar to Generalized Likelihood Ratio (GLR) but adapted for unknown pre- and post-distributions. At a true change point $\tau_j$, the SNR is $\text{SNR}_j^*(t) = \frac{(\tau_j - \tau_{j-1})(t - \tau_j)}{t - \tau_{j-1}} \frac{\Delta_j^2}{\sigma^2}$, where $\Delta_j$ is the jump size.
    - **Design Motivation**: SNR increases monotonically with $t$, ensuring sufficiently large changes are detected within logarithmic delay; scanning all $k$ ensures that change points at unknown locations are not missed.

2. **Time-varying Adaptive Threshold (Core Innovation)**:
    - **Function**: Dynamically adjusts the threshold to balance stability (low false alarms) and adaptivity (fast detection of true change points) without pre-knowledge of $T$ or $S$.
    - **Mechanism**: $\gamma_t^r = \sqrt{6 \log(t - r) + 2 \log(1/\alpha_r) + 2 \log(\pi^2/3)}$, where $\alpha_r = \frac{6 \alpha}{\pi^2 r^2}$ is the cumulative false alarm budget allocated decreasingly by restart time, satisfying $\sum_r \alpha_r \leq \alpha$. The detection condition is $C_t^r \geq \gamma_t^r$.
    - **Design Motivation**: The $\log(t - r)$ growth rate ensures uniform concentration of the scanning statistic over all $k$ and $t$ within the current segment. The convergence of the $\alpha_r$ series ensures bounded total anytime false alarms, naturally supporting true online/anytime operations.

3. **Endogenous Confounding Quantification (SNR Degradation Bound)**:
    - **Function**: Quantitatively constrains the negative impact of missing a specific change point on the detection power of subsequent ones, ensuring no cascading collapse occurs.
    - **Mechanism**: If the $j$-th change point is missed, the reference mean becomes a mixture $\mu_{\text{pre}}^{\text{eff}}(r, j) = \frac{\sum_{\ell = i}^{j-1} n_\ell \mu_\ell}{\sum n_\ell}$. The effective jump size $\Delta_j^{\text{eff}} = |\mu_{\text{pre}}^{\text{eff}} - \mu_j|$ may be much smaller than the true $\Delta_j$, leading to a drop in effective SNR. Proposition 3.1 provides a **logarithmic** degradation upper bound: $(\text{SNR}_j^*(t) - \text{SNR}_j^{\text{eff}}(t; r))_+ \leq C \log\frac{\tau_j - r + 1}{\alpha_r}$.
    - **Design Motivation**: This is the theoretical core of the paper—it demonstrates that missed detections delay subsequent detections but do not lead to algorithmic collapse, thereby justifying the "selective detection" principle.

### Loss & Training
The goal is to minimize dynamic regret $\mathcal{R}_T(\pi) = \mathbb{E}[\sum_{t=2}^T (\hat{\mu}_t - \mu_t)^2]$. There is no offline optimization or training; the process is purely online tracking.

## Key Experimental Results

### Main Results (Theory + Synthetic + Real Data)

| Environment | Change Points $S$ | Time $T$ | ATC Regret | Theoretical Upper Bound | Theoretical Lower Bound | Remarks |
|------|----------|---------|---------|--------|--------|------|
| Synthetic (Mean Jump) | 5 | 300+ | $O(\log T)$ | $O(\sigma^2 (S+1) \log T)$ | $\Omega(\sigma^2 (S+1) \log(T/(S+1)))$ | 5000 MC trials |
| NAB AWS CPU Data | Unknown | 4000+ | Lowest Regret | ~40% lower than baseline | — | Explicit restart outperforms sliding window |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full ATC (Log Threshold) | Regret + False Alarm Balance | Default configuration |
| Constant Threshold ATC | Regret +30% | Fixed threshold fails to adapt to segment growth, increasing missed detections |
| Computationally Efficient Variant (Geometric Grid) | Regret nearly unchanged, Cost $O(\log(t-r))$ | Limits candidate scanning points; preserves asymptotic rates |

### Key Findings
- Figure 4(c) shows that regret grows linearly with $\log T$ across 5000 MC trials, consistent with Theorem 4.1. The computationally efficient variant's curve is parallel to the full version, differing only by a constant.
- The 5th change point was successfully detected at $t \approx 900$ even after undergoing two missed detections, validating the "logarithmic, non-collapsing" nature of the SNR degradation bound.
- On NAB data, the method adapts immediately after large jumps compared to sliding window/discounting baselines, reducing regret by 40% at the large jump near step 4000+.
- Sensitivity to variance $\sigma$ mis-specification: Underestimating $\sigma$ increases false alarms but does not change the asymptotic rate.

## Highlights & Insights
- **Formalization of Endogenous Confounding**: This is the first work to characterize the hidden traps of multi-change point online learning as an explicit logarithmic bound in Proposition 3.1. All restart-based online learning analysis should refer to this framework.
- **Selective Detection Principle**: The core philosophy is that "one need not detect every change"—allowing changes below the statistical resolution to go undetected (with controllable cost) enables near-minimax optimality in a fully general setting. This is counter-intuitive yet powerful.
- **Near-Minimax Optimality**: Theorem 4.1 (upper bound) and Theorem 4.2 (lower bound) differ only by a $\log(S+1)$ factor, providing a tight characterization for anytime algorithms.
- **Dynamic Regret Framework**: Tracking moving targets using squared loss is transferable to problems like non-stationary RL and dynamic pricing.

## Limitations & Future Work
- While the upper bound is achieved in a fully online manner, it remains an open question whether knowing $T$ and $S$ beforehand could further improve the bound to $O(\sigma^2 S \log(T/S))$.
- The algorithm assumes the sub-Gaussian proxy $\sigma$ is known; in practice, it needs to be estimated online.
- Multi-dimensional extensions ($\mathbb{R}^d$) would introduce a $\sqrt{d}$ factor in the threshold, limiting high-dimensional efficiency.
- The results are specific to squared loss; the lower bound for $L_1$ is $\Omega(\sqrt{ST})$ (linear growth), implying that a qualitative shift in algorithm design would be required.

## Related Work & Insights
- **vs. Classical CUSUM (Page 1954; Lorden 1971)**: Classical CUSUM targets a single change point with a known pre-distribution; this work generalizes to multiple change points without detectability assumptions, with the core innovation being the handling of endogenous confounding.
- **vs. Sliding Window / Discounting (Garivier & Moulines 2011)**: Passive methods rely on forgetting old data and require manual parameter tuning; the explicit restart in this work automates hyperparameter optimization.
- **vs. Multi-Armed Bandit Active Restart (Liu et al. 2018; Cao et al. 2019)**: Previous works required high-confidence detection assumptions; this paper provides regret bounds without such assumptions, potentially inspiring new algorithms for non-stationary bandits.
- **Insights**: The "three-piece suite" of selective detection, adaptive thresholds, and missed-detection degradation bounds holds methodological value for all restart-based online learning and reinforcement learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to strictly formalize endogenous confounding and achieve near-minimax optimality in a fully general setting; a significant breakthrough in online learning theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic data clearly validates the theory; performance on NAB demonstrates practical advantages; thorough ablation; lacks comparison on a wider range of real-world multi-change point datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ The visualization of endogenous confounding in Figure 1 is intuitive and powerful; proof sketches are complete in the main text; definitions are precise.
- Value: ⭐⭐⭐⭐⭐ Theoretically fills a gap in multi-change point non-stationary learning; practically applicable to demand tracking, resource management, and online compression; provides a universal reference for algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](../../NeurIPS2025/time_series/planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](../../NeurIPS2025/time_series/frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[ICML 2026\] Divide and Contrast: Learning Robust Temporal Features Without Augmentation](divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)
- [\[ICML 2026\] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)

</div>

<!-- RELATED:END -->
