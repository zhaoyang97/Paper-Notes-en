---
title: >-
  [Paper Note] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift
description: >-
  [ICML 2025][Optimization][Preference Drift] This paper proposes NS-DPO, which incorporates a single exponential decay parameter $\gamma$ into the Dynamic Bradley-Terry model to temporally weight training data. This enables robust DPO alignment under preference drift over time, without sacrificing performance in stationary scenarios.
tags:
  - "ICML 2025"
  - "Optimization"
  - "Preference Drift"
  - "Non-Stationary Optimization"
  - "DPO"
  - "Bradley-Terry Model"
  - "Preference Alignment"
date: 2026-05-08
content_hash: a930cc816477ade0
---

# Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift

**Conference**: ICML 2025  
**arXiv**: [2407.18676](https://arxiv.org/abs/2407.18676)  
**Code**: Yes (mentioned in the paper, specific links in paper footnotes)  
**Area**: LLM Alignment / RLHF  
**Keywords**: Preference Drift, Non-Stationary Optimization, DPO, Bradley-Terry Model, Preference Alignment

## TL;DR

This paper proposes NS-DPO, which incorporates a single exponential decay parameter $\gamma$ into the Dynamic Bradley-Terry model to temporally weight training data. This enables robust DPO alignment under preference drift over time, without sacrificing performance in stationary scenarios.

## Background & Motivation

Existing preference optimization algorithms (e.g., DPO, IPO) assume that human preferences are **stationary**, meaning they do not change over time. However, in reality, preferences drift due to various reasons:

- **Emergence of new information**: Social events or scientific discoveries alter public opinions.
- **Socio-cultural trends**: Moral standards and safety requirements Passenger-side safety evolve over time.
- **Increasing time spans of data collection**: Long-term preference datasets naturally contain changing preferences.

When preference drift occurs, stationary algorithms treat it as **label noise**, treating outdated and fresh data equally, leading to severe misalignment. Prior research shows that data quality is critical for fine-tuning performance, meaning preference drift heavily impacts LLM alignment.

**Core Problem**: How to design computationally efficient preference optimization algorithms when only the upper bound of the total preference drift (and not which specific preferences changed) is known?

## Method

### Overall Architecture

The core idea of NS-DPO is elegant: it introduces an **exponential temporal decay weight** $\gamma^{T-t_i-1}$ into the standard DPO loss function, maximizing the weight of recent data points while discounting older ones. This modification adds only a single hyperparameter $\gamma$.

The pipeline is as follows:

1. Each preference pair in the dataset is annotated with a timestamp $t_i$.
2. Each data point in the DPO loss is multiplied by the exponential decay weight.
3. The optimal setup for $\gamma$ is determined via theoretical analysis.
4. The LLM policy is trained normally.

### Key Designs

1. **Dynamic Bradley-Terry Model**: The standard BT model is extended to a time-varying version $p(a_i \succ a_i'|x_i, t_i) = \sigma(r(x_i, a_i, t_i) - r(x_i, a_i', t_i))$, where the reward function $r(x,a,t)$ explicitly depends on the timestep $t$. This is the first time a dynamic BT model is used within a direct preference optimization framework to capture the time-varying nature of preferences.

2. **Exponentially Weighted Loss Function**: The loss of NS-DPO is defined as $\mathcal{L}^{NS}(\theta_T) = \sum_{(x_i,a_i,a_i',t_i) \in \mathcal{D}} -\gamma^{T-t_i-1} \log\sigma(\tau h_{\pi_{\theta_T}}(x_i, a_i, a_i'))$, where $\gamma \in (0,1)$ controls the discount rate of old data. As $\gamma \to 1$, it reduces to standard DPO. **Key Insight**: Since it is unknown which data points have experienced preference shifts, NS-DPO uniformly downweights all past data, which is a robust strategy under uncertainty.

3. **Variation Budget Assumption**: Instead of assuming how preferences drift, the authors only assume an upper bound $B_T$ on the total drift of the optimal policy parameters $\theta_t^*$, i.e., $\sum_{t=1}^{T-1}\|\theta_{t+1}^* - \theta_t^*\|_2 \leq B_T$. This mild assumption allows for sharp changes at any moment, as long as the total variation is bounded.

### Loss & Training

**Derivation of the NS-DPO Objective**:

Similar to standard DPO, the RLHF objective $\mathcal{J}_T(\pi)$ at timestep $T$ is defined first, deriving the implicit reward $r(x,a,T) = \tau\log\frac{\pi_T^*(a|x)}{\pi_{ref}(a|x)} + \tau\log Z_T^*(x)$. Substituting this into the exponentially weighted negative log-likelihood of the dynamic BT model yields the final NS-DPO loss.

**Regularized Version (for theoretical analysis)**: $$\mathcal{L}_{reg}^{NS}(\theta) = \frac{1}{n}\mathcal{L}^{NS}(\theta) + \frac{\lambda c_{\sigma,\tau}\tau^2}{2}\|\theta\|^2$$

**Selection of Optimal $\gamma$**: Theoretical analysis shows that the optimal regret bound is achieved when $\gamma = 1 - (B_T/T)^{3/4}$. In practice, $\gamma=0.95$ is used for Llama-2-7b, and $\gamma=0.85$ is used for Llama-3.2-1b. On the TV-HH dataset, $\gamma$ is adaptively adjusted based on the change-point $t_{cp}$ as $\gamma = 1 - \frac{1}{(100-t_{cp})\log(100)}$.

**Gradient Analysis**: NS-DPO only adjusts the **scaling term** of the gradient (further reducing the gradient contribution of data points far from $T$ via temporal discounting) and **does not change the gradient direction**. Specifically, the scaling term $\gamma^{T-t_i-1}\sigma(-h_\theta)$ in the gradient exponentially decays the gradient contribution of old data.

**Theoretical Guarantees**:

- **Estimation Error** = Learning Error + Tracking Error: The learning term $O(\sqrt{d/n})$ is identical to standard DPO, while the tracking term $O(\sqrt{T/(1-\gamma)^3} \cdot B_T)$ characterizes the non-stationary cost.
- **Regret Bound**: When $\gamma = 1-(B_T/T)^{3/4}$, $R_T^{off} = \tilde{O}(dB_T^{3/4}n^{-1/4})$.
- **Stationary Recovery**: As $B_T \to 0$, standard $O(n^{-1/2})$ complexity is recovered, showing that NS-DPO strictly generalizes DPO theory.

## Key Experimental Results

### Main Results

Experiments validate the effectiveness of NS-DPO across multiple non-stationary preference datasets:

| Dataset (ρ_diff, t_cp) | Model | Metric | NS-DPO | DPO | SW-DPO |
|----------------------|------|------|--------|-----|--------|
| UltraFeedback-LM (0.7, 21) | Llama-3.2-1b-it | LCWR | **8.93** | 7.29 | 6.09 |
| UltraFeedback-LM (0.7, 51) | Llama-3.2-1b-it | LCWR | **8.38** | 7.85 | 4.93 |
| UltraFeedback-LM (0.7, 81) | Llama-3.2-1b-it | LCWR | **7.85** | 7.17 | 4.63 |
| UltraFeedback-LM (1.0, 21) | Llama-3.2-1b-it | LCWR | **9.00** | 8.23 | 6.71 |
| UltraFeedback-LM (1.0, 51) | Llama-3.2-1b-it | LCWR | **7.41** | 6.99 | 5.59 |
| UltraFeedback-LM (1.0, 81) | Llama-3.2-1b-it | LCWR | **7.36** | 6.49 | 4.83 |
| UltraFeedback-LM (Stationary) | Llama-3.2-1b-it | LCWR | **9.12** | 8.81 | 8.81 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| γ=0.5~0.9 (Synthetic experiments) | Reward accuracy >80% | Stable performance across a wide range of γ |
| γ>0.97 | Performance degrades close to DPO | Excessively large γ nullifies discounting, degenerating to stationary DPO |
| SW-DPO (w=33) | Final accuracy ≈ NS-DPO | Convergence is significantly slower than NS-DPO |
| SW-DPO (w=50) | Performance decreases | Overly large window includes outdated data |
| tDPO (Prompting with time info) | ≈ DPO | ICL cannot effectively resolve preference drift |
| Stationary dataset (tcp=0) | NS-DPO ≈ DPO | No performance degradation when there is no preference drift |
| Gradual preference drift (TV-HH) | NS-DPO > DPO +10% | Significantly outperforms baselines in gradual drift scenarios |
| 2C NSGO (US→Germany) | NS-DPO >60%, DPO ~55% | ~10% margin in socio-cultural preference shifts |

### Key Findings

1. **Later change points show larger NS-DPO advantage**: When preference shifts occur later ($t_{cp}=81$), the gap between NS-DPO and DPO is largest. This is because most data points carry stale preferences, "drowning" DPO in incorrect labels.
2. **Larger $\rho_{diff}$ means higher value for NS-DPO**: When more preference pairs are flipped ($\rho_{diff}=0.9$), non-stationary algorithms gain the most.
3. **No side effects in stationary scenarios**: On drift-free datasets, NS-DPO matches or slightly outperforms DPO (LCWR 9.12 vs 8.81).
4. **Win rate experiments**: On TV-HH with late change points, Llama-3.2-1b-it using NS-DPO consistently achieves >0.5 win rate against DPO.
5. **NS-DPO is robust to $\gamma$**: In synthetic experiments, $\gamma \in [0.5, 0.97]$ maintains $>80\%$ accuracy, whereas SW-DPO is highly sensitive to window size.

## Highlights & Insights

- **Minimalistic changes, highly practical**: It only adds a $\gamma^{T-t_i-1}$ weighting factor to the DPO loss, making it incredibly easy to implement and integrate into existing DPO codebases.
- **Theoretical and empirical guarantees**: It provides regret bounds under log-linear settings and validates effectiveness across various LLM scales.
- **Default-on strategy**: Since there is no performance penalty in stationary scenarios, NS-DPO can serve as a "safe alternative" to standard DPO when preference drift in data is suspected but unconfirmed.
- **Non-stationary preference dataset construction**: The proposed methods for constructing non-stationary datasets (abrupt/gradual drift, multi-reward model switching, cross-cultural preference interpolation) represent valuable experimental infrastructure in their own right.

## Limitations & Future Work

1. **Requirement of timestamps**: NS-DPO relies on timestamp metadata for data points, which is missing from most existing datasets, limiting out-of-the-box applicability.
2. **Prior knowledge required for selecting $\gamma$**: Although theory prescribes the optimal $\gamma$ via $B_T$, $B_T$ is unknown in practice, requiring hyperparameter tuning.
3. **Focus on offline settings**: The paper focuses on offline settings and does not extend to online RLHF or iterative alignment scenarios (though it mentions this as a possible extension).
4. **Risk of amplifying bias in new data**: By placing higher weight on recent data, NS-DPO may amplify biases present in new data.
5. **Theoretical analysis bounded by log-linear policies**: Practical LLMs use deep neural networks, meaning theoretical guarantees have limited direct applicability.

## Related Work & Insights

- **DPO Variants**: Rafailov et al. 2024 (DPO), Azar et al. 2024 (IPO), SimPO, ORPO, etc., do not account for time-varying preferences.
- **Non-stationary Bandits**: The exponentially weighted strategy of Russac et al. 2019 serves as direct inspiration for this work.
- **Carroll et al. 2024**: Discussed various objectives for LLM alignment under preference drift but did not propose concrete algorithms.
- **Insight**: This approach can be extended to multi-objective alignment (where different preference dimensions drift at different rates) or continuous online alignment (where deployed models continuously receive updated preference feedback).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Inspiring problem definition, but the technical implementation (exponential decay weights) is adapted from classical strategies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Thoroughly evaluated across 4 datasets, 2 LLMs, abrupt/gradual/cultural drift, and synthetic settings.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, clear experimental setups, and compelling motivation.
- **Value**: ⭐⭐⭐⭐ — Minimally invasive with no side effects, though the lack of timestamp metadata in real-world data remains a practical hurdle.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICML 2025\] BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization](bopo_neural_combinatorial_optimization_via_best-anchored_and_objective-guided_pr.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](../../NeurIPS2025/optimization/non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment](../../NeurIPS2025/optimization/clean_first_align_later_benchmarking_preference_data_cleaning_for_reliable_llm_a.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](../../NeurIPS2025/optimization/preference_learning_with_response_time_robust_losses_and_guarantees.md)

</div>

<!-- RELATED:END -->
