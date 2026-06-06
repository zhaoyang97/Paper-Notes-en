---
title: >-
  [Paper Note] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving
description: >-
  [ICML 2026][LLM Efficiency][AFD] This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where prefill length…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "AFD"
  - "A/F ratio"
  - "Bilevel Optimization"
  - "Geometric Distribution"
  - "Roofline Model"
date: 2026-05-08
content_hash: 23639a46d3afbcc4
---

# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

**Conference**: ICML 2026  
**arXiv**: [2601.21351](https://arxiv.org/abs/2601.21351)  
**Code**: Available ([anonymous.4open.science/r/AF-release-1C11](https://anonymous.4open.science/r/AF-release-1C11))  
**Area**: LLM Efficiency / Inference Systems / Attention-FFN Disaggregation  
**Keywords**: AFD, A/F ratio, Bilevel Optimization, Geometric Distribution, Roofline Model

## TL;DR
This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where prefill lengths have a finite mean and decode lengths follow a geometric distribution, it derives a closed-form solution for the optimal A/F ratio $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$ under an rA-1F topology. Theoretical predictions are validated with a trace-calibrated simulator, showing a deviation of <10% from measured optimal values.

## Background & Motivation

**Background**: LLM inference services have evolved from monolithic architectures to disaggregation. This began with **PD disaggregation** (separating compute-bound prefill from memory-bound decode, Zhong et al. 2024). More recently, **AFD (Attention-FFN Disaggregation)** has emerged, noting that within the decode phase, Attention (stateful, memory-bound, dominated by KV cache reads) and FFN (stateless, compute-bound with batching) exhibit distinct computational characteristics. These are deployed into separate hardware pools, allowing multiple Attention instances to share a single FFN instance (rA-1F topology).

**Limitations of Prior Work**: AFD performance is extremely sensitive to the A/F ratio $r$. If $r$ is too small, FFN remains idle waiting for data; if $r$ is too large, Attention instances are blocked waiting for the FFN. Existing AFD systems (Wang et al. 2025, Zhu et al. 2025, Zuo et al. 2025) rely on **empirical search** for $r$, lacking theoretical guidance on the "optimal value and its rationale."

**Key Challenge**: **Attention workload is non-stationary**. As KV cache grows at each step and completed requests are replaced by new ones (continuous batching), the token load $T_k$ is a stochastic process that drifts over time. Conversely, **FFN workload is stable** (depending only on batch size). Consequently, static microbatch schedules cannot remain optimal, inevitably generating pipeline bubbles. To select $r^*$, this non-stationary stochastic dynamic must be simplified into an optimizable scalar.

**Goal**: (1) Establish a probabilistic workload model capturing microbatch pipelining, synchronization barriers, and continuous batching; (2) Derivie a closed-form solution for the optimal A/F ratio; (3) Validate the theoretical predictive power using a simulator.

**Key Insight**: The authors observe that decode lengths in production LLM traces closely follow a **geometric distribution** $D\sim \text{Geo}(p)$. This memoryless property ensures that $X_b(k)$ (whether a slot continues) is independent of $i_b(k)$ (the current decode index), simplifying the complex non-stationary process into a solvable Markov chain. For the prefill length $P$, only the mean $\mu_P$ is required, regardless of its specific distribution.

**Core Idea**: Replace the instantaneous $T_k$ with the "horizon-average token load $\bar{T}=B(\mu_P+\mu_D)$." The cycle time model $\tau=\max\{t_A, t_C, t_F\}$ is decomposed into three regimes (Attention, Communication, and FFN bottlenecks). The global $r^*$ is the maximum of the optimal values calculated for each regime.

## Method

### Overall Architecture
The model treats an AFD bundle as an rA-1F topology. Each Attention instance maintains $B$ slots. A decode step involves four phases: Attention calculation → A→F communication → FFN processing of an aggregated batch $(rB)$ → F→A communication. The cycle time is $\tau(B;r)=\max\{t_A(T), t_C(B), t_F(rB)\}$, and the objective is to maximize per-instance throughput: $\text{Throughput}=\frac{1}{r+1}\cdot \frac{rB}{\tau(B;r)}$. Probabilistic analysis replaces $T$ with the horizon-average value $\bar{T}$ before performing regime analysis.

### Key Designs

1. **Probabilistic Workload Model + Geometric Distribution Insight**:
    - **Function**: Simplifies non-stationary stochastic dynamics on the Attention-side into an analytical expectation recurrence.
    - **Mechanism**: Let $X_b(k)\sim \text{Bernoulli}(1-p)$ indicate if slot $b$ continues at step $k$. The decode index updates as $i_b(k+1)=X_b(k)\cdot(i_b(k)+1)$, and prefill length as $s_b(k+1)=X_b(k)\cdot s_b(k)+(1-X_b(k))\cdot S_b'(k)$. The **memoryless property of the geometric distribution** ensures $X_b(k)$ is independent of $i_b(k)$, yielding clean expectation recurrences: $\mathbb{E}[P_k]=B\mu_P$ (constant) and $\mathbb{E}[D_k]=B\frac{1-p}{p}(1-(1-p)^k)$ (exponentially saturating from 0 to $B\mu_D$). Thus, $\mathbb{E}[T_k]=B\mu_P+B\frac{1-p}{p}(1-(1-p)^k)$.
    - **Design Motivation**: The geometric distribution reflects the **physical reality of LLM autoregressive generation**, where an EOS token is produced with approximately constant probability per step. Traces from SGLang and AzureLLM confirm this (Figure 3). Using only the mean $\mu_P$ for prefill makes the model robust against specific distributions.

2. **Horizon-Average Token Load Convergence**:
    - **Function**: Compresses the time-drifting $\mathbb{E}[T_k]$ into a single representative scalar for the optimization objective.
    - **Mechanism**: Define horizon-average $\bar{T}(B;N):=\frac{1}{K(B)}\sum_{k=0}^{K(B)-1}\mathbb{E}[T_k]$, where $K(B)=N/(Bp)$ is the expected steps to serve $N$ requests. Proposition 4.3 proves that as $N\to\infty$, $\bar{T}\to (\mu_P+\frac{1-p}{p})B = B(\mu_P+\mu_D)$. This formalizes the intuition that the total length per slot is the sum of expected prefill and decode lengths.
    - **Design Motivation**: This step is crucial for solvability. Using $\mathbb{E}[T_k]$ directly in max-min optimization prevents closed-form solutions. The horizon-average maintains long-run behavior while allowing for a closed form, with the Law of Large Numbers providing asymptotic correctness.

3. **Three-Regime Analysis and Closed-form $r^*$**:
    - **Function**: Decomposes the piecewise function $\tau=\max\{t_A, t_C, t_F\}$ to solve for each regime's optimum.
    - **Mechanism**: Define $\bar{t}_A=\alpha_A\bar{T}+\beta_A$, $\bar{t}_C=\alpha_C B+\beta_C$, and $\bar{t}_F(r)=\alpha_F rB+\beta_F$. 
        - **Regime I (Attention-bottleneck)**: For $r\leq r_A:=(\bar{t}_A-\beta_F)/(\alpha_F B)$, throughput $\propto r/(r+1)$ is increasing, peaking at $r_A$.
        - **Regime II (Comm-bottleneck)**: For $r\leq r_C:=(\bar{t}_C-\beta_F)/(\alpha_F B)$, optimal at $r_C$.
        - **Regime III (FFN-bottleneck)**: For $r\geq r_{\text{crit}}$, throughput $f(r)=rB/[(r+1)(\alpha_F rB+\beta_F)]$ is unimodal. Solving $f'(r)=0$ yields $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$.
        - **Theorem 4.4** states $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$.
    - **Design Motivation**: The max form emerges because each regime seeks to maximize $r$ without becoming the binding bottleneck. $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ reveals that under an FFN bottleneck, $r$ scales with $\sqrt{1/B}$: larger batches require fewer Attention instances per FFN.

### Loss & Training
**Ours is purely system-theoretical and requires no training.** The workflow includes: (1) Inputs of hardware parameters $(\alpha_A, \beta_A, \alpha_F, \beta_F, \alpha_C, \beta_C)$ and workload $(\mu_P, \mu_D)$; (2) Calculation of $\bar{T}\approx B(\mu_P+\mu_D)$; (3) Calculation of $r_A, r_C, r_{\text{peak}}$; (4) $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. Parameter calibration was performed using linear regression on DeepSeek-V3 + Huawei Ascend 910C NPU traces.

## Key Experimental Results

### Main Results: Theoretical $r^*$ vs. Simulated Optimum (DeepSeek-V3 + Ascend 910C)

| Workload Config | Theoretical $r^*$ | Simulated Optimum | Relative Error |
|:---|:---|:---|:---|
| Typical Chat ($\mu_P$=200, $\mu_D$=300, $B$=32) | (Value) | (Value) | <10% |
| Long Context ($\mu_P$=2000, $B$=16) | (Increases) | (Matches) | <10% |
| Short Response ($\mu_D$=50) | (Decreases) | (Matches) | <10% |

Across various combinations of batch size $B$ and context length $\mu_P$, the deviation between theoretical $r^*$ and the brute-forced simulated optimum remains consistently below 10%.

### Key Findings

| Configuration | Trend | Explanation |
|:---|:---|:---|
| Batch size $B$ ↑ | Optimal $r^*$ ↑ | $r_A$ grows with $\bar{T}=B(\mu_P+\mu_D)$. |
| Context length $\mu_P+\mu_D$ ↑ | Optimal $r^*$ ↑ | Attention workload increases, requiring more Attention instances. |
| In FFN-bottleneck regime | $r^*=r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ | $r^* \propto 1/\sqrt{B}$; $r^*$ decreases as $B$ increases. |
| In Attention-bottleneck regime | $r^*=r_A$ grows linearly | KV cache dominates; $r$ must scale with token load. |

- **Geometric distribution is a critical engineering choice**: It is validated by real traces (Figure 3) rather than assumed, and its memoryless property makes the recurrence solvable.
- **Three-regime perspective provides explainability**: Operators can determine the system's regime based on the relative magnitudes of $\bar{t}_A, \bar{t}_C, \bar{t}_F$.
- **Counter-intuitive $r^*$ vs. $\sqrt{1/B}$ relationship**: In the FFN-bottleneck regime, larger batches allow FFN to process more tokens per cycle, actually reducing the required number of Attention instances.
- **Robustness**: The <10% error across workloads suggests the horizon-average approximation captures the essential dynamics.

## Highlights & Insights
- **Theoretical Trifecta**: Combining geometric distribution, LLN, and three-regime analysis transforms a non-stationary queuing problem into a simple closed-form formula. This provides a blueprint for modeling more complex topologies like PD-AFD-MoE.
- **Standardized Abstraction**: The use of Roofline models and linear latency ($t_A=\alpha_A T+\beta_A$) aligns with established LLM serving literature (Yuan et al. 2024), ensuring portability across hardware.
- **Scientific Rigor**: The authors candidly admit to using trace-calibrated simulators due to the lack of mature open-source AFD implementations, prioritizing theoretical blueprints over premature prototyping.

## Limitations & Future Work
- Validated only on a simulator; may miss hardware nuances like NUMA, network jitter, or scheduler overhead.
- Assumption that only the mean prefill length matters might be challenged by heavy-tailed distributions.
- Geometric distribution parameter $p$ varies by task; mixing workloads (chat, code, etc.) may require more than a simple mean $\mu_D$.
- Does not explicitly account for SLA/TPOT constraints or power/cost efficiency.
- **Future Directions**: Extending to joint PD-AFD disaggregation, Expert-FFN grouping in MoE, and heterogeneous hardware ratios.

## Related Work & Insights
- **Vs. PD Disaggregation (Zhong et al. 2024)**: PD is stage-level (prefill vs. decode); AFD is component-level (Attention vs. FFN). These are complementary.
- **Vs. AFD Implementations (MoonCake / SpeedyLLM)**: These provide system implementations and search for $r$ empirically; this work provides the missing theoretical foundation.
- **Vs. Sarathi (Agrawal et al. 2024)**: Sarathi uses chunked prefill for batch stability; this work assumes stability to optimize component ratios.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical framework with closed-form solutions for the AFD domain.
- Experimental Thoroughness: ⭐⭐⭐ Rigorous simulator validation across workloads, though lacking real-system deployment.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear derivation chain from workload modeling to Theorem 4.4.
- Value: ⭐⭐⭐⭐ Provides principled formulas that eliminate trial-and-error in industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[NeurIPS 2025\] UMoE: Unifying Attention and FFN with Shared Experts](../../NeurIPS2025/llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)

</div>

<!-- RELATED:END -->
