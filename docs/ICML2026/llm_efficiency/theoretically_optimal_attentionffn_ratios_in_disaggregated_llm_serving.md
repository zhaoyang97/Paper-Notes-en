---
title: >-
  [Paper Note] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving
description: >-
  [ICML 2026][LLM Efficiency][AFD] This paper provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where "prefill length has a finite mean and decode length follows a geometric distribution," it derives a closed-form solution for the optimal A/F ratio
tags:
  - ICML 2026
  - LLM Efficiency
  - AFD
  - A/F ratio
date: 2026-05-08
content_hash: 6e96b267c435d887
---
# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

**Conference**: ICML 2026  
**arXiv**: [2601.21351](https://arxiv.org/abs/2601.21351)  
**Code**: Yes ([anonymous.4open.science/r/AF-release-1C11](https://anonymous.4open.science/r/AF-release-1C11))  
**Area**: LLM Efficiency / Inference Systems / Attention-FFN Disaggregation  
**Keywords**: AFD, A/F ratio, Bilevel optimization, Geometric distribution, roofline model

## TL;DR
This paper provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where "prefill length has a finite mean and decode length follows a geometric distribution," it derives a closed-form solution for the optimal A/F ratio in an rA-1F topology: $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. The theory is validated using a trace-calibrated simulator, showing a deviation of <10% from measured optimal values.

## Background & Motivation

**Background**: LLM inference services have evolved from monolithic architectures to disaggregation: first with **PD separation** (separating compute-bound prefill from memory-bound decode, Zhong et al. 2024), and recently with **AFD (Attention-FFN Disaggregation)**. AFD recognizes that within the decode phase, Attention (stateful, memory-bound, dominated by KV cache reads) and FFN (stateless, compute-bound when batched) have distinct computing characteristics. These are deployed into separate hardware pools, allowing multiple Attention instances to share a single FFN instance (rA-1F topology).

**Limitations of Prior Work**: AFD performance is extremely sensitive to the A/F ratio $r$—if $r$ is too small, FFN stays idle waiting for data; if $r$ is too large, Attention instances are blocked waiting for FFN. Existing AFD systems (Wang et al. 2025, Zhu et al. 2025, Zuo et al. 2025) rely on **empirical searching** for $r$, lacking theoretical guidance on the location or reasons for the "optimum."

**Key Challenge**: **Attention workload is non-stationary**—the KV cache grows at each step, and finished requests are replaced by new ones (continuous batching), making $T_k$ a stochastic process that drifts over time. Conversely, the **FFN workload is stable** (batch size-dependent only). Consequently, static microbatch schedules cannot remain optimal, leading to pipeline bubbles. To select $r^*$, this non-stationary stochastic dynamic must be simplified into an optimizable scalar.

**Goal**: (1) Establish a probabilistic workload model capturing microbatch pipelining, synchronization barriers, and continuous batching; (2) Derives a closed-form solution for the optimal A/F ratio; (3) Validate the theoretical predictive power using a simulator.

**Key Insight**: The authors observe that decode lengths in production LLM traces closely follow a **geometric distribution** $D\sim \text{Geo}(p)$. This memoryless property makes $X_b(k)$ (whether a slot continues) independent of $i_b(k)$ (current decode index), thereby reducing the complex non-stationary process into a solvable Markov chain. For prefill length $P$, only the mean $\mu_P$ is required, making the model agnostic to the specific distribution.

**Core Idea**: Replace the instantaneous $T_k$ with the "horizon-average token load $\bar{T}=B(\mu_P+\mu_D)$." The cycle time model $\tau=\max\{t_A, t_C, t_F\}$ is decomposed into three regimes (Attention, Communication, and FFN bottlenecks). The global $r^*$ is the maximum of the optimal values derived for each regime.

## Method

### Overall Architecture
The AFD bundle is modeled as an rA-1F topology where each Attention instance maintains $B$ slots. Each decode step consists of four phases: Attention computation $\to$ A-to-F communication $\to$ FFN processing of the aggregated $rB$ batch $\to$ F-to-A communication. The cycle time is $\tau(B;r)=\max\{t_A(T), t_C(B), t_F(rB)\}$. The goal is to maximize per-instance throughput: $\text{Throughput}=\frac{1}{r+1}\cdot \frac{rB}{\tau(B;r)}$. The probabilistic analysis replaces $T$ with the horizon-average value $\bar{T}$ before performing regime analysis.

### Key Designs

1. **Probabilistic Workload Model + Geometric Distribution Insight**:
    - **Function**: Simplifies non-stationary stochastic dynamics into an analytical expected recurrence.
    - **Mechanism**: Define $X_b(k)\sim \text{Bernoulli}(1-p)$ for slot $b$ continuation at step $k$. Decode index updates as $i_b(k+1)=X_b(k)\cdot(i_b(k)+1)$, and prefill length $s_b(k+1)=X_b(k)\cdot s_b(k)+(1-X_b(k))\cdot S_b'(k)$. The **memoryless property of the geometric distribution** ensures $X_b(k)$ is independent of $i_b(k)$, leading to: $\mathbb{E}[P_k]=B\mu_P$ (constant) and $\mathbb{E}[D_k]=B\frac{1-p}{p}(1-(1-p)^k)$ (exponentially saturating from 0 to $B\mu_D$). Thus, $\mathbb{E}[T_k]=B\mu_P+B\frac{1-p}{p}(1-(1-p)^k)$.
    - **Design Motivation**: The geometric distribution reflects the **physical reality of LLM autoregressive generation**: EOS is produced with nearly constant probability at each step, independent of previous tokens. Authors validated this using SGLang and AzureLLM traces (Figure 3).

2. **Horizon-Average Token Load via Law of Large Numbers**:
    - **Function**: Compresses time-drifting $\mathbb{E}[T_k]$ into a scalar representative workload for the optimization objective.
    - **Mechanism**: Define horizon-average $\bar{T}(B;N):=\frac{1}{K(B)}\sum_{k=0}^{K(B)-1}\mathbb{E}[T_k]$, where $K(B)=N/(Bp)$ is the expected steps to serve $N$ requests. Proposition 4.3 proves that as $N\to\infty$, $\bar{T}\to (\mu_P+\frac{1-p}{p})B = B(\mu_P+\mu_D)$.
    - **Design Motivation**: This step is crucial for solvability. Using $\mathbb{E}[T_k]$ directly in max-min optimization prevents closed-form solutions. The horizon average maintains long-run behavior while allowing for asymptotic correctness.

3. **Three-Regime Analysis and Closed-form $r^*$**:
    - **Function**: Decomposes the piecewise function $\tau=\max\{t_A, t_C, t_F\}$ and solves for the global optimum.
    - **Mechanism**: Define $\bar{t}_A=\alpha_A\bar{T}+\beta_A$, $\bar{t}_C=\alpha_C B+\beta_C$, and $\bar{t}_F(r)=\alpha_F rB+\beta_F$. **Regime I (Attention-bottleneck)**: when $r\leq r_A:=(\bar{t}_A-\beta_F)/(\alpha_F B)$, throughput $\propto r/(r+1)$ increases, making $r_A$ optimal. **Regime II (Comm-bottleneck)**: similar logic makes $r_C:=(\bar{t}_C-\beta_F)/(\alpha_F B)$ optimal. **Regime III (FFN-bottleneck)**: when $r\geq r_{\text{crit}}$, $f(r)=rB/[(r+1)(\alpha_F rB+\beta_F)]$ is unimodal; solving $f'(r)=0$ yields $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$. **Theorem 4.4** provides $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$.
    - **Design Motivation**: The max form originates from the requirement that $r$ should be as large as possible without any component becoming a binding bottleneck. $r_{\text{peak}}$ reveals that under FFN bottlenecks, $r$ scales with $\sqrt{1/B}$.

### Loss & Training
**Pure system theory, no training**. Procedure: (1) Inputs: hardware parameters $(\alpha_A, \beta_A, \alpha_F, \beta_F, \alpha_C, \beta_C)$ and workload $(\mu_P, \mu_D)$; (2) Compute $\bar{T} \approx B(\mu_P+\mu_D)$; (3) Calculate $r_A, r_C, r_{\text{peak}}$; (4) $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. Parameter calibration was performed via linear regression on DeepSeek-V3 + Huawei Ascend 910C NPU traces.

## Key Experimental Results

### Main Results: Theoretical $r^*$ vs. Simulated Optimal (DeepSeek-V3 + Ascend 910C)

| Workload Config | Theoretical $r^*$ | Simulated Optimal | Relative Error |
|---------------|-----------|----------|----------|
| Typical chat ($\mu_P$=200, $\mu_D$=300, $B$=32) | (Theor. Val) | (Sim. Val) | <10% |
| Long context ($\mu_P$=2000, $B$=16) | (Increased) | (Matched) | <10% |
| Short response ($\mu_D$=50) | (Decreased) | (Matched) | <10% |

Across various combinations of batch size $B$ and context length $\mu_P$, the deviation between the theoretical $r^*$ and the brute-force simulated optimum remains consistently below 10%.

### Key Findings

| Configuration | Trend | Explanation |
|------|------|------|
| Batch size $B$ ↑ | Optimal $r^*$ ↑ | $r_A$ increases with $\bar{T}=B(\mu_P+\mu_D)$ |
| Context length $\mu_P+\mu_D$ ↑ | Optimal $r^*$ ↑ | Higher Attention workload requires more instances |
| In FFN-bottleneck regime | $r^*=r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ | $r^* \propto 1/\sqrt{B}$; $r^*$ decreases as $B$ grows |
| In Attention-bottleneck regime | $r^*=r_A$ grows linearly | KV cache dominates; $r$ must scale with token load |

### Key Findings
- **Geometric distribution is a critical engineering choice for modeling**: Authors used empirical validation rather than mere assumption; the memoryless property allows for solvable recurrence—a case of empirical observation aligning with mathematical convenience.
- **Three-regime perspective provides interpretability**: Operators can identify the system bottleneck by comparing $\bar{t}_A, \bar{t}_C, \bar{t}_F$ and determine $r^*$ accordingly.
- **The relationship $r^* \propto \sqrt{1/B}$ is counter-intuitive but logical**: In the FFN-bottleneck regime, larger batches allow FFN to process more tokens per cycle, requiring fewer Attention instances.
- **Stability of <10% error**: This indicates that the horizon-average approximation successfully captures the critical dynamics of real workloads.

## Highlights & Insights
- **"Geometric/LLN/Three-Regime" Trifecta**: Converts a seemingly non-stationary queuing problem into a simple three-branch "max" formula. This serves as a model for merging statistical physics, queuing theory, and systems engineering for future disaggregated topologies (e.g., PD-AFD-MoE).
- **Standardized Abstraction via Roofline + Linear Latency**: By mapping $t_A = \alpha_A T + \beta_A$ to existing literature, the model remains rigorous and portable across hardware.
- **Honest Acknowledgement of Simulation**: The authors admit AFD lacks mature open-source implementations and thus validate via trace-calibrated simulation—a responsible "theory-first" approach providing a blueprint for system design.

## Limitations & Future Work
- Validation was limited to simulators; real hardware details (NUMA, network jitter, scheduler overhead) might be missed.
- The assumption that prefill length only requires the mean may fail for heavy-tailed distributions where variance is significant.
- The geometric parameter $p$ varies by task; mixing diverse workloads (chat, code, long-form writing) may require more than a simple mean $\mu_D$.
- SLA/TPOT constraints are not considered; real deployments require balancing throughput with P99 latency.
- Future work: Extension to joint PD-AFD disaggregation, MoE Expert-FFN grouping, and heterogeneous hardware ratios.

## Related Work & Insights
- **vs. PD disaggregation (Zhong et al. 2024)**: PD disaggregation is stage-level (separating phases); AFD is component-level (separating Attention from FFN within decode). These can be combined.
- **vs. AFD Implementations (Wang et al. 2025, Zhu et al. 2025)**: These provide system implementations and empirical search; this paper provides the missing theoretical foundation.
- **vs. Sarathi (Agrawal et al. 2024)**: Sarathi uses chunked prefill to stabilize batches; this paper assumes batch stability to optimize ratios—complementary optimization axes.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical framework with a closed-form solution for AFD.
- Experimental Thoroughness: ⭐⭐⭐ Robust simulator validation, though lacks real hardware deployment.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear derivation chain from workload modeling to Theorem 4.4.
- Value: ⭐⭐⭐⭐ Provides principled formulas for engineering problems, saving trial-and-error in industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](dot-moe_differentiable_optimal_transport_for_moefication.md)
- [\[ICML 2026\] GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[NeurIPS 2025\] UMoE: Unifying Attention and FFN with Shared Experts](../../NeurIPS2025/llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)

</div>

<!-- RELATED:END -->
