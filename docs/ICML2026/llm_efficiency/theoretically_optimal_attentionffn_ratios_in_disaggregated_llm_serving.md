---
title: >-
  [Paper Note] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving
description: >-
  [ICML 2026][LLM Efficiency][AFD] This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model with "finite mean pr…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "AFD"
  - "A/F ratio"
  - "bi-level optimization"
  - "geometric distribution"
  - "roofline model"
date: 2026-05-08
content_hash: 69d5caecc6dfe051
---

# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

**Conference**: ICML 2026  
**arXiv**: [2601.21351](https://arxiv.org/abs/2601.21351)  
**Code**: Available ([anonymous.4open.science/r/AF-release-1C11](https://anonymous.4open.science/r/AF-release-1C11))  
**Area**: LLM Efficiency / Inference Systems / Attention-FFN Disaggregation  
**Keywords**: AFD, A/F ratio, bi-level optimization, geometric distribution, roofline model

## TL;DR
This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model with "finite mean prefill length + decode length following a geometric distribution," it derives a closed-form solution for the optimal A/F ratio under the rA-1F topology: $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. A trace-calibrated simulator verifies that the theoretical and empirical optima differ by less than 10%.

## Background & Motivation

**Background**: LLM inference has evolved from monolithic architectures to disaggregation: first, **PD disaggregation** (separating prefill compute-bound and decode memory-bound phases, Zhong et al. 2024), and more recently, **AFD (Attention-FFN Disaggregation)**. It is observed that within the decode phase, Attention (stateful + memory-bound, dominated by KV cache reads) and FFN (stateless + compute-bound after batching) have distinct computational characteristics. Deploying them on separate hardware pools allows multiple Attention instances to share a single FFN instance (rA-1F topology).

**Limitations of Prior Work**: AFD performance is highly sensitive to the A/F ratio $r$—if $r$ is too small, FFN resources are underutilized; if too large, Attention instances are bottlenecked waiting for FFN. Existing AFD systems (Wang et al. 2025, Zhu et al. 2025, Zuo et al. 2025) rely on empirical search for $r$, lacking theoretical guidance and understanding of the optimal point.

**Key Challenge**: **Attention workload is non-stationary**—the KV cache grows at each step, and completed requests are replaced by new ones (continuous batching), making $T_k$ a time-varying stochastic process; **FFN workload is stable** (depends only on batch size). Thus, static microbatch scheduling cannot remain optimal, inevitably causing pipeline bubbles. To select $r^*$, this non-stationary stochastic dynamic must be reduced to an optimizable scalar.

**Goal**: (1) Build a probabilistic workload model capturing microbatch pipelining, synchronization barriers, and continuous batching; (2) Derive a closed-form solution for the optimal A/F ratio; (3) Validate the theory's predictive power using a simulator.

**Key Insight**: The authors observe that decode lengths in production LLM traces closely follow a **geometric distribution** $D\sim \text{Geo}(p)$—this memoryless property ensures $X_b(k)$ (whether a slot continues) is independent of $i_b(k)$ (current decode index), reducing the complex non-stationary process to a solvable Markov chain. The prefill length $P$ only requires its mean $\mu_P$, not the full distribution.

**Core Idea**: Replace instantaneous $T_k$ with the "horizon-average token load" $\bar{T}=B(\mu_P+\mu_D)$, decompose the cycle time model $\tau=\max\{t_A, t_C, t_F\}$ into three regimes (Attention / Communication / FFN bottleneck), optimize each, and take the maximum as the global $r^*$.

## Method

### Overall Architecture
The model considers an AFD bundle with an rA-1F topology. Each Attention instance maintains $B$ slots; each decode step has four stages: Attention computation → A→F communication → FFN processes the aggregated $rB$ batch → F→A communication. The cycle time is $\tau(B;r)=\max\{t_A(T), t_C(B), t_F(rB)\}$, and the objective is to maximize per-instance throughput $\text{Throughput}=\frac{1}{r+1}\cdot \frac{rB}{\tau(B;r)}$. Probabilistic analysis replaces $T$ with the horizon-average $\bar{T}$, followed by regime analysis.

### Key Designs

1. **Probabilistic Workload Model & Geometric Distribution Insight**:

    - **Function**: Reduces the non-stationary stochastic dynamics on the Attention side to an analytically tractable expectation recursion.
    - **Mechanism**: Use $X_b(k)\sim \text{Bernoulli}(1-p)$ to indicate whether slot $b$ continues at step $k$; decode index updates as $i_b(k+1)=X_b(k)\cdot(i_b(k)+1)$, prefill length as $s_b(k+1)=X_b(k)\cdot s_b(k)+(1-X_b(k))\cdot S_b'(k)$. The **memoryless property of the geometric distribution** ensures $X_b(k)$ is independent of $i_b(k)$, yielding clean expectation recursions: $\mathbb{E}[P_k]=B\mu_P$ (constant), $\mathbb{E}[D_k]=B\frac{1-p}{p}(1-(1-p)^k)$ (exponentially saturates from 0 to $B\mu_D$), so $\mathbb{E}[T_k]=B\mu_P+B\frac{1-p}{p}(1-(1-p)^k)$.
    - **Design Motivation**: The geometric distribution is not just mathematically convenient—it **reflects the true physics of LLM autoregressive generation**: each step emits EOS with approximately constant probability, independent of the number of tokens generated. The authors validate the geometric fit for decode length using production traces from SGLang, AzureLLM, etc. (Figure 3); using only the mean $\mu_P$ for prefill makes the model robust to distributional details.

2. **Law of Large Numbers Convergence for Horizon-Average Token Load**:

    - **Function**: Compresses the time-varying $\mathbb{E}[T_k]$ into a representative scalar workload for optimization.
    - **Mechanism**: Define the horizon-average $\bar{T}(B;N):=\frac{1}{K(B)}\sum_{k=0}^{K(B)-1}\mathbb{E}[T_k]$, where $K(B)=N/(Bp)$ is the expected number of steps to serve $N$ requests. Proposition 4.3 proves that as $N\to\infty$, $\bar{T}\to (\mu_P+\frac{1-p}{p})B = B(\mu_P+\mu_D)$—a rigorous version of the intuition that "average total slot length = expected prefill + expected decode."
    - **Design Motivation**: This step is key for tractable optimization—directly using $\mathbb{E}[T_k]$ in max-min optimization does not yield a closed-form solution; using the horizon average preserves long-run behavior and enables a closed form. The law of large numbers provides asymptotic correctness rather than hand-wavy approximation.

3. **Three-Regime Analysis and Global Optimal $r^*$ Closed-Form Solution**:

    - **Function**: Decomposes the piecewise function $\tau=\max\{t_A, t_C, t_F\}$, optimizes each segment separately, and combines for the global optimum.
    - **Mechanism**: Define $\bar{t}_A=\alpha_A\bar{T}+\beta_A$, $\bar{t}_C=\alpha_C B+\beta_C$, $\bar{t}_F(r)=\alpha_F rB+\beta_F$. **Regime I (Attention-bottleneck)**: for $r\leq r_A:=(\bar{t}_A-\beta_F)/(\alpha_F B)$, throughput increases with $r/(r+1)$, optimal at $r_A$; **Regime II (Comm-bottleneck)**: for $r\leq r_C:=(\bar{t}_C-\beta_F)/(\alpha_F B)$, optimal at $r_C$; **Regime III (FFN-bottleneck)**: for $r\geq r_{\text{crit}}$, throughput $f(r)=rB/[(r+1)(\alpha_F rB+\beta_F)]$ is unimodal, with $f'(r)=0$ yielding $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$. **Theorem 4.4** gives $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$, i.e., take the maximum of the regime-wise optima.
    - **Design Motivation**: The max form arises because each regime wants "$r$ as large as possible without becoming the bottleneck"—taking the maximum ensures no component is the binding bottleneck. $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ reveals that "under FFN bottleneck, $r$ scales with $\sqrt{1/B}$"—the larger the batch, the fewer Attention instances are needed, matching intuition.

### Loss & Training
**Purely a systems-theoretic analysis, no training involved.** Calculation steps: (1) Given hardware parameters $(\alpha_A, \beta_A, \alpha_F, \beta_F, \alpha_C, \beta_C)$ and workload $(\mu_P, \mu_D)$; (2) Compute $\bar{T}\approx B(\mu_P+\mu_D)$; (3) Compute $r_A, r_C, r_{\text{peak}}$; (4) $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. Parameter calibration: DeepSeek-V3 + Huawei Ascend 910C NPU trace linear regression.

## Key Experimental Results

### Main Results: Theoretical $r^*$ vs Simulated Optimum (DeepSeek-V3 + Ascend 910C)

| Workload Configuration | Theoretical $r^*$ | Simulated Optimum | Relative Error |
|-----------------------|-------------------|-------------------|---------------|
| Typical chat ($\mu_P$=200, $\mu_D$=300, $B$=32) | (theoretical) | (simulated) | <10% |
| Long context ($\mu_P$=2000, $B$=16) | (increases) | (matches) | <10% |
| Short reply ($\mu_D$=50) | (decreases) | (matches) | <10% |

Across various combinations of batch size $B$ and context length $\mu_P$, the theoretical $r^*$ and the simulator's exhaustive optimum differ by less than 10%.

### Key Findings (from paper ablation)

| Configuration | Trend | Explanation |
|---------------|-------|-------------|
| Batch size $B$ ↑ | Optimal $r^*$ ↑ | $r_A$ term grows with $\bar{T}=B(\mu_P+\mu_D)$ |
| Context length $\mu_P+\mu_D$ ↑ | Optimal $r^*$ ↑ | Attention workload increases, requiring more Attention instances |
| In FFN-bottleneck regime | $r^*=r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ | $r^* \propto 1/\sqrt{B}$; larger $B$ means smaller $r^*$ |
| In Attention-bottleneck regime | $r^*=r_A$ grows linearly | KV cache dominates; $r$ must keep up with token load |

### Key Findings
- **Geometric distribution is a crucial modeling choice**: The authors validate this with real traces (Figure 3), and its memoryless property makes the recursion solvable—a rare case where empirical observation and mathematical convenience align.
- **Three-regime perspective explains "why $r$ should be set this way"**: Operators no longer need to guess; by comparing $\bar{t}_A, \bar{t}_C, \bar{t}_F$, one can determine the regime and thus $r^*$.
- **The $r^*$–$\sqrt{1/B}$ relationship is counterintuitive but matches the trade-off**: In the FFN-bottleneck regime, larger batches allow each FFN cycle to process more tokens, so fewer Attention instances are needed—showing that blindly increasing Attention for large batches is suboptimal.
- **<10% error is stable across workloads**: This demonstrates that the horizon-average approximation captures the essential dynamics.

## Highlights & Insights
- **"Geometric distribution + law of large numbers + three-regime" toolkit**: Transforms a seemingly simulation-dependent, non-stationary queuing problem into a closed-form, three-branch max formula—a beautiful example of combining statistical physics, queuing theory, and systems engineering. The approach can be extended to more complex topologies like PD-AFD-MoE.
- **Roofline model + linear latency is a standard abstraction for LLM inference modeling**: The authors explicitly tie $t_A=\alpha_A T+\beta_A$ and similar linear models to the roofline and existing LLM serving literature (Yuan et al. 2024), making the approach rigorous and portable to other hardware (Appendix B provides a general derivation framework).
- **Candid admission of "simulation validation, not real deployment"**: The authors acknowledge that AFD lacks a mature open-source implementation, so validation is only on a trace-calibrated simulator—this "theory-first, blueprint for system design" stance is more responsible than rushing a prototype.

## Limitations & Future Work
- Only validated on a simulator, not on a real AFD system—while the simulator is trace-calibrated, it may miss hardware details (NUMA, network jitter, scheduler overhead).
- Assumes prefill length only needs the mean, but in practice, heavy-tailed distributions (popular long prompts) may make variance non-negligible.
- The geometric distribution parameter $p$ varies across tasks/models; if a service mixes workloads (chat + code + long-form), using mean $\mu_D$ may be insufficient.
- Does not consider SLA / TPOT constraints—real deployments must satisfy P99 latency, not just maximize throughput.
- Ignores energy/cost constraints, focusing only on throughput per instance.
- Future directions: extend to PD-AFD joint disaggregation, MoE model Expert-FFN grouping, and heterogeneous hardware (mixed GPU/NPU) ratios.

## Related Work & Insights
- **vs PD disaggregation (Zhong et al. 2024, Patel et al. 2024)**: PD disaggregation separates at the stage level (prefill vs decode), while this work's AFD is a further component-level split (Attention vs FFN within decode); both can be combined into a PD-AFD joint architecture.
- **vs MoonCake / SpeedyLLM and other AFD implementations (Wang et al. 2025, Zhu et al. 2025)**: These works provide system implementations and empirically search for $r$; this work supplies the missing theoretical foundation.
- **vs Sarathi (Agrawal et al. 2024)**: Sarathi uses chunked prefill for more stable batching; this work assumes stable batches and optimizes component ratios—complementary optimization axes.
- **vs VirtualFlow / FlexFlow pipeline optimizations**: Those works optimize graph partitioning for a given model; this work optimizes deployment ratios for a given model architecture—different goals and granularity.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical framework with closed-form solution in AFD, novel decomposition via geometric distribution and three regimes.
- Experimental Thoroughness: ⭐⭐⭐ Solid cross-workload simulator validation with <10% error, but lack of real hardware deployment is a minor drawback.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation chain from workload modeling to Lemma 4.1, Prop 4.3, Theorem 4.4; the Practical Recipe section is directly actionable for engineers.
- Value: ⭐⭐⭐⭐ Provides a principled formula for AFD ratio selection, saving substantial trial-and-error in industrial deployment; would be even stronger with real-machine validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[NeurIPS 2025\] UMoE: Unifying Attention and FFN with Shared Experts](../../NeurIPS2025/llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[NeurIPS 2025\] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](../../NeurIPS2025/llm_efficiency/yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)

</div>

<!-- RELATED:END -->
