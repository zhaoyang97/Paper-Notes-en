---
title: >-
  [Paper Note] Threshold-Based Exclusive Batching for LLM Inference
description: >-
  [ICML 2026][Autonomous Driving][LLM Inference] This paper systematically characterizes the performance crossover conditions between mixed batching (MB) and exclusive batching (EB) in LLM inference. It proves that combini…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "LLM Inference"
  - "Batching Scheduling"
  - "Exclusive Batching"
  - "Mixed Batching"
  - "Memory Bandwidth"
date: 2026-05-08
content_hash: ff7c6cea04a88ac3
---

# Threshold-Based Exclusive Batching for LLM Inference

**Conference**: ICML 2026  
**arXiv**: [2606.00516](https://arxiv.org/abs/2606.00516)  
**Code**: https://github.com/weifang231/eb-vllm  
**Area**: LLM Efficiency / Inference Scheduling  
**Keywords**: LLM Inference, Batching Scheduling, Exclusive Batching, Mixed Batching, Memory Bandwidth

## TL;DR
This paper systematically characterizes the performance crossover conditions between mixed batching (MB) and exclusive batching (EB) in LLM inference. It proves that combining prefill and decode in the same batch on bandwidth-constrained GPUs slows down Attention due to bandwidth contention. Based on this, it derives the optimal phase-switching threshold $\theta^*$ and memory-safe batch size using the hazard rate. The designed online adaptive scheduler, EB+, achieves up to 41.9% throughput improvement on bandwidth-constrained hardware and up to 36.4% improvement over MB under non-stationary traffic.

## Background & Motivation
**Background**: LLM inference consists of two distinct phases: prefill (compute-bound) and decode (memory-bandwidth-bound). Mainstream inference engines (vLLM v1, SGLang, TGI, TensorRT-LLM) default to mixed batching, which combines prefill and decode tokens into a single forward pass to utilize both compute power and bandwidth. However, some production systems still prefer exclusive batching, where prefill and decode phases alternate in separate batches.

**Limitations of Prior Work**: The rationality of adopting MB as the default has never been strictly questioned. Through controlled experiments, the authors found that on high-bandwidth H200 (4.8 TB/s), the marginal cost of MB exceeds pure decode only when the decode token ratio $r$ exceeds 80%. Conversely, on bandwidth-constrained RTX PRO 6000 (1.792 TB/s), this threshold drops to 20%. This implies that MB is not universally optimal, yet existing works provide neither analytical criteria nor adaptive scheduling strategies.

**Key Challenge**: Decode Attention requires streaming the entire KV-cache token-by-token, which is inherently limited by memory bandwidth. Mixing prefill into the same batch competes for bandwidth, significantly increasing decode Attention latency. This interference is absorbed by high-bandwidth GPUs but amplified on low-bandwidth GPUs. Current FlashAttention kernels are not specialized for mixed batching in bandwidth-constrained scenarios, making the "universal MB" paradigm suboptimal for many hardware configurations.

**Goal**: (1) Provide closed-form criteria for the EB vs. MB performance crossover; (2) derive the optimal phase-switching threshold $k^*$ and memory-safe batch size $N^*$ for EB mode; (3) design a hybrid scheduler capable of online adaptation to workloads and hardware.

**Key Insight**: The authors model the single-step iteration time linearly as $T_{\text{iter}} = \alpha + \beta \cdot n_{\text{tok}}$ and characterize the batch by the decode ratio $r = n_{\text{decode}} / n_{\text{tok}}$. Both $\alpha$ and $\beta$ depend on hardware (bandwidth) and batch composition ($r$). Using the saturated assumption and fluid approximation, throughput optimization is transformed into a scalar optimization problem depending only on a few measurable parameters.

**Core Idea**: Use the *hazard rate of the output length distribution* to determine when to switch between prefill and decode phases, and use the marginal cost difference $\beta_{\mathrm{MB}}^e - \beta_{\mathrm{EB}}^w$ to decide online between EB and MB.

## Method

### Overall Architecture
The system utilizes $N$ slots (maximum concurrent requests). During the decode phase, one slot is freed as each request completes. When the number of free slots reaches threshold $k$, the system enters the prefill phase, parallelly prefilling $k$ new requests into the vacant slots before returning to the decode phase. The scheduler consists of two layers:

1.  **Offline Derivation Layer** — Under the saturated assumption (saturated queue, full batch) and fluid approximation, it derives (i) the optimal normalized threshold $\theta_0 = k_0^* / N$; (ii) the IFR correction term $\Delta\theta$; (iii) the maximum memory-safe batch size $N^*$; and (iv) the throughput crossover condition for EB vs. MB.
2.  **Online Control Layer** — Estimates $(\hat p_0, \hat\eta, \hat\mu_L)$ using sliding window statistics, periodically recalculates $(\hat k^*, \hat N^*)$, and evaluates the crossover criterion (8) using an EMA-smoothed active batch occupancy $N_{\text{obs}}$ to decide the scheduling mode. A KV-aware gate prevents memory overflow during runtime.

### Key Designs

1.  **CFR Baseline Threshold $\theta_0$ (Constant Hazard Rate)**:
    - **Function**: Provides the normalized switching threshold $\theta_0 = \lim_{N\to\infty} k^*/N$ that maximizes EB throughput under the geometric output length assumption $h(t) = p_0$.
    - **Mechanism**: The average duration of the decode phase is expressed as $\mathbb{E}[T_d(k;N)] = [\beta_d N\theta - \alpha_d \ln(1-\theta)] / p_0$. Substituting this into the saturated throughput $\mathrm{TP}_{\mathrm{EB}}$ and differentiating with respect to $k$ yields the equation for $\theta_0$: $\theta_0 (1-\theta_0)^{-1} + \ln(1-\theta_0) = p_0 \alpha_p \alpha_d^{-1}$. This equation depends only on the ratio $p_0 \alpha_p / \alpha_d$ and is independent of $N$, $\mu_L$, or per-token costs $\beta_p, \beta_d$.
    - **Design Motivation**: Traditionally, $k$ is determined heuristically (e.g., v0 uses $k=1$). This work transforms it into a scalar root equation depending only on the hazard rate and fixed overhead ratio, making the threshold interpretable, portable, and estimable online without exhaustive search.

2.  **IFR Correction and Decoupled Optimization**:
    - **Function**: Extends the CFR threshold to the more common increasing-failure-rate (IFR) scenario $h(t) = p_0 + \eta t$ and decouples the "threshold selection + batch size selection" into two solvable sub-problems.
    - **Mechanism**: Using perturbation expansion on $\eta$, the optimal threshold is $\theta^* = \theta_0 + \Delta\theta + O(\eta^2)$, where $\Delta\theta = \frac{\eta(1-\theta_0)^2}{p_0^2 \theta_0}\big[\zeta(\frac{\theta_0}{1-\theta_0} - \frac{\zeta}{2}) + \frac{\beta_d N}{\alpha_d}(\zeta - \theta_0)\big]$ and $\zeta = -\ln(1-\theta_0)$. The positive correction reflects that under IFR, completions become denser over time, justifying a longer wait before switching. The maximum batch size $N^*$ is then determined under an OOM probability constraint $\le \epsilon$.
    - **Design Motivation**: Real workloads are typically IFR (EOS becomes more likely over time), but joint optimization of $(k, N)$ is intractable. Decoupling by solving for $\theta$ at the $N\to\infty$ limit and then finding the largest feasible $N$ retains closed-form solutions while keeping errors at $O(1/N)$.

3.  **EB+ Online Phase/Mode Switching Criterion**:
    - **Function**: Uses a single inequality to drive both phase switching within EB and mode switching between EB and MB.
    - **Mechanism**: The steady-state MB throughput $\mathrm{TP}_{\mathrm{MB}}(N)$ is compared with $\mathrm{TP}_{\mathrm{EB}}$. MB wins if $\beta_{\mathrm{MB}}^e - \beta_{\mathrm{EB}}^w < \frac{1}{\mu_L + \mu_O}\big[\frac{\alpha_p + \alpha_d \zeta \mu_O}{k_0^*} - \frac{\alpha_{\mathrm{MB}}(1+\mu_O)}{N}\big]$. Online, the scheduler replaces $N$ with $N_{\text{obs}}$ and looks up $\beta_{\mathrm{MB}}^e(\hat r)$ from a hardware profile table.
    - **Design Motivation**: The LHS represents the "marginal cost difference introduced by prefill-decode mixing," while the RHS represents the "fixed cost advantage of MB due to fewer kernel launches." The scheduler favors MB at light loads (lower TTFT) and EB at heavy loads with bandwidth constraints (higher TPOT).

### Loss & Training
This work involves no training; all optimizations occur at the scheduling layer. The online controller maintains sliding windows for output lengths $\mathcal{W}_O$ and input lengths $\mathcal{W}_L$. It estimates the empirical hazard rate $\hat h(t)$ and fits $\hat h(t) = \hat p_0 + \hat\eta t$ using weighted least squares. In each cycle, the scheduler solves for $\hat\theta_0$, applies the correction to get $\hat\theta^*$, and calculates $\hat k^*$ and $\hat N^*$.

## Key Experimental Results

### Main Results
Evaluations were conducted on four GPUs (B300 8.0 TB/s, H200 4.8 TB/s, RTX PRO 6000 1.792 TB/s, L40S 0.864 TB/s) using Qwen3-8B and Qwen3-30B-A3B (MoE). Comparisons include v0 (EB $k=1$), v1 (MB), and EB($\hat k^*$).

| GPU / Model | Workload | v1 (MB) | EB($\hat k^*$) | Gain vs v1 |
|---|---|---|---|---|
| RTX 6000 / Qwen3-8B | ShareGPT | 17.07 RPS | 19.68 RPS | **+15.3%** |
| RTX 6000 / Qwen3-8B | WildChat | 12.75 RPS | 14.19 RPS | **+11.3%** |
| RTX 6000 / Qwen3-8B | LongBench | 8.35 RPS | 8.66 RPS | +3.7% |
| RTX 6000 / Qwen3-8B | NuminaMath | 0.73 RPS | 0.74 RPS | +1.4% |
| RTX 6000 / Qwen3-8B | **Mean** | — | — | **+7.9%** |
| RTX 6000 / Qwen3-30B | **Mean** | — | — | +1.4% |
| H200 / Qwen3-8B | **Mean** | — | — | +1.5% |
| H200 / Qwen3-30B | **Mean** | — | — | −2.9% |

On bandwidth-constrained GPUs, EB($\hat k^*$) achieves up to 41.9% gain over v1. On H200, v1 performs better for larger models, consistent with theoretical predictions.

### Ablation Study

| Config (RTX 6000, $c=2048$) | Throughput (tok/s) | TTFT (s) | TPOT (ms) |
|---|---|---|---|
| v1 (MB) | 8,830 | 83.8 | 207.3 |
| EB($\hat k^*$) | 13,179 | 70.8 | 82.7 |
| **EB+ (Ours)** | **13,214** | 68.2 | 101.6 |
| EB+ $c=32$ | 4,930 | 0.061 | 19.5 (Match v1) |
| EB+ Distribution Drift | 9,582 (+36.4% vs v1) | — | — |

EB+ automatically reverts to MB under low concurrency to match v1's TTFT and switches to EB under high concurrency for throughput.

### Key Findings
-   **Bandwidth is Decisive**: The threshold ratio $r$ where MB's marginal cost exceeds pure decode is 80% for H200 but only 20% for RTX PRO 6000. Kernel breakdown shows the bottleneck is in Attention, not GEMM.
-   **Closed-form $\theta_0$ is Effective**: EB($\hat k^*$) outperforms the best fixed-$k$ by adjusting $N$ synchronously, validating the decoupled approximation.
-   **Scale Reduces EB Gains**: As model size increases (8B to 30B), fixed costs $\alpha$ grow, making the kernel amortization of MB more competitive.
-   **TTFT vs. TPOT Trade-off**: v1 optimizes TTFT through phase overlapping, while EB optimizes TPOT by avoiding bandwidth contention. EB+ allows tuning this via a priority margin $\delta$.

## Highlights & Insights
-   **First Closed-form Criterion for EB vs. MB**: Instead of engineering intuition, this work provides a quantifiable $\beta_{\mathrm{MB}}^e - \beta_{\mathrm{EB}}^w$ vs. $O(1/N)$ comparison, cleanly separating hardware and workload factors.
-   **Hazard Rate as a Scheduling Signal**: Using $h(t)$ to predict future completion rhythms is a novel application of survival analysis to LLM serving, outperforming reactive signals like queue length.
-   **Value of Decoupled Approximation**: Solving $\theta$ at the limit and then back-calculating $N$ provides a practical path for "jointly unsolvable" optimization problems in inference scheduling.

## Limitations & Future Work
-   **Fluid Approximation Sensitivity**: The accuracy of analytical throughput decreases under light or extremely bursty traffic where the saturated batch assumption fails.
-   **Linear Iteration Time Model**: The $T_{\text{iter}} = \alpha + \beta n_{\text{tok}}$ model simplifies complex kernel behaviors and may not fully capture MoE routing or extremely long context jitters.
-   **Single GPU Focus**: The benefits of EB+ might be diluted in disaggregated multi-card topologies where KV-cache transmission costs are significant.

## Related Work & Insights
-   **vs. vLLM v0**: v0 is a degenerate EB ($k=1$). This work generalizes it to an optimal $k = \lfloor \theta_0 N \rfloor$.
-   **vs. vLLM v1 / Sarathi**: Proves that MB is not universally optimal, especially on lower-bandwidth platforms like RTX 6000 or L40S.
-   **vs. Disaggregated Serving (DistServe/Splitwise)**: EB+ achieves similar interference elimination within a single pool without the overhead of KV-cache transfer or static resource splitting.

## Rating
-   Novelty: ⭐⭐⭐⭐
-   Experimental Thoroughness: ⭐⭐⭐⭐
-   Writing Quality: ⭐⭐⭐⭐
-   Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)
- [\[CVPR 2026\] C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination](../../CVPR2026/autonomous_driving/c2t_llm_traffic_coordination.md)
- [\[ICCV 2025\] CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving](../../ICCV2025/autonomous_driving/colmdriver_llm-based_negotiation_benefits_cooperative_autonomous_driving.md)
- [\[AAAI 2026\] Smart: A GNN-LLM Hybrid Surrogate Model for Dragonfly System Application Runtime Prediction](../../AAAI2026/autonomous_driving/smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
