---
title: >-
  [Paper Note] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding
description: >-
  [ICML 2026][LLM Efficiency][Speculative Decoding] This paper proposes PipeSD: transforming speculative decoding from sequential cloud-edge execution to a token-batch pipeline…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Speculative Decoding"
  - "Cloud-Edge"
  - "Pipeline Scheduling"
  - "Bayesian Optimization"
  - "Dynamic Programming"
date: 2026-05-08
content_hash: 8c186d552e164766
---

# PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2605.13319](https://arxiv.org/abs/2605.13319)  
**Code**: [anonymous.4open.science/r/PipeSD](https://anonymous.4open.science/r/PipeSD)  
**Area**: LLM Inference Systems / Cloud-Edge Collaboration / Speculative Decoding  
**Keywords**: Speculative Decoding, Cloud-Edge, Pipeline Scheduling, Bayesian Optimization, Dynamic Programming

## TL;DR
This paper proposes PipeSD: transforming speculative decoding from sequential cloud-edge execution to a token-batch pipeline, replacing fixed draft length with dual-threshold NAV triggering and Bayesian autotuning. On a real 5G cloud-edge testbed, PipeSD achieves 1.16×–2.16× speedup and 14–25% reduction in cloud energy consumption.

## Background & Motivation
**Background**: The bottleneck of large model inference lies in the serial dependency of autoregressive generation. Speculative decoding breaks this serialism by having a small draft model generate $N$ tokens, then a large target model verifies them via NAV in one shot. Cloud-edge collaborative deployment is naturally suited: draft runs on the edge for energy efficiency and privacy, target runs on the cloud for compute. Existing frameworks include HSL, HAT, and SpecEdge.

**Limitations of Prior Work**: (1) Existing frameworks follow a "generate all drafts → upload as a whole → NAV" sequential pipeline, causing the edge to wait for NAV feedback and the cloud to wait for draft upload, leaving both bandwidth and compute underutilized; (2) NAV triggering either uses a fixed draft length (HSL uses $N=6$), or a single confidence signal (HSL uses single token confidence, EdgeLLM uses cumulative sequence confidence), which cannot jointly reflect token difficulty—leading to either premature triggering (wasting compute) or late triggering (causing large rollbacks).

**Key Challenge**: Communication startup cost $\alpha$ is significant, so sending each token immediately (extreme pipelining) can be slower than batching; but fully sequential execution wastes waiting time. The optimal batch strategy—how many tokens to merge before sending—must be found. NAV triggering should consider both "is the whole segment still credible" (sequence confidence) and "is any token already unreliable" (single token confidence); a single signal is biased.

**Goal**: (1) Formalize token generation-communication pipeline scheduling and find the optimal batch boundaries; (2) Use dual-threshold NAV to combine sequence and token-level signals; (3) Automatically tune thresholds on the edge to adapt to dynamic network and compute.

**Key Insight**: Communication startup cost $\alpha$, per-token transmission time $\beta$, and per-token compute $\gamma$ can be measured online. The scheduling problem is a classic DAG scheduling problem, solvable by DP in $O(\hat N^2)$ time; the effect of thresholds on TPT is non-analytic but sample-efficient, so Bayesian optimization can approach optimality within 16 samples.

**Core Idea**: The combination of "DP-optimal token-batch pipelining + dual-threshold NAV + BO autotuning" pushes cloud-edge speculative decoding's compute/bandwidth utilization close to the Pareto frontier.

## Method

### Overall Architecture
A speculative round in PipeSD consists of four steps: (1) The edge draft model autoregressively generates draft tokens; (2) The edge Token-batch Pipeline Scheduler uses DP to determine batch boundaries $\mathbb B=(b_1,\dots,b_K)$ and uploads batches in real time, overlapping with generation; (3) The Dual-threshold NAV Trigger continuously monitors both single token and cumulative sequence confidence, triggering NAV if either threshold is crossed, with the BO autotuner periodically updating thresholds; (4) The cloud target model performs NAV and returns accept/reject. The system is implemented with llama-cpp-python (edge) + PyTorch + FastAPI (cloud); the edge also has an Environment Monitor that continuously measures $(\alpha,\beta,\gamma)$ and triggers DP rerun on significant changes.

### Key Designs

1. **DP-Optimal Scheduling for Token-batch Pipelining**:

    - **Function**: Given communication startup cost $\alpha$, per-token transmission time $\beta$, and per-token compute $\gamma$, find batch boundaries $\mathbb B$ that minimize total generation and transmission time.
    - **Mechanism**: For batch $k$, communication time is $t_c^{(k)}=\alpha+\beta\cdot(b_{k+1}-b_k)$, generation time is $t_{ag}^{(k)}=\gamma\cdot(b_{k+1}-b_k)$. Communication depends on "previous batch communication finished and current batch generation finished", recursively $\tau_c^{(k)}=\max\{\tau_c^{(k-1)}+t_c^{(k-1)},\tau_{ag}^{(k)}+t_{ag}^{(k)}\}$. The objective is $\min T=\tau_c^{(K)}+t_c^{(K)}-\tau_{ag}^{(1)}$. Algorithm 1 uses $dp[j]$ to represent the optimal time for the first $j$ tokens, enumerating previous batch start $i<j$ to get $dp[j]=\min_i\{\max(dp[i],\gamma j)+\alpha+\beta(j-i)\}$; backtracking yields $\mathbb B$, with complexity $O(\hat N^2)$. The paper proves this is optimal (Theorem 4.1).
    - **Design Motivation**: Naive batching (all tokens together or sending each token immediately) is not Pareto optimal since $\alpha$ is non-negligible; DP considers both amortizing $\alpha$ and the "cover window" of generation time, dynamically deciding whether batching 2 or 3 tokens is more efficient.

2. **Dual-threshold NAV Triggering Mechanism**:

    - **Function**: Simultaneously monitors "whether the whole sequence is still worth drafting" and "whether any token falls below the alert line", avoiding premature or delayed triggering caused by single-signal approaches.
    - **Mechanism**: Define cumulative sequence confidence $C_1=\prod_{n}P(D_n)$ (product of probabilities for draft tokens not yet verified) and single token confidence $P(D_n)$. For each new token, compute tentative $C_1^*=C_1\cdot P(D_n)$; if $C_1^*\le R_1$ or $P(D_n)\le R_2$, trigger NAV and reset $C_1=1$.
    - **Design Motivation**: HSL only considers $P(D_n)$, so if each token is moderately credible, NAV is never triggered—leading to over-generation; EdgeLLM only considers $C_1$, which can mask single-point failures—delaying error detection. The dual-threshold covers both failure modes.

3. **Bayesian Autotuning + Dynamic Scheduling Window**:

    - **Function**: The mapping from thresholds to TPT is non-analytic and varies with task difficulty and network jitter; BO is used to find near-optimal $(R_1,R_2)$ online.
    - **Mechanism**: The BO autotuner aims to minimize average TPT, sampling $(R_1,R_2,\text{TPT})$ triplets and using a Gaussian process to predict the next optimal query; near-optimality is reached within 16 samples. When TPT changes significantly (monitor triggered), BO is rerun; when $(\alpha,\beta,\gamma)$ change, DP is rerun. A scheduling window $\hat N$ is introduced (sliding average of the last 100 draft sequence lengths, initially 20), with two rules: when NAV is triggered, any untransmitted tokens are immediately sent as a batch; while waiting for NAV, continue generating the next window of drafts to further overlap.
    - **Design Motivation**: All adaptive logic is placed on the edge (no dependency on the cloud framework), making PipeSD compatible with any cloud inference backend such as vLLM or TensorRT-LLM; BO, rather than grid/random search, ensures sample efficiency, aligning with the "edge must be lightweight" deployment constraint.

### Loss & Training
PipeSD has no training loss (inference framework). Key parameters: DP algorithm inputs $(\hat N,\alpha,\beta,\gamma)$ are measured in real time by the Environment Monitor; BO autotuner converges in 16 samples by default; average draft length is about 6 for programming tasks, 4 for math tasks; NAV thresholds $(R_1,R_2)$ are searched by BO in the 0–1 range.

## Key Experimental Results

### Main Results
Four scenarios (Scenario 1: laptop + static 20/200Mbps; Scenarios 2/3: simulated phone/IoT compute 2.5/1.2 GHz; Scenario 4: dynamic bandwidth 10–80 Mbps), two model pairs (DeepSeek-Coder 1.3B→6.7B, TinyLlama 1.1B→Llama-2 7B), two datasets (HumanEval, GSM8K), compared with Vanilla / HSL / EdgeLLM:

| Scenario | Dataset | Vanilla TPT(ms) | HSL | EdgeLLM | PipeSD | vs Vanilla |
|----------|---------|-----------------|-----|---------|--------|------------|
| 1 | HumanEval | 194 | 155 | 153 | 129 | 1.50× |
| 1 | GSM8K | 193 | 174 | 169 | 145 | 1.33× |
| 3 (IoT) | HumanEval | 306 | 244 | 201 | 152 | 2.01× |
| 3 (IoT) | GSM8K | 402 | 296 | 231 | 186 | 2.16× |
| 4 (Dynamic BW) | HumanEval | 160 | 132 | 127 | 108 | 1.48× |

Cloud energy consumption (Scenario 1, per 100 accepted tokens):

| Dataset | Vanilla(J) | HSL | EdgeLLM | PipeSD | Reduction vs EdgeLLM |
|---------|------------|-----|---------|--------|----------------------|
| HumanEval | 68 | 71 | 75 | 56 | 25.3% |
| GSM8K | 98 | 102 | 100 | 84 | 16.0% |

### Ablation Study
**BO Tuning Comparison (HumanEval Scenario 1, TPT ms):**

| Tuning Strategy | HumanEval | GSM8K |
|-----------------|-----------|-------|
| BO Autotuner | 129 | 145 |
| Grid Search | 139 | 155 |
| Random Search | 148 | 162 |

**Bandwidth Sensitivity (HumanEval Scenario 1, PipeSD vs Vanilla):** 10 Mbps 1.32×, 20 Mbps 1.47×, 40 Mbps 1.45×, 80 Mbps 1.34×; after 80 Mbps, communication is no longer the bottleneck and speedup saturates.

### Key Findings
- The weaker the compute (IoT scenario), the higher the speedup (2.16×), confirming the intuition that "pipeline mainly benefits from hiding communication"—the slower the edge, the more generation time can cover the communication window.
- DP rerun overhead is <0.013% of total time, essentially negligible; BO converges in only 16 samples, so autotuning cost is also negligible.
- The dual-threshold mechanism is the main contributor to ECS reduction: it reduces invalid NAV requests (fewer target model computations on the cloud), so energy savings exceed TPT reduction.

## Highlights & Insights
- **Reframing cloud-edge speculative decoding as a scheduling problem**: Previous works patched individual components (HSL modifies triggering, HAT modifies accuracy constraints), while PipeSD formalizes it as DP pipelining + BO adaptation, providing a complete system-level abstraction.
- **DP algorithm is simple yet precise**: Complexity $O(\hat N^2)$, with $\hat N\sim 20$, is fully online; this "cheap exact algorithm + online monitoring-triggered rerun" pattern is well-suited for system optimization with slowly drifting parameters.
- **Dual-threshold approach is transferable**: Any early stopping/triggering problem involving "cumulative score + instantaneous score" can benefit—for example, early stopping in multi-step chain-of-thought reasoning, or truncation in long-sequence retrieval.
- **BO tuning as a core component**: Making Bayesian optimization a general-purpose "thresholding" module on the edge is lighter than RL-trained controllers and enables zero cold-start for new deployment environments.

## Limitations & Future Work
- Only implemented for a single draft-target pair and single client; for multi-client or heterogeneous batching with multiple draft models, the DP formula needs to be re-derived.
- BO uses global average TPT as the objective, but draft acceptance rates differ greatly between tasks (code vs math), so per-task thresholds or multi-task BO may be needed.
- Edge energy consumption is only theoretically analyzed, not empirically measured; frequent BO + DP triggering on CPU may impact power in battery-constrained scenarios, which remains to be validated.
- Security/privacy aspects are not considered: the dual-threshold exposes per-token confidence from the draft model, which could be a side-channel for inferring private data—this is not discussed in the paper.

## Related Work & Insights
- **vs HSL (hao2024)**: HSL triggers on single token confidence, uses fixed draft length, and lacks pipelining; this method uses dual thresholds + DP pipelining, achieving 1.61× speedup in Scenario 3.
- **vs EdgeLLM (xu2025)**: EdgeLLM uses cumulative sequence confidence and continues generation while waiting for NAV; PipeSD adds token-level thresholds and DP-optimal batching, reducing ECS by 16–25%.
- **vs HAT, SpecEdge**: HAT focuses on accuracy constraints, SpecEdge on multi-edge collaboration; these are orthogonal to PipeSD and can be combined.
- **vs Medusa, EAGLE (cloud-side SD)**: These works improve the acceptance rate of the draft head itself; PipeSD improves deployment and triggering, and can be combined for cloud-edge scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of DP pipelining, dual thresholds, and BO autotuner is the first Pareto-complete framework for cloud-edge speculative decoding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Real cloud-edge testbed, 4 scenarios × 2 models × 2 datasets + bandwidth sweep + energy measurement.
- Writing Quality: ⭐⭐⭐⭐ Consistent from bottleneck analysis to DP derivation to system implementation, with clear notation.
- Value: ⭐⭐⭐⭐ A directly reusable framework for mobile/IoT cloud-edge inference in the 5G era, with open-source code as a bonus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)
- [\[ICML 2026\] KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](../../NeurIPS2025/llm_efficiency/flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)

</div>

<!-- RELATED:END -->
