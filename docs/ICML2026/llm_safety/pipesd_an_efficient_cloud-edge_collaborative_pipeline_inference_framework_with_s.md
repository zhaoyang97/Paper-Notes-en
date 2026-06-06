---
title: >-
  [Paper Note] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding
description: >-
  [ICML 2026][LLM Safety][Speculative Decoding] A cloud-edge pipeline inference framework named PipeSD is proposed, which transforms speculative decoding from sequential execution into a token-batch pipeline. By replacing…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Speculative Decoding"
  - "Cloud-Edge"
  - "Pipeline Scheduling"
  - "Bayesian Optimization"
  - "Dynamic Programming"
date: 2026-05-08
content_hash: 3ddf7d9532f0f917
---

# PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2605.13319](https://arxiv.org/abs/2605.13319)  
**Code**: [anonymous.4open.science/r/PipeSD](https://anonymous.4open.science/r/PipeSD)  
**Area**: LLM Inference Systems / Cloud-Edge Synergy / Speculative Decoding  
**Keywords**: Speculative Decoding, Cloud-Edge, Pipeline Scheduling, Bayesian Optimization, Dynamic Programming

## TL;DR
A cloud-edge pipeline inference framework named PipeSD is proposed, which transforms speculative decoding from sequential execution into a token-batch pipeline. By replacing fixed draft lengths with a dual-threshold NAV trigger and Bayesian autotuning, PipeSD achieves 1.16×–2.16× acceleration and a 14–25% reduction in cloud energy consumption on 5G testbeds.

## Background & Motivation
**Background**: The bottleneck of LLM inference lies in the serial dependency of autoregressive generation. Speculative decoding (SD) breaks this seriality by using "small draft model generation of $N$ tokens $\rightarrow$ one-time large target model NAV verification." Cloud-edge collaborative deployment is a natural fit: draft models are deployed at the edge for energy efficiency and privacy, while target models remain in the compute-rich cloud. Previous frameworks include HSL, HAT, and SpecEdge.

**Limitations of Prior Work**: (1) Existing frameworks follow a sequential workflow of "generate draft $\rightarrow$ upload $\rightarrow$ NAV," leading to idle bandwidth and compute as the edge waits for NAV and the cloud waits for uploads. (2) NAV triggering relies either on fixed draft lengths (e.g., $N=6$ in HSL) or a single confidence signal (per-token confidence in HSL or cumulative sequence confidence in EdgeLLM), which fails to jointly reflect token complexity, resulting in either premature triggering or delayed error detection.

**Key Challenge**: The communication startup overhead $\alpha$ is significant, making extreme pipelining (sending each token immediately) slower than batching. However, fully sequential execution wastes waiting time. The challenge is identifying the optimal batch strategy for token transmission. Furthermore, NAV triggering requires a balance between sequence-level reliability and single-token sanity, which a single signal cannot achieve.

**Goal**: (1) Formalize the token generation-communication pipeline scheduling and solve for optimal batch boundaries; (2) Implement a dual-threshold NAV trigger considering both sequence and per-token signals; (3) Automatically tune thresholds at the edge to adapt to dynamic network and compute conditions.

**Key Insight**: Communication startup overhead $\alpha$, per-token transmission time $\beta$, and per-token compute time $\gamma$ can be measured online. The scheduling problem is a DAG scheduling task solvable via DP in $O(\hat N^2)$ time. Although the threshold's impact on TPT (Time Per Token) is non-analytical, samples are cheap, allowing Bayesian Optimization (BO) to approximate the optimum within 16 samples.

**Core Idea**: By combining "DP-optimal token-batch pipelining + dual-threshold NAV + BO autotuning," PipeSD pushes cloud-edge speculative decoding performance toward the Pareto frontier of compute and bandwidth utilization.

## Method

### Overall Architecture
A speculative round in PipeSD consists of four steps: (1) The edge draft model generates draft tokens autoregressively; (2) The Token-batch Pipeline Scheduler uses DP-determined batch boundaries $\mathbb B=(b_1,\dots,b_K)$ to pack and upload tokens immediately, overlapping with the generation process; (3) The Dual-threshold NAV Trigger monitors per-token and cumulative sequence confidence, triggering NAV if either exceeds boundaries, while the BO autotuner periodically updates thresholds; (4) The cloud target model performs NAV and returns accept/reject status. The system is implemented using llama-cpp-python (edge) + PyTorch + FastAPI (cloud). An Environment Monitor at the edge measures $(\alpha,\beta,\gamma)$ and triggers DP re-runs upon significant changes.

### Key Designs

1. **Token-batch Pipeline DP Optimal Scheduling**:
    - **Function**: Solves for batch boundaries $\mathbb B$ that minimize the "total generation + transmission duration" given $\alpha, \beta, \gamma$.
    - **Mechanism**: Communication time for batch $k$ is $t_c^{(k)}=\alpha+\beta\cdot(b_{k+1}-b_k)$ and generation time is $t_{ag}^{(k)}=\gamma\cdot(b_{k+1}-b_k)$. Communication depends on the completion of the previous batch transmission and the current batch generation: $\tau_c^{(k)}=\max\{\tau_c^{(k-1)}+t_c^{(k-1)},\tau_{ag}^{(k)}+t_{ag}^{(k)}\}$. The objective is $\min T=\tau_c^{(K)}+t_c^{(K)}-\tau_{ag}^{(1)}$. Algorithm 1 uses $dp[j]$ to represent the optimal duration for the first $j$ tokens, iterating over previous batch start $i < j$: $dp[j]=\min_i\{\max(dp[i],\gamma j)+\alpha+\beta(j-i)\}$. Backtracking yields $\mathbb B$ in $O(\hat N^2)$.
    - **Design Motivation**: Naive batching is not Pareto optimal due to non-negligible $\alpha$. DP considers $\alpha$ amortization and generation duration "masking windows" simultaneously to determine the most cost-effective batch size.

2. **Dual-threshold NAV Triggering Mechanism**:
    - **Function**: Monitors both whether the sequence is worth continuing and if a specific token has fallen below safety levels to avoid suboptimal triggering.
    - **Mechanism**: Defines cumulative sequence confidence $C_1=\prod_{n}P(D_n)$ (product of probabilities for unverified draft tokens) and per-token confidence $P(D_n)$. For each new token, it computes $C_1^*=C_1\cdot P(D_n)$. If $C_1^*\le R_1$ or $P(D_n)\le R_2$, NAV is triggered and $C_1$ is reset to 1.
    - **Design Motivation**: HSL only monitors $P(D_n)$, leading to over-generation when tokens are moderately reliable. EdgeLLM only monitors $C_1$, which can mask single-point failures. Dual thresholds cover both failure modes.

3. **Bayesian Autotuner + Dynamic Scheduling Window**:
    - **Function**: Since the threshold-to-TPT mapping is non-analytical and varies by task and network, BO is used to find near-optimal threshold pairs $(R_1,R_2)$ online.
    - **Mechanism**: The BO autotuner minimizes average TPT by sampling $(R_1,R_2,\text{TPT})$ triplets and using Gaussian Processes to predict the next query point, converging within 16 samples. BO is re-run if TPT changes significantly. The scheduling window $\hat N$ uses a moving average of recent draft lengths.
    - **Design Motivation**: Keeping adaptation logic at the edge ensures compatibility with any cloud backend (e.g., vLLM). BO is more sample-efficient than grid or random search, meeting the lightweight edge deployment constraint.

### Loss & Training
Ours is an inference framework and does not involve training losses. Key parameters: DP input $(\hat N,\alpha,\beta,\gamma)$ is measured in real-time. BO autotuner typically converges in 16 samples. Average draft lengths are ~6 for coding and ~4 for math tasks. NAV thresholds $(R_1,R_2)$ are searched in the range $(0,1)$.

## Key Experimental Results

### Main Results
Evaluated in 4 scenarios (Scenario 1: Laptop + 20/200Mbps; Scenario 2/3: Phone/IoT at 2.5/1.2 GHz; Scenario 4: Dynamic 10–80 Mbps), using 2 model pairs (DeepSeek-Coder 1.3B $\rightarrow$ 6.7B, TinyLlama 1.1B $\rightarrow$ Llama-2 7B) on HumanEval and GSM8K. Comparison against Vanilla, HSL, and EdgeLLM:

| Scenario | Dataset | Vanilla TPT(ms) | HSL | EdgeLLM | PipeSD | vs. Vanilla |
|------|--------|------|------|---------|--------|--------------|
| 1 | HumanEval | 194 | 155 | 153 | 129 | 1.50× |
| 1 | GSM8K | 193 | 174 | 169 | 145 | 1.33× |
| 3 (IoT) | HumanEval | 306 | 244 | 201 | 152 | 2.01× |
| 3 (IoT) | GSM8K | 402 | 296 | 231 | 186 | 2.16× |
| 4 (Dyn-BW) | HumanEval | 160 | 132 | 127 | 108 | 1.48× |

Cloud energy consumption (Scenario 1, per 100 accepted tokens):

| Dataset | Vanilla(J) | HSL | EdgeLLM | PipeSD | Reduction vs EdgeLLM |
|--------|------|------|---------|--------|------|
| HumanEval | 68 | 71 | 75 | 56 | 25.3% |
| GSM8K | 98 | 102 | 100 | 84 | 16.0% |

### Ablation Study
**Comparison of Autotuning Strategies (Scenario 1, TPT ms):**

| Strategy | HumanEval | GSM8K |
|----------|-----------|-------|
| BO Autotuner | 129 | 145 |
| Grid Search | 139 | 155 |
| Random Search | 148 | 162 |

**Bandwidth Sensitivity (Scenario 1, PipeSD vs Vanilla):** 1.32× at 10 Mbps, 1.47× at 20 Mbps, 1.45× at 40 Mbps, 1.34× at 80 Mbps. Gains saturate beyond 80 Mbps as communication is no longer the bottleneck.

### Key Findings
- Lower compute capacity (IoT scenario) yields higher speedup (2.16×), confirming that pipelining primarily benefits from hiding communication—the slower the edge, the larger the communication window that can be masked.
- DP overhead is $<0.013\%$ of total time, making it virtually free. BO converges quickly, ensuring negligible adaptation cost.
- The dual-threshold mechanism is the main contributor to cloud energy reduction, as it minimizes invalid NAV requests (reducing cloud target model compute).

## Highlights & Insights
- **Formalized Cloud-Edge SD as a Scheduling Problem**: Unlike prior works that applied heuristic patches, PipeSD formalizes the process as a DP pipeline + BO adaptation, providing a system-level abstraction.
- **Precision vs. Cost in DP**: The $O(\hat N^2)$ complexity is perfectly suited for online execution. This "cheap exact algorithm + online trigger" pattern is ideal for system optimization where parameters drift.
- **Generalizability of Dual Thresholds**: The "cumulative score + point score" logic can be applied to early-exit in Chain-of-Thought or truncation in long-sequence retrieval.
- **BO as a Lightweight Component**: Using BO as a general edge "thresholder" is lighter than training RL controllers and requires zero cold-start for new environments.

## Limitations & Future Work
- Implementation is limited to single draft-target pairs and single clients; heterogeneous batching for multiple clients requires new DP derivations.
- BO currently targets global average TPT, but variance across tasks (e.g., code vs. math) suggests a need for per-task thresholds or multi-task BO.
- Edge energy consumption was analyzed theoretically but not measured; the impact of frequent BO/DP runs on battery power remains to be verified.
- Privacy implications: Dual thresholds expose per-token draft confidence, which could potentially act as a side-channel for inferring private data.

## Related Work & Insights
- **vs. HSL**: HSL uses single-token confidence and fixed lengths without pipelining; Ours uses dual thresholds and DP pipelining, achieving 1.61× faster TPT in Scenario 3.
- **vs. EdgeLLM**: EdgeLLM uses cumulative confidence and continuous generation; PipeSD adds per-token thresholds and DP-optimal batching, reducing cloud energy by up to 25%.
- **vs. Medusa/EAGLE**: These optimize the draft acceptance rate itself. PipeSD focuses on deployment and triggering, making it complementary to improved draft architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ (First complete Pareto-optimal framework for cloud-edge SD via DP and BO).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Real-world testbeds, multiple scenarios, bandwidth and energy analysis).
- Writing Quality: ⭐⭐⭐⭐ (Clear consistency from bottleneck analysis to DP derivation).
- Value: ⭐⭐⭐⭐ (Ready-to-use framework for 5G cloud-edge synergy with open-source code).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](../../AAAI2026/llm_safety/prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)
- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](../../ACL2026/llm_safety/fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ICML 2026\] Efficient DP-SGD for LLMs with Randomized Clipping](efficient_dp-sgd_for_llms_with_randomized_clipping.md)
- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)

</div>

<!-- RELATED:END -->
