---
title: >-
  [Paper Note] DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services
description: >-
  [ACL 2025][LLM (Other)][Device-Server Collaboration] This work proposes DiSCo, a device-server collaborative LLM inference scheduler that optimizes user-perceived Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT) under cost constraints through cost-aware request dispatching and token-level migration mechanisms.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Device-Server Collaboration"
  - "QoE"
  - "TTFT"
  - "TBT"
  - "Token Migration"
  - "LLM Serving"
date: 2026-05-08
content_hash: 7250c9df50e5427c
---

# DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services

**Conference**: ACL 2025  
**arXiv**: [2502.11417](https://arxiv.org/abs/2502.11417)  
**Code**: Not open-sourced  
**Area**: LLM Inference Systems / Device-Server Collaboration  
**Keywords**: Device-Server Collaboration, QoE, TTFT, TBT, Token Migration, LLM Serving  

## TL;DR

This work proposes DiSCo, a device-server collaborative LLM inference scheduler that optimizes user-perceived Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT) under cost constraints through cost-aware request dispatching and token-level migration mechanisms.

## Background & Motivation

**Core Problem:** LLM-based text streaming services face severe challenges in Quality of Experience (QoE) and operational cost. TTFT (Time-to-First-Token) and TBT (Time-Between-Tokens) are critical metrics for real-time interaction, yet existing deployment paradigms fail to optimize both simultaneously.

**Limitations of Prior Work:** (1) Server-only deployment is highly expensive, and TTFT fluctuates severely due to request queuing, batching contention, and network latency (e.g., GPT-4o-mini's TTFT spikes from 0.3s to several seconds under high load); (2) Device-only deployment is constrained by limited hardware resources, suffering from slow prefill time for long prompts and high energy consumption (e.g., running a 7B model on an iPhone can last for less than 2 hours).

**Design Motivation:** It is observed that server-side TTFT is unpredictable but weakly correlated with prompt length, whereas device-side TTFT is predictable and scales linearly with prompt length. Furthermore, the token generation speed of both deployment paradigms exceeds the human consumption rate. Using these complementary characteristics motivates a device-server collaborative scheduling approach.

## Method

### Overall Architecture

Operating as a middleware, DiSCo consists of two core controllers: the **Dispatch Controller**, which determines the initial execution endpoint of a request, and the **Migration Controller**, which dynamically switches the execution endpoint during the generation process. Together, they optimize TTFT and TBT under cost constraints.

### Key Designs

1. **Unified Cost Model and Cost-Aware Dispatching Strategy:** A dynamic exchange rate $\lambda$ is introduced to unify server monetary costs and device energy costs. In device-constrained scenarios, a waiting-time strategy is adopted: execution is first attempted on the server-side, and device-side inference is initiated after a waiting time $w(l)$, allocating the budget in two stages (tail protection + average optimization). In server-constrained scenarios, requests are routed based on a prompt length threshold $l_{th}$—short prompts are sent to the device to save server budget, while long prompts are processed in parallel on both ends to select the faster result.

2. **Token-Level Migration Framework:** A token buffer $B = r_c \times t_m$ is constructed by exploiting the difference between the token generation rate $r_g$ and the human consumption rate $r_c$. Migration is triggered when the buffer accumulates enough tokens to cover the migration overhead; the source endpoint ceases generation, and the target endpoint seamlessly takes over. Migration is only executed when the expected cost savings exceed the migration overhead: $C_{migration} = \Delta c_{decode} \cdot l_{remaining} > \text{Overhead}_{migration}$.

3. **Efficient Token Transmission:** When endpoints share the same vocabulary, token IDs are transmitted instead of the full text, reducing data volume by 35-54%. For different vocabularies, the data is converted to text first and then re-tokenized. Transmitting intermediate states (e.g., KV cache) is avoided because endpoints often utilize different model architectures.

### Loss & Training

The optimization goal is to minimize average and tail TTFT while maintaining stable TBT, under the cost constraint $\mathbb{E}[I_d(l)l] \leq b \cdot \mathbb{E}[l]$ or $\mathbb{E}[I_s(l)l] \leq b \cdot \mathbb{E}[l]$.

## Experimental Results

### Main Results

Evaluation of four commercial LLM services (GPT-4o-mini, DeepSeek-V2.5, Cohere Command, LLaMA-3-70b) and three device configurations:

| Platform/Model | Constraint | Tail TTFT Reduction (Pixel 7 Pro B-1.1B) | Tail TTFT Reduction (Xiaomi 14 Q-0.5B) |
|-----------|------|--------------------------------------|-------------------------------------|
| GPT | Server | 23.85% | 44.04% |
| GPT | Device | 26.39% | 16.32% |
| LLaMA | Server | 11.08% | 26.29% |
| LLaMA | Device | 35.67% | 21.29% |
| Command | Server | 47.93% | 52.23% |
| Command | Device | 34.78% | 24.42% |

### Ablation Study

| Aspect | Conclusion |
|------|------|
| Migration Mechanism | Up to 72.7% cost reduction in device-constrained scenarios, and up to 83.6% in server-constrained scenarios |
| Request Arrival Interval | Benefits persist under real-world DiffusionDB workload patterns |
| Migration Impact on Generation Quality | Evaluations by three LLM judges (GPT-4o, Gemini, Qwen) show consistent quality retention |
| Scalability | DiSCo-S requires only 9.08ms on 100K samples, and DiSCo-D requires 14.86ms |

### Key Findings

- DiSCo reduces average TTFT by 6-78% and tail TTFT by 11-52%. The migration mechanism saves up to 84% in serving costs while preserving comparable QoE.
- Only a few tokens (3-17 on average) are delayed during migration, which is negligible compared to generation lengths of hundreds or thousands of tokens, leaving the P99 TBT unaffected.
- The stability of device-side TTFT and TBT is significantly superior to that of the server side, providing a reliable predictive foundation for the collaborative strategy.

## Highlights & Insights

- Presents the first device-server collaborative LLM inference scheduling paradigm, rather than basic routing and dispatching.
- Elegantly designs a token-level migration mechanism that leverages the delta between generation and consumption speeds to enable imperceptible switching.
- Backed by extensive empirical data from real-world commercial LLM services (GPT, DeepSeek, etc.).

## Limitations & Future Work

- Focuses on application scenarios where device-side LLMs have achieved sufficient accuracy (e.g., chat, translation), but is not applicable to complex reasoning tasks.
- Uses a FLOPs-based linear model for device energy consumption, whereas real-world energy consumption is more complex and influenced by factors such as battery health and temperature.
- Only considers single-device scenarios, leaving coordination overheads and resource allocation problems in multi-device collaboration unexplored.
- Does not consider privacy preservation issues, assuming users accept the transmission of data between the device and the server.

## Related Work & Insights

- **Device-Server Collaborative LLMs:** EdgeShard (Zhang et al., 2024) and WDMoE (Xue et al., 2024) deploy model shards across endpoints; LLMCad (Xu et al., 2023) leverages device-side models to reduce server costs.
- **LLM Routing:** Ong et al. (2024) and Ding et al. (2024) route to different models based on query complexity, but do not optimize token delivery metrics.
- **LLM Inference Optimization:** vLLM (Kwon et al., 2023) and Sarathi-Serve (Agrawal et al., 2024) optimize server-side throughput-latency trade-offs.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Practicality | 9 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Overall Rating | 8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)

</div>

<!-- RELATED:END -->
