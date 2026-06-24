---
title: >-
  [Paper Note] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models
description: >-
  [ACL 2025][LLM Efficiency][speculative decoding] This paper proposes a drop-in adaptive solution for speculative decoding that dynamically adjusts the speculative window size $\gamma$ (and potentially the choice of draft models) during inference, thereby maximizing the end-to-end speedup of speculative decoding under diverse input distributions.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "speculative decoding"
  - "window adaptation"
  - "LLM inference acceleration"
  - "draft model selection"
  - "end-to-end speed optimization"
date: 2026-05-08
content_hash: b93602adafa4d4da
---

# A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: speculative decoding, window adaptation, LLM inference acceleration, draft model selection, end-to-end speed optimization

## TL;DR

This paper proposes a drop-in adaptive solution for speculative decoding that dynamically adjusts the speculative window size $\gamma$ (and potentially the choice of draft models) during inference, thereby maximizing the end-to-end speedup of speculative decoding under diverse input distributions.

## Background & Motivation

**Background**: Large Language Models (LLMs) built on the Transformer architecture face extremely high demands for memory and computational resources during real-time inference. Speculative decoding is one of the mainstream LLM inference acceleration strategies, where a smaller draft model predicts within a given window size $\gamma$, and the target model verifies these predicted tokens in parallel, thus partially parallelizing the serial process of autoregressive generation.

**Limitations of Prior Work**: Existing speculative decoding methods usually employ a fixed window size $\gamma$ and a fixed draft model. However, the optimal $\gamma$ is highly dependent on the input data distribution. When the draft model is well-aligned with the target model on certain inputs, a larger $\gamma$ yields a greater speedup. Conversely, when there is a significant discrepancy between them, a larger $\gamma$ wastes computational resources because a higher number of rejected tokens diminishes the acceleration effect. Consequently, static configurations of speculative decoding cannot achieve optimal performance across all scenarios in practical deployments.

**Key Challenge**: The choice of the speculation window size $\gamma$ faces a fundamental trade-off: a larger $\gamma$ increases the number of potentially accepted tokens in each verification step (high throughput) but also increases the probability of draft predictions being rejected (low acceptance rate), leading to wasted computations. Different input prompts, generation phases, and even differing draft-target model combinations affect this equilibrium.

**Goal**: To design a scheme capable of dynamically adjusting speculative decoding parameters in real-time during inference. The requirements are: (1) no modification to the core logic of the existing speculative decoding framework, (2) dynamic adjustment of the window size based on online-observed token acceptance rates, and (3) convenient integration as a drop-in module.

**Key Insight**: The authors observe that the token acceptance rate varies significantly across different inputs and generation phases. Therefore, online statistical information (e.g., the acceptance rates of recent steps) can be utilized to predict the optimal window size. Rather than searching offline for a fixed parameter, the system should dynamically adapt based on actual runtime performance.

**Core Idea**: By monitoring the token acceptance rate of speculative decoding online, a lightweight adaptive controller is designed to adjust the window size $\gamma$ in real-time, ensuring that speculative decoding consistently maintains near-optimal acceleration across inputs of different distributions.

## Method

### Overall Architecture

The proposed method is built on top of the standard speculative decoding framework. In standard speculative decoding, the draft model generates $\gamma$ candidate tokens at once, and the target model verifies all candidate tokens in a single forward pass, accepting the first $k$ tokens ($k \le \gamma$) that align with the target model's distribution. This adaptive scheme acts as an external control module, monitoring the token acceptance of each verification step and dynamically adjusting the window size $\gamma$ for the next speculation round.

### Key Designs

1. **Online Acceptance Rate Monitoring Module**:

    - **Function**: Track the token acceptance rate of each verification step in speculative decoding in real-time.
    - **Mechanism**: Maintain a sliding window to record the ratio of accepted tokens to draft length in each of the past $W$ verification steps. The current average acceptance rate $\bar{\alpha}$ is calculated using Exponential Moving Average (EMA) or a simple moving average. When $\bar{\alpha}$ is high, indicating good alignment between the draft and target models on the current input, $\gamma$ is increased for greater speedup. When $\bar{\alpha}$ is low, $\gamma$ is decreased to reduce useless computation.
    - **Design Motivation**: Avoid sub-optimal performance caused by using a fixed $\gamma$, allowing the system to make adjustment decisions based on actual runtime data.

2. **Window Size Adaptive Controller**:

    - **Function**: Determine the window size for the next round of speculation based on the acceptance rate information collected by the monitoring module.
    - **Mechanism**: Establish a mapping relationship between the acceptance rate and the optimal window size. In high acceptance rate regions (e.g., $\bar{\alpha} > 0.8$), $\gamma$ is set to a large value (e.g., 8-10); in low acceptance rate regions (e.g., $\bar{\alpha} < 0.5$), $\gamma$ is set to a small value (e.g., 2-3); linear interpolation or lookup tables are used for the intermediate regions. The specific mapping can be derived through theoretical analysis: the expected speedup of speculative decoding with respect to $\gamma$ and acceptance rate $\alpha$ is $\text{speedup} \approx \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(c \cdot \gamma + 1)}$, where $c$ is the latency ratio between the draft and target models. Taking the derivative of this formula with respect to $\gamma$ yields the optimal $\gamma$ value.
    - **Design Motivation**: A theory-guided adaptive strategy is more reliable than heuristic methods and incurs negligible computational overhead (only a few floating-point operations).

3. **Drop-In Integration Interface**:

    - **Function**: Ensure that the adaptive module can be seamlessly integrated into any existing speculative decoding implementation.
    - **Mechanism**: Encapsulate the controller as an independent component. It only requires querying the controller for the recommended $\gamma$ value before each speculation round and feeding the acceptance results back to the controller after verification. It does not modify any internal logic of the draft or target models.
    - **Design Motivation**: Lower the barrier to entry, enabling existing speculative decoding systems to gain adaptive capabilities with zero modifications.

### Loss & Training

The proposed method is an inference-time adaptive scheme that does not involve any additional training processes. The parameters of the controller (such as the EMA decay coefficient, upper/lower bounds of window size, etc.) can be set using a small number of calibration samples or directly using default values derived from theory.

## Key Experimental Results

### Main Results

| Model Combination | Task | Optimal Fixed $\gamma$ | Adaptive Method | Gain |
|----------|------|-----------|-----------|-----------|
| LLaMA-2-70B + LLaMA-2-7B | Code Generation | 1.85x | 2.12x | +14.6% |
| LLaMA-2-70B + LLaMA-2-7B | Text Summarization | 1.72x | 1.95x | +13.4% |
| LLaMA-2-70B + LLaMA-2-7B | QA | 1.91x | 2.08x | +8.9% |
| Vicuna-33B + Vicuna-7B | Code Generation | 1.78x | 2.05x | +15.2% |
| Vicuna-33B + Vicuna-7B | Dialogue | 1.65x | 1.88x | +13.9% |

### Ablation Study

| Configuration | Average Speedup | Description |
|------|-----------|------|
| Fixed $\gamma=4$ (Default) | 1.72x | Static configuration |
| Fixed $\gamma=\text{Optimal}$ (Oracle) | 1.85x | Manual tuning for each task |
| Adaptive (Window Adjustment Only) | 2.02x | Online adjusting $\gamma$ |
| Adaptive (Full Scheme) | 2.08x | Includes all components |
| No Sliding Window (Instantaneous Rate) | 1.93x | Large fluctuations in $\gamma$, unstable performance |

### Key Findings

- The adaptive method outperforms the fixed $\gamma$ configuration across all tested model combinations and task types, improving the speedup ratio by about 12-15% on average.
- In mixed-task scenarios where the input distribution varies significantly, the advantage of the adaptive method is even more pronounced because a single fixed $\gamma$ cannot simultaneously accommodate multiple distributions.
- The additional computational overhead of the controller is negligible, requiring only microsecond-level time per decision.

## Highlights & Insights

- Shifting speculative decoding parameter tuning from offline search to online adaptation is a highly practical idea. Since the input distribution keeps changing in actual deployment environments, static configurations are destined to be sub-optimal.
- The drop-in design philosophy is worth emulating—good system optimization should be transparent to users and non-intrusive to existing systems.
- Utilizing the token acceptance rate as an online proxy metric to guide parameter adjustment is both intuitive and highly efficient, avoiding complex online learning approaches.

## Limitations & Future Work

- This work mainly focuses on the adaptation of the window size $\gamma$. However, in scenarios where multiple draft models are available, dynamically selecting the draft model is also an important future direction.
- In terms of attention KV cache management, a dynamically changing $\gamma$ may present challenges for memory allocation.
- The current method assumes that the acceptance rate changes smoothly; there might be an adaptation latency when abrupt distribution shifts occur.
- Future work could explore joint optimization with other acceleration techniques such as quantization and pruning.

## Related Work & Insights

- **vs SpecDec**: The standard method uses a fixed $\gamma$, whereas this work extends it to an adaptive version, boosting the speedup ratio without adding significant overhead.
- **vs Medusa**: Medusa achieves parallel speculation by appending multiple prediction heads to the target model, which requires extra training. The proposed method requires no training and offers better compatibility.
- **vs EAGLE**: EAGLE utilizes feature-level speculation. The proposed method is orthogonal to it and can further introduce adaptive windows on top of EAGLE.

## Rating

- Novelty: ⭐⭐⭐ The idea is intuitive yet practical; the concept of an adaptive speculative window is of great engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing across multiple model combinations and tasks.
- Writing Quality: ⭐⭐⭐⭐ ACL long-paper style, well-structured and clear.
- Value: ⭐⭐⭐⭐ Speculative decoding is currently a hot topic in LLM deployment, and this adaptive scheme holds direct engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[ICLR 2026\] Speculative Speculative Decoding](../../ICLR2026/llm_efficiency/speculative_speculative_decoding.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)

</div>

<!-- RELATED:END -->
