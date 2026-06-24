---
title: >-
  [Paper Note] Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously
description: >-
  [CVPR 2025][Video Understanding][Streaming Video Understanding] The Video Streaming Thinking (VST) paradigm is proposed to alternate between "watching" and "thinking" during video playback. The model generates intermediate reasoning chains while receiving video frames, amortizing CoT computation into the pre-query phase. This achieves a state-of-the-art (SOTA) score of 79.5% on StreamingBench while maintaining real-time responsiveness (0.56s QA latency).
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Streaming Video Understanding"
  - "Chain of Thought"
  - "Reinforcement Learning"
  - "Knowledge Graph"
  - "Online Inference"
date: 2026-05-08
content_hash: 77b274181bb88a04
---

# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

**Conference**: CVPR 2025  
**arXiv**: [2603.12262](https://arxiv.org/abs/2603.12262)  
**Code**: [https://github.com/1ranGuan/VST](https://github.com/1ranGuan/VST)  
**Area**: Video Understanding  
**Keywords**: Streaming Video Understanding, Chain of Thought, Reinforcement Learning, Knowledge Graph, Online Inference

## TL;DR

The Video Streaming Thinking (VST) paradigm is proposed to alternate between "watching" and "thinking" during video playback. The model generates intermediate reasoning chains while receiving video frames, amortizing CoT computation into the pre-query phase. This achieves a state-of-the-art (SOTA) score of 79.5% on StreamingBench while maintaining real-time responsiveness (0.56s QA latency).

## Background & Motivation

**Background**: Streaming video understanding requires VideoLLMs to process continuous video inputs in real time and respond instantly. Existing methods primarily manage the context window through visual token compression or KV cache retrieval to achieve efficient streaming perception.

**Limitations of Prior Work**: Two major categories of methods have their respective drawbacks: (a) Streaming perception methods (StreamForest, TimeChatOnline) focus on visual token management, where the LLM rarely participates in reasoning and analysis, lacking deep understanding capabilities; (b) Offline CoT reasoning methods (Video-R1) execute step-by-step reasoning after the query, leading to high QA latency of up to 8.8s, failing to meet real-time requirements.

**Key Challenge**: There is an inherent conflict between explicit reasoning capabilities and real-time responsiveness—reasoning takes time, but real-time systems require low latency.

**Goal**: How to endow streaming VideoLLMs with strong reasoning capabilities without sacrificing real-time performance?

**Key Insight**: Inspired by the neural coupling mechanism of the human brain, where the logical stream synchronizes with the external information stream. Rather than reasoning only after a query, the model continuously performs intermediate reasoning during video playback, "amortizing" the reasoning cost to the pre-query stage.

**Core Idea**: Transform CoT reasoning from "passive post-query generation" to "active generation during video playback," achieving temporal parallelism between reasoning and perception.

## Method

### Overall Architecture

VST models streaming video understanding as a multi-turn conversation task. Video streams are segmented into clips. As each clip arrives, the model generates a "streaming thought" (intermediate reasoning text) based on the current clip and historical memory. This is written into a dual-memory system: a short-term visual buffer (raw visual tokens of the current clip) and a long-term textual memory (a FIFO queue of historical thoughts). When a user query arrives, the model directly generates the final response based on the accumulated reasoning memory and current visual context, achieving extremely low QA latency.

The joint probability factorization is formulated as:

$$p(\mathbf{y}|\mathbf{q}, \mathcal{V}) = p(\mathbf{y}|\mathbf{q}, \mathbf{c}^K, \mathbf{m}^K) \prod_{k=1}^{K-1} p(\mathbf{z}^k|\mathbf{c}^k, \mathbf{m}^{k-1})$$

The product term represents the "streaming thought" process (completed before the query arrives), and the final term represents the "direct answer" (completed instantly after the query arrives).

### Key Designs

1. **VST-SFT (Supervised Fine-Tuning Stage)**:

    - **Function**: Adapts offline VideoLLMs into streaming reasoning mode.
    - **Mechanism**: Explicitly organizes the video sequence into a multi-turn format: $(memory, (clip_1, thought_1), ..., (clip_{K-1}, thought_{K-1}), clip_K, query, answer)$. A streaming video attention mask restricts visual tokens to seeing only the most recent $L$ steps, while text tokens remain globally visible under causal constraints.
    - **Design Motivation**: Streaming reasoning strictly demands temporal causality—step $k$ can only observe information up to step $k$, with no "previewing" of the future. The attention mask enforces this constraint.
    - **Long Video Processing**: Sequences are partitioned into multiple segments, passing memory states across segment boundaries.

2. **VST-RL (Reinforcement Learning Stage)**:

    - **Function**: Transitions from off-policy imitation to on-policy autonomous exploration, enhancing the quality of intermediate reasoning.
    - **Mechanism**: Employs the GRPO policy where the model generates trajectories $\mathcal{T}$ via an agentic loop in the streaming environment. Verifiable rewards are computed only for the final answer, but advantages are backpropagated to all tokens in the trajectory (including intermediate thoughts).
    - **Design Motivation**: Intermediate reasoning lacks ground truth, but superior intermediate reasoning should lead to a correct final answer. RL enables the model to autonomously learn "what kind of streaming thought is most helpful for downstream answering."
    - **Key Findings**: VST-SFT primarily improves backward memory (+9.2%), while VST-RL mainly enhances forward prediction (+12.7%), making them highly complementary.

3. **Knowledge Graph Data Synthesis Pipeline**:

    - **Function**: Synthesizes 100K high-quality streaming reasoning training data.
    - **Mechanism**: (a) Segments video scenes using PySceneDetect $\rightarrow$ (b) Extracts entities and relations with Gemini 3.0 Flash to build a knowledge graph $\rightarrow$ (c) Samples multi-hop evidence chains using DFS $\rightarrow$ (d) Generates streaming QA pairs and intermediate CoTs based on the evidence chains.
    - **Design Motivation**: Existing CoT datasets are designed for offline settings (viewing the entire video at once), which are unsuitable for streaming setups. The knowledge graph ensures temporal causality and multi-hop reasoning quality.
    - **Quality Control**: World-knowledge check, format alignment, logical consistency, repetition check, and thought validation.

### Loss & Training

- Base model: Qwen2.5-VL-7B, video sampled at 2fps.
- VST-SFT: Freezes the visual encoder, LR 5e-6, 1 epoch, maximum of 384 frames per video.
- VST-RL: DAPO algorithm, rollout batch 256, group size 8, LR 5e-7.
- Training data: 100K VST + 50K LLaVA-Vid QA (SFT); 11K multiple-choice/counting questions (RL).

## Key Experimental Results

### Main Results

| Model | StreamingBench | OVO-Bench | VideoMME (Long) | LongVideoBench | VideoHolmes | QA Latency |
|------|---------------|-----------|-----------------|----------------|-------------|---------|
| GPT-4o | 73.3% | 59.5% | 65.3% | 66.7% | 42.0% | — |
| Qwen2.5-VL-7B | 73.7% | 55.0% | — | 54.7% | 32.9% | 0.54s |
| Video-R1 w/CoT | — | — | — | — | 36.5% | 8.80s |
| StreamForest-7B | 77.3% | 55.6% | — | — | — | — |
| **VST-7B** | **79.5%** | **59.3%** | **55.3%** | **58.0%** | **41.9%** | **0.56s** |

VST-7B outperforms all open-source models on streaming benchmarks (including StreamForest by +2.2%) and exceeds GPT-4o by +6.2%. It also remains highly competitive on offline benchmarks, notably on VideoHolmes (+5.4% vs. Video-R1), while achieving a QA latency that is only 1/15.7 of Video-R1's.

### Ablation Study

| Configuration | OVO-Bench Overall | VideoMME Overall |
|------|------------------|------------------|
| Qwen2.5-VL-7B baseline | 50.5% | 62.9% |
| + LLaVA-Vid 50K SFT | 52.3% | 61.8% |
| + VST-SFT only | 57.4% | 63.0% |
| + VST-RL only | 56.8% | 62.8% |
| + VST-SFT & VST-RL | **59.3%** | **64.9%** |

### Key Findings

- **VST-SFT and VST-RL are complementary in function**: SFT mainly enhances backward tracing (+9.2%), while RL mainly enhances forward prediction (+12.7%), with their combined use yielding the best results.
- **Model scale scalability**: Consistent improvements are observed from 3B $\rightarrow$ 7B $\rightarrow$ 32B, with absolute gains of +7.7/+7.8/+9.2% on StreamingBench.
- **Impact of reasoning steps**: Backward tasks consistently improve as thinking steps increase (from 1 to 16 steps), whereas Real-Time and Forward tasks saturate after $\ge$ 4 steps, where excessive memory introduces redundancy.

## Highlights & Insights

- **"Amortizing reasoning costs into playback time" is an elegant insight**: Video playback inherently involves waiting times (intervals between frames). Utilizing this gap to perform reasoning enables "zero-latency" test-time scaling. This approach can be extended to any scenario with natural waiting periods (e.g., voice dialogue, real-time sensor data processing).
- **RL only rewards the final answer, but advantages propagate to intermediate thoughts**: There is no need to annotate ground truth for intermediate reasoning steps; as long as the final answer is correct, the steps are considered successful. This represents a highly practical "weakly supervised" training method for intermediate processes.
- **Knowledge graph-driven data synthesis** guarantees the quality of multi-hop reasoning and temporal causality, which is considerably more reliable than simply prompting an LLM to generate CoT data.

## Limitations & Future Work

- **Token Consumption**: Streaming thoughts are generated as text, consuming extra LLM tokens. The authors mention that exploring latent reasoning (reasoning in a latent space instead of generating text) could mitigate this overhead.
- **Text-only Memory**: Long-term memory is purely textual, losing visual details. Future work could integrate this with visual KV-cache management methods.
- **Fixed Thinking Frequency**: A thought is generated for every clip, but some clips (e.g., static scenes) may not require reasoning. Dynamically deciding when to think could further improve efficiency.
- **Data Synthesis Dependency on Gemini**: The upper performance limit of the pipeline is constrained by the capabilities of Gemini 3.0 Flash.

## Related Work & Insights

- **vs Video-R1**: Video-R1 performs CoT reasoning after a query. It suffers from an 8.8s latency but only achieves a 36.5% accuracy on VideoHolmes. VST performs pre-query reasoning with a 0.56s latency and 41.9% accuracy, completely dominating Video-R1. This suggests that "when to reason" is more crucial than "how much to reason."
- **vs StreamForest/TimeChatOnline**: These methods only perform streaming perception (visual token management) without explicit reasoning. VST integrates a thought stream on top, improving StreamingBench by +2.2%.
- **vs LongVILA-R1**: While also utilizing RL to enhance video reasoning, LongVILA-R1 is designed for offline settings. VST is the first work to apply RL to streaming video understanding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "thinking while watching" paradigm is innovative and natural, inspired by neuroscience.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 5 benchmarks, multiple scales (3B/7B/32B), with detailed ablations, latency analysis, and case studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with standard formulas and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Exerts a significant impact on the field of streaming video understanding by proposing a novel research paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[CVPR 2026\] Time Blindness: Why Video-Language Models Can't See What Humans Can?](../../CVPR2026/video_understanding/time_blindness_why_video-language_models_cant_see_what_humans_can.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)

</div>

<!-- RELATED:END -->
