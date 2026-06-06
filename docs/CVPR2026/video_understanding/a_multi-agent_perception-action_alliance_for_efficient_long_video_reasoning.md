---
title: >-
  [Paper Note] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning
description: >-
  [CVPR2026][Video Understanding][long video reasoning] This paper proposes A4VL, a training-free multi-agent perception-action alliance framework that achieves state-of-the-art performance across five VideoQA benchmarks—s…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "long video reasoning"
  - "multi-agent collaboration"
  - "video question answering"
  - "perception-action exploration"
  - "training-free framework"
date: 2026-05-08
content_hash: 6d98cbd31c7203e8
---

# A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning

**Conference**: CVPR2026
**arXiv**: [2603.14052](https://arxiv.org/abs/2603.14052)  
**Code**: [git-disl/A4VL](https://github.com/git-disl/A4VL)  
**Area**: Video Understanding
**Keywords**: long video reasoning, multi-agent collaboration, video question answering, perception-action exploration, training-free framework

## TL;DR

This paper proposes A4VL, a training-free multi-agent perception-action alliance framework that achieves state-of-the-art performance across five VideoQA benchmarks—surpassing 28 baseline methods—while significantly reducing inference latency, through event-driven video segmentation, clue-guided keyframe selection, and a multi-round agent negotiation-and-pruning mechanism.

## Background & Motivation

**Computational bottleneck in long video reasoning**: When MLLMs process long videos, memory and time costs grow quadratically with the number of frames; GPT-4o requires over 150 seconds per question on Video-MME on average.

**Redundant frames introduce noise**: Prior work has shown that naively increasing frame sampling density can actually hurt performance, as redundant frames distract the model and hinder alignment with truly informative keyframes.

**Slow existing agent methods**: VideoAgent requires over 10 minutes to process a one-hour video; most methods rely on a single MLLM for decision-making and lack effective multi-agent collaboration.

**Difficulty in keyframe localization**: When a question concerns events covered by only a small number of frames in a long video, precisely locating relevant frames is highly challenging; MoReVQA shows significant accuracy degradation on long-video datasets.

**Limitations of single-model approaches**: Different MLLMs have complementary strengths and blind spots; a single model is prone to errors in complex reasoning scenarios and cannot self-correct, whereas collaborative mechanisms enable mutual complementation.

**Lack of efficient multi-agent coordination**: Existing multi-agent methods either lack iterative correction capabilities or lack pruning strategies that lead to inefficiency, motivating the design of a framework that is both accurate and efficient.

## Method

### Overall Architecture

A4VL consists of three core components operating within a multi-round perception-action exploration loop:

- **Agent Teaming**: An optional preprocessing stage. From $M$ candidate MLLMs, $m=3$ maximally complementary agents are selected to form a team—each agent performs inference on $K$ unannotated samples and is scored by the frequency of agreement with the majority answer; the top-3 scoring agents are selected.
- **Perception Exploration**: A two-stage process—(1) each agent randomly samples $N_1=4$ frames to generate a query-relevant perception clue; (2) the video is segmented into at most $B$ blocks using an event-driven method, CLIP similarity is computed between the clue and each block, and $N_2=16$ frames are sampled from high-relevance blocks via softmax-weighted allocation.
- **Action Exploration**: A two-stage process—(1) each agent generates an answer and reasoning rationale based on its sampled frames; (2) if consensus is reached the final answer is output; otherwise, agents cross-score one another (1–10), the lowest-scoring agent is pruned, and the remaining agents refine their clues based on the current state before entering the next round.

### Key Designs

**Event-driven video segmentation**: DINOv2 embeddings and pixel-level cues (HSV, motion, sharpness) are used to detect scene changes; candidate boundary points are generated via KTS, PELT, and SSM novelty detection, merged through NMS, and the top-$(B-1)$ boundaries are retained. Segmentation completes within 2 seconds for most videos in experiments.

**Adaptive frame allocation**: When all block similarities fall below threshold $\rho=0.8$, each agent samples $N_2$ frames only from the best-matching block; when high-similarity blocks exist, frame counts are allocated proportionally via softmax normalization: $\mathbf{c}^{(i)} = \lfloor N_2 \cdot \text{SoftMax}(\mathbf{s}^{(i)}) \rfloor$, with the remainder filled randomly.

**Multi-round negotiation and pruning**: Consensus detection employs a Full Consensus strategy—termination occurs only when all agents agree. When consensus is not reached, each agent scores all answers (including its own), and the agent with the lowest total score is pruned. Remaining agents refine their perception clues based on the previous round's answer set, rationales, and information from the pruned agent:

$$P_{i,j+1} = A_{i,\text{refine}}(P_{i,j}, S_{a,j}, S_{r,j}, A_{\min}, Q, O)$$

A maximum of 3 rounds is run (upper-bounded by the number of agents).

### Loss & Training

A4VL is a fully training-free framework involving no gradient updates or loss functions. All components—clue generation, answer reasoning, cross-scoring, and clue refinement—are driven entirely by prompting the constituent MLLMs.

## Experiments

### Main Results

Comparison against 2 closed-source MLLMs, 16 open-source MLLMs, and 10 agent/long-video methods across five VideoQA benchmarks:

| Method | NeXT-QA | EgoSchema | LongVideoBench | MLVU | Video-MME (avg, w/o sub) |
|--------|---------|-----------|----------------|------|--------------------------|
| GPT-4o | - | 72.2 | 66.7 | 54.9 | 71.9 |
| Gemini 1.5 Pro | - | 71.1 | 64.0 | - | 75.0 |
| InternVL3-78B | 84.0 | 76.8 | 56.4 | 55.3 | 66.9 |
| LVAgent | 83.0 | 78.4 | 66.9 | 50.0 | 73.9 |
| **A4VL** | **85.1** | **82.2** | **72.2** | **58.0** | **77.2** |

A4VL achieves the best performance on all five benchmarks. It is the only method to exceed 80% on EgoSchema; on LongVideoBench it outperforms GPT-4o by 5.5 points while using only open-source models.

### Inference Efficiency

| Method | NeXT-QA | EgoSchema | MLVU |
|--------|---------|-----------|------|
| GPT-4o | 23s | 54s | 127s |
| InternVL3-78B | 15s | 50s | 204s |
| VideoAgent | 20s | 83s | 175s |
| TraveLER | 101s | 94s | 450s |
| **A4VL** | **18s** | **37s** | **74s** |

On MLVU long videos, A4VL requires only 74 seconds per sample—42% faster than GPT-4o and 83% faster than TraveLER.

### Ablation Study

- **Effect of rounds**: Accuracy improves steadily as the maximum number of rounds increases. On harder datasets, agents tend to utilize more negotiation rounds.
- **Sampling strategy**: RESampling (random sampling in the perception phase + event-block sampling in the action phase) achieves the best result at 82.2%, indicating that perception benefits from global coverage while action requires event-focused sampling.
- **Consensus criterion**: Full Consensus (82.2%) outperforms Majority Consensus (81.4%), at the cost of a slight increase in latency (37s vs. 26s).
- **Necessity of pruning**: Removing pruning (NoPruneSum 80.8%, NoPruneMaj 79.4%) not only reduces accuracy but also increases latency from 37s to 60s, demonstrating that pruning is critical for both effectiveness and efficiency.

### Key Findings

1. The agent team selected by Agent Teaming differs across benchmarks (e.g., MLVU selects LLaVA-72B rather than QwenVL-72B), validating the value of task-adaptive teaming.
2. Even when all agents answer incorrectly in the initial round, subsequent rounds of negotiation and clue refinement can correct the output to the right answer (demonstrated via an EgoSchema example).

## Highlights & Insights

- **Training-free and plug-and-play**: Entirely prompt-driven, enabling flexible combination of arbitrary VLMs without retraining or fine-tuning.
- **Elegant decoupled perception-action design**: A coarse-to-fine two-stage frame selection achieves precise localization with very few frames (4+16).
- **Efficient pruning-consensus mechanism**: Cross-score-based pruning of weak agents, rather than simple voting, simultaneously improves accuracy and speed.
- **Comprehensive experimental design**: Covers five benchmarks spanning short/medium/long videos, 28 comparison methods, and thorough ablations.

## Limitations & Future Work

- The agent pool is restricted to 8 specific VLMs; generalization to other models (e.g., Gemini, GPT series) has not been validated.
- Simultaneous deployment of multiple large models (including 78B) requires 6 H200 GPUs, posing a high hardware barrier.
- Only visual and text (subtitle) inputs are processed; audio modality information is not utilized.
- CLIP similarity serves as the sole block-matching signal, which may be insufficient for abstract or causal questions.
- The maximum of 3 negotiation rounds is hard-coded by the number of agents, lacking a dynamic adjustment mechanism.

## Related Work & Insights

- **Token optimization**: DYTO (dynamic bipartite token merging) and AuroraLong (linear RNN + token merging) reduce processing overhead.
- **Agent methods**: VideoAgent (memory-augmented single agent), TraveLER (multi-step planning), MoReVQA (modular reasoning), LVAgent (dynamic multi-agent collaboration)—A4VL outperforms all in comparison.
- **Memory retrieval**: VideoRAG employs retrieval-augmented generation for long video processing and is competitive on Video-MME, but reports no EgoSchema/NeXT-QA results.
- **Architectural improvements**: DynFocus (dynamic focusing) and BOLT (efficient sampling) optimize from a model architecture perspective, orthogonal to agent-based approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ (the combined design of multi-agent perception-action alliance, event-driven segmentation, and cross-score pruning is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 benchmarks, 28 comparisons, multi-dimensional ablations—highly comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rich illustrations, well-formatted formulations)
- Value: ⭐⭐⭐⭐ (the training-free framework is highly practical, though high hardware requirements limit some application scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)
- [\[CVPR 2026\] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[CVPR 2026\] MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning](minerva-cultural_a_benchmark_for_cultural_and_multilingual_long_video_reasoning.md)

</div>

<!-- RELATED:END -->
