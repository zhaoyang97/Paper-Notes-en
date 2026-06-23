---
title: >-
  [Paper Note] Fostering Video Reasoning via Next-Event Prediction
description: >-
  [ICLR 2026][vlm_reasoning][Next-Event Prediction] This paper proposes **Next-Event Prediction (NEP)**, a learning task that splits video into "past" and "future" segments. It requires MLLMs to predict textual descriptions of future events based solely on past frames, leveraging the video's inherent future content as a self-supervised signal to elicit temporal reasonin
tags:
  - ICLR 2026
  - vlm_reasoning
  - Next-Event Prediction
  - MLLM
  - Self-Supervised Learning
  - FutureBench
date: 2026-05-08
content_hash: ff07a3664736e9cb
---
# Fostering Video Reasoning via Next-Event Prediction

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=8nUgzuvskm](https://openreview.net/forum?id=8nUgzuvskm)  
**Code**: To be confirmed  
**Area**: Multi-modal Video Understanding / Temporal Reasoning  
**Keywords**: Next-Event Prediction, Video Temporal Reasoning, MLLM, Self-supervised Learning, FutureBench  

## TL;DR
This paper proposes **Next-Event Prediction (NEP)**, a learning task that splits video into "past" and "future" segments. It requires MLLMs to predict textual descriptions of future events based solely on past frames, leveraging the video's inherent future content as a self-supervised signal to elicit temporal reasoning capabilities. The work also introduces the V1-33K training set and the FutureBench evaluation benchmark.

## Background & Motivation
**Background**: While MLLMs have shown rapid progress in video understanding, mainstream video instruction-tuning tasks (Video QA, Captioning, Grounding) are essentially **observational**. They focus on "describe what you see" perceptual tasks like object recognition, event identification, and factual recall.

**Limitations of Prior Work**: (1) These tasks prioritize cross-modal alignment but neglect the core dimension distinguishing video from static images—**time**. VQA often relies on a few keyframes, and Captioning maps frames to text sequentially, failing to capture the evolution of dynamic events. (2) VQA and Grounding typically require **manual annotation or distillation from stronger MLLMs**, leading to poor scalability and high costs.

**Key Challenge**: The reasoning ability of LLMs stems from next-token prediction, a simple yet infinitely scalable self-supervised task. Video understanding lacks a corresponding learning task that effectively instills temporal reasoning capabilities into MLLMs.

**Goal**: To identify a video version of "next-token prediction"—a self-supervised, scalable task specifically designed to elicit temporal reasoning.

**Core Idea**: **Use the "future" of the video as a supervisory signal**. By splitting a video at a causal turning point, the model observes only the first half and is required to predict what happens next. Since the target event is absent from the input, the model must transition from perception to prediction, combining "perceived facts" from the visual encoder with "common-sense knowledge" (physical laws, social norms, typical behaviors) from the LLM for causal inference.

## Method

### Overall Architecture
NEP formalizes the video frame sequence $V=[v_1,\dots,v_T]$ by splitting it at a point $t<T$ into a past segment $V_{\le t}=[v_1,\dots,v_t]$ and a future segment $V_{>t}=[v_{t+1},\dots,v_T]$. The MLLM is trained to take $V_{\le t}$ as input and generate text $Y$ describing the events in the future segment. This is essentially a seq-to-seq language modeling problem conditioned on video frames. The work consists of three components: **NEP task formalization → V1-33K data construction pipeline → four instruction-tuning strategies**, supported by FutureBench for multi-hop temporal reasoning evaluation.

```mermaid
flowchart LR
    A[Raw Video + Captions] --> B[Caption Analysis<br/>LLM Identifies Causal Pivot]
    B --> C[Grounding to Locate Timestamp t]
    C --> D[Split into Past/Future Segments<br/>Split Captions into Past/Future]
    D --> E[Optional: Reasoning + Critique<br/>Generate Reasoning Trace]
    E --> F[V1-33K<br/>33K Past-Future Pairs]
    F --> G[MLLM Inputs Past Frames]
    G --> H[Predict Future Event Text]
    H --> I[Four Strategies: SFT/CFT/Distill/Mix]
```

### Key Designs

**1. Leveraging the Future for Self-Supervision**: The essence of NEP is that the supervisory target describes events not visible in the input, forcing the model from "perception" (object detection, action recognition) to "prediction." To predict a plausible next event, the MLLM must integrate visual evidence with world knowledge. The authors categorize tasks into three types of logical inference: VQA as **induction**, NEP as **deduction**, and previous-event prediction as **abduction**. Experiments show that deductive NEP yields the highest gains on temporal benchmarks due to its higher cognitive load and reliance on abstract logical principles.

**2. Low-Cost Automated Data Pipeline (V1-33K)**: The pipeline converts video into training samples in three steps: (i) **Caption Analysis** (optional): LLMs parse video captions to identify scene transitions and causal turning points; (ii) **Grounding and Splitting**: MLLMs align textual split points to the video timeline to obtain timestamp $t$, splitting the video into past/future segments and the caption into past/future components; (iii) **Reasoning and Critique** (optional): Textual reasoning models generate predictions and traces for the past-caption, which are refined by another LLM. A key insight is that the only prerequisite capability is **timestamp grounding**, which is learned early in fine-tuning, allowing NEP to produce "auto-labeled self-supervised signals" more cheaply than VQA.

**3. Controlled Task Comparison**: To isolate the effect of the NEP task, the authors construct NEP data using the **exact same video sources as Captioning/QA**. This allows for a comparison where only the task format changes while the model, video content, and data volume remain constant, eliminating confounding factors like data quality or source bias.

**4. Four Instruction-Tuning Strategies**: The authors compare training strategies: **SFT** (standard cross-entropy with ground-truth future captions), **CFT** (critique fine-tuning using GPT-generated signals), **Distill** (using structured reasoning traces), and **Mix** (proportional mixing of supervision types). Results indicate that while SFT is simple, it is highly efficient, often outperforming more complex strategies in terms of cost-effectiveness.

### FutureBench Evaluation Design
FutureBench utilizes a multiple-choice QA format. Each video includes an "end anchor" derived from the final state of the full video. Models must reason forward and backward to identify intermediate events. It features two paradigms: **Extrapolation** (predicting a sequence of 1-hop/2-hop/3-hop future events) and **Interpolation** (filling in non-continuous intermediate events given start and end anchors). Distractors are designed to be "common-sense plausible but logically inconsistent with the trajectory." The textual model o4-mini achieves only **32.0%** accuracy without visual input, confirming the benchmark's strong dependence on visual perception.

## Key Experimental Results

### Main Results: NEP vs. Other Video Instruction-Tuning Tasks (Qwen2.5-VL-7B, 3K samples)

| Task | Observation Range | G-Avg. (General) | T-Avg. (Temporal) | FutureBench |
|------|-------------------|------------------|-------------------|-------------|
| Instruct (Original) | — | 60.3 | 49.7 | 52.6 |
| Captioning | Full Video | 60.0 | 49.7 | 55.8 |
| MCQA | Full Video | 58.5 | 47.7 | 60.3 |
| OEQA | Full Video | 60.4 | 51.2 | 58.8 |
| **NEP** | Partial Video | **60.9** | **53.5** | **61.3** |

NEP leads significantly on temporal benchmarks (T-Avg. 53.5 vs. others $\le 51.2$) while maintaining or slightly improving general benchmark performance (VideoMME/MVBench), demonstrating that it strengthens temporal reasoning without sacrificing general understanding.

### Ablation Study: Four Tuning Strategies (Partial Results)

| Model | Strategy | G-Avg. | T-Avg. |
|-------|----------|--------|--------|
| Qwen2.5-VL-3B | Instruct | 57.2 | 45.8 |
| | SFT | 56.3 | **48.2** |
| | Distill | 58.1 | 48.4 |
| | Mix | 57.9 | 48.5 |
| Qwen2.5-VL-7B | Instruct | 60.3 | 49.7 |
| | **SFT** | 59.7 | **52.6** |
| | Distill | 61.2 | 51.9 |
| | Mix | 59.9 | 53.3 |

SFT, the simplest strategy, achieves substantial gains on temporal benchmarks (from 49.7 to 52.6 on the 7B model). While CFT and Distill are effective, they rely on additional annotations, making them less cost-effective than SFT.

### Key Findings
- **NEP enhances temporal reasoning without degrading general performance**: Consistent improvements are seen across TempCompass, TemporalBench, SEED-Bench-R1, and FutureBench.
- **Deduction > Induction/Abduction**: When using the same 3K data, the deductive NEP format yields higher gains on temporal benchmarks than VQA (induction) or previous-event prediction (abduction).
- **Simplicity is Effective**: The most basic SFT strategy captures the majority of the benefits, echoing the philosophy that task scalability is more important than complex algorithmic designs.

## Highlights & Insights
- **Identified a video version of "next-token prediction"**: Treating the "future" as a free self-supervised signal is an elegant, scalable task design specifically aligned with temporal reasoning.
- **Robust Controlled Variables**: By reusing the same videos across tasks, the authors isolate the "task format" as the primary variable, ensuring high credibility.
- **Logical Framework**: Mapping VQA/NEP to induction/deduction provides a cognitive science perspective on why NEP better stimulates reasoning.
- **Effective Benchmark**: FutureBench's multi-hop design and the low performance of text-only models (o4-mini at 32%) prove it to be a discriminative temporal reasoning benchmark.

## Limitations & Future Work
- **Data Quality Variance**: The automated pipeline produces future segments of varying difficulty; the authors anticipate better results with refined data quality control.
- **Dependence on Grounding**: The self-supervised signal relies on the model's existing timestamp grounding capability; poor grounding in early stages could degrade signal quality.
- **Modest Absolute Gains**: While consistently better, the gains are in the range of a few percentage points, representing a conceptual contribution rather than a massive SOTA jump.
- **Future Directions**: Scaling NEP to larger datasets, integrating R1-style RL to amplify reasoning gains, and extending to multi-hop predictions across longer temporal spans.

## Related Work & Insights
- **vs. Video Instruction Tuning** (Video-LLaVA, LLaVA-NeXT, Qwen-VL): These models focus on observational tasks. NEP shifts toward predictive goals to model "world dynamics" rather than just static frame understanding.
- **vs. CV Future Prediction** (Action/Motion/Frame Prediction): Earlier CV works focused on low-level representation learning for pre-training. NEP focuses on **high-level, semantic, natural language** future event prediction by improving the MLLM's projector and LLM while freezing the visual encoder.
- **Insights**: NEP demonstrates a path for using the inherent temporal structure of data to build self-supervised signals. This is transferable to embodied domains like robotics and autonomous driving.

## Rating
- Novelty: ⭐⭐⭐⭐ — Highly intuitive "future as self-supervision" design with a strong logical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Robust controlled comparisons and ablation across multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and well-structured arguments.
- Value: ⭐⭐⭐⭐ — Provides a scalable, low-cost paradigm for fostering temporal reasoning in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning](../../CVPR2026/vlm_reasoning/oasis_on-demand_hierarchical_event_memory_for_streaming_video_reasoning.md)
- [\[ICLR 2026\] ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](expvid_a_benchmark_for_experiment_video_understanding_reasoning.md)
- [\[CVPR 2026\] VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction](../../CVPR2026/vlm_reasoning/vgent_visual_grounding_via_modular_design_for_disentangling_reasoning_and_predic.md)
- [\[ICLR 2026\] VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning](videozoomer_reinforcement-learned_temporal_focusing_for_long_video_reasoning.md)
- [\[ICLR 2026\] Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy](vid-llm_a_compact_video-based_3d_multimodal_llm_with_reconstructionreasoning_syn.md)

</div>

<!-- RELATED:END -->
