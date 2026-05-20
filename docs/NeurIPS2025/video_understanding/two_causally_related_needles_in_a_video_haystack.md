---
title: >-
  [Paper Note] Two Causally Related Needles in a Video Haystack
description: >-
  [NeurIPS 2025][Video Understanding][Long video understanding] This paper proposes Causal2Needles, a benchmark of 4,100 QA pairs that binds the understanding of two causally related events via a "bridging entity…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Long video understanding"
  - "causal reasoning"
  - "needle-in-a-haystack"
  - "video-language models"
  - "benchmark"
date: 2026-05-08
content_hash: 5c90aea33b93dc59
---

# Two Causally Related Needles in a Video Haystack

**Conference**: NeurIPS 2025
**arXiv**: [2505.19853](https://arxiv.org/abs/2505.19853)  
**Code**: [Project Page](https://limiaoyu.github.io/Causal2Needles)  
**Area**: Video Understanding / Causal Reasoning
**Keywords**: Long video understanding, causal reasoning, needle-in-a-haystack, video-language models, benchmark

## TL;DR

This paper proposes Causal2Needles, a benchmark of 4,100 QA pairs that binds the understanding of two causally related events via a "bridging entity," forcing VLMs to jointly retrieve and reason over two needles scattered across a long video. It reveals severe deficiencies in state-of-the-art models on the causal dual-needle task (ChatGPT-4o achieves only 13.4% Both accuracy on the dual-needle setting).

## Background & Motivation

**Background**: Long video understanding benchmarks have proliferated, yet most evaluate only single-needle information extraction (retrieving an answer from one location) or track objects purely through visual appearance matching.

**Limitations of Prior Work**: (1) NLP research has shown that models perform substantially worse on multi-needle than single-needle tasks, yet the multimodal community lacks systematic multi-needle evaluation; (2) existing benchmarks assess VLMs' "world model" capacity only through physical motion prediction, neglecting causal reasoning over human actions; (3) narrative text input may introduce "text bias," allowing models to answer directly from text without genuinely understanding the video.

**Key Challenge**: Do models that perform well on single-needle tasks truly understand the causal structure of long videos, or are they merely performing local information retrieval?

**Goal**: Construct a long video understanding benchmark that simultaneously evaluates joint dual-needle comprehension and causal reasoning ability.

**Key Insight**: Exploit causally related event pairs in movie summary videos, using a "bridging entity" design that forces the model to first retrieve the effect event before localizing the cause event.

**Core Idea**: Obfuscate the bridging entity to transform the dual-needle problem into an indivisible joint reasoning task, preventing it from degenerating into two independent single-needle problems.

## Method

### Overall Architecture

Causal2Needles is built upon 192 movie summary videos (from the YMS and SyMoN datasets) and comprises four question types: (1) non-causal single-needle (1,704 questions)—asking about event details; (2) causal single-needle (902 questions)—inferring cause from effect, with the answer at a single location; (3) causal dual-needle–visual grounding format (747 questions)—requiring selection of the video segment containing the answer; (4) causal dual-needle–image description format (747 questions)—requiring description of visual details of the cause event. The construction pipeline proceeds as: LLM extracts causal relations → generates bridging entities → generates two-part questions → automatic and human quality evaluation.

### Key Designs

1. **Bridging-Entity-Driven Dual-Needle Question Design**:

    - Function: Design a question structure that forces the model to jointly understand both cause and effect events.
    - Mechanism: Identify a bridging entity shared by a causal event pair (e.g., "Superman's death"), then replace it with an obfuscated expression in the question (e.g., "tragedy"). Part 1 requires resolving the bridging entity from the effect event; Part 2 requires retrieving the video segment of the cause event based on that resolution.
    - Design Motivation: If the bridging entity is stated explicitly, the dual-needle problem degenerates into two independently answerable single-needle problems—obfuscation is the key to ensuring joint reasoning.

2. **Dual Complementary QA Formats (Visual Grounding + Image Description)**:

    - Function: Design two complementary question formats to offset each other's evaluation biases.
    - Mechanism: The visual grounding format requires selecting the correct video segment, preventing purely text-based answers but potentially causing out-of-distribution (OOD) issues; the image description format requires answering multiple-choice questions about visual details, circumventing OOD but potentially susceptible to pretrained knowledge leakage. Both formats are used jointly for comprehensive evaluation.
    - Design Motivation: A single format will either overestimate or underestimate model capability—visual grounding may underestimate (OOD), while image description may overestimate (movie knowledge leakage).

3. **Global + Local Causal Relation Extraction**:

    - Function: Use an LLM to extract a comprehensive causal relation graph from video narratives.
    - Mechanism: A global graph extracts long-range causal relations from the full narrative; a sliding window (15 sentences, stride 5) extracts local graphs to capture fine-grained relations. After merging, only causal pairs with distance $\geq 3$ events apart are retained.
    - Design Motivation: LLMs exhibit reduced attention to content in the middle of long texts ("lost in the middle"); local windows compensate for missed relations.

### Loss & Training

This is a purely evaluative benchmark; no training is involved. Questions are generated using LLMs (GPT-4o-mini, Gemini-2.0-flash); quality is assessed by ChatGPT-4.1 and Gemini-2.0-flash as well as five human annotators. Bridging entity co-occurrence rate exceeds 95%, and factual correctness of questions scores 4.6+/5.

## Key Experimental Results

### Main Results

VLM accuracy on Causal2Needles (%, averaged over forward/reverse):

| Model | Non-Causal 1-Needle | Causal 1-Needle | VG 2-Needle Both | ID 2-Needle |
|-------|:-------------------:|:---------------:|:----------------:|:-----------:|
| Human | - | 78.2 | 79.3 | 88.2 |
| ChatGPT-4o | 56.8 | 39.2 | 13.4 | 59.2 |
| Gemini-1.5-pro | 55.4 | 35.6 | 8.4 | 60.9 |
| ChatGPT-4o-mini | 39.9 | 33.4 | 5.2 | 52.3 |
| Qwen2.5VL-32B | 30.7 | 11.7 | 1.9 | 53.5 |
| LLaVA-OneVision-7B | 12.3 | 18.0 | 0.1 | 28.3 |

### Ablation Study

Effects of causality and dual-needle distance:

| Analysis Dimension | Finding |
|-------------------|---------|
| Non-causal → Causal (1-needle) | ChatGPT-4o: 56.8% → 39.2% (causal reasoning is substantially harder) |
| 1-needle → 2-needle Both | ChatGPT-4o: 39.2% → 13.4% (joint dual-needle reasoning causes a dramatic drop) |
| Distance correlation | Greater inter-needle distance correlates with lower performance (significant negative correlation) |
| Forward vs. reverse order | Most models perform better with forward-order input, though differences vary by model |

### Key Findings

1. **Causal reasoning is far harder than non-causal retrieval**: All models exhibit large performance drops from non-causal to causal questions.
2. **Joint dual-needle comprehension is the true bottleneck**: The strongest model, ChatGPT-4o, achieves only 13.4% VG dual-needle Both accuracy, far below the single-needle setting.
3. **Open-source models nearly completely fail**: Qwen2.5VL-32B achieves only 1.9% on dual-needle Both.
4. **Image description format consistently outperforms visual grounding**: Likely because models can leverage pretrained knowledge when answering appearance-based multiple-choice questions.

## Highlights & Insights

- **Elegant bridging entity design**: Obfuscating a shared reference ensures the dual-needle task is indivisible—this is the core innovation for evaluating joint reasoning.
- **Exposing the "high-score trap"**: Models may achieve high scores on existing benchmarks while entirely lacking causal reasoning ability.
- **Dual complementary format design**: Simultaneously controls for OOD bias and knowledge leakage bias.
- **High dataset quality**: Automatic and human evaluation confirms >95% valid bridging entities and 4.6+/5 factual correctness.

## Limitations & Future Work

- Videos are sourced from movie summaries (edited clips), which may not fully represent natural long videos.
- Narrative text as auxiliary input may introduce text bias that is difficult to fully eliminate.
- Human baseline evaluation is limited in scale (5 annotators).
- Causal relation extraction itself relies on LLMs and may miss implicit causal chains.
- Only a limited set of VLMs is evaluated; additional architectures (e.g., video-native models) remain to be assessed.

## Related Work & Insights

- **vs. VideoMME/MLVU**: These benchmarks include multiple needles but the needles are only visually associated and can be understood independently—Causal2Needles requires causal joint understanding.
- **vs. EgoSchema**: Lacks diagnostic category labels—Causal2Needles precisely isolates single-needle vs. dual-needle and causal vs. non-causal conditions.
- **Insight**: "Capable of retrieval ≠ capable of reasoning"—the next frontier in long video understanding is advancing from information extraction to causal reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The bridging-entity-driven dual-needle causal reasoning design is unique among long video benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10+ models, 4 question types, forward/reverse comparison, and comprehensive quality evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear and motivation is well-argued.
- Value: ⭐⭐⭐⭐⭐ Reveals fundamental weaknesses of existing VLMs in joint reasoning; highly diagnostic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](../../CVPR2026/video_understanding/rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[NeurIPS 2025\] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)

</div>

<!-- RELATED:END -->
