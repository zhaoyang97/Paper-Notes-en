---
title: >-
  [Paper Note] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering
description: >-
  [ICCV 2025][Video Generation][Text-video alignment evaluation] This paper proposes ETVA, a text-to-video alignment evaluation method based on fine-grained question generation and answering. It employs a multi-agent scene…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Text-video alignment evaluation"
  - "question-answering framework"
  - "scene graph"
  - "multi-agent"
  - "knowledge-augmented reasoning"
date: 2026-05-08
content_hash: 789f815f6fc0db08
---

# ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering

**Conference**: ICCV 2025
**arXiv**: [2503.16867](https://arxiv.org/abs/2503.16867)
**Code**: [eftv-eval.github.io/etva-eval](https://eftv-eval.github.io/etva-eval)
**Area**: Video Generation
**Keywords**: Text-video alignment evaluation, question-answering framework, scene graph, multi-agent, knowledge-augmented reasoning

## TL;DR

This paper proposes ETVA, a text-to-video alignment evaluation method based on fine-grained question generation and answering. It employs a multi-agent scene graph traversal to generate atomic questions and a knowledge-augmented multi-stage reasoning pipeline to answer them. ETVA substantially outperforms existing metrics in correlation with human judgments (Spearman's ρ 58.47 vs. 31.0) and introduces an evaluation benchmark containing 2k prompts and 12k questions.

## Background & Motivation

### State of the Field

Text-to-video (T2V) generation models (e.g., Sora, Kling, HunyuanVideo) are advancing rapidly, yet reliable automatic metrics for measuring the semantic alignment between generated videos and text descriptions remain lacking.

### Limitations of Prior Work

**Coarse-grained scoring**: Existing metrics (CLIPScore, VideoScore, etc.) produce a single scalar score, providing no information about which specific semantic elements are aligned or misaligned.

**Poor correlation with human judgments**: For instance, in the motivating example, human annotators judge Video 2 as better capturing microgravity effects aboard a space station, whereas existing metrics systematically prefer Video 1.

**Low-quality question generation**: Naïve ICL-based approaches generate overly complex questions (e.g., simultaneously querying action + object + environment), which Video LLMs cannot answer accurately.

**Severe hallucination in Video LLMs**: When Video LLMs directly answer questions, they lack commonsense knowledge (e.g., the behavior of water under microgravity) and perform no explicit deep reasoning.

### Root Cause

How to generate questions that are simultaneously atomic and comprehensive, and enable Video LLMs to answer them reliably?

### Starting Point

The method simulates the human annotation process: first parsing the text into a scene graph and traversing it to generate atomic questions (addressing C1), then retrieving relevant knowledge and applying multi-stage reasoning so that the Video LLM recalls domain knowledge before observing and analyzing the video (addressing C2).

## Method

### Overall Architecture

ETVA operates in two stages:
1. **Question Generation (QG)**: A multi-agent system parses the text prompt → constructs a semantic scene graph → traverses it to generate atomic yes/no questions.
2. **Question Answering (QA)**: An auxiliary LLM retrieves commonsense knowledge → the Video LLM answers each question via multi-stage reasoning.

The final alignment score is: $S = \frac{1}{n}\sum_{i=1}^{n} S_i$, where $S_i \in \{0, 1\}$.

### Key Designs

#### 1. **Multi-Agent Question Generation**

- **Function**: Decompose the text prompt into atomic questions, ensuring complete coverage without redundancy.
- **Mechanism**: Three collaborative agents —
    - **Element Extractor**: Identifies entities (e.g., "cup," "space station"), attributes (e.g., "glass material," "transparent"), and relations (e.g., "poured from," "inside").
    - **Graph Builder**: Organizes the elements into a hierarchical scene graph, with entity nodes as central anchors; all relation and attribute nodes must be connected to at least one entity. Attribute nodes have only outgoing edges; relation nodes maintain bidirectional connections.
    - **Graph Traverser**: Processes nodes in order — first confirming entities → then verifying attributes → finally checking relations (only after both endpoint entities are verified).
- **Design Motivation**: Scene-graph-driven traversal guarantees atomicity (each question addresses exactly one aspect) and logical dependency (objects are confirmed before their relations are queried), avoiding the redundancy and unanswerable questions produced by ICL-based methods.

#### 2. **Knowledge-Augmented Multi-Stage Reasoning**

- **Function**: Reduce Video LLM hallucinations by simulating the human cognitive process.
- **Mechanism**:
    - **Knowledge Augmentation (KA)**: An auxiliary LLM (Qwen2.5-72B-Instruct) retrieves relevant commonsense knowledge based on the prompt. For example, "pouring water in a space station" → under microgravity, liquid forms floating spheres rather than falling.
    - **Multi-Stage Reasoning**:
        1. **Video Understanding Stage**: The Video LLM independently extracts frame-by-frame descriptions without access to the text prompt.
        2. **General Reflection Stage**: The model cross-analyzes its observations together with the question and commonsense knowledge.
        3. **Conclusion Stage**: The model produces a Yes/No answer with an explicit visual–linguistic alignment check.
- **Design Motivation**: This pipeline mirrors the human annotation process — recalling relevant knowledge → carefully observing the video → reasoning deeply → drawing a conclusion — thereby mitigating the hallucination that arises when Video LLMs answer without deliberation.

#### 3. **ETVABench Construction**

- **Function**: Construct a comprehensive benchmark specifically for text-to-video alignment evaluation.
- **Mechanism**:
    - 2k diverse prompts are collected from VBench, EvalCrafter, T2V-ComBench, and other benchmarks.
    - Questions are categorized into 10 types based on question type: existence, action, material, spatial, number, shape, color, camera, physics, other.
    - ETVABench-2k (2k prompts, 12k questions) + ETVABench-105 (a compact 105-prompt subset).

## Key Experimental Results

### Main Results (Correlation with Human Judgments)

| Metric | Kendall's τ | Spearman's ρ |
|--------|-------------|--------------|
| BLIP-BLEU | 8.5 | 12.1 |
| CLIPScore | 10.3 | 13.8 |
| ViCLIPScore | 19.4 | 25.9 |
| VideoScore | 23.7 | 31.0 |
| **ETVA** | **47.2** | **58.5** |

**Per-dimension comparison** (Spearman's ρ):

| Dimension | VideoScore | ETVA | Gain |
|-----------|-----------|------|------|
| Existence | 30.6 | 57.4 | +87.6% |
| Material | 37.3 | 66.1 | +77.2% |
| Spatial | 31.7 | 66.8 | +110.7% |
| Shape | 35.7 | 75.1 | +110.4% |
| Physics | 23.9 | 60.4 | +152.7% |
| Camera | 26.3 | 44.2 | +68.1% |

### Ablation Study

**QG component ablation**:

| Configuration | Kendall's τ | Spearman's ρ | Note |
|---------------|-------------|--------------|------|
| Multi-agent QG | **47.16** | **58.47** | Full method |
| Vanilla ICL QG | 35.04 | 42.87 | Direct ICL question generation |
| Gain | +12.12 | +15.60 | +34.6% |

**QA component ablation**:

| Configuration | Accuracy | Kendall's τ | Spearman's ρ |
|---------------|----------|-------------|--------------|
| ETVA (full) | **89.27** | **47.16** | **58.47** |
| w/o Knowledge Augmentation (KA) | 67.34 | 27.34 | 35.54 |
| w/o Video Understanding (VU) | 82.73 | 37.56 | 44.81 |
| w/o Critical Reflection (CR) | 68.74 | 28.73 | 38.21 |
| KA only | 65.48 | 24.72 | 33.12 |
| Direct answering | 63.07 | 18.18 | 23.84 |

### T2V Model Evaluation (ETVABench-105)

| Model | Existence | Action | Spatial | Physics | Camera | Avg |
|-------|-----------|--------|---------|---------|--------|-----|
| Latte | 0.519 | 0.504 | 0.444 | 0.350 | 0.105 | 0.474 |
| CogVideoX-5B | 0.644 | 0.664 | 0.630 | 0.500 | 0.474 | 0.620 |
| HunyuanVideo | 0.727 | 0.693 | 0.704 | 0.300 | 0.421 | 0.686 |
| Kling-1.5 | 0.754 | 0.675 | 0.754 | 0.500 | 0.383 | 0.707 |
| Pika-1.5 | 0.801 | 0.752 | 0.778 | 0.450 | 0.421 | 0.738 |
| Sora | 0.815 | 0.759 | **0.870** | 0.550 | 0.316 | 0.757 |
| **Vidu-1.5** | 0.792 | **0.766** | 0.862 | **0.600** | 0.421 | **0.761** |

### Key Findings

- **ETVA improves correlation with human judgments by 88%**: Spearman's ρ rises from 31.0 to 58.5.
- **Knowledge augmentation is the most critical module**: Removing KA causes the largest accuracy drop of 21.93%, demonstrating that commonsense knowledge is essential for combating hallucination.
- **Multi-agent QG contributes a 14.67% gain**: Structured scene-graph-based generation yields substantially higher quality than vanilla ICL.
- **All T2V models perform worst on Physics and Camera**: Top scores reach only 0.600 and 0.474 respectively, exposing fundamental deficiencies in physical simulation and camera control.
- **Closed-source models lead overall but do not dominate universally**: Open-source HunyuanVideo matches or even surpasses Sora on the Shape dimension (0.824 vs. 0.765).
- **KA alone is insufficient**: Multi-stage reasoning is required to effectively leverage the retrieved knowledge (KA-only accuracy: 65.48 vs. full method: 89.27).

## Highlights & Insights

1. **Scene-graph-driven atomic question generation**: Directly addresses the core challenge of identifying questions that Video LLMs can answer accurately.
2. **Cognitive architecture simulation**: The pipeline of knowledge retrieval → video observation → reflective reasoning → conclusion closely mirrors the human annotation workflow.
3. **Fine-grained diagnostic capability**: Beyond a single overall score, ETVA precisely identifies whether each semantic element is aligned, providing actionable guidance for model improvement.
4. **10-dimension classification system**: Question-based categorization is more precise than prompt-based categorization, as a single prompt can span multiple dimensions.
5. **Systematic evaluation of 15 T2V models**: Reveals that physics simulation and camera control are the most significant weaknesses of current T2V models.

## Limitations & Future Work

1. **Dependence on large-scale LLMs**: Both QG (Qwen2.5-72B) and QA (Qwen2-VL-72B) require very large models, resulting in high evaluation costs.
2. **Restricted to Yes/No questions**: Degree-level semantic alignment (e.g., the precision of an action) cannot be assessed.
3. **KA may introduce bias**: If the auxiliary LLM's commonsense knowledge is inaccurate, it may mislead the evaluation.
4. **Limited closed-source model evaluation**: ETVABench-2k evaluates only open-source models due to the prohibitive cost of large-scale closed-source API calls.
5. **Color dimension yields the lowest performance** (Spearman's ρ 39.7), suggesting that color assessment may require more specialized approaches.
6. **Temporal dynamics are not deeply evaluated**: Aspects such as action fluency and causal accuracy are not addressed.

## Related Work & Insights

- **TIFA** pioneered QG&QA-based evaluation for text-image alignment; however, direct transfer to video faces substantially greater complexity and hallucination challenges.
- **VBench** is the most comprehensive existing T2V evaluation benchmark, but its text-alignment dimension relies on ViCLIPScore, which is relatively coarse-grained.
- **VideoScore** achieves scoring by fine-tuning an MLLM, yet this paper demonstrates its human-judgment correlation reaches only 31.0.
- The proposed multi-agent QG framework and knowledge-augmented QA framework have potential for extension to other multimodal evaluation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of scene-graph-driven QG and knowledge-augmented multi-stage QA is novel, though the individual techniques (scene graphs, chain-of-thought) have prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comparisons against 7 baseline metrics, detailed ablations, evaluation of 15 models, and human annotation validation constitute a very comprehensive study.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear, the C1/C2 challenge framing is persuasive, and case analyses are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a text-to-video evaluation approach that substantially surpasses existing metrics, and the constructed benchmark offers important reference value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](../../AAAI2026/video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)

</div>

<!-- RELATED:END -->
