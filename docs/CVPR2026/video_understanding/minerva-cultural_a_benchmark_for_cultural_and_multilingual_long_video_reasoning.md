---
title: >-
  [Paper Note] MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning
description: >-
  [CVPR 2026][Video Understanding][Video QA] This paper introduces MINERVA-Cultural, a benchmark comprising 2,400 manually annotated video reasoning questions spanning 18 language/region locales, and reveals severe deficiencies in cultural visual perception among state-of-the-art Video-LLMs through evidence graphs and an iterative error isolation strategy (best model Gemini-2.5-Pro: 45.07% vs. human: 95.22%).
tags:
  - CVPR 2026
  - Video Understanding
  - Video QA
  - cross-cultural understanding
  - multilingual reasoning
  - long video
  - evidence graph error analysis
date: 2026-05-08
content_hash: 8eca431c3c9fb9ee
---

# MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning

**Conference**: CVPR 2026
**arXiv**: [2601.10649](https://arxiv.org/abs/2601.10649)
**Code**: Coming soon
**Area**: Video Understanding / Multicultural Benchmarking
**Keywords**: Video QA, cross-cultural understanding, multilingual reasoning, long video, evidence graph error analysis

## TL;DR

This paper introduces MINERVA-Cultural, a benchmark comprising 2,400 manually annotated video reasoning questions spanning 18 language/region locales, and reveals severe deficiencies in cultural visual perception among state-of-the-art Video-LLMs through evidence graphs and an iterative error isolation strategy (best model Gemini-2.5-Pro: 45.07% vs. human: 95.22%).

## Background & Motivation

1. **Background**: Video understanding has advanced substantially, with long video comprehension emerging as a focal research area. Benchmarks such as EgoSchema, LongVideoBench, and MLVU have driven model progress, and frontier models including GPT-5 and Gemini-2.5 achieve strong performance on standard benchmarks.

2. **Limitations of Prior Work**: (a) Existing video benchmarks are dominated by Western content and English, introducing significant evaluation bias; (b) cross-cultural benchmarks such as ViMUL-Bench rely on automatic translation while their visual content still centers on Western concepts; (c) evaluation focuses solely on final answer correctness, ignoring specific failure modes within the reasoning process.

3. **Key Challenge**: Training data for current models is dominated by Euro-American and English content, leading to severely inadequate understanding of low-resource languages and cultures (e.g., Tamil, Telugu). Moreover, simple accuracy metrics cannot reveal precisely where in the reasoning chain a model fails.

4. **Goal**: (a) Construct a genuinely multicultural, multilingual video reasoning benchmark annotated entirely by local experts; (b) provide human reasoning chains as diagnostic tools; (c) develop fine-grained error analysis methods to localize the root causes of model failures.

5. **Key Insight**: Every question is required to demand "visual-cultural understanding" as a skill, tightly coupling perception with cultural knowledge. Human reasoning processes are modeled as directed acyclic graphs (DAGs), enabling iterative isolation and classification of errors.

6. **Core Idea**: Expose and quantify systemic deficiencies of Video-LLMs in cultural visual perception through a long-video reasoning benchmark fully annotated by local experts from 18 regions (with no reliance on translation), combined with an evidence-graph-based iterative analysis methodology.

## Method

### Overall Architecture

MINERVA-Cultural consists of two components: (1) a benchmark dataset — 540 videos and 2,400 questions covering 18 language/region locales across 6 cultural domains, each question accompanied by a manually authored multi-step reasoning chain; and (2) a diagnostic methodology — reasoning chains are formalized as evidence graphs, and iterative error isolation is applied to precisely localize model failure points.

### Key Designs

1. **Culture-Centric Curation**:

    - **Function**: Ensures that every question genuinely requires deep understanding of visual-cultural content to answer correctly.
    - **Mechanism**: A four-stage pipeline — (1) *Cultural video selection*: local reviewers filter YouTube videos according to a cultural taxonomy, requiring native-language audio, culturally salient scenes, and duration exceeding one minute; (2) *Difficulty calibration*: 10% of samples are annotated first to verify that questions are sufficiently challenging for LLMs (not answerable from a single frame, audio alone, or common sense); (3) *Correctness calibration*: an independent reviewer answers each question without access to the reference answer; disagreements are resolved through revision until consensus is reached; (4) *Final annotation and audit*.
    - **Design Motivation**: Avoids the pseudo-multiculturalism prevalent in other benchmarks that combine automatic translation with Western imagery. Each question must require at least two reasoning skills, one of which must be visual-cultural understanding.

2. **Evidence Graph**:

    - **Function**: Formalizes unstructured human reasoning chains into directed acyclic graphs to precisely localize reasoning failure points.
    - **Mechanism**: An LLM decomposes each human reasoning chain into atomic evidence nodes — including visual observations (with timestamps), external knowledge retrieval, and logical inference steps. Prerequisite dependency edges are established between nodes, such that an error at a given node blocks all downstream reasoning. On average, each question requires 5.0 atomic evidence nodes, and 63% of evidence nodes are anchored to specific video timestamps.
    - **Design Motivation**: Simple accuracy cannot distinguish between "failure to perceive cultural elements" and "logical reasoning errors." The graph structure captures causal dependencies as well as spatiotemporal relationships.

3. **Iterative Error Isolation**:

    - **Function**: Exhaustively identifies all model failure modes, preventing early errors from masking subsequent failures.
    - **Mechanism**: A three-stage loop — (1) *Traversal*: the model's reasoning is compared against human evidence along the evidence graph; traversal halts upon encountering a missing evidence node; (2) *Error labeling*: "divergences" (where the model takes a reasonable alternative path, occurring in only 2% of cases) are distinguished from "errors" (labeled according to a taxonomy: temporal localization, spatial localization, attribute misidentification, hallucination, etc.); (3) *Prompted correction and re-evaluation*: the model receives a corrective prompt, evaluated nodes are pruned from the graph, and unevaluated nodes are re-assessed. This loop continues until all nodes are evaluated (99.7% of cases are resolved within 5 iterations).
    - **Design Motivation**: A single-pass error analysis can only detect the first error, causing subsequent reasoning errors to be masked. The iterative approach uncovers an additional 22% of errors, including 78 reasoning errors that were originally obscured by perception errors.

### Loss & Training

This is a benchmark paper; no model training is involved. Evaluation employs an LLM Judge (Gemini-2.5-Flash), scoring open-ended responses on a 0–2 three-point scale.

## Key Experimental Results

### Main Results

**Model performance across 18 regions (accuracy %):**

| Model | Aggregate | Highest Region | Lowest Region |
|-------|-----------|---------------|---------------|
| Qwen-2.5-VL | 12.75 | en-GB (25.70) | ta-IN (3.60) |
| Qwen-3-VL | 21.50 | en-GB (34.58) | te-IN (12.40) |
| Claude-Sonnet-4 | 23.36 | en-GB (29.91) | te-IN (14.40) |
| GPT-5-mini | 36.64 | ko-KR (51.90) | ta-IN (16.40) |
| GPT-5 | 42.20 | id-ID (56.34) | te-IN (23.60) |
| Gemini-2.5-Flash | 35.84 | de-DE (51.90) | ta-IN (20.00) |
| **Gemini-2.5-Pro** | **45.07** | **ko-KR (64.29)** | **te-IN (28.00)** |
| **Human Baseline** | **95.22** | it-IT (98.24) | de-DE (90.51) |

### Ablation Study

| Analysis Dimension | Key Finding |
|-------------------|-------------|
| Audio vs. video-only | Adding audio yields an average gain of 4.32% (zh-TW: +8.15%, id-ID: +7.09%) |
| Thinking budget (tokens) | Accuracy increases from 35.9% to 45.9% as budget grows from 128 to 2k tokens, then saturates |
| Frame count (1→512) | Monotonically increasing with diminishing returns, indicating temporal reasoning is required |
| Error type analysis | 75% of errors are attributable to cultural visual perception (temporal localization + spatial localization + attribute misidentification + hallucination) |

### Key Findings

- **Large human–machine gap**: The strongest model, Gemini-2.5-Pro (45.07%), lags behind human performance (95.22%) by approximately 50 percentage points.
- **Pronounced cultural disparity**: The same model shows a 36-percentage-point gap between Korean (ko-KR: 64.29%) and Telugu (te-IN: 28.00%), revealing severe cultural bias.
- **South Indian languages are the weakest**: ta-IN (31.60%), te-IN (28.00%), and mr-IN (38.72%) all fall far below English-locale performance.
- **75% of errors stem from cultural visual perception**: Models fail not due to insufficient reasoning ability but because they cannot "see" or "interpret" culturally specific visual elements (attire, rituals, symbols, etc.).
- **Iterative error analysis is essential**: 22% of errors are missed in a single-pass analysis, particularly reasoning errors masked by perception errors.
- **Perception errors in low-resource languages are 1.4× those in high-resource languages**: Cultural perception errors in ar-EG and ta-IN exceed those in en-GB and ja-JP by approximately 40%.

## Highlights & Insights

- **Key finding that "cultural understanding is a visual perception problem"**: Models do not fail because they lack reasoning capability; they fail because they cannot perceive culturally specific visual elements. This pinpoints the direction for improvement — diversifying pretraining data culturally rather than merely enhancing reasoning capacity.
- **Evidence graph + iterative error isolation as a diagnostic methodology**: This approach provides substantially deeper analytical insight than simple accuracy metrics and is transferable to any benchmark involving multi-step reasoning (e.g., mathematical reasoning, code generation).
- **High-standard fully manual annotation**: Every question is annotated by a local cultural expert and calibrated by an independent reviewer, eliminating translation artifacts. This is a resource-intensive but highly valuable contribution to the research community.

## Limitations & Future Work

- **Limited scale**: 2,400 questions across 18 regions yield approximately 130 questions per region, which may be insufficient to cover all cultural scenarios.
- **Dependence on LLM Judge**: Although majority voting mitigates the issue, LLM-based evaluation may itself carry biases toward certain languages.
- **Incomplete cultural coverage**: Parts of Africa, South America, and Southeast Asian minority language communities are not included.
- **Future directions**: (a) Expansion to additional regions and languages; (b) development of culturally aware visual pretraining data strategies; (c) open-sourcing automated diagnostic tools based on evidence graphs; (d) exploration of strategies to balance cultural distributions in pretraining data.

## Related Work & Insights

- **vs. ViMUL-Bench**: ViMUL-Bench covers 14 languages but incorporates non-cultural videos and partially translated annotations. MINERVA-Cultural is annotated entirely in native languages by local experts, with video content that is uniformly culturally specific.
- **vs. MINERVA**: MINERVA provides reasoning chains and error taxonomies; MINERVA-Cultural extends this foundation by adding a multicultural dimension and an evidence-graph-based analysis methodology, enabling finer-grained diagnosis.
- **vs. LongVideoBench / MLVU**: These benchmarks target general video understanding without addressing cultural or multilingual dimensions; MINERVA-Cultural fills this gap.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First large-scale, fully manual multicultural multilingual video reasoning benchmark; evidence graph analysis methodology is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 models, 18 regions, multi-dimensional analysis (audio, frame count, thinking budget, error types), and human baseline.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is well-grounded; annotation pipeline is described in detail; analysis is deep and insightful.
- **Value**: ⭐⭐⭐⭐⭐ — Exposes cultural bias in AI systems, defines a critical direction for model improvement, and carries significant implications for fair AI and global deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)
- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)

</div>

<!-- RELATED:END -->
