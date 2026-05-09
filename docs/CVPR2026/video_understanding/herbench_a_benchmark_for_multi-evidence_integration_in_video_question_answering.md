---
title: >-
  [Paper Note] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering
description: >-
  [CVPR 2026][Video Understanding][Video QA benchmark] HERBench is a video question answering benchmark specifically designed for multi-evidence integration, comprising 26,806 five-choice questions, each structurally requiring the fusion of $\ge 3$ temporally dispersed, non-overlapping visual cues. By introducing the Minimum Required Frame Set (MRFS) metric, the benchmark exposes two critical bottlenecks in current Video-LLMs: insufficient frame retrieval and evidence fusion failure.
tags:
  - CVPR 2026
  - Video Understanding
  - Video QA benchmark
  - multi-evidence integration
  - frame selection
  - long video understanding
  - temporal reasoning
date: 2026-05-08
content_hash: 5fc6caca1623ce21
---

# HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering

**Conference**: CVPR 2026
**arXiv**: [2512.14870](https://arxiv.org/abs/2512.14870)
**Code**: None
**Area**: Video Understanding / Multimodal VLM
**Keywords**: Video QA benchmark, multi-evidence integration, frame selection, long video understanding, temporal reasoning

## TL;DR

HERBench is a video question answering benchmark specifically designed for multi-evidence integration, comprising 26,806 five-choice questions, each structurally requiring the fusion of $\ge 3$ temporally dispersed, non-overlapping visual cues. By introducing the Minimum Required Frame Set (MRFS) metric, the benchmark exposes two critical bottlenecks in current Video-LLMs: insufficient frame retrieval and evidence fusion failure.

## Background & Motivation

1. **State of the Field**: Video-LLMs (e.g., GPT-4o, Gemini, Qwen2.5-VL) have achieved strong scores on existing VideoQA benchmarks, suggesting rapid progress in video understanding capabilities.

2. **Limitations of Prior Work**: Recent auditing studies reveal that high scores often stem from language priors or single-cue shortcuts rather than genuine temporal reasoning. Models can answer questions by observing a single frame or exploiting linguistic bias, and existing benchmarks fail to distinguish "true video understanding" from "shortcut exploitation."

3. **Root Cause**: Existing VideoQA benchmarks permit single-cue shortcuts—one key frame or textual common sense suffices to answer the question—making it impossible to determine whether models genuinely possess the ability to integrate multiple pieces of evidence across time.

4. **Paper Goals**: (1) Design a benchmark that structurally requires multi-evidence integration ($\ge 3$ dispersed cues); (2) Propose the quantitative MRFS metric to measure "evidential demand"; (3) Diagnose specific failure modes of current Video-LLMs—distinguishing frame selection failures from evidence fusion failures.

5. **Starting Point**: The authors define the concept of *Evidential Requirement* (ER)—the minimum number of non-redundant visual evidence pieces needed to answer a question. By enforcing $\text{ER} \ge 3$, single-cue shortcuts are fundamentally eliminated, making multi-evidence reasoning an unavoidable requirement.

6. **Core Idea**: Through structural design, each question requires at least three temporally dispersed visual cues. Combined with the MRFS metric for quantifying frame-fusion difficulty, the benchmark systematically exposes Video-LLMs' dual deficiencies in frame retrieval and evidence fusion.

## Method

### Overall Architecture

HERBench is an evaluation benchmark rather than a model. Its core components are: (1) a taxonomy of 12 compositional reasoning task types organized into 4 reasoning families; (2) a three-channel data construction pipeline (object tracking + shot segmentation + manual annotation integration); and (3) the MRFS metric for cross-benchmark comparison of evidential demand. The benchmark comprises 336 long videos (average 395 seconds) and 26,806 five-choice questions.

### Key Designs

1. **Four Reasoning Families and 12 Sub-tasks**:

    - **Function**: Covers diverse multi-evidence reasoning patterns, ensuring no question can be answered via a single frame or language prior.
    - **Mechanism**: (a) **Temporal Reasoning & Chronology (TR&C)**: Includes shot temporal ordering (TSO), multi-person duration reasoning (MPDR), and action sequence integrity identification (ASII)—requiring understanding of event order, temporal overlap, and duration comparison. (b) **Reference & Tracking (R&T)**: Includes appearance-grounded behavior interaction (AGBI), appearance-grounded attribute recognition (AGAR), and appearance-grounded localization trajectory (AGLT)—requiring maintenance of target identity binding across time. (c) **Global Consistency & Verification (GC&V)**: Includes false action memory (FAM), scene verification arrangement (SVA), and false object memory (FOM)—requiring full-video scanning to verify existence and detect absence. (d) **Multi-Entity Aggregation & Counting (MEA&N)**: Includes multi-entity grounding and localization (MEGL), action counting (AC), and region-limited people counting (RLPC)—requiring cross-temporal deduplication and set-level aggregation.
    - **Design Motivation**: These tasks reformulate existing VideoQA categories (temporal ordering, counting, etc.), but the critical distinction is that each question structurally enforces $k \ge 3$: answers must be derived from the combination of cues at multiple distinct temporal moments, precluding solutions from a single frame or local window.

2. **Three-Channel Data Construction Pipeline**:

    - **Function**: Extracts spatiotemporal information from videos at different granularities to construct high-quality multi-evidence questions.
    - **Mechanism**: **Pipeline I (Object Tracking & Trajectory Analysis)**: Employs RF-DETR + DeepSORT to obtain entity trajectories, retains the top 20% of entities by TrackRank score, and generates non-overlapping A-cards (appearance descriptions) and B-cards (behavior/trajectory descriptions) for each tracklet, deliberately separating appearance identification from behavior queries across different temporal frames. **Pipeline II (Shot Segmentation)**: Applies shot boundary detection to discretize videos into semantic segments, then uses an MLLM to generate scene cards for each segment. **Pipeline III (Manual Annotation Integration)**: Integrates human-verified narrative event logs to establish ground-truth event chronology and counts.
    - **Design Motivation**: The three channels are complementary—Pipeline I provides continuous micro-level entity dynamics, Pipeline II provides macro-level scene structure, and Pipeline III provides human-verified factual anchors. The A/B-card separation design ensures that identity-binding tasks cannot be resolved through local attribute lookup.

3. **Minimum Required Frame Set (MRFS) Metric**:

    - **Function**: Quantifies the minimum number of frames a model must integrate to answer a question, enabling fair cross-benchmark comparison.
    - **Mechanism**: Given a fixed MLLM $f$, frame selector $r$, and frame budget $x$, MRFS is defined as the minimum frame count $k$ at which the model transitions from incorrect to correct. Questions solvable from text alone ($E(f(q, \varnothing), y) = 0$) are excluded, and an adaptive binary search over $k \in [1, x]$ identifies the minimum successful index, requiring only $O(\log x)$ model calls per question.
    - **Design Motivation**: Existing metrics (Temporal Indispensability, Certificate Length) either only contrast single-frame vs. multi-frame performance or rely on manual annotation. MRFS is an automated, model-centric metric that directly quantifies the multi-evidence aggregation challenge.

### Quality Control

- Token-level similarity checks and manual review ensure A/B-cards do not leak information across channels.
- Questions answered correctly by $\ge 3/4$ blind LLMs are discarded to remove language bias.
- Stratified sampling of 15% undergoes expert verification to confirm $k \ge 3$ compliance and answer uniqueness.
- Human annotators achieve 88.8% accuracy under full video access and 95.7% accuracy under oracle frames.

## Key Experimental Results

### Main Results

Thirteen SOTA Video-LLMs are evaluated; overall accuracy ranges from only 31–42% (random chance: 20%):

| Model | TR&C Avg. | R&T Avg. | GC&V Avg. | MEA&N Avg. | Overall |
|-------|-----------|----------|-----------|------------|---------|
| GPT-4.1 | 25.4 | 66.0 | 37.1 | 29.0 | 39.4 |
| Gemini-2.5-Flash | 29.7 | 69.9 | 34.9 | 26.8 | 40.3 |
| Qwen2.5-VL-72B | 26.9 | 70.9 | 36.6 | 24.4 | 39.7 |
| Ovis-2.5-9B | 18.9 | 73.5 | 46.8 | 29.2 | **42.1** |
| InternVL3.5-8B | 33.6 | 70.2 | 29.7 | 30.8 | 41.1 |

### Cross-Benchmark MRFS Comparison

| Benchmark | # Videos | # Questions | MRFS↑ | Language Debiasing | Forced Fusion |
|-----------|----------|-------------|-------|--------------------|---------------|
| MVBench | 4,000 | 4,000 | 3.52 | ✗ | ✗ |
| Video-MME | 900 | 2,700 | 5.31 | ✗ | ✗ |
| MINERVA | 223 | 1,515 | 5.14 | ✓ | ✗ |
| **HERBench** | **336** | **26,806** | **5.49** | **✓** | **✓** |

### Key Findings

- **Frame Retrieval Bottleneck (Finding 1)**: Although adaptive frame selectors outperform uniform sampling, they fall significantly short of oracle keyframes—models simply fail to locate the critical evidence frames.
- **Fusion Bottleneck (Finding 2)**: Even when oracle frames are provided, model accuracy improves only modestly, indicating that models cannot correctly allocate attention across all key frames and integrate the information.
- The R&T family achieves relatively higher scores (~60–73%), as appearance descriptions in these tasks provide strong visual anchors; TR&C and MEA&N families score the lowest (<30%), reflecting severe deficiencies in temporal reasoning and multi-entity aggregation.
- Smaller models (e.g., Ovis-2.5-9B) outperform larger models (e.g., GPT-4.1) on certain tasks, suggesting that model scale alone does not explain the performance gap.

## Highlights & Insights

- **Elegance of the MRFS Metric**: Rather than naively counting the number of required frames, MRFS fixes a frame selector and uses binary search to find the minimum successful frame count, while excluding text-only-solvable questions—making cross-benchmark comparisons both fair and computationally efficient.
- **A/B-Card Separation Design**: Appearance descriptions and behavior queries are deliberately placed at different temporal frames, compelling the model to first localize the target via appearance description and then track it to the moment of action occurrence. This design elegantly transforms identity binding into an unavoidable multi-frame reasoning requirement.
- **Dual-Bottleneck Diagnostic Framework**: By decoupling frame selection from fusion reasoning through oracle-frame experiments, the benchmark clearly identifies "finding the frames" and "effectively using the frames" as two independent challenges, providing clear directions for future research.

## Limitations & Future Work

- Portions of the benchmark are generated through automated pipelines, potentially introducing residual systematic biases.
- With only 336 videos, scene diversity may be insufficient to represent all real-world scenarios.
- MRFS depends on a specific frame selector and model; different combinations may yield different orderings.
- The benchmark diagnoses model failures without proposing concrete remedies (it is purely diagnostic in nature).
- The relatively high accuracy on R&T tasks may indicate that the ER design for this family is not sufficiently strict.

## Related Work & Insights

- **vs. Video-MME**: Video-MME emphasizes longer contexts but does not control evidential demand; HERBench ensures high ER through structural design, positioning evidence density—rather than video length—as the source of difficulty.
- **vs. MINERVA**: MINERVA also applies language debiasing and achieves a relatively high MRFS, but focuses on multi-step reasoning and reasoning-chain auditing rather than enforced multi-frame evidence aggregation.
- **vs. MVBench**: MVBench covers diverse temporal reasoning tasks but uses short videos, and its questions can often be answered from a single frame.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first VideoQA benchmark centered on evidential requirement as a core design principle; the MRFS metric is a genuine innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluates 13 models, performs cross-benchmark MRFS comparison, and conducts multi-dimensional diagnostic analysis.
- Writing Quality: ⭐⭐⭐⭐ — Framework is clear and the task taxonomy is comprehensive, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ — Diagnoses critical deficiencies in Video-LLMs and provides an important reference for future progress in the field.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](egopointvqa_gesture_based_egocentric_video_qa.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)

<!-- RELATED:END -->
