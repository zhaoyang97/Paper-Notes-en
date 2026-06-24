---
title: >-
  [Paper Note] FAVE: A Structured Benchmark for Fine-Grained Audio-Visual Temporal Evaluation in Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Audio-visual large models] FAVE is a three-layer benchmark specifically designed to evaluate whether Audio-Visual Large Language Models (AVLLMs) can align audio and video streams within the same time window and perform fine-grained temporal reasoning. Using a scalable pipeline involving shot segmentation, dual-modal captioning, GPT synthesis, and human verification, it constructs nearly 10,000 timestamped QA pairs based on QVHighlights. Evaluations…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Audio-visual large models"
  - "fine-grained temporal reasoning"
  - "cross-modal alignment"
  - "multimodal benchmark"
  - "AVLLM"
date: 2026-05-08
content_hash: 67a3929ac9a80bd0
---

# FAVE: A Structured Benchmark for Fine-Grained Audio-Visual Temporal Evaluation in Multimodal LLMs

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_FAVE_A_Structured_Benchmark_for_Fine-Grained_Audio-Visual_Temporal_Evaluation_in_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Multimodal VLM / Audio-Visual Understanding  
**Keywords**: Audio-visual large models, fine-grained temporal reasoning, cross-modal alignment, multimodal benchmark, AVLLM  

## TL;DR
FAVE is a three-layer benchmark specifically designed to evaluate whether Audio-Visual Large Language Models (AVLLMs) can align audio and video streams within the same time window and perform fine-grained temporal reasoning. Using a scalable pipeline involving shot segmentation, dual-modal captioning, GPT synthesis, and human verification, it constructs nearly 10,000 timestamped QA pairs based on QVHighlights. Evaluations of 13 SoTA models show that even the strongest model, Gemini 1.5, performs significantly below human levels, while open-source models suffer near-total failure, indicating that joint cross-modal temporal understanding remains an open problem.

## Background & Motivation
**Background**: Audio-Visual Large Language Models (AVLLMs, such as VideoLLaMA, PandaGPT, and Qwen2.5-Omni) encode audio and visual features into a single LLM, achieving success in tasks like video QA and captioning. Existing audio-visual datasets (AVQA, Music-AVQA, AVSD, etc.) mostly consist of short clips and single events with simple question types.

**Limitations of Prior Work**: Existing benchmarks rarely evaluate "cross-modal temporal" capabilities. They either label only a single modality (visual-only or audio-only, which is unsuitable for AVLLMs) or provide both modalities but with simple task types, lacking event-level temporal relationships or precise timestamps. Ours observes a counter-intuitive phenomenon: adding audio to Vision-Language Models (VLMs) for Moment Retrieval actually degrades performance; native AVLLMs struggle similarly. This suggests that the "joint audio-visual temporal capability" of existing models has never been systematically quantified.

**Key Challenge**: In real videos, audio and visual signals are densely complementary (what is on screen and when sounds occur). Understanding a video requires aligning both streams on the **same timestamp** before reasoning. However, most existing models process visual tokens followed by audio tokens; this sequential encoding is naturally detrimental to cross-modal alignment at the same moment. Furthermore, the community lack both metrics to characterize this ability and data to elicit it.

**Goal**: Deconstruct the vague "audio-visual temporal understanding" into quantifiable sub-abilities: (1) cross-modal alignment within the same time window; (2) multi-scale temporal perception, including basic relationships (order, proximity, position) and fine-grained descriptions for specific segments.

**Key Insight**: Speech is deemed the most semantically dense audio signal in everyday videos. Thus, focusing primarily on speech (supplemented by ~20% manually annotated ambient sounds), tasks are designed around "what should be said or seen at the same timestamp" to force models to expose alignment and temporal weaknesses.

**Core Idea**: A **three-layer progressive** structured benchmark (Alignment → Basic Temporal Relations → Fine-grained Segment Description) called FAVE is proposed. Combined with a construction and filtering mechanism to "prevent single-modality shortcuts," it measures the real performance gap in the cross-modal temporal capabilities of AVLLMs.

## Method
FAVE is essentially an "evaluation protocol + data construction pipeline" rather than a new model. To address the lack of joint temporal evaluation in existing data, the authors designed a three-layer task system and a scalable annotation pipeline to produce high-quality samples that are timestamped, dual-modality balanced, and resistant to textual priors.

### Overall Architecture
FAVE is built upon real videos from QVHighlights (QVH), with an average length of ~150s. The construction side uses a pipeline: shot-level segmentation for semantically coherent clips, dual-modal (visual and speech) captioning for each clip, and GPT-based synthesis of `[start, end, caption]` events and modality-specific QA, followed by multiple rounds of human verification. The evaluation side organizes samples into three subsets: FAVE-align (cross-modal alignment), FAVE-low (basic temporal relations), and FAVE-high (fine-grained segment description), each using different inputs, outputs, and metrics.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["QVHighlights Real Videos<br/>(Avg ~150s)"] --> B["Data Construction Pipeline<br/>Shot Seg → Dual-modal Captions → GPT Synthesis → Human Check"]
    B --> C["Modality Parity & Shortcut Prevention<br/>Speaker Clustering + Parity Verification + Textual Prior Filtering"]
    C --> D["Three-layer Task System<br/>Alignment / Basic Temporal / Fine-grained Description"]
    D -->|Single Modality Segment → Opposite Caption<br/>GPT Score (1-5)| E["FAVE-align (3,557)"]
    D -->|Two Events → Temporal Label<br/>Accuracy| F["FAVE-low (4,546)"]
    D -->|Video Segment → Generated Description<br/>V/A/Overall Score (1-5)| G["FAVE-high (1,781)"]
```

### Key Designs

**1. Three-layer Task System: Decomposing temporal understanding into quantifiable progressive abilities**

To address the limited task types in existing benchmarks, FAVE splits capabilities into three levels, increasing in difficulty from alignment to reasoning to generation. **FAVE-align** targets "alignment within the same time window": given a description of one modality at a timestamp, the model must describe the other modality at that same moment. This includes Vision-to-Audio (V2A) and Audio-to-Vision (A2V). Misalignment leads to temporal hallucinations, exposing alignment weaknesses. **FAVE-low** evaluates basic temporal relations independent of absolute timestamps, including: Relative order (which of two events came first), Temporal Proximity (whether two events are adjacent), and Event position (whether an event occurs in the early or late part of the video). These tasks are architecture-agnostic. **FAVE-high** is Moment-to-Caption: given a timestamped segment, the model generates a fine-grained description integrating both audio and video, testing high-level temporal synthesis.

**2. Scalable Dual-modal Annotation Pipeline: Leveraging pre-trained models, GPT synthesis, and human oversight**

To scale fine-grained annotation with precise timestamps, FAVE uses a pipeline that minimizes human labor to the verification stage. First, **TransNetV2** performs shot boundary detection. Second, dual-path captioning is used: **LongVA** and **InternVL2.5** for visual dynamic events and frame details; **Whisper** and **3D-Speaker** for speech recognition and multi-speaker diarization. Ambient sounds (~20%) are **manually annotated** due to the noise of automatic models. Third, **GPT event synthesis** follows a structured flow to identify time units where both modalities contribute and generate `[start, end, multi-modal caption, visual-QA, audio-QA]`. A critical protocol is the **strict isolation of GPT units**: the GPT used for annotation is different (distinct prompts, seeds, and non-shared text) from the one used for evaluation to prevent data leakage. Finally, 15 trained annotators conducted two rounds of independent verification (inter-annotator agreement ~85%).

**3. Defense against Single-modality Shortcuts & Textual Prior Filtering**

To prevent models from guessing correctly based on a single modality or textual common sense, FAVE employs two mechanisms. First, **modality parity**: GPT-assisted verification ensures audio and visual captions contribute equally to semantic understanding. Second, **textual prior filtering for FAVE-low**: the authors use strict GPT quality control to **remove QA pairs where the temporal order can be guessed solely from the captions**. Samples are only retained if the correct answer requires "grounding the captions into the actual video content" rather than relying on linguistic priors.

### Evaluation Protocol
The three task layers are defined as follows:

| Task | Input | Output | Metric |
|------|------|------|------|
| FAVE-align | Single modality segment (V or A) | Opposite modality caption | GPT 5-point similarity |
| FAVE-low | Two event captions | Temporal label (Order/Proximity/Pos) | Accuracy (%) |
| FAVE-high | Video segment | Generated description | 5-point relevance (V/A/Overall) |

Open-ended tasks use a rubric: 1 (Irrelevant), 2 (Partially relevant), 3 (Coarse match), 4 (Fine match), 5 (Semantically equivalent).

## Key Experimental Results

### Main Results (Ranking across FAVE layers)
Models are grouped by paradigm: Sequential (independent encoding), Interleaved (token-level interleaving), and Vision-only. Align/high are 1–5 scores; low is Accuracy (%).

| Model | Paradigm | align Avg. | low Overall | high Overall |
|------|------|-----------|-------------|--------------|
| **Human Baseline** | / | **4.63** | **88.98** | **4.23** |
| Gemini 1.5 Flash | / | 4.04 | 75.34 | 2.81 |
| Gemini 1.5 Pro | / | 3.93 | 74.86 | 2.75 |
| Qwen2.5-Omni | Interleaved | 3.09 | 67.77 | 2.01 |
| LongVALE | Sequential | 3.25 | 62.73 | 2.37 |
| VideoLLaMA2 | Sequential | 3.01 | 59.39 | 2.25 |
| NextGPT | Sequential | 1.96 | 52.63 | 2.03 |

Core conclusions: (1) Even Gemini 1.5 Flash scores only 2.81 in the high task, far below the human 4.23; (2) Gemini excels in alignment; (3) No open-source model achieves effective joint audio-visual temporal understanding.

### Ablation Study (Temporal relationship sub-tasks)

| Model | Order | Proximity | Position |
|------|-------|-----------|----------|
| Gemini 1.5 Flash | 80.00 | 63.03 | 83.00 |
| Gemini 1.5 Pro | 78.73 | 59.05 | **86.80** |
| Qwen2.5-Omni | 73.73 | 60.88 | 68.69 |
| Human | 92.72 | 83.11 | 91.11 |

**Proximity (judging event adjacency)** is the weakest sub-ability across all models (even Gemini ~60%), indicating that distinguishing event boundaries is the most difficult temporal sub-task.

### Key Findings
- **Information loss at the input or model level?** Increasing sample frames (Figure 5) does not significantly improve scores. The bottleneck is the **model architecture** rather than the input; temporal information is lost during feature extraction or intra-LLM processing.
- **Sequential vs. Interleaved Paradigm**: Sequential processing hinders alignment between characteristics at the same moment. Interleaved encoding attempts to mitigate this but often **disrupts intra-modal coherence** (e.g., breaking a continuous speech segment). Balancing "cross-modal alignment" and "intra-modal coherence" remains an open dilemma.

## Highlights & Insights
- **Decomposition into quantifiable protocols**: Translating the abstract "temporal understanding" into V2A/A2V alignment, order/proximity/position relations, and V/A-specific generation scores provides a clear diagnostic of model failures.
- **Textual Prior Filtering**: Actively removing QA pairs guessable from text prevents the benchmark from degrading into a text-only common sense test.
- **Engineering Discipline**: Strict isolation between the labeling GPT and the judge GPT addresses the "self-evaluation" loophole common in LLM-as-judge benchmarks.

## Limitations & Future Work
- **Speech Centricity**: Focuses heavily on speech; coverage of environmental sounds and music is limited.
- **Dependency on Commercial LLMs**: Synthesis and scoring rely heavily on GPT, potentially inheriting its biases.
- **Error Propagation**: The pipeline relies on multiple pre-trained models (TransNetV2, Whisper, etc.), where errors can cascade.
- **Diagnostic only**: FAVE identifies architectural bottlenecks but does not propose a new model architecture to solve them.

## Related Work & Insights
Compared to **AVQA/Music-AVQA**, FAVE uses longer videos (~150s) and multi-level temporal tasks with precise timestamps. Compared to **LongVALE**, FAVE introduces missing "event temporal relationship" modeling. Compared to **TimeChat/VTimeLLM** (vision-only temporal VLMs), FAVE shows that many AVLLMs do not effectively utilize audio to outperform vision-only counterparts in visual accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HAVE-Bench: Hierarchical Audio-Visual Evaluation from Perception to Interaction](have-bench_hierarchical_audio-visual_evaluation_from_perception_to_interaction.md)
- [\[CVPR 2026\] CoV-Align: Efficient Fine-grained Cross-Modal Alignment with Cohesive Visual Semantics Priority](cov-align_efficient_fine-grained_cross-modal_alignment_with_cohesive_visual_sema.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] HanDyVQA: A Video QA Benchmark for Fine-Grained Hand-Object Interaction Dynamics](handyvqa_a_video_qa_benchmark_for_fine-grained_hand-object_interaction_dynamics.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)

</div>

<!-- RELATED:END -->
