---
title: >-
  [Paper Note] ViTED: Video Temporal Evidence Distillation
description: >-
  [CVPR 2025][Video Understanding][Video Question Answering] ViTED proposes a framework that automatically generates temporal grounding chain of evidence, unifying evidence collection, temporal grounding, and question-answering reasoning into a single video-language model, enhancing complex video QA capabilities through evidence distillation.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Question Answering"
  - "Chain-of-Evidence Reasoning"
  - "Temporal Grounding"
  - "Chain-of-Thought"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: c94a26c014db8f5e
---

# ViTED: Video Temporal Evidence Distillation

**Conference**: CVPR 2025  
**arXiv**: [2503.12855](https://arxiv.org/abs/2503.12855)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Question Answering, Chain-of-Evidence Reasoning, Temporal Grounding, Chain-of-Thought, Knowledge Distillation

## TL;DR

ViTED proposes a framework that automatically generates temporal grounding chain of evidence, unifying evidence collection, temporal grounding, and question-answering reasoning into a single video-language model, enhancing complex video QA capabilities through evidence distillation.

## Background & Motivation

Video Question Answering (VideoQA) is a core task of video understanding. Existing video large models possess two key limitations:

- **Keyframe omission due to uniform sampling**: Models sample a fixed number of frames at fixed intervals, which may miss unevenly distributed key evidence in the video (e.g., a brief waving gesture).
- **Lack of temporal grounding and multi-step reasoning capabilities**: Models cannot associate evidence with specific temporal segments in the video, rendering them unable to perform multi-step reasoning like "first locate evidence -> then reason -> then answer."

For example, answering "Why does the baby put their hand in their mouth?" requires: (1) locating the segment where the mom is feeding with a spoon; (2) observing the baby's uncomfortable expression; (3) reasoning that the baby is trying to get the food out. Existing models cannot complete this chain of reasoning.

Although existing temporal grounding models can locate temporal segments for specific descriptions, they require **knowing what to ground in advance**—they cannot autonomously identify and ground relevant evidence starting from the question.

## Method

### Overall Architecture

ViTED consists of three stages: (1) **Evidence Pool Generation**: Partitioning the video into hierarchical, multi-granular segments, and using a VLM to generate question-related descriptions for each segment; (2) **Evidence Chain Search**: Employing an LLM to search for the sequence of evidence chains that best supports the correct answer within the evidence pool via beam search; (3) **Evidence Distillation Training**: Incorporating the searched evidence chains into the training data to train the model to simultaneously generate evidence chains and answers.

### Key Designs

**1. Hierarchical Evidence Pool Generation**

- **Function**: Comprehensively extract potential evidence at different temporal scales from the video, covering from global contexts to fine-grained local actions.
- **Mechanism**: Non-uniformly partition the video into $N=5$ levels, $(L,S) \in \{(1/16, 1/16), (1/8, 1/16), (1/4, 1/8), (1/2, 1/4), (1, 1)\}$. Generate question-related descriptions for each segment using LLaMA-3.2-Vision-11B to form the evidence pool $E = \{(t_s, t_e, \epsilon)_i\}$.
- **Design Motivation**: Evidence is distributed non-uniformly in videos (a global activity vs. a brief action), requiring multi-granular coverage. Non-uniform partitioning is less likely to miss key information than uniform sampling.

**2. Evidence Chain Search and Refinement**

- **Function**: Find the evidence chain sequence from a large amount of noisy evidence that is most likely to infer the correct answer.
- **Mechanism**: First use an LLM to narrow down the evidence pool $E \rightarrow E^*$ (retaining top-K), then perform beam search: initialize a beam of width $W=K/2$, iteratively add new evidence to the chain, calculate $P(A|Q, C_i \oplus ev_j)$ and retain the top-W chains. After convergence, use an LLM to summarize the optimal chain to make it temporally and causally coherent. Finally, filter and retain chains that can correctly infer the answer.
- **Design Motivation**: A single piece of evidence only provides partial information; multiple pieces of evidence must be combined to form a reasoning chain. Beam search balances efficiency and quality.

**3. Curriculum Evidence Distillation Training**

- **Function**: Distill the ability to generate and reason with evidence chains into a single VLM.
- **Mechanism**: Two-stage training—Stage-1 standard instruction tuning (Q→A), Stage-2 evidence distillation (Q→Evidence Chain + A). The standard next token prediction cross-entropy loss is used during training. During inference, the model first generates the timestamped evidence chain, then answers the question based on the evidence.
- **Design Motivation**: Curriculum learning avoids learning complex tasks from the very beginning. The model first learns basic QA capability, then learns evidence reasoning.

### Loss & Training

Standard next token prediction cross-entropy loss, optimized on Stage-1 (answer tokens) and Stage-2 (evidence chain + answer tokens) respectively.

## Key Experimental Results

### Main Results: VideoQA Benchmark Comparison (7B Models)

| Method | CinePile | PerceptionTest | NExT-QA | STAR | NExT-GQA |
|------|----------|---------------|---------|------|----------|
| LLaMA-3.2V (11B) | 39.55 | 52.65 | 67.58 | 45.62 | 11.64 |
| LLaVA-OneVision | 46.42 | - | - | - | - |
| SeViLA (4B) | - | - | 73.8 | 64.9 | 16.6 |
| **ViTED** | **48.2** | **64.8** | **80.1** | **66.2** | **22.4** |

### Ablation Study: Impact of Evidence Distillation

| Training Method | NExT-QA | NExT-GQA |
|---------|---------|----------|
| w/o CoT | 75.3 | 14.2 |
| + "step-by-step" prompt | 76.1 | 15.1 |
| + Evidence Distillation (ViTED) | **80.1** | **22.4** |

### Key Findings

- ViTED outperforms GPT-4-driven Agent methods on NExT-GQA (temporal grounding QA) in a zero-shot manner, proving that the distilled model internalizes temporal grounding capabilities.
- Human evaluation shows that the quality of evidence chains generated by ViTED is significantly higher than the reasoning explanations of baseline VLMs.
- 54% of the questions in NExT-QA require locating and reasoning over one or more temporal windows, where the evidence chain method demonstrates the greatest advantage.
- Evidence chains average 2-3 hops, requiring the collection of different granular clues from different temporal positions in the video.
- Simple "step-by-step" prompting has limited effectiveness (+0.8%), whereas genuine evidence distillation brings a substantial improvement (+4.8%).

## Highlights & Insights

1. **Expanding CoT from the text domain to the video domain**: Not just simple textual reasoning chains, but timestamped video evidence chains where each piece of evidence is associated with a specific temporal segment of the video.
2. **Automated data generation pipeline**: High-quality evidence chain training data can be generated from existing VideoQA datasets without manual annotation.
3. **Single model as a replacement for Agent systems**: Competencies of multi-module Agents (evidence collection + grounding + reasoning) are compressed into a single forward pass via distillation.

## Limitations & Future Work

- Evidence pool generation relies on an external VLM (LLaMA-3.2-Vision), whose quality directly impacts downstream performance.
- The computational overhead of beam search is relatively high, making it unsuitable for real-time applications.
- Currently, only data augmentation for existing QA pairs is handled, without exploring open-ended questions.
- Future work could explore online evidence search (dynamically searching during inference instead of relying on distilled knowledge from training).

## Related Work & Insights

- **Relationship with SeViLA**: SeViLA uses a frame selector to locate keyframes, while ViTED goes a step further to locate temporal windows and generate textual evidence.
- **Relationship with Agent methods**: Agent systems (like VIP, etc.) perform reasoning through multiple API calls, whereas ViTED distills this into single-model, single-pass inference.
- **Insights**: The upgrade of video understanding from "perception" to "reasoning" requires explicit evidence support, rather than end-to-end black-box prediction.

## Rating

⭐⭐⭐⭐

Introduces timestamped evidence chain reasoning to video question answering for the first time, with an elegantly designed automated data generation pipeline. Achieves SOTA on multiple benchmarks, notably surpassing GPT-4 Agents on NExT-GQA. The technical pipeline is complete and reproducible. The main limitation lies in the computational overhead of training and data generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)
- [\[ICML 2026\] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](../../ICML2026/video_understanding/foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [\[ACL 2025\] From Teacher to Student: Tracking Memorization Through Model Distillation](../../ACL2025/video_understanding/from_teacher_to_student_tracking_memorization_through_model_distillation.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

</div>

<!-- RELATED:END -->
