---
title: >-
  [Paper Note] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection
description: >-
  [CVPR 2025][Reasoning][Video Chain-of-Thought] VideoEspresso constructs a large-scale video CoT reasoning dataset of over 200k samples (containing spatial bounding box and temporal grounding annotations). It also proposes a hybrid framework, VideoQA-SC, which employs a lightweight 1.5B model to select an average of 2.36 core frames, followed by an 8B reasoning model performing two-stage evidence extraction and answer generation. With only 1.8% of the frames and 14.7% of the c…
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Video Chain-of-Thought"
  - "Core Frame Selection"
  - "Video QA"
  - "Multimodal Reasoning"
  - "Dataset"
date: 2026-05-08
content_hash: 785757301d9b941e
---

# VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection

**Conference**: CVPR 2025  
**arXiv**: [2411.14794](https://arxiv.org/abs/2411.14794)  
**Code**: [https://github.com/hshjerry/VideoEspresso](https://github.com/hshjerry/VideoEspresso)  
**Area**: LLM Reasoning  
**Keywords**: Video Chain-of-Thought, Core Frame Selection, Video QA, Multimodal Reasoning, Dataset

## TL;DR

VideoEspresso constructs a large-scale video CoT reasoning dataset of over 200k samples (containing spatial bounding box and temporal grounding annotations). It also proposes a hybrid framework, VideoQA-SC, which employs a lightweight 1.5B model to select an average of 2.36 core frames, followed by an 8B reasoning model performing two-stage evidence extraction and answer generation. With only 1.8% of the frames and 14.7% of the computation, it outperforms GPT-4o and all open-source LVLMs.

## Background & Motivation

**Background**: LVLMs have made significant progress in multimodal understanding, but still fall short in video reasoning tasks, primarily limited by the scarcity of high-quality, large-scale VideoQA datasets.

**Limitations of Prior Work**: (1) Manual annotation is costly and lacks fine-grained details; (2) Automatic construction approaches rely on frame-by-frame analysis, which is computationally expensive and introduces heavy redundancy; (3) Existing video CoT efforts perform reasoning mainly at the textual level, neglecting visual grounding (e.g., where is the object? which frame is it?); (4) Current LVLMs typically sample a large number of frames uniformly (e.g., 128 frames) during video processing, leading to immense computational overhead where most frames are redundant.

**Key Challenge**: Video information is highly redundant, yet complex reasoning demands precise localization of key frames and key objects. Uniform sampling wastes computational resources, while under-sampling can potentially lose critical information.

**Goal**: (1) Construct a large-scale VideoQA dataset containing multimodal CoT annotations (spatial + temporal grounding); (2) Design an efficient video reasoning framework that achieves high-quality reasoning using minimal frames.

**Key Insight**: First use a semantic-aware approach for redundancy removal and QA pair generation, then utilize GPT-4o to annotate multimodal CoT (including core frames, key objects, and evidence chains), and finally train a hybrid framework that sequentially operates as "frame selection before reasoning".

**Core Idea**: Employ a lightweight model to select 2–3 core frames, and then utilize a larger model to perform two-stage fine-grained video reasoning in an "evidence extraction $\rightarrow$ evidence-based reasoning" manner.

## Method

### Overall Architecture

The proposed method consists of two parts: data construction and the model framework. **Data Construction**: Videos are collected from 7 video datasets, followed by adaptive FPS sampling $\rightarrow$ InternVL2-8B frame description $\rightarrow$ BGE-M3 semantic redundancy removal $\rightarrow$ GPT-4o QA pair generation $\rightarrow$ Claude/Gemini quality filtering $\rightarrow$ GPT-4o multimodal CoT annotation (core frame selection + key object extraction + evidence generation) $\rightarrow$ GroundingDINO spatial labeling + BGE-M3 temporal retrieval. **Model Framework** (VideoQA-SC): Stage 1: A lightweight Frame Selector (InternVL2-1B + Qwen-0.5B) selects core frames; Stage 2: Two-stage SFT for the reasoning LVLM—first training on evidence extraction, and then training on evidence-based reasoning for answer generation.

### Key Designs

1. **Semantic-Aware Frame Redundancy Elimination**:

    - **Function**: Efficiently remove redundant frames from videos while retaining key information.
    - **Mechanism**: First apply adaptive FPS sampling (FPS 2–4 for dynamic scenes, FPS 1 for static scenes), then use InternVL2-8B to generate descriptions for each frame. Compute semantic similarity of adjacent frame descriptions via the BGE-M3 model, and eliminate redundant frames whose similarity exceeds a threshold $\tau$ (using a LIFO filtering strategy).
    - **Design Motivation**: Determining redundancy based on textual semantics rather than pixel-level differences better captures changes at the content level.

2. **Multimodal CoT Annotation Pipeline**:

    - **Function**: Generate a reasoning chain containing spatial and temporal grounding details for each QA pair.
    - **Mechanism**: A three-step process: (1) GPT-4o selects the core frame descriptions most relevant to the question from the generated frame descriptions; (2) Key objects are extracted from these core frame descriptions to serve as evidence; (3) These key objects are organized into a natural language reasoning chain. For spatial annotations, GroundingDINO is used to label bounding boxes, which are then verified for consistency via CLIP. For temporal annotations, BGE-M3 semantic retrieval is employed to match the original frames.
    - **Design Motivation**: Text-only CoT lacks visual grounding. Multimodal CoT uses spatial and temporal localization to make the reasoning process traceable and verifiable.

3. **VideoQA-SC Hybrid LVLM Collaborative Framework**:

    - **Function**: Achieve highly efficient and precise video reasoning.
    - **Mechanism**: **Frame Selector**: A 1B-parameter LVLM generates frame descriptions, and a 0.5B-parameter LLM selects core frames based on the question (averaging only 2.36 frames). **Two-Stage Reasoning LVLM**: Stage 1 trains on evidence extraction ("Please provide evidence helpful to answer the question"), and Stage 2 trains on answer generation ("Please answer the question based on the evidence"). This progressive two-stage training ensures step-by-step integration of multimodal information.
    - **Design Motivation**: Decoupling frame selection and reasoning allows a highly compact model to perform frame selection (saving compute) so that the larger model only processes the 2–3 most critical frames. Meanwhile, two-stage SFT prevents the model from bypassing evidence and directly guessing the answer.

### Loss & Training

Both stages of SFT utilize LoRA fine-tuning: learning rates are $2 \times 10^{-5}$ and $10^{-5}$ respectively, batch size = 16, $8 \times \text{A100}$, LoRA rank = 16, alpha = 32. The input resolution is $224 \times 224$, max tokens = 6144, 1 epoch of training with cosine decay.

## Key Experimental Results

### Main Results

| Model | Params | Frames | TFLOPs | Avg Acc (14 tasks) |
|------|-----|------|--------|-------------------|
| GPT-4o | - | FPS3 | - | 26.4% |
| Qwen-VL-Max | - | FPS3 | - | 26.0% |
| InternVL2-8B | 8B | FPS1 | 73.2 | 28.7% |
| Qwen2-VL-7B | 7B | FPS1 | 64.6 | 28.5% |
| LongVA-DPO-7B | 7B | 128 frames | 465.4 | 24.4% |
| **VideoEspresso** | **8.5B** | **2.36 frames** | **9.26** | **34.1%** |

Subjective evaluation (scores based on a 10-point scale):

| Model | Logicality | Factuality | Accuracy | Conciseness | Overall |
|------|--------|--------|--------|--------|------|
| GPT-4o | 73.2 | 63.1 | 61.7 | 70.0 | 66.1 |
| InternVL2 | 70.6 | 56.3 | 54.5 | 66.8 | 60.1 |
| **VideoEspresso** | **72.3** | **61.3** | **59.7** | **75.7** | **65.8** |

### Ablation Study

| Configuration | Accuracy | Delta |
|------|----------|------|
| Full model | 34.13% | - |
| GT-CoT (oracle evidence) | 72.95% | +38.82% |
| w/o Bbox (w/o spatial labels) | 33.14% | -0.99% |
| w/o CoT (w/o reasoning chain) | 31.32% | -2.81% |

Generalizability of the Core Frame Selector:

| Model | Uniform Sampling | + Selector | Delta |
|------|---------|-----------|------|
| GPT-4o (16 frames) | 26.9% | 29.5% (2.36 frames) | +2.6% |
| InternVL2 (16 frames) | 28.6% | 30.0% (2.36 frames) | +1.4% |

### Key Findings

- **Just 2.36 frames outperforming 128 frames**: Using only 1.8% of the frames and 2% of the FLOPs of LongVA-DPO, VideoEspresso surpasses it by 9.7% in accuracy, proving that "carefully selected sparse frames" is far superior to "brute-force dense frames".
- **Huge room for growth with GT-CoT**: Utilizing ground truth evidence can skyrocket performance from 34.1% to 73.0% (+38.8%), indicating that the evidence extraction capabilities of current models still have immense room for improvement.
- **Core Frame Selector as a plug-and-play module**: Directly plugging it into GPT-4o yields a 2.6% accuracy improvement while reducing the frame count by 85%.
- **Superiority over GPT-4o in conciseness**: The conciseness score in subjective evaluation reaches 75.7 vs. 70.0 for GPT-4o, indicating that selecting core frames effectively reduces redundant output.

## Highlights & Insights

- **The philosophy of "less is more"** works surprisingly well in video reasoning: 2.36 frames > 128 frames. Rather than uniform sampling, the core frame selector achieves ultimate information compression through semantic understanding.
- **The two-stage "evidence-first, reasoning-second" training strategy** is highly elegant: it prevents the model from taking shortcuts like "predicting the answer from the question directly" and forces it to explicitly extract visual evidence first.
- **Highly automated dataset construction pipeline**: The entire pipeline (InternVL2 frame description $\rightarrow$ BGE-M3 redundancy removal $\rightarrow$ GPT-4o QA generation $\rightarrow$ multi-LLM cross-verification $\rightarrow$ GroundingDINO spatial annotation) exhibits strong scalability.

## Limitations & Future Work

- The GT-CoT experiment reveals a huge room for improving the quality of automatically generated CoT annotations (34.1% vs. 72.9%).
- The test set contains only 1,382 items, which is relatively small and might invite statistical variance.
- The selector slightly degrades performance on certain models (e.g., -0.6% for LongVA), suggesting that the generalizability of the selector needs further improvement.
- The distribution among the 14 tasks is highly imbalanced (e.g., 87k for causal reasoning vs. 276 for cooking steps), which might affect the fairness of the evaluation.

## Related Work & Insights

- **vs. LongVA-DPO**: While LongVA processes videos with 128 frames in a brute-force manner, VideoEspresso dramatically outperforms it (+9.7%) using only 2.36 frames, consuming only 2% of the FLOPs.
- **vs. MVBench**: MVBench focuses on basic video understanding (e.g., "what object is this"), whereas VideoEspresso emphasizes complex reasoning ("why", "inferring"), which presents a fundamental difference at the dataset design level.
- **vs. Text-only CoT methods**: Text-only CoT lacks visual grounding, whereas VideoEspresso introduces spatial bounding boxes and temporal localization to make the reasoning chain traceable.

## Rating

- Novelty: ⭐⭐⭐⭐ The multimodal CoT annotation and the hybrid frame selection-reasoning framework are designed novelly, though individual sub-modules (frame descriptions, semantic retrieval, etc.) are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons across 14 tasks and 9 models, coupled with thorough ablation studies and cross-model generalization validation.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly described, accompanied by rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ The 200k high-quality video CoT dataset combined with an efficient reasoning framework acts as a significant catalyst for video understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)

</div>

<!-- RELATED:END -->
