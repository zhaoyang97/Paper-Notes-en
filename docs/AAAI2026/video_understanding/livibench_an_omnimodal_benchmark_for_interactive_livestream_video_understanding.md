---
title: >-
  [Paper Note] LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding
description: >-
  [AAAI 2026][Video Understanding][livestream video understanding] This paper presents LiViBench, the first omnimodal benchmark for interactive livestream video understanding (3,168 videos, 3,175 MCQs, 24 tasks), introduces a multi-agent seed-guided semi-automatic annotation pipeline, and develops LiVi-LLM-7B — a specialized model featuring a Video-to-Comment Retrieval (VCR) module and two-stage instruction tuning — which surpasses 72B open-source models at the 7B scale.
tags:
  - AAAI 2026
  - Video Understanding
  - livestream video understanding
  - multimodal benchmark
  - real-time comments
  - instruction tuning
  - omnimodal model
date: 2026-05-08
content_hash: c8fa0ce042c1098e
---

# LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding

**Conference**: AAAI 2026  
**arXiv**: [2601.15016](https://arxiv.org/abs/2601.15016)  
**Code**: [github](https://github.com/Wang-Xiaodong1899/LiViBench)  
**Area**: Video Understanding  
**Keywords**: livestream video understanding, multimodal benchmark, real-time comments, instruction tuning, omnimodal model

## TL;DR

This paper presents LiViBench, the first omnimodal benchmark for interactive livestream video understanding (3,168 videos, 3,175 MCQs, 24 tasks), introduces a multi-agent seed-guided semi-automatic annotation pipeline, and develops LiVi-LLM-7B — a specialized model featuring a Video-to-Comment Retrieval (VCR) module and two-stage instruction tuning — which surpasses 72B open-source models at the 7B scale.

## Background & Motivation

### State of the Field

Multimodal large language models (MLLMs) have achieved remarkable progress in general video understanding, with benchmarks such as Video-MME, LongVideoBench, and MLVU continuously pushing model capabilities forward. However, existing video benchmarks **focus primarily on non-interactive content** — films, recordings, short clips — which lack real-time interaction between audiences and creators.

### Limitations of Prior Work

**Interactive video is neglected**: Livestreaming is the fastest-growing segment of online video consumption (e.g., Instagram Live, TikTok Live), yet no benchmark specifically evaluates model comprehension of livestream video.

**Unique livestream characteristics are not covered**: Livestreams involve interactive behaviors such as gifting, real-time dialogue, bullet-screen comments (danmaku), and multi-party co-streaming, which are fundamentally distinct from conventional video.

**Annotation pipelines lack transparency**: Existing benchmarks either rely entirely on manual annotation (high cost) or on a single model for automatic annotation (introducing bias), without well-designed semi-automatic workflows.

**Real-time comment processing is challenging**: Livestreams are accompanied by massive volumes of danmaku (the dataset contains approximately 1.45 million comments), posing significant challenges to model context length and information extraction capability.

### Root Cause

The defining characteristic of livestream video is **real-time interactivity** (danmaku, gifting, co-streaming, etc.), yet existing MLLMs and evaluation frameworks are entirely incapable of capturing and assessing this interactive understanding. Even the top closed-source model GPT-4o performs poorly in livestream scenarios.

### Starting Point

The paper adopts a three-pronged approach: (1) constructing the first omnimodal livestream video benchmark covering audio, speech, and danmaku; (2) designing a multi-agent + seed-question + human-in-the-loop annotation pipeline; and (3) developing the specialized model LiVi-LLM-7B with a danmaku retrieval module.

## Method

### Overall Architecture

The work comprises three main components: a benchmark construction pipeline, an instruction tuning strategy, and a comment retrieval module.

### Key Designs

#### 1. Multi-Agent Seed-Guided Annotation Pipeline

**Function**: Efficiently construct high-quality video question-answering evaluation data.

**Mechanism**:

**(a) Multi-agent video description**: A multi-agent system composed of four large-parameter models — LLaVA-Video, Qwen2.5-VL, Intern3VL, and Seed1.5-VL — is employed, with each model responsible for specific description subtasks to generate comprehensive video descriptions, thereby mitigating the bias of any single model.

**(b) Seed question bank**:
- Closed-source models automatically generate candidate seed questions
- Human reviewers remove unreasonable or overly simplistic questions
- A curated seed question bank covering 24 tasks is established

**(c) Question generation + human-in-the-loop**:
- Using the seed question bank and detailed video descriptions, models generate candidate questions per video
- Human annotators filter and revise ambiguous, overly simple, or irrelevant questions
- Both models and humans provide answers independently
- A final round of comprehensive quality control is conducted by human reviewers

**Design Motivation**: To balance annotation efficiency and quality — the multi-agent setup reduces single-model bias, seed questions ensure controllability, and multi-stage human-in-the-loop review guarantees annotation quality.

#### 2. Video-to-Comment Retrieval (VCR) Module

**Function**: Retrieve the most relevant danmaku comments from the massive comment stream with respect to video content.

**Mechanism**:
- Video frames are uniformly sampled and frame embeddings are extracted using Chinese-CLIP
- All danmaku comments are encoded into text embeddings using a text encoder
- Frame-text similarity scores are computed to obtain the top-$k$ most relevant comments per frame
- All retrieved comments are **arranged in chronological order** and provided as textual context to the model alongside the question

**Design Motivation**: The volume of livestream danmaku is enormous (hundreds to thousands of comments per video on average), making direct input infeasible due to context overflow. VCR leverages vision-text similarity to retain only the most relevant comments, alleviating information overload.

#### 3. Two-Stage Instruction Tuning Strategy

**Function**: Enhance the ability of open-source MLLMs to understand interactive livestream video.

**Mechanism**:

**Model Architecture**: Initialized from Qwen2.5-Omni weights; video tokens are extracted using the Qwen2.5-VL visual encoder, audio tokens using Qwen2-Audio, and both are fused through a Transformer Decoder before being fed into the LLM.

**Stage 1: Domain Alignment**
- 37,953 machine-annotated synthetic samples are used
- General video data is incorporated to preserve generalization ability
- Objective: align the model to the interactive video domain

**Stage 2: Fine-Grained Tuning**
- 11,180 human-refined samples are used
- Further improves model accuracy and robustness

**Design Motivation**: The two-stage strategy balances data volume and quality — Stage 1 leverages large-scale synthetic data for rapid domain knowledge acquisition, while Stage 2 uses a smaller set of high-quality data to refine performance.

### Benchmark Task Design

The 24 tasks are organized into 5 major categories:
- **Coarse-Grained Perception** (4 tasks): Basic understanding of scenes, actions, and appearance
- **Fine-Grained Perception** (6 tasks): Multi-person interaction, behavioral details, temporal changes
- **Knowledge-Based QA** (3 tasks): Reasoning that requires external knowledge
- **General Reasoning** (4 tasks): Causal reasoning, sentiment analysis, etc.
- **Livestream-Specific Tasks** (7 tasks): Danmaku comprehension, gift recognition, co-streaming interaction, etc.

## Key Experimental Results

### Main Results

| Model | Params | Overall | Coarse | Fine | Know | Reason | Livestream |
|------|--------|---------|--------|------|------|--------|------------|
| GPT-4o | — | 56.3 | 67.0 | 66.5 | 57.6 | 55.2 | 47.4 |
| Gemini 2.5 Pro | — | 56.1 | 65.0 | 68.4 | 58.1 | 51.3 | 48.2 |
| Seed1.5-VL | — | **66.2** | 70.9 | 71.4 | **68.8** | **70.7** | 59.1 |
| Qwen2.5-VL-72B | 72B | 62.3 | 73.4 | 72.4 | 61.9 | 64.6 | 52.0 |
| InternVL3-78B | 78B | 64.4 | 72.0 | 69.8 | 65.8 | 69.3 | 56.3 |
| InternVL3-38B | 38B | 64.1 | 70.9 | 72.6 | 66.6 | 68.3 | 54.5 |
| **LiVi-LLM-7B** | **7B** | **64.4** | 70.1 | 68.7 | 62.8 | 63.6 | **60.9** |
| Qwen2.5-Omni-7B | 7B | 60.3 | 68.1 | 68.5 | 59.4 | 60.7 | 53.1 |

Key findings:
- LiVi-LLM-7B achieves 64.4% at the 7B scale, **matching InternVL3-78B**
- It achieves the **best performance on Livestream-specific tasks at 60.9%**, surpassing all closed-source and large-scale open-source models
- GPT-4o and Gemini 2.5 Pro show limited performance on livestream tasks (47.4% and 48.2%, respectively)

### Ablation Study: Modality Impact Analysis

| Model | V (video only) | +A (+audio) | +S (+speech/danmaku) | Livestream (V) | Livestream (+S) |
|------|---------|---------|------------|--------------|----------------|
| LLaVA-Video-7B | 52.6 | NA | 55.4↑ | 43.5 | 48.4↑ |
| MiniCPM-o-26 | 56.0 | 54.7 | 57.9↑ | 46.5 | 51.2↑ |
| Qwen2.5-Omni-7B | 57.8 | 60.3↑ | 60.2↑ | — | 53.1 |
| **LiVi-LLM-7B** | — | — | **64.4** | — | **60.9** |

### Performance on General Video Benchmarks

| Model | Video-MME | LongVB | MLVU | VideoEval-Pro |
|------|-----------|--------|------|---------------|
| InternVL3-8B | 71.2 | 60.0 | 73.8 | 31.2 |
| Qwen2.5-VL-7B | 72.3 | 61.4 | 72.3 | 31.5 |
| **LiVi-LLM-7B** | **73.1** | 60.8 | 73.5 | **33.2** |

LiVi-LLM-7B achieves the best results on Video-MME and VideoEval-Pro, demonstrating that domain-specific fine-tuning does not sacrifice generalization ability.

### Key Findings

1. **Livestream-specific tasks represent the largest weakness across all models**: Even Seed1.5-VL (66.2% Overall) achieves only 59.1% on Livestream tasks.
2. **Speech and danmaku modalities consistently improve all categories**: Adding speech and danmaku yields an overall gain of +2–5 points.
3. **A 7B model with domain fine-tuning can surpass a 72B general-purpose model**: This demonstrates that domain knowledge outweighs model scale.
4. **The VCR module effectively addresses danmaku overload**: Visual-text retrieval efficiently filters relevant comments.

## Highlights & Insights

1. **Fills a critical gap in interactive video evaluation**: For the first time, livestream video understanding capabilities are systematically defined and assessed.
2. **Reusable annotation pipeline**: The multi-agent + seed-question + human-in-the-loop framework is transferable to other video domains.
3. **VCR module is simple yet effective**: Cross-modal retrieval via CLIP resolves danmaku information overload with minimal additional computational cost.
4. **Careful data curation at scale**: 1.45 million danmaku comments, 3,168 videos, and 24 tasks provide comprehensive coverage of the livestream domain.
5. **Practical two-stage tuning strategy**: Synthetic data for domain alignment followed by human-refined data for fine-tuning strikes an effective balance between cost and performance.

## Limitations & Future Work

1. **Evaluation is predominantly in Chinese**: Both danmaku and ASR transcripts are in Chinese; cross-lingual generalization remains to be verified.
2. **Video duration is relatively short**: Most videos are 1–5 minutes, whereas real livestreams typically run for hours.
3. **Only multiple-choice format is supported**: Open-ended generation capabilities cannot be evaluated.
4. **VCR module relies on Chinese-CLIP**: Retrieval quality is bounded by CLIP's cross-modal alignment capability.
5. **Domain coverage skews toward entertainment**: Chat, singing, and dancing content predominates; e-commerce livestreams, educational broadcasts, and similar scenarios are underrepresented.

## Related Work & Insights

- **Multi-agent annotation approach**: Using multiple models for cross-validation reduces bias and is more reliable than annotation by a single GPT-4o instance.
- **Seed question bank design**: Controlled question generation yields higher quality than fully unconstrained generation, offering a valuable reference for future benchmark construction.
- **Danmaku as a distinct modality**: Real-time comments encode audience-perspective information unavailable in other video types, opening a new research direction.
- **Domain fine-tuning vs. scale expansion**: The result showing a 7B model surpassing a 72B model through domain knowledge provides important implications for resource-constrained deployment scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first omnimodal benchmark for livestream video; both the problem formulation and task design are entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluates 24 models including closed-source and open-source systems, with coverage across multiple general benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rich figures and tables, though certain details could be presented more concisely.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses an important gap; both the benchmark and the model are highly valuable contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)

</div>

<!-- RELATED:END -->
