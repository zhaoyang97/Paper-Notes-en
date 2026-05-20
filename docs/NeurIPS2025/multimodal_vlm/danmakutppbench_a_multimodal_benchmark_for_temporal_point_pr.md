---
title: >-
  [Paper Note] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding
description: >-
  [NeurIPS 2025][Multimodal VLM][temporal point process] This paper introduces DanmakuTPPBench, the first multimodal temporal point process benchmark. DanmakuTPP-Events provides 7…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "temporal point process"
  - "multimodal benchmark"
  - "Danmaku"
  - "LLM evaluation"
  - "multi-agent pipeline"
date: 2026-05-08
content_hash: da8703f2de7c1f23
---

# DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2505.18411](https://arxiv.org/abs/2505.18411)  
**Code**: [GitHub](https://github.com/FRENKIE-CHIANG/DanmakuTPPBench)  
**Area**: Multimodal Temporal Modeling / TPP / LLM Benchmark
**Keywords**: temporal point process, multimodal benchmark, Danmaku, LLM evaluation, multi-agent pipeline

## TL;DR

This paper introduces DanmakuTPPBench, the first multimodal temporal point process benchmark. DanmakuTPP-Events provides 7,250 sequences comprising 10.8 million Danmaku events with natural three-modal alignment (time–text–video). DanmakuTPP-QA automatically generates 10 categories of reasoning question–answer pairs via a multi-agent pipeline. The benchmark systematically reveals significant deficiencies of both classical TPP models and MLLMs in understanding multimodal event dynamics.

## Background & Motivation

**Background**: Temporal point processes (TPPs) are a classical framework for modeling continuous-time event sequences, with broad applications in social media forecasting, medical monitoring, and financial analysis. Recent advances in LLM/MLLM multimodal reasoning have made it promising to incorporate TPP modeling into multimodal language models.

**Limitations of Prior Work**: Existing TPP datasets (Retweet, StackOverflow, Taobao, etc.) contain only timestamps and event categories, entirely lacking textual semantics and visual context. RNCNIX and Amazon Review include text but no visual information. No existing TPP dataset simultaneously covers temporal, textual, and visual modalities, and no QA evaluation benchmark exists for TPP understanding.

**Key Challenge**: Strong multimodal capability (e.g., MLLMs performing well on VQA) does not imply strong multimodal temporal event modeling ability. No benchmark currently exists to evaluate whether models can jointly understand the interactions among temporal dynamics, textual semantics, and visual content.

**Key Insight**: The Danmaku system on Bilibili serves as an ideal multimodal TPP data source — each Danmaku comment naturally carries a precise timestamp (aligned to a video frame), textual content (users' real-time reactions), and the corresponding video frame, forming perfect three-modal alignment. Based on this, DanmakuTPP-Events is constructed for classical TPP modeling evaluation, and DanmakuTPP-QA is automatically built via a multi-agent pipeline for deeper temporal reasoning evaluation.

## Method

### Overall Architecture

DanmakuTPPBench comprises two complementary components: (1) DanmakuTPP-Events — a multimodal event dataset for classical TPP modeling (7,250 sequences, 10.8 million events, 14 video categories); and (2) DanmakuTPP-QA — a multi-task question-answering benchmark for deep temporal reasoning (2,605 videos, 10 task categories), automatically constructed by five specialized collaborative agents.

### Key Designs

1. **DanmakuTPP-Events Dataset Construction**:
    - Function: Collects Danmaku data from all videos of Bilibili's Top 100 creators in 2024, constructing the first three-modal TPP dataset.
    - Mechanism: Each Danmaku event is modeled as a quadruple $(t_i, e_i, m_i^{\text{text}}, m_i^{\text{image}})$ — timestamp, event type (9 categories: sarcasm, Danmaku meme, emotional expression, etc.), Danmaku text, and the corresponding video frame. The dataset contains 7,250 sequences with an average length of 1,494 (far exceeding existing TPP datasets' 27–197), covering 14 video categories (gaming 23%, education 18%, lifestyle 12%, animation 10%, etc.).
    - Design Motivation: Danmaku is one of the rare data sources that naturally possesses temporal, textual, and visual attributes simultaneously without requiring manual alignment. The data volume (10.8 million events) and sequence length (average 1,494) far surpass all existing TPP datasets, providing an ideal testbed for studying long-sequence multimodal event modeling.

2. **DanmakuTPP-QA Multi-Agent Construction Pipeline**:
    - Function: Automatically constructs 10 evaluation task categories (8 closed-ended + 2 open-ended), covering temporal prediction, sentiment analysis, event attribution, and more.
    - Mechanism: Five specialized agents collaborate with distinct roles — the **Task-Design Agent** (Deepseek-R1) designs 10 task types and their input/output formats from the data structure; the **Annotation Agent** (Qwen2.5 for text, Qwen2.5-VL for vision, RAM for object tagging) labels events; the **Quality-Control Agent** (Qwen3) coordinates and validates annotation consistency via majority voting and gap-filling; the **Visualization Agent** (Qwen2.5-Coder) automatically generates Python visualization scripts; and the **Task-Solve Agent** (Qwen3 + Qwen2.5-VL + Gemma-3 multi-model voting) generates reference answers.
    - Design Motivation: Manually constructing large-scale multimodal QA is prohibitively costly; the multi-agent pipeline achieves a balance between scale and quality. Cross-validation, majority voting, and manual spot-checking ensure data quality. Test set answers undergo rigorous human verification.

3. **10-Category Evaluation Task System**:
    - Function: Designs a graduated set of evaluation tasks ranging from simple prediction to complex causal reasoning.
    - Mechanism: **Closed-ended tasks** (8 categories) — Danmaku burst count (ACC), next Danmaku/burst time prediction (RMSE), average sentiment polarity assessment (RMSE), sentiment polarity prediction (RMSE), event type inference (ACC), burst trigger type prediction (ACC). **Open-ended tasks** (2 categories) — global sentiment dynamics analysis (LLM-Eval score 0–1), burst causal attribution analysis (LLM-Eval). Data split: 2,005 training, 300 validation, 300 test.
    - Design Motivation: Task difficulty escalates progressively from numerical prediction to logical reasoning to causal attribution, comprehensively evaluating models' temporal awareness, textual understanding, and cross-modal reasoning capabilities.

### Loss & Training

As a benchmark paper, no new training methods are proposed. Classical TPP models are trained using default settings of the EasyTPP framework. MLLM evaluation employs zero-shot inference with 3 randomly sampled video frames as visual input. Fine-tuning experiments apply LoRA to Qwen2.5-VL-3B (single GPU RTX 4090, 3 epochs, lr=1e-4).

## Key Experimental Results

### Main Results

| Model | T-1 (ACC↑) | T-2 (RMSE↓) | T-4 (RMSE↓) | T-7 (ACC↑) | T-8 (ACC↑) |
|------|-----------|-----------|-----------|-----------|-----------|
| Qwen2.5-7B | 0.33 | 27.64 | 0.65 | 10.67 | 32.67 |
| Qwen2.5-72B | 0.67 | 1.28 | 0.30 | 16.00 | 43.83 |
| Qwen3-30B-A3B | 0.67 | 1.33 | **0.20** | **23.00** | 43.67 |
| DeepSeek-V3 | **25.00** | 1.30 | 0.34 | 13.67 | 34.50 |
| Qwen2.5-VL-72B | 0.33 | **1.14** | 0.28 | 15.98 | **47.17** |
| Gemma3-27B | 0.33 | 1.33 | 0.28 | 15.67 | 36.17 |
| Fine-tuned VL-3B | **27.0** | 1.35 | **0.05** | 15.33 | 43.00 |

### Ablation Study

| Configuration | T-4 RMSE↓ | T-5 RMSE↓ | T-6 RMSE↓ | Note |
|------|-----------|-----------|-----------|------|
| Best pretrained model | 0.20 | 0.26 | 0.20 | Qwen3-30B / DeepSeek-V3 / Gemma3 |
| Fine-tuned VL-3B | **0.05** | **0.16** | **0.08** | Error reduced 4–6×; small fine-tuned model greatly surpasses large zero-shot models |
| Fine-tuned VL-3B @ T-3 | 220.43 | — | — | Severe overfitting on temporal prediction task |

Classical TPP models: NHP achieves the best log-likelihood (0.799) and RMSE (0.932), while attention-based models score lower (THP 0.619, AttNHP 0.550).

### Key Findings

- MLLMs do not consistently outperform text-only LLMs on multimodal TPP understanding: Qwen2.5-VL-72B achieves the best performance on only a subset of tasks, while Llama-3.3-70B performs better on temporal prediction (T-2 RMSE=1.11).
- Model scale has a significant effect: across the Qwen2.5 series, T-2 RMSE decreases from 27.64 (7B) to 1.28 (72B).
- Fine-tuning a small model on sentiment tasks greatly surpasses large-model zero-shot performance (T-4 RMSE: 0.05 vs. 0.20), but may overfit on temporal prediction (severe degradation on T-3).
- On open-ended tasks, Qwen2.5-VL-72B and Qwen3-235B lead; Task-9 (sentiment dynamics analysis) scores 0.48 and Task-10 (causal attribution) scores 0.52 — far from saturation.
- Danmaku burst causal attribution (Task-10) is the most challenging, requiring simultaneous understanding of video content changes, user sentiment evolution, and event trigger mechanisms.

## Highlights & Insights

- The data source selection is elegant: Danmaku is one of the rare naturally three-modal-aligned data sources — each comment inherently carries a precise timestamp, text, and corresponding video frame, requiring no manual alignment. This data design insight is exemplary.
- The multi-agent pipeline design is mature: five specialized agents with distinct responsibilities, combined with cross-validation and majority voting, is generalizable to automated benchmark construction in other domains. In particular, the paradigm of using a reasoning model (R1) for task design and multi-model voting for answer generation in the Task-Solve Agent is highly instructive.
- The paper reveals an important gap: "strong multimodal understanding" does not equate to "strong multimodal temporal event modeling" — current MLLMs, while powerful on VQA and image understanding, face fundamental limitations in long-sequence temporal dynamics analysis.

## Limitations & Future Work

- The data domain is concentrated in Bilibili's Danmaku ecosystem, introducing platform-specific cultural and behavioral biases; generalization to other languages and cultural contexts requires further validation.
- Automated QA generation is bounded by LLM capability, and task definitions and reference answers may contain noise.
- The paper only contributes a benchmark without proposing a new strong modeling method — no multimodal TPP model designed for this data is presented.
- Event types are limited to 9 categories; behavioral differences across video categories may warrant finer-grained type taxonomies.
- MLLMs use only 3 sampled frames as visual input; denser frame sampling or dedicated video encoders may improve performance.

## Related Work & Insights

- **vs. Classical TPP datasets** (Retweet / Taobao / StackOverflow): All are unimodal. DanmakuTPP-Events is the first complete three-modal TPP dataset, with an average sequence length of 1,494 far exceeding theirs (27–197).
- **vs. TSQA** (Kong et al.): TSQA targets time-series QA but does not address point processes or multimodality; DanmakuTPP-QA focuses specifically on TPP understanding with visual information.
- **vs. Amazon Review**: Contains text but no visual information and has an average sequence length of only 27; DanmakuTPP-Events comprehensively surpasses it in both modality completeness and sequence scale.

## Rating

- Novelty: ⭐⭐⭐⭐ First multimodal TPP benchmark; unique and natural data source selection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers classical TPP models and diverse LLMs/MLLMs with rich task design.
- Writing Quality: ⭐⭐⭐⭐ Data construction pipeline is clearly and completely described; experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ Fills the gap in multimodal TPP evaluation and bridges the TPP and LLM communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](../../CVPR2026/multimodal_vlm/videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)

</div>

<!-- RELATED:END -->
