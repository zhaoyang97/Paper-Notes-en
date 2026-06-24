---
title: >-
  [Paper Note] Giraffe: Design Choices for Extending the Context Length of Visual Language Models
description: >-
  [ACL2025][LLM Efficiency][VLM] This work systematically explores the design space for extending the context window of existing Visual Language Models (VLMs) to 128K. It proposes best practices across three dimensions—data recipe, positional encoding extension, and context utilization—and introduces two techniques: M-RoPE++ and hybrid-resolution training. The resulting Giraffe model achieves state-of-the-art (SOTA) performance among long-context VLMs.
tags:
  - "ACL2025"
  - "LLM Efficiency"
  - "VLM"
  - "Long Context"
  - "M-RoPE++"
  - "Hybrid-Resolution"
  - "Context Extension"
  - "Video Understanding"
date: 2026-05-08
content_hash: 35c31b4ee1f6a2a7
---

# Giraffe: Design Choices for Extending the Context Length of Visual Language Models

**Conference**: ACL2025  
**arXiv**: [2412.12735](https://arxiv.org/abs/2412.12735)  
**Code**: [GitHub](https://github.com/kiaia/GIRAFFE)  
**Area**: LLM Efficiency / Long-Context Visual Language Models  
**Keywords**: VLM, Long Context, M-RoPE++, Hybrid-Resolution, Context Extension, Video Understanding  

## TL;DR

This work systematically explores the design space for extending the context window of existing Visual Language Models (VLMs) to 128K. It proposes best practices across three dimensions—data recipe, positional encoding extension, and context utilization—and introduces two techniques: M-RoPE++ and hybrid-resolution training. The resulting Giraffe model achieves state-of-the-art (SOTA) performance among long-context VLMs.

## Background & Motivation

### Problem Definition
Visual Language Models (VLMs) have demonstrated exceptional capabilities in handling multimodal inputs. However, advanced scenarios such as multi-image and high-resolution long video inputs impose higher demands on the model's long-range modeling capabilities. For example, a context length of 2K can only process a few video frames, which severely limits the performance ceiling on long video understanding tasks.

### Key Challenge
Existing open-source VLMs lack a systematic exploration of context window extension, while proprietary commercial models do not disclose technical details. Prior works (e.g., LongVA, LongVILA, LongLLaVA) adopt various strategies but lack a systematic comparison of the effectiveness of different design choices.

### Three Core Research Questions
1. How to effectively organize and curate training data?
2. How to efficiently train VLMs with longer contexts?
3. How to better utilize the extended context window?

## Method

### Overall Architecture

Focusing on three research questions, the paper conducts a systematic experimental study across three dimensions: data curation, context extension training, and context utilization. Ultimately, the Giraffe model is constructed based on the Qwen-VL series.

### Key Design 1: ETVLM Data Recipe

The data sources comprise four categories:

| Category | Data Source | Proportion |
|------|--------|------|
| Long-context text-only instructions | LongAlign, LongAlpaca | 20% |
| Short visual instruction data | LLaVA-Instruct, M3IT | 25% |
| Interleaved image-text data | MMDU, Mantis, ArXivQA | 25% |
| Video QA + Summarization | ShareGPT4O, MLVU, LLaVA-Video | 30% |

**Findings from data ratio experiments:**
- Short multimodal instruction data is crucial both for extending long-context capabilities and for maintaining short-context performance.
- A balanced data ratio contributes to balanced performance across downstream tasks.
- Increasing the proportion of any single data type primarily boosts performance only on its corresponding tasks.

**Findings from data length experiments:**
- When the proportion of long data (>8K tokens) reaches 60%, the performance of long-context tasks tends to saturate.
- When the proportion of long data exceeds 60%, the performance on short-context tasks decreases significantly.
- Therefore, a 60% proportion of long data is selected as the optimal balance point.

### Key Design 2: M-RoPE++ Positional Encoding Extension

**Problem Analysis**: M-RoPE factorizes the rotary positional encoding into three dimensions: temporal, height, and width, with an allocation ratio of 2:3:3.

**Limitations of Prior Work**:
- Position Interpolation (PI) and NTK methods compress high-frequency signals indiscriminately, which may confuse the model's perception of temporal order in adjacent frames.
- Experiments reveal that similar to LLMs, VLMs exhibit an "effective length shortage" (falls short) phenomenon—the effective length after extension is much shorter than the training length.

**Core Idea of M-RoPE++**:
- **Temporal dimension (low dimension $\rightarrow$ high frequency)**: Keep extrapolation, since temporal information resides in the high-frequency components and has been fully covered during the pre-training phase.
- **Spatial dimensions (high dimension $\rightarrow$ low frequency)**: Apply interpolation, as height and width occupy higher-dimensional spaces and may not be fully covered in the rotary domain during pre-training.

Piecewise function definition:
$$\theta_d' = \begin{cases} \theta_d & \text{if } 0 < d \leq 2x \text{ (temporal)} \\ (\frac{1}{s} + (1-\frac{1}{s})\cdot\frac{d-r_{5x}}{r_{2x}-r_{5x}})\cdot\theta_d & \text{if } 2x < d \leq 5x \text{ (height)} \\ \frac{\theta_d}{s} & \text{if } 5x < d \leq 8x \text{ (width)} \end{cases}$$

where $s = L'/L_V$ is the extension scale.

### Key Design 3: Training Strategy Selection

The paper compares three training strategies:

| Strategy | MMBench | BLINK | VideoMME |
|------|---------|-------|----------|
| One-stage multimodal instruction tuning | **82.8** | **54.6** | **58.5** |
| Two-stage: Text extension + MM instruction | 79.8 | 52.9 | 58.1 |
| Two-stage: MM alignment + MM instruction | 80.5 | 51.2 | 57.8 |

**Conclusion**: Directly applying long-context multimodal instruction tuning to the VLM is sufficient; multi-stage training is unnecessary. The reasons are:
- Long-context multimodal data itself already covers a diverse distribution of lengths.
- Qwen2-VL has already undergone instruction tuning, and performing alignment training again would disrupt the learned distribution.

### Key Design 4: Hybrid-Resolution Training (Hybrid-Resolution)

Inspired by the SlowFast network:
- Video frames are divided into $N$ groups, with each group containing $L$ frames.
- The first frame of each group uses high resolution ($m$ tokens).
- The subsequent $L-1$ frames use low resolution ($m/s$ tokens).

The token budget is reduced from $L \times N \times m$ to $(1+\frac{L-1}{s}) \times N \times m$.

Experiments demonstrate that under an equivalent token budget, hybrid-resolution inference can improve the resolution of keyframes and enhance performance on downstream tasks.

## Key Experimental Results

### Evaluation Setup
- **Short-Context**: MME, MMBench (single-image understanding)
- **Multi-Image**: Mantis-Eval, QBench, BLINK
- **Long-Video**: VideoMME, LongVideoBench
- **Effective Length**: Visual Haystack (needle-in-a-haystack test for vision)

### Comparison of Positional Encoding Methods

| Method | VideoMME Long (512 frames) | Visual Haystack (100 images) |
|------|----------------------|------------------------|
| Direct Extrapolation | 55.4 | 51.3 |
| PI | 56.0 | 57.8 |
| NTK | 56.2 | 56.7 |
| **M-RoPE++** | **58.5** | **61.3** |

M-RoPE++ consistently outperforms other methods across all settings.

### Video Understanding Tasks (Main Results)

| Model | Frame Count | VideoMME Overall | LongVideoBench Avg |
|------|------|-----------------|-------------------|
| GPT-4V | 10 | 59.9 | 59.1 |
| GPT-4o | 384 | 71.9 | 66.7 |
| Qwen2-VL-7B | 256 | 63.2 | 61.5 |
| **Giraffe** | 768 | 64.8 | 63.3 |
| **Giraffe + Hybrid-res** | 1024 | **65.9** | **64.3** |

Giraffe achieves SOTA performance among open-source long-context VLMs and outperforms GPT-4V in several categories.

### Multi-Image and Single-Image Evaluation

- **Multi-Image**: Giraffe-QwenVL shows a significant improvement compared to Qwen-VL (Mantis-Eval: 39.2 $\rightarrow$ 48.3, QBench: 45.9 $\rightarrow$ 57.4).
- **Single-Image**: Giraffe maintains short-context performance comparable to Qwen2-VL (MMBench: 82.1 vs 82.8).

### Frame Count - Resolution Trade-off

| Frame Count | Tokens per Frame | VideoMME Medium | VideoMME Long |
|------|----------|----------------|---------------|
| 128 | 960 | 62.5 | 55.6 |
| 512 | 240 | 64.6 | 58.2 |
| 768 | 160 | 64.8 | 58.5 |
| 1024 | 120 | 64.7 | 58.5 |

- Medium-length tasks tend to saturate after 512 frames.
- Long-video tasks benefit from larger frame counts, but performance drops when single-frame resolution becomes too low.

## Highlights & Insights

1. **Systematic Design Space Exploration**: Rather than simply proposing a method, the paper systematically addresses three key design questions through extensive controlled experiments, yielding highly practical and actionable insights.
2. **Differentiated Treatment of Dimensions in M-RoPE++**: Based on a deep understanding of the intrinsic differences between temporal (high-frequency) and spatial (low-frequency) dimensions in RoPE, the paper proposes tailored extrapolation/interpolation strategies.
3. **One-Stage Training is Optimal**: Shattering the intuition of multi-stage training, the authors find that direct instruction tuning yields the best performance, significantly simplifying the training pipeline.
4. **Practicality of Hybrid Resolution**: The SlowFast concept is cleverly adapted to VLMs, effectively improving performance under a fixed token budget.
5. **Effective Length Analysis**: The paper reveals an "effective length shortage" phenomenon in VLMs similar to LLMs, establishing an important benchmark for future research.

## Limitations & Future Work

1. Validated only on the Qwen-VL series, meaning the transferability of the conclusions requires further verification.
2. Extending context to 128K requires substantial GPU resources (8x80G H100), making the training costly.
3. The effective length is still far shorter than the training length (approx. 40K vs 128K), indicating that there remains room for improvement in positional encoding extension.
4. The selection strategy for high-resolution frames in hybrid-resolution training is relatively simple (fixed to the first frame of each group), and adaptive selection remains unexplored.
5. The evaluation primarily focuses on English datasets, leaving multilingual scenarios unaddressed.

## Related Work & Insights

- **Long-Context LLMs**: Positional encoding extension and efficient inference methods such as NTK, PI, YaRN, StreamingLLM, and InfLLM.
- **VLMs**: Models like LLaVA, MiniGPT-4, Qwen-VL, and VideoLLaVA.
- **Long-Context VLMs**: LongVA (extending LLM first then transferring to VLM), LongVILA (multi-stage + sequence parallelism), LongLLaVA (Mamba + Transformer).

## Rating ⭐⭐⭐⭐

A highly solid and systematic research work, featuring thorough experiments and highly practical guiding value. The design of M-RoPE++ offers theoretical insights, and hybrid-resolution training is highly practical. The main shortcomings lie in its validation being limited to the Qwen-VL series, and the fact that the effective length still has substantial room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Collapse to Control: Understanding and Extending Context Length in Emerging Hybrid Models via Universal Position Interpolation](../../ICLR2026/llm_efficiency/from_collapse_to_control_understanding_and_extending_context_length_in_emerging_.md)
- [\[ICLR 2026\] UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models](../../ICLR2026/llm_efficiency/ultrallada_scaling_the_context_length_to_128k_for_diffusion_large_language_model.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)

</div>

<!-- RELATED:END -->
