---
title: >-
  [Paper Note] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval
description: >-
  [AAAI 2026][LLM Evaluation][zero-shot video moment retrieval] This paper proposes GranAlign, a training-free granularity-aware alignment framework that addresses the core challenge of semantic granularity mismatch in zer…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "zero-shot video moment retrieval"
  - "semantic granularity alignment"
  - "query rewriting"
  - "LLM"
  - "vision-language models"
date: 2026-05-08
content_hash: 448b84fe9a516fd4
---

# GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval

**Conference**: AAAI 2026
**arXiv**: [2601.00584](https://arxiv.org/abs/2601.00584)  
**Code**: N/A  
**Area**: LLM Evaluation
**Keywords**: zero-shot video moment retrieval, semantic granularity alignment, query rewriting, LLM, vision-language models

## TL;DR

This paper proposes GranAlign, a training-free granularity-aware alignment framework that addresses the core challenge of semantic granularity mismatch in zero-shot video moment retrieval (ZVMR). By rewriting queries into simplified and detailed variants and matching them against query-agnostic and query-aware video descriptions respectively, GranAlign achieves a 3.23% improvement in mAP@avg on QVHighlights.

## Background & Motivation

Video Moment Retrieval (VMR) aims to localize temporal segments in untrimmed videos given natural language queries. Traditional supervised approaches rely on expensive annotated data, making zero-shot VMR (ZVMR)—which leverages pretrained VLMs and LLMs—an increasingly valuable paradigm.

However, ZVMR faces a fundamental challenge: **Granularity Mismatch**. Users may describe the same event at varying levels of abstraction, e.g., "a cute dog" vs. "a golden retriever puppy wandering around." Coarse-grained queries yield high recall but low precision (broad coverage but imprecise localization), while fine-grained queries yield high precision but low recall (semantically specific but fragile to minor discrepancies). This creates an inevitable trade-off.

Through quantitative analysis, the authors categorize queries into types (simple/detailed/erroneous/other) and demonstrate significant performance variance across categories in existing methods—providing empirical evidence of the practical impact of granularity mismatch. Prior methods either perform query rewriting at a single granularity level ("one-size-fits-all") or rely solely on query-agnostic video descriptions that lack semantic alignment with the query. This single-channel inference paradigm is identified as the core bottleneck limiting robust retrieval.

The paper's **Core Idea** is to abandon single-channel design and establish dual coarse-to-fine channels on both the query side and the video side, aligning them by granularity level—simplified queries matched with generic descriptions (high recall), and detailed queries matched with query-aware descriptions (high precision)—combining the strengths of both.

## Method

### Overall Architecture

GranAlign is a fully training-free three-stage framework: (1) **granularity-aware alignment**—rewriting queries into simplified and detailed variants while generating both query-agnostic and query-aware frame-level descriptions; (2) **moment proposal generation**—generating candidate temporal segments based on frame-level semantic similarity scores; (3) **post-processing**—applying NMS to select final predictions.

### Key Designs

1. **Granularity-based Query Rewriting**

    - **Function**: Rewrites the original query into two semantically complementary versions via LLaMA-3.
    - **Mechanism**: The simplified query $Q_s$ replaces rare words with common ones, retains core entities and actions, and removes incidental details—providing high generalizability for retrieving broadly relevant candidates. The detailed query $Q_d$ preserves fine-grained expressions, temporal context, and specific lexical choices—enabling more precise alignment and localization.
    - **Design Motivation**: Single-granularity rewriting cannot simultaneously achieve high recall and high precision. Multiple manually designed prompt pairs are used to generate both variants; experiments show the framework is robust to the specific choice of prompt pairs.

2. **Query-Aware Captioning**

    - **Function**: Generates two types of descriptions for video frames—generic and query-aware.
    - **Mechanism**: Query-agnostic descriptions $C_{agn} \in \mathbb{R}^{L_v \times l}$ are first generated for all frames as a baseline. The top-K% frames (totaling $L_k$ frames) with the highest query similarity are then selected, and query-aware descriptions $C_{awr} \in \mathbb{R}^{L_k \times l}$ are generated for these frames only using Qwen2.5-VL, guided by entities and actions extracted from the query.
    - **Design Motivation**: Generating query-aware descriptions for all frames is computationally prohibitive. This hybrid strategy applies semantically precise descriptions to critical regions while maintaining global computational efficiency. Since query-aware descriptions may suffer from hallucinations or over-imitation of the query's linguistic structure, error-tolerant mechanisms are incorporated at the scoring stage.

3. **Granular Moment Scoring**

    - **Function**: Fuses similarity scores from two query–description pairs.
    - **Mechanism**: For each frame $f$, a composite similarity score is computed as: $S_f = \frac{1}{2m}\sum_{i=1}^{m}[g(q_s^{(i)}, C_{agn,f}) + g(q_d^{(i)}, C_{awr,f})]$, where $m$ is the number of rewriting pairs and $g(\cdot, \cdot)$ denotes normalized cosine similarity.
    - **Design Motivation**: The simplified–agnostic pair $(Q_s, C_{agn})$ provides broad coverage and high recall, while the detailed–aware pair $(Q_d, C_{awr})$ delivers precise alignment but is susceptible to hallucinations. Fusing the complementary scores from both pairs eliminates biases and false positives introduced by either pair alone.

4. **Moment Proposal Generation and Post-processing**

    - **Function**: Generates and filters candidate temporal segments from frame-level scores.
    - **Mechanism**: Adjacent high-scoring frames are merged into a single proposal if their gap does not exceed threshold $\tau$; proposals with average similarity in the bottom $n$% are discarded. Each candidate segment is scored as $\text{Score}(p) = (1-\lambda)\mu_p + \lambda\rho_p$, where $\mu_p$ is the average semantic similarity and $\rho_p$ is a normalized length regularization term, with $\lambda = 0.3$. NMS is applied to remove redundant proposals.
    - **Design Motivation**: Length regularization prevents both overly long low-quality proposals and overly short fragmented ones.

### Loss & Training

GranAlign is a fully zero-shot, training-free framework requiring no training data or fine-tuning. Query rewriting is performed with LLaMA3-8B, caption generation with Qwen2.5-VL-7B, and initial frame filtering with CLIP ViT-B/32. Query-agnostic descriptions are generated offline; query-aware descriptions are generated online only for key frames, achieving significantly shorter inference time than Moment-GPT (6.2s vs. 16.1s).

## Key Experimental Results

### Main Results

| Dataset | Metric | GranAlign | Prev. SOTA (Moment-GPT) | Gain |
|--------|------|-----------|---------------------|------|
| QVHighlights val | R1@0.5 | 61.94 | 58.9 | +3.04 |
| QVHighlights val | mAP@avg | 39.12 | 35.9 | +3.22 |
| QVHighlights test | R1@0.5 | 59.92 | 58.3 | +1.62 |
| QVHighlights test | mAP@avg | 38.23 | 35.0 | +3.23 |
| Charades-STA | R1@0.5 | 39.6 | 38.4 | +1.2 |
| Charades-STA | mIoU | 38.0 | 36.5 | +1.5 |
| ActivityNet | R1@0.5 | 34.0 | 31.1 | +2.9 |
| ActivityNet | mIoU | 33.1 | 30.8 | +2.3 |

### Ablation Study (QVHighlights val)

| Query → $C_{agn}$ | Query → $C_{awr}$ | R1@0.5 | mAP@avg |
|-----------------|-----------------|--------|---------|
| $Q_r$ (original) | - | 57.94 | 31.80 |
| - | $Q_r$ | 58.19 | 32.13 |
| $Q_s$ (simplified) | - | 58.97 | 37.13 |
| - | $Q_d$ (detailed) | 59.48 | 37.65 |
| $Q_s$ | $Q_d$ (full GranAlign) | **61.94** | **39.12** |

### Key Findings
- Granularity matching is critical: pairing the simplified query with query-aware descriptions (granularity mismatch) degrades performance, while matched-granularity pairings yield clear improvements.
- Dual-channel fusion consistently outperforms any single-channel variant: the simplified pair contributes high recall, the detailed pair contributes high precision, and their combination benefits both.
- On the Video Highlight Detection (VHD) task, GranAlign achieves 39.35% mAP, surpassing the fully supervised QD-DETR (39.04%), demonstrating the substantial potential of zero-shot approaches.
- Inference efficiency is excellent: 6.2s per query far outperforms Moment-GPT's 16.1s, owing to the two-stage captioning strategy.
- The framework is robust to hyperparameter choices (rewriting count $m=3$, $\lambda=0.3$ yields stable performance across nearby values).

## Highlights & Insights
- The formulation of "granularity mismatch" is precise and well-motivated, supported by quantitative evidence through query-type categorization (Error/Simple/Detail/Else).
- The dual-channel granularity alignment design is both intuitive and effective—one channel handles simple queries, the other handles detailed ones, and score averaging avoids complex fusion mechanisms.
- The framework is fully zero-shot with no training cost, yet approaches or surpasses fully supervised methods on certain QVHighlights metrics, demonstrating strong practical value.
- The two-stage captioning strategy (offline generic + online key-frame-aware) represents a sound engineering trade-off between efficiency and accuracy.

## Limitations & Future Work
- Query-aware descriptions may suffer from hallucinations—generating visual content absent from the video or over-imitating the linguistic structure of the query.
- LLM-based query rewriting may alter the original intent, necessitating a semantic validation step.
- The framework depends on multiple large models (LLaMA3 + Qwen2.5-VL + CLIP + SentenceTransformer), resulting in non-trivial deployment costs.
- For event-dense long videos, simplified queries may cover too much irrelevant content.

## Related Work & Insights
- Moment-GPT is the direct predecessor, using LLaMA-3 for query rewriting and Video-ChatGPT for scoring, but remains a single-channel, single-granularity design.
- The granularity-aware paradigm proposed here is generalizable to other retrieval settings—such as handling varying formulations of the same question in text retrieval or abstract/concrete query pairs in image retrieval.
- Combining the "generate multi-granularity representations then align" paradigm with RAG or multi-step reasoning may yield more powerful multimodal understanding systems.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](../../ACL2026/llm_evaluation/the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)

</div>

<!-- RELATED:END -->
