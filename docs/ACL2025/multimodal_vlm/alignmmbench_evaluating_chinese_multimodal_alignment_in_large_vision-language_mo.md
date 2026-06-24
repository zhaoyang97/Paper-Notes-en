---
title: >-
  [Paper Note] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models
description: >-
  [ACL 2025][Multimodal VLM][VLM Evaluation] Proposes AlignMMBench, the first multimodal alignment evaluation benchmark for Chinese visual contexts, covering 13 tasks across 3 major categories, with 1054 images and 4978 QA pairs (including single-turn/multi-turn dialogues). Additionally, a ChatGLM3-6B-based evaluator, CritiqueVLM, is trained, which outperforms GPT-4 in evaluation consistency.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "VLM Evaluation"
  - "Chinese Multimodal"
  - "alignment benchmark"
  - "CritiqueVLM"
  - "prompt robustness"
date: 2026-05-08
content_hash: aff88032aeeb831c
---

# AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models

**Conference**: ACL 2025  
**arXiv**: [2406.09295](https://arxiv.org/abs/2406.09295)  
**Code**: [https://github.com/THUDM/AlignMMBench](https://github.com/THUDM/AlignMMBench)  
**Area**: Multimodal VLM / Evaluation Benchmark  
**Keywords**: VLM Evaluation, Chinese Multimodal, alignment benchmark, CritiqueVLM, prompt robustness

## TL;DR
Proposes AlignMMBench, the first multimodal alignment evaluation benchmark for Chinese visual contexts, covering 13 tasks across 3 major categories, with 1054 images and 4978 QA pairs (including single-turn/multi-turn dialogues). Additionally, a ChatGLM3-6B-based evaluator, CritiqueVLM, is trained, which outperforms GPT-4 in evaluation consistency.

## Background & Motivation
**Background**: VLMs exhibit strong visual understanding capabilities after SFT and RLHF alignment training. Chinese VLMs (e.g., QwenVL, CogVLM, InternVL) have approached GPT-4o on public leaderboards.

**Limitations of Prior Work**: Existing benchmarks (e.g., MME, MMBench, MMMU) primarily evaluate fundamental capabilities using non-verbal forms like yes/no or multiple-choice questions, lacking fine-grained open-ended evaluation of alignment performance, especially benchmarks tailored for Chinese visual scenes.

**Key Challenge**: Chinese multimodal corpora are scarce and difficult to annotate (due to higher ambiguity in Chinese contexts), requiring iterative validation by multiple annotators; moreover, significant differences in image features and cultural backgrounds between Chinese and English mean that English-only datasets cannot comprehensively evaluate Chinese VLMs.

**Goal**: To build a high-quality Chinese multimodal alignment evaluation benchmark covering three major dimensions—perception and understanding, reasoning and analysis, and dialogue context—while addressing the difficulty of evaluating open-ended questions.

**Key Insight**: Manually collecting Chinese images from real-world scenarios and internet resources, designing prompt paraphrase strategies to generate semantically equivalent but differently phrased variant questions, and training a dedicated evaluator to replace GPT-4.

**Core Idea**: Build a reproducible and controllable Chinese multimodal alignment evaluation system using carefully curated Chinese visual scenes, prompt paraphrase strategies, and a rule-calibrated small evaluator.

## Method

### Overall Architecture
AlignMMBench consists of three core components:
- **Evaluation Dataset**: 1054 images, 4978 QA pairs, covering 13 tasks across 3 major categories.
- **Evaluator CritiqueVLM**: An automatic scoring model fine-tuned based on ChatGLM3-6B.
- **Alignment Score Metric**: Measures model robustness under different prompt variants.

### Key Designs

1. **Dataset Construction (3 Categories, 13 Tasks)**:

    - **Function**: Covers perception & understanding (description, recognition, counting, OCR, memory, knowledge), reasoning & analysis (reasoning, charts, programming, comparison, writing), and dialogue context (coherent/incoherent multi-turn dialogues).
    - **Mechanism**: ① Define task types $ \rightarrow $ ② Collect images from Chinese websites like Baidu via web scrapers $ \rightarrow $ ③ Manually filter low-quality images $ \rightarrow $ ④ Create seed questions $ \rightarrow $ ⑤ Paraphrase via LLM to generate semantically equivalent variants $ \rightarrow $ ⑥ Manually annotate reference answers $ \rightarrow $ ⑦ Conduct two-stage quality review.
    - **Design Motivation**: Chinese multimodal corpora are extremely scarce and cannot leverage existing datasets like VQAv2 as in English, requiring construction from scratch; dialogue context tasks (especially incoherent ones) evaluate the VLM's ability to detect errors, which is critical in practical applications.

2. **Prompt Paraphrase Strategy**:

    - **Function**: Uses LLMs to paraphrase each seed question into multiple stylistically diverse but semantically equivalent variants (expanding from 1054 seeds to 4978 QA pairs).
    - **Mechanism**: Real-world users express the same intent in various ways; this evaluates the model's robustness to different formulations.
    - **Design Motivation**: Introduces the "alignment score"—the consistency of the model's performance under different phrasings of the same question—to measure alignment stability.

3. **CritiqueVLM Evaluator**:

    - **Function**: Fine-tuned on ChatGLM3-6B, taking the question + reference answer + model response as input, and outputting a score of 1-10 along with a CoT explanation.
    - **Mechanism**: Designs general prompts (scoring scale, criteria, format) + task-specific prompts (category-level scoring keypoints) to perform SFT using human-annotated scoring data (containing CoT explanations).
    - **Training Details**: 32 x A800 GPUs, batch_size=128, 1000 iterations, with loss decreasing from 3.8 to 0.3.
    - **Design Motivation**: The GPT-4 API is a black box, expensive, and unstable due to updates; CritiqueVLM is open-source, controllable, and shows higher consistency with human ratings than GPT-4 (reducing MAE by 34.8%).

4. **Alignment Score**:

    - **Function**: Quantifies the scoring stability of the model under different prompt variants.
    - **Mechanism**: Calculates the inverse of the score variance across multiple variants of the same question; higher values indicate greater robustness.
    - **Design Motivation**: Models with high performance but low robustness are unreliable in real-world applications.

### Evaluation Metric System
- Consistency between CritiqueVLM and human scoring is measured using 6 metrics: MAE, Pearson/Spearman/Kendall correlation coefficients, Fuzzy Accuracy, and Strict Accuracy.

## Key Experimental Results

### CritiqueVLM Evaluator Comparison

| Evaluator | MAE (↓) | Pearson (↑) | Fuzzy Acc (↑) | Strict Acc (↑) |
|--------|---------|-------------|---------------|-----------------|
| ChatGLM3-6B | 2.424 | 0.230 | 0.350 | 0.285 |
| ChatGPT | 1.720 | 0.572 | 0.427 | 0.347 |
| GPT-4 | 1.256 | 0.839 | 0.677 | 0.565 |
| **CritiqueVLM** | **0.818** | **0.846** | **0.747** | **0.646** |

### VLM Benchmark Leaderboard (Partial)

| Model | Params | Avg Score | Perception & Understanding | Reasoning & Analysis | Dialogue Context | Align. Score |
|------|--------|--------|---------|---------|-----------|-------------|
| Qwen2-VL | 72B | 6.51 | ~7.0 | ~5.5 | ~5.8 | 1.54 |
| Claude | - | 6.51 | ~7.1 | ~5.6 | ~6.3 | 1.45 |
| GPT-4o | - | 6.41 | ~6.9 | ~5.8 | ~5.4 | 1.18 |
| CogVLM2 | 19B | 5.81 | ~6.5 | ~4.7 | ~5.4 | 1.49 |
| InternVL-Chat | 26B | 5.62 | ~6.0 | ~4.8 | ~5.4 | 1.12 |

### Key Findings
- All VLMs perform relatively well on perception and understanding (average score of 5.07) but poorly on reasoning and analysis (average score of 4.38).
- Incoherent scenarios in multi-turn dialogues score significantly lower than coherent scenarios, demonstrating that VLMs struggle to detect errors in preceding dialogue.
- English-centric VLMs (such as Phi-3-Vision) exhibit a significant performance drop on the Chinese benchmark.
- High-performing models are not necessarily highly robust: GPT-4o achieves high performance but has a lower alignment score (1.18), whereas Qwen2-VL excels in both.

## Highlights & Insights
- **First Chinese Visual Alignment Benchmark**: Fills the gap in Chinese multimodal evaluation, utilizing data from real-world Chinese scenarios with tasks spanning both single-turn and multi-turn dialogues.
- **Small-Model Evaluator Outperforming GPT-4**: With only 6B parameters, CritiqueVLM surpasses GPT-4's evaluation capability through rule calibration and SFT, demonstrating that domain-specific fine-tuned evaluators hold massive advantages, remaining highly controllable and economical.
- **Alignment Score Metric**: Incorporates robustness into the evaluation, assessing stability rather than only peak scores; this methodology can be generalized to other evaluation benchmarks.
- **Prompt Paraphrase Strategy**: Cost-effectively scales the evaluation size and introduces a robustness dimension.

## Limitations & Future Work
- The data scale is relatively limited (1054 images); the task coverage can be further expanded.
- CritiqueVLM is based on ChatGLM3-6B (an older model); upgrading to a newer base model may yield further improvements.
- The construction method for incoherent scenarios in dialogue context tasks is not detailed.
- The evaluation focuses solely on Chinese, leaving cross-lingual comparative analysis (Chinese-English) unexplored.
- Some images are sourced from web scrapers, requiring continuous attention to copyright compliance.
- Lacks theoretical analysis regarding the alignment score (e.g., why certain models are more robust).

## Related Work & Insights
- **vs MMBench**: MMBench includes both English and Chinese versions but only utilizes multiple-choice questions; AlignMMBench evaluates deeper alignment capability using open-ended questions.
- **vs LLaVABench / MM-Vet**: These are English open-ended benchmarks evaluated by GPT; AlignMMBench targets Chinese and employs a self-trained evaluator.
- **vs TouchStone**: TouchStone is also an open-ended VLM evaluation but only in English and without dialogue context tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The first Chinese multimodal alignment benchmark, with innovations in CritiqueVLM and the alignment score.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 15+ models, with comprehensive comparative validation for the evaluator.
- Writing Quality: ⭐⭐⭐⭐ Clear data construction pipeline and comprehensive evaluation methodology.
- Value: ⭐⭐⭐⭐ Holds significant practical value for the Chinese VLM community; CritiqueVLM can be directly reused.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](../../ICCV2025/multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)

</div>

<!-- RELATED:END -->
