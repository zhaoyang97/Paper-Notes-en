---
title: >-
  [Paper Note] FunQA: Towards Surprising Video Comprehension
description: >-
  [ECCV 2024][LLM (Other)][video question answering] The authors construct a large-scale counter-intuitive video question answering benchmark, FunQA (consisting of 4.3K videos and 312K QA pairs), covering three categories of surprising videos: Humor, Creativity, and Magic. They also propose the FunMentor agent, which enhances the counter-intuitive reasoning capabilities of VLMs through multi-turn dialogue.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "video question answering"
  - "counter-intuitive reasoning"
  - "surprising videos"
  - "VLM"
  - "benchmark"
date: 2026-05-08
content_hash: 604c88cf3b77ee57
---

# FunQA: Towards Surprising Video Comprehension

**Conference**: ECCV 2024  
**arXiv**: [2306.14899](https://arxiv.org/abs/2306.14899)  
**Code**: [https://github.com/Jingkang50/FunQA](https://github.com/Jingkang50/FunQA)  
**Area**: LLM/NLP  
**Keywords**: video question answering, counter-intuitive reasoning, surprising videos, VLM, benchmark

## TL;DR

The authors construct a large-scale counter-intuitive video question answering benchmark, FunQA (consisting of 4.3K videos and 312K QA pairs), covering three categories of surprising videos: Humor, Creativity, and Magic. They also propose the FunMentor agent, which enhances the counter-intuitive reasoning capabilities of VLMs through multi-turn dialogue.

## Background & Motivation

### Background

**Background**: Existing VideoQA benchmarks mainly focus on common, everyday scenarios (cooking, instruction, etc.) and lack evaluation regarding "surprising" video comprehension.

### Limitations of Prior Work

**Limitations of Prior Work**: Understanding entertaining videos goes beyond basic visual perception; it requires comprehending deviations from common sense—i.e., why a certain scene is funny, creative, or mind-boggling.

### Key Challenge

**Key Challenge**: GPT-4V has already achieved an accuracy of 80% on NExT-QA, necessitating the creation of more challenging benchmarks.

### Key Insight

**Key Insight**: Existing datasets for humor/creativity comprehension heavily rely on audio and narrative cues, with the role of visual understanding being underemphasized.

### Supplementary Note

**Supplementary Note**: The average response length in FunQA is 34.2 words, which far exceeds NExT-QA's 2.6 words, indicating a demand for deeper video comprehension.

## Method

### Overall Architecture

**The FunQA dataset consists of three subsets**:
1. **HumorQA**: Funny videos, featuring unexpected contrasts and plot twists.
2. **CreativeQA**: Creative performance videos, highlighting ingenious disguises and creative techniques.
3. **MagicQA**: Magic performance videos, centering on seemingly impossible events.

**Four types of task design**:
1. Counter-intuitive timestamp localization: Locating the specific video segment where the surprising event occurs.
2. Detailed video description: Generating coherent and objective descriptions of the video content.
3. Counter-intuitive reasoning: Explaining why the video is surprising or funny.
4. Advanced tasks: Title generation, creativity scoring, and magic trick explanation.

### Key Designs

**FunMentor Agent**:
- Plays a role similar to a mentor in a variety show.
- Guides the VLM through multi-turn dialogue:
  1. Basic Description → Guiding the model to focus on key details.
  2. Comparative Analysis → Guiding the model to identify anomalous elements.
  3. Reasoning & Integration → Guiding the model to formulate a comprehensive explanation.
- Employs precise prompting strategies to produce fluent and logical answers.

**Data Construction Pipeline** (~900 hours, 50+ annotators):
1. Preprocessing: YouTube scraping → Two-stage manual filtering and trimming.
2. Manual Annotation: Chinese annotation → 10% secondary verification → Consensus evaluation.
3. Post-processing: GPT-3.5 translation and expansion → 312K QA pairs.
4. Provides both FunQA-MC (multiple-choice) and FunQA-DIA (dialogue) formats.

### Loss & Training

- FunMentor does not involve model training; it is an inference-time prompting strategy.
- Evaluation utilizes a variety of metrics: GPT-4 assisted evaluation, BLEU, ROUGE, BERTScore, etc.

## Key Experimental Results

### Main Results

| Model | H1 (Locating) | H2 (Description) | H3 (Reasoning) | C3 (Reasoning) | M3 (Reasoning) |
|------|-----------|-----------|-----------|-----------|-----------|
| Video-ChatGPT | 1.23 | 2.05 | 1.89 | 1.67 | 1.54 |
| VideoChat2 | 1.31 | 2.14 | 2.03 | 1.83 | 1.72 |
| GPT-4V | 1.98 | 2.89 | 2.67 | 2.45 | 2.19 |
| GPT-4V + FunMentor | **2.34** | **3.12** | **3.01** | **2.78** | **2.51** |

### Ablation Study

| FunMentor Component | Improvement on Reasoning Task |
|---------------|-----------|
| W/o FunMentor | Baseline |
| + Single-turn guidance | +0.15 |
| + Multi-turn dialogue | +0.34 |
| + Full FunMentor | **+0.34** |

### Key Findings

- All existing VLMs perform significantly worse on FunQA compared to typical VideoQA benchmarks.
- Timestamp localization is the most challenging task, as models generally struggle to pinpoint counter-intuitive moments precisely.
- FunMentor delivers massive improvements across all tasks, validating the effectiveness of multi-turn dialogue guidance.
- The annotation consensus rate exceeds 90% in the "high consensus" category, and only 1% falls into "low consensus", demonstrating high data quality.
- MagicQA is the most challenging subset, as it requires models to understand physical intuition and deduce the principles behind magic tricks.

## Highlights & Insights

1. **Fills a crucial gap**: The first large-scale benchmark systematically targeting counter-intuitive and entertaining video comprehension.
2. **Hierarchical task design**: Scales up from perception (localization) to understanding (description) and finally to reasoning (explanation).
3. **Simple yet effective FunMentor**: A training-free multi-turn prompting strategy that significantly bolsters the reasoning capability of VLMs.
4. Exposes severe deficiencies in state-of-the-art VLMs concerning counter-intuitive reasoning.
5. Strict quality control: Involves 900+ annotation hours and multi-stage consensus validation.

## Limitations & Future Work

- The video sources are primarily from YouTube, which may introduce cultural biases (skewed toward Western humor styles).
- A substantial portion of the 312K QA pairs was translated/expanded using GPT-3.5, posing quality control challenges.
- FunMentor relies heavily on specific prompt engineering, and its generalizability across different VLMs warrants further validation.
- Evaluation remains primarily reliant on automated metrics, with human evaluation conducted on a relatively small scale.

## Related Work & Insights

- **NExT-QA**: A pioneer in open-ended VideoQA, but lacks sufficient difficulty.
- **CLEVRER**: A synthetic video reasoning benchmark.
- **Whoops**: Focuses on counter-intuitive image understanding.
- Insight: Models' "understanding" capability should not merely be evaluated on everyday scenarios; counter-to-common-sense reasoning constitutes a far deeper test of cognitive abilities.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 9 |
| Technical Depth | 6 |
| Experimental Thoroughness | 8 |
| Value | 8 |
| Writing Quality | 8 |
| Overall Rating | 7.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)
- [\[ICML 2026\] Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models](../../ICML2026/llm_nlp/masks_can_be_distracting_on_context_comprehension_in_diffusion_language_models.md)
- [\[CVPR 2026\] Single-step Diffusion-based Video Coding with Semantic-Temporal Guidance](../../CVPR2026/llm_nlp/single-step_diffusion-based_video_coding_with_semantic-temporal_guidance.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](../../ACL2025/llm_nlp/scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](../../CVPR2025/llm_nlp/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)

</div>

<!-- RELATED:END -->
