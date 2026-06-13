---
title: >-
  [Paper Note] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding
description: >-
  [CVPR 2026][Video Understanding][Video Understanding Benchmark] This paper proposes VirtueBench, the first long video understanding benchmark for evaluating VLM trustworthiness under uncertainty. By constructing multi-le…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Understanding Benchmark"
  - "Trustworthiness under Uncertainty"
  - "Long Video Understanding"
  - "Refusal Behavior Evaluation"
  - "VLM Evaluation"
date: 2026-05-08
content_hash: 7c7fbc033ba2d1bf
---

# VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.07071](https://arxiv.org/abs/2603.07071)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: Video Understanding Benchmark, Trustworthiness under Uncertainty, Long Video Understanding, Refusal Behavior Evaluation, VLM Evaluation

## TL;DR
This paper proposes VirtueBench, the first long video understanding benchmark for evaluating VLM trustworthiness under uncertainty. By constructing multi-level frame sampling for each video and annotating answerable/unanswerable ground truth at each level, it reveals that existing models tend to guess rather than honestly refuse to answer.

## Background & Motivation
Vision-language models (VLMs) have achieved remarkable progress in multimodal understanding, yet evaluation of long video understanding remains unreliable.

**Key Challenge**: Due to input frame limits (typically 256–512 frames), key frames of long videos may not be included in the model's input. Under existing evaluation protocols:
- Models that honestly refuse to answer ("insufficient video information") are penalized as incorrect
- Models that happen to guess correctly receive inflated accuracy scores
- This **incentivizes guessing over honest responses**, producing misleading evaluation results

**Concrete Example**: When tested on a 64-frame subset of VideoEval-Pro, Qwen2.5-VL-72B honestly indicated insufficient information (marked wrong), while LLaVA-Video-72B guessed correctly (marked right)—yet the latter never actually saw the key frames needed to answer the question.

**Limitations of Prior Work**:
1. Standard long video benchmarks (Video-MME, MLVU, etc.) use full-video answers as ground truth across all frame settings—without considering the actual visible frame range
2. Video hallucination benchmarks (VideoHallucer, VIDHALLUC) diagnose specific hallucination types but do not evaluate honest refusal under insufficient information
3. Multiple-choice question formats further exacerbate the guessing problem

**Core Idea**: Construct a benchmark that provides distinct ground truth for each frame sampling level—when frames do not contain key information, the correct answer is "insufficient information," and only an honest refusal is counted as correct.

## Method

### Overall Architecture
VirtueBench collects video-question pairs from existing long video benchmarks, constructs 5-level frame sampling (64/128/256/512/1024 frames) for each video, and annotates independent ground truth (including an "insufficient information" label) for each level.

### Key Designs

1. **Multi-Level Frame Sampling Construction**:

    - Function: Each video is uniformly downsampled to 1 FPS, then uniformly sampled into 5-level clips of 64/128/256/512/1024 frames.
    - Design Motivation: Different frame levels capture different visual content; the same question may be answerable at some levels but not others.
    - Key Design: Independent ground truth annotation is provided for each frame level.

2. **Data Collection and Quality Filtering**:

    - Sources: MLVU, LVBench, LongVideoBench, MovieChat, Video-MME, ALLVB
    - Initial scale: 3,042 videos, 33,400 questions
    - **MCQ→Open-ended Conversion**: All multiple-choice questions are converted to open-ended format using the correct option as the reference answer, eliminating guessing.
    - **Multi-Stage Filtering**:
        - Remove questions with excessively long answers (>6 words)
        - Remove questions that rely on option context, reference timestamps/subtitles, or involve subjective judgment (automatically detected by Gemini-2.5-Flash)
        - **Common Sense Filtering**: A single randomly sampled frame is presented to Gemini-2.5-Flash; questions answerable from this single frame do not require video understanding and are discarded
    - Approximately 2,500 open-ended QA pairs remain after filtering

3. **Annotation and Verification Pipeline**:

    - Function: Annotate correct answers for each question at different frame levels.
    - Procedure:
        - Gemini-2.5-Pro first generates reference answers at each frame level
        - Human annotators carefully review all clips and provide final annotations by combining the original full-video answer with AI-generated references
        - Questions are labeled "The video does not provide enough information" when key frames are absent from the clip
        - Annotators are required to specify timestamp evidence supporting their answers
    - **Dual Review**: Each instance is reviewed by at least two annotators—primary annotation followed by correction review
    - Disputed questions are discarded; random spot-checks are conducted; substandard annotations are returned for re-labeling
    - Final dataset: 1,328 high-quality annotated instances

4. **Evaluation Pipeline (LLM-as-Judge)**:

    - Design Motivation: Open-ended questions cannot be evaluated through rule-based matching.
    - Evaluation model: GPT-4o
    - **Two-Stage Judgment**:
        - Refusal detection: Determine whether the model refuses to answer
        - Correctness evaluation: For questions with definite answers, verify semantic consistency; for "insufficient information" questions, only a refusal is counted as correct

### Evaluation Metrics
- **Overall accuracy**: Total accuracy including refusal judgment
- **Non-refusal accuracy**: Accuracy on the subset with definite answers only
- **Refusal accuracy**: The proportion of correct refusals on the subset where ground truth is "insufficient information"
- Results are further broken down by Perception/Reasoning

## Key Experimental Results

### Dataset Statistics
- 1,328 instances, 901 source videos
- 767 perception + 561 reasoning questions (balanced coverage)
- Unanswerable rate by frame level: ~50% at 64 frames → ~25% at 1024 frames (decreasing as frame count increases)
- Instance-level distribution is balanced: from fully unanswerable to fully answerable

### Main Results — Overall Accuracy (64 frames)

| Model | Overall | P/R |
|-------|---------|-----|
| Gemini-2.5-Flash | **58.96** | 63.60/54.70 |
| GPT-4o | 55.43 | 59.81/49.74 |
| Qwen3VL-32B | 50.83 | 53.00/48.01 |
| Qwen2.5VL-72B | 49.32 | 52.86/44.71 |
| GPT-5 | 50.30 | 51.40/48.87 |
| Mimo-VL-7B-RL | 39.98 | 42.74/36.40 |
| LLaVA-Video-72B | 25.53 | 29.83/19.93 |

### Key Findings — Refusal Behavior Analysis

| Model Type | Refusal Accuracy |
|------------|-----------------|
| Qwen-VL series | Strongest open-source refusal capability (>50%) |
| Gemini-2.5-Flash | Best among commercial models |
| LLaVA-Video | Near-zero refusal behavior |
| GPT-5 | Performance improves with increasing frame count |

### Ablation Study — Effect of Honesty Instruction in Prompt

| Model | With Honesty Instruction | Without Honesty Instruction | Change in Refusal Accuracy |
|-------|------------------------|----------------------------|---------------------------|
| Most models | Refusal capability present | Refusal nearly disappears | ~50% decrease |

### Key Findings
- **Accuracy decreases as frame count increases**: This is contrary to the conventional assumption in standard benchmarks that "more frames is better"—because VirtueBench provides independent ground truth per frame level, denser sampling means fewer unanswerable questions, but reasoning on answerable questions also becomes more demanding.
- **Refusal behavior is highly prompt-dependent**: Removing the honesty instruction causes the refusal accuracy of most models to drop by approximately half, indicating that refusal capability is primarily elicited by prompting rather than being an intrinsic model property.
- **Perception outperforms Reasoning**: Reasoning tasks require cross-frame information integration and higher-order inference, making them significantly harder.
- Gemini-2.5-Flash achieves the best performance across all frame levels with consistent stability.
- The open-source Qwen3-VL-32B substantially narrows the gap with closed-source models.
- **LLaVA-Video series exhibits nearly 0% refusal rate**—this model is trained to always produce an answer.

## Highlights & Insights
- **The problem formulation itself is a major contribution**: This is the first work to explicitly identify evaluation bias in long video benchmarks—correct guesses are rewarded while honest responses are penalized.
- The multi-level frame sampling with per-level ground truth annotation is both simple and effective.
- Common sense filtering (discarding questions answerable from a single frame) ensures evaluation fairness.
- The MCQ-to-open-ended conversion further reduces the guessing space.
- The work reveals a deeper issue: **the training paradigms of existing VLMs (RLHF, SFT) encourage models to "always give an answer" rather than "know what they do not know."**

## Limitations & Future Work
- The scale of 1,328 instances is relatively limited, and domain coverage may be insufficient.
- The reliability of LLM-as-Judge (GPT-4o) as an evaluator has not been thoroughly validated.
- The definition of "honest refusal" may be overly strict—models may express uncertainty in vague or hedged ways rather than explicitly refusing.
- Uniform frame sampling is adopted; intelligent sampling strategies (e.g., keyframe detection) may alter the conclusions.
- Only binary refusal/non-refusal behavior is evaluated; finer-grained uncertainty expressions such as confidence calibration are not assessed.
- The paper does not explore how to improve model refusal capability through training.

## Related Work & Insights
- VideoEval-Pro: Identifies the guessing problem in MCQ; this paper further addresses evaluation bias under insufficient information.
- Video hallucination benchmarks (VideoHallucer, etc.): Diagnose hallucination types, whereas this paper evaluates honesty under uncertainty.
- Implications for VLM evaluation: All benchmarks requiring inference from limited inputs should account for "insufficient information" scenarios.
- Implications for model training: Honest refusal should be incorporated into RLHF reward functions rather than optimizing accuracy alone.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic definition and evaluation of VLM trustworthiness under uncertainty; the problem formulation is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 25 models (including commercial) + 5-level frame sampling + refusal behavior analysis + prompt ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation, excellent figures, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Provides an important corrective for long video understanding evaluation and advances the development of trustworthy VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)

</div>

<!-- RELATED:END -->
