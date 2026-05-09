---
title: >-
  [Paper Note] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks
description: >-
  [CVPR 2026][Multimodal VLM][human-centric video understanding] This paper presents HumanVBench, a video benchmark comprising 16 fine-grained tasks, systematically evaluating the human-centric video understanding capabilities of MLLMs via two automated pipelines (video annotation and distractor generation). The benchmark reveals significant deficiencies in current models regarding emotion perception and speech-visual alignment.
tags:
  - CVPR 2026
  - Multimodal VLM
  - human-centric video understanding
  - MLLM benchmarking
  - automated benchmark construction
  - emotion perception
  - speech-visual alignment
date: 2026-05-08
content_hash: 9aba0c8dac198b59
---

# HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks

**Conference**: CVPR 2026
**arXiv**: [2412.17574](https://arxiv.org/abs/2412.17574)
**Code**: [https://github.com/datajuicer/data-juicer/tree/HumanVBench](https://github.com/datajuicer/data-juicer/tree/HumanVBench)
**Area**: Video Understanding / Multimodal Evaluation
**Keywords**: human-centric video understanding, MLLM benchmarking, automated benchmark construction, emotion perception, speech-visual alignment

## TL;DR

This paper presents HumanVBench, a video benchmark comprising 16 fine-grained tasks, systematically evaluating the human-centric video understanding capabilities of MLLMs via two automated pipelines (video annotation and distractor generation). The benchmark reveals significant deficiencies in current models regarding emotion perception and speech-visual alignment.

## Background & Motivation

1. **Background**: Video MLLMs are advancing rapidly, yet existing benchmarks primarily assess general content understanding and lack fine-grained evaluation of human-centric capabilities such as emotion, behavior, and cross-modal alignment.
2. **Limitations of Prior Work**: Emotion benchmarks are limited to discrete classification; action benchmarks overlook emotional states and speaker identification; speech-visual synchronization has rarely been systematically evaluated.
3. **Key Challenge**: Humans can readily detect audio-visual mismatches, yet models exhibit severe deficiencies in speaker identification and lip-speech alignment.
4. **Goal**: To construct a benchmark that systematically evaluates the fundamental human-centric perceptual capabilities of MLLMs.
5. **Key Insight**: Two automated pipelines substantially reduce the need for manual annotation while leveraging model errors to generate high-quality distractors.
6. **Core Idea**: Model-induced errors are repurposed as semantically plausible distractors, simultaneously ensuring question difficulty and reducing human effort.

## Method

### Overall Architecture

Two core pipelines are proposed: (1) a human-centric video annotation pipeline that employs 20+ state-of-the-art operators to produce dense multimodal annotations; and (2) a QA synthesis pipeline with distractors that generates high-quality multiple-choice questions, using erroneous answers from a multi-model ensemble as distractors. The resulting benchmark contains 2,475 questions spanning 16 tasks.

### Key Designs

1. **Human-Centric Video Annotation Pipeline**:

    - **Function**: Automatically generates dense, fine-grained human-centric annotations from raw video.
    - **Mechanism**: Begins with human tracking (`video_human_tracks_extraction`) to obtain reliable person trajectories and counts, followed by extraction of demographic information, appearance descriptions, and facial expression descriptions from the tracks. On the audio side, active speaker detection, ASR transcription, speech emotion recognition, and acoustic feature analysis are performed.
    - **Design Motivation**: Automation is achieved by integrating multiple task-specific operators, avoiding large-scale manual annotation.

2. **QA Synthesis Pipeline with Distractors**:

    - **Function**: Generates semantically plausible and discriminative multiple-choice questions.
    - **Mechanism**: Multiple MLLMs (Gemini, VideoLLaMA3, ShareGPT4Video) independently generate candidate answers, which are ranked via preference voting. The highest-voted answer serves as the ground truth, while incorrect answers—reflecting typical model errors—are retained as distractors. When semantic diversity is insufficient, an LLM introduces task-specific perturbations.
    - **Design Motivation**: Retaining common model errors as distractors ensures both plausibility and difficulty. Approximately 72% of questions require no human correction.

3. **16 Fine-Grained Task Design**:

    - **Function**: Comprehensively evaluates human-centric video understanding capabilities.
    - **Mechanism**: Tasks are organized into intrinsic emotions (emotion recognition, emotion temporal analysis, attitude recognition, emotion intensity comparison) and extrinsic expressions (4 person identification tasks, 4 behavior analysis tasks, 4 speech-visual alignment tasks), totaling 16 tasks.
    - **Design Motivation**: Covers a complete evaluation hierarchy ranging from basic perception to advanced reasoning.

### Loss & Training

No training is involved, as this is a benchmark construction effort. Approximately 6% of questions with frequently correct responses under vision-free conditions are removed via answer leakage detection.

## Key Experimental Results

### Main Results

| Model | Emotion | Person ID | Behavior | Speech-Visual | Overall |
|-------|---------|-----------|----------|---------------|---------|
| Gemini-2.5-Pro | 52.9 | 83.5 | 70.7 | 86.5 | 73.4 |
| Qwen-VL3 (7B) | 43.2 | 67.6 | 54.3 | 48.3 | 53.4 |
| GPT-5 | 46.8 | 69.5 | 67.3 | — | — |
| Human (Graduate) | 84.6 | 88.5 | 87.0 | 94.4 | 88.6 |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| ER no-edit rate | 81% | Efficiency gain 5.3× |
| ETA no-edit rate | 83% | Efficiency gain 5.9× |
| Average no-edit rate | 72.3% | Average efficiency gain 3.6× |

### Key Findings

- Emotion perception is a universal weakness across all models; even Gemini-2.5-Pro achieves only 52.9%, far below the human score of 84.6%.
- Models frequently misclassify "open mouth" gestures in speaker-sampled frames as expressions of "surprise."
- For speech-visual alignment, most open-source audio-visual models perform near chance level, with only the Gemini series demonstrating strong performance.
- Qwen2.5-Omni achieves 71.8% on SCM (Speech Content Matching), exhibiting a distinctive advantage on purely audio-based tasks.

## Highlights & Insights

- **Model-Driven Benchmark Construction**: Human effort is shifted from manual creation to efficient verification, yielding a 3.6× efficiency improvement.
- **Pipeline Generalizability**: The pipeline can be adapted to non-human domains (e.g., pet attribute recognition, vehicle tracking) by substituting the underlying detectors.
- **Revealing Critical Gaps**: The work quantifies model-to-human performance gaps in two underexplored areas: emotion perception and lip-speech alignment.

## Limitations & Future Work

- The video sources are limited (primarily Pexels and MF2 films), and scene diversity could be further expanded.
- Systematic biases in audio annotation models may affect the quality of certain questions.
- The 16 tasks focus on the foundational perceptual layer and do not cover higher-level social intelligence reasoning.

## Related Work & Insights

- **vs. Video-MME**: Video-MME covers general video understanding but only 1% of its questions involve emotion, whereas HumanVBench is 100% human-centric.
- **vs. Social-IQ**: Social-IQ conflates perception and high-level reasoning; HumanVBench focuses on the foundational perceptual layer to provide purer diagnostic signals.

## Rating

- Novelty: ⭐⭐⭐⭐ The automated pipeline design and distractor generation strategy are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 30 models with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in human-centric video understanding evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)

<!-- RELATED:END -->
