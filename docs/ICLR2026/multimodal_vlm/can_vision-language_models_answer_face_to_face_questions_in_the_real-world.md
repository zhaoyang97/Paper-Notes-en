---
title: >-
  [Paper Note] Can Vision-Language Models Answer Face to Face Questions in the Real-World?
description: >-
  [ICLR2026][Multimodal VLM][situated understanding] This paper introduces QIVD (Qualcomm Interactive Video Dataset), a face-to-face real-time QA benchmark comprising 2,900 videos with audio and timestamp annotations. It reveals that existing VLMs fall far short of human performance in real-time situated understanding (best model 60% vs. humans 87%), with primary bottlenecks in referential disambiguation, response timing judgment, and situated commonsense. Fine-tuning on this data can substantially close the gap.
tags:
  - ICLR2026
  - Multimodal VLM
  - situated understanding
  - real-time interaction
  - video QA
  - multimodal benchmark
  - streaming VLM
date: 2026-05-08
content_hash: 7ce86302a8f30573
---

# Can Vision-Language Models Answer Face to Face Questions in the Real-World?

**Conference**: ICLR2026
**arXiv**: [2503.19356](https://arxiv.org/abs/2503.19356)
**Code**: [https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd](https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd)
**Area**: Multimodal VLM
**Keywords**: situated understanding, real-time interaction, video QA, multimodal benchmark, streaming VLM

## TL;DR
This paper introduces QIVD (Qualcomm Interactive Video Dataset), a face-to-face real-time QA benchmark comprising 2,900 videos with audio and timestamp annotations. It reveals that existing VLMs fall far short of human performance in real-time situated understanding (best model 60% vs. humans 87%), with primary bottlenecks in referential disambiguation, response timing judgment, and situated commonsense. Fine-tuning on this data can substantially close the gap.

## Background & Motivation

**Background**: Large multimodal models (LMMs) have achieved notable progress on image captioning and VQA, and are beginning to support real-time audio-visual dialogue. However, existing capabilities remain confined to "offline reasoning"—receiving complete visual input and full question text before generating a response.

**Limitations of Prior Work**: (a) Existing video understanding benchmarks adopt an offline paradigm in which models observe the entire video and complete question in advance; (b) No benchmark exists for evaluating "face-to-face dialogue" capabilities, i.e., a model connected to a camera and microphone that answers questions in real time; (c) Models have no mechanism to determine *when* to respond—the when-to-speak problem in conversational interaction has been largely overlooked.

**Key Challenge**: Real-world AI assistants and humanoid robots must simultaneously parse dynamic scenes, comprehend spoken questions, and determine the appropriate moment to respond. Yet both training data and evaluation benchmarks remain offline, leaving models without these capabilities.

**Goal**: (a) Construct the first face-to-face real-time QA dataset; (b) systematically evaluate the capability boundaries of existing models; (c) demonstrate that fine-tuning on such data improves real-time interaction performance.

**Key Insight**: The paper designs a straightforward online QA paradigm in which a user records a video with a mobile device, performs actions, and poses questions simultaneously. The model must understand the scene from synchronized video and audio inputs and respond at the correct moment.

**Core Idea**: By constructing a real-time interactive QA dataset with synchronized audio, video, and answer timestamps, this work provides the first systematic measurement of VLMs' face-to-face interaction capabilities and identifies three primary failure modes.

## Method

### Overall Architecture
QIVD is a combined dataset, benchmark, and streaming baseline contribution. Data consists of short videos (~5 seconds) crowdsourced from workers recording with mobile devices, during which they verbally pose questions while demonstrating scenes. Annotations include: (1) question transcription; (2) answer; and (3) optimal response timestamp (the point in the video at which sufficient information is available to answer). Evaluation is conducted under both streaming and offline settings.

### Key Designs

1. **Dataset Design (QIVD)**:

    - **Function**: Construct a real-time face-to-face QA dataset of 2,900 videos.
    - **Mechanism**: Crowdsource workers record short videos using mobile phones or computers, simultaneously demonstrating scenes and posing spoken questions (e.g., "What color is this?" / "How many apples am I holding?"). Each video is annotated with three elements: question transcription, answer, and optimal response timestamp. Videos span 13 semantic categories: object attribute/counting/detection/referencing, action attribute/counting/detection/understanding, scene understanding, audio-visual fusion, OCR, and subjective judgment.
    - **Design Motivation**: Existing benchmarks follow an "offline" paradigm in which questions are posed after viewing the complete video. In QIVD, questions are embedded within the audio stream, and answering may require continued observation after the question ends. This naturally evaluates both real-time comprehension and response timing.

2. **Response Timestamp Annotation (When-to-Answer)**:

    - **Function**: Annotate the point in the video at which sufficient information exists to answer the question.
    - **Mechanism**: The response timestamp does not necessarily coincide with the end of the spoken question. For instance, if a user first asks "What action am I doing?" and then performs the action, the optimal response moment is after the action completes. The average response timestamp falls at 81.47% of video duration; action counting questions have the latest timestamps (92.22%) and object detection questions have the earliest (76.95%).
    - **Design Motivation**: This annotation enables QIVD to assess not only comprehension ability but also timing judgment—knowing *when to speak* is central to real-time dialogue.

3. **Streaming Baseline**:

    - **Function**: Propose a simple modular streaming processing framework.
    - **Mechanism**: Whisper-Streaming ASR transcribes audio in 0.25-second chunks in real time and detects question completion → upon detecting question end, video frames up to that timestamp plus the transcribed question are fed to a Video-LMM → the model generates an answer. Additionally, Qwen2.5-Omni is fine-tuned to detect the optimal response moment.
    - **Design Motivation**: Existing LMMs do not natively support synchronized audio-visual streaming. This modular pipeline serves as a baseline demonstrating that even state-of-the-art ASR+VLM combinations find real-time interaction challenging.

4. **Evaluation Protocol**:

    - **Function**: Comprehensively evaluate open-source and closed-source VLMs on face-to-face interaction.
    - **Mechanism**: Answer correctness is assessed using an LLM judge (Qwen3-8B), supplemented by BERT, METEOR, BLEU, and ROUGE-L similarity metrics. Evaluation is conducted under two settings: streaming (ASR-extracted questions and timestamps) and offline (ground-truth questions and timestamps).
    - **Design Motivation**: Free-form answers cannot be evaluated by exact match; an LLM judge provides greater flexibility. The two settings decouple the effects of ASR errors from visual comprehension errors.

### Loss & Training
Fine-tuning experiments employ VideoLLaMA2.1-7B-AV on QIVD using 5-fold cross-validation, with the visual encoder frozen and the LLM backbone and audio pathway fine-tuned for 2 epochs per fold. Stream-Qwen-Omni fine-tunes Qwen2.5-Omni to support streaming input, training the model to emit a special token to signal the optimal response moment.

## Key Experimental Results

### Main Results (Offline Setting: Ground-Truth Questions and Timestamps)

| Model | Corr. | BERT | METEOR | BLEU | ROUGE-L |
|-------|-------|------|--------|------|---------|
| **Human** | **87.33** | 93.01 | 53.21 | 17.40 | 49.76 |
| Qwen3-VL-8B | 60.07 | 87.58 | 36.72 | 6.64 | 35.89 |
| GPT-4o | 58.76 | 89.36 | 51.18 | 15.72 | 42.55 |
| Gemini-2.5-Flash | 58.07 | 90.43 | 43.07 | 8.33 | 36.05 |
| VideoLLaMA3-7B | 56.38 | 91.63 | 48.56 | 12.72 | 43.84 |
| VideoLLaMA2-72B | 50.83 | 92.29 | 51.13 | 16.12 | 45.76 |
| Qwen2.5-VL-7B | 50.62 | 87.58 | 37.37 | 4.66 | 29.44 |

### Ablation Study (Audio Contribution + Fine-Tuning Effect)

| Configuration | Overall Corr. | Action Counting | Audio-Visual | Subjective |
|---------------|--------------|-----------------|--------------|------------|
| VideoLLaMA2.1-7B (video only) | ~43% | ~13% | ~27% | ~23% |
| VideoLLaMA2.1-7B (video + audio) | ~40% | ~10% | ~23% | ~16% |
| Fine-tuned (video + audio) | **~52%** | **~30% (+17%)** | **~45% (+17%)** | **~47% (+23%)** |
| Fine-tuned (video only) | ~47% | ~22% | ~27% | ~37% |

### Key Findings
- **27% gap between humans and the best model**: Humans achieve 87.33% vs. Qwen3-VL's 60.07%; even with ground-truth questions and timestamps, models remain substantially behind.
- **Audio does not always help**: Without fine-tuning, adding audio actually degrades performance (likely due to insufficient audio-visual alignment during pretraining); after fine-tuning, audio becomes a critical asset, particularly for audio-visual tasks and subjective judgment.
- **Response timing has a significant impact**: Stream-Qwen-Omni achieves a timestamp detection MAE of 0.52s (vs. 0.83s for Whisper-Streaming), and more accurate timing directly improves answer correctness.
- **Action counting is the hardest category**: Even after fine-tuning, accuracy remains ~30%, suggesting that temporal reasoning may require stronger inductive biases at the architectural level.
- **Referential disambiguation is a core challenge**: Object referencing accounts for the largest proportion of samples (24.34%), and models frequently misinterpret deictic expressions such as "this" and "that."

## Highlights & Insights
- **First genuinely face-to-face interaction benchmark**: Embedding questions within the video audio stream rather than appending them post hoc naturally produces challenges such as referential disambiguation and response timing—reflecting real conversational scenarios more faithfully than post-hoc annotation.
- **Independent research value of When-to-Answer**: Knowing when to speak is fundamental to dialogue yet has received almost no research attention. Response timestamp annotations make this problem independently evaluable and optimizable.
- **Heterogeneous gains from fine-tuning**: Fine-tuning yields large improvements in action counting (+17%) and audio-visual tasks (+17%), but minimal gains in object attribute and scene understanding—indicating that different situated comprehension capabilities require different training strategies.
- **Modular streaming design**: The architecture of ASR-based timing detection followed by VLM response generation is simple yet effective, and can serve as a starting point for real-time interactive systems.

## Limitations & Future Work
- **Videos are too short**: The average duration of ~5 seconds precludes evaluation of extended interaction, multi-turn dialogue, and context switching.
- **Single-turn QA only**: Only single question–single answer pairs are evaluated; real conversations require multi-turn comprehension and context maintenance.
- **English only**: Audio is limited to English; cross-lingual generalization and accent diversity are not covered.
- **Imbalanced category distribution**: Audio-visual fusion (0.76%) and OCR (0.79%) have very few samples, potentially leading to unstable evaluation.
- **Reliance on LLM judge**: Evaluation of free-form answers is inherently subjective, and different LLM judges may produce divergent conclusions.

## Related Work & Insights
- **vs. AVSD / SocialIQ (video dialogue benchmarks)**: These datasets are built on pre-recorded videos with post-hoc question annotation. In QIVD, questions are posed synchronously during recording, naturally incorporating referential expressions and temporal dependencies.
- **vs. VideoLLM-online / FlashVStream (streaming models)**: These works attempt real-time processing but lack audio and are limited to specific domains. QIVD provides a more general evaluation platform.
- **vs. Ego4D Social**: Both involve face-to-face interaction, but Ego4D uses task-specific labels rather than free-form QA. QIVD's open-ended question answering more closely approximates real dialogue.

## Rating
- Novelty: ⭐⭐⭐⭐ First face-to-face real-time interaction QA benchmark; the when-to-answer annotation is a particular highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 20+ models across streaming/offline/audio ablation/fine-tuning/timing analysis dimensions.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Directly addresses core capability gaps in real-time AI assistants and robots, with meaningful implications for the field's development.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](../../CVPR2026/multimodal_vlm/demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)
- [\[ICLR 2026\] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/multimodal_vlm/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/multimodal_vlm/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

<!-- RELATED:END -->
