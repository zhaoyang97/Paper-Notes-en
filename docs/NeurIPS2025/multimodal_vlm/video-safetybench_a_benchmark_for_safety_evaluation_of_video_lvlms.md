---
title: >-
  [Paper Note] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][Video Safety] This paper presents Video-SafetyBench, the first comprehensive benchmark for safety evaluation of video LVLMs. It comprises 2…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Video Safety"
  - "LVLM Evaluation"
  - "Attack Success Rate"
  - "Multimodal Safety Benchmark"
  - "RJScore"
date: 2026-05-08
content_hash: c6caf255fa4d3ba1
---

# Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs

**Conference**: NeurIPS 2025
**arXiv**: [2505.11842](https://arxiv.org/abs/2505.11842)  
**Code**: [https://liuxuannan.github.io/Video-SafetyBench.github.io/](https://liuxuannan.github.io/Video-SafetyBench.github.io/)  
**Area**: Multimodal / VLM Safety
**Keywords**: Video Safety, LVLM Evaluation, Attack Success Rate, Multimodal Safety Benchmark, RJScore

## TL;DR

This paper presents Video-SafetyBench, the first comprehensive benchmark for safety evaluation of video LVLMs. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, constructed via a controllable video generation pipeline. A confidence-based evaluation metric, RJScore, is proposed to assess model outputs. Large-scale evaluation across 24 LVLMs reveals an average attack success rate of 67.2% under benign queries.

## Background & Motivation

As large vision-language models (LVLMs) are increasingly deployed in real-world settings, systematic safety evaluation becomes critical. However, significant gaps remain in existing multimodal safety assessment:

**Existing benchmarks focus on static images**: Works such as FigStep, MM-SafetyBench, HADES, and VLSBench exclusively consider image-text inputs, overlooking the unique safety risks introduced by the temporal dynamics of video (e.g., harmful actions that evolve over time).

**Video inputs expand the attack surface**: Compared to single-frame images, continuous frame sequences in video pose greater challenges for safety alignment, as adversaries can exploit temporal information to circumvent safety mechanisms.

**Evaluation metrics are insufficient for boundary cases**: Existing automated judges have limited capability in handling uncertain or borderline harmful outputs, lacking calibration mechanisms aligned with human judgment.

**Key Challenge**: The safety risks of video LVLMs are increasingly prominent, yet no systematic video-text attack benchmark or reliable safety evaluation methodology exists.

**Key Insight**: (1) Construct compositional video-text attack tasks encompassing both harmful queries with explicit malicious intent and benign queries that implicitly convey malice through video context; (2) Design a controllable video generation pipeline to ensure semantic alignment between video content and harmful intent; (3) Propose RJScore, an LLM confidence-based evaluation metric with calibrated decision thresholds.

## Method

### Overall Architecture

Video-SafetyBench consists of three main components: (1) a safety taxonomy covering 13 primary categories and 48 subcategories; (2) a three-stage controllable video generation pipeline; and (3) the RJScore evaluation metric based on LLM token-level confidence. Each video is paired with both a harmful query and a benign query variant.

### Key Designs

1. **Two-Level Safety Taxonomy**:

    - Function: Defines a systematic hierarchy of video safety risks.
    - Mechanism: 13 primary categories (violent crime, non-violent crime, sexual crime, child sexual exploitation, defamation, professional advice, privacy, intellectual property, weapons of mass destruction, hate speech, self-harm/suicide, sexual content, elections) and 48 fine-grained subcategories.
    - Design Motivation: Adapted from existing LLM safety taxonomies and extended for video-specific scenarios to ensure comprehensive coverage.

2. **Three-Stage Controllable Video Generation Pipeline**:

    - Function: Synthesizes videos semantically aligned with harmful intent.
    - Mechanism: Video semantics are decomposed into "what to show" (subject image) and "how to move" (motion text), executed in three steps:
        - **Stage 1 (Text)**: Harmful queries are generated based on safety policies and then rewritten into benign variants via LLM (replacing harmful phrases with video-referential expressions, e.g., "high-explosive device" → "the device shown in the video").
        - **Stage 2 (Text → Image)**: An LLM (GPT-4o) transforms abstract queries into rich scene descriptions, which are then fed to a T2I model (Midjourney-V6, KLING 1.5) to generate subject images.
        - **Stage 3 (Image + Text → Video)**: An LVLM infers motion trajectory descriptions; combined with the subject image, these are input to an I2V model (KLING 1.6, Sora, Jimeng) to generate 10-second videos.
    - Design Motivation: Direct T2V generation lacks precise semantic control. Decomposing into subject and motion dimensions substantially improves controllability. The resulting dataset achieves FID=73 (video quality) and VQAScore=0.522 (text-video alignment), both superior to existing benchmarks.

3. **RJScore Evaluation Metric**:

    - Function: Quantifies the harmfulness of model outputs and aligns judgment with human annotations.
    - Mechanism: Qwen2.5-72B assigns a 5-level toxicity score to each output. Rather than taking the argmax prediction, the logit values of the 5 candidate tokens are converted to softmax probabilities, and the expected score is computed as $RJScore = \sum_{k=1}^{5} k \cdot p(k)$. A decision threshold of $\tau=2.85$ is calibrated via 5-fold cross-validation.
    - Design Motivation: Binary classification fails to handle uncertain and borderline cases. Token-level logit distributions capture judgment confidence, and threshold calibration yields 91% agreement with human annotations, surpassing GPT-4o (88.2%).

### Attack Modalities

- **Harmful Query Attack**: Text explicitly expresses malicious intent, with video amplifying the harmful effect.
- **Benign Query Attack**: Text itself is innocuous but implicitly conveys malice through video references (e.g., "drop the device shown in the video" instead of "throw a high-explosive device").

## Key Experimental Results

### Main Results

Attack success rate (ASR) evaluated across 24 LVLMs (7 closed-source + 17 open-source):

| Model | Harmful Query ASR | Benign Query ASR | Notes |
|-------|-------------------|------------------|-------|
| Qwen-VL-Max | 25.4% | **78.3%** | Benign ASR higher by 52.9% |
| GPT-4o | 14.8% | 43.3% | Safest among closed-source |
| Claude 3.5 Sonnet | 7.8% | 19.9% | Safest overall |
| Qwen2-VL-72B | 44.6% | 83.3% | Most vulnerable among open-source 72B |
| Qwen2.5-VL-7B | — | 68.7% | 7B safer than 72B |
| Qwen2.5-VL-72B | — | 74.0% | Larger model ≠ safer |
| **Average (all models)** | 39.1% | **67.2%** | Benign queries higher by 28.1% |

### Ablation Study

| Judge Model | Human Agreement | F1 | FPR | FNR |
|-------------|----------------|----|-----|-----|
| Rule-based | 76.5% | 75.1% | 46.4% | 0.7% |
| HarmBench | 77.1% | 76.1% | 2.7% | 43.0% |
| Llama Guard 3 | 79.5% | 79.4% | 12.2% | 28.6% |
| GPT-4o | 88.2% | 88.1% | 19.7% | 3.9% |
| Qwen-2.5-72B | 88.4% | 88.3% | 18.4% | 4.7% |
| **RJScore (τ=2.85)** | **91.0%** | **91.0%** | **12.3%** | 5.8% |

### Key Findings

- **Benign queries are far more dangerous than harmful queries**: The average ASR under benign queries exceeds that under harmful queries by 28.1%, indicating that models struggle to detect malicious intent implicitly conveyed through video references.
- **Model scale does not imply greater safety**: Within the Qwen2.5-VL series, benign-query ASR for 7B/32B/72B models is 68.7%/73.2%/74.0%, respectively — larger models are actually more susceptible.
- **Video inputs are more dangerous than static images**: Video-based ASR is on average 8.6% higher than image-based ASR, with temporal information introducing additional risk.
- **Professional advice (S6-SA) is the most vulnerable category**: Nearly all models exhibit ASR exceeding 70% in this category, particularly under benign queries.

## Highlights & Insights

- This is the first systematic study of video-text compositional attacks on LVLM safety, filling a critical gap in video safety evaluation.
- The subject-plus-motion decomposition in the controllable video generation pipeline is elegant and effective, ensuring semantic alignment between video content and harmful intent.
- The RJScore approach — combining token-level logit confidence with cross-validation threshold calibration — provides a more reliable paradigm for LLM-as-Judge evaluation.
- The high ASR of benign query attacks reveals a fundamental weakness of LVLMs in reasoning about implicitly conveyed malicious intent.

## Limitations & Future Work

- The benchmark currently relies solely on synthetic videos; attack effectiveness on real-world videos may differ.
- The benign query rewriting strategy is relatively simple (template-based substitution); more sophisticated semantic obfuscation techniques warrant exploration.
- Audio modality safety risks are not addressed.
- Evaluation focuses exclusively on output harmfulness, without considering over-refusal (excessive defense against legitimate requests).

## Related Work & Insights

- This work provides strong directional guidance for extending existing image safety benchmarks (MM-SafetyBench, VLSBench, etc.) to the video domain.
- The confidence quantification methodology underlying RJScore is transferable to other LLM-as-Judge scenarios.
- The benign query attack paradigm is conceptually consistent with VLSBench's "benign text + harmful image" approach, but poses greater risks in the video domain.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] Learning Skill-Attributes for Transferable Assessment in Video](learning_skill-attributes_for_transferable_assessment_in_video.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)

</div>

<!-- RELATED:END -->
