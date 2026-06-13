---
title: >-
  [Paper Note] VGGSounder: Audio-Visual Evaluations for Foundation Models
description: >-
  [ICCV 2025][Audio & Speech][Audio-visual classification] To address the limitations of the VGGSound dataset — including missing multi-labels, category overlap, and modality misalignment — this work constructs VGGSounder…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "Audio-visual classification"
  - "multi-label benchmark"
  - "modality annotation"
  - "foundation model evaluation"
  - "VGGSound"
date: 2026-05-08
content_hash: d00e1587d8b8a8d8
---

# VGGSounder: Audio-Visual Evaluations for Foundation Models

**Conference**: ICCV 2025
**arXiv**: [2508.08237](https://arxiv.org/abs/2508.08237)  
**Code**: [Project Page](https://vggsounder.github.io/)  
**Area**: Audio-Visual Learning / Benchmark Evaluation
**Keywords**: Audio-visual classification, multi-label benchmark, modality annotation, foundation model evaluation, VGGSound

## TL;DR

To address the limitations of the VGGSound dataset — including missing multi-labels, category overlap, and modality misalignment — this work constructs VGGSounder, a multi-label audio-visual classification benchmark with modality-level annotations, and proposes a "modality confusion" metric to expose deficiencies in foundation models' multimodal fusion capabilities.

## Background & Motivation

VGGSound is the most widely used benchmark dataset in audio-visual classification, comprising approximately 200,000 10-second video clips across 309 categories. However, as audio-visual foundation models have rapidly advanced, the limitations of VGGSound have become increasingly apparent:

**Incomplete annotations**: VGGSound assigns only a single label per sample, yet the vast majority of videos contain multiple simultaneous categories (e.g., a band performance may include multiple instruments). This causes models to be penalized for correctly predicting valid additional categories.

**Category overlap**: Among the 309 automatically generated categories, there exist synonymous classes (e.g., "timpani" and "tympany"), sub-class/super-class relationships (e.g., "male speech" and "people speaking"), and frequently co-occurring classes (e.g., "playing drums" and "playing drum kit").

**Modality misalignment**: Despite VGGSound's claim of both visual and auditory verification, approximately **48.43%** of test samples exhibit modality misalignment — the annotated category is either inaudible (e.g., a static image with background music) or invisible (e.g., content only mentioned in a voiceover narration). This challenges the widely held assumption that VGGSound is strongly modality-aligned.

**Core Motivation**: The absence of modality-aware multi-label annotations makes it impossible to accurately evaluate the audio and visual capabilities of foundation models — a model that correctly predicts the audible category is penalized if that category is absent from the original single label. This systematic underestimation particularly affects foundation models used in a zero-shot setting.

## Method

### Overall Architecture

The construction pipeline of VGGSounder consists of: proposal generation → human annotation → automatic completion → label merging.

### Key Designs

1. **Gold-standard reference set**: Four computer vision experts manually annotated 417 randomly selected samples, ensuring at least one sample per category was covered. Annotations were merged via majority voting. This reference set is used to evaluate the recall of automated proposal strategies.

2. **Label proposal generation**: Top-$k$ predictions from multiple state-of-the-art models (CAV-MAE, AV-Siam, Equi-AV, DeepAVFusion, Gemini 1.5 Flash/Pro) are combined. Predictions from different modalities (audio/visual/audio-visual) and different values of $k$ are aggregated. High-frequency categories (e.g., speech, bird sounds) are forced into proposals. The final strategy achieves a **93% recall rate**, with approximately 30 proposals per sample on average.

3. **Human annotation (Amazon Mechanical Turk)**: For each sample, annotators answer:

    - Whether the clip contains background music / voiceover narration / static images (meta-labels)
    - Whether each proposed category is **audible** and/or **visible**
    - Whether any categories have been missed

   Each batch of 20 samples includes 2 gold-standard quality control samples; batches with F1 below 25% are rejected and re-annotated.

4. **Automatic completion**: Super-classes are automatically added when a sub-class is detected (e.g., detecting "eagle screaming" → automatically adding "bird squawking"); synonymous classes are mutually completed.

5. **Modality Confusion metric ($\mu$)**: Measures the proportion of samples for which a model produces correct predictions under a single modality but incorrect predictions upon receiving an additional modality:

$$\mu_M = 100 \cdot \frac{\sum_{x \in M} \mathcal{I}[a(x)\text{-correct} \cap av(x)\text{-wrong}]}{N_{total}}$$

This metric reveals that **multimodal input is not always beneficial** — models can be misled by the additional modality.

### Loss & Training

VGGSounder is an evaluation benchmark and does not involve model training. Evaluation employs multi-label classification metrics (Subset Accuracy, F1, Hit). For embedding models, top-$k$ predictions are used; for foundation models, LLM-assisted evaluation is employed (Qwen-3 judges the match between model outputs and target categories).

## Key Experimental Results

### Main Results

VGGSounder evaluation results (F1 scores) for 11 audio-visual models:

| Model | a(A) | v(V) | av(AV) | $\mu_A$↓ | $\mu_V$↓ |
|-------|:---:|:---:|:---:|:---:|:---:|
| CAV-MAE | 34.46 | 34.91 | 42.62 | 3.96 | 6.01 |
| AV-Siam | 33.30 | 35.41 | 39.43 | 10.30 | 8.69 |
| Gemini 1.5 Pro | 19.26 | 49.73 | 53.74 | 3.07 | 4.23 |
| VideoLLaMA 2 | 38.87 | 47.82 | 52.35 | 14.34 | 5.43 |
| Ola | 47.70 | 24.85 | 46.48 | 17.07 | 6.32 |

- Embedding models generally outperform on audio inputs compared to visual inputs, whereas foundation models exhibit the **opposite trend** — a systematic bias toward visual information.
- All models exhibit significant modality confusion ($\mu$), with 4–17% of samples being predicted incorrectly after the addition of an extra modality.

### Ablation Study (Meta-label Analysis)

Effect of different meta-labels on F1 scores ($\Delta$ F1):

| Condition | Embed. (Audio) | Found. (Audio) | Embed. (Visual) | Found. (Visual) |
|-----------|:---:|:---:|:---:|:---:|
| Background music present | −3.4~−4.1 | −0.5~−11.8 | −2.5~−4.9 | −0.9~−4.6 |
| Voiceover narration present | −7.1~−9.1 | −4.0~+18.2 | +4.3~+5.2 | −3.7~−8.2 |
| Static image present | +15.9~+22.1 | +1.4~+19.0 | +10.4~+19.8 | +4.2~+22.1 |
| No meta-labels | — | — | — | — |

- Background music poses a challenge for all models.
- Foundation models are more robust to voiceover narration (some even show performance gains), whereas embedding models are severely affected.
- Static images degrade visual classification performance but improve audio classification (audio classification is more accurate in the absence of visual distractors).

### Key Findings

- Performance on VGGSounder is substantially higher than on VGGSound (due to the elimination of false negatives through multi-labeling), with average gaps of 15–29%.
- Human annotation contributes far more than automatic completion (Hit improvement of 8–28% vs. 0.2–2.1%).
- The Gemini series performs extremely poorly on audio-only input, indicating that these models **rely almost entirely on the visual modality**.
- The performance gap between specialized embedding models and general-purpose foundation models has largely closed.

## Highlights & Insights

- **The modality confusion metric is the core contribution**: it is the first quantification of the phenomenon that "multimodal input can perform worse than unimodal input," revealing a fundamental deficiency in current models' modality fusion.
- **The introduction of modality annotations is critical**: it enables, for the first time, separate evaluation of model performance on samples that are "audible but not visible" and "visible but not audible."
- **Methodological transferability**: the pipeline of label proposal + human verification + automatic completion can be generalized to the construction of other multi-label benchmarks.
- The work uncovers a systematic divergence: foundation models are biased toward visual information, while embedding models are biased toward audio.

## Limitations & Future Work

- The completeness of multi-label annotation still depends on the recall of the proposal generation stage (93%), and rare categories may be missed.
- Only the test set (15,446 samples) has been re-annotated; the training set likely suffers from the same issues.
- Meta-labels cover only three conditions (background music / voiceover / static images); other sources of interference are not considered.
- LLM-assisted evaluation may introduce additional bias.
- The category set is fixed at VGGSound's 309 classes, precluding evaluation of open-vocabulary capabilities.

## Related Work & Insights

- Inspired by ImageNet re-labeling (Beyer et al., 2020) and MMLU correction (Gema et al., 2024).
- Complementary to AudioSet's multi-label design, with the addition of modality-level annotations.
- The concept of the modality confusion metric can be extended to the evaluation of any multimodal fusion system.
- Provides an important diagnostic tool for the training strategies of future audio-visual foundation models.

## Rating

- **Novelty**: 7/10 — Benchmark reconstruction is not a new methodology per se, but the modality confusion metric is a genuine new contribution.
- **Technical Quality**: 8/10 — The annotation pipeline is rigorous and the evaluation is comprehensive, covering 11 models.
- **Practicality**: 9/10 — A directly usable benchmark that reveals the true capabilities of foundation models.
- **Writing Quality**: 8/10 — Well-structured with rich visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](../../NeurIPS2025/audio_speech/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](../../NeurIPS2025/audio_speech/e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)

</div>

<!-- RELATED:END -->
