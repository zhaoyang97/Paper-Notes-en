---
title: >-
  [Paper Note] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps
description: >-
  [ACL 2026][Medical Imaging][Speech LLM] This work proposes four audio-attention-based metrics (AudioRatio, AudioConsistency, AudioEntropy, TextEntropy) and trains lightweight logistic regression classifiers to detect hallucinations in Speech Large Language Models (SpeechLLMs) at inference time, achieving up to +0.23 PR-AUC improvement on in-domain data.
tags:
  - ACL 2026
  - Medical Imaging
  - Speech LLM
  - hallucination detection
  - attention maps
  - inference-time detection
  - lightweight classifier
date: 2026-05-08
content_hash: ed18f1894df865ed
---

# Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps

**Conference**: ACL 2026
**arXiv**: [2604.19565](https://arxiv.org/abs/2604.19565)
**Code**: N/A
**Area**: Medical Imaging / Speech Processing
**Keywords**: Speech LLM, hallucination detection, attention maps, inference-time detection, lightweight classifier

## TL;DR

This work proposes four audio-attention-based metrics (AudioRatio, AudioConsistency, AudioEntropy, TextEntropy) and trains lightweight logistic regression classifiers to detect hallucinations in Speech Large Language Models (SpeechLLMs) at inference time, achieving up to +0.23 PR-AUC improvement on in-domain data.

## Background & Motivation

**Background**: SpeechLLMs have achieved remarkable progress on tasks such as automatic speech recognition (ASR) and speech-to-text translation (S2TT), yet they remain susceptible to hallucinations—fluent outputs that are inconsistent with the input audio.

**Limitations of Prior Work**: (1) Existing hallucination detection methods rely on gold-standard references for comparison, which is costly and infeasible in deployment settings. (2) Hallucination detection methods developed for text LLMs cannot directly capture audio-specific signals, as audio representations are substantially longer than text and the alignment between input frames and output tokens differs fundamentally from text-to-text generation.

**Key Challenge**: Hallucinations must be detected at inference time without reference transcripts, yet the attention dynamics of the audio modality are intrinsically different from those of the text modality, preventing direct transfer of existing approaches.

**Goal**: To develop lightweight inference-time hallucination detectors by leveraging internal attention patterns of SpeechLLMs.

**Key Insight**: The authors observe that hallucination generation is accompanied by pathological attention patterns—degradation of the diagonal attention structure and attention regression toward the beginning of the audio input.

**Core Idea**: Four audio-specific attention metrics are designed to capture hallucination-related attention patterns, and logistic regression classifiers are trained to enable efficient detection.

## Method

### Overall Architecture

Inference is performed on SpeechLLMs (Qwen-2-Audio and Voxtral-3B). At each decoding step, attention weights are extracted and used to compute four audio attention metrics, which serve as feature vectors for training a logistic regression classifier to detect hallucinations.

### Key Designs

1. **AudioRatio**:

    - **Function**: Measures the ratio of attention allocated to audio input versus the autoregressive text prefix.
    - **Mechanism**: $AR^{l,h}_t = \frac{A^{l,h}_t(\text{Audio})}{A^{l,h}_t(\text{Audio}) + A^{l,h}_t(\text{ART})}$, analogous to Lookback-Lens but restricted to audio tokens on the input side.
    - **Design Motivation**: During hallucination, the model may attend excessively to the autoregressive prefix rather than the input audio.

2. **AudioConsistency**:

    - **Function**: Measures the consistency of audio attention vectors across consecutive decoding steps.
    - **Mechanism**: Computes the Pearson correlation coefficient between audio attention vectors at adjacent decoding steps to capture attention regression behavior.
    - **Design Motivation**: During hallucination, model attention often collapses to the beginning of the audio input, causing consecutive attention distributions to become highly similar.

3. **AudioEntropy / TextEntropy**:

    - **Function**: Measures the entropy of audio and text attention weights, respectively.
    - **Mechanism**: Entropy is computed after re-normalizing the attention weights: $AE^{l,h}_t = H(\frac{a^{l,h,t}_{1:N}}{\sum_i a^{l,h,t}_i})$.
    - **Design Motivation**: AudioEntropy captures uncertainty over the audio input and is suitable for attention heads without a clear diagonal pattern; TextEntropy captures uncertainty on the text side.

### Loss & Training

A logistic regression classifier is employed, with L2 regularization for feature ranking and L1 regularization for feature pruning (Stable Features variant). Training data consists of 40,000 samples from the VoxPopuli training set (10,000 per language across 4 languages). Hallucination labels are automatically generated using a WER + SHS > 0.7 threshold, calibrated against a manually annotated subset.

## Key Experimental Results

### Main Results (Voxtral-3B, VoxPopuli In-Domain)

| Method | F1 | PR-AUC | PRR@10% |
|---|---|---|---|
| Mean Entropy (baseline) | 0.42 | 0.44 | 0.43 |
| Perplexity (baseline) | 0.40 | 0.41 | 0.40 |
| AudioRatio Only (LR) | 0.64 | 0.67 | 0.56 |
| Combined (LR) | 0.64 | 0.69 | 0.56 |

### Qwen-2-Audio Results

| Dataset | Method | F1 | PR-AUC |
|---|---|---|---|
| VoxPopuli | Mean Entropy | 0.50 | 0.49 |
| VoxPopuli | AudioRatio (LR) | 0.56 | 0.56 |
| VoxPopuli | Combined (LR) | 0.55 | 0.58 |
| CALLHOME | Mean Entropy | 0.58 | 0.67 |
| CALLHOME | Combined (LR) | 0.41 | 0.61 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| All features (4096) | PR-AUC 0.58 | Too many features may cause overfitting |
| AudioRatio Only (1024) | PR-AUC 0.56 | Single metric approaches optimal performance |
| Top 75 (300 features) | PR-AUC 0.58 | A small number of heads suffices for optimal in-domain performance |
| Stable Features | Better out-of-domain generalization | ~100 attention heads yield the best results |

### Key Findings
- Attention features significantly outperform uncertainty estimation baselines on in-domain data, achieving a +0.23 PR-AUC improvement on Voxtral-3B.
- Approximately 100 attention heads suffice for strong detection performance and generalize better out-of-domain than using all heads.
- The effectiveness is model-dependent: improvements are more pronounced on Voxtral-3B than on Qwen-2-Audio.
- Out-of-domain generalization (CALLHOME noisy data) remains the primary challenge; feature selection helps mitigate this.
- Hallucination rates are low on clean data (VoxPopuli: 1–6%) but reach up to 20% on noisy data (CALLHOME).

## Highlights & Insights
- This work is the first to extend attention-based hallucination detection from text LLMs to speech LLMs, introducing audio-specific metrics.
- The lightweight approach (logistic regression) supports real-time deployment at inference time and can be applied to online filtering or offline analysis.
- Visualizations clearly demonstrate pathological attention patterns during hallucination: diagonal structure degradation and attention regression to the audio onset.
- Feature selection is found not only to reduce computational cost but also to improve out-of-domain generalization.

## Limitations & Future Work
- Performance is highly model- and task-dependent, requiring task-specific training.
- Out-of-domain generalization remains the primary bottleneck, particularly from clean to noisy data.
- Hallucination labels rely on automatic thresholding (WER + SHS > 0.7), which may introduce label noise.
- Future directions include combining with uncertainty estimation, exploring additional SpeechLLM architectures, and end-to-end training.

## Related Work & Insights
- **vs. Lookback-Lens**: Lookback-Lens computes the input/output attention ratio for text LLMs; this work adapts it to the audio modality by computing the attention ratio exclusively over audio tokens.
- **vs. SHALLOW**: SHALLOW is a reference-based hallucination detection benchmark; this work proposes reference-free inference-time detection.
- **vs. Uncertainty Estimation**: Uncertainty-based methods (Mean Entropy, Perplexity) provide general signals, whereas the proposed attention features specifically capture audio–text alignment failures.

## Rating
- **Novelty**: ⭐⭐⭐ Adapts existing text hallucination detection ideas to the speech modality; the primary contribution lies in metric design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation covers two models, two tasks, and multiple datasets with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly presented, visualizations are intuitive, and the experimental design is sound.
- **Value**: ⭐⭐⭐ Practically useful but with a relatively narrow scope due to dependence on specific models and tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](../../ICLR2026/medical_imaging/driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)

<!-- RELATED:END -->
