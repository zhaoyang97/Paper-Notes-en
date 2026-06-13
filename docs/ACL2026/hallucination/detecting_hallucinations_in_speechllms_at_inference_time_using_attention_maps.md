---
title: >-
  [Paper Note] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps
description: >-
  [ACL 2026][Hallucination Detection][SpeechLLM] This paper proposes four audio-attention-based metrics (AudioRatio, AudioConsistency, AudioEntropy…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "SpeechLLM"
  - "Attention Maps"
  - "Inference-time Detection"
  - "Lightweight Classifier"
date: 2026-05-08
content_hash: 035f65ee67f090ca
---

# Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19565](https://arxiv.org/abs/2604.19565)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: SpeechLLM, Hallucination Detection, Attention Maps, Inference-time Detection, Lightweight Classifier

## TL;DR

This paper proposes four audio-attention-based metrics (AudioRatio, AudioConsistency, AudioEntropy, TextEntropy) and trains a lightweight logistic regression classifier to detect SpeechLLM hallucinations at inference time, achieving up to a +0.23 PR-AUC improvement on in-domain data.

## Background & Motivation

**Background**: Speech Large Language Models (SpeechLLMs) have achieved significant progress in tasks such as Automatic Speech Recognition (ASR) and Speech-to-Text Translation (S2TT), but they still produce hallucinations—content that is fluent but mismatches the input audio.

**Limitations of Prior Work**: (1) Existing hallucination detection methods rely on comparisons with gold-standard outputs, which is costly and infeasible in deployment scenarios; (2) Hallucination detection methods developed for text LLMs cannot directly capture audio-specific signals because audio representations are much longer than text, and the alignment between input frames and output tokens differs from text-to-text generation.

**Key Challenge**: There is a need for inference-time (reference-free) detection, but the attention dynamics of the audio modality are fundamentally different from the text modality, preventing direct migration of existing methods.

**Goal**: To develop a lightweight inference-time hallucination detector utilizing the internal attention patterns of SpeechLLMs.

**Key Insight**: It is observed that pathological patterns appear in attention during hallucination generation—specifically, the degradation of the diagonal attention structure and attention falling back to the starting positions of the audio input.

**Core Idea**: Design four audio-specific attention metrics to capture hallucination-related attention patterns and train a logistic regression classifier for efficient detection.

## Method

### Overall Architecture

Inference is performed on SpeechLLMs (Qwen-2-Audio and Voxtral-3B), attention weights are extracted at each decoding step, four audio attention metrics are calculated, and these are used as feature vectors to train a logistic regression classifier for hallucination detection.

### Key Designs

1. **AudioRatio**:

    - **Function**: Measures the ratio of attention allocation between the audio input versus the auto-regressive text prefix.
    - **Mechanism**: $AR^{l,h}_t = \frac{A^{l,h}_t(\text{Audio})}{A^{l,h}_t(\text{Audio}) + A^{l,h}_t(\text{ART})}$, similar to Lookback-Lens but restricting the input side specifically to audio tokens.
    - **Design Motivation**: During hallucination generation, the model might over-focus on the auto-regressive prefix rather than the input audio.

2. **AudioConsistency**:

    - **Function**: Measures the consistency of audio attention vectors between consecutive decoding steps.
    - **Mechanism**: Calculates the Pearson correlation coefficient between audio attention vectors of adjacent decoding steps to capture fallback behavior.
    - **Design Motivation**: During hallucinations, model attention often collapses to the initial audio positions, leading to highly similar consecutive attention distributions.

3. **AudioEntropy / TextEntropy**:

    - **Function**: Measures the entropy of audio and text attention weights, respectively.
    - **Mechanism**: Entropy is calculated after re-normalizing the attention weights: $AE^{l,h}_t = H(\frac{a^{l,h,t}_{1:N}}{\sum_i a^{l,h,t}_i})$.
    - **Design Motivation**: AudioEntropy captures uncertainty regarding audio input, suitable for attention heads without clear diagonal patterns; TextEntropy captures text-side uncertainty.

### Loss & Training

A Logistic Regression classifier is used, with L2 regularization for feature ranking and L1 regularization for feature pruning (Stable Features variant). The training data consists of 40,000 samples from the VoxPopuli training set (10,000 for each of the 4 languages). Hallucination labels are automatically generated using a threshold of WER + SHS > 0.7, with a manually annotated subset used to calibrate this threshold.

## Key Experimental Results

### Main Results (Voxtral-3B, VoxPopuli In-domain)

| Method | F1 | PR-AUC | PRR@10% |
|------|-----|--------|---------|
| Mean Entropy (baseline) | 0.42 | 0.44 | 0.43 |
| Perplexity (baseline) | 0.40 | 0.41 | 0.40 |
| AudioRatio Only (LR) | 0.64 | 0.67 | 0.56 |
| Combined (LR) | 0.64 | 0.69 | 0.56 |

### Qwen-2-Audio Results

| Dataset | Method | F1 | PR-AUC |
|--------|------|-----|--------|
| VoxPopuli | Mean Entropy | 0.50 | 0.49 |
| VoxPopuli | AudioRatio (LR) | 0.56 | 0.56 |
| VoxPopuli | Combined (LR) | 0.55 | 0.58 |
| CALLHOME | Mean Entropy | 0.58 | 0.67 |
| CALLHOME | Combined (LR) | 0.41 | 0.61 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| All Features (4096) | PR-AUC 0.58 | Too many features may lead to overfitting. |
| AudioRatio Only (1024) | PR-AUC 0.56 | Performance of a single metric is close to optimal. |
| Top 75 (300 features) | PR-AUC 0.58 | Optimal in-domain performance achieved with a few heads. |
| Stable Features | Better OOD | Optimal OOD generalization with ~100 attention heads. |

### Key Findings
- Attention features significantly outperform uncertainty estimation baselines on in-domain data, with a PR-AUC increase of up to +0.23 on Voxtral-3B.
- Approximately 100 attention heads are sufficient for strong detection performance, and out-of-distribution (OOD) generalization improves when using a subset of heads compared to all heads.
- Effectiveness is model-dependent: improvements on Voxtral-3B are more significant than on Qwen-2-Audio.
- OOD generalization (e.g., to noisy CALLHOME data) remains a major challenge, though feature selection helps mitigate this.
- Hallucination rates are low (1-6%) on clean data (VoxPopuli) but reach up to 20% on noisy data (CALLHOME).

## Highlights & Insights
- This work is the first to extend attention-based hallucination detection from text LLMs to SpeechLLMs, designing metrics specific to the audio modality.
- The lightweight approach (Logistic Regression) can be deployed in real-time during inference for online filtering or offline analysis.
- Visualizations clearly demonstrate pathological attention patterns during hallucinations: diagonal degradation and attention fallback to the start of the audio.
- Feature selection is found not only to reduce computational overhead but also to enhance OOD generalization capabilities.

## Limitations & Future Work
- Effectiveness is highly dependent on the specific model and task, requiring task-specific training.
- OOD generalization is still a primary bottleneck, particularly when moving from clean data to noisy data.
- Hallucination labels rely on an automatic threshold (WER + SHS > 0.7), which may introduce label noise.
- Future directions include integration with uncertainty estimation, exploring more SpeechLLM architectures, and end-to-end training.

## Related Work & Insights
- **vs Lookback-Lens**: While Lookback-Lens calculates input/output attention ratios for text LLMs, this paper adapts the concept to the audio modality by specifically calculating the attention ratio for audio tokens.
- **vs SHALLOW**: SHALLOW is a reference-based hallucination detection benchmark; this paper proposes reference-free detection at inference time.
- **vs Uncertainty Estimation**: While uncertainty methods (Mean Entropy, Perplexity) provide general signals, the attention features in this study specifically capture audio-text alignment failures.

## Rating
- Novelty: ⭐⭐⭐ Adapts existing text hallucination detection concepts to the speech modality; innovation lies in the metric design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across two models, two tasks, and multiple datasets with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology, intuitive visualizations, and sound experimental design.
- Value: ⭐⭐⭐ High practicality but scope is currently limited by dependency on specific models and tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)
- [\[ICLR 2026\] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](../../ICLR2026/hallucination/lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)
- [\[AAAI 2026\] LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction](../../AAAI2026/hallucination/llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)

</div>

<!-- RELATED:END -->
