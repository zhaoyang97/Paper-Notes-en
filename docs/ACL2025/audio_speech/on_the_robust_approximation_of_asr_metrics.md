---
title: >-
  [Paper Note] On the Robust Approximation of ASR Metrics
description: >-
  [ACL 2025][Audio & Speech][ASR evaluation] This paper proposes a label-free approximation method for ASR performance metrics. By utilizing speech-text similarity in a unified multimodal embedding space and proxy metrics from high-quality proxy models, an ensemble regression model is trained to predict WER/CER. The absolute error is maintained within single digits across over 40 models and 14 datasets, outperforming the latest baseline by more than 50%.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "ASR evaluation"
  - "label-free metrics"
  - "WER approximation"
  - "multimodal embeddings"
  - "proxy reference"
date: 2026-05-08
content_hash: d58dc64eae520f0f
---

# On the Robust Approximation of ASR Metrics

**Conference**: ACL 2025  
**arXiv**: [2502.12408](https://arxiv.org/abs/2502.12408)  
**Code**: None  
**Area**: Speech  
**Keywords**: ASR evaluation, label-free metrics, WER approximation, multimodal embeddings, proxy reference

## TL;DR

This paper proposes a label-free approximation method for ASR performance metrics. By utilizing speech-text similarity in a unified multimodal embedding space and proxy metrics from high-quality proxy models, an ensemble regression model is trained to predict WER/CER. The absolute error is maintained within single digits across over 40 models and 14 datasets, outperforming the latest baseline by more than 50%.

## Background & Motivation

**Background**: ASR models are typically evaluated using WER and CER, which rely heavily on ground-truth transcriptions. While large-scale speech foundation models perform excellently on standard benchmarks, their generalization capabilities under diverse domains and testing conditions remain unclear.

**Limitations of Prior Work**: 
   - Annotating data is expensive and time-consuming, limiting the evaluation of model performance in new domains.
   - Existing reference-free evaluation methods (such as NoRefER) mainly provide relative quality assessment and cannot provide precise error rates.
   - Existing metric approximation methods (such as eWER3) are primarily validated in IID settings, lacking evaluations of OOD generalization.

**Key Challenge**: The need to obtain reliable quantitative indicators of ASR performance without labels, while maintaining robustness across out-of-distribution (OOD) data and different ASR models.

**Goal**: To achieve robust ASR metric approximation across four evaluation scenarios (IID-Source, IID-Target, OOD-Source, and OOD-Target), covering over 40 models and 14 datasets.

**Key Insight**: Combining speech-text similarity from the SONAR multimodal unified embedding space with the WER/CER of high-quality proxy models as features to train an ensemble regression model.

**Core Idea**: Using the cosine similarity of speech-transcription in a unified representation space and the error rates of proxy models as features to train a regression model that predicts the ground-truth WER/CER.

## Method

### Overall Architecture

The pipeline consists of three components:
1. Similarity calculation within a unified representation space.
2. Consistency measurement against proxy references.
3. Training regression models to predict ASR metrics.

### Key Designs

1. **Similarity in a Unified Representation Space**: Using the SONAR model, the speech signal $x_{\text{speech}}$ and the ASR transcription $x_{\text{text}}$ are mapped into a shared 1024-dimensional embedding space to compute the cosine similarity:

$$\text{Similarity}(x_{\text{speech}}, x_{\text{text}}) = \frac{e_{\text{speech}} \cdot e_{\text{text}}}{\|e_{\text{speech}}\| \|e_{\text{text}}\|}$$

Intuition: Higher similarity indicates better transcription quality and superior alignment with the actual content.

2. **Proxy Reference Mechanism**: A high-quality ASR model is selected as a proxy to compute the WER (pWER) and CER (pCER) between the target model's transcription and the proxy's transcription as features. The proxy is selected dynamically: 41 models are ranked based on their average performance on the datasets, and for each target model, the top-ranked model that is not itself is selected as the proxy.

3. **Ensemble Regression Model**: The similarity and proxy metrics are concatenated into a feature vector $z = [\text{Similarity}, \text{pWER}/\text{pCER}]$, which is input into ensemble regressors to predict aWER/aCER. The ensemble includes Random Forest, Gradient Boosting, Histogram-based Gradient Boosting, and Ridge Regression with non-negativity constraints. Hyperparameters are tuned using RandomizedSearchCV to minimize MAE.

### Four Evaluation Settings

- **Case 1**: IID data + Source model — trained on $\mathcal{D}_{S,B}^{\text{train}}$ and tested on $\mathcal{D}_{S,B}^{\text{test-IID}}$
- **Case 2**: IID data + Target model — tested on $\mathcal{D}_{T,B}^{\text{test-IID}}$
- **Case 3**: OOD data + Source model — tested on $\mathcal{D}_{S,W}$ (wild dataset)
- **Case 4**: OOD data + Target model — tested on $\mathcal{D}_{T,W}$

### Loss & Training

- The regression models minimize MAE (Mean Absolute Error).
- Leave-one-out strategy: Training on 9 out of 10 standard benchmarks and testing on the remaining 1.
- The regression target is the absolute number of errors (word/character level), which is normalized to obtain aWER/aCER.
- Extraction of SONAR embeddings for 1,000 samples per dataset takes only about 1 minute.

## Key Experimental Results

### Main Results — WER Approximation on Wild Datasets (Selected Models)

| Model | LS_Noise (WER/aWER) | Primock57 | ATCOsim | VP_Acc |
|------|--------|-----------|---------|--------|
| canary-1b | 4.1/6.4 | 16.2/13.4 | 30.4/35.5 | 23.2/12.1 |
| whisper-l-v3 | 4.6/5.9 | 18.7/12.0 | 64.7/73.9 | 19.2/18.1 |
| parakeet-tdt-1.1b | 3.4/6.0 | 13.5/13.2 | 28.3/35.7 | 17.9/10.2 |
| data2vec-large | 7.2/8.6 | 28.3/30.7 | 44.0/51.1 | 21.4/26.5 |
| mms-1b-f102 | 24.0/24.9 | 70.2/67.8 | 93.4/99.0 | 39.4/38.2 |

### Key Findings in Benchmark Datasets

- Whisper-large-v3 achieves a WER of 19.0% vs aWER of 17.1% on AMI_IHM, with a gap of only 1.9%.
- Approximations for high-performance models (low WER) are more accurate, while low-performance models exhibit larger deviation, though still remaining within a reasonable range.
- For most models on most datasets, |WER - aWER| < 5%.

### Cross-Lingual Experiments

| Training Language→Testing Language | MAE (word error count) |
|---|---|
| EN→EN (IID) | 0.56-0.66 |
| EN→DE (OOD) | 0.75-1.60 |
| DE→EN (OOD) | 0.50-1.59 |

### Ablation Study

- **Similarity only**: Performance drops significantly, indicating that proxy metrics contribute critical information.
- **Proxy metrics only**: Outperforms Similarity-only, but the combination of both yields optimal performance.
- **Training Data Size**: Training with only 20% of the data can achieve over 90% of the full-data performance.
- **vs. eWER3 Baseline**: Outperforms eWER3 by more than **50%** in all settings.

### Key Findings

- High-performance ASR models (low WER) are easier to approximate accurately, while low-performance models show larger deviation, but the absolute gap still remains in the single-digits.
- The method is agnostic to both models and data, showing strong generalizability across models and domains.
- It maintains reasonable accuracy even cross-linguistically (EN↔DE).
- OOD generalization performance on wild datasets (real-world scenarios) is highly satisfactory.
- The approximation deviation increases at extremely high error rates (e.g., WER > 90%), but distinguishing high from extremely high error rates has limited practical significance in such cases.

## Highlights & Insights

- **High Practical Value**: Evaluates ASR model performance in new domains without labels, which is highly useful for domain adaptation assessment before large-scale model deployment.
- **Ingenious Use of SONAR**: Leverages the pre-trained multimodal unified embedding space as a zero-shot feature extractor, avoiding end-to-end training.
- **Flexible Proxy Selection Strategy**: Dynamically ranks and selects the best proxy to avoid self-referential bias.
- **Impressive Scale**: 40+ models $\times$ 14 datasets $\times$ 4 evaluation settings = extremely comprehensive evaluation.

## Limitations & Future Work

1. The approximation accuracy for models with extremely high error rates (WER > 80%) is lower, which may require non-linear features.
2. The selection of proxy models depends on the pre-evaluation of multiple models, leading to a high initial setup cost.
3. The regression models are based on traditional ML (RandomForest/GBT), leaving neural network-based regressors unexplored.
4. Mostly validated on English data, with cross-lingual experiments only involving German.
5. Sentence-level approximations might be less stable than corpus-level ones.

## Related Work & Insights

- The capabilities of the SONAR (Duquenne et al., 2023) multimodal unified embedding model extend beyond translation evaluation, proving equally effective in ASR quality assessment.
- eWER3 (Chowdhury and Ali, 2023) uses an end-to-end approach based on wav2vec2 + RoBERTa but lacks proxy information and OOD generalization.
- Potential application in pseudo-labeling: Helps filter high-quality transcriptions for knowledge distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of Proxy + SONAR similarity is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 40+ models $\times$ 14 datasets $\times$ 4 settings, extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework and rigorous evaluation setup.
- **Value**: ⭐⭐⭐⭐ — High practical value for the area of label-free ASR evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion](double_entendre_robust_audio-based_ai-generated_lyrics_detection_via_multi-view_.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](gigaspeech2_low_resource_asr.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation](../../ACL2026/audio_speech/sn-wer_script-normalized_wer_for_multi-script_indic_asr_evaluation.md)

</div>

<!-- RELATED:END -->
