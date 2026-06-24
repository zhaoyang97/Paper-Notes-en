---
title: >-
  [Paper Note] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors
description: >-
  [ACL 2025][Audio & Speech][Multilingual audio-text retrieval] This paper theoretically analyzes that the root cause of cross-lingual inconsistency in Multilingual Audio-Text Retrieval (ML-ATR) is the training data distribution error. It proposes two strategies, namely 1-to-K Contrastive Learning (KCL) and Audio-English Common-Anchor Contrastive Learning (CACL), to reduce this error, achieving SOTA performance in both recall and consistency.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Multilingual audio-text retrieval"
  - "cross-lingual consistency"
  - "contrastive learning"
  - "modality alignment"
  - "data distribution error"
date: 2026-05-08
content_hash: 914d18d6f55cfe88
---

# ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors

**Conference**: ACL 2025  
**arXiv**: [2502.14627](https://arxiv.org/abs/2502.14627)  
**Code**: [github.com/ATRI-ACL/ATRI-ACL](https://github.com/ATRI-ACL/ATRI-ACL)  
**Area**: Audio/Speech  
**Keywords**: Multilingual audio-text retrieval, cross-lingual consistency, contrastive learning, modality alignment, data distribution error

## TL;DR

This paper theoretically analyzes that the root cause of cross-lingual inconsistency in Multilingual Audio-Text Retrieval (ML-ATR) is the training data distribution error. It proposes two strategies, namely 1-to-K Contrastive Learning (KCL) and Audio-English Common-Anchor Contrastive Learning (CACL), to reduce this error, achieving SOTA performance in both recall and consistency.

## Background & Motivation

Audio-Text Retrieval (ATR) aims to search for matching audio segments or text descriptions in a database based on cross-modal queries. While English monolingual ATR performance continues to improve, research on Multilingual Audio-Text Retrieval (ML-ATR) remains limited and faces two core challenges:

**Inadequate Multilingual Recall**: Existing ML-ATR solutions randomly select text from one language per epoch to pair with audio for training. This prevents the model from fully learning the embedding space relationships between audio and multilingual text.

**Inconsistent Cross-Lingual Retrieval Results**: When querying the same audio with different languages, the retrieval rankings differ significantly. For example, querying the same sound with English and French descriptions might yield completely different ranking results.

The training method of existing ML-CLAP solutions essentially samples one language randomly per epoch for contrastive learning, which not only degrades retrieval recall but also causes retrieval inconsistency issues. The authors are the first to theoretically analyze the root cause of this problem.

## Method

### Overall Architecture

The core idea of the ATRI framework is to theoretically prove that the root cause of inconsistency is the data distribution error, and then design two strategies to reduce this error. The framework uses CED-Base as the audio encoder and SONAR as the multilingual text encoder.

### Key Designs

1. **Theoretical Analysis — Derivation of Weight Error Upper Bound**: The authors first visualize the inconsistency problem from the perspective of modality alignment direction error. Ideally, the audio embedding should align with the arithmetic mean direction of the multilingual text embeddings (green arrow). However, random sampling forces the audio to align only with a single language text (red arrow), and the angle between them represents the modality alignment direction error. They further derive the weight error upper bound formula:

    $$\|\mathbf{w}_{eT} - \mathbf{w}'_{eT}\| \leq a^T\|\mathbf{w}_{(e-1)T} - \mathbf{w}'_{(e-1)T}\| + \eta\sum_{(a,t)}\|p(a,t) - p'_e(a,t)\|\cdot(\text{gradient-related term})$$

   Expanding it reveals that the root cause of weight error stems entirely from the data distribution error $\sum\|p(a,t) - p'_i(a,t)\|$ across epochs.

2. **1-to-K Contrastive Learning (KCL)**: Multilingual texts of all $K$ languages are paired with the audio simultaneously in every epoch to perform contrastive learning, theoretically eliminating the data distribution error. The loss function covers both audio-to-text and text-to-audio directions, calculating contrastive losses for each language independently before summing them up. The drawback is that GPU VRAM consumption scales linearly with the number of languages $K$.

3. **Audio-English Common-Anchor Contrastive Learning (CACL)**: To address the high VRAM overhead of KCL, CACL is proposed as a lightweight alternative. For each data instance, a triplet of (audio, English text, random other language text) is sampled for three groups of contrastive learning:

    - Audio-English alignment $\mathcal{L}^{ae}_{cacl}$
    - Audio-multilingual alignment $\mathcal{L}^{at}_{cacl}$
    - English-multilingual alignment $\mathcal{L}^{et}_{cacl}$ 
   
   The effectiveness of CACL can be understood from two perspectives: (a) English-multilingual alignment narrows the gap between different language embeddings, reducing the modality alignment direction bias; (b) more audio-text pairs (including high-quality English texts) are trained in each epoch, making the empirical data distribution closer to the theoretically optimal distribution. Its chief advantage is that VRAM overhead does not scale with the number of languages.

### Loss & Training

- KCL Loss: $\mathcal{L}_{kcl} = \frac{1}{2NK}(\mathcal{L}^{a2t}_{kcl} + \mathcal{L}^{t2a}_{kcl})$
- CACL Loss: $\mathcal{L}_{cacl} = \frac{1}{6N}(\mathcal{L}^{ae}_{cacl} + \mathcal{L}^{at}_{cacl} + \mathcal{L}^{et}_{cacl})$
- Initialized with ML-CLAP pre-trained weights and fine-tuned for 10 epochs on cross-translated multilingual AudioCaps and Clotho datasets.
- Batch size is 24, learning rate is $5\times 10^{-6}$, temperature parameter is $\tau = 0.07$, using the Adam optimizer.
- Trained on a single A100 80GB GPU.

## Key Experimental Results

### Main Results

Average T2A R@1 performance on the AudioCaps dataset (8 languages):

| Ours | T2A R@1 (avg) | A2T R@1 (avg) | Gain over ML-CLAP |
|------|-------------|-------------|-----------------|
| ML-CLAP | 44.84 | 61.19 | - |
| CACL | 46.03 (+1.19) | 62.28 (+1.09) | Consistent recall gains |
| KCL | **46.81** (+1.97) | **62.91** (+1.72) | SOTA, R@1 improved by ~2% |

English monolingual ATR results (AudioCaps T2A R@1):

| Ours | R@1 | R@5 | mAP10 |
|------|-----|-----|-------|
| ML-CLAP | 47.31 | 80.65 | 61.44 |
| CACL | 49.05 | 82.14 | 63.07 |
| KCL | **49.68** (+5%) | **82.44** | **63.34** |

### Consistency Evaluation

| Ours | AudioCaps MRV↓ | Clotho MRV↓ | Notes |
|------|---------------|-------------|------|
| ML-CLAP | High | High | Poor cross-lingual consistency |
| CACL | Reduced | Reduced | Narrowed embedding space gap and distance |
| KCL | **Lowest** | **Lowest** | Best consistency |

### Key Findings

- **Consistency between theory and experiment**: KCL completely eliminates data distribution error, achieving the best performance; CACL reduces distribution error, yielding second-best performance. Both outperform the randomly-sampled ML-CLAP.
- **KCL > CACL > ML-CLAP**: KCL continuously leads across the vast majority of languages and metrics.
- **Performance gaps between languages**: Metrics for Japanese and Chinese are relatively low, owing to their larger syntactic differences compared to other languages.
- **Practical value of CACL**: While offering performance close to KCL, its VRAM and runtime overheads are comparable to ML-CLAP, making it a better option for practical deployments.
- **Occasional anomalies**: On very few metrics, KCL performs lower than CACL, which is attributed to dataset noise (more commonly found in Clotho).

## Highlights & Insights

1. **Theory-driven method design**: Starting from the derivation of the weight error upper bound, the paper uncovers that the data distribution error is the fundamental cause of inconsistency, followed by a targeted solution design, building a rigorous logical chain.
2. **Exact correspondence between theory and experiment**: KCL (eliminating error) > CACL (reducing error) > ML-CLAP (random), perfectly validating the theoretical predictions.
3. **Practical considerations**: It provides two schemes for different scenarios — KCL for performance-first contexts, and CACL for resource-limited ones.
4. **Clever design of using English as an anchor**: Since English is typically the source language of translations and of the highest quality, utilizing it as a common anchor is a natural and effective choice.

## Limitations & Future Work

- Only validated on translated datasets (AudioCaps and Clotho), lacking evaluation on native multilingual data.
- Translation quality may affect results, especially for languages with major syntactic differences like Japanese and Chinese.
- The coverage of 8 languages is limited; performance on low-resource languages remains unknown.
- The assumption of selecting English as the anchor in CACL might not hold for all scenarios.
- This work only explores the SONAR text encoder; the efficacy of other multilingual encoders (e.g., mBERT) remains unverified.

## Related Work & Insights

- **Relationship with ML-CLAP**: Improves directly upon ML-CLAP, overcoming its inherent random language sampling limitations.
- **CLIP inspiration**: Builds on CLIP-style contrastive learning frameworks and extends them to multilingual multi-modal scenarios.
- **Insights for other fields**: The analytical approach showing how data distribution errors bias model weights away from the optimum can be generalized to other multilingual multi-modal tasks (e.g., multilingual vision-language pre-training).
- **Generality of the common-anchor strategy**: Aligning low-quality modalities by anchoring them to high-quality ones (e.g., English text) can be applied to other multi-modal alignment scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The theoretical perspective is novel, and the derivation of the weight error upper bound provides a solid foundation for the methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 8 languages, 2 datasets, multiple metrics, and detailed consistency analysis, though lacking native multilingual data.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations, detailed empirical analysis, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Makes substantial contributions to the multilingual audio retrieval field, and the theoretical findings offer broader general insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](audio_token_consistency.md)
- [\[ACL 2025\] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)

</div>

<!-- RELATED:END -->
