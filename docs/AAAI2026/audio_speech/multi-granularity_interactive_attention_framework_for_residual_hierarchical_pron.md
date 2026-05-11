---
title: >-
  [Paper Note] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment
description: >-
  [AAAI2026][Audio & Speech][pronunciation assessment] This paper proposes the HIA framework, which employs an Interactive Attention Module to enable bidirectional information exchange across phoneme, word…
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "pronunciation assessment"
  - "multi-granularity interaction"
  - "attention mechanism"
  - "residual hierarchical structure"
  - "CAPT"
  - "speech scoring"
date: 2026-05-08
content_hash: 3127d78679966014
---

# Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment

**Conference**: AAAI2026  
**arXiv**: [2601.01745](https://arxiv.org/abs/2601.01745)  
**Authors**: Hong Han, Hao-Chen Pei, Zhao-Zheng Nie, Xin Luo, Xin-Shun Xu  
**Code**: Not released  
**Area**: Audio & Speech  
**Keywords**: pronunciation assessment, multi-granularity interaction, attention mechanism, residual hierarchical structure, CAPT, speech scoring  

## TL;DR

This paper proposes the HIA framework, which employs an Interactive Attention Module to enable bidirectional information exchange across phoneme, word, and utterance granularities. Combined with a residual hierarchical structure to mitigate feature forgetting, HIA achieves state-of-the-art results on the speechocean762 dataset across all granularities and aspects.

## Background & Motivation

### Importance of Automatic Pronunciation Assessment

Computer-Assisted Pronunciation Training (CAPT) systems provide learners with real-time feedback to improve pronunciation, with Automatic Pronunciation Assessment (APA) as the core component—scoring speakers' pronunciation quality across multiple aspects. Early APA methods focused on a single granularity: phoneme-level accuracy, word-level or utterance-level aspect detection. While effective for specific tasks, these single-granularity approaches fail to account for the inherently hierarchical structure of speech signals.

### Necessity of Multi-granularity Assessment

Speech signals possess an intrinsic hierarchical structure: phonemes compose words, and words compose sentences. Lower-granularity pronunciation outcomes directly influence higher-granularity scores—inaccurate phoneme production inevitably degrades word-level scores. Single-granularity modeling cannot reveal implicit cross-granularity associations. Consequently, integrating multi-aspect, multi-granularity assessment into a unified model has become a research trend.

### Limitations of Existing Multi-granularity Methods

Existing methods only consider unidirectional dependencies between adjacent granularities (phoneme→word→utterance), lacking bidirectional cross-granularity interaction: (1) GOPT processes each granularity in parallel but without cross-granularity interaction; (2) HiPAMA employs a hierarchical structure but with unidirectional information flow; (3) Gradformer focuses on utterance-level modeling and neglects phoneme–word associations; (4) HierGAT's fixed graph structure constrains dynamic interaction. In particular, the same word may carry different stress patterns across sentence contexts, and the absence of top-down interaction modeling explains the poor performance of prior methods on word stress. Furthermore, initial encoded features may be forgotten as hierarchical depth increases.

## Core Problem

How can bidirectional dynamic interaction among phoneme, word, and utterance granularities be achieved in multi-aspect, multi-granularity pronunciation assessment, while simultaneously mitigating the feature forgetting problem caused by hierarchical modeling?

## Method

### Overall Architecture

HIA takes GOP features and canonical phoneme embeddings as input. After encoding via a Transformer encoder to obtain acoustic embeddings, a residual hierarchical structure sequentially models scores at each granularity. The core components are the Interactive Attention Module and residual connections.

### Acoustic Feature Processing

GOP features (84-dimensional) are extracted using the Librispeech acoustic model, comprising Log Phone Posterior (LPP) and Log Posterior Ratio (LPR):

$$\text{LPP}(p) \approx \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \log P(p|o_t)$$

$$\text{LPR}(p_j|p_i) = \log P(p_j|\mathbf{o}; t_s, t_e) - \log P(p_i|\mathbf{o}; t_s, t_e)$$

With 42 pure phonemes in total, the GOP feature is an 84-dimensional vector. The projected GOP features, canonical phoneme embeddings, and learnable positional embeddings are summed and fed into the Transformer encoder.

### Interactive Attention Module

**Core Innovation**: The first work to achieve bidirectional interaction across all three granularities in pronunciation assessment.

1. **Granularity Query Initialization**: Granularity-specific query vectors $Q^l \in \mathbb{R}^{B \times D}$ are projected from acoustic embeddings.
2. **Concatenated Self-Attention**: The three granularity queries are concatenated as $Q = \{Q^{phn}, Q^{word}, Q^{utt}\} \in \mathbb{R}^{B \times 3 \times D}$, and bidirectional interaction is realized via self-attention: $Q_{self} = \text{SelfAttn}(Q)$
3. **Cross-Attention Mapping**: The self-attended queries serve as queries while the acoustic embedding $X$ serves as keys/values, mapping representations into the acoustic feature space: $Q_{cross} = \text{CrossAttn}(Q_{self}, X)$
4. **Projection Output**: After passing through an FFN, per-granularity interactive attention heads $H^{phn}$, $H^{word}$, $H^{utt}$ are obtained via projection.

### Residual Hierarchical Multi-granularity Modeling

**Phoneme Level**: The acoustic embedding $X$ is added with the interactive attention head $H^{phn}$, and phoneme accuracy scores are produced via a 1-D convolution and regression head:

$$S^{phn} = \text{Conv}(X + H^{phn})$$

**Word Level**: Acoustic embeddings, phoneme scoring results, and the word-level attention head are combined; word-level multi-aspect associations are modeled via an aspect attention mechanism:

$$X^{word} = X + S^{phn} + H^{word}, \quad S^{word} = \text{AspectAttn}(X^{word})$$

**Utterance Level**: A Transformer decoder captures long-range dependencies; learnable query vectors are initialized, with acoustic embeddings plus word-level scores plus the utterance-level attention head serving as keys/values:

$$S^{utt} = \text{TransDecoder}(Q^{utt}, X + S^{word} + H^{utt})$$

**Residual Connections**: The original acoustic embedding $X$ is incorporated at each granularity level, mitigating the forgetting of initial features as hierarchy depth increases.

### Loss & Training

MSE loss is applied to each aspect at each granularity; the total loss is the sum of all granularity-aspect losses:

$$L_{\text{total}} = \sum_{i=1}^M \frac{1}{N} \sum_{j=1}^N L_{ij}$$

## Key Experimental Results

Dataset: speechocean762 (5,000 English sentences, 250 non-native speakers including children). Adam optimizer, lr=1e-3; results averaged over 5 different random seeds with standard deviations reported.

### Main Results (PCC↑, comparison with SOTA)

| Model | Phoneme PCC↑ | Word Accuracy↑ | Word Stress↑ | Word Total↑ | Utt. Fluency↑ | Utt. Prosodic↑ | Utt. Total↑ |
|------|----------|-------------|-----------|----------|-----------|------------|----------|
| GOPT | 0.612 | 0.533 | 0.291 | 0.549 | 0.753 | 0.760 | 0.742 |
| HiPAMA | 0.616 | 0.575 | 0.320 | 0.591 | 0.749 | 0.751 | 0.754 |
| Gradformer | 0.646 | 0.598 | 0.334 | 0.614 | 0.769 | 0.767 | 0.756 |
| **HIA** | **0.657** | **0.613** | **0.436** | **0.628** | **0.778** | **0.784** | **0.764** |
| Human Expert | 0.555 | 0.589 | 0.212 | 0.602 | 0.665 | 0.651 | 0.675 |

HIA achieves a PCC of 0.436 on word stress, a 30.5% improvement over Gradformer (+0.102), representing the most significant gain. HIA surpasses inter-rater agreement among five human expert annotators on all metrics except utterance-level completeness.

### Ablation Study on Interactive Attention Module

| Configuration | Phoneme PCC | Word Stress | Word Total | Utt. Total |
|------|---------|----------|---------|---------|
| w/o all interaction heads | 0.626 | 0.335 | 0.605 | 0.748 |
| word + utt heads only | 0.621 | 0.429 | 0.617 | 0.758 |
| phoneme + utt heads only | 0.661 | 0.328 | 0.604 | 0.759 |
| phoneme + word heads only | 0.653 | 0.421 | 0.621 | 0.754 |
| **All interaction heads (HIA)** | **0.657** | **0.436** | **0.628** | **0.764** |

### Ablation Study on Residual Hierarchical Structure

| Configuration | Phoneme PCC | Word Stress | Word Total | Utt. Total |
|------|---------|----------|---------|---------|
| w/o residual | 0.647 | 0.382 | 0.603 | 0.748 |
| w/o hierarchy | 0.645 | 0.374 | 0.593 | 0.753 |
| **HIA** | **0.657** | **0.436** | **0.628** | **0.764** |

### Ablation Study on Number of Convolutional Layers

| Layers | Phoneme PCC | Word Stress | Word Total | Utt. Total |
|------|---------|----------|---------|---------|
| 0 | 0.638 | 0.415 | 0.601 | 0.754 |
| 1 (HIA) | **0.657** | **0.436** | **0.628** | **0.764** |
| 2 | 0.646 | 0.427 | 0.618 | 0.759 |
| 3 | 0.645 | 0.421 | 0.617 | 0.755 |

## Highlights & Insights

- **First Bidirectional Granularity Interaction**: Through the elegant design of concatenating three-granularity queries for self-attention followed by cross-attention, full bidirectional information flow among phoneme↔word↔utterance is realized, yielding over 30% improvement on word stress assessment in particular.
- **Residual Hierarchical Structure**: Incorporating residual connections from the original acoustic embedding $X$ at each granularity level effectively mitigates feature forgetting caused by increasing hierarchical depth.
- **Surpassing Human Expert Agreement**: HIA exceeds inter-annotator consistency among five expert raters on nearly all metrics, demonstrating strong practical value for pronunciation assessment.
- **Comprehensive Ablation Study**: Detailed ablation analyses are conducted over interaction attention (per-granularity ablation), residual/hierarchical structure, number of convolutional layers, embedding dimensions, and number of attention heads.

## Limitations & Future Work

- **Single Dataset**: Evaluation is limited to speechocean762, which is relatively small (5,000 sentences); the completeness score distribution is extremely skewed (4,975/5,000 are perfect scores), limiting the reliability of evaluation on certain metrics.
- **Dependence on GOP Features**: The framework relies on conventional GOP features as input without leveraging representations from self-supervised speech models (e.g., wav2vec 2.0, HuBERT), potentially capping performance.
- **Read-Aloud Scenario Only**: The framework is designed for read-aloud pronunciation assessment and is not applicable to open-ended spoken response scenarios.
- **Limited Model Scale**: The small model configuration (embedding dimension 48, single-head attention) is constrained by dataset size; optimal configurations would need to be re-explored when larger datasets become available.

## Related Work & Insights

- **GOPT** (2022): Transformer-based multi-task parallel assessment without cross-granularity interaction; HIA surpasses it by 7.4% on phoneme PCC and 14.4% on word Total.
- **HiPAMA** (2023): Introduces hierarchical structure to model granularity dependencies but with unidirectional information flow; HIA outperforms it on word Stress by 36.3% (0.436 vs. 0.320).
- **Gradformer** (2024): Convolution-enhanced Transformer with granularity decoupling, focusing on utterance-level modeling while neglecting phoneme–word associations; HIA achieves comprehensive superiority across all metrics.
- **HierGAT** (2024): Graph neural network-based hierarchical modeling; its fixed graph structure limits dynamic interaction, whereas HIA's attention-based dynamic interaction is more flexible.
- **Non-GOP Methods** (wav2vec2-based, LAS, etc.): Using self-supervised features, these methods approach competitive performance on utterance-level Total (0.725/0.766), but do not provide multi-granularity assessment capability.

The "concatenate multi-granularity queries → self-attention → cross-attention" design pattern of the Interactive Attention Module is transferable to other multi-granularity tasks, such as document summarization (word→sentence→paragraph) and video understanding (frame→clip→full video). The residual hierarchical structure's strategy for mitigating feature forgetting echoes cross-layer connection ideas in DenseNet and U-Net, and its application in sequential modeling warrants further exploration. The substantial improvement in word stress assessment (+30%) validates the importance of bidirectional interaction for capturing context-dependent pronunciation patterns, and suggests that utterance-level information could similarly guide word-level prosody generation in speech synthesis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Bidirectional granularity interaction is proposed for the first time in pronunciation assessment; the Interactive Attention Module design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations cover nearly all design choices; correlation analyses with data distributions further strengthen the arguments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with thorough motivation; figures and tables are clear and intuitive.
- **Value**: ⭐⭐⭐⭐ — Achieves comprehensive SOTA in the pronunciation assessment subfield with high practical utility, though applicability remains relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](../../ICLR2026/audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[AAAI 2026\] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](../../ICLR2026/audio_speech/efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)

</div>

<!-- RELATED:END -->
