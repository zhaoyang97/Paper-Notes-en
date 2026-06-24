---
title: >-
  [Paper Note] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment
description: >-
  [AAAI2026][Audio & Speech][Pronunciation Assessment] This paper proposes the HIA framework, which achieves bidirectional information interaction among the three granularities of phonemes, words, and sentences via an Interactive Attention Module. Combined with a residual hierarchical structure to alleviate feature forgetting, it achieves SOTA performance across all granularity and aspect metrics on the speechocean762 dataset.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Pronunciation Assessment"
  - "Multi-granularity Interaction"
  - "Attention Mechanism"
  - "Residual Hierarchical Structure"
  - "CAPT"
  - "Speech Scoring"
date: 2026-05-08
content_hash: 89e663205eb6d30c
---

# Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment

**Conference**: AAAI2026  
**arXiv**: [2601.01745](https://arxiv.org/abs/2601.01745)  
**Authors**: Hong Han, Hao-Chen Pei, Zhao-Zheng Nie, Xin Luo, Xin-Shun Xu  
**Code**: Not released  
**Area**: Audio & Speech  
**Keywords**: Pronunciation Assessment, Multi-granularity Interaction, Attention Mechanism, Residual Hierarchical Structure, CAPT, Speech Scoring  

## TL;DR

This paper proposes the HIA framework, which achieves bidirectional information interaction among the three granularities of phonemes, words, and sentences via an Interactive Attention Module. Combined with a residual hierarchical structure to alleviate feature forgetting, it achieves SOTA performance across all granularity and aspect metrics on the speechocean762 dataset.

## Background & Motivation

### Importance of Automatic Pronunciation Assessment

Computer-Assisted Pronunciation Training (CAPT) systems help language learners improve their pronunciation through instant feedback, with the core being Automatic Pronunciation Assessment (APA)—scoring the speaker's pronunciation quality across multiple aspects. Early APA methods focused on a single granularity: phoneme-level pronunciation accuracy assessment, or word-level or sentence-level multi-aspect evaluation. Although these single-granularity methods perform well on specific tasks, they do not consider the natural multi-granularity hierarchical properties of speech signals.

### Necessity of Multi-granularity Assessment

Speech signals have an inherent hierarchical structure: phonemes form words, and words form sentences. Lower-granularity pronunciation results directly impact higher-granularity scores—if phonemes are mispronounced, the overall score of the word is inevitably affected. Single-granularity modeling cannot reveal implicit associations between different granularities. Therefore, integrating multi-aspect and multi-granularity assessment tasks into a unified model has become a research trend.

### Limitations of Prior Work

Existing methods only consider unidirectional dependencies between adjacent granularities (phoneme → word → sentence) and lack bidirectional interaction between granularities: (1) GOPT processes each granularity in parallel but lacks cross-granularity interaction; (2) HiPAMA employs a hierarchical structure but with unidirectional information flow; (3) Gradformer focuses on sentence-level modeling while neglecting the phoneme-word association; (4) HierGAT's fixed graph structure limits dynamic interaction. In particular, the same word may have different stress patterns in different sentences, and the lack of top-down interactive modeling is why existing methods perform poorly on word stress. Furthermore, as the hierarchical depth increases, the initial encoded features may be forgotten.

## Core Problem

How to achieve bidirectional dynamic interaction among the three granularities of phonemes, words, and sentences in multi-aspect and multi-granularity pronunciation assessment, while alleviating the problem of feature forgetting caused by hierarchical modeling?

## Method

### Overall Architecture

HIA receives GOP features and canonical phoneme embeddings as input, encodes them using a Transformer encoder to obtain acoustic embeddings, and then models the scoring of each granularity sequentially through a residual hierarchical structure. The core components are the Interactive Attention Module and residual connections.

### Acoustic Feature Processing

The Librispeech acoustic model is used to extract GOP features (84-dimensional), including Log Phone Posterior (LPP) and Log Posterior Ratio (LPR):

$$\text{LPP}(p) \approx \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \log P(p|o_t)$$

$$\text{LPR}(p_j|p_i) = \log P(p_j|\mathbf{o}; t_s, t_e) - \log P(p_i|\mathbf{o}; t_s, t_e)$$

There are 42 canonical phonemes in total, making the GOP feature an 84-dimensional vector. The projected GOP features, canonical phoneme embeddings, and trainable positional embeddings are summed and fed into the Transformer encoder.

### Interactive Attention Module

**Core Innovation**: The first to achieve bidirectional interaction among three granularities in pronunciation assessment.

1. **Initialize Granularity Queries**: Project query vectors $Q^l \in \mathbb{R}^{B \times D}$ for each granularity from the acoustic embeddings.
2. **Concatenated Self-Attention**: Concatenate the queries of the three granularities as $Q = \{Q^{phn}, Q^{word}, Q^{utt}\} \in \mathbb{R}^{B \times 3 \times D}$, and achieve bidirectional interaction via self-attention: $Q_{self} = \text{SelfAttn}(Q)$.
3. **Cross-Attention Mapping**: Take the self-attention heads as query and the acoustic embeddings $X$ as key/value to map back to the acoustic feature space: $Q_{cross} = \text{CrossAttn}(Q_{self}, X)$.
4. **Project Outputs**: After passing through an FFN, project to obtain the interactive attention heads $H^{phn}$, $H^{word}$, and $H^{utt}$ for each granularity.

### Residual Hierarchical Multi-granularity Modeling

**Phoneme level**: Acoustic embedding $X$ plus the interactive attention head $H^{phn}$ are passed through a 1-D convolution and a regression head to output phoneme accuracy:

$$S^{phn} = \text{Conv}(X + H^{phn})$$

**Word level**: Combined with the acoustic embeddings, phoneme scoring results, and the word-level attention head, word-level multi-aspect associations are modeled through an aspect attention mechanism:

$$X^{word} = X + S^{phn} + H^{word}, \quad S^{word} = \text{AspectAttn}(X^{word})$$

**Sentence level**: A Transformer decoder is used to capture long-range dependencies, initializing a learnable query vector and using acoustic embeddings + word-level scores + sentence-level attention head as key/value:

$$S^{utt} = \text{TransDecoder}(Q^{utt}, X + S^{word} + H^{utt})$$

**Residual Connection**: The original acoustic embedding $X$ is introduced during the modeling of each granularity, alleviating the forgetting of initial features caused by deeper hierarchies.

### Loss & Training

An MSE loss is utilized for each aspect of each granularity, and the total loss is the sum of losses across all granularities and aspects:

$$L_{\text{total}} = \sum_{i=1}^M \frac{1}{N} \sum_{j=1}^N L_{ij}$$

## Key Experimental Results

Dataset: speechocean762 (5000 English sentences, 250 non-native speakers, including children). Adam optimizer, lr=1e-3, average and standard deviation calculated across 5 different seeds.

### Main Results

| Model | Phoneme PCC↑ | Word Accuracy↑ | Word Stress↑ | Word Total↑ | Sentence Fluency↑ | Sentence Prosodic↑ | Sentence Total↑ |
|------|----------|-------------|-----------|----------|-----------|------------|----------|
| GOPT | 0.612 | 0.533 | 0.291 | 0.549 | 0.753 | 0.760 | 0.742 |
| HiPAMA | 0.616 | 0.575 | 0.320 | 0.591 | 0.749 | 0.751 | 0.754 |
| Gradformer | 0.646 | 0.598 | 0.334 | 0.614 | 0.769 | 0.767 | 0.756 |
| **HIA** | **0.657** | **0.613** | **0.436** | **0.628** | **0.778** | **0.784** | **0.764** |
| Human Expert | 0.555 | 0.589 | 0.212 | 0.602 | 0.665 | 0.651 | 0.675 |

HIA achieves a PCC of 0.436 on word stress, which is a 30.5% improvement (+0.102) compared to Gradformer, marking the most significant progress. HIA outperforms human expert evaluator consistency on all metrics except sentence-level completeness.

### Interactive Attention Module Ablation

| Configuration | Phoneme PCC | Word Stress | Word Total | Sentence Total |
|------|---------|----------|---------|---------|
| w/o All Interaction Heads | 0.626 | 0.335 | 0.605 | 0.748 |
| Word & Sentence Interaction Heads Only | 0.621 | 0.429 | 0.617 | 0.758 |
| Phoneme & Sentence Interaction Heads Only | 0.661 | 0.328 | 0.604 | 0.759 |
| Phoneme & Word Interaction Heads Only | 0.653 | 0.421 | 0.621 | 0.754 |
| **All Interaction Heads (HIA)** | **0.657** | **0.436** | **0.628** | **0.764** |

### Residual Hierarchical Structure Ablation

| Configuration | Phoneme PCC | Word Stress | Word Total | Sentence Total |
|------|---------|----------|---------|---------|
| Remove Residuals | 0.647 | 0.382 | 0.603 | 0.748 |
| Remove Hierarchy | 0.645 | 0.374 | 0.593 | 0.753 |
| **HIA** | **0.657** | **0.436** | **0.628** | **0.764** |

### Number of Convolutional Layers Ablation

| Layers | Phoneme PCC | Word Stress | Word Total | Sentence Total |
|------|---------|----------|---------|---------|
| 0 layers | 0.638 | 0.415 | 0.601 | 0.754 |
| 1 layer (HIA) | **0.657** | **0.436** | **0.628** | **0.764** |
| 2 layers | 0.646 | 0.427 | 0.618 | 0.759 |
| 3 layers | 0.645 | 0.421 | 0.617 | 0.755 |

## Highlights & Insights

- **First Bidirectional Cross-Granularity Interaction**: A concise design using self-attention and cross-attention on concatenated three-granularity queries achieves fully bidirectional information flow among phonemes ↔ words ↔ sentences, yielding a 30%+ improvement particularly in word stress assessment.
- **Residual Hierarchical Structure**: Introducing residual connections of the original acoustic embeddings during hierarchical step-by-step granulation modeling effectively mitigates feature forgetting caused by deeper hierarchies.
- **Outperforming Human Expert Consistency**: HIA surpasses the consistency among 5 expert evaluators on almost all metrics, demonstrating the practical value of the model in pronunciation assessment.
- **Extremely Thorough Ablation Studies**: Detailed ablation analyses are conducted on the interactive attention (granularity-by-granularity ablation), residual/hierarchical structure, number of convolutional layers, embedding dimension, and number of attention heads.

## Limitations & Future Work

- **Single Dataset Limitation**: The model is only evaluated on speechocean762, which is a small-scale dataset (5000 sentences) and has an extremely unbalanced distribution of the completeness score (4975/5000 are perfect scores), limiting the reliability of several metrics.
- **Input Features Dependent on GOP**: The model relies on traditional GOP features as input and does not leverage the representation power of self-supervised speech models (e.g., wav2vec 2.0, HuBERT), which might limit its performance upper bound.
- **Only Supports Read-Aloud Scenarios**: The framework is tailored for read-aloud pronunciation assessment and is not applicable to open-ended oral response scenarios.
- **Limited Model Scale**: Configured as a small model with an embedding dimension of 48 and single-head attention due to dataset size limitations; optimal configurations need to be re-explored with larger datasets.

## Related Work & Insights

- **GOPT** (2022): Transformer-based multitask parallel evaluation without cross-granularity interaction. HIA outperforms it by 7.4% on phoneme PCC and 14.4% on word Total.
- **HiPAMA** (2023): Introduces a hierarchical structure to model granularity dependencies, but with unidirectional information flow. HIA outperforms it by 36.3% on word Stress through bidirectional interaction (0.436 vs. 0.320).
- **Gradformer** (2024): Convolution-augmented Transformer + granularity decoupling, focusing on sentence-level modeling while neglecting phoneme-word associations. HIA leads across all metrics.
- **HierGAT** (2024): Hierarchical modeling with Graph Neural Networks, where the fixed graph structure restricts dynamic interaction. HIA's attention mechanism offers more flexible dynamic interaction.
- **Non-GOP Methods** (wav2vec 2.0-based, LAS, etc.): Leveraging self-supervised features achieves comparable performance on sentence-level Total (0.725 / 0.766) but lacks multi-granularity assessment capability.

## Related Work & Insights

- The design paradigm of the Interactive Attention Module ("concatenate multi-granularity queries → self-attention → cross-attention") is transferable to other multi-granularity tasks, such as document summarization (word → sentence → paragraph) and video understanding (frame → clip → full video).
- The feature-forgetting mitigation strategy of the residual hierarchical structure shares its lineage with cross-layer connection concepts like DenseNet and U-Net, but its application in sequence modeling deserves further exploration.
- The substantial improvement in word stress assessment (+30%) validates the significance of bidirectional interaction for capturing context-dependent pronunciation patterns, suggesting that sentence-level information can also be leveraged to guide word-level prosody generation in speech synthesis.

## Rating

- Novelty: ⭐⭐⭐⭐ — Bidirectional granularity interaction is proposed for the first time in pronunciation assessment, and the design of the Interactive Attention Module is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Ablations cover almost all design choices, and data correlation analysis increases persuasiveness.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with fully discussed problem motivation and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Achieves comprehensive SOTA in the niche field of pronunciation assessment with high practical value, though the scope of application is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Semantic-Acoustic Modeling via Semi-Discrete Residual Representations for Expressive End-to-End Speech Synthesis](../../ICLR2026/audio_speech/hierarchical_semantic-acoustic_modeling_via_semi-discrete_residual_representatio.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](../../ICLR2026/audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[CVPR 2026\] AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](../../CVPR2026/audio_speech/amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
