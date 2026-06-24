---
title: >-
  [Paper Note] Different Speech Translation Models Encode and Translate Speaker Gender Differently
description: >-
  [ACL 2025][Audio & Speech][Speech Translation] Through an attention-based probing analysis, this study investigates how speech translation models of different architectures encode speaker gender. It finds that traditional encoder-decoder models preserve gender information well, whereas adapters in modern speech+MT architectures significantly erase gender information, leading to more severe masculine default bias in translation.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Speech Translation"
  - "Gender Encoding"
  - "Interpretability"
  - "Probing Analysis"
  - "Translation Bias"
date: 2026-05-08
content_hash: b277dbf5c4870135
---

# Different Speech Translation Models Encode and Translate Speaker Gender Differently

**Conference**: ACL 2025  
**arXiv**: [2506.02172](https://arxiv.org/abs/2506.02172)  
**Code**: [https://github.com/hlt-mt/speech-translation-gender](https://github.com/hlt-mt/speech-translation-gender)  
**Area**: Speech  
**Keywords**: Speech Translation, Gender Encoding, Interpretability, Probing Analysis, Translation Bias

## TL;DR

Through an attention-based probing analysis, this study investigates how speech translation models of different architectures encode speaker gender. It finds that traditional encoder-decoder models preserve gender information well, whereas adapters in modern speech+MT architectures significantly erase gender information, leading to more severe masculine default bias in translation.

## Background & Motivation

**Background**: Audio representation learning studies show that internal model representations can capture phonetic and speaker-related features (including gender). The field of speech translation (ST) is transitioning from traditional encoder-decoder architectures to new speech+MT architectures—specifically, connecting a pre-trained speech encoder to a machine translation system via an adapter.

**Limitations of Prior Work**: When translating from "notional gender" languages (such as English) to "grammatical gender" languages (such as French, Italian, and Spanish), models need to infer gender from the context and correctly apply grammatical inflections. For example, "I was born in..." → "Je suis né/née à...". However, existing systems generally exhibit a masculine default bias.

**Key Challenge**: An architectural shift is occurring in the ST field, but little is known about how old and new architectures encode and utilize gender information—and how this affects gender bias in translation.

**Goal**: (1) Do different ST architectures encode speaker gender information? (2) How do architectural differences affect gender assignment in translation? (3) Is there a correlation between gender encoding capability and translation accuracy?

**Key Insight**: Utilizing probing methods—a well-established interpretability technique—the authors train classifiers to predict speaker gender from the model's hidden states, followed by analyzing the relationship between gender encoding capability and translation accuracy.

**Core Idea**: Gender encoding capability is highly correlated with translation gender accuracy; the adapter in speech+MT architectures erases gender information, exacerbating the masculine default bias.

## Method

### Overall Architecture

A probing analysis is conducted on three categories of ST models: traditional encoder-decoder (enc-dec), SeamlessM4T (speech+MT), and ZeroSwot (speech+MT). The relationship between gender encoding and translation accuracy is analyzed across three translation directions (En→Fr/It/Es).

### Key Designs

**Module: Attention-based Probe**

- **Function**: Extracts gender information from the sequence of model hidden states for binary classification.
- **Mechanism**: Inspired by Q-Former, it performs an attention operation using a single learnable query vector $\mathbf{q} \in \mathbb{R}^d$ and the hidden state sequence $\mathbf{X} = \langle \mathbf{x}_1, ..., \mathbf{x}_L \rangle$. The hidden states are projected into Key and Value via learnable weight matrices $\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$. The output $\mathbf{o} \in \mathbb{R}^d$ is classified through a linear layer. The attention weights $\mathbf{a} \in \mathbb{R}^L$ indicate which positions contribute the most to gender encoding.
- **Design Motivation**: 
    - Avoids the issue where mean/max pooling might mask structural variations over time.
    - Avoids the limitation where fixed-position probes cannot support variable-length sequences.
    - Keeps the architecture simple (conforming to probing design principles) while being more expressive than linear models.
    - Mimics how the ST decoder accesses encoder states through the cross-attention mechanism.

**Probing Locations**:
- enc-dec: Encoder output
- speech+MT: Probed before the adapter (pre-ad) and after the adapter (post-ad), respectively.

### Loss & Training

- Probe training utilizes gender-balanced training and validation sets sampled from the MuST-C training set (en→es).
- Two test sets: test-generic (generic unbalanced set) and test-speaker (a gender-balanced set from MuST-SHE, focusing on speaker reference).
- Evaluation metrics: Probing utilizes macro F1 + per-class recall; translation quality is evaluated using COMET; gender translation accuracy is evaluated with accuracy + coverage.

## Key Experimental Results

### Main Results

Gender probe F1 scores (average on test-speaker):

| Model | Probing Location | en→es | en→fr | en→it | Average |
|------|---------|-------|-------|-------|------|
| Seamless | post-ad | 51.72 | 54.51 | - | 53.95 |
| Seamless | pre-ad | 67.32 | 67.52 | - | 68.47 |
| ZeroSwot | post-ad | 61.80 | 61.62 | - | 61.36 |
| ZeroSwot | pre-ad | 90.25 | 90.02 | - | 89.61 |
| **enc-dec** | encoder | **93.14** | **94.59** | - | **94.64** |

Translation gender accuracy (test-speaker):

| Model | She Acc | He Acc | All Acc | COMET |
|------|---------|--------|---------|-------|
| Seamless | 14.33 | 90.25 | 53.35 | 80.36 |
| ZeroSwot | 50.69 | 74.90 | 62.80 | 83.94 |
| **enc-dec** | **78.53** | **92.25** | **85.57** | 74.77 |

### Ablation Study

Impact of the adapter on gender information (F1 drop):

| Model | pre-ad F1 | post-ad F1 | Drop |
|------|-----------|------------|------|
| Seamless | 68.47 | 53.95 | ~21% |
| ZeroSwot | 89.61 | 61.36 | ~32% |

Correlation between gender encoding and translation accuracy: $R^2 = 0.99$, $p < 0.01$.

### Key Findings

1. **Traditional enc-dec models** demonstrate the strongest gender encoding capability (F1 > 94), whereas speech+MT models see a substantial drop after the adapter (Seamless post-ad is only 53.95).
2. **The adapter acts as a key bottleneck**: ZeroSwot's pre-ad F1 is 89.61, but post-ad it plummets to 61.36, representing a loss of approximately 32%.
3. **Gender encoding is highly correlated with translation accuracy** ($R^2=0.99$), where stronger encoding yields more accurate translations.
4. **Masculine default bias**: Seamless's translation accuracy for females is as low as 12.09% (en→es) compared to 90.55% for males, showing a stark disparity.
5. Even enc-dec models exhibit a slight masculine bias: female average 78.53 vs. male 92.25.
6. ST models primarily encode gender information at early positions in the sequence.

## Highlights & Insights

- **Counter-intuitive finding**: While standard NLP research often assumes that removing gender information improves fairness, this study demonstrates that preserving gender encoding in ST generates fairer translations (leading to higher translation accuracy for female speakers).
- **Elegant attention-based probe design**: It handles variable-length sequences flexibly (like Q-Former) while maintaining the characteristic simplicity expected of a probe.
- **The adapter acts as a "bottleneck" for gender information**: This finding has important implications for the design of speech+MT architectures, as mapping to text embedding spaces filters out speaker-related acoustic information.
- The strong correlation ($R^2=0.99$) between gender encoding and translation accuracy provides robust evidence pointing toward a causal link.

## Limitations & Future Work

1. **Limited model and language coverage**: Only three models and three translation directions are evaluated, without covering LLM-based ST systems.
2. **Binary gender framework**: Due to data availability constraints, only She/He binary classification is analyzed, leaving non-binary genders unexplored.
3. **Causal inference limitations of probing**: Correlation between probe performance and translation performance does not imply causation, requiring further analyses like amnesic probing.
4. Modifying adapter architectures to better preserve gender information remains unexplored.
5. Ethical considerations: Inferring gender from acoustic features may lead to misclassification (e.g., for transgender individuals, children, etc.).

## Related Work & Insights

- **MuST-SHE** provides a dedicated corpus for evaluating gender translation in ST; its self-reported gender labels mitigate the risk of misclassification.
- **SeamlessM4T** and **ZeroSwot** represent two training strategies in speech+MT architectures (joint training vs. frozen MT).
- **Q-Former** (BLIP-2) inspired the design of the attention-based probe.
- Sociophonetics research offers a theoretical foundation for the presence of acoustic gender cues.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First systematic analysis of gender encoding in ST models; uncovers the adapter bottleneck)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Three architectures $\times$ three language directions; strong correlation with $R^2=0.99$)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Rigorous discussion, thorough ethical considerations, and clear structure)
- **Value**: ⭐⭐⭐⭐ (Practical guidance for both ST architecture design and fairness)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](improving_language_and_modality_transfer_in.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ACL 2025\] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation](eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)

</div>

<!-- RELATED:END -->
