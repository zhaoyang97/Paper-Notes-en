---
title: >-
  [Paper Note] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-lingual transfer] By constructing an interpretable experimental framework, this study investigates the cross-lingual transfer of cultural knowledge during the language adaptation process of LLMs. It finds a bidirectional transfer between high-resource languages (Chinese, Korean) and English, whereas low-resource languages (Tibetan, Mongolian) exhibit an asymmetric transfer—knowledge mainly flows from low-resource languages…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-lingual transfer"
  - "cultural knowledge"
  - "low-resource languages"
  - "language adaptation"
  - "frequency hypothesis"
date: 2026-05-08
content_hash: 8669ca85375ebf54
---

# Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon

**Conference**: ACL 2025  
**arXiv**: [2506.01675](https://arxiv.org/abs/2506.01675)  
**Code**: [GitHub](https://github.com/luciusssss/cross-lingual-culture)  
**Area**: Multilingual Translation  
**Keywords**: Cross-lingual transfer, cultural knowledge, low-resource languages, language adaptation, frequency hypothesis

## TL;DR

By constructing an interpretable experimental framework, this study investigates the cross-lingual transfer of cultural knowledge during the language adaptation process of LLMs. It finds a bidirectional transfer between high-resource languages (Chinese, Korean) and English, whereas low-resource languages (Tibetan, Mongolian) exhibit an asymmetric transfer—knowledge mainly flows from low-resource languages to English, with limited reverse flow. The frequency hypothesis is proposed to explain this phenomenon.

## Background & Motivation

Large Language Models (LLMs) face challenges in processing global cultural diversity. Existing research primarily evaluates whether LLMs possess cultural knowledge of non-English communities, but little is known about the acquisition mechanisms of cultural knowledge, especially under multilingual environments. This paper focuses on the following core problems:

**Opacity of cross-lingual transfer**: The training data and processes of LLMs are opaque, making it difficult to conduct interpretable experiments to trace the sources and influencing factors of knowledge transfer.

**Factor confounding issue**: When the performance of cultural knowledge question-answering in a language improves, it is difficult to distinguish whether this is due to enhanced language proficiency or cross-lingual knowledge transfer.

**Underrepresented culture of low-resource languages**: Existing cultural evaluations rarely focus on communities using low-resource languages (such as Tibetan and Mongolian), leaving their cultural knowledge severely underrepresented in LLMs.

The authors point out that understanding the dynamics of cross-lingual cultural knowledge transfer is of great significance for building culture-aware models, especially for serving low-resource language communities.

## Method

### Overall Architecture

A three-step research framework is proposed: English Pre-training → Controlled Continued Pre-training → Bilingual Evaluation. The framework has three core characteristics:

1. **Transparent training data**: The model is trained from scratch using filtered English Wikipedia data.
2. **Decoupled transfer effects**: Two continued pre-training settings are designed to distinguish cross-lingual transfer from language proficiency improvement.
3. **Bilingual parallel evaluation**: Bilingual parallel probing questions are used to analyze bidirectional transfer.

### Key Designs

#### 1. Transparent Pre-training

- **Function**: Trains a 0.5B-parameter model from scratch, instead of using closed-source LLMs.
- **Mechanism**: The model is pre-trained on English Wikipedia (5B tokens) with all non-Latin characters filtered out, ensuring clear observation of the knowledge transfer process and tracing the corpus source of the acquired knowledge.
- **Design Motivation**: If pre-trained large models are used, the training data is opaque, making it impossible to know where the cultural knowledge originates.
- The model architecture adopts Qwen-2.5-0.5B.

#### 2. Continued Pre-training with Decoupled Transfer Effects

- **Function**: Designs two continued pre-training settings: one promoting cross-lingual transfer, and the other minimizing transfer.
- **Mechanism**:
    - **With bridges setting**: Parallel sentence pairs are added to the continued pre-training data, where each parallel pair is concatenated and mixed with other monolingual data.
    - **Without bridges setting**: The same data is used, but the co-occurrence of parallel sentence pairs is deliberately prevented by splitting the two sentences in each pair into independent documents before mixing them into the training data.
- **Design Motivation**: The performance gap between the two settings can serve as an estimate of the cross-lingual transfer effect. Non-Latin script languages (Korean, Chinese, Tibetan, Mongolian) are selected to maximize isolation from English.
- Continued pre-training is performed for 1500 steps with a batch size of 0.5M tokens.

#### 3. Bilingual Parallel Evaluation

- **Function**: Evaluates the model using English and non-English versions of cultural probing questions, respectively.
- **Mechanism**:
    - Non-English question testing: The performance difference between the "with bridges" and "without bridges" settings represents the amount of "English → non-English" transfer.
    - English question testing: The performance difference between the "with bridges" and "without bridges" settings represents the amount of "non-English → English" transfer.
- Evaluation adopts a cloze-style format, which is suitable for small models trained from scratch with limited capability.
- Korean cultural questions are derived from the CLIcK dataset, and Chinese ethnic minority cultural questions are from the book *Chinese Ethnic Minorities*, generated by GPT-4o and verified/translated by native speakers.

### Frequency Hypothesis

- **Hypothesis content**: Cultural knowledge that appears with higher frequency in the training corpus is more likely to undergo cross-lingual transfer.
- **Verification method**:
    - For each cultural probing question, the top 50 most relevant documents are retrieved from the corpus.
    - Llama-3.1-70B is used to judge whether the retrieved documents contain the cultural knowledge in the question.
    - A "cultural density" metric is introduced: number of knowledge occurrences / total documents in the corpus.

## Key Experimental Results

### Main Results: Direction of Cross-Lingual Transfer

| Cultural Community | Associated Language | English → Non-English Transfer | Non-English → English Transfer |
|---------|---------|---------------|---------------|
| Korean | Korean (High-resource) | Significant ✓ | Significant ✓ |
| Han Chinese | Chinese (High-resource) | Significant ✓ | Significant ✓ |
| Tibetan | Tibetan (Low-resource) | Insignificant ✗ | Significant ✓ |
| Mongolian | Mongolian (Low-resource) | Insignificant ✗ | Significant ✓ |

**Key Findings**: High-resource languages exhibit bidirectional transfer, while low-resource languages show asymmetric transfer (mainly in the non-English → English direction).

### Ablation Study: Cultural Density Analysis

| Cultural Community | Density in English Corpus | Density in Non-English Corpus |
|---------|------------|-------------|
| Korean | 2.86e-7 | 5.21e-7 |
| Han Chinese | 2.97e-7 | 2.84e-7 |
| Tibetan | 1.49e-7 | **9.19e-6** |
| Mongolian | 1.55e-7 | **3.72e-6** |

The cultural density of low-resource languages in non-English corpora is an order of magnitude higher than in English corpora, explaining the asymmetric transfer phenomenon.

### Instance-level Analysis

| Transfer Direction | Average Occurrences of Successfully Transferred Knowledge in Source Corpus | Overall Average Occurrences |
|---------|--------------------------------------|--------------|
| English → Non-English | 9.0 (English Corpus) | 4.2 |
| Non-English → English | 4.7 (Non-English Corpus) | 2.2 |

The frequency of successfully transferred knowledge in the source language corpus is significantly higher than average, further supporting the frequency hypothesis.

### Key Findings

1. **Asymmetric transfer is widespread**: Cultural knowledge of low-resource languages easily transfers to English, but cultural knowledge in English is difficult to transfer back to low-resource languages.
2. **Frequency is a key factor**: Cultural knowledge with higher frequency in the corpus is more likely to undergo cross-lingual transfer.
3. **Divergent roles of cross-lingual bridges**: For high-resource languages, parallel sentence pairs are effective in both directions; for low-resource languages, they are primarily effective in the non-English → English direction.
4. **Bridge settings can compensate for forgetting**: For most languages, English performance under the with-bridges setting continues to improve, indicating that cross-lingual transfer can mitigate the forgetting of English capability caused by continued pre-training.

## Highlights & Insights

- **Interpretable experimental design**: Training models from scratch with transparent data and controlled settings offers a "clean" experimental environment to study knowledge transfer, which holds prominent methodological value.
- **First systematic study on the cross-lingual transfer mechanism of cultural knowledge**: Fills an important gap regarding "how LLMs acquire cultural knowledge."
- **Concise and explanatory frequency hypothesis**: Connects cross-lingual transfer with frequency effects in monolingual knowledge acquisition, forming a unified explanation.
- **Attention to genuinely low-resource cultures**: Selecting Tibetan and Mongolian as research objects, communities that have been historically overlooked in existing NLP cultural research.

## Limitations & Future Work

1. **Model scale limitations**: Only a 0.5B model is used (a computational cost trade-off for 16 settings); whether the conclusions generalize to larger models remains to be verified.
2. **Limited cultural coverage**: The experimental design requires non-Indo-European languages with non-Latin scripts, which significantly restricts the choices; meanwhile, question collection and validation require substantial manual effort.
3. **Imperfect retrieval system**: Cultural density analysis relies on a retrieval system, which may introduce inaccuracies.
4. **Only 4 cultures/languages**: The sample size is small, limiting statistical significance.
5. **No exploration of mitigation methods for asymmetric transfer**: The study only analyzes the phenomenon and its causes, without proposing solutions on how to improve the transfer of cultural knowledge for low-resource languages.

## Related Work & Insights

- It aligns with the findings of Kandpal et al. (2023) regarding "LLMs struggling to learn tail knowledge."
- The frequency hypothesis can be generalized to other types of knowledge (not limited to cultural knowledge).
- Provides a new perspective for enhancing the cultural awareness of LLMs in the future: increasing the exposure frequency of cultural knowledge in low-resource language data.
- Offers important references for designing continued pre-training strategies for multilingual models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First systematic study of cross-lingual transfer mechanisms of cultural knowledge, ingenious experimental design)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Controlled experiments + frequency analysis + instance-level verification, complete chain of evidence)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear problem definition, precise framework description, rigorous logic)
- **Value**: ⭐⭐⭐⭐ (Important insights into how LLMs acquire cultural knowledge, highly significant for low-resource language research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)

</div>

<!-- RELATED:END -->
