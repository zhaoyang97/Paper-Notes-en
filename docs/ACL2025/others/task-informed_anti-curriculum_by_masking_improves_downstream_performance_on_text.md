---
title: >-
  [Paper Note] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text
description: >-
  [ACL 2025][Masked Language Modeling] TIACBM proposes a task-informed anti-curriculum masking fine-tuning strategy: leveraging downstream task knowledge (e.g., sentiment polarity, part-of-speech tags) to determine which tokens are masked, and employing a cyclically decaying masking rate. It achieves statistically significant performance improvements across three tasks: sentiment analysis, text classification, and authorship attribution.
tags:
  - "ACL 2025"
  - "Masked Language Modeling"
  - "Anti-curriculum Learning"
  - "Task-informed Masking"
  - "Fine-tuning Strategy"
  - "Sentiment Analysis"
date: 2026-05-08
content_hash: 3a9bb0e97c58be27
---

# Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text

**Conference**: ACL 2025  
**arXiv**: [2502.12953](https://arxiv.org/abs/2502.12953)  
**Code**: Yes ([https://github.com/JarcaAndrei/TIACBM](https://github.com/JarcaAndrei/TIACBM))  
**Area**: Other  
**Keywords**: Masked Language Modeling, Anti-curriculum Learning, Task-informed Masking, Fine-tuning Strategy, Sentiment Analysis

## TL;DR

TIACBM proposes a task-informed anti-curriculum masking fine-tuning strategy: leveraging downstream task knowledge (e.g., sentiment polarity, part-of-speech tags) to determine which tokens are masked, and employing a cyclically decaying masking rate. It achieves statistically significant performance improvements across three tasks: sentiment analysis, text classification, and authorship attribution.

## Background & Motivation

Masked Language Modeling (MLM) is a core technology for pre-training language models, but two overlooked issues remain:

**Random selection of masked tokens**: In standard MLM, the tokens to be masked are selected completely at random without utilizing task-relevant knowledge.

**Fixed masking ratio**: The masking rate typically remains constant at 15% throughout the entire training process.

Existing studies (Ankner et al., 2024; Yang et al., 2023) have found that decaying the masking rate is more effective for text pre-training—this essentially corresponds to an **anti-curriculum strategy** (hard-to-easy), as a higher masking rate makes the learning task more challenging. However, these works only focus on the pre-training stage and do not exploit downstream task information.

The core innovation of TIACBM lies in: **introducing MLM as an auxiliary objective during the fine-tuning phase** and combining task knowledge to selectively mask important tokens.

## Method

### Overall Architecture

TIACBM consists of two core components:
1. **Cyclically Decaying Masking Rate**: Utilizing a decaying masking rate vector $\mathbf{r} = \{r_1 \geq \cdots \geq r_K\}$ during training, which is reset every K steps (cyclically).
2. **Task-Informed Token Selection**: Computing the importance of each token according to a task-specific `task_relevance` function, and sampling the tokens to be masked based on probabilities.

### Key Designs

1. **Masking Strategy for Sentiment Analysis Task**

    - Core Assumption: The most subjective words are the most important features.
    - Implementation: SentiWordNet 3.0 is used to find the polarity score (positive + negative = subjectivity) of each word; words with higher subjectivity are more likely to be masked.
    - The Lesk algorithm is employed to find the most probable synset to determine the positive/negative scores.
    - Importance score: $s_i = s_{pos}^i + s_{neg}^i$

2. **Masking Strategy for Text Classification (Topic) Task**

    - Core Assumption: Content words (nouns, verbs, adjectives, adverbs) are more topic-relevant than functional words.
    - Implementation: Non-content words are assigned an importance of 0, whereas content words utilize the attention weights of the pre-trained model as importance.
    - Attention importance: Averaged over all attention blocks and heads: $\mathbf{a} = \frac{1}{B \cdot H \cdot |\mathbf{x}|} \sum_h \sum_b \sum_j A_{b,j}^h$
    - Finally: $s_i = a_i$ if $x_i$ is a content word, otherwise $s_i = 0$

3. **Masking Strategy for Authorship Attribution Task**

    - Core Assumption: Functional words (prepositions, articles, conjunctions, symbols, punctuation) reflect writing style.
    - Implementation: Inverse to text classification, functional words are masked instead of content words.
    - Finally: $s_i = a_i$ if $x_i$ is a functional word, otherwise $s_i = 0$

4. **Cyclically Decaying Masking Rate**

    - Create a vector of K decaying masking rates, resetting every K iteration cycles.
    - Motivation: Cyclic resets prevent the model from overfitting during the low masking rate phase, while maintaining the hard-to-easy anti-curriculum effect.
    - Number of masked tokens: $N = \lfloor |\mathbf{x}| \cdot r_t \rfloor$

### Loss & Training

Fine-tuning simultaneously optimizes the classification loss and the MLM reconstruction loss, where the masking strategy influences the reconstruction target of the MLM component. Strategically prioritizing the masking of discriminative features prevents feature co-adaptation, yielding a regularization effect similar to Dropout.

## Key Experimental Results

### Main Results — BERT/RoBERTa (Table 1 Summary)

| Strategy | Reuters(F1) | 20News(Acc) | SST2(Acc) | PAN19-P1(Acc) | PAN19-P5(Acc) |
|------|------------|-------------|-----------|---------------|---------------|
| Standard Fine-tuning | 90.61 | 84.63 | 93.38 | 58.24 | 66.10 |
| Fixed Masking (15%) | 90.81 | 84.98 | 93.94 | 47.50 | 65.76 |
| Poesina (CL++) | 90.72 | 82.30 | 94.00 | 44.76 | 68.66 |
| Ankner (Decay) | 90.99 | 85.39 | 93.83 | 46.03 | 65.55 |
| Cyclic Decay | 90.96 | 84.88 | 94.10 | 51.94 | 69.28 |
| **TIACBM** | **91.20** | **85.65** | **94.61** | **60.60** | **69.94** |

Consistent improvements are also observed on RoBERTa, with all results passing Cochran's Q test (p < 0.001).

### GPT-2 Experiments (Table 2)

| Strategy | SST2(Acc) | PAN19-P1(Acc) | PAN19-P5(Acc) |
|------|-----------|---------------|---------------|
| Standard Fine-tuning | 92.35 | 67.96 | 38.16 |
| **TIACBM** | **92.96** | **73.44** | **42.90** |

This demonstrates that TIACBM is not limited to masked language models and is also effective for autoregressive models such as GPT-2.

### Key Findings

1. **TIACBM achieves the best performance across all tasks and models**, and the improvements are statistically significant.
2. **The improvement is most significant on the PAN19 authorship attribution task** (from 58.24% to 60.60% on BERT), because masking functional words directly targets author style features.
3. **Cyclic decay outperforms simple decay**: Compared to Ankner's non-cyclic decay, the cyclic version (without task information) already shows improvement on PAN19.
4. **Task information is the core contribution**: In comparing cyclic decay (without task information) and TIACBM (with task information), the latter performs better across all tasks.
5. **MLM is beneficial not only for pre-training but also for fine-tuning**—this is an important empirical finding.

## Highlights & Insights

1. **Simplistic yet Effective**: The method is straightforward to implement, requiring no additional models or data, only a task-specific token importance function.
2. **Clear Theoretical Intuition**: Masking discriminative features -> Preventing feature co-adaptation -> Similar to targeted data augmentation/regularization.
3. **Broad Applicability**: Effective on both masked LMs (BERT/RoBERTa) and autoregressive LMs (GPT-2).
4. **Novel Curriculum Learning Perspective**: This work is the first to explicitly link masking rate scheduling with anti-curriculum learning and validate it during the fine-tuning phase.

## Limitations & Future Work

- There is no universal optimal masking rate schedule; $K$ and decay schemas require user tuning.
- Designing task-specific token importance functions can be non-intuitive for certain tasks (e.g., regression tasks, multi-label tasks).
- Experiments were only conducted on 3 types of tasks, without covering more task categories such as NER, QA, or summarization.
- The interaction effects with other regularization techniques (e.g., Dropout, R-Drop) have not been explored.

## Related Work & Insights

- Ankner et al. (2024) and Yang et al. (2023) found that decaying masking rates are beneficial for pre-training; this work extends it to fine-tuning and incorporates task information.
- Cart-Stra-CL++ by Poesina et al. (2024) utilizes data maps for easy-to-hard curriculum learning, but requires double the training time.
- Jarca et al. (2024) found in the vision domain that easy-to-hard curricula are better for masked modeling, whereas the conclusion is opposite in the text domain (where hard-to-easy is superior).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of task-informed masking and cyclic anti-curriculum is an original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A systematic comparison of 3 models x 4 datasets x 5 baselines with rigorous statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Methodology is clearly described and algorithm pseudocode is well-structured.
- **Value**: ⭐⭐⭐ — The performance gain is modest (~1%), but the method is simple and easy to integrate into existing fine-tuning pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)

</div>

<!-- RELATED:END -->
