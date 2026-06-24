---
title: >-
  [Paper Note] Large Vocabulary Size Improves Large Language Models
description: >-
  [ACL 2025][LLM Pretraining][Vocabulary size] Experiments demonstrate that larger subword vocabulary sizes consistently improve LLM performance on downstream tasks. This work also proposes a simple vocabulary replacement method (Swap & Insert) for switching to a more appropriate vocabulary in continual training scenarios.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Vocabulary size"
  - "subword"
  - "pretraining"
  - "continual training"
  - "embedding replacement"
  - "monolingual LLMs"
date: 2026-05-08
content_hash: 8481d72343fa74f5
---

# Large Vocabulary Size Improves Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2406.16508](https://arxiv.org/abs/2406.16508)  
**Code**: Based on Megatron-LM / SentencePiece  
**Area**: LLM Pretraining / Tokenization  
**Keywords**: Vocabulary size, subword, pretraining, continual training, embedding replacement, monolingual LLMs  

## TL;DR

Experiments demonstrate that larger subword vocabulary sizes consistently improve LLM performance on downstream tasks. This work also proposes a simple vocabulary replacement method (Swap & Insert) for switching to a more appropriate vocabulary in continual training scenarios.

## Background & Motivation

- **Problem Definition**: When building an LLM, how does the subword vocabulary size affect model performance? Is the current mainstream choice (30k-60k) truly optimal?
- **Limitations of Prior Work**: From BERT to GPT and Llama, vocabulary sizes of 30k-60k have become "magic numbers" with almost no papers justifying their choice. Although some studies discuss the benefits of larger vocabularies on inference efficiency (e.g., Falcon), a systematic study of their impact on downstream task quality is still lacking.
- **Core Motivation**: (1) A larger vocabulary means each token carries more information, allowing the same text to be represented with fewer tokens. Does this enable the model to learn better representations? (2) In continual training scenarios (such as adapting an English LLM to Japanese), can performance be improved by replacing the vocabulary?
- **Key Challenge**: A larger vocabulary increases the parameter size of the embedding and output layers. This requires a fair comparison under two settings: fixed token count and fixed epoch count. Furthermore, replacing the vocabulary during continual training requires addressing the initialization of the embedding matrix.

## Method

### Overall Architecture

The study is divided into two parts: (1) Pretraining from scratch—training 680M parameter GPT-3 Large models on English and Japanese with vocabulary sizes of 5k/10k/50k/100k/500k, respectively; (2) Continual training—switching vocabularies based on Llama2-7B for Japanese continual training.

### Key Designs

1. **Fair Experimental Design**: Since different vocabulary sizes yield different numbers of tokens (e.g., an English 5k vocabulary generates 830B tokens while 500k generates 640B), two training configurations are designed—fixed 1T tokens and fixed 1 epoch—to ensure fairness without favoring either large or small vocabularies.
2. **Swap & Insert Vocabulary Replacement**: For continual training, a new target language vocabulary $V_{new}$ is constructed independently. A new embedding matrix is built from the original embedding matrix $E_{orig}$: $E_{new} = \frac{W \cdot E_{orig}}{\sqrt{|V_{orig}|}}$, where $W$ is a standard normal random matrix. The Insert strategy further retains the pretrained embeddings of subwords in the intersection of the original and new vocabularies to inherit existing knowledge.
3. **Scaled Embed for Stable Training**: Scaled embed technology (Takase et al., 2024) is used to maintain training stability when training with large vocabularies.

### Loss & Training

Standard autoregressive language modeling negative log-likelihood loss (next-token prediction).

## Key Experimental Results

### Main Results I: English Pretraining from Scratch

| Vocabulary Size| PIQA | OBQA | HellaSwag | WinoGrande | ARC-e | ARC-c | Average |
|-----------|------|------|-----------|------------|-------|-------|------|
| 5k (1T tokens) | 69.9 | 33.2 | 51.0 | 55.2 | 49.6 | 27.7 | 47.8 |
| 10k | 71.2 | 33.4 | 51.5 | 55.2 | 50.6 | 27.1 | 48.2 (+0.4) |
| 50k | 71.7 | 32.8 | 53.9 | 54.5 | 50.8 | 27.7 | 48.6 (+0.8) |
| 100k | 70.9 | 33.4 | 53.9 | 54.8 | 54.3 | 27.7 | 49.2 (+1.4) |
| **500k** | **71.4** | **34.0** | **55.3** | **57.5** | **55.1** | **28.3** | **50.3 (+2.5)** |

Expanding the vocabulary size from 5k to 500k yields a gain of +2.5 points in the average score of English commonsense reasoning tasks.

### Main Results II: Japanese Pretraining from Scratch

| Vocabulary Size | JSQuAD | JCQA | XWinograd | JAQKET | Average |
|-----------|--------|------|-----------|--------|------|
| 5k (1T tokens) | 58.1 | 68.1 | 58.9 | 12.5 | 49.4 |
| 100k | 62.1 | 71.9 | 59.6 | 34.9 | 57.1 (+7.7) |
| **500k** | **64.5** | **71.6** | **59.3** | **38.9** | **58.6 (+9.2)** |

The improvement for Japanese is even more significant (+9.2), especially on the JAQKET task (which requires answer generation), rising from 12.5 to 38.9.

### Ablation Study: Vocabulary Replacement in Continual Training (Llama2 → Japanese)

| Method | JSQuAD | JCQA | XWinograd | JAQKET | Average |
|------|--------|------|-----------|--------|------|
| Llama2 Original Vocabulary (32k) | 80.7 | 79.4 | 72.6 | 47.7 | 70.1 (+17.7) |
| Swap (100k, Random Initialization) | 79.2 | 80.2 | 67.5 | 56.3 | 70.8 (+18.4) |
| **Swap & Insert (100k)** | **81.9** | **80.2** | **69.2** | **61.2** | **73.1 (+20.7)** |
| Fujii et al. (2024) Method | 81.6 | 77.6 | 69.1 | 61.1 | 72.4 (+20.0) |

Even a randomly initialized new embedding (Swap) can outperform the original vocabulary. Adding the Insert strategy provides further improvements, surpassing prior specialized methods.

### Key Findings

- **Consistent Improvement with Large Vocabulary**: Whether for English or Japanese, and under either a fixed token count or a fixed epoch count, large vocabularies consistently deliver better performance.
- **Generation Tasks Benefit the Most**: Japanese JAQKET (Q&A without candidate answers) shows the largest improvement from 12.5 to 38.9, indicating that large vocabularies are particularly beneficial for generation capabilities.
- **Improved Training Efficiency**: With a larger vocabulary, the same text requires fewer tokens, reducing GPU computation time by approximately 30% (Japanese 100k vs 5k).
- **Feasibility of Vocabulary Replacement in Continual Training**: Even with a completely new vocabulary and randomly initialized embeddings, the model can still converge after continual training and outperform the original vocabulary.

## Highlights & Insights

- This is the first systematic study of the effect of vocabulary size on LLM downstream task performance, filling a long-standing empirical gap.
- Rigorous experimental design: Fair comparisons are ensured by controlling both token count and epoch count.
- The proposed Swap & Insert method is extremely simple yet effective, providing a practical vocabulary replacement scheme for cross-lingual continual training.
- The finding that a larger vocabulary improves both performance and training efficiency is a win-win conclusion with broad practical significance.

## Limitations & Future Work

- The method is only validated on English and Japanese; although it does not depend on language-specific features, multilingual generalization experiments are lacking.
- The model scale is limited (only 680M for pretraining from scratch, and 7B for continual training). The conclusions might differ on larger models (10B+).
- The vocabulary size is scaled up to 500k, making it impossible to observe the trends and upper bounds for more extreme sizes (e.g., 1M+).
- Large vocabularies increase the computational overhead of the output layer, requiring optimization techniques such as adaptive softmax for practical applications.

## Related Work & Insights

- **Vocabulary Construction**: BPE (Sennrich et al., 2016) and Unigram (Kudo, 2018) are the two mainstream tokenization algorithms. BERT used 30k and GPT used 40k, with almost no explanation ever provided.
- **Studies on Vocabulary Size**: Kiyono et al. (2020) studied the relationship between vocabulary size and performance, but only up to 32k. Ali et al. (2024) conducted a broader analysis but concluded that the "impact is negligible", which contradicts the findings of this paper.
- **Multilingual Vocabulary Expansion**: Studies like Fujii et al. (2024) and Kim et al. (2024) have explored expanding vocabularies during continual training, but their initialization strategies are more complex. The random projection initialization in this work is much simpler.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 6 |
| Experimental Thoroughness | 8 |
| Writing Quality | 9 |
| Value | 8 |
| **Total Score** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)

</div>

<!-- RELATED:END -->
