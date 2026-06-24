---
title: >-
  [Paper Note] Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities
description: >-
  [ACL 2025][Self-Supervised Learning][Decoder Augmentation] This paper proposes Magnet, a method that augments decoder-only LLMs simultaneously into text encoders and infilling models using a hybrid attention mechanism (bidirectional + causal) and three self-supervised objectives (masked prediction + contrastive learning + missing span generation). It outperforms specialized methods like LLM2Vec on token-level and sentence-level representation learning tasks while avoiding the…
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Decoder Augmentation"
  - "Bidirectional Attention"
  - "Representation Learning"
  - "Text Infilling"
  - "Joint Training"
  - "Repetition Issue"
date: 2026-05-08
content_hash: 1f98923a62fdf3b8
---

# Magnet: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities

**Conference**: ACL 2025  
**arXiv**: [2501.08648](https://arxiv.org/abs/2501.08648)  
**Code**: Not released  
**Authors**: Savya Khosla, Aditi Tiwari, Kushal Kafle, Simon Jenni, Handong Zhao, John Collomosse, Jing Shi  
**Institutions**: Adobe Research, University of Illinois Urbana-Champaign  
**Area**: Self-Supervised Learning / Language Model Unification  
**Keywords**: Decoder Augmentation, Bidirectional Attention, Representation Learning, Text Infilling, Joint Training, Repetition Issue

## TL;DR

This paper proposes Magnet, a method that augments decoder-only LLMs simultaneously into text encoders and infilling models using a hybrid attention mechanism (bidirectional + causal) and three self-supervised objectives (masked prediction + contrastive learning + missing span generation). It outperforms specialized methods like LLM2Vec on token-level and sentence-level representation learning tasks while avoiding the severe text repetition issue caused by bidirectionality.

## Background & Motivation

**Limitations of Decoder LLMs**:
   - Causal attention limits the understanding of bidirectional context, leading to poor performance in tasks requiring global context, such as sentiment analysis and NER.
   - Text infilling tasks require understanding both left and right contexts, an ability naturally lacking in causal decoders.

**Limitations of Prior Work**:
   - Methods for representation learning (like LLM2Vec and Echo Embeddings) convert causal attention into bidirectional attention, which breaks the generation capability.
   - Methods for text infilling (like GLM and InCoder) enable infilling capabilities but fail to produce high-quality text representations.
   - **No prior work has simultaneously endowed LLMs with both representation learning and infilling capabilities.**

**Key Challenge**: When converting LLMs to bidirectional models, text generation suffers from severe sentence/phrase repetition (e.g., LLM2Vec increases the repetition rate by 36.5 times). The fundamental cause is the lack of autoregressive objectives during training.

**Goal**: To unify text understanding and generation within a single framework, leveraging the synergetic effects of joint training to improve various capabilities.

## Method

### Overall Architecture

Magnet fine-tunes pretrained LLMs (such as Llama-2-7B). Its core innovations lie in two aspects: (1) a hybrid attention mask, and (2) three self-supervised training objectives.

### 3.1 Hybrid Attention Mechanism

The input tokens are divided into two categories:

1. **Context tokens (blue)**: Utilize **fully bidirectional attention** among themselves, meaning each context token can attend to all other context tokens. This achieves encoder-style bidirectional understanding.
2. **Span tokens (green)**: Can attend to **all context tokens** (bidirectionally), but use **causal attention** among themselves. This enables infilling-style generation.

**Key Insight**: Maintaining causal attention among span tokens is crucial to avoiding the repetition issue—the purely bidirectionalized LLM2Vec suffers from generation degradation precisely because it fully enables bidirectional attention.

Three modes are supported during inference:
- Pure Causal: Traditional left-to-right text generation.
- Pure Bidirectional: Representation learning tasks.
- Hybrid: Text infilling.

### 3.2 Three Training Objectives

#### Objective 1: Masked Next Token Prediction (MNTP)

- Randomly masks 20% of the input tokens (80% replaced with [MASK], 10% random words, 10% kept unchanged, inheriting the BERT strategy).
- **Key Design**: Uses the output at position $l$ to predict the masked token at position $l+1$ (instead of predicting position $l$ at position $l$ as in BERT), maintaining consistency with the "predicting the next token" scheme in LLM pre-training.
- Loss Function: Cross-entropy, calculated only at masked positions.
- Applies only to context tokens.

#### Objective 2: Self-Supervised Contrastive Learning (SSCL)

- Uses a paraphrase model to generate an augmented view $x^+$ of the input.
- Uses the representation of the final token [EOS] as the sentence embedding.
- InfoNCE loss + in-batch negative samples.
- **Clever Design**: Selecting the final token decouples SSCL from MNTP (in MNTP, the output at position $l$ is used to predict the token at $l+1$, and the output of the final token does not participate in MNTP).
- Employs an instruction prefix: "Given the sentence, find its representation:"

#### Objective 3: Missing Span Generation (MSG)

- Removes one or more continuous spans from the input and requires the model to autoregressively generate the infilling content.
- Each span token $y_l$ is conditioned on all context tokens and preceding span tokens.
- Cross-entropy loss, computed only at span positions.
- **Side Benefit**: When all tokens are span tokens, it degrades to the standard next-token prediction task, preserving the generation capability.

#### Total Loss

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{MNTP}} + \lambda_2 \mathcal{L}_{\text{SSCL}} + \lambda_3 \mathcal{L}_{\text{MSG}}$$

### 3.3 Training Pipeline

For each training sample $x$:
1. Generate three views: (a) masked and spanned $x^m$, (b) original $x$ for SSCL, and (c) paraphrased $x^+$.
2. Process the three views through the same base model using different attention masks.
3. Calculate the three losses separately and compute their weighted sum.

## Key Experimental Results

### Token-level Representation Learning (CoNLL-2003)

| Model | Chunking | NER | POS-Tags |
|------|:---:|:---:|:---:|
| BERT-Large | 71.77 | 90.09 | 75.12 |
| DeBERTa-Large | 85.74 | 94.97 | 86.49 |
| StructBERT-Large | 89.99 | 97.31 | 90.86 |
| Llama-2-7B (Original) | 88.23 | 96.59 | 91.53 |
| $\text{LLM2Vec}^{\text{MNTP}}$ | 91.61 | 97.16 | 92.61 |
| **Magnet** | **92.64** | **98.31** | **93.34** |

**Key Findings**: Magnet consistently outperforms $\text{LLM2Vec}^{\text{MNTP}}$ (the version trained solely with MNTP) across all three tasks, demonstrating the synergetic effects of joint training. Notably, although Magnet incorporates additional SSCL and MSG objectives compared to LLM2Vec, these "extra" objectives do not interfere with representation learning but rather enhance it.

### Sentence-level Representation Learning (STS Benchmark)

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Average |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RoBERTa-Large + SimCSE | 72.86 | 83.99 | 75.62 | 84.77 | 81.80 | 81.98 | 71.26 | 78.90 |
| Llama-2-7B (Original) | 50.98 | 74.02 | 62.86 | 67.09 | 71.03 | 63.56 | 67.22 | 65.25 |
| LLM2Vec | 65.39 | 79.26 | 72.98 | 82.72 | 81.02 | 78.32 | 71.77 | 75.92 |
| **Magnet** | **67.98** | **84.66** | **77.67** | **84.17** | 79.44 | **82.88** | **78.77** | **79.36** |

Magnet outperforms LLM2Vec by approximately 3.4 percentage points on the average STS score, even surpassing encoders specifically trained for this purpose, such as RoBERTa-Large + SimCSE.

### Text Infilling (Perplexity)

| Method | ROC Stories PPL | Wikitext-103 PPL |
|------|:---:|:---:|
| Llama-2-7B | 13.93 | 22.04 |
| **Magnet** | **9.52** | **15.46** |

Magnet significantly reduces infilling perplexity. In human evaluations, the infills generated by Magnet are judged to be "contextually appropriate" in 62% of cases (compared to 53.5% for original Llama-2-7B, 5.5% for zero-shot, and 54.5% for five-shot).

### Repetition Analysis (One of the Core Contributions)

| Method | Rep-Sen (Wiki) | Rep-4 (Wiki) | Rep-Sen (ROC) | Rep-4 (ROC) |
|------|:---:|:---:|:---:|:---:|
| Llama-2-7B | 0.0056 | 0.0601 | 0.0381 | 0.0163 |
| LLM2Vec | 0.2044 | 0.4747 | 0.2945 | 0.5243 |
| **Magnet** | **0.0151** | 0.2047 | **0.0737** | 0.2573 |

- LLM2Vec increases the sentence repetition rate of Llama-2-7B by **36.5 times** (on Wikitext), whereas Magnet only increases it by 2.7 times.
- As training iterations increase, the repetition issue in LLM2Vec continuously worsens, while Magnet shows no such trend.
- **Analysis of Root Causes**: LLM2Vec is trained solely with bidirectional attention, reverting the decoder into a BERT-like model; Magnet's MSG objective preserves the autoregressive generation capability.

### Retention of Knowledge and Reasoning Capabilities

| Model | HellaSwag | BBH | ARC-Easy | ARC-Challenge | MMLU (Avg) |
|------|:---:|:---:|:---:|:---:|:---:|
| Llama-2-7B | 75.51 | 33.57 | 73.95 | 44.28 | 46.81 |
| Magnet | 75.08 | 32.22 | 74.33 | 44.52 | 45.98 |

Magnet has a minimal impact on pretrained knowledge, with variations across benchmarks limited to within 1-2 percentage points.

## Highlights & Insights

1. **Superiority of the Unified Framework**: By jointly training different objectives, positive synergies emerge among various capabilities. For instance, token-level representation learning is enhanced thanks to the MSG objective, which challenges the intuition that "multi-task learning inevitably leads to interference".
2. **Root Cause Analysis of the Repetition Issue**: This work presents the first systematic analysis of the repetition issue in text generation caused by bidirectionality, pointing out that purely bidirectional attention degrades LLMs to BERT-like models, which inherently suffer from generation repetition.
3. **Elegant Hybrid Attention Design**: The partitioned strategy—bidirectional for context tokens and causal for span tokens—allows a single attention mask to simultaneously serve both understanding and generation.
4. **Clever Motivation for Using the Final Token as Sentence Representation**: It decouples the representation from the MNTP objective, preventing the two tasks from competing at the same position.
5. **Parameter Efficiency**: It only requires fine-tuning on existing LLMs without the need for pre-training from scratch.

## Limitations & Future Work

1. The approach is only validated on Llama-2-7B, without testing larger models (13B/70B) or other architectures (e.g., Mistral, GPT).
2. SSCL relies on a paraphrase model to generate augmented views, and the quality of augmentation may affect the sentence representation performance.
3. Magnet still shows a noticeable degradation in the Rep-4 metric compared to the original model (Wikitext increases from 0.06 to 0.20), indicating that the repetition issue is not completely resolved.
4. Infilling inference requires knowing the missing positions and contexts, limiting its practical application scenarios.
5. Comparisons with newer models like Mistral/Qwen or instruction-tuned versions are missing.

## Related Work & Insights

- **Representation Learning**: LLM2Vec (BehnamGhader et al., 2024) transforms LLMs into encoders using MNTP+SimCSE but sacrifices generation capabilities; Echo Embeddings (Springer et al., 2024) captures bidirectional information by duplicating the input.
- **Text Infilling**: GLM (Du et al., 2021) employs autoregressive blank infilling, InCoder (Fried et al., 2022) rearranges training samples, and FIM (Bavarian et al., 2022) utilizes Fill-in-the-Middle pre-training.
- **Unification of Understanding and Generation**: XLNet (Yang et al., 2019) uses a permutation objective, and UniLM (Dong et al., 2019) uses multi-directional attention masks, but these methods require pre-training from scratch.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ It is the first to simultaneously endow LLMs with encoding, infilling, and generation capabilities within a single framework. The hybrid attention design is highly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ It covers multi-dimensional evaluations across token/sentence-level representations, infilling, generation, and knowledge retention, with an in-depth analysis of the repetition issue.
- **Value**: ⭐⭐⭐⭐ It equips LLMs with multiple capabilities without sacrificing their original functions, offering high practicality.
- **Writing Quality**: ⭐⭐⭐⭐ The diagrams and tables are well-designed with clear motivation and methodology explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Residual Connections Harm Generative Representation Learning](../../CVPR2026/self_supervised/residual_connections_harm_generative_representation_learning.md)
- [\[ACL 2025\] QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](qaencoder_aligned_representation.md)
- [\[ACL 2025\] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)
- [\[CVPR 2025\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2025/self_supervised/representation_learning_for_spatiotemporal_physical_systems.md)
- [\[ICLR 2026\] Disentanglement of Variations with Multimodal Generative Modeling](../../ICLR2026/self_supervised/disentanglement_of_variations_with_multimodal_generative_modeling.md)

</div>

<!-- RELATED:END -->
