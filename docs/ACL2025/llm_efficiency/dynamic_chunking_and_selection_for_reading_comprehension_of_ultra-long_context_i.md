---
title: >-
  [Paper Note] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models
description: >-
  [ACL 2025][LLM Efficiency][long-context QA] Proposes Dynamic Chunking and Selection (DCS), which addresses semantic fragmentation caused by fixed chunking in long texts through semantic similarity-based dynamic chunking and question-aware classifier-based chunk selection. Using Llama3 as the base model, it achieves a single-hop average of 35.50 (+28.6%) and a multi-hop average of 29.07 (+20.0%) across 12 long-text QA datasets, while maintaining robustness under 256k token inp…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "long-context QA"
  - "dynamic chunking"
  - "chunk selection"
  - "reading comprehension"
  - "semantic segmentation"
date: 2026-05-08
content_hash: 33ea13858d4205e2
---

# Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.00773](https://arxiv.org/abs/2506.00773)  
**Code**: [GitHub](https://github.com/ECNU-Text-Computing/DCS)  
**Area**: LLM Efficiency  
**Keywords**: long-context QA, dynamic chunking, chunk selection, reading comprehension, semantic segmentation

## TL;DR

Proposes Dynamic Chunking and Selection (DCS), which addresses semantic fragmentation caused by fixed chunking in long texts through semantic similarity-based dynamic chunking and question-aware classifier-based chunk selection. Using Llama3 as the base model, it achieves a single-hop average of 35.50 (+28.6%) and a multi-hop average of 29.07 (+20.0%) across 12 long-text QA datasets, while maintaining robustness under 256k token inputs.

## Background & Motivation

**Background**: LLMs face two major bottlenecks in long-context reading comprehension: position encoding limits the context window length, and quadratic attention computational complexity constrains the actual processable length. Meanwhile, LLMs tend to focus on the beginning and end of inputs (the "lost in the middle" phenomenon).

**Limitations of Prior Work**: Existing long-text processing methods (e.g., InfLLM, StreamingLLM) typically partition inputs into chunks of **fixed lengths**. However, fixed truncation often cuts sentences in the middle, disrupting semantic integrity. As shown in Figure 1, splitting "Deep learning" into two chunks prevents the LLM from understanding the full meaning, leading to incorrect answers.

**Key Challenge**: The simplicity of fixed-length chunking vs. the necessity of semantic coherence — either keeping chunking simple but losing semantics, or requiring more intelligent chunking strategies.

**Goal**: How to compress long-context inputs and maintain answer accuracy under the premise of not modifying the LLM architecture, by utilizing semantic-aware dynamic chunking and question-related chunk selection.

**Key Insight**: Employ Sentence-BERT to encode sentence-level semantics, leveraging the semantic distance between adjacent sentences to adaptively determine chunk boundaries; then train a lightweight classifier to select relevant chunks based on the question.

**Core Idea**: The synergy of semantic similarity-based dynamic chunking to preserve semantic integrity, and a question-aware classifier to select relevant chunks, collaboratively resolving long-context comprehension.

## Method

### Overall Architecture

DCS consists of two core modules: **(1) Dynamic Chunking**—adaptive chunking based on semantic similarity; and **(2) Chunk Selection**—chunk filtering based on a question-aware classifier. Finally, the selected chunks are concatenated in their original order and fed into the LLM to generate answers.

### Key Designs

**1. Dynamic Chunking**

- Segment the long text into sentences based on punctuation, obtaining a sentence sequence $[s_0, s_1, \ldots, s_{n-1}]$
- Expand the context using neighbor merging: $s'_i = s_{i-1} \oplus s_i \oplus s_{i+1}$
- Encode with Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) to obtain embeddings
- Calculate the cosine distance between adjacent sentences: $\text{dis}(i) = 1 - \text{sim}(i, i+1)$
- Select the top-$(1-\alpha)$ proportion of positions with the largest distances as chunk boundaries
- Iterative refinement: Ensure each chunk does not exceed a predefined length $l$ (defaulting to 512 tokens)
- Merge overly small chunks to make them as close to the target length as possible

**2. Chunk Selection**

- Concatenate each chunk $c_i$ with the question, and feed it into the LLM to extract features
- Feature extraction strategy: Take the hidden states of boundary tokens + attention-weighted context/question representations, yielding a total of 6 $d$-dimensional vectors
- Train a 3-layer MLP classifier to predict the question relevance probability $T_i$ for each chunk
- Select the top-$k$ most relevant chunks based on the compression ratio $\alpha_c = l_C / l_T$
- Concatenate the selected chunks in their original order, then feed them along with the initial info and question into the LLM

### Loss & Training

The classifier is trained using binary cross-entropy loss:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log \sigma(h_\theta(H_i)) + (1-y_i)\log(1-\sigma(h_\theta(H_i)))]$$

The training data is based on AdversarialQA, with positive and negative samples constructed using a negative sampling strategy.

## Key Experimental Results

### Main Results (Llama-3-8B-Instruct)

| Method | Single-hop Avg | Multi-hop Avg | Total Avg |
|------|---------------|--------------|--------|
| Llama3 Original | 27.60 | 24.22 | - |
| + StreamingLLM | 24.26 | 22.51 | - |
| + LM-Infinite | 24.20 | 22.79 | - |
| + InfLLM | 27.15 | 23.58 | - |
| **+ DCS (Ours)** | **35.50** | **29.07** | - |

Representative single-hop results: Loogle_SD 45.10 (Original 21.25), Factrecall 29.89 (Original 15.50). Multi-hop: Musique 28.90 (Original 21.72), HotpotwikiQA 25.40 (Original 14.22).

### Ablation Study

| Component | Single-hop | Multi-hop | Avg |
|------|-----------|----------|-----|
| Llama3 + DC (Dynamic Chunking) | 38.10 | 38.06 | 38.08 |
| Llama3 + FC (Fixed Chunking) | 36.66 | 37.26 | 36.96 |

Dynamic chunking consistently outperforms fixed chunking, with an average improvement of 1.12.

### Hyperparameter Selection

| Chunk Length $l$ | Avg |
|---------------|-----|
| 256 | 37.33 |
| **512** | **38.08** |
| 768 | 36.87 |
| 1024 | 36.72 |

| Threshold $\alpha$ | Avg |
|--------------|-----|
| 55 | 37.43 |
| **60** | **38.08** |
| 65 | 37.67 |
| 70 | 36.97 |

### Key Findings

- DCS achieves improvements of 5.8% (single-hop) and 7.6% (multi-hop) with Mistral, and 24.9% and 7.3% with Vicuna.
- Across the input length range of 16k to 256k, the performance degradation of DCS is significantly smaller than the baselines, with a widening advantage particularly above 64k.
- The MLP classifier significantly outperforms the cosine similarity-based selection scheme.

## Highlights & Insights

1. **Intuitive and effective**: Using semantic distance to determine chunk boundaries avoids semantic fragmentation from fixed division; the approach is natural and well-justified.
2. **Extremely low training overhead**: Requires training only a 3-layer MLP classifier without fine-tuning the LLM itself.
3. **Highly robust**: Maintains stable performance even on ultra-long texts up to 256k, significantly outperforming the degradation curves of baselines.
4. **Plug-and-play**: Compatible with any LLM (Llama3/Mistral/Vicuna) without altering the model architecture.
5. **Clever iterative refinement strategy**: Initial coarse division followed by merging ensures each chunk is neither overly long nor too fragmented.

## Limitations & Future Work

1. **Classifier reliance on base LLM feature extraction**: Requires a forward pass for each chunk-question pair to obtain hidden states, which incurs a non-trivial overhead when the chunk count is large.
2. **Validated only on QA tasks**: Performs no assessment on other long-context tasks such as summarization or translation.
3. **Classifier training data is sourced from short texts** (AdversarialQA); whether this can generalize to more diverse domains remains to be explored.
4. **Limited base model scale (7B–8B)**: The performance on larger models or commercial APIs has not been verified.
5. **Sentence-BERT encoding quality directly impacts final chunking performance**: Text from different domains may require parameter tuning.

## Related Work & Insights

- **Compared to InfLLM**: InfLLM uses fixed chunking + memory block indexing, whereas DCS uses dynamic chunking + classifier selection; their directions are complementary.
- **Compared to HMT**: HMT simulates human hierarchical memory, whereas DCS is more concise and direct.
- **Compared to Token Eviction (H2O/TOVA)**: Token-level eviction disrupts the original semantic structures, whereas chunk-level operations preserve higher fidelity.
- **Chunk-level processing is an effective granularity for long-context understanding**—it is neither too fine-grained like token-level methods nor too coarse-grained like full-paragraph approaches.

## Rating

- Novelty: ⭐⭐⭐ — While the idea of dynamic chunking is not new, the combined design of semantic-distance driving and question-aware classification is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage with 12 datasets, 3 LLMs, hyperparameter tuning, ablation, and ultra-long text evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical formulations, intuitive diagrams, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ — A simple yet effective long-context processing solution with high practicality, suitable for engineering deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)

</div>

<!-- RELATED:END -->
