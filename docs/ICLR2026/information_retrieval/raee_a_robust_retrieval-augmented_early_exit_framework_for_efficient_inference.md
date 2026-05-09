---
title: >-
  [Paper Note] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference
description: >-
  [ICLR 2026][Early Exit] This paper proposes RAEE, a retrieval-augmented early exit framework that requires no classifier training. By retrieving exit information from semantically similar samples, RAEE dynamically determines the optimal exit layer, simultaneously accelerating inference and correcting model mispredictions — achieving a dual gain in both efficiency and performance.
tags:
  - ICLR 2026
  - Early Exit
  - Retrieval Augmentation
  - Distribution Prediction
  - Inference Acceleration
  - Error Correction
date: 2026-05-08
content_hash: d50a966e1a5274fa
---

# RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference

**Conference**: ICLR 2026
**arXiv**: [2405.15198](https://arxiv.org/abs/2405.15198)
**Code**: [GitHub](https://github.com/HugeRaabbit/RAEE)
**Area**: Information Retrieval
**Keywords**: Early Exit, Retrieval Augmentation, Distribution Prediction, Inference Acceleration, Error Correction

## TL;DR

This paper proposes RAEE, a retrieval-augmented early exit framework that requires no classifier training. By retrieving exit information from semantically similar samples, RAEE dynamically determines the optimal exit layer, simultaneously accelerating inference and correcting model mispredictions — achieving a dual gain in both efficiency and performance.

## Background & Motivation

Inference efficiency of large language models (LLMs) is a core challenge in deployment. Early exit, which terminates inference at intermediate layers to reduce latency and memory overhead, represents an advanced model compression technique.

Existing early exit frameworks fall into three categories:
- **Training-based methods** (e.g., DeeBERT): jointly optimize internal classifiers and the backbone, incurring high training costs
- **Semi-training methods** (e.g., AdaInfer): freeze the backbone and train lightweight classifiers, relying on manual feature engineering
- **Training-free methods** (e.g., HashEE): employ heuristic exit criteria, lacking adaptability with notable performance degradation

**Key Observation**: Existing methods universally trade accuracy for speed, overlooking the error-correction potential of early exit. The authors identify:

**Early exit as an error-correction mechanism**: Intermediate layers sometimes yield better predictions than the final layer. When the full model predicts incorrectly, on average 90.66% of errors can be corrected by intermediate-layer outputs.

**Highly consistent exit behavior among semantically similar samples**: The correct-prediction probability patterns of the top-8 nearest neighbors are nearly identical.

## Method

### Overall Architecture

RAEE operates in two phases: the **Build Phase** and the **Inference Phase**.

### 1. Exit Feature Collection and Retrieval Database Construction

Given training data $\mathcal{D} = \{(x_i^{train}, y_i^{train})\}$ and a backbone model $\mathcal{M}$ with $m$ layers, key-value pairs are constructed as follows:

**Keys**: Input embedding representations obtained via encoder $\mathcal{E}$:

$$\mathcal{K} = \{e_i\}_{i=1}^{|\mathcal{D}|} = \{\mathcal{E}(x_i^{train})\}_{i=1}^{|\mathcal{D}|}$$

**Values**: Candidate exit layers and their corresponding correct-prediction probabilities for each sample:

$$\mathcal{V} = \{v_i\}_{i=1}^{|D|} = \left\{\{(l_i^j, p_i^j)\}_{j=1}^{m_i}\right\}_{i=1}^{|D|}$$

An approximate nearest neighbor index is built using FAISS.

### 2. Retrieval-Augmented Early Exit Mechanism

The exit layer is modeled as a random variable $z \in \{1, \ldots, m\}$. The distribution $F$ is approximated by retrieving exit information from the top-$k$ nearest neighbors:

$$P(z=l|x) = \sum_{i=1}^{k} P(v_i|x) \cdot S_i$$

where the contribution weight is based on the inverse distance:

$$P(v_i|x) = \frac{\min(\{distance(v_j, x)\}_{j=1}^k)}{distance(v_i, x)}$$

The final exit layer is selected as the layer that maximizes the probability:

$$f(x) = \arg\max_l P(z=l|x)$$

### 3. Loss & Training

A threshold of $\tau = 0.9$ is used to filter low-confidence exit layer information, and the number of retrieved neighbors is set to $k = 12$. When multiple exit layers share the same maximum probability, the earliest layer is selected to maximize acceleration.

## Key Experimental Results

### Main Results: Performance Comparison on 8 Downstream Tasks

| Method | Backbone | SST-2 | SST-5 | MR | CR | MPQA | SUBJ | TREC | CoLA | Avg |
|--------|----------|-------|-------|-----|-----|------|------|------|------|-----|
| RoBERTa-Large | Full Model | 83.60 | 34.98 | 80.80 | 79.55 | 67.60 | 51.45 | 32.40 | 2.03 | 54.05 |
| DeeBERT | RB-L | 52.29 | 18.05 | 50.60 | 50.00 | 75.95 | 80.85 | 16.20 | 0.00 | 42.99 |
| AdaInfer | RB-L | 50.92 | 24.48 | 50.00 | 50.00 | 60.90 | 50.85 | 22.60 | -1.62 | 38.52 |
| **RAEE** | **RB-L** | **84.63** | **33.57** | **81.55** | **68.05** | **78.55** | **84.05** | **62.40** | **14.48** | **63.41** |
| SLEB | Llama-3 | 54.01 | 21.09 | 51.10 | 49.45 | 55.65 | 49.95 | 14.00 | 0.92 | 37.02 |
| **RAEE** | **Llama-3** | **73.05** | **35.25** | **66.45** | **57.95** | **75.05** | **90.05** | **51.80** | **9.55** | **57.39** |

**Key Findings**: Across all backbone models, RAEE not only accelerates inference but also surpasses the full model's average performance (63.41 vs. 54.05), breaking the traditional early exit paradigm of trading accuracy for speed.

### Ablation Study: Impact of the Correct-Prediction Retrieval Database

| Model | SST-2 | SST-5 | TREC | CoLA | Avg |
|-------|-------|-------|------|------|-----|
| Llama-3-8B Full Model | 62.84 | 26.06 | 8.40 | 0.00 | 41.80 |
| RAEE w/o correct filtering | 60.55 | 24.52 | — | — | — |
| RAEE w/ correct filtering | 73.05 | 35.25 | 51.80 | 9.55 | 57.39 |

Restricting the retrieval database to correctly predicted samples is the key factor behind RAEE's performance gains.

## Highlights & Insights

1. **Paradigm shift**: This work is the first to demonstrate that early exit functions not only as an acceleration technique but also as a dynamic error-correction mechanism, challenging the traditional efficiency–accuracy trade-off.
2. **Training-free design**: No classifiers or model parameters need to be trained; the retrieval database is constructed solely through model inference.
3. **Cross-model generality**: The framework proves effective across diverse architectures including RoBERTa, T5, Llama-3, and Gemma.
4. **Theoretical insight**: Early exit is formulated as a distribution prediction problem, with the exit distribution approximated via exit information from semantically similar samples.

## Limitations & Future Work

- Constructing and maintaining an external retrieval database introduces additional storage overhead.
- The retrieval process adds extra latency, though retrieval is performed only once at the start of inference.
- Performance depends on the embedding quality of the pre-trained encoder.
- Generalization to out-of-distribution data remains to be validated.

## Related Work & Insights

- **Training-based early exit**: DeeBERT, ElasticBERT, PABEE — require training internal classifiers
- **Training-free early exit**: HashEE, CALM — rely on heuristic exit criteria
- **Semi-training methods**: AdaInfer — SVM-based approach
- **Retrieval-augmented inference**: REALM, RAG — related work on retrieval-augmented generation

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐ | Combines retrieval augmentation with early exit; introduces an error-correction perspective |
| Practicality | ⭐⭐⭐⭐ | Training-free, cross-model generalizable, directly deployable |
| Experimental Thoroughness | ⭐⭐⭐⭐ | 8 tasks, 4 backbone models, comprehensive comparisons |
| Writing Quality | ⭐⭐⭐⭐ | Clear observation–method–experiment narrative structure |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)

<!-- RELATED:END -->
