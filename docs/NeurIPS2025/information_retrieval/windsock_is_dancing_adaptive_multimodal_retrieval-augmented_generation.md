---
title: >-
  [Paper Note] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation
description: >-
  [NeurIPS 2025][Multimodal RAG] This paper proposes a dual-component framework (Windsock + DANCE) to address three core challenges in multimodal RAG: the Windsock module adaptively determines **when to retrieve** and **which modality to retrieve** (text/image/none) based on the query; the DANCE instruction fine-tuning strategy improves **how to utilize** retrieved information by dynamically selecting the model's weakest modality for noise-robust training. The overall framework achieves a 17.07% performance improvement while reducing retrieval calls by 8.95%.
tags:
  - NeurIPS 2025
  - Multimodal RAG
  - Adaptive Retrieval
  - Modality Selection
  - Noise-Robust Training
  - Retrieval-Augmented Generation
date: 2026-05-08
content_hash: 5a849115fd9e8bd0
---

# Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.22694](https://arxiv.org/abs/2510.22694)
**Code**: Not available
**Area**: Information Retrieval
**Keywords**: Multimodal RAG, Adaptive Retrieval, Modality Selection, Noise-Robust Training, Retrieval-Augmented Generation

## TL;DR

This paper proposes a dual-component framework (Windsock + DANCE) to address three core challenges in multimodal RAG: the Windsock module adaptively determines **when to retrieve** and **which modality to retrieve** (text/image/none) based on the query; the DANCE instruction fine-tuning strategy improves **how to utilize** retrieved information by dynamically selecting the model's weakest modality for noise-robust training. The overall framework achieves a 17.07% performance improvement while reducing retrieval calls by 8.95%.

## Background & Motivation

Multimodal Retrieval-Augmented Generation (MRAG) enhances the factuality and timeliness of MLLMs by incorporating external knowledge. However, existing methods suffer from three critical limitations:

**The *When* Problem — Indiscriminate Retrieval**: Existing methods apply retrieval to all queries uniformly (retrieve-for-all strategy), even when the model's parametric knowledge suffices. This introduces unnecessary computational overhead, and noisy documents returned by unreliable retrievers can degrade answer quality.

**The *What* Problem — Rigid Modality Selection**: Existing methods either fix retrieval to images or to text (e.g., Wikipedia), without accounting for the varying informational needs of different queries. In practice, historical event queries benefit more from textual retrieval, whereas questions about painting styles are better served by image retrieval—different queries inherently require different modalities.

**The *How* Problem — Underutilization of Retrieved Information**: MLLMs are sensitive to irretrievable documents; both statistical (BM25) and vector-based (CLIP) retrievers may return irrelevant content, causing factual errors and hallucinations. Models must learn to extract useful information while discarding noise.

Prior works (e.g., ReflectiVA, mR2AG) have attempted adaptive retrieval but rely on **expensive manual or GPT-4 annotations** and overlook multimodal selection. Windsock addresses both the *when* and *what* problems simultaneously by constructing training data via **self-evaluation**, without any external annotation.

## Method

### Overall Architecture

The complete pipeline consists of three core components:
1. **Windsock Module** (lightweight classifier): Receives the user query and outputs a three-way decision—no retrieval (NA) / visual retrieval / textual retrieval.
2. **Retriever**: Retrieves top-$k$ documents from the corresponding modality knowledge base according to the Windsock decision.
3. **MLLM Generator** (fine-tuned with DANCE): Integrates the query and retrieved documents to generate the final answer.

### Key Designs

1. **Windsock — Query-Aware Adaptive Retrieval Decision Maker**:

   Built on a Flan-T5-Small backbone, Windsock implements the following three-way classification:
   $$c = \mathcal{W}(Q) \in \{\text{NA}, \text{Visual}, \text{Textual}\}$$

   The corresponding strategy is executed based on the classification result:
   $$\begin{cases} r^\varnothing = \mathcal{G}(Q, \varnothing), & \text{if } c = \text{NA} \\ r^V = \mathcal{G}(Q, \mathcal{R}(Q, \mathbb{D}^I)), & \text{if } c = \text{Visual} \\ r^T = \mathcal{G}(Q, \mathcal{R}(Q, \mathbb{D}^T)), & \text{if } c = \text{Textual} \end{cases}$$

   Design advantages: (a) skipping unnecessary retrievals reduces overhead and noise; (b) selecting the most appropriate modality improves information quality; (c) modular design enables **plug-and-play** integration with arbitrary MLLMs. Experiments show that Windsock adds only 10.25 ms (1.83%) of inference overhead.

2. **Self-Evaluation Training Data Construction** (no GPT-4 annotation required):

   For each QA pair $\{Q, A\}$, the MLLM generates answers under three strategies (no retrieval / visual retrieval / textual retrieval) and scores each using downstream task metrics (e.g., F1):
   $$s^\varnothing = \epsilon(r^\varnothing, A), \quad s^I = \epsilon(r^I, A), \quad s^T = \epsilon(r^T, A)$$
   $$c^* = \arg\max_c(s^\varnothing, s^I, s^T)$$

   The optimal strategy $c^*$ serves as the training label for Windsock. This approach leverages the MLLM's own capabilities to evaluate different strategies without external annotation. It also identifies cases where retrieval is harmful (when $s^\varnothing > \max(s^I, s^T)$).

3. **DANCE — Dynamic Noise-Resistant Instruction Fine-Tuning**:

   Core Idea: Rather than injecting noise randomly, DANCE intelligently identifies the model's **weakest modality** for targeted training. Specifically, for each sample, the modality on which the MLLM performs worst is selected:
   $$\arg\min_M (s^I, s^T) \in \{I, T\}$$

   Retrieved results from this modality are highly likely to contain noise or irrelevant information. These "hard samples" are used to construct instruction fine-tuning data: $\{Q, \mathcal{R}(Q, \mathbb{D}^M), A\}$, and the model is trained via standard instruction tuning to extract useful information from noisy contexts.

   Compared to SURf: SURf uses image similarity for hard sample mining and requires per-document response generation, whereas DANCE efficiently identifies difficult cases via downstream metrics with parallel processing across modalities, achieving 2× faster data construction.

### Loss & Training

- **Windsock Training**: AdamW optimizer (lr=5e-4), batch size 16, 5 epochs, cross-entropy loss with class weights to handle training data imbalance.
- **DANCE Instruction Fine-Tuning**: Default LoRA configuration from LLaMA-Factory, 1 epoch.
- **Retriever**: VBGE-base, returning top-3 retrieved results.

## Key Experimental Results

### Main Results — WebQA F1 Score

| Method | Generator | Single | Multiple | All |
|--------|-----------|:---:|:---:|:---:|
| Zero-Shot | Qwen2-VL-7B | 61.76 | 37.09 | 44.04 |
| Vanilla RAG | Qwen2-VL-7B | 62.96 | 38.36 | 45.29 |
| Windsock only | Qwen2-VL-7B | 65.92 | 38.63 | 46.32 |
| SURf | Qwen2-VL-7B-SURf | 62.72 | 55.60 | 57.61 |
| **DANCE** | Qwen2-VL-7B-DANCE | 66.42 | 57.45 | 59.97 |
| **Windsock+DANCE** | Qwen2-VL-7B-DANCE | **70.12** | **59.32** | **62.36** |

### Ablation Study — Retrieval Strategy Efficiency Analysis

| Retrieval Strategy | Time (s)↓ | Single↑ | Multiple↑ | All↑ |
|--------------------|:---:|:---:|:---:|:---:|
| NA (no retrieval) | 0.46 | 61.76 | 37.09 | 44.04 |
| Visual only | 0.67 | 64.88 | 36.46 | 44.47 |
| Textual only | 0.79 | 52.87 | 36.70 | 41.25 |
| **Windsock** | **0.56** | **65.92** | **38.63** | **46.32** |

### Ablation Study — Instruction Fine-Tuning Strategy Comparison

| Strategy | Single | Multiple | All |
|----------|:---:|:---:|:---:|
| Easy (train on best modality) | 58.83 | 51.24 | 53.38 |
| Random (random modality selection) | 60.98 | 53.94 | 55.12 |
| **DANCE (train on worst modality)** | **66.42** | **57.45** | **59.97** |

### Key Findings

- **Adaptive retrieval outperforms fixed strategies across the board**: Windsock surpasses both pure Visual and pure Textual retrieval while maintaining inference time between the two. On WebQA, 8.96% of queries bypass retrieval; when simple MS-COCO queries are included, the skip rate rises to 26.99%, demonstrating strong adaptability.
- **Training on weaknesses is key to DANCE**: Training on the model's worst-performing modality (DANCE) outperforms training on the best (Easy) by 6.59% and random selection (Random) by 4.85%, demonstrating the superiority of targeted training.
- **Complementary gains from Windsock+DANCE**: Windsock addresses the input side (what information to provide), while DANCE addresses the model side (how to use that information); their combination yields the greatest improvement.
- **Side effects of DANCE**: Performance on the general MLLM benchmark MME decreases somewhat, reflecting a trade-off between specialized and general capabilities.

## Highlights & Insights

1. **Clarity of problem decomposition**: The MRAG problem is cleanly decomposed into three dimensions—*when*, *what*, and *how*—each addressed by a targeted solution.
2. **Self-evaluation as a substitute for GPT-4 annotation** is a practical and economical approach: directly using the target model to evaluate the effectiveness of different strategies eliminates the need for expensive external annotators. Data construction is 2× faster than SURf.
3. **"Learn from failure" training philosophy**: DANCE selects the model's weakest modality for training—analogous to curriculum learning but in the opposite direction (hard-first), and experiments confirm this counterintuitive strategy yields the best results.
4. **Extremely lightweight design of Windsock**: Using Flan-T5-Small as the backbone, Windsock achieves significant efficiency and performance gains with only a 1.83% increase in inference overhead.

## Limitations & Future Work

- Currently supports only text and image modalities; other modalities such as tables are not yet supported.
- DANCE fine-tuning leads to performance degradation on the general MLLM benchmark (MME); the balance between specialization and generalization warrants further exploration.
- Windsock uses a text-only backbone (Flan-T5) and does not leverage visual information potentially present in the query.
- The quality of self-evaluation training data depends on the quality of both the base MLLM's responses and the retriever's results.
- The three-way classification (NA/Visual/Textual) may be too coarse-grained; hybrid retrieval across modalities is not supported.

## Related Work & Insights

- **Inspiration from Self-RAG**: Self-RAG uses special tokens to decide whether to retrieve; Windsock extends this idea to the multimodal setting and adds a modality selection dimension.
- **Comparison with SURf**: SURf enhances the model's ability to select and utilize information through instruction fine-tuning, but relies on image similarity for hard sample mining. DANCE uses downstream metrics for direct evaluation, which is more efficient and better aligned with the task.
- **Implications for MRAG system design**: Future MRAG systems should treat "whether to retrieve" and "what to retrieve" as learnable decisions rather than fixed strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The when+what+how three-dimensional decomposition and self-evaluation data construction approach are novel and practically valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets (WebQA and MultimodalQA), multiple baselines, and comprehensive ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, thorough experimental analysis, and rich visualizations.
- **Value**: ⭐⭐⭐⭐ Provides direct guidance for real-world deployment of MRAG systems; the self-evaluation pipeline is ready to use out of the box.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)

<!-- RELATED:END -->
