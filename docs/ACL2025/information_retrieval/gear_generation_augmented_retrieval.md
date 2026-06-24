---
title: >-
  [Paper Note] GeAR: Generation Augmented Retrieval
description: >-
  [ACL 2025][Information Retrieval & RAG][document retrieval] GeAR introduces a fusion encoder and a text decoder into the traditional bi-encoder retrieval framework. It enhances the retrieval model's comprehension of fine-grained internal semantics of documents through generation tasks, without introducing additional computational overhead for global retrieval.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "document retrieval"
  - "fine-grained retrieval"
  - "generation-augmented"
  - "bi-encoder"
  - "contrastive learning"
date: 2026-05-08
content_hash: 290763eeb8ab442d
---

# GeAR: Generation Augmented Retrieval

**Conference**: ACL 2025  
**arXiv**: [2501.02772](https://arxiv.org/abs/2501.02772)  
**Code**: [Available](https://github.com/microsoft/LMOps)  
**Area**: Others  
**Keywords**: document retrieval, fine-grained retrieval, generation-augmented, bi-encoder, contrastive learning

## TL;DR

GeAR introduces a fusion encoder and a text decoder into the traditional bi-encoder retrieval framework. It enhances the retrieval model's comprehension of fine-grained internal semantics of documents through generation tasks, without introducing additional computational overhead for global retrieval.

## Background & Motivation

The dominant approach in document retrieval is the bi-encoder, which encodes queries and documents into vectors separately and computes matching scores via cosine similarity. This approach suffers from several fundamental limitations:

**Information Bottleneck**: The complex semantic relationship between queries and documents is compressed into a single scalar similarity score, making it difficult to provide interpretable retrieval results.

**Coarse-grained Matching**: For long documents (256-512+ tokens), bi-encoders only provide a global matching score, failing to locate specific passages or sentences within the document that are most relevant to the query.

**Downstream Task Demands**: Many downstream NLP tasks (such as sentence selection, search result highlighting, fine-grained citation, and needle-in-a-haystack) require the retrieval model to possess local, fine-grained comprehension capabilities, whereas traditional methods only optimize for global semantic alignment.

The authors pose a core question: **How can a retriever be endowed with the dual capabilities of both global and local comprehension?**

Achieving this goal faces two main challenges: (1) a lack of sufficient training data consisting of (query, document, local information) triplets; and (2) the optimal combination of model architecture, training objectives, and training methodologies has not been fully explored.

## Method

### Overall Architecture

Inspired by multimodal representation learning (e.g., ALBEF、BLIP), GeAR views documents and queries as two distinct "modalities." The architecture comprises three core components:

- **Bi-Encoder**: Encodes queries and documents independently for global document retrieval (with zero extra overhead).
- **Fusion Encoder**: Deeply fuses document embeddings with query embeddings using cross-attention.
- **Text Decoder**: Generates fine-grained information related to the query from the document based on the fused representation.

### Key Designs

1. **Bi-Encoder (Global Retrieval)**: This component initializes two independent encoders, $E_d(\cdot)$ and $E_q(\cdot)$, to encode documents and queries separately, applying mean pooling to obtain textual embeddings. During inference, bi-encoder similarity is computed directly, which is entirely consistent with classic retrieval methods and introduces no additional computational cost. The design motivation is to retain retrieval efficiency while incorporating local comprehension capabilities.

2. **Fusion Encoder (Local Information Interaction)**: This component shares most of its parameters with the query encoder but introduces lightweight, learnable cross-attention modules at each layer. Document embeddings are fused with query embeddings through cross-attention. The core idea is to enable each token in the document to perceive the query intent, thereby yielding a meaningful attention weight distribution. These cross-attention weights can be utilized directly to rank sentences within the document, facilitating local information retrieval.

3. **Text Decoder (Generation Task Augmentation)**: Receiving the output representations from the fusion encoder, this component employs unidirectional causal attention to autoregressively generate fine-grained text relevant to the query. It generates answers in question-answering (QA) scenarios, and outputs the most relevant sentences from the document in relevant information retrieval (RIR) scenarios. A special `[Decode]` token is introduced to signify the start of generation. The design motivation is to explicitly model the alignment between queries and fine-grained semantic segments of documents through generation tasks.

4. **Data Construction**: Two retrieval scenarios are defined:

    - Question-Answering Retrieval (QAR): 30M training samples extracted from the PAQ dataset.
    - Relevant Information Retrieval (RIR): An LLM-assisted synthetic data pipeline is constructed, where sentences are sampled from high-quality Wikipedia documents and rewritten as queries. This yields 5.8M triplets after deduplication and relevance filtering.

### Loss & Training

The total loss of GeAR is a weighted sum of two components:

$$\mathcal{L}_{\text{GeAR}} = \mathcal{L}_{\text{CL}} + \alpha \cdot \mathcal{L}_{\text{LM}}$$

- **Contrastive Learning Loss $\mathcal{L}_{\text{CL}}$**: Optimizes the semantic similarity between queries and documents. A momentum bi-encoder (similar to MoCo/BLIP) is introduced to provide richer soft-label supervision signals.
- **Language Modeling Loss $\mathcal{L}_{\text{LM}}$**: Optimizes the generation task via cross-entropy loss to maximize the probability of producing the ground-truth text. This is key to enhancing local information retrieval capabilities.
- A weight of $\alpha = 0.25$ yields the best performance, indicating that the generation task functions best as an auxiliary task with a lower weight.

The encoders are initialized using `bert-base-uncased`, while the 110M-parameter decoder is randomly initialized. Training is conducted on 16 AMD MI200 GPUs for 10 epochs.

## Key Experimental Results

### Main Results - Global Document Retrieval (Table)

| Method | SQuAD R@5 | NQ R@5 | TriviaQA R@5 | PAQ R@5 | RIR R@5 |
|------|-----------|--------|-------------|---------|---------|
| BGE (Pre-trained) | 0.829 | 0.674 | 0.690 | 0.752 | 0.451 |
| GTE (Pre-trained) | 0.866 | **0.767** | **0.726** | 0.836 | 0.528 |
| BGERT (Retrained) | 0.841 | 0.751 | 0.640 | 0.901 | 0.953 |
| **GeAR** | **0.887** | 0.762 | 0.664 | **0.952** | **0.964** |

GeAR achieves state-of-the-art (SOTA) performance on SQuAD, PAQ, and RIR, delivering an average relative improvement of 3.5% in Recall@5 compared to BGERT.

### Main Results - Local Information Retrieval (Table)

| Method | SQuAD R@1 | NQ R@1 | TriviaQA R@1 | PAQ R@1 | RIR R@3 |
|------|-----------|--------|-------------|---------|---------|
| BGE-Reranker-L | 0.751 | 0.670 | 0.464 | 0.704 | 0.891 |
| **GeAR** | **0.814** | **0.761** | **0.510** | **0.884** | **0.897** |

GeAR achieves an average relative improvement of 12.9% in local information retrieval compared to the runner-up, BGE-Reranker-L.

### Ablation Study (Table)

| LM Loss Weight α | Global Retrieval Avg Recall | Local Retrieval Avg Recall |
|-------------|-------------------|-------------------|
| 0           | 0.844             | 0.663             |
| 0.25        | **0.846**         | 0.781             |
| 0.5         | 0.844             | **0.785**         |
| 1.0         | 0.838             | 0.784             |

The impact of the generation task on retrieval performance exhibits an inverted U-shape: a moderate LM loss weight significantly enhances local retrieval (from 0.663 to 0.785), whereas excessively high weights degrade global retrieval.

### Key Findings

1. The generation task not only enhances local information retrieval but also has a minor yet consistent positive impact on global document retrieval.
2. Local retrieval in GeAR does not require re-encoding document chunks; instead, it directly ranks sentences using cross-attention weights.
3. The cross-attention weights in the middle layers of the fusion encoder (around layer 10) are more suitable for local retrieval than those in the top layer.
4. Despite having a decoder with only 110M parameters, the generation performance of GeAR on in-domain data is surprisingly comparable to that of LLaMA 3.2 3B.

## Highlights & Insights

- **Core Innovation**: Using the generation task as an auxiliary means to enhance the fine-grained comprehension of the retrieval model, rather than as a replacement for retrieval itself—this design philosophy is highly practical.
- **Zero Extra Inference Overhead**: The global retrieval phase is completely equivalent to a standard bi-encoder. The fusion encoder and decoder are only activated when local information is required.
- **Visualization Analysis**: This breaks the "black box" nature of traditional retrieval. By highlighting cross-attention weights and generating relevant text, it transforms retrieval results from sheer numerical scores into interpretable outputs.

## Limitations & Future Work

1. The context length is limited to 512 tokens, and long-text retrieval scenarios have not yet been explored.
2. The diversity and quality of the synthetic data may not match those of large-scale real retrieval data.
3. The decoder has only 110M parameters and is unable to handle complex generation tasks; scaling up the decoder can be explored in future work.
4. The model has not been evaluated on standard retrieval benchmarks like BEIR, so its generalizability requires further validation.

## Related Work & Insights

- Key inspiration is derived from multimodal representation learning (e.g., CLIP, ALBEF, BLIP), which treats documents and queries as two distinct "modalities" to be aligned.
- Unlike the late interaction in ColBERT, GeAR captures fine-grained semantics through generation tasks rather than multi-vector matching.
- This offers a new perspective for "post-retrieval localization" in RAG systems: the retriever itself can locate local information without requiring an additional reranker or reader.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Enhancing the dual capability of both global and local retrieval via generation tasks is a novel and self-consistent concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across two scenarios, five datasets, various baselines, and ablations, but lacks standard benchmarks like BEIR.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and intuitive visualization analysis, though some mathematical notation could be more concise.
- **Value**: ⭐⭐⭐⭐ — Practically advances the interpretability and fine-grained capabilities of retrieval systems, particularly suitable for RAG scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] VISA: Retrieval Augmented Generation with Visual Source Attribution](visa_retrieval_augmented_generation_with_visual_source_attribution.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)

</div>

<!-- RELATED:END -->
