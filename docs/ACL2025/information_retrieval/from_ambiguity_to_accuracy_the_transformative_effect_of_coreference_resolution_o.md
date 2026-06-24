---
title: >-
  [Paper Note] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems
description: >-
  [ACL 2025][Information Retrieval & RAG][Coreference Resolution] This paper systematically investigates the impact of coreference resolution on the two stages of RAG systems: document retrieval and QA generation. It finds that coreference resolution consistently improves retrieval performance (with mean pooling models benefiting the most). In QA tasks, the performance improvement for small models is significantly greater than that for large models…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Coreference Resolution"
  - "RAG"
  - "Retrieval-Augmented Generation"
  - "Embedding Models"
  - "Question Answering Systems"
date: 2026-05-08
content_hash: aa235c535f68935a
---

# From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems

**Conference**: ACL 2025  
**arXiv**: [2507.07847](https://arxiv.org/abs/2507.07847)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Coreference Resolution, RAG, Retrieval-Augmented Generation, Embedding Models, Question Answering Systems

## TL;DR

This paper systematically investigates the impact of coreference resolution on the two stages of RAG systems: document retrieval and QA generation. It finds that coreference resolution consistently improves retrieval performance (with mean pooling models benefiting the most). In QA tasks, the performance improvement for small models is significantly greater than that for large models, even allowing small models to reach the baseline performance of large models.

## Background & Motivation

**Background**: RAG has become a core framework in NLP for improving factual consistency and reducing hallucinations by combining external document retrieval with LLM generation. However, the effectiveness of RAG is often hindered by the complexity of coreference in retrieved documents—ambiguities introduced by numerous pronouns and abbreviations in the documents interfere with in-context learning.

**Limitations of Prior Work**: Coreference complexity affects RAG systems on two levels: (1) **Retrieval Stage**, where pronouns and abbreviations make it difficult for embedding models to accurately capture document semantics, leading to query-document matching failures; (2) **Generation Stage**, where ambiguous reference relationships disrupt the reasoning chain and reduce the factual accuracy of answers. These retrieval errors are amplified during generation, ultimately undermining user trust in the system's output.

**Design Motivation**: Existing research lacks a systematic empirical analysis of how coreference resolution separately affects each core component of RAG (retrieval and in-context learning). The authors aim to answer: (1) How do embedding models with different pooling strategies differ in performance after coreference resolution? (2) Do LLMs of different scales benefit to varying degrees from coreference resolution?

## Method

### Overall Architecture

This paper does not propose a new model architecture but rather represents a **systematic empirical study**. The core pipeline consists of:

1. **Coreference Resolution Preprocessing**: For each document $d_i$, an LLM-driven coreference resolution function $f_{\text{coref}}$ is used to generate the resolved document $d_i' = f_{\text{coref}}(d_i)$, replacing pronouns/abbreviations with explicit antecedents (e.g., "GR" $\rightarrow$ "general relativity", "it" $\rightarrow$ "the basketball").
2. **Retrieval Experiments**: Evaluate the retrieval performance of multiple embedding models on original vs. resolved documents.
3. **QA Experiments**: Evaluate the QA performance of multiple LLMs on original vs. resolved documents.

### Key Designs

**1. LLM-driven Coreference Resolution**

- A GPT-4o-mini is utilized to implement the coreference resolution function. It takes text containing unresolved coreferences as input and outputs text where multiple expressions referring to the same entity are explicitly linked.
- Example: In the document, "GR", "it", and "Its" are replaced with "general relativity", "the basketball", and "The basketball's" respectively.
- This approach improves the embedding models' capacity to capture document semantics by increasing explicit semantic information in the text.

**2. Model Architectures and Pooling Strategy Comparison in Retrieval Experiments**

The experiments cover 8 embedding models across two main architectural categories:

- **Encoder-based (4 models)**: e5-large-v2, stella_en_400M_v5 (Mean pooling); gte-modernbert-base ([CLS] pooling); bge-large-en-v1.5 ([CLS] pooling)
- **Decoder-based (4 models)**: NV-Embed-v2, LLM2Vec (Mean pooling); gte-Qwen2-1.5B, Linq-Embed-Mistral (Last token pooling)

**3. Model Scale Comparison in QA Experiments**

Seven instruction-tuned models are selected, covering three families with two scales each:
- Llama 3.2-3B / 3.1-8B
- Qwen2.5-3B / 7B
- gemma-2-2b / 9b
- Mistral-7B

### Loss & Training

This paper is a purely empirical study and does not involve model training. The evaluation metrics include:
- **Retrieval Task**: nDCG@k (k=1,3,5), evaluating retrieval ranking quality.
- **QA Task**: Accuracy calculated via log likelihood for BELEBELE and BoolQ; F1 score for SQuAD2.0.

## Key Experimental Results

### Main Results 1: Retrieval Performance Comparison

| Model | Pooling | Original AVG@1 | +CR AVG@1 | Original AVG@5 | +CR AVG@5 |
|------|---------|-----------|-----------|-----------|-----------|
| e5-large-v2 | Mean | 0.809 | **0.814** | 0.810 | 0.809 |
| stella_en_400M_v5 | Mean | 0.785 | **0.790** | 0.803 | **0.804** |
| NV-Embed-v2 | Mean | 0.836 | **0.843** | 0.836 | 0.836 |
| LLM2Vec | Mean | 0.814 | **0.826** | 0.824 | **0.827** |
| gte-modernbert-base | [CLS] | 0.793 | **0.794** | 0.811 | 0.807 |
| bge-large-en-v1.5 | [CLS] | 0.776 | **0.777** | 0.799 | **0.800** |
| gte-Qwen2-1.5B | Last | 0.816 | 0.816 | 0.812 | **0.814** |
| Linq-Embed-Mistral | Last | 0.810 | **0.815** | 0.830 | **0.832** |

**Key Findings**: Mean pooling models benefit the most (LLM2Vec +0.012 @1), while [CLS]/Last pooling gains are smaller.

### Main Results 2: QA Performance Comparison

| Model | Scale | BoolQ Original→+CR | BELEBELE Original→+CR | SQuAD Original→+CR |
|------|------|---------------|-------------------|---------------|
| Qwen2.5-3B | 3B | 0.780→0.780 | 0.780→**0.858** (+7.8%) | 0.297→**0.550** (+25.3%) |
| Qwen2.5-7B | 7B | 0.860→0.860 | 0.862→**0.902** (+4.0%) | 0.398→**0.798** (+40.0%) |
| gemma-2-2b | 2B | 0.801→0.802 | 0.263→**0.307** (+4.3%) | 0.519→**0.621** (+10.2%) |
| gemma-2-9b | 9B | 0.865→0.865 | 0.541→**0.547** (+0.6%) | 0.765→**0.842** (+7.8%) |
| Llama3.2-3B | 3B | 0.764→0.764 | 0.812→**0.839** (+2.7%) | 0.644→**0.689** (+4.5%) |
| Llama3.1-8B | 8B | 0.820→0.821 | 0.883→**0.913** (+3.0%) | 0.558→**0.783** (+22.4%) |

### Ablation Study

The experimental design of this paper itself serves as a systematic ablation:
- **Ablation of Pooling Strategies**: Mean pooling achieves the most significant improvement after coreference resolution. Since it treats all tokens equally, replacing pronouns with explicit antecedents allows each token to carry more semantic information.
- **Ablation of Model Scale**: On BELEBELE, the improvement of Qwen2.5-3B (+7.8%) is much larger than Qwen2.5-7B (+4.0%); the improvement of gemma-2-2b (+4.3%) is much larger than gemma-2-9b (+0.6%).
- **Cross-Scale Comparison**: After resolution, gemma-2-2b (F1=0.621) and Qwen2.5-3B (F1=0.550) on SQuAD can match or even surpass unresolved larger models (Llama3.1-8B=0.558, Qwen2.5-7B=0.398).

### Key Findings

- **Synergistic effect between Mean pooling and coreference resolution**: Mean pooling treats each token equally. Replacing pronouns with explicit entities after resolution allows each token to carry richer semantics.
- **Small models benefit disproportionately**: Small models have limited inherent ability to handle referential ambiguity. Coreference resolution acts as "external assistance," significantly lowering the barrier to understanding.
- **Coreference resolution increases document length**: Replacing pronouns with antecedents lengthens the text, which further amplifies the advantage of mean pooling (enabling more effective integration of information across different text lengths).

## Highlights & Insights

1. **Novel Problem Perspective**: Reconceptualizes coreference resolution as a general preprocessing step for RAG systems rather than a traditional NLP subtask, providing a unified cross-task perspective.
2. **Deep Mechanical Explanation of Mean Pooling**: It not only reports the phenomenon but also provides a rational explanation from the perspective of token-level semantic capacity.
3. **High Practical Value**: Discovers a low-cost performance improvement scheme for resource-constrained scenarios (where only small models can be used) — applying coreference resolution prior to retrieval/QA.
4. The finding that **small model + coreference resolution $\approx$ large model** offers strong engineering implications.

## Limitations & Future Work

1. **Bias of Coreference Resolution Itself**: Utilizing GPT-4o-mini for resolution may introduce model-specific biases, which do not necessarily align perfectly with human understanding.
2. **Insufficient Domain Generalization**: Although 4 datasets were used, specialized technical texts (e.g., medical, legal) were not covered, leaving the effectiveness of resolution in these fields unknown.
3. **Trade-off in Generation Flexibility**: While explicit references enhance clarity, they may constrain the language model's ability to generate natural and diverse responses.
4. **Computational Cost**: Invoking LLMs for coreference resolution on every document introduces additional overhead, which may be impractical for large-scale corpus scenarios.
5. **Lack of End-to-End Evaluation**: Retrieval and QA are evaluated separately; the full cascaded effect of "coreference resolution $\rightarrow$ better retrieval $\rightarrow$ better QA" is not demonstrated.

## Related Work & Insights

- **Coreference Resolution Methods**: From rule-based approaches to end-to-end neural methods (Lee et al., 2017; Kantor & Globerson, 2019) and LLM-driven resolution.
- **RAG System Improvements**: Complements Dense X Retrieval (Chen et al., 2024) in enhancing retrieval granularity and is orthogonal to retrieval re-ranking studies.
- **Insights**: Coreference resolution can serve as a standard preprocessing pipeline component for RAG, particularly when deploying small models. Performing resolution prior to chunking should also be considered to enhance the self-containment of chunks.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 6 | Limited methodological novelty (purely empirical), but the perspective and findings are valuable |
| Technical Depth | 5 | No new models/algorithms; the core contribution lies in the comprehensive empirical analysis |
| Experimental Thoroughness | 8 | Covers 8 embedding models + 7 LLMs + 4 datasets, offering broad coverage |
| Writing Quality | 7 | Logical and clear, with deep analysis and an intuitive Figure 1 example |
| Value | 7 | Findings can be directly deployed in RAG preprocessing pipelines |
| Overall Score | 6.5 | A solid empirical research with inspiring findings, although lacking in methodological innovation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Distracting Effect: Understanding Irrelevant Passages in RAG](the_distracting_effect_understanding_irrelevant_passages_in_rag.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](knowshiftqa_rag_knowledge_shifts.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)

</div>

<!-- RELATED:END -->
