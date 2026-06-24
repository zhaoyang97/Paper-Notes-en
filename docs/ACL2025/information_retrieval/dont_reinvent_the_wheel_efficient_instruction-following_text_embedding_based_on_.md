---
title: >-
  [Paper Note] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation
description: >-
  [ACL 2025][Information Retrieval & RAG][Instruction-following embeddings] The GSTransform framework is proposed, which adapts pre-computed generic embeddings in real-time to the semantic space specified by user instructions via a lightweight space transformation. This avoids re-encoding the entire corpus for each new instruction, achieving an average score of 66.01 across 9 datasets (compared to the SOTA baseline of 55.31) while delivering a 6x to 300x speedup in real-time la…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Instruction-following embeddings"
  - "space transformation"
  - "text embeddings"
  - "clustering"
  - "efficiency optimization"
date: 2026-05-08
content_hash: ef43bf536fd170d3
---

# Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation

**Conference**: ACL 2025  
**arXiv**: [2505.24754](https://arxiv.org/abs/2505.24754)  
**Code**: [Yes](https://github.com/YingchaojieFeng/GSTransform)  
**Area**: Information Retrieval  
**Keywords**: Instruction-following embeddings, space transformation, text embeddings, clustering, efficiency optimization

## TL;DR

The GSTransform framework is proposed, which adapts pre-computed generic embeddings in real-time to the semantic space specified by user instructions via a lightweight space transformation. This avoids re-encoding the entire corpus for each new instruction, achieving an average score of 66.01 across 9 datasets (compared to the SOTA baseline of 55.31) while delivering a 6x to 300x speedup in real-time latency.

## Background & Motivation

Text embeddings are fundamental infrastructure in NLP, but generic embedding models are static and cannot adapt to specific task requirements. For instance, given the same set of news articles, users might focus on "topics" or "countries", requiring different structures for the embedding space.

**Instruction-following text embedding** allows users to guide embedding generation via instructions, enabling embeddings to dynamically adjust their semantic focus. The core limitation of existing methods is:

- **InstructOR**: Concatenating instructions with text to fine-tune Transformers $\rightarrow$ Requires re-encoding the entire corpus for every new instruction.
- **InBedder**: Treating instructions as questions to encode corresponding answers $\rightarrow$ Similarly requires $O(m \times n)$ forward passes.

With $m$ instructions and $n$ texts, the computational overhead grows linearly with the size of the data, making it impractical for real-world scenarios where vector databases have already pre-stored embeddings.

**Key Observation**: Generic embeddings already contain instruction-related information, which is just not explicitly utilized. Instead of re-encoding, it is more efficient to directly transform existing embeddings.

## Method

### Overall Architecture

GSTransform consists of two phases:
1. **Instruction-based Label Construction**: Constructing a label classification hierarchy starting from user instructions.
2. **Label-guided Embedding Transformation**: Training a lightweight transformation model using the constructed labels.

### Key Designs

1. **Instruction-following Clustering**:

    - Randomly sample 3,000 texts from the corpus.
    - Use an LLM (GPT-4o-mini) to generate a summary for each text based on the user instruction (extracting instruction-relevant semantics).
    - Encode the summaries using a generic embedding model (e.g., UAE).
    - Run k-means++ clustering (k=50) to obtain instruction-related clusters.
    - Design Motivation: Directly clustering original embeddings fails to capture instruction-specific distinctions.

2. **Contrastive Label Generation**:

    - For each cluster, construct an LLM prompt using positive samples (texts within the cluster) and negative samples (texts from other clusters).
    - Guide the LLM to focus on the discriminative features of the cluster to generate semantically aligned, mutually exclusive, and generalizable labels.
    - Avoid labels that are overly broad or excessively specific.

3. **Embedding Transformation**:

    - Use the LLM to classify the sampled texts based on the label classification hierarchy (classifying original texts rather than summaries to preserve full context).
    - Train a lightweight encoder-decoder (both are single-layer linear layers) with dual-objective optimization:
    - **Contrastive Loss**: Pull embeddings with the same label closer and push different label embeddings further apart (with a margin).
    - **Reconstruction Loss**: Reconstruct the original embeddings from the transformed embeddings to preserve generic semantic information.
    - Once trained, real-time transformation can be applied to any pre-computed embedding without accessing the original text.

### Loss & Training

Total loss: $\mathcal{L} = \beta_1 \cdot \mathcal{L}_{contr} + \beta_2 \cdot \mathcal{L}_{recon}$ ($\beta_1 = \beta_2 = 1.0$)

- Contrastive loss uses Euclidean distance + margin penalty.
- Reconstruction loss is MSE.
- 80/20 train/validation split with early stopping to prevent overfitting.
- Requires only 3,000 labeled samples for training.

## Key Experimental Results

### Main Results — Average scores across 9 datasets (Table)

| Model | Clustering Mean | STS Mean | Triplet Mean | Overall Mean |
|------|----------------|----------|-------------|--------|
| InstructOR | 38.57 | 37.99 | 59.71 | 45.42 |
| InBedder-Llama2 | 45.06 | 43.38 | 77.51 | 55.31 |
| UAE (generic) | 38.86 | 41.30 | 54.68 | 44.95 |
| **GSTransform (UAE)** | **57.27** | **58.12** | **82.29** | **65.89** |
| **GSTransform (Mxbai)** | **57.59** | **58.07** | **82.37** | **66.01** |
| **GSTransform (BGE)** | **57.33** | **56.98** | **81.65** | **65.32** |

GSTransform improves the generic model UAE from 44.95 to 65.89 (+20.94), surpassing the SOTA baseline InBedder-Llama2 (55.31).

### Efficiency Comparison

| Model | AG-News RT Latency (s) | Big Patent RT Latency (s) |
|------|-------------------|---------------------|
| InstructOR | 511 | 1,679 |
| InBedder-Roberta | 749 | 1,781 |
| InBedder-Llama2 | 16,206 | 29,756 |
| **GSTransform** | **77** | **87** |

The real-time latency of GSTransform is only 77-87 seconds, which is **6x to 300x faster** than InBedder-Llama2.

### Ablation Study

| Variant | NYTClust | MultiHate | IntEmo | Mean |
|------|----------|-----------|--------|------|
| GSTransform (Mxbai) | 73.92 | 52.45 | 96.30 | 74.22 |
| W/o summarization step | 68.32 | 16.21 | 85.14 | 56.56 |
| Direct LLM label generation | 42.50 | 20.55 | 86.76 | 49.94 |
| FDA replacing transformation model | 67.53 | 48.49 | 89.19 | 68.40 |

Performance drops sharply without the instruction-guided summarization step (74.22 $\rightarrow$ 56.56), and direct label generation yields even worse results ($\rightarrow$ 49.94).

### Key Findings

- Instruction-guided summarization is the most critical component: without it, the score on MultiHate plummets from 52.45 to 16.21.
- Near-optimal performance is achieved with just 3,000 samples, with diminishing returns when increased to 5,000.
- Performance remains stable with k values in the range of 10-90, demonstrating the robustness of the method to hyperparameters.
- Consistent improvements of over 20 points across different backbones (UAE, Mxbai, BGE) validate its model-agnostic nature.
- The largest gain is achieved on the AmzCF dataset (1.49 $\rightarrow$ 34.68), which confirms that generic embeddings indeed contain instruction-relevant information.

## Highlights & Insights

- **Valuable shift in perspective**: Shifting from "re-encoding" to "transforming existing embeddings" aligns with actual industrial deployment scenarios (where embeddings are pre-stored in vector databases).
- Requires only a small number of LLM calls (~3,000 times for summarization + classification), followed by pure local linear transformations, keeping the cost extremely low.
- The contrastive label generation strategy ensures high label quality, outperforming direct LLM label generation by 24 points.
- A single linear layer is sufficient for the encoder-decoder, outperforming traditional dimensionality reduction methods like FDA.

## Limitations & Future Work

- Random sampling is sensitive to data imbalance; coreset selection could be a better alternative.
- A single-layer linear transformation may be insufficient to capture complex non-linear relationships.
- The transformation quality depends heavily on the base quality of the generic embeddings.
- Improvements are relatively limited on some domain-specific datasets (e.g., Big Patent).

## Related Work & Insights

- InstructOR and InBedder represent the concatenation-based and QA-based approaches to instruction-following embeddings, respectively.
- Contrastive learning frameworks (like SimCSE, AnglE) provide the foundation for GSTransform's contrastive loss.
- It complements prompt-based retrieval methods (e.g., PromptRiever, FollowIR); while the former focuses on retrieval, this work focuses on generic embedding adaptation.
- The framework is highly generalizable and can be seamlessly integrated with more efficient LLMs and embedding models in the future.

## Rating

- **Novelty**: 8/10 — The idea of transforming existing embeddings is simple yet effective, with a clearly defined problem.
- **Experimental Thoroughness**: 9/10 — Evaluation across 3 tasks and 9 datasets + ablation studies + parameter sensitivity analysis + efficiency comparisons + case studies.
- **Writing Quality**: 8/10 — Clear logic and well-designed figures/tables.
- **Value**: 8/10 — High practical deployment value; the 6x to 300x speedup has direct significance for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)
- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)

</div>

<!-- RELATED:END -->
