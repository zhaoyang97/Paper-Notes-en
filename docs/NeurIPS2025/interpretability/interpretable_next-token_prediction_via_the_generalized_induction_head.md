---
title: >-
  [Paper Note] Interpretable Next-token Prediction via the Generalized Induction Head
description: >-
  [NeurIPS 2025][Interpretability] This paper proposes Induction-Gram (GIM), an interpretable language model that combines exact n-gram matching with fuzzy matching. By constructing a "generalized induction head" to retrieve similar sequences from the input context for next-token prediction, it achieves up to 25 percentage points improvement over interpretable baselines and a 20% improvement in fMRI brain response prediction.
tags:
  - NeurIPS 2025
  - Interpretability
  - Next-token Prediction
  - Induction Head
  - n-gram Models
  - fMRI
date: 2026-05-08
content_hash: 05f6112024e96bb8
---

# Interpretable Next-token Prediction via the Generalized Induction Head

**Conference**: NeurIPS 2025

**arXiv**: [2411.00066](https://arxiv.org/abs/2411.00066)

**Code**: [Available](https://github.com/ejkim47/generalized-induction-head)

**Area**: Interpretable AI / Language Modeling

**Keywords**: Interpretability, Next-token Prediction, Induction Head, n-gram Models, fMRI

## TL;DR

This paper proposes Induction-Gram (GIM), an interpretable language model that combines exact n-gram matching with fuzzy matching. By constructing a "generalized induction head" to retrieve similar sequences from the input context for next-token prediction, it achieves up to 25 percentage points improvement over interpretable baselines and a 20% improvement in fMRI brain response prediction.

## Background & Motivation

Large Transformer models achieve strong predictive performance but their black-box nature limits their deployment in high-stakes domains such as science, medicine, and policy-making. Furthermore, the enormous scale of LLMs entails prohibitive energy costs and deployment challenges.

Traditional n-gram models maintain full interpretability and computational efficiency, yet exhibit a substantial performance gap relative to black-box LLMs. The existing Infini-Gram model, while scalable to trillions of tokens in a reference corpus, struggles to find long exact matches under distribution shift (i.e., when the reference corpus differs from the input context), leading to suboptimal performance.

This work is inspired by the *induction head* mechanism identified in LLMs — induction heads perform in-context learning by identifying similar patterns within the context and copying subsequent tokens. The authors seek to reconstruct this mechanism in an interpretable manner to bridge the gap between n-gram models and LLMs.

## Method

### Overall Architecture

Induction-Gram consists of three components:

1. **Infini-Gram**: Exact n-gram matching against a reference corpus.
2. **Induction-only (exact)**: Exact n-gram matching within the input context.
3. **Induction-only (fuzzy)**: Fuzzy matching within the input context.

The final prediction is composed according to the following priority (Eq. 5):
- If the effective $n$ from the reference corpus exceeds that from the input context and surpasses threshold $\tau$, use Infini-Gram.
- If the effective $n$ from the input context is larger and exceeds $\tau$, use exact induction matching.
- Otherwise, use fuzzy induction matching.

### Key Designs

**1. Similarity Definition for Fuzzy Matching**

The similarity between two sequences $x_1$ and $x_2$ is defined based on whether they induce similar next-token distributions, measured via Jensen-Shannon divergence:

$$s(x_1, x_2) = \exp(-\text{JSD}(P_{\text{next}}(x_1), P_{\text{next}}(x_2)))$$

**2. Fuzzy Matching Model**

To compute similarity efficiently, a lightweight Transformer (3–4 layers) is trained via knowledge distillation from an LLM. The model produces feature embeddings, and cosine similarity is used to approximate the original similarity:

$$s_{\text{FM}}(x_1, x_2) = \exp(-(1 - \text{CosineSim}(e_1, e_2))/T)$$

where $T=0.1$ is a temperature parameter. The model is jointly trained with cross-entropy loss and reverse KL divergence loss.

**3. Next-token Prediction via Fuzzy Matching**

A sliding window is applied over the input context to search for sequences similar to the end of the query. Similarity scores are used as soft counts to estimate the next-token probability distribution:

$$P_{\text{induction(fuzzy)}}(w_i | x) = \frac{c_{\text{fuzzy}}(w_{i-k-1:i-1} w_i | x)}{\sum_{w_j \in \mathcal{V}} c_{\text{fuzzy}}(w_{i-k-1:i-1} w_j | x)}$$

### Loss & Training

- The Fuzzy Matching Model is trained with **cross-entropy loss + reverse KL divergence loss** (weight 1.0 each).
- LLaMA2-7B is used as the teacher model.
- AdamW optimizer with learning rate 0.0001 and weight decay 0.1.
- Cosine learning rate schedule with 1,000 warm-up steps; trained for 15,000–20,000 steps.
- Threshold $\tau$ is determined via cross-validation on BabyLM: 8 for the GPT-2 tokenizer and 9 for LLaMA-2.

## Key Experimental Results

### Main Results

**Table 1: Next-token Prediction Accuracy (%) — GPT-2 Tokenizer**

| Reference Corpus | Model | BabyLM-test | FineWeb | Pile-val |
|:---|:---|:---:|:---:|:---:|
| — | Induction-only (exact) | 36.7 | 17.2 | 37.0 |
| — | Induction-only (fuzzy) | 41.1 | 25.2 | 38.7 |
| BabyLM-dev (17.4M) | Infini-Gram | 37.6 | 14.7 | 16.0 |
| BabyLM-dev (17.4M) | **Induction-Gram** | **42.2 (+4.6)** | **25.3 (+10.6)** | **40.0 (+24.0)** |
| OpenWebText (9.04B) | Infini-Gram | 16.7 | 25.5 | 22.7 |
| OpenWebText (9.04B) | **Induction-Gram** | **41.8 (+25.1)** | **27.2 (+1.7)** | **42.7 (+20.0)** |
| Pile-train (383B) | Infini-Gram | 33.5 | 39.3 | 49.2 |
| Pile-train (383B) | **Induction-Gram** | **49.4 (+15.9)** | **38.0 (−1.3)** | **50.3 (+1.1)** |
| Unknown | LLM (GPT-2) | 46.9 | 39.0 | 52.3 |

**Table 2: Speculative Decoding Speed (A40 × 1 GPU)**

| Draft Model | Large Model | BabyLM ms/token (↓) | Speedup (↑) | Pile ms/token (↓) | Speedup (↑) |
|:---|:---|:---:|:---:|:---:|:---:|
| — | LLaMA2-7B | 30.2 | 1.00× | 30.2 | 1.00× |
| TinyLLaMA-1.1B | LLaMA2-7B | 21.3 | 1.42× | 21.3 | 1.42× |
| **Induction-only (fuzzy)** | LLaMA2-7B | **17.7** | **1.71×** | **20.1** | **1.50×** |
| — | LLaMA2-13B | 52.4 | 1.00× | 52.0 | 1.00× |
| TinyLLaMA-1.1B | LLaMA2-13B | 26.7 | 1.96× | 26.3 | 1.98× |
| **Induction-only (fuzzy)** | LLaMA2-13B | **24.x** | **~2.1×** | **~25** | **~2.0×** |

### Ablation Study

**fMRI Brain Response Prediction Results (Table 3)**

| Feature Model | Mean Correlation (All Voxels) | Mean Correlation (Top-10% Voxels) |
|:---|:---:|:---:|
| Eng1000 (baseline) | 0.072 | 0.220 |
| Random matching + Eng1000 | ~0.069 | — |
| Naive n-gram matching + Eng1000 | ~0.068 | — |
| Infini-Gram matching + Eng1000 | — | 0.200 |
| **Induction matching + Eng1000** | **0.087 (+20%)** | **0.265 (+20%)** |
| Black-box LLaMA-2 | — | 0.268 |

### Key Findings

1. **In-context matching vs. corpus matching**: Induction-only (exact), using only a 1,024-token input context, outperforms Infini-Gram using a 10B-token reference corpus by 5.5–20 percentage points on BabyLM and Pile.
2. **Value of fuzzy matching**: Induction-only (fuzzy) improves over the exact variant by 1.5–8.7 percentage points, with more pronounced gains when the effective $n$ is low.
3. **Complementarity**: Induction-Gram still yields a 15.9 percentage point gain over Infini-Gram even when using the Pile-train (383B) reference corpus.
4. **Speculative decoding speedup**: As a draft model, it achieves up to 2.1× speedup while preserving interpretability.
5. **fMRI prediction**: Induction matching trails black-box LLaMA-2 by only 1% (0.265 vs. 0.268) while being fully interpretable.

## Highlights & Insights

1. **Bridging mechanistic and engineering interpretability**: Starting from the induction head mechanism discovered in LLMs and reconstructing it via hand-crafted engineering, this work represents a new paradigm for reverse-engineering interpretable components from LLMs.
2. **Distributional advantage of the input context**: Experiments show that in-context matching naturally reflects the distribution of the input query, yielding higher accuracy than matching against a fixed reference corpus.
3. **Cross-domain generalization**: The same framework transfers seamlessly from language modeling to fMRI brain response prediction, demonstrating the method's generality.
4. **Interpretable and efficient**: The fuzzy matching model requires only 3–4 Transformer layers, and as a speculative decoding draft model it is faster even than LLM-based draft models.

## Limitations & Future Work

1. **Limited by short contexts**: When the input context is short or informationally sparse, the gains from the induction head are limited.
2. **Insufficient reasoning capability**: Reliance on n-gram-level matching makes the model ill-suited for tasks requiring deep reasoning, analogous to the limitations of kNN-LM.
3. **Potential integration with RAG**: Retrieval-augmented generation could supplement the context with relevant documents, potentially alleviating the short-context limitation.
4. **Further LLM components**: Additional mechanisms such as the indirect object identifier and retrieval head could be incorporated into the framework.

## Related Work & Insights

- **Infini-Gram** (Liu et al., 2024): Extends n-gram models to a reference corpus of trillions of tokens.
- **Induction Head** (Olsson et al., 2022): A key mechanism for in-context learning identified in Transformers.
- **kNN-LM** (Khandelwal et al., 2020): A nearest-neighbor-based language model.
- **Speculative Decoding** (Leviathan et al., 2023): Uses a small model to accelerate inference with a large model.
- Insight: Interpretable models need not sacrifice all performance; engineerable interpretable components can be extracted from LLM internal mechanisms.

## Rating

| Dimension | Score (1–5) | Notes |
|:---|:---:|:---|
| Novelty | 4 | A new paradigm for extracting interpretable components from LLM mechanisms |
| Technical Quality | 4 | Elegant and concise design with thorough experiments |
| Experimental Thoroughness | 5 | Multiple datasets, tokenizers, and two application scenarios |
| Practicality | 4 | Directly applicable to speculative decoding and fMRI analysis |
| Writing Quality | 4 | Clear structure with well-motivated problem formulation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[AAAI 2026\] GenePheno: Interpretable Gene Knockout-Induced Phenotype Abnormality Prediction Framework](../../AAAI2026/interpretability/genepheno_interpretable_gene_knockout-induced_phenotype_abnormality_prediction_f.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)

</div>

<!-- RELATED:END -->
