---
title: >-
  [Paper Note] Efficient Long Context Language Model Retrieval with Compression
description: >-
  [ACL 2025][Model Compression][Long-context Language Models] CoLoR (Compression for Long context Retrieval) is proposed to jointly train a passage compression model using preference optimization and length regularization, compressing the context length by 1.91x while improving the retrieval performance of long-context language models by 6%.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Long-context Language Models"
  - "Information Retrieval"
  - "Passage Compression"
  - "Preference Optimization"
  - "Length Regularization"
date: 2026-05-08
content_hash: 34fe662842aff139
---

# Efficient Long Context Language Model Retrieval with Compression

**Conference**: ACL 2025  
**arXiv**: [2412.18232](https://arxiv.org/abs/2412.18232)  
**Code**: [GitHub](https://github.com/going-doer/CoLoR)  
**Area**: Model Compression  
**Keywords**: Long-context Language Models, Information Retrieval, Passage Compression, Preference Optimization, Length Regularization

## TL;DR

CoLoR (Compression for Long context Retrieval) is proposed to jointly train a passage compression model using preference optimization and length regularization, compressing the context length by 1.91x while improving the retrieval performance of long-context language models by 6%.

## Background & Motivation

**Background**: Long-context language models (LCLMs) have emerged as a new paradigm for information retrieval. They can process entire document corpora directly within an single context window, bypassing traditional sparse or dense retrieval indexes. This "all-in-context retrieval" approach has demonstrated potential to outperform traditional BM25 and DPR methods across multiple tasks.

**Limitations of Prior Work**: LCLM retrieval faces severe efficiency bottlenecks. When massive passages are loaded into the context, the computational overhead grows superlinearly with context length (due to the $O(n^2)$ complexity of self-attention in Transformers). Processing these intermediate representations during inference is also highly time-consuming. Existing general-purpose text compression methods are primarily designed for summarization tasks and lack optimization for retrieval scenarios, often discarding key information required for effective retrieval.

**Key Challenge**: Retrieval accuracy requires preserving fine-grained information in passages relevant to potential queries, whereas efficiency demands minimizing passage length. A natural trade-off exists between these two objectives: over-compression discards critical discriminative details, causing retrieval to fail, while under-compression fails to reduce computational costs effectively.

**Goal**: To design a passage compression method tailored specifically for LCLM retrieval, maximizing retrieval performance while minimizing compressed passage length.

**Key Insight**: The authors propose formulating passage compression as a preference learning problem. For a given query, a compressed passage version that leads to successful retrieval is designated as "chosen", while one that causes failure is designated as "rejected". This allows preference data to be automatically generated without human annotation. Length regularization is also integrated to reinforce conciseness.

**Core Idea**: A compression model is trained using preference optimization (e.g., DPO), leveraging "retrieval success" as an automatic reward signal to generate chosen/rejected pairs, while incorporating a length regularization loss to simultaneously optimize retrieval quality and compression ratio.

## Method

### Overall Architecture

The pipeline of CoLoR is as follows: (1) First, a pre-trained language model is used to perform multi-sample compression on passages, generating compressed candidates of varying lengths. (2) These compressed candidates are fed into an LCLM for retrieval evaluation, where they are automatically labeled as chosen/rejected pairs based on whether retrieval succeeds. (3) The compression model is trained via DPO on these preference data, integrated with a length regularization loss to constrain output length. (4) During inference, the compression model first compresses all passages before they are fed into the LCLM for retrieval.

### Key Designs

1. **Automated Preference Data Generation**:

    - **Function**: Constructs chosen/rejected training pairs automatically for the compression model without human annotation.
    - **Mechanism**: For each query-passage pair, multiple compression candidates are generated using an existing compression model. Each candidate replaces the original passage in the LCLM for retrieval evaluation. Successful candidates are labeled as chosen, while failed candidates are labeled as rejected. This directly employs end-to-end retrieval performance as the metric for compression quality.
    - **Design Motivation**: Traditional compression methods usually evaluate summarization quality (e.g., ROUGE), which does not necessarily correlate with retrieval success. Utilizing retrieval outcomes as reward signals ensures that the compression model learns to prioritize information critical strictly to retrieval.

2. **Preference Optimization Training (DPO)**:

    - **Function**: Steers the compression model toward generating compressed texts that facilitate successful retrieval.
    - **Mechanism**: Adopts the Direct Preference Optimization (DPO) framework. Given the original passage as input and the chosen/rejected compressed versions as preference pairs, the compression model is trained to maximize the generation likelihood of chosen versions and minimize that of rejected versions. This bypasses the need for an additional reward model required by RLHF.
    - **Design Motivation**: Retrieval success/failure is a binary signal that naturally fits the preference learning framework. DPO consolidates reward modeling and policy optimization into a single step, enabling more stable and efficient training.

3. **Length Regularization Loss**:

    - **Function**: Enforces conciseness constraints on top of preference optimization to prevent the compression model from generating excessively long outputs.
    - **Mechanism**: A length regularization term is appended to the DPO loss to penalize the length of the compressed output. The two losses are combined via weighted summation to form the final training objective: $L = L_{\text{DPO}} + \lambda \cdot L_{\text{len}}$, where $\lambda$ controls the balance between compression ratio and retrieval quality.
    - **Design Motivation**: Relying solely on preference optimization might lead the model to adopt a conservative deletion strategy (i.e., deleting as little as possible) to ensure retrieval success, which contradicts the goal of compression. Length regularization explicitly introduces a penalty to drive conciseness.

### Loss & Training

The final objective is a weighted combination of the DPO loss and length regularization. Training is based on Phi-3-mini-4k-instruct (3.8B) as the backbone of the compression model. During inference, Gemini-1.5-Flash or other LCLMs are utilized as retrieval engines. Evaluation is conducted across 9 diverse datasets.

## Key Experimental Results

### Main Results

Retrieval performance comparison across 9 datasets (average results):

| Method | Context Compression Ratio | Average Retrieval Accuracy | Relative Gain |
|------|------------|-------------|---------|
| Original Passages (No Compression) | 1.0x | Baseline | - |
| General Summarization Compression | ~2.0x | Baseline - 3% | Performance Drops Post-Compression |
| CoLoR | 1.91x | Baseline + 6% | Win-Win (Compression + Gain) |

Detailed cross-dataset results:

| Dataset Type | Original Passages | CoLoR | Description |
|-----------|---------|-------|------|
| Natural QA (NQ, etc.) | Baseline | +5~8% | Substantial improvement on factual questions |
| Multi-hop Reasoning | Baseline | +3~5% | Effectively handles cross-passage evidence |
| Domain-Specific Retrieval | Baseline | +6~9% | Strong domain adaptability |

### Ablation Study

| Configuration | Retrieval Accuracy | Compression Ratio | Description |
|------|----------|--------|------|
| Full CoLoR | Baseline + 6% | 1.91x | Complete model |
| W/o Length Regularization | Baseline + 5% | 1.3x | Insufficient compression, slight performance drop |
| W/o Preference Optimization (SFT Only) | Baseline + 1% | 1.8x | Lack of retrieval awareness |
| Random Chosen/Rejected | Baseline - 2% | 1.7x | Validates the importance of label quality |

### Key Findings

- **Compression paradoxically boosts retrieval**: CoLoR's compressed passages are not only shorter but also net a 6% average improvement over uncompressed passages. This suggests that the compression process removes noise while preserving core content relevant to the retrieval, acting as an implicit denoising step.
- **Length regularization is critical**: Disabling length regularization drops the compression ratio from 1.91x to 1.3x. This indicates that preference optimization alone tends to be highly conservative with deletions, whereas length regularization successfully applies pressure for conciseness.
- **Preference Optimization vs. SFT**: Compared to vanilla Supervised Fine-Tuning (SFT), preference optimization delivers a massive retrieval performance gain (+5% vs. +1%), demonstrating the efficacy of contrastive learning on chosen/rejected pairs.
- **Strong cross-dataset generalization**: Consistent performance gains across 9 distinct types of datasets indicate that the learned compression policy is highly generalizable.

## Highlights & Insights

- **Adopting retrieval success rate as an automatic metric for compression quality**: This bypasses the challenging prerequisite of manually defining what a "good compression" should preserve, allowing downstream task performance to guide the compression policy. This "end-to-end, task-driven compression" paradigm can also be readily applied to RAG retrieval modules to compress retrieved passages before feeding them to the reader.
- **Denoising through compression insight**: The observation that retrieval improves post-compression reveals the existence of noise in the original passages that harms LCLM retrieval. This implies that inserting a task-aware compression step between reranking and generation in RAG pipelines could be highly beneficial.
- **Application of preference learning to text generation control**: Adapting DPO from alignment to compression scenarios demonstrates the versatile applicability of the preference learning framework in controlling specific text generation behaviors.

## Limitations & Future Work

- **Computational overhead of the compression model**: Although reducing the context length for LCLMs, the inference cost of the compression model itself (3.8B parameters) must be factored into the overall system efficiency.
- **Dependence on specific LCLMs**: Preference data are generated based on feedback from a specific LCLM. Utilizing a different target LCLM might require model retraining.
- **English-centric validation**: Although 9 datasets were evaluated, they are predominantly in English. The model's generalizability across multilingual scenarios remains to be tested.
- **Limited compression ratio**: A 1.91x compression ratio might not be aggressive enough for extremely long documents. Exploring how performance scales under higher compression ratios is an important future direction.

## Related Work & Insights

- **Vs. LongLLMLingua**: Prompt compression methods like LongLLMLingua primarily operate via token-level pruning without rewriting. In contrast, CoLoR relies on generative compression to reorganize information, theoretically offering a higher ceiling for performance.
- **Vs. Traditional Dense Retrieval**: Traditional methods require offline indexing and online retrieval. CoLoR combined with LCLMs places all passages directly into context, bypassing index building but suffering from heavy context engineering costs. CoLoR directly mitigates this overhead gap.
- **Vs. DPO in alignment**: Although using the same DPO framework, the reward signal is modified from "human preference" to "retrieval success rate", displaying the flexibility of DPO in non-alignment contexts.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing preference optimization to retrieval-oriented passage compression is a highly innovative combination, and the insight that "compression is denoising" is enlightening.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers a wide spectrum of 9 datasets with ablation studies on key components, though dataset-level breakdowns require checking the full paper.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, highly readable method descriptions, and the central value proposition ("retrieval success = good compression") is compellingly delivered.
- Value: ⭐⭐⭐⭐ Provides a highly practical solution to the efficiency bottlenecks of LCLM retrieval. The results of +6% performance and 1.91x compression possess significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](../../ICML2025/model_compression/core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)
- [\[ACL 2025\] APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)

</div>

<!-- RELATED:END -->
