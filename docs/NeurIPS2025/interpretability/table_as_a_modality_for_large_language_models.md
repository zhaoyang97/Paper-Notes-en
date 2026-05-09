---
title: >-
  [Paper Note] Table as a Modality for Large Language Models
description: >-
  [NeurIPS 2025][Table Reasoning] This paper proposes TaMo, a framework that treats tables as an independent modality, encoding their structural information via a hypergraph neural network and fusing the resulting structural embeddings with the text modality of an LLM. TaMo achieves an average improvement of 42.65% over pure-text methods across multiple table reasoning benchmarks, and approaches GPT-4 in terms of structural robustness.
tags:
  - NeurIPS 2025
  - Table Reasoning
  - Multimodal LLM
  - Hypergraph
  - Permutation Invariance
  - Table QA
date: 2026-05-08
content_hash: 663e8b797486e597
---

# Table as a Modality for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2512.00947](https://arxiv.org/abs/2512.00947)
**Code**: [Available](https://github.com/liyaooi/TAMO)
**Area**: Interpretability
**Keywords**: Table Reasoning, Multimodal LLM, Hypergraph, Permutation Invariance, Table QA

## TL;DR

This paper proposes TaMo, a framework that treats tables as an independent modality, encoding their structural information via a hypergraph neural network and fusing the resulting structural embeddings with the text modality of an LLM. TaMo achieves an average improvement of 42.65% over pure-text methods across multiple table reasoning benchmarks, and approaches GPT-4 in terms of structural robustness.

## Background & Motivation

The dominant paradigm for handling table tasks in LLMs is to serialize tables into text (e.g., Markdown format) before feeding them into the model. This approach, however, has a fundamental flaw: **structural information is lost during serialization**.

The paper introduces the StructQA benchmark—a diagnostic dataset focused on table structure understanding—to expose this problem. StructQA explicitly accounts for the **permutation invariance** property inherent to tabular data: arbitrary row/column reordering should not alter the semantics of a table, and a model should produce consistent answers for structurally equivalent tables. Experiments reveal that:

- Mainstream LLMs, including Llama2-7B, GPT-3.5, and GPT-4, suffer significant performance degradation when presented with permuted tables.
- All models except GPT-4 achieve answer robustness below 40%.
- TableLlama, a model specifically trained on tables, reaches only 6.47% on StructQA.

Such failures are trivial for humans, indicating that the root cause is not insufficient comprehension ability but rather a **representational bottleneck introduced by serialization**. The paper's core insight is that, just as images and audio require dedicated encoders, tables should also be treated as an independent modality.

## Method

### Overall Architecture

TaMo is a multimodal framework consisting of three components:
1. **Hypergraph-augmented table encoder**: captures table structural information and produces structural embeddings.
2. **Modality alignment interface**: projects structural embeddings into the LLM's semantic space.
3. **LLM reasoning**: fuses structural and textual embeddings and generates answers autoregressively.

### Key Designs

1. **Hypergraph modeling of table structure**:

    - A table is modeled as a hypergraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: leaf cells (those containing no sub-cells) serve as nodes, while branch cells (e.g., headers containing sub-cells) serve as hyperedges.
    - For simple flat tables: each cell is a node, and each row/column constitutes a hyperedge.
    - For complex hierarchical tables (e.g., HiTab): the hierarchical relationships naturally translate into a hypergraph structure.
    - Core motivation: row/column permutation only changes the ordering, not the graph structure (the sets of nodes and edges remain invariant), thereby satisfying permutation invariance by construction.

2. **HyperTrans encoder**:

    - Two multiset functions alternately update node and hyperedge representations.
    - Node→hyperedge aggregation: $\mathbf{x}_e^{t+1} = \text{Fusion}(\mathbf{x}_e^t, \text{Multiset}_1(\{\mathbf{x}_v^t | v \in e\}))$
    - Hyperedge→node aggregation: $\mathbf{x}_v^{t+1} = \text{Multiset}_2(\{\mathbf{x}_e^{t+1} | v \in e\})$
    - The multiset functions are parameterized via Set Transformers, incorporating multi-head attention and feed-forward networks.
    - The permutation invariance of the multiset functions guarantees the overall permutation invariance of the encoder.

3. **Modality alignment and fusion**:

    - An MLP projects the pooled node and hyperedge representations from the hypergraph encoder into the LLM embedding space: $\mathbf{X}_{st} = \text{MLP}(\text{Pooling}(\hat{\mathbf{X}}_\mathcal{V}, \hat{\mathbf{X}}_\mathcal{E}))$
    - The text-serialized table embeddings $\mathbf{X}_{tt}$ are retained in parallel, providing fine-grained semantic content ("what").
    - The structural embeddings $\mathbf{X}_{st}$ are injected at the front of the LLM as soft prompts, supplying global relational context ("where").
    - The two streams are complementary and non-redundant: ablation experiments confirm that removing either degrades performance.

### Loss & Training

- Standard next-token prediction (autoregressive cross-entropy loss) is employed.
- Three training modes are supported:
    - **Frozen LLM**: only the table encoder and alignment layer are trained.
    - **LoRA**: additionally applies LoRA fine-tuning to the LLM.
    - **SFT**: full-parameter supervised fine-tuning.
- Models are trained independently on each dataset to establish an upper-bound performance reference.

## Key Experimental Results

### Main Results

| Setting | StructQA | HiTab | WikiTQ | WikiSQL | FeTaQA |
|---------|----------|-------|--------|---------|--------|
| Zero-shot | 8.60 | 7.77 | 14.50 | 21.44 | 20.08 |
| Prompt Tuning | 37.80 | 26.26 | 29.86 | 61.24 | 29.94 |
| **TaMo (Frozen)** | **59.07** | **48.86** | **37.06** | **76.45** | **36.52** |
| △ vs Prompt Tuning | ↑56.27% | ↑86.06% | ↑24.11% | ↑24.84% | ↑21.98% |
| LoRA | 45.67 | 50.76 | 37.13 | 57.10 | 35.80 |
| **TaMo+LoRA** | **70.80** | **59.22** | **43.53** | **84.43** | **37.43** |
| SFT | 62.73 | 54.80 | 43.28 | 79.86 | 37.37 |
| **TaMo+SFT** | **71.60** | **63.89** | **45.81** | **85.90** | **39.01** |
| GPT-4 | 51.40 | 48.40 | 68.40 | 47.60 | 21.70 |
| DeepSeek-R1 | 57.47 | 63.89 | 75.76 | 71.91 | 13.10 |

TaMo achieves an average gain of **42.65%** under the Frozen LLM setting. On StructQA and WikiSQL, TaMo+SFT substantially outperforms GPT-4.1 and DeepSeek-R1.

### Ablation Study: Evaluation of Structural Learning in the Table Encoder

| Configuration | F1 Score | Notes |
|---------------|----------|-------|
| MLP head (no encoder) | 5.39 | Unable to recognize row/column structure |
| + randomly initialized encoder | 49.73 | Hypergraph inductive bias itself is effective |
| + StructQA pre-trained encoder | **71.32** | Best structural learning |
| + WikiTQ pre-trained encoder | 62.63 | Cross-dataset generalization |
| + WikiSQL pre-trained encoder | 68.00 | Cross-dataset generalization |

### Key Findings

- Structural information yields the largest gains for frozen LLMs (+42.65%), demonstrating that structure conveys information that text serialization cannot capture.
- Attention visualizations show that TaMo causes the LLM to attend more to tokens relevant to the correct answer (e.g., the correct cell and related context).
- Under permutation robustness testing, TaMo consistently outperforms pure-text methods and achieves the highest answer consistency.
- TaMo+SFT with 7B parameters surpasses GPT-4.1 and DeepSeek-R1 on structure-intensive tasks.

## Highlights & Insights

- The notion of **treating tables as an independent modality** is highly novel; the analogy to multimodal LLM designs for images and audio is both natural and effective.
- Modeling tables as hypergraphs is an elegant choice: it handles hierarchical tables naturally, incorporates permutation invariance by design, and provides a unified representation for both simple and complex tables.
- The StructQA benchmark is itself a significant contribution, exposing a fundamental deficiency in current LLMs' table understanding capabilities.
- As a plug-and-play module that requires no modification to the LLM architecture, TaMo exhibits strong generalizability.

## Limitations & Future Work

- The framework relies on pre-structured table inputs; tables embedded within unstructured text require additional preprocessing.
- Only single-turn, static table understanding is currently supported; dynamic multi-step reasoning and multi-turn dialogue remain unexplored.
- The interaction between different text serialization templates (Markdown vs. SQL) and the structural modality has not been systematically studied.
- Although cross-dataset generalization is partially demonstrated, large-scale pre-training on multimodal instruction data is absent.

## Related Work & Insights

- Compared to traditional table models such as TAPAS/TAPEX: TaMo designs a modality interface specifically for decoder-only LLMs.
- Compared to table encoders (TaBERT, TabNet, HyTrel): those approaches perform representation learning only and cannot support joint text-table reasoning.
- The design principles of TaMo offer valuable reference for future LLM systems handling structured data such as knowledge graphs and databases.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of "table as a modality" is proposed and systematically validated for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five datasets, multiple training settings, ablation studies, and attention visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; the diagnostic experiments on StructQA are highly convincing.
- Value: ⭐⭐⭐⭐⭐ — Represents a substantive advance in LLM-based table reasoning with a practically strong plug-and-play design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)

<!-- RELATED:END -->
