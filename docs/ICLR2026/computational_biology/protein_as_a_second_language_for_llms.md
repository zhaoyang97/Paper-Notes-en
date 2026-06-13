---
title: >-
  [Paper Note] Protein as a Second Language for LLMs
description: >-
  [Computational Biology] This work treats amino acid sequences as a "second language" for LLMs. By constructing a protein–natural language bilingual dataset and an adaptive context construction mechanism…
tags:
  - "Computational Biology"
date: 2026-05-08
content_hash: f47144f80f20e27c
---

# Protein as a Second Language for LLMs

- **Conference**: ICLR 2026
- **arXiv**: [2510.11188](https://arxiv.org/abs/2510.11188)
- **Code**: To be released
- **Area**: Bioinformatics / Protein Understanding / LLM Applications
- **Keywords**: protein understanding, in-context learning, LLM, second language acquisition, bilingual dataset

## TL;DR

This work treats amino acid sequences as a "second language" for LLMs. By constructing a protein–natural language bilingual dataset and an adaptive context construction mechanism, the proposed framework enables general-purpose LLMs to achieve an average ROUGE-L improvement of 7%—up to 17.2%—on protein question-answering tasks without any training, even surpassing domain-specific fine-tuned models.

## Background & Motivation

**Background**: Protein function understanding is dominated by two paradigms: (1) protein representation learning—self-supervised pre-training on amino acid sequences followed by task-specific decoders; and (2) protein–language alignment—establishing mappings between sequences and text via contrastive learning or multimodal LLMs. Both paradigms require large-scale training data, substantial computational resources, and task-specific fine-tuning.

**Limitations of Prior Work**: Embeddings from protein representation learning require an additional "translator" to produce human-readable explanations; protein–language alignment depends on large-scale paired data and must be re-fine-tuned whenever the downstream objective changes. Both lines of work are constrained in generalizability and scalability.

**Key Challenge**: Protein sequences are intrinsically "language-like" (fixed character set, compositional structure, contextual semantics), yet existing methods have not genuinely exploited this analogy—they still treat protein sequences as an independent modality rather than a language that LLMs can directly "read."

**Goal**: To design a **training-free** framework that enables general-purpose LLMs to understand the functional meaning of protein sequences solely through in-context examples.

**Key Insight**: The work draws on the cognitive science principles of **second language acquisition**—humans learning a new language rely on prior knowledge of their native language and infer word meanings through repeated encounters in context. LLMs can "acquire" protein language in the same manner.

**Core Idea**: Construct a protein–natural language **bilingual dataset** (79,926 QA pairs), design an **adaptive context construction mechanism** (dual-criterion retrieval based on sequence homology and text similarity), and enable LLMs to perform protein reasoning through in-context examples.

## Method

### Overall Architecture

The "Protein-as-Second-Language" framework proceeds in three stages:
1. **Bilingual Dataset Construction**: Swiss-Prot → GO-DAG pruning and grouping → sequence and functional redundancy removal → QA generation via DeepSeek-R1 → 79,926 protein QA triples
2. **Adaptive Context Construction**: For each query protein, examples are retrieved using both sequence homology and text similarity, then assembled into a bilingual context
3. **LLM Inference**: The constructed context and query are fed directly into a frozen LLM, which generates answers in a zero-training manner

### Key Designs

#### 1. Bilingual Dataset Construction (Three-Step Pipeline)

**Function**: Construct a balanced and diverse protein QA corpus from 573,661 Swiss-Prot entries.

**Mechanism**:
- **GO-DAG Pruning and Grouping**: A decision-tree-style pruning strategy is applied to the Gene Ontology DAG, using a depth-adaptive minimum support threshold $m(d) = \lambda \cdot C_{tot} \cdot (1 + \beta d)$ and a child-node imbalance ratio $\rho(v)$ to identify functional class nodes to retain.
- **Bilingual Redundancy Removal**: MMseqs2 is first applied at a 70% sequence similarity threshold to remove sequence redundancy; functional redundancy is then removed based on per-protein information content $\text{IC}_{\text{protein}}$.
- **LLM-Generated QA**: DeepSeek-R1 generates four QA types: attribute QA (11,693 pairs), knowledge QA, descriptive QA, and judgment QA (32,444 pairs).

**Design Motivation**: Directly converting all Swiss-Prot annotations introduces severe redundancy and severe class imbalance. The three-step pipeline controls data scale while preserving diversity.

#### 2. Adaptive Context Construction Mechanism

**Function**: Dynamically select the most informative in-context examples for each query protein.

**Mechanism**: A **dual-criterion retrieval** approach is employed:
- (i) Amino acid sequence homology (computed via MMseqs2)—capturing structural/functional similarity signals
- (ii) Descriptive text / QA text similarity—providing semantic grounding

Retrieved candidates are assembled into a coherent bilingual context by a context integration module and provided to the LLM as in-context examples together with the query.

**Design Motivation**: Sequence homology alone cannot capture functional semantics; text similarity alone lacks sequence-level patterns. Ablation experiments show that the dual-criterion approach outperforms sequence-only retrieval by 5.2% and text-only retrieval by 2.8%.

#### 3. Bilingual In-Context Learning

**Function**: Enable LLMs to infer the function of a query protein through analogical reasoning from examples.

**Mechanism**: The process fully mirrors second language acquisition—the LLM already possesses natural language as its "native tongue." By presenting paired examples of amino acid sequences and functional descriptions (bilingual corpus), the LLM infers the correspondence between sequence patterns and function within the context window, requiring no parameter updates.

### Loss & Training

This method is a **training-free** framework; no loss functions or gradient updates are involved. Evaluation employs the ROUGE-L automatic metric and human ratings on a 0–5 scale.

## Key Experimental Results

### Main Results: ROUGE-L Improvement Across LLMs

| Model | ProtDescribe (zero-shot → +context) | Protein2Text-QA (zero-shot → +context) | Mol-Inst. Avg (zero-shot → +context) |
|------|---------------------------|------------------------------|---------------------------|
| Qwen2.5-3B | 18.45 → **27.32** (+8.87) | 23.21 → **28.66** (+5.45) | 18.54 → **21.35** (+2.81) |
| Mistral-7B | 15.02 → **29.39** (+14.37) | 20.97 → **28.59** (+7.62) | 17.17 → **19.29** (+2.12) |
| Qwen3-14B | 23.20 → **35.53** (+12.33) | 21.02 → **25.93** (+4.91) | 14.61 → **19.82** (+5.21) |
| GPT-4o | 18.29 → **35.53** (+17.22) | 20.84 → **26.86** (+6.02) | 17.03 → **19.89** (+2.85) |
| ProLLaMA-7B (fine-tuned) | 12.77 | 10.09 | 16.85 |
| BioT5+ (fine-tuned) | 9.97 | 6.96 | 3.60 |

> Frozen general-purpose LLMs with context construction **substantially outperform** domain-fine-tuned models ProLLaMA and BioT5+. GPT-4o achieves an improvement of up to 17.22% on ProtDescribe.

### Ablation Study: Dual-Criterion vs. Single-Criterion Retrieval

| Retrieval Strategy | ProtDescribe (avg) | Protein2Text-QA (avg) | Mol-Inst. (avg) |
|---------|-------------------|---------------------|----------------|
| Dual (dual-criterion) | **32.73** | **26.22** | **19.14** |
| SeqOnly (sequence only) | 23.75 (−8.98) | 22.77 (−3.45) | 15.96 (−3.18) |
| QAOnly (text only) | 29.20 (−3.53) | 23.76 (−2.46) | 16.77 (−2.37) |

> Dual-criterion retrieval outperforms single-criterion retrieval across all datasets, confirming that sequence homology and text similarity provide complementary signals.

### Key Findings

1. **Larger models benefit more**: Qwen3-14B and GPT-4o exhibit the largest gains, suggesting that stronger in-context learning capacity better exploits bilingual context.
2. **Optimal number of examples varies by task**: ProtDescribe performs best at $k=10$–$11$; Protein2Text-QA peaks at $k=3$–$4$.
3. **General-purpose LLMs outperform domain fine-tuned models**: ProLLaMA-7B (fine-tuned) scores only 12.77 on ProtDescribe, whereas Qwen2.5-3B with context construction reaches 27.32.

## Highlights & Insights

- The "protein as a second language" conceptual framework is elegant, organically integrating second language acquisition theory with ICL.
- The approach is entirely training-free—surpassing fine-tuned models through context engineering alone, highlighting the untapped potential of ICL.
- The three-step redundancy-removal pipeline (GO pruning → sequence clustering → functional IC sampling) offers a replicable paradigm for constructing other bioinformatics datasets.
- The ablation analysis of dual-criterion retrieval clearly demonstrates the complementarity of the two information sources.

## Limitations & Future Work

- ROUGE-L primarily measures lexical overlap and cannot adequately assess biological accuracy or reasoning depth.
- Context window constraints limit the number of examples that can be provided, which may be insufficient for long-sequence proteins and complex reasoning chains.
- The dataset relies on automatic generation by DeepSeek-R1, potentially introducing LLM hallucinations and biases.
- Performance on higher-order tasks such as protein design and mutation analysis has not been evaluated.

## Related Work & Insights

- **ESM/ProtTrans**: Protein language models provide general-purpose embeddings but require task-specific decoders.
- **ProLLaMA** (Lv et al., 2024): Fine-tunes LLaMA on protein corpora, but with limited generalizability.
- **BioT5+** (Pei et al., 2024): Multi-task biomedical text fine-tuning; scores lower than frozen LLM + context.
- **Mol-Instructions** (Fang et al., 2024): Provides a protein instruction dataset used as an evaluation benchmark in this work.
- **Insights**: Treating biological sequences as a "new language" learnable by LLMs—rather than as an independent modality—opens new avenues for genomics, metabolomics, and related fields.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — The "protein as a second language" concept is original, and the result of a training-free approach outperforming fine-tuned models is impressive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple LLMs and datasets with thorough ablations; human evaluation enhances credibility.
- **Value**: ⭐⭐⭐⭐ — Zero-cost deployment, though reliant on a high-quality bilingual dataset.
- **Writing Quality**: ⭐⭐⭐ — The framework is presented clearly, though some tables are densely formatted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)
- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)

</div>

<!-- RELATED:END -->
