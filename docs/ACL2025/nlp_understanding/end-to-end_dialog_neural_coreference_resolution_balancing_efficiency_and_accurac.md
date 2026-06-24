---
title: >-
  [Paper Note] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems
description: >-
  [ACL 2025][NLP Understanding][Coreference Resolution] An end-to-end neural coreference resolution system is proposed. By combining contextual embeddings, a hierarchical attention mechanism, and optimization strategies (pruning/quantization), it achieves a balance between efficiency and accuracy, with SpanBERT reaching 87.3 F1 on benchmark datasets such as OntoNotes.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Coreference Resolution"
  - "End-to-End Systems"
  - "Attention Mechanism"
  - "Contextual Embeddings"
  - "Large-Scale Systems"
date: 2026-05-08
content_hash: a0b14bd573da3088
---

# End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems

**Conference**: ACL 2025  
**arXiv**: [2504.05824](https://arxiv.org/abs/2504.05824)  
**Code**: None  
**Area**: NLP Understanding  
**Keywords**: Coreference Resolution, End-to-End Systems, Attention Mechanism, Contextual Embeddings, Large-Scale Systems

## TL;DR

An end-to-end neural coreference resolution system is proposed. By combining contextual embeddings, a hierarchical attention mechanism, and optimization strategies (pruning/quantization), it achieves a balance between efficiency and accuracy, with SpanBERT reaching 87.3 F1 on benchmark datasets such as OntoNotes.

## Background & Motivation

**Background**: Coreference Resolution is a fundamental task in NLP, aiming to identify different expressions in text that refer to the same entity. In recent years, BERT-based methods have made significant progress, but efficiency remains a bottleneck in large-scale applications.

**Limitations of Prior Work**: Existing methods face various challenges: cache miss issues when processing long documents (Guo et al., 2023), performance disparities across multilingual environments (Pražák & Konopík, 2024), and deficiencies in singleton detection (Zou et al., 2024). High-performance models (such as large models with 13B parameters) incur massive computational overhead, making them difficult to deploy in actual production environments.

**Key Challenge**: In large-scale text processing, there is a trade-off between model accuracy and inference efficiency. Increasing model depth and the number of attention heads can improve the F1 score but simultaneously increases inference latency and computational resource requirements.

**Goal**: Design an end-to-end coreference resolution system that runs efficiently while maintaining high accuracy, making it suitable for real-world deployment.

**Key Insight**: Redesign the coreference decision process through a hierarchical attention mechanism and combine it with model optimization strategies (pruning and quantization) to reduce computational overhead.

**Core Idea**: Construct an efficiency-accuracy balanced end-to-end coreference resolution system using multi-layer contextual embeddings, hierarchical attention, and directed graph optimization.

## Method

### Overall Architecture

The system is divided into two stages:
1. **Mention Representation**: Generate a dynamic embedding representation $\mathcal{C} = \{c_1, c_2, ..., c_n\}$ for each mention through contextual embeddings, utilizing a combination of RNN and Transformer.
2. **Coreference Linkage Decision**: Construct an affinity matrix via the attention mechanism, and then select the optimal coreference links using directed graph optimization.

### Key Designs

**Contextual Embedding Module**: For an input text $x = \{w_1, w_2, ..., w_n\}$, it is projected into a high-dimensional embedding space through a mapping function $\phi$:

$$\mathcal{E} = \{\phi(w_1), \phi(w_2), ..., \phi(w_n)\}$$

Attention score calculation:

$$a_{ij} = \frac{\exp(\alpha(\mathcal{E}_i, \mathcal{E}_j))}{\sum_{k=1}^{n} \exp(\alpha(\mathcal{E}_i, \mathcal{E}_k))}$$

The refined representation is $R_i = \sum_{j=1}^{n} a_{ij} \mathcal{E}_j$, which merges original semantics and contextual information.

**Coreference Linkage Resolution**: Construct an affinity matrix $A \in \mathbb{R}^{n \times n}$. The attention scores take the scaled dot-product form:

$$a_{ij} = \text{softmax}\left(\frac{e(c_i, c_j)}{\sqrt{d}}\right)$$

The optimization objective is to find the optimal coreference link subset $\mathcal{L} \subseteq \mathcal{C} \times \mathcal{C}$, constraining each mention to link to at most one antecedent.

### Loss & Training

The loss function considers both precision and recall:

$$\mathcal{L}_{CR} = -(\alpha \cdot \log(P) + \beta \cdot \log(1-P))$$

where $P$ is the probability of establishing a coreference link, and $\alpha$ and $\beta$ control the weights of precision and recall, respectively.

Training configuration: 30 epochs, batch size=16, learning rate=3e-5 (AdamW), sequence length 512, trained on NVIDIA V100 GPU. The data is split into 80%/10%/10% (train/validation/test).

## Key Experimental Results

### Main Results

| Method | Dataset | F1 | Precision | Recall |
|---|---|:---:|:---:|:---:|
| BERT-large | OntoNotes v5.0 | 86.2 | 85.0 | 87.5 |
| **SpanBERT** | **OntoNotes v5.0** | **87.3** | **86.5** | **88.1** |
| Z-coref | Winograd Schema | 82.1 | 80.3 | 83.5 |
| Seq2seq | NusaCrowd | 81.5 | 79.7 | 83.3 |
| Knowledge Integration | BARThez | 84.8 | 83.2 | 86.5 |
| Event Coreference | CliCR | 79.9 | 78.6 | 81.0 |
| Event-aware Learning | OntoNotes v5.0 | 85.0 | 83.5 | 86.5 |

### Ablation Study

| Configuration | Dataset | F1 | Change |
|---|---|:---:|:---:|
| BERT-large (No Attention) | OntoNotes v5.0 | 84.5 | -1.7 |
| BERT-large (Static Embeddings) | OntoNotes v5.0 | 85.2 | -1.0 |
| Seq2seq (No Optimization) | NusaCrowd | 79.7 | -1.8 |
| Z-coref (Fixed Learning Rate) | Winograd Schema | 80.6 | -1.5 |
| Event Coreference (Reduced Layers) | CliCR | 77.5 | -2.4 |
| Event-aware (No Attention) | OntoNotes v5.0 | 82.0 | -3.0 |

### Key Findings

1. **Attention mechanism is crucial**: Removing attention leads to an F1 drop of 1.7-3.0 points. Hierarchical attention (87.3 F1) outperforms multi-head attention (86.2 F1) and adaptive attention (86.8 F1).
2. **Transformer embeddings significantly outperform traditional methods**: BERT (86.0) and RoBERTa (86.5) far exceed Word2Vec, GloVe, and FastText (83.0).
3. **Hybrid models perform best**: The hybrid identification technique reaches 88.2 F1, outperforming single deep learning methods (86.5).
4. **Model depth has a significant impact**: Reducing layers drops the F1 on CliCR from 79.9 to 77.5.

## Highlights & Insights

- **Two-stage design**: First generates mention embeddings, and then optimizes linkage decisions through directed graphs, decomposing a complex problem into manageable sub-problems.
- **Efficiency-accuracy trade-offs**: Significantly reduces inference overhead while maintaining accuracy through pruning and quantization strategies.
- **Unified ablation framework**: Systematically compares attention mechanisms, embedding types, and identification techniques, providing a reference for follow-up research.

## Limitations & Future Work

- Resource consumption remains high on highly complex texts, limiting deployment in resource-constrained environments.
- Mainly relies on benchmark datasets for evaluation; generalization capability on diverse, real-world texts has not been fully verified.
- Robustness to noisy data needs to be strengthened.
- The potential of larger-scale pretrained models (e.g., decoder-only architectures like Llama) has not been explored.
- Different datasets use different baselines but identical training configurations (unified lr=3e-5, batch=16), raising questions about fairness and targeted tuning.

## Related Work & Insights

- Inspiration for long document processing from Guo et al. (2023)'s dual-cache design—separating global and local entities.
- The sentence-level incremental method of Grenander et al. (2023) provides insight for real-time processing.
- The success of hybrid methods (rules + neural networks) suggests that explicit injection of domain knowledge remains valuable in coreference resolution.
- Comparison with Maverick by Martinelli et al. (2024): achieving SOTA with 500M parameters shows that parameter-efficient design is a key direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The overall framework is relatively standard, representing an engineering integration and optimization of BERT-based methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets and ablation experiments are provided, but the practice of using a unified training configuration for all methods is debatable.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, but some descriptions are rather verbose, and the methodological innovation is not prominently articulated.
- **Value**: ⭐⭐⭐⭐ — Provides a comprehensive system design reference for coreference resolution, but with limited breakthrough novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [\[ICML 2025\] Cover Learning for Large-Scale Topology Representation](../../ICML2025/nlp_understanding/cover_learning_for_large-scale_topology_representation.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)

</div>

<!-- RELATED:END -->
