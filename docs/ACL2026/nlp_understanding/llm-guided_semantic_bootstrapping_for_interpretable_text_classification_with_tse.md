---
title: >-
  [Paper Note] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines
description: >-
  [ACL 2026][NLP Understanding][Tsetlin Machine] Ours proposes an LLM-guided semantic bootstrapping framework that uses LLMs to discover sub-intents and generate synthetic data via a three-stage curriculum. This data train…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Tsetlin Machine"
  - "Semantic Guidance"
  - "Symbolic Learning"
  - "Sub-intent Discovery"
  - "Interpretable Classification"
date: 2026-05-08
content_hash: 1819b78d0d7c367e
---

# LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.12223](https://arxiv.org/abs/2604.12223)  
**Code**: None  
**Area**: Interpretability / Text Classification  
**Keywords**: Tsetlin Machine, Semantic Guidance, Symbolic Learning, Sub-intent Discovery, Interpretable Classification

## TL;DR

Ours proposes an LLM-guided semantic bootstrapping framework that uses LLMs to discover sub-intents and generate synthetic data via a three-stage curriculum. This data trains a Non-Negated Tsetlin Machine (NTM) to extract high-confidence symbolic features for injection into real data, allowing standard TMs to approach BERT-level performance while maintaining full interpretability.

## Background & Motivation

**Background**: Tsetlin Machines (TM) have gained attention in interpretable NLP for their clause-level transparency, having been applied to document classification and sentiment analysis. Meanwhile, pre-trained language models like BERT provide powerful semantic representations but are costly and opaque.

**Limitations of Prior Work**: (1) TMs based on Boolean Bag-of-Words (BoW) fail to generalize across semantically similar but morphologically different expressions unless explicitly present in training data; (2) Enhancing TM inputs with Word2Vec/GloVe provides only limited semantic alignment; (3) BERT lacks decision traceability in high-stakes fields like law and healthcare.

**Key Challenge**: A fundamental contradiction exists between symbolic interpretability and semantic generalization—BoW representations ensure transparency at the cost of semantic understanding, while embedding representations capture semantics but sacrifice interpretability.

**Goal**: To transfer semantic knowledge from LLMs to TMs in symbolic form without introducing embedding layers or runtime LLM calls.

**Key Insight**: Utilize LLMs to generate interpretable sub-intents (e.g., positive_due_to_plot) and corresponding synthetic data, bridging the semantic gap through symbolic enhancement rather than embedding enhancement.

**Core Idea**: The LLM does not participate in classification inference; instead, it acts as a "semantic teacher" during the offline training phase, providing symbolic semantic priors for the TM through sub-intent decomposition and curriculum data generation.

## Method

### Overall Architecture

The framework consists of three stages: (1) LLM-guided sub-intent discovery and three-stage synthetic data generation (Seed→Core→Enriched); (2) Pre-training a Non-Negated TM (NTM) on synthetic data to extract high-confidence symbolic features; (3) Injecting NTM-extracted semantic features into the BoW representation of real data and fine-tuning a standard TM on the augmented representation. Final inference is entirely symbolic, requiring no LLM or embeddings.

### Key Designs

1.  **LLM-Guided Sub-intent Discovery and Three-stage Data Generation**:
    - **Function**: Decomposes class labels into interpretable semantic drivers and generates diverse training data.
    - **Mechanism**: The LLM decomposes each category into fine-grained sub-intents (e.g., positive→positive_due_to_plot, positive_due_to_acting). Synthetic data is then generated via a three-stage curriculum: the Seed stage generates 15-20 word canonical expressions as anchors; the Core stage maintains vocabulary stability but varies syntactic structure; the Enriched stage introduces synonyms and compositional phrases to expand the vocabulary space.
    - **Design Motivation**: Single-step LLM generation tends to collapse into high-probability patterns or overly generalized phrases; the three-stage strategy follows curriculum learning principles to ensure coverage, lexical diversity, and semantic fidelity—all vital for clause formation in Boolean symbolic models.

2.  **Non-Negated Tsetlin Machine (NTM)**:
    - **Function**: Extracts stable, high-confidence symbolic semantic features from synthetic data.
    - **Mechanism**: NTM modifies two aspects of the standard TM: (1) It eliminates negated literals, turning clauses into pure monotonic conjunctions $C_\iota^\kappa = \bigwedge_{k \in I_\iota^\kappa} x_k$; (2) It strengthens Type I feedback ($P_{\text{reward}}=1.0$, $P_{\text{penalty}}=0.0$), forcing Tsetlin Automata (TA) to converge rapidly to high-confidence literal sets. The literals with the deepest TA states are extracted as semantic indicators.
    - **Design Motivation**: Removing negated literals ensures monotonic interpretability of clause semantics—all learned rules reflect positively correlated lexical patterns; enhanced feedback ensures rapid and stable convergence on synthetic data.

3.  **Semantic Feature Injection and TM Fine-tuning**:
    - **Function**: Injects LLM-derived symbolic semantic knowledge into real data.
    - **Mechanism**: Real samples are passed through the NTM to predict sub-intents; high-confidence literals corresponding to activated clauses are collected, and their binary presence indicators are appended to the original BoW representation. A standard TM is then fine-tuned on this hybrid representation.
    - **Design Motivation**: Augmentation occurs offline, introducing no new components during inference—the final model remains purely symbolic and efficient. Semantic features provide cross-lexical associations absent in the raw BoW.

### Loss & Training

The NTM is trained using modified Type I/II feedback (150 clauses/sub-intent, T=5000, s=5). The standard TM is fine-tuned on augmented data using an integer-weighted variant. All synthetic data is generated by GPT-4o (nucleus sampling, p=0.9, temp=0.7).

## Key Experimental Results

### Main Results

**Performance Comparison across Six Classification Benchmarks**

| Method | AG-News | R8 | R52 | IMDB | SST2 | HoC |
|------|---------|-----|-----|------|------|-----|
| TM | 88.34 | 96.16 | 84.62 | 90.62 | 75.61 | 77.42 |
| TM (GloVe) | 90.12 | 97.50 | 89.14 | 90.88 | 76.38 | 78.78 |
| BERT | 94.75 | 97.49 | 94.26 | 93.46 | 94.00 | 82.90 |
| **LLM-Guided TM** | **93.10** | **97.88** | **94.45** | **92.10** | **85.24** | **81.90** |

### Ablation Study

**Improvement of TM Variant across Datasets**

| Dataset | TM→LLM-TM Gain | vs BERT Gap |
|--------|----------------|-------------|
| AG-News | +4.76% | -1.65% |
| R8 | +1.72% | +0.39% |
| R52 | +9.83% | +0.19% |
| SST2 | +9.63% | -8.76% |
| HoC | +4.48% | -1.00% |

### Key Findings

- LLM-Guided TM outperforms BERT on R8 and R52 while maintaining full symbolic interpretability.
- The largest gain occurs on SST2 (+9.63%), though the gap with BERT remains largest (-8.76%), indicating that short-text sentiment analysis still requires contextual understanding.
- Performance approaches BERT on the HoC biomedical dataset (81.90% vs 82.90%), where semantic decomposition effectively recovered compound words (e.g., immunosuppression → immune + suppression).
- Symbolic feature groups are semantically coherent: e.g., the politics sub-intent extracts {parliament, election, results}.
- The entire inference pipeline remains purely symbolic—no embeddings, no runtime LLM calls.

## Highlights & Insights

- The philosophy of "LLM as a semantic teacher rather than a classifier" is elegant—leveraging the LLM's world knowledge while completely avoiding its inference overhead.
- Sub-intent decomposition ensures the augmented features themselves are interpretable, unlike embedding enhancements which introduce black boxes.
- The three-stage curriculum generation strategy is particularly important for clause learning in Boolean symbolic models—the balance between lexical stability and diversity is key.

## Limitations & Future Work

- Dependency on LLM generation quality—sub-intents may be inaccurate in complex or overlapping domains.
- Removing negated literals improves interpretability but reduces expressiveness, preventing the capture of negative logic.
- Lack of systematic hyperparameter ablation (number of clauses, number of synthetic samples, weighting schemes, etc.).
- The significant gap with BERT on SST2 suggests a bottleneck in contextual understanding for short texts.

## Related Work & Insights

- **vs TM (GloVe)**: GloVe enhancement provides static word vector alignment; the sub-intent guidance in this paper provides structured semantic associations, resulting in a +5.31% gain on R52.
- **vs BERT**: BERT maintains an advantage across most tasks (except R8/R52) but at the cost of interpretability. Ours closes most of the gap while maintaining symbolic transparency.
- **vs Symbolic Distillation**: Existing methods typically distill into decision trees or linear rules; ours is the first to distill into clause logic.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of symbolically transferring LLM semantic knowledge to Tsetlin Machines is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 datasets covering multiple domains, though missing minor ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, and case studies are persuasive.
- Value: ⭐⭐⭐⭐ Provides a practical solution for high-stakes scenarios requiring interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[ACL 2026\] Accurate and Efficient Statistical Testing for Word Semantic Breadth](accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)

</div>

<!-- RELATED:END -->
