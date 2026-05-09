---
title: >-
  [Paper Note] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning
description: >-
  [AAAI 2026][Graph Learning][Scholarly Information Extraction] This paper introduces GSAP-ERE — a fine-grained scholarly entity and relation extraction dataset for the machine learning domain, comprising 10 entity types and 18 relation types, with 63K entities and 35K relations annotated across 100 full-text papers. Experiments show that fine-tuned models (NER: 80.6%, RE: 54.0%) substantially outperform LLM prompting approaches (NER: 44.4%, RE: 10.1%).
tags:
  - AAAI 2026
  - Graph Learning
  - Scholarly Information Extraction
  - Named Entity Recognition
  - Relation Extraction
  - Knowledge Graph
  - ML Reproducibility
  - Fine-Grained Annotation
date: 2026-05-08
content_hash: 09ab13afe0e88c83
---

# GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning

**Conference**: AAAI 2026  
**arXiv**: [2511.09411](https://arxiv.org/abs/2511.09411)  
**Authors**: Wolfgang Otto, Lu Gan, Sharmila Upadhyaya, Saurav Karmakar, Stefan Dietze (GESIS)  
**Code**: [https://data.gesis.org/gsap/gsap-ere](https://data.gesis.org/gsap/gsap-ere)  
**Area**: Graph Learning  
**Keywords**: Scholarly Information Extraction, Named Entity Recognition, Relation Extraction, Knowledge Graph, ML Reproducibility, Fine-Grained Annotation  

## TL;DR

This paper introduces GSAP-ERE — a fine-grained scholarly entity and relation extraction dataset for the machine learning domain, comprising 10 entity types and 18 relation types, with 63K entities and 35K relations annotated across 100 full-text papers. Experiments show that fine-tuned models (NER: 80.6%, RE: 54.0%) substantially outperform LLM prompting approaches (NER: 44.4%, RE: 10.1%).

## Background & Motivation

### Problem Background
Machine learning research is advancing rapidly, yet reproducibility continues to decline — prior studies have found that only 4% of ML papers can be reproduced without a response from the original authors. Understanding the dependencies among models, datasets, and tasks is critical for improving research reproducibility and reusability. Scholarly Information Extraction (Scholarly IE) provides a scalable pathway to construct knowledge graphs and monitor research reproducibility by automatically extracting entities and their relations from papers.

### Limitations of Prior Work
- **Coarse-grained entity types**: SciERC defines only 6 entity types; SciER only 3 (Dataset, Task, Method), making it impossible to identify key meta-information such as model architectures and data sources.
- **Limited annotation scope**: ScienceIE annotates only paragraphs; SciERC and SemEval 2018 annotate only abstracts, failing to cover the diverse linguistic styles found in full text.
- **Poor direct LLM performance**: Existing LLMs fall far short of fine-tuned models on fine-grained domain-specific IE tasks and are unsuitable for high-quality scholarly IE.
- **Lack of informal entity annotation**: SciER includes only explicitly named entities, ignoring a large number of informal mentions (e.g., "the model", "this dataset"), leading to incomplete relation annotation.
- **Incomplete relation coverage**: Existing datasets define at most 9 relation types (SciER), insufficient to capture multi-dimensional relations such as model design, data provenance, and peer comparison.

### Core Motivation
To construct a comprehensive dataset with full-text coverage, fine-grained entity and relation annotations, supporting a variety of downstream tasks ranging from knowledge graph construction to AI research reproducibility monitoring.

## Method

### Data Model Design
Built upon an extension of the GSAP-NER dataset, the paper defines a complete entity-relation schema:

**10 Entity Types** (three major categories):
- ML model-related: MLModel, ModelArchitecture, MLModelGeneric, Method, Task
- Dataset-related: Dataset, DatasetGeneric, DataSource
- Others: ReferenceLink, URL

**18 Relation Types** (seven semantic groups):
1. **Model Design**: usedFor, architecture, isBasedOn — capturing compositional and derivation relations among models/methods
2. **Task Binding**: appliedTo, benchmarkFor — linking models/datasets to tasks
3. **Data Usage**: trainedOn, evaluatedOn — training and evaluation dependencies
4. **Data Provenance**: sourcedFrom, transformedFrom, generatedBy — data origins and transformations
5. **Data Properties**: size, hasInstanceType — dataset scale and modality
6. **Peer Relations**: coreference, isPartOf, isHyponymOf, isComparedTo — relations among entities of the same type
7. **Referencing**: citation, url — links to external sources

### Annotation Strategy
A two-phase "annotate-then-refine" strategy is adopted:
- **Annotation phase**: Two student annotators with computer science backgrounds; 10 papers doubly annotated, 90 papers singly annotated, using the INCEpTION platform.
- **Refinement phase**: Two PhD students and two postdoctoral researchers review annotation alignment, extract inconsistency patterns, and apply corrections.

### Evaluation Setup
Four levels of RE evaluation strictness are defined:
- **RE+**: Strict matching of entity types, relation labels, and entity boundaries.
- **RE**: Matching of relation labels and entity boundaries only, without entity type constraints.
- **RE+≈**: Strict label matching with partial overlap allowed for entity boundaries.
- **RE≈**: Correct relation label with overlapping entity boundaries sufficient.

### Baseline Models
- **Supervised Pipeline**: PL-Marker — NER followed by RE, using Packed Levitated Markers to model entity-pair interactions.
- **Supervised Joint Model**: HGERE — built on the PL-Marker framework with a hypergraph neural network, jointly optimizing NER and RE.
- **LLM Prompting**: Qwen 2.5 (32B/72B) and LLaMA 3.1 (70B), using a two-stage pipeline prompt (NER then RE).

## Key Experimental Results

### Experiment 1: Supervised Models vs. LLM Prompting

| Method | Model | NER | NER≈ | RE | RE≈ | RE+ | RE+≈ |
|--------|-------|-----|------|----|-----|-----|------|
| Supervised Joint | HGERE | **80.6** | **85.8** | **54.0** | **59.8** | **46.9** | **51.3** |
| Supervised Pipeline | PL-Marker | 72.6 | 77.7 | 41.4 | 46.2 | 36.3 | 39.9 |
| LLM Pipeline | Qwen 2.5 72B | 44.4 | 59.1 | 10.1 | 15.7 | 8.2 | 11.9 |
| LLM Pipeline | Qwen 2.5 32B | 42.0 | 56.9 | 7.2 | 14.6 | 7.2 | 10.9 |
| LLM Pipeline | LLaMA 3.1 70B | 40.5 | 55.0 | 6.4 | 9.6 | 5.7 | 7.8 |

Supervised HGERE outperforms all baselines across all metrics: NER exceeds the best LLM by 36.2 percentage points, and RE by 43.9 percentage points. In terms of inference speed, PLM-based methods are 182× faster than LLMs (4 minutes vs. 12.5 hours).

### Experiment 2: Effect of Few-Shot Example Selection Strategy on NER (Qwen2.5 32B, validation set)

| Selection Strategy | k=0 | k=1 | k=2 | k=5 | k=10 | k=20 |
|-------------------|-----|-----|-----|-----|------|------|
| random (micro-F1) | 19.1 | 24.7 | 23.1 | 29.7 | 34.1 | 34.4 |
| similar+diverse (micro-F1) | 19.1 | 34.7 | 38.2 | 40.4 | **40.9** | 27.8 |
| random (NER≈ micro-F1) | 33.0 | 41.8 | 37.1 | 50.1 | 53.3 | 50.3 |
| similar+diverse (NER≈ micro-F1) | 33.0 | 53.8 | 56.7 | 58.1 | **58.4** | 39.4 |

The similar+diverse strategy achieves optimal performance at k=10, outperforming the random strategy by approximately 5 percentage points; performance drops sharply at k=20. For RE, the best configuration is k=1 (micro-F1: 10.7%), and adding more examples is counterproductive.

### Dataset Scale Comparison

| Dataset | Annotation Unit | Papers | Entity Types | Relation Types | Entities | Relations | Relations/Paper |
|---------|----------------|--------|-------------|---------------|----------|-----------|----------------|
| **GSAP-ERE** | Full text | 100 | **10** | **18** | **62,619** | **35,302** | **353.0** |
| SciER | Full text | 106 | 3 | 9 | 24,518 | 12,083 | 114.0 |
| SciERC | Abstract | 500 | 6 | 7 | 8,094 | 4,648 | 9.3 |
| SemEval18 | Abstract | 500 | - | 6 | 7,505 | 1,583 | 3.3 |
| ScienceIE | Paragraph | 500 | 3 | 2 | 9,946 | 672 | 3.1 |

GSAP-ERE surpasses all existing datasets in entity count, relation count, type richness, and annotation density.

## Highlights & Insights

- **Largest scholarly IE dataset**: 63K entities + 35K relations, with 18 relation types covering 7 semantic dimensions; annotation density (353 relations/paper) far exceeds comparable datasets.
- **Fine-grained data model**: Distinguishes formal and informal entity mentions, captures multi-dimensional relations including model design, data provenance, and peer comparison, directly supporting ML research reproducibility monitoring.
- **Full-text annotation**: Compared to datasets annotating only abstracts or paragraphs, full-text annotation covers richer linguistic styles and information.
- **Rigorous quality control**: A two-stage annotate-then-refine pipeline achieves inter-annotator NER consistency of 0.82 macro-F1, with significant improvement after refinement.
- **Revealing LLM limitations**: Empirical results demonstrate that even the strongest current LLMs lag far behind fine-tuned models on fine-grained scholarly IE (RE gap of 43.9%), providing strong evidence for the necessity of domain-specific datasets.

## Limitations & Future Work

- **Domain scope**: Coverage is limited to ML and applied ML papers; generalizability to other disciplines (e.g., biomedicine, physics) remains to be verified.
- **Sentence-level annotation**: Current annotation supports only sentence-level entities and relations, lacking document-level cross-sentence relation annotation, making it impossible to capture long-range dependencies.
- **Dataset size**: With only 100 papers and 80 in the training set, the dataset may be insufficient for deep learning models.
- **Low RE performance ceiling**: Even the best supervised model achieves only 46.9% RE+ F1, indicating either extremely high task difficulty or the need for further refinement of the data model.
- **Variable inter-annotator agreement**: RE+ consistency for the Model Design semantic group is only 38.4%, reflecting ambiguous boundary definitions for some relations.
- **Nested relations not addressed**: Although the dataset contains nested and overlapping entities, the impact of these complex structures on model performance is not thoroughly analyzed.

## Related Work & Insights

- **SciERC (Luan et al. 2018)**: 6 entity types + 7 relation types, annotating 500 abstracts only; GSAP-ERE substantially surpasses it in type richness and annotation density, and provides full-text annotation.
- **SciER (Zhang et al. 2024)**: 3 entity types + 9 relation types, excluding informal entities; GSAP-ERE retains informal mentions, improving relation coverage completeness.
- **DMDD (Huitong et al. 2023)**: Automatically annotated via distant supervision with no relation annotation; GSAP-ERE is manually curated with rich relation annotations.
- **SciREX (Jain et al. 2020)**: Relation annotation is limited to mention clustering rather than pairwise relations, ignoring contextual information.
- **PL-Marker (Ye et al. 2022)**: The pipeline method achieves NER 72.6% and RE 41.4% on GSAP-ERE, below the joint method HGERE.
- **HGERE (Yan et al. 2023)**: The joint method achieves the best performance on GSAP-ERE (NER 80.6%, RE+ 46.9%), validating the effectiveness of hypergraph networks for scholarly IE.
- **LLM prompting methods**: Even with a 10-shot similar+diverse strategy, Qwen 2.5 72B and LLaMA 3.1 achieve RE below 11%, consistent with observations by Zhang et al. (2024) on SciER.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first full-text scholarly IE dataset combining 10 fine-grained entity types and 18 semantically grouped relation types, filling a clear gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both supervised and unsupervised methods with few-shot strategy ablations, though cross-domain generalization experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed data model definitions and comprehensive comparison tables.
- Value: ⭐⭐⭐⭐ — Provides a high-quality benchmark for ML reproducibility monitoring and knowledge graph construction, while exposing the limitations of LLMs on domain-specific IE.
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](../../ACL2026/graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)

</div>

<!-- RELATED:END -->
