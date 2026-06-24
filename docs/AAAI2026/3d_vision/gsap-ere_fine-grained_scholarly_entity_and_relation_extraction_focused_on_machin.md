---
title: >-
  [Paper Note] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning
description: >-
  [AAAI 2026][3D Vision][Scholarly Information Extraction] Proposes GSAP-ERE, a fine-grained scholarly entity and relation extraction dataset for the machine learning domain. It contains 10 entity types and 18 relation types, with 63K entities and 35K relations annotated across 100 full-text papers. Experimental results demonstrate that fine-tuned models (NER: 80.6%, RE: 54.0%) significantly outperform LLM prompting methods (NER: 44.4%, RE: 10.1%).
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Scholarly Information Extraction"
  - "Named Entity Recognition"
  - "Relation Extraction"
  - "Knowledge Graphs"
  - "ML Reproducibility"
  - "Fine-grained Annotation"
date: 2026-05-08
content_hash: 8141a25a24b36984
---

# GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning

**Conference**: AAAI 2026  
**arXiv**: [2511.09411](https://arxiv.org/abs/2511.09411)  
**Authors**: Wolfgang Otto, Lu Gan, Sharmila Upadhyaya, Saurav Karmakar, Stefan Dietze (GESIS)  
**Code**: [https://data.gesis.org/gsap/gsap-ere](https://data.gesis.org/gsap/gsap-ere)  
**Area**: Graph Learning  
**Keywords**: Scholarly Information Extraction, Named Entity Recognition, Relation Extraction, Knowledge Graphs, ML Reproducibility, Fine-grained Annotation  

## TL;DR

Proposes GSAP-ERE, a fine-grained scholarly entity and relation extraction dataset for the machine learning domain. It contains 10 entity types and 18 relation types, with 63K entities and 35K relations annotated across 100 full-text papers. Experimental results demonstrate that fine-tuned models (NER: 80.6%, RE: 54.0%) significantly outperform LLM prompting methods (NER: 44.4%, RE: 10.1%).

## Background & Motivation

### Background
Machine learning research is expanding rapidly, but reproducibility continues to decline—prior studies show that only 4% of ML papers are reproducible without author communication. Understanding dependencies among models, datasets, and tasks is crucial for improving reproducibility and reusability. Scholarly Information Extraction (Scholarly IE) provides a scalable pathway for knowledge graph construction and reproducibility monitoring by automatically extracting entities and their relations from papers.

### Limitations of Prior Work
- **Coarse-grained entity types**: SciERC contains only 6 entity types, and SciER only 3 (Dataset, Task, Method), failing to recognize key meta-information such as model architectures and data sources.
- **Limited annotation scope**: ScienceIE only annotates paragraphs, while SciERC and SemEval 2018 only annotate abstracts, failing to cover the diverse linguistic styles of full-text articles.
- **Poor direct application of LLMs**: Existing LLMs underperform fine-tuned models on fine-grained domain-specific IE tasks, making them unsuitable for direct deployment in high-quality scholarly IE.
- **Lack of informal entity annotations**: SciER only includes explicit named entities and ignores many informal mentions (e.g., "the model", "this dataset"), leading to incomplete relation annotations.
- **Incomplete relation coverage**: Existing datasets have at most 9 relation types (SciER), which fail to capture multidimensional relations such as model design, data provenance, and peer comparisons.

### Design Motivation
To build a comprehensive full-text dataset with fine-grained entity and relation annotations to support various downstream tasks, from knowledge graph construction to AI research reproducibility monitoring.

## Method

### Data Model Design
Extended from the GSAP-NER dataset, a comprehensive entity-relation schema is defined:

**10 Entity Types** (three major categories):
- ML Model-related: MLModel, ModelArchitecture, MLModelGeneric, Method, Task
- Dataset-related: Dataset, DatasetGeneric, DataSource
- Others: ReferenceLink, URL

**18 Relation Types** (seven semantic groups):
1. **Model Design**: usedFor, architecture, isBasedOn—capturing compositional and derivative relations between models/methods.
2. **Task Binding**: appliedTo, benchmarkFor—associating models/datasets with tasks.
3. **Data Usage**: trainedOn, evaluatedOn—dependencies of training and evaluation.
4. **Data Provenance**: sourcedFrom, transformedFrom, generatedBy—data sources and transformations.
5. **Data Properties**: size, hasInstanceType—dataset scale and modality.
6. **Peer Relations**: coreference, isPartOf, isHyponymOf, isComparedTo—relations between entities of the same category.
7. **Referencing**: citation, url—links to external sources.

### Annotation Strategy
A two-stage "annotation-refinement" strategy is employed:
- **Annotation Phase**: Two annotators with computer science backgrounds double-annotated 10 papers and single-annotated 90 papers using the INCEpTION platform.
- **Refinement Phase**: Two PhD students and two postdoctoral researchers reviewed the annotation alignment, extracted inconsistency patterns, and corrected them.

### Evaluation Setup
Four levels of strictness for RE evaluation are designed:
- **RE+**: Strict match of entity types, relation labels, and entity boundaries.
- **RE**: Only requires relation labels and entity boundaries to match, without constraining entity types.
- **RE+≈**: Strict match of labels with partial overlap allowed for entity boundaries.
- **RE≈**: Only requires the relation label to be correct and entity boundaries to overlap.

### Baseline Models
- **Supervised Pipeline**: PL-Marker—separates NER and RE, modeling entity pair interactions via Packed Levitated Markers.
- **Supervised Joint Model**: HGERE—introduces hypergraph neural networks based on the PL-Marker framework, optimizing NER and RE jointly.
- **LLM Prompting**: Qwen 2.5 (32B/72B) and LLaMA 3.1 (70B) under a two-stage pipeline prompting strategy (NER followed by RE).

## Key Experimental Results

### Main Results: Supervised Models vs. LLM Prompting

| Method | Model | NER | NER≈ | RE | RE≈ | RE+ | RE+≈ |
|------|------|-----|------|-----|-----|-----|------|
| Supervised Joint | HGERE | **80.6** | **85.8** | **54.0** | **59.8** | **46.9** | **51.3** |
| Supervised Pipeline | PL-Marker | 72.6 | 77.7 | 41.4 | 46.2 | 36.3 | 39.9 |
| LLM Pipeline | Qwen 2.5 72B | 44.4 | 59.1 | 10.1 | 15.7 | 8.2 | 11.9 |
| LLM Pipeline | Qwen 2.5 32B | 42.0 | 56.9 | 7.2 | 14.6 | 7.2 | 10.9 |
| LLM Pipeline | LLaMA 3.1 70B | 40.5 | 55.0 | 6.4 | 9.6 | 5.7 | 7.8 |

Supervised HGERE leads comprehensively across all metrics: outperforming the best LLM by 36.2 percentage points on NER and 43.9 percentage points on RE. In terms of inference speed, PLM-based methods are 182 times faster than LLMs (4 minutes vs. 12.5 hours).

### Ablation Study: Impact of Few-Shot Example Selection Strategies on NER (Qwen2.5 32B Dev Set)

| Selection Strategy | $k=0$ | $k=1$ | $k=2$ | $k=5$ | $k=10$ | $k=20$ |
|---------|-----|-----|-----|-----|------|------|
| random (micro-F1) | 19.1 | 24.7 | 23.1 | 29.7 | 34.1 | 34.4 |
| similar+diverse (micro-F1) | 19.1 | 34.7 | 38.2 | 40.4 | **40.9** | 27.8 |
| random (NER≈ micro-F1) | 33.0 | 41.8 | 37.1 | 50.1 | 53.3 | 50.3 |
| similar+diverse (NER≈ micro-F1) | 33.0 | 53.8 | 56.7 | 58.1 | **58.4** | 39.4 |

The similar+diverse strategy performs best at $k=10$, exceeding the random strategy by approximately 5 percentage points; performance drops sharply at $k=20$. The optimal configuration for RE is $k=1$ (micro-F1: 10.7%), whereas increasing the number of examples yields no benefit.

### Dataset Scale Comparison

| Dataset | Annotation Unit | Papers | Entity Types | Relation Types | Entities | Relations | Relations/Paper |
|--------|---------|--------|---------|---------|--------|--------|---------|
| **GSAP-ERE** | Full-text | 100 | **10** | **18** | **62,619** | **35,302** | **353.0** |
| SciER | Full-text | 106 | 3 | 9 | 24,518 | 12,083 | 114.0 |
| SciERC | Abstract | 500 | 6 | 7 | 8,094 | 4,648 | 9.3 |
| SemEval18 | Abstract | 500 | - | 6 | 7,505 | 1,583 | 3.3 |
| ScienceIE | Paragraph | 500 | 3 | 2 | 9,946 | 672 | 3.1 |

GSAP-ERE outperforms existing datasets across the board in the number of entities, relations, type richness, and annotation density.

## Highlights & Insights

- **Largest scholarly IE dataset**: 63K entities and 35K relations, with 18 relation types spanning 7 semantic dimensions; the annotation density (353 relations/paper) far exceeds similar datasets.
- **Fine-grained data model**: Distinguishes between formal and informal entity mentions, capturing multidimensional relations such as model design, data provenance, and peer comparisons, directly serving ML reproducibility monitoring.
- **Full-text level annotation**: Compared to datasets that only annotate abstracts or paragraphs, full-text annotation covers richer linguistic styles and information.
- **Rigorous quality control**: A two-stage annotation-refinement process, with inter-annotator agreement on NER reaching 0.82 macro-F1, which was further improved after refinement.
- **Revealing LLM limitations**: Empirical results show that current state-of-the-art LLMs lag significantly behind fine-tuned models on fine-grained scholarly IE (with a 43.9% gap in RE), providing strong evidence for the necessity of domain-specific datasets.

## Limitations & Future Work

- **Domain limitation**: Only covers papers in ML and applied ML, leaving other disciplines (e.g., biomedicine, physics) unaddressed; generalization remains to be validated.
- **Sentence-level annotation**: Currently only supports sentence-level entity and relation annotation, lacking document-level cross-sentence relation annotation, which fails to capture long-distance dependencies.
- **Dataset scale**: Only 100 papers, with 80 in the training set, which may be insufficient for training deep learning models.
- **Low ceiling for RE performance**: Even the best supervised model achieves only 46.9% RE+ F1, indicating extreme task difficulty or a need for further optimization of the data model.
- **Varying inter-annotator agreement**: The inter-annotator agreement (RE+) for the Model Design semantic group is only 38.4%, reflecting fuzzy boundaries in some relation definitions.
- **Nested relations not analyzed**: Although the dataset contains nested and overlapping entities, the impact of these complex structures on performance has not been analyzed thoroughly.

## Related Work & Insights

- **SciERC (Luan et al. 2018)**: 6 entities + 7 relations, annotating only 500 abstracts; GSAP-ERE significantly outperforms it in type richness and annotation density, with full-text annotation.
- **SciER (Zhang et al. 2024)**: 3 entities + 9 relations, excluding informal entities; GSAP-ERE preserves informal mentions, enhancing relation coverage completeness.
- **DMDD (Huitong et al. 2023)**: Automatically annotated via distant supervision with no relation annotations; GSAP-ERE is manually and meticulously annotated with rich relations.
- **SciREX (Jain et al. 2020)**: Relation annotation is limited to mention clustering rather than pairwise relations, ignoring contextual information.
- **PL-Marker (Ye et al. 2022)**: The pipeline method achieves 72.6% NER and 41.4% RE on GSAP-ERE, underperforming the joint method HGERE.
- **HGERE (Yan et al. 2023)**: The joint method achieves the best performance on GSAP-ERE (NER 80.6%, RE+ 46.9%), validating the effectiveness of hypergraph neural networks in scholarly IE.
- **LLM Prompting Methods**: Even with a 10-shot similar+diverse strategy, Qwen 2.5 72B and LLaMA 3.1 struggle with RE performance (under 11%), consistent with observations on SciER by Zhang et al. (2024).

## Rating

- Novelty: ⭐⭐⭐⭐ — The first full-text scholarly IE dataset combining 10 fine-grained entities and 18 semantically grouped relations, filling a gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both supervised and unsupervised methods and ablates few-shot strategies, but lacks cross-domain generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with a detailed data model definition and rich comparative tables.
- Value: ⭐⭐⭐⭐ — Provides a high-quality benchmark for ML reproducibility monitoring and knowledge graph construction, revealing LLM limitations in domain-specific IE.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] iSplat: Iterative Learning for Fine-Grained Gaussian Splatting](../../CVPR2026/3d_vision/isplat_iterative_learning_for_fine-grained_gaussian_splatting.md)
- [\[CVPR 2026\] Animator-Centric Skeleton Generation on Objects with Fine-Grained Details](../../CVPR2026/3d_vision/animator-centric_skeleton_generation_on_objects_with_fine-grained_details.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] Fresco: Frequency-Spatial Consistent Optimization for Fine-Grained Head Avatar Modeling](../../CVPR2026/3d_vision/fresco_frequency-spatial_consistent_optimization_for_fine-grained_head_avatar_mo.md)
- [\[CVPR 2026\] InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields](../../CVPR2026/3d_vision/infinidepth_arbitrary-resolution_and_fine-grained_depth_estimation_with_neural_i.md)

</div>

<!-- RELATED:END -->
