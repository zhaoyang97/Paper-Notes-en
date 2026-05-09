---
title: >-
  [Paper Note] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment
description: >-
  [ICLR 2026][LLM Evaluation][Network Alignment] This paper presents PlanetAlign, a PyTorch-based network alignment benchmark library integrating 18 datasets across 6 domains, 14 methods spanning three categories (consistency-based, embedding-based, and optimal transport-based), and a standardized evaluation pipeline. Through large-scale systematic experiments, PlanetAlign reveals that OT-based methods (PARROT/JOENA) achieve comprehensive superiority in effectiveness, while different method categories exhibit distinct trade-offs in scalability and robustness.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Network Alignment
  - Benchmark Library
  - Graph Matching
  - Optimal Transport
  - Evaluation Framework
date: 2026-05-08
content_hash: 1ff6302fd9fe075f
---

# PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment

**Conference**: ICLR 2026
**arXiv**: [2505.21366](https://arxiv.org/abs/2505.21366)
**Code**: [GitHub](https://github.com/yq-leo/PlanetAlign)
**Area**: LLM Evaluation
**Keywords**: Network Alignment, Benchmark Library, Graph Matching, Optimal Transport, Evaluation Framework

## TL;DR

This paper presents PlanetAlign, a PyTorch-based network alignment benchmark library integrating 18 datasets across 6 domains, 14 methods spanning three categories (consistency-based, embedding-based, and optimal transport-based), and a standardized evaluation pipeline. Through large-scale systematic experiments, PlanetAlign reveals that OT-based methods (PARROT/JOENA) achieve comprehensive superiority in effectiveness, while different method categories exhibit distinct trade-offs in scalability and robustness.

## Background & Motivation

**State of the Field**: Network Alignment (NA) aims to discover node correspondences across different networks, serving as a critical foundation for downstream tasks such as cross-social-network user matching, protein homology discovery, knowledge graph fusion, and fraud detection. The field has developed three major categories of methods: consistency-based (e.g., IsoRank, FINAL), embedding-based (e.g., REGAL, BRIGHT), and optimal transport-based (e.g., PARROT, JOENA). However, systematic comparisons among these methods have been largely absent.

**Limitations of Prior Work**: The five existing NA benchmarks/libraries (SGAPBSA, CAPABN, ASNets, NAB, OpenEA) all exhibit notable limitations: (1) datasets are confined to a single domain—SGAPBSA and CAPABN cover only biological networks, ASNets only social networks, and OpenEA only knowledge graphs; (2) method coverage is incomplete—none of the existing libraries includes the latest and best-performing OT-based methods; (3) evaluation dimensions are narrow—most assess only effectiveness while ignoring scalability and robustness, and inconsistent dataset splitting strategies hinder reproducibility.

**Root Cause**: While the repertoire of NA methods has grown rapidly, the evaluation infrastructure has lagged far behind. Researchers report results under heterogeneous datasets, splitting strategies, and metrics, making fair performance comparisons impossible and impeding research progress.

**Paper Goals**: To construct a comprehensive, unified, and user-friendly NA benchmark library covering multi-domain datasets, multi-category methods, and multi-dimensional evaluation.

**Starting Point**: Drawing inspiration from successful benchmark library designs in CV/NLP (e.g., MMDetection, HuggingFace), this work addresses the fragmented evaluation landscape in NA research through unified API design, standardized data splitting, and reproducible evaluation pipelines.

**Core Idea**: By constructing PlanetAlign—a unified benchmark library spanning 6 domains and 3 method categories—this work enables, for the first time, systematic and fair comparison of NA methods across four dimensions: effectiveness, scalability, robustness, and supervision sensitivity.

## Method

### Overall Architecture

PlanetAlign is a PyTorch-based Python library organized into three layers: (1) **Data Layer**—18 datasets covering 6 domains including social networks, publication networks, biological networks, knowledge graphs, infrastructure networks, and communication networks; (2) **Algorithm Layer**—14 NA methods uniformly encapsulated in classes inheriting from `BaseModel`, accessible via `.train()` and `.test()` APIs; (3) **Evaluation Layer**—standardized Hits@K and MRR metrics, along with tools for time/memory profiling and robustness testing. Users can complete the full pipeline of dataset loading, model training, and evaluation with just a few lines of code.

### Key Designs

1. **Comprehensive Dataset Collection and Synthesis**:

    - **Function**: Provides 18 datasets across 6 domains, comprising 11 real-world datasets and 7 synthetic datasets.
    - **Mechanism**: Synthetic datasets are generated via the classical network perturbation strategy—inserting 10% noisy edges and removing 15% existing edges from the original network to produce two networks with a known permutation. Domains covered include social (Foursquare-Twitter, Douban, and 2 others), publication (ACM-DBLP, Cora, ArXiv, and 1 other), biological (SacchCere, PPI, GGI, and others), knowledge graph (DBP15K ZH-EN/JA-EN/FR-EN), infrastructure (Italy, Airport, PeMS08, and others), and communication (Phone-Email, Arenas, and others).
    - **Design Motivation**: Prior libraries cover at most 1–2 domains, making it impossible to assess cross-domain generalization. Broad domain coverage reveals performance differences of NA methods under varying network structural characteristics.

2. **Unified Implementation of 14 Methods across Three Categories**:

    - **Function**: Provides unified PyTorch implementations of consistency-based methods (IsoRank, FINAL), embedding-based methods (IONE, REGAL, CrossMNA, NetTrans, WAlign, BRIGHT, NeXtAlign, WLAlign), and OT-based methods (PARROT, SLOTAlign, HOT, JOENA).
    - **Mechanism**: All methods inherit from the `BaseModel` base class with unified `.train()` / `.test()` interfaces. Built-in utility functions such as random walk with restart (RWR) embeddings and anchor node embeddings facilitate integration of new methods with minimal code. Compared to official implementations, PlanetAlign achieves up to 3× speedup while maintaining comparable effectiveness.
    - **Design Motivation**: This is the first unified benchmark to incorporate OT-based methods—the most recent and highest-performing direction in NA—filling the largest gap in existing libraries.

3. **Multi-Dimensional Standardized Evaluation Tools**:

    - **Function**: Supports evaluation across four dimensions: effectiveness (Hits@K, MRR), scalability (time/memory), robustness (injection of edge/attribute/supervision noise), and supervision sensitivity (varying training ratios).
    - **Mechanism**: Effectiveness metrics support bidirectional alignment ($\mathcal{G}_1 \to \mathcal{G}_2$ and the reverse), reporting the average; scalability is tracked automatically via a built-in Logger for runtime and peak memory; robustness is assessed by injecting various types and levels of noise via utility functions to measure method degradation; unified random seeds and data splitting ensure reproducibility.
    - **Design Motivation**: Prior libraries rely on a single evaluation dimension (typically Hits@1 only), which is insufficient to characterize the practical applicability of methods.

## Key Experimental Results

### Main Results: Effectiveness and Efficiency

14 methods are evaluated on datasets from 6 domains (training ratio 20%), reporting average Hits@1, Hits@10, and MRR (%):

| Method | Category | Social H@1 | Publication H@1 | Bio H@1 | KG H@1 | Infra H@1 | Comm H@1 |
|--------|----------|-----------|----------------|---------|--------|-----------|---------|
| JOENA | OT | **18.7** | **73.2** | **63.7** | **66.3** | **62.9** | **66.3** |
| PARROT | OT | 12.6 | 66.6 | 61.6 | 66.0 | 51.8 | 63.3 |
| NetTrans | Embedding | 7.2 | 40.7 | 34.2 | 28.8 | 29.3 | 45.2 |
| BRIGHT | Embedding | 5.1 | 40.4 | 30.5 | 30.4 | 29.9 | 50.9 |
| NeXtAlign | Embedding | 7.1 | 43.2 | 25.9 | 27.5 | 28.0 | 29.6 |
| FINAL | Consistency | 4.9 | 22.3 | 22.9 | 13.9 | 15.1 | 21.7 |
| IsoRank | Consistency | 4.2 | 18.9 | 21.6 | 11.5 | 14.2 | 22.1 |
| REGAL | Embedding | 0.3 | 1.8 | 1.0 | 0.8 | 2.8 | 45.3 |

OT-based methods (JOENA, PARROT) achieve the best Hits@1 across all 6 domains, with particularly pronounced advantages on knowledge graphs and infrastructure networks.

### Ablation Study: Efficiency and Scalability

| Method | Category | Social Time (s) | Social Mem (GB) | Pub. Time (s) | Pub. Mem (GB) |
|--------|----------|----------------|----------------|--------------|--------------|
| WAlign | Embedding | **0.61** | 2.65 | 9.41 | 9.88 |
| REGAL | Embedding | 9.38 | **1.16** | 16.14 | **3.18** |
| FINAL | Consistency | 5.91 | 5.39 | 6.75 | 10.06 |
| PARROT | OT | — | — | — | — |
| JOENA | OT | — | — | — | — |
| IONE | Embedding | $6.34\times10^3$ | 1.94 | $1.43\times10^4$ | 4.16 |

In terms of efficiency, WAlign and REGAL are the fastest and most memory-efficient; IONE's training time exceeds other methods by several orders of magnitude (6,000+ seconds on social networks), making it the least scalable method.

### Key Findings

- **OT methods dominate across the board**: JOENA ranks first in Hits@1 on all 6 domains, with PARROT consistently in second place, validating the superiority of the OT framework for NA tasks.
- **Large variance among embedding methods**: Despite belonging to the same category, REGAL achieves only 0.3% Hits@1 on social networks (near complete failure) yet reaches 45.3% on communication networks, underscoring the critical importance of method–data compatibility.
- **Consistency methods are stable but not competitive**: IsoRank and FINAL exhibit the most consistent performance across domains (low variance), rarely suffering catastrophic failures despite their lower absolute performance.
- **Significant efficiency–effectiveness trade-offs**: WAlign is the fastest but achieves only moderate effectiveness; IONE has above-average effectiveness but unacceptably long training times; OT-based methods strike a reasonable balance between effectiveness and efficiency.
- **High implementation quality in PlanetAlign**: Compared to official implementations, PlanetAlign achieves up to 3× speedup while maintaining comparable effectiveness.

## Highlights & Insights

- **First unified benchmark to include OT-based methods**: This is the first NA benchmark library incorporating optimal transport methods, and the experimental results confirm their comprehensive superiority—a finding of significant value for guiding future research directions.
- **Cross-domain evaluation exposes method bias**: The performance of a single method can vary by more than 50-fold across domains (e.g., REGAL), making single-domain benchmarks prone to misleading conclusions. This highlights the necessity of multi-domain validation when developing new NA methods.
- **Transferable API design principles**: The three-layer abstraction of `BaseData` + `BaseModel` + `Logger` in PlanetAlign serves as an excellent template for constructing domain-specific benchmark libraries, and can be directly transferred to other graph learning tasks such as link prediction and community detection.

## Limitations & Future Work

- **Incomplete method coverage**: GNN-based end-to-end NA methods (e.g., DGMC) and emerging LLM-based methods are not included; continuous updates will be required as the field evolves.
- **Limited dataset scale**: The largest datasets contain tens of thousands of nodes, with no million-scale datasets to thoroughly stress-test scalability.
- **Homogeneous synthetic dataset generation**: All synthetic datasets employ the same 10% insertion + 15% deletion strategy, which may fail to capture more complex network divergence patterns found in real-world scenarios.
- **Lack of dedicated evaluation for unsupervised methods**: While the framework supports varying training ratios, it is not specifically optimized for purely unsupervised settings, making fair evaluation of unsupervised methods less straightforward.

## Related Work & Insights

- **vs. NAB** (Trung et al., 2020): NAB covers effectiveness, scalability, and robustness but is limited to social network datasets and lacks OT-based methods. PlanetAlign comprehensively surpasses NAB in both domain coverage and method coverage.
- **vs. OpenEA** (Sun et al., 2020): OpenEA focuses on embedding-based methods for knowledge graph alignment and serves as an excellent benchmark in that niche. PlanetAlign is broader in scope but may lack OpenEA's depth within the KG domain.
- **vs. PyG/DGL ecosystems**: PlanetAlign can be viewed as a vertical benchmark library within the graph learning ecosystem, specifically targeting NA tasks, complementing general-purpose graph learning frameworks.

## Rating

- **Novelty**: ⭐⭐⭐ As a benchmark library paper, the core contribution lies in engineering integration rather than algorithmic innovation; however, the first-time inclusion of OT-based methods adds meaningful novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 14 methods × 18 datasets × 4 evaluation dimensions, with mean and standard deviation reported over 5 repeated runs—an exceptionally comprehensive experimental scale.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich tables, and intuitive API examples; some sections are slightly verbose.
- **Value**: ⭐⭐⭐⭐ Directly advances the NA research community; the systematic experimental finding (OT methods lead) carries significant reference value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ICLR 2026\] LCA: Local Classifier Alignment for Continual Learning](lca_local_classifier_alignment_for_continual_learning.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](../../ICCV2025/llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)

<!-- RELATED:END -->
