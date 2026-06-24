---
title: >-
  [Paper Note] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment
description: >-
  [ICLR 2026][Network Alignment] PlanetAlign is proposed as a PyTorch-based network alignment benchmark library that integrates 18 datasets across 6 domains, 14 methods covering three major categories (consistency, embedding, and Optimal Transport), and standardized evaluation workflows. Through large-scale systematic experiments, it reveals the comprehensive lead of OT-based methods (PARROT/JOENA) in effectiveness and the differentiated performance of various methods in scalab…
tags:
  - "ICLR 2026"
  - "Network Alignment"
  - "Benchmark Library"
  - "Graph Matching"
  - "Optimal Transport"
  - "Evaluation Framework"
date: 2026-05-08
content_hash: 6c09dd243cb1f9e9
---

# PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment

**Conference**: ICLR 2026  
**arXiv**: [2505.21366](https://arxiv.org/abs/2505.21366)  
**Code**: [GitHub](https://github.com/yq-leo/PlanetAlign)  
**Area**: LLM Evaluation  
**Keywords**: Network Alignment, Benchmark Library, Graph Matching, Optimal Transport, Evaluation Framework

## TL;DR

PlanetAlign is proposed as a PyTorch-based network alignment benchmark library that integrates 18 datasets across 6 domains, 14 methods covering three major categories (consistency, embedding, and Optimal Transport), and standardized evaluation workflows. Through large-scale systematic experiments, it reveals the comprehensive lead of OT-based methods (PARROT/JOENA) in effectiveness and the differentiated performance of various methods in scalability and robustness.

## Background & Motivation

**Background**: Network Alignment (NA) aims to discover node correspondences across different networks. it is a critical foundation for downstream tasks such as cross-social network user matching, protein homology discovery, knowledge graph fusion, and anti-fraud detection. The field has developed three main categories of methods: consistency-based (e.g., IsoRank, FINAL), embedding-based (e.g., REGAL, BRIGHT), and Optimal Transport (OT)-based (e.g., PARROT, JOENA). However, a systematic comparison between these methods has been missing.

**Limitations of Prior Work**: Existing NA benchmarks or libraries (SGAPBSA, CAPABN, ASNets, NAB, OpenEA) have significant limitations: (1) Datasets are restricted to single domains—SGAPBSA and CAPABN only feature biological networks, ASNets focuses on social networks, and OpenEA targets knowledge graphs; (2) Incomplete method coverage—none of the existing libraries include the latest and best-performing OT-based methods; (3) Single evaluation dimension—most only evaluate effectiveness, ignoring scalability and robustness, while inconsistent dataset partitioning leads to non-reproducible results.

**Key Challenge**: While methods in the NA field are increasingly diverse, the evaluation infrastructure lags behind. Researchers report results using different datasets, splits, and metrics, making it impossible to fairly compare the true performance differences between methods, which hinders research progress.

**Goal**: To build a comprehensive, unified, and easy-to-use NA benchmark library that covers multi-domain datasets, multi-category methods, and multi-dimensional evaluation.

**Key Insight**: Drawing from successful benchmark library designs in CV/NLP (e.g., MMDetection, HuggingFace), this work addresses the evaluation fragmentation in NA through unified API design, standardized data partitioning, and reproducible evaluation workflows.

**Core Idea**: By constructing PlanetAlign, a unified benchmark library covering 6 domains and 3 major method categories, systematic and fair comparisons of NA methods across four dimensions—effectiveness, scalability, robustness, and supervision sensitivity—are achieved for the first time.

## Method

### Overall Architecture

Methods in the Network Alignment (NA) field are increasing, but researchers use disparate datasets, partitions, and metrics, preventing fair comparisons. PlanetAlign addresses this evaluation fragmentation by unifying data, methods, and evaluation into a single PyTorch-based library, allowing users to complete the full "load data → train algorithm → multi-dimensional evaluation" workflow with a few lines of code.

The overall architecture corresponds to three key designs linked in three layers: (1) **Data Layer**—18 datasets covering 6 domains: social, publication, biology, knowledge graph, infrastructure, and communication (Design 1); (2) **Algorithm Layer**—14 NA methods unified within classes inheriting from `BaseModel`, sharing `.train()` / `.test()` interfaces (Design 2); (3) **Evaluation Layer**—standardized Hits@K / MRR metrics, along with time/memory overhead tracking and robustness testing tools (Design 3). Data flows unidirectionally through the layers: two networks to be aligned and anchor node pairs are retrieved from the Data Layer, fed into the Algorithm Layer to compute the alignment matrix $\mathbf{S}$, and then passed to the Evaluation Layer for scoring across effectiveness, scalability, robustness, and supervision sensitivity.

### Key Designs

**1. Comprehensive Dataset Collection and Synthesis: Making Cross-Domain Generalization Measurable**

Previous libraries covered at most 1–2 domains, making it impossible to compare the actual performance of methods under different network structures. PlanetAlign collects 18 datasets, including 11 from the real world and 7 synthesized, spanning Social (4, e.g., Foursquare-Twitter, Douban), Publication (3, e.g., ACM-DBLP, Cora, ArXiv), Biology (3, e.g., SacchCere, PPI, GGI), Knowledge Graph (3 variants of DBP15K: ZH-EN/JA-EN/FR-EN), Infrastructure (3, e.g., Italy, Airport, PeMS08), and Communication (2, e.g., Phone-Email, Arenas). Synthetic datasets are generated using a classic network perturbation strategy: inserting 10% noise edges into the original network and deleting 15% of existing edges to obtain two networks with a known permutation (ground-truth alignment). Consequently, the generalization ability of methods on networks with vastly different structural characteristics can be measured horizontally for the first time.

**2. Unified Implementation of 14 Methods Across Three Categories: Including OT Methods for the First Time**

NA methods are increasingly diverse, yet no existing library covers all three major categories, notably missing the latest and strongest OT methods. PlanetAlign rewrites 14 methods, including consistency-based (IsoRank, FINAL), embedding-based (IONE, REGAL, CrossMNA, NetTrans, WAlign, BRIGHT, NeXtAlign, WLAlign), and OT-based (PARROT, SLOTAlign, HOT, JOENA), into unified PyTorch implementations. They inherit from the same `BaseModel` base class and share `.train()` / `.test()` interfaces. The library includes built-in utility functions for common tasks like Random Walk with Restart (RWR) embeddings and anchor node embeddings, allowing new methods to be integrated with minimal code. Compared to original official implementations, these rewritten versions achieve up to 3x acceleration while maintaining similar effectiveness—filling the largest gap in existing libraries (OT methods) while ensuring comparisons occur on a consistent, optimized codebase.

**3. Multi-dimensional Standardized Evaluation Tools: Expanding from Hits@1 to Four Dimensions**

Previous library evaluations were one-dimensional, usually only reporting Hits@1, which fails to characterize whether a method is fast or stable enough for practical deployment. PlanetAlign expands evaluation into four dimensions. Effectiveness uses Hits@K and MRR, averaged over bidirectional alignments ($\mathcal{G}_1 \to \mathcal{G}_2$ and its reverse) to avoid directional bias; Scalability is tracked via a built-in Logger for runtime and peak memory; Robustness is tested by injecting different types and levels of edge/attribute/supervision noise via utility functions to observe degradation; Supervision Sensitivity is evaluated by varying training ratios to see dependency on labels. The entire workflow uses unified random seeds and data splits to ensure results are reproducible and fairly comparable.

## Key Experimental Results

### Main Results: Effectiveness and Efficiency

14 methods were evaluated across 6 domain datasets (20% training ratio), reporting average Hits@1, Hits@10, and MRR (%):

| Method | Category | Social H@1 | Publication H@1 | Bio H@1 | KG H@1 | Infra H@1 | Comm H@1 |
|------|------|---------|-----------|---------|------------|------------|---------|
| JOENA | OT | **18.7** | **73.2** | **63.7** | **66.3** | **62.9** | **66.3** |
| PARROT | OT | 12.6 | 66.6 | 61.6 | 66.0 | 51.8 | 63.3 |
| NetTrans | Embedding | 7.2 | 40.7 | 34.2 | 28.8 | 29.3 | 45.2 |
| BRIGHT | Embedding | 5.1 | 40.4 | 30.5 | 30.4 | 29.9 | 50.9 |
| NeXtAlign | Embedding | 7.1 | 43.2 | 25.9 | 27.5 | 28.0 | 29.6 |
| FINAL | Consistency | 4.9 | 22.3 | 22.9 | 13.9 | 15.1 | 21.7 |
| IsoRank | Consistency | 4.2 | 18.9 | 21.6 | 11.5 | 14.2 | 22.1 |
| REGAL | Embedding | 0.3 | 1.8 | 1.0 | 0.8 | 2.8 | 45.3 |

OT-based methods (JOENA, PARROT) achieved the best Hits@1 across all 6 domains, with the lead being particularly significant in Knowledge Graphs and Infrastructure networks.

### Ablation Study: Efficiency and Scalability

| Method | Category | Social Time(s) | Social Mem(GB) | Pub Time(s) | Pub Mem(GB) |
|------|------|-----------|-------------|-------------|---------------|
| WAlign | Embedding | **0.61** | 2.65 | 9.41 | 9.88 |
| REGAL | Embedding | 9.38 | **1.16** | 16.14 | **3.18** |
| FINAL | Consistency | 5.91 | 5.39 | 6.75 | 10.06 |
| PARROT | OT | — | — | — | — |
| JOENA | OT | — | — | — | — |
| IONE | Embedding | $6.34\times10^3$ | 1.94 | $1.43\times10^4$ | 4.16 |

Regarding efficiency, WAlign and REGAL are the fastest with the smallest memory footprints; IONE's training time is several orders of magnitude higher than other methods (over 6000 seconds on social networks), making it the least scalable method.

### Key Findings

- **OT Methods Lead Comprehensively**: JOENA ranked first in Hits@1 across all 6 domains, with PARROT consistently second, validating the superiority of the OT framework for NA tasks.
- **Large Variance in Embedding Methods**: Despite being the same category, REGAL's Hits@1 was only 0.3% on social networks (nearly failing) but reached 45.3% on communication networks, highlighting the critical importance of method-data alignment.
- **Consistency Methods are Stable but Not Dominant**: IsoRank and FINAL showed the highest consistency (small variance) across all domains; though absolute performance was lower, they rarely suffered catastrophic failures.
- **Significant Efficiency-Effectiveness Trade-off**: WAlign is the fastest but has moderate effectiveness, while IONE has upper-middle effectiveness but unacceptable training time; OT methods strike a good balance between efficacy and efficiency.
- **High Implementation Quality of PlanetAlign**: Compared to official implementations, PlanetAlign's versions maintain similar effectiveness while achieving up to 3x speedup.

## Highlights & Insights

- **First Unified Benchmark Including OT Methods**: This is the first benchmark library in the NA field to include Optimal Transport methods. Experimental results confirm the overall lead of OT methods, a finding that serves as a valuable guide for future research.
- **Cross-Domain Evaluation Reveals Method Bias**: The performance of the same method can vary by over 50x across different domains (e.g., REGAL). Single-domain benchmarks are prone to misleading conclusions, suggesting that new NA methods must be validated across multiple domains.
- **Transferable API Design Philosophy**: The three-layer abstract design of `BaseData` + `BaseModel` + `Logger` in PlanetAlign is an excellent paradigm for building domain benchmarks and can be directly migrated to other graph learning tasks such as link prediction or community detection.

## Limitations & Future Work

- **Method Coverage Still Gaps**: End-to-end GNN-based NA methods (e.g., DGMC) and emerging LLM-based methods are not included and require continuous updates as the field evolves.
- **Limited Dataset Scale**: The maximum node count is around samples of ten thousand; there is a lack of large-scale datasets with millions of nodes to fully test scalability.
- **Single Synthetic Method**: All synthetic datasets use the same 10% insertion + 15% deletion strategy, which might not cover more complex network difference patterns in real-world scenarios.
- **Missing Fair Evaluation for Unsupervised Methods**: Some methods are unsupervised while others are semi-supervised. Although the current framework supports different training ratios, it is not specifically optimized for purely unsupervised scenarios.

## Related Work & Insights

- **vs NAB** (Trung et al., 2020): NAB covers effectiveness, scalability, and robustness but only includes social network datasets and lacks OT methods. PlanetAlign comprehensively surpasses it in domain and method coverage.
- **vs OpenEA** (Sun et al., 2020): OpenEA focuses on embedding methods for knowledge graphs and is an excellent benchmark for KG alignment. PlanetAlign is broader but may lack the depth of OpenEA specifically in the KG domain.
- **vs PyG/DGL Ecosystem**: PlanetAlign can be viewed as a vertical benchmark library specifically for NA tasks within the graph learning ecosystem, complementing general graph learning frameworks.

## Rating

- Novelty: ⭐⭐⭐ As a benchmark library paper, the core contribution lies in engineering integration rather than algorithmic innovation, though including OT methods for the first time provides some novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 methods × 18 datasets × 4 evaluation dimensions, with means and standard deviations from 5 runs; the experimental scale is remarkably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich tables, and intuitive API examples, though some content is slightly verbose.
- Value: ⭐⭐⭐⭐ It directly promotes the NA research community, and its systematic experimental conclusions (OT lead) provide significant reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](../../ACL2025/others/popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)
- [\[ICLR 2026\] Fractional-Order Spiking Neural Network](fractional-order_spiking_neural_network.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](../../ACL2025/others/ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ICLR 2026\] Scaling Direct Feedback Learning with Jacobian Alignment Guarantees](scaling_direct_feedback_learning_with_jacobian_alignment_guarantees.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)

</div>

<!-- RELATED:END -->
