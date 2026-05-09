---
title: >-
  [Paper Note] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights
description: >-
  [NeurIPS 2025][Image Restoration][Graph Condensation] This paper proposes GC4NC—the first systematic benchmark framework for graph condensation (GC)—which evaluates multiple GC methods across 8 dimensions (performance / efficiency / privacy protection / denoising / NAS effectiveness / transferability, etc.), finding that trajectory matching methods achieve the best performance, structure-free methods are most efficient, and graph condensation significantly outperforms image condensation under 1000× compression.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Graph Condensation
  - Dataset Distillation
  - Node Classification
  - Benchmark Evaluation
  - Privacy Protection
date: 2026-05-08
content_hash: a3e11eb62c053955
---

# GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights

**Conference**: NeurIPS 2025
**arXiv**: [2406.16715](https://arxiv.org/abs/2406.16715)
**Code**: [https://github.com/Emory-Melody/GraphSlim](https://github.com/Emory-Melody/GraphSlim)
**Area**: Image Restoration
**Keywords**: Graph Condensation, Dataset Distillation, Node Classification, Benchmark Evaluation, Privacy Protection

## TL;DR
This paper proposes GC4NC—the first systematic benchmark framework for graph condensation (GC)—which evaluates multiple GC methods across 8 dimensions (performance / efficiency / privacy protection / denoising / NAS effectiveness / transferability, etc.), finding that trajectory matching methods achieve the best performance, structure-free methods are most efficient, and graph condensation significantly outperforms image condensation under 1000× compression.

## Background & Motivation

**State of the Field**: Graph condensation compresses large-scale graphs into extremely small synthetic graphs (e.g., with 1000× node reduction) while preserving GNN training performance. It is a data compression technique inspired by dataset distillation in the image domain.

**Limitations of Prior Work**:
- Lack of unified evaluation protocols: different methods use different validation models, some select models on the test set, and validation frequencies are inconsistent.
- Incomplete evaluation dimensions: most methods only assess performance and transferability, neglecting privacy protection and denoising capability.
- Unclear impact of design choices: the effects of condensation objectives (gradient matching / distribution matching / trajectory matching), initialization strategies, and whether to generate graph structure have not been systematically compared.

**Root Cause**: Graph condensation methods are developing rapidly but lack a fair comparison benchmark, making it impossible to reliably determine which design choices truly matter.

**Paper Goals**: Establish a unified evaluation benchmark, systematically compare existing methods, and reveal the impact of key design choices.

**Starting Point**: Constructing a multi-dimensional evaluation framework + standardized experimental protocol + open-source codebase.

**Core Idea**: 8-dimensional unified evaluation + in-depth design choice analysis = the first comprehensive benchmark for graph condensation.

## Method

### Overall Architecture
The GC4NC evaluation pipeline: select a graph condensation method → run on standard datasets → use a fair validation protocol to select the best condensed graph → score across 8 evaluation dimensions → perform cross-method comparison and analyze the impact of design choices.

### Key Designs

1. **Fair Evaluation Protocol**:

    - Function: Eliminate unfair factors in the evaluation of existing methods.
    - Mechanism: Uniformly use the validation set (rather than the test set) to select the best condensed graph; unify validation models (cross-evaluation with MLP + GCN + GAT); control intermediate validation frequency to prevent excessive selection.
    - Design Motivation: Different papers use different protocols, making results incomparable; standardization is a prerequisite for a benchmark.

2. **8-Dimensional Evaluation System**:

    - Function: Comprehensively measure the quality of condensation methods.
    - Mechanism: (a) Performance—node classification accuracy; (b) Efficiency—condensation time and memory; (c) Privacy Protection—degree to which the condensed graph leaks information about the original graph; (d) Denoising—robustness to noisy features/labels; (e) NAS Effectiveness—performance on the original graph of architectures searched on the condensed graph; (f) Transferability—using a graph condensed for one GNN with another GNN; (g) Scalability—performance on large graphs; (h) Structure Preservation—degree of topological property retention.
    - Design Motivation: Accuracy alone is far from sufficient—practical deployment also requires consideration of privacy (the condensed graph must not allow reconstruction of the original data), efficiency, and transferability.

3. **Design Choice Analysis**:

    - Function: Systematically compare key design decisions in the condensation process.
    - Mechanism: Experimental comparison of—(i) condensation objectives: gradient matching vs. distribution matching vs. trajectory matching; (ii) initialization: random vs. clustering vs. real node selection; (iii) structure-based vs. structure-free methods; (iv) graph property preservation strategies.
    - Design Motivation: Understanding "which choices matter most for which dimensions" guides the design of future methods.

### Loss & Training
- This is not a methodology paper—it is an evaluation framework.
- The open-source codebase supports one-click reproduction of all experiments.

## Key Experimental Results

### Main Results
Node classification accuracy of various methods on standard datasets (1% condensation ratio):

| Method Type | Representative Method | Cora | Citeseer | Ogbn-arxiv | Avg. Rank |
|---|---|---|---|---|---|
| Gradient Matching | GCond | Medium | Medium | Medium | 3 |
| Distribution Matching | GCDM | Medium | Lower | Lower | 4 |
| Trajectory Matching | SFGC | **Highest** | **Highest** | **Highest** | **1** |
| Structure-Free | DosCond | Lower | Lower | Medium | 5 |

### Ablation Study: Impact of Design Choices

| Design Choice | Effect on Performance | Effect on Efficiency | Key Finding |
|---|---|---|---|
| Condensation Objective | Trajectory matching is best | Trajectory matching is slowest | Performance–efficiency trade-off |
| Graph Structure Generation | Improves performance | Increases computation | Structure-generating methods perform better at high compression ratios |
| Initialization Strategy | Has impact | Minor impact | Clustering initialization generally outperforms random |
| Graph Property Preservation | Helpful | Low overhead | Degree distribution preservation is most important |

### Key Findings
- **Trajectory matching methods are best but slowest**: SFGC leads in performance, but its condensation time is several times that of gradient matching.
- **Structure-free methods are highly efficient**: Methods such as DosCond bypass graph structure learning, achieving high speed at the cost of noticeable performance degradation.
- **Graph condensation outperforms image condensation at high compression ratios**: At equivalent compression rates, graph data retains more usable information than image data.
- **Privacy protection: condensation does not imply anonymization**: The condensed graphs of certain methods still leak sensitive information from the original graph.
- **NAS effectiveness: condensed graphs are highly useful for NAS**—architectures searched on condensed graphs also perform well on the original graph, substantially accelerating NAS.
- **Feature noise has a greater impact than label noise**: Most methods exhibit weak denoising capability against feature noise.

## Highlights & Insights
- The **first comprehensive benchmark for graph condensation** fills a gap in the field; the 8-dimensional evaluation system is well-designed.
- The finding that **"graph condensation outperforms image condensation"** suggests that graph structure confers unique advantages in data compression.
- The introduction of the **privacy dimension** is forward-looking—as data protection regulations strengthen, whether condensed graphs leak private information is a critical consideration for practical deployment.

## Limitations & Future Work
- Only node classification is addressed; link prediction and graph classification are not covered.
- Experiments on some large-scale datasets are insufficiently comprehensive due to computational resource constraints.
- The attack models used for privacy evaluation are relatively simple; stronger privacy attacks could be considered.
- The fairness dimension is not considered (i.e., whether condensation preserves representation of minority classes).

## Related Work & Insights
- **vs. DC-Bench (image condensation benchmark)**: GC4NC is the graph-domain counterpart, but with a broader set of evaluation dimensions.
- **vs. individual method papers**: GC4NC unifies their evaluation conditions, eliminating unfair comparisons.
- Provides standardized objectives and references for the development of graph condensation methods.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive multi-dimensional graph condensation benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 dimensions, multiple methods, multiple datasets, design choice analysis
- Writing Quality: ⭐⭐⭐⭐ Clear organization, findings well summarized
- Value: ⭐⭐⭐⭐⭐ Foundational contribution to the graph condensation field

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](../../ICCV2025/image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

<!-- RELATED:END -->
