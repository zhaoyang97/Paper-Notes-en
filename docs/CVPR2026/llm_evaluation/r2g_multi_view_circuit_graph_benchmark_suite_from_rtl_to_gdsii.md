---
title: >-
  [Paper Note] R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII
description: >-
  [CVPR 2026][LLM Evaluation][circuit graph] This paper introduces R2G, the first standardized multi-view circuit graph benchmark suite…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "circuit graph"
  - "GNN benchmark"
  - "multi-view"
  - "physical design"
  - "EDA"
date: 2026-05-08
content_hash: d352ce3f222fd464
---

# R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII

**Conference**: CVPR 2026
**arXiv**: [2604.08810](https://arxiv.org/abs/2604.08810)  
**Code**: [https://github.com/ShenShan123/R2G](https://github.com/ShenShan123/R2G)  
**Area**: AI for EDA / Graph Neural Network Benchmarks
**Keywords**: circuit graph, GNN benchmark, multi-view, physical design, EDA

## TL;DR

This paper introduces R2G, the first standardized multi-view circuit graph benchmark suite, providing five stage-aware graph representations with information equivalence across 30 IP cores. A systematic study reveals that graph representation choice has a greater impact on performance than GNN model choice.

## Background & Motivation

Graph neural networks are increasingly applied to physical design tasks such as congestion prediction and wirelength estimation, yet progress is hindered by inconsistent circuit representations and evaluation protocols that lack controlled variables. Existing EDA datasets couple graph representations with task labels, making it impossible to determine whether model accuracy stems from architectural advantages or representation choices.

The core contribution of R2G is to decouple representation selection from model selection—by fixing the circuit and task while varying only the graph view, it isolates the effect of representation, establishing the first controlled-variable circuit graph benchmark.

## Method

### Overall Architecture

Five standardized graph views are extracted from DEF files. Each view encodes the same attribute set but differs in where features are attached (information equivalence). The benchmark covers three physical design stages: synthesis, placement, and routing.

### Key Designs

1. **Five Complementary Views**: Including node-centric views (features on nodes) and edge-centric views (features on edges), each view maintains the same attribute set while differing only in representational structure. This information equivalence is the critical prerequisite for controlled-variable experiments.

2. **End-to-End DEF-to-Graph Pipeline**: Graph structures, features, and labels are extracted directly from standard DEF design files, with unified splits, domain metrics, and reproducible baselines. The 30 open-source IP cores span scales from ~500 to >$10^6$ nodes/edges, covering diverse categories including audio controllers (ss_pcm, ac97_ctrl), cryptographic cores (des3_area, SHA256, AES), and video controllers (vga_lcd), across synthesis, placement, and routing stages.

3. **Systematic Cross-View Study**: Three representative GNNs—GINE, GAT, and ResGatedGCN—are systematically evaluated across all five views to isolate representation effects.

### Loss & Training

Standard regression losses are used for node-level placement tasks (HPWL prediction) and edge-level routing tasks (wirelength prediction). Unified train/validation/test splits ensure reproducibility.

## Key Experimental Results

### Key Findings

| Finding | Evidence | Explanation |
|---------|----------|-------------|
| View > Model | Test R² varies >0.3 across views | View selection dominates performance under a fixed GNN |
| Model ranking reversal | Best model differs across views | Severe representation–model coupling |
| Node-centric view most robust | View (b) achieves best cross-stage results | Performs best on both placement and routing |
| Decoder head depth is critical | 3–4 layer head: R² from −0.17 to 0.99 | Far exceeds the impact of GNN depth |

### Key Findings

- Graph representation choice matters far more than GNN architecture choice.
- Decoder head depth (3–4 layers) is the primary driver of accuracy.
- Node-centric views generalize best across both placement and routing stages.
- The five views maintain information equivalence (same attribute set, differing only in feature attachment location), which is the key prerequisite for controlled-variable experiments.
- Increasing head depth from 1 to 4 layers raises R² from −0.17 to 0.99 on placement tasks, and causes routing tasks to converge from a non-converging state.
- The optimal GNN model varies across views, indicating severe representation–model coupling.

## Highlights & Insights

- First work to treat graph representation as an independent variable in controlled experiments.
- The finding that "view selection dominates model selection" carries significant guidance for the EDA-ML community.
- The surprising importance of decoder head depth may reshape GNN architectural design thinking.
- The information equivalence design is the foundation for rigorous ablation studies.

## Limitations & Future Work

- Only 30 IP cores, limiting diversity.
- The five views do not exhaust all possible circuit representations.
- The work focuses primarily on back-end physical design; front-end logic design is not addressed.
- Heterogeneous GNNs (e.g., distinguishing cell and net node types) are not explored in the multi-view setting.
- Dataset scale ranges from ~500 to >$10^6$ nodes/edges, spanning a wide range but with limited samples per scale segment.
- R2G inherits best practices from graph ML benchmarks such as OGB: unified splits, scalable loaders, and reproducible baselines.
- Existing EDA datasets couple graph representations with task labels; R2G decouples them to enable the first controlled-variable experiments.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First controlled-variable multi-view circuit graph benchmark.
- **Technical Depth**: ⭐⭐⭐⭐ — Information equivalence design is rigorous, ensuring experimental controllability.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic cross-view and cross-model evaluation.
- **Value**: ⭐⭐⭐⭐ — Provides a standardized tool for EDA-ML research with open-source code and dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](../../ACL2026/llm_evaluation/diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/llm_evaluation/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/llm_evaluation/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[ICLR 2026\] Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](../../ICLR2026/llm_evaluation/can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)

</div>

<!-- RELATED:END -->
