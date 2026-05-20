---
title: >-
  [Paper Note] PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations
description: >-
  [NeurIPS 2025][LLM Evaluation][Power Flow] PFΔ is the first power flow benchmark dataset to simultaneously encompass load, generation dispatch, and topology variations. It comprises 859…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Power Flow"
  - "Benchmark Dataset"
  - "Graph Neural Networks"
  - "Topology Perturbation"
  - "Power Grid Simulation"
date: 2026-05-08
content_hash: f4e8f5b5bf436606
---

# PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations

**Conference**: NeurIPS 2025
**arXiv**: [2510.22048](https://arxiv.org/abs/2510.22048)  
**Code**: [GitHub](https://github.com/MOSSLab-MIT/pfdelta)  
**Area**: Power Systems / Graph Neural Network Benchmarks
**Keywords**: Power Flow, Benchmark Dataset, Graph Neural Networks, Topology Perturbation, Power Grid Simulation

## TL;DR

PFΔ is the first power flow benchmark dataset to simultaneously encompass load, generation dispatch, and topology variations. It comprises 859,800 solved instances across six grid scales, includes close-to-infeasible extreme operating conditions, and introduces a standardized evaluation task suite for systematically assessing ML methods under diverse operating conditions.

## Background & Motivation

Power flow (PF) computation is the cornerstone of real-time grid operation, underpinning virtually every critical decision-making process:
- **N-k security analysis**: Requires solving PF for thousands of contingency scenarios within a 5-minute operational window to assess grid security.
- **Topology optimization**: Searches for optimal topologies in combinatorially explosive action spaces (e.g., a single substation in a 118-bus system can have 65,000 configurations), requiring PF at every step.
- **Extreme weather response**: Climate change drives more frequent equipment failures, demanding larger-scale and faster PF simulations.

Traditional Newton-Raphson (NR) solvers offer high accuracy but at significant computational cost, making real-time large-scale applications challenging. ML methods—particularly GNNs—show promise as fast approximators, yet current research faces two critical bottlenecks:

**Lack of standardized benchmarks**: Different papers employ disparate data generation and evaluation protocols, precluding fair comparison.

**Insufficient variation dimensions in existing datasets**: Most datasets cover only a subset of load variations, failing to capture generation dispatch diversity and topology perturbations, far from the complex operating conditions faced by real-world grids.

Unlike existing datasets (e.g., OPFData, which considers only N-1 topology and load perturbations, or OPF-Learn, which focuses solely on load diversity), PFΔ is the first benchmark to integrate perturbations across load, generation, and topology dimensions within a single framework, while also introducing close-to-infeasible extreme operating conditions.

## Method

### Overall Architecture

PFΔ's core contributions consist of three components: (1) a multi-dimensional perturbation data generation pipeline; (2) a dataset construction procedure incorporating extreme operating conditions; and (3) a standardized evaluation task suite spanning four scenario categories.

### Key Designs

1. **Three-dimensional perturbation data generation**: Each data sample is subjected to three simultaneous perturbations:

    - **Load perturbation**: Adopts the convex-set sampling method from OPF-Learn, uniformly sampling load configurations from a convex set that contains the ACOPF feasible region. When an infeasible point is sampled, the convex set is contracted. This approach yields richer and more diverse load distributions compared to simple ±20% uniform sampling.
    - **Topology perturbation**: Simulates N-1 and N-2 contingency scenarios by equiprobably executing one of the following operations: removing up to 2 generators, removing up to 2 lines, removing one of each, or retaining the original topology.
    - **Generation dispatch perturbation**: Diverse active power and voltage setpoints are produced by randomly permuting generator cost parameters, solved via a modified ACOPF to obtain varied dispatch configurations.

   For each perturbed input, a modified ACOPF (with output variable constraints removed to render it equivalent to PF equations) is solved using PowerModels.jl with Ipopt; only converged feasible solutions are retained.

2. **Close-to-Infeasible (C2I) sample generation**: The loadability boundary near the steady-state voltage stability limit is identified by progressively increasing power injection/withdrawal and repeatedly solving PF. At this boundary, the PF Jacobian becomes singular and traditional solvers frequently fail to converge. "Approaching-infeasible" samples are also generated as data augmentation. These extreme operating conditions are critical for testing ML model robustness.

3. **Standardized evaluation suite**: Twelve evaluation tasks organized into four groups:

    - **Group 1 (In-distribution generalization)**: Training on different topology combinations (N / N+N-1 / N+N-1+N-2) and testing generalization to unseen topologies.
    - **Group 2 (Data efficiency)**: Progressively reducing training samples from 54,000 to 18,000.
    - **Group 3 (Out-of-distribution generalization)**: Training on one grid scale and testing on different scales (e.g., train on 118-bus, test on 57/500-bus).
    - **Group 4 (Extreme conditions)**: Incorporating C2I samples during training and evaluating adaptation to extreme scenarios.

   The primary evaluation metric is the unsupervised power balance mismatch $|\Delta S_i| = \sqrt{(\Delta p_i)^2 + (\Delta q_i)^2}$, which avoids solver-induced bias arising from the existence of multiple solutions.

### Loss & Training

The three evaluated GNN models employ distinct training strategies:
- **CANOS-PF** (adapted from an ACOPF model): Interaction network with L2 loss and constraint violation penalty; branch power flows are computed analytically.
- **PowerFlowNet**: Bus-type embeddings with TAGConv message passing and L2 supervised loss.
- **GNS-S** (self-supervised): Directly minimizes power balance equations via iterative voltage updates, requiring no labels.

## Key Experimental Results

### Main Results

Experiments are conducted primarily on the IEEE 118-bus system, with models trained on Task 1.3 (N+N-1+N-2 topologies) and evaluated across multiple scenarios.

| Model | Feasible Mean PBL | Feasible Max PBL | C2I Mean PBL | Runtime vs. NR |
|------|------------------|-----------------|-------------|-------------|
| Newton-Raphson | ~$10^{-6}$ | ~$10^{-5}$ | Frequently fails | 1× (baseline) |
| CANOS-PF | Lowest (overall best) | Lowest | Moderate | ~5× speedup |
| PowerFlowNet | Moderate | Moderate | Worst | ~5× speedup |
| GNS-S | High variance | High variance | **Lowest** | ~5× speedup |

Out-of-distribution generalization (train on 118-bus, test on 57/500-bus):

| Test Scale | CANOS-PF | PowerFlowNet | GNS-S | NR Convergence Rate |
|---------|---------|-------------|-------|----------|
| 57-bus | Moderate error | Large error | Moderate error | 95.2% |
| 118-bus | Best | Moderate | High variance | 65.7% |
| 500-bus | Large error (all) | Large error (all) | Large error (all) | 43.4% |

### Ablation Study

Effect of training topology on generalization (Task 1.1 vs. 1.2 vs. 1.3):

| Training Topology | N-1 Test Performance | N-2 Test Performance | Key Finding |
|---------|------------|------------|---------|
| N only | All models degrade sharply on N-1/N-2 | Worse | CANOS-PF degrades most severely (analytic formulas amplify errors) |
| N + N-1 | Significant improvement | Also improves | N-1 training data is critical for generalization |
| N + N-1 + N-2 | Best | Best | Topology diversity matters more than data volume |

Data efficiency experiments (Task 2.1–2.3, 18k/36k/54k samples):

| Model | Low-Data Performance | High-Data Performance | Note |
|------|----------|----------|------|
| CANOS-PF | Largely stable | Largely stable | Minimal sensitivity to data volume |
| PowerFlowNet | Largely stable | Largely stable | Similar |
| GNS-S | Degrades to N-only training level | Normal | More sensitive to topology diversity under low-data regimes |

### Key Findings

- **No GNN model approaches the $10^{-6}$-level accuracy of NR solvers**, even those incorporating physics-informed components.
- Topology diversity has a substantially greater effect on model generalization than data volume.
- Cross-scale generalization remains a major challenge: all models exhibit significant performance degradation on grids of different sizes than those used for training.
- GNS-S's self-supervised physical loss enables the best performance under C2I extreme conditions.
- NR solver convergence drops sharply on large-scale grids (completely failing on 2000-bus systems), a gap that GNNs may fill.
- GNN methods achieve approximately 5× inference speedup.

## Highlights & Insights

- **Comprehensiveness**: The first benchmark to simultaneously address load, generation dispatch, and topology dimensions, more faithfully reflecting real-world grid operations.
- **C2I design**: The introduction of extreme operating conditions is a key highlight—these represent a recognized weakness of traditional solvers and a potential strength of ML approaches.
- **Unsupervised metric**: Using power balance mismatch as the evaluation metric circumvents bias introduced by the presence of multiple solutions.
- The standardized task framework enables fair comparison across future methods.

## Limitations & Future Work

- The GOC-2000 scale dataset contains only half the samples of other scales; data generation for large-scale grids remains constrained by computational cost.
- The ability of models to identify infeasible operating conditions (i.e., detecting when PF has no solution) is not evaluated.
- N-k contingencies with $k > 2$ are not covered.
- The load sampling method may not accurately represent the complete feasible space for large-scale networks.
- All evaluated models remain far from the accuracy required for practical deployment.

## Related Work & Insights

- PFΔ is complementary to OPFData and OPF-Learn: it focuses on PF rather than ACOPF and covers a richer set of variation dimensions.
- PowerGraph supports only the 118-bus system and lacks topology perturbations; PFΔ substantially surpasses it in both scale and diversity.
- Physics-informed components (e.g., GNS-S's power balance loss) are critical for extreme operating conditions, suggesting that future architectures should integrate physical constraints more deeply.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A benchmark dataset paper; key innovations lie in C2I generation and multi-dimensional perturbation design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Twelve standardized tasks, three GNN baselines plus NR comparison; analysis is highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough power systems background, accessible to readers outside the domain.
- **Value**: ⭐⭐⭐⭐ Fills a critical gap in ML benchmarks for power flow, providing important impetus for the power systems AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](../../CVPR2026/llm_evaluation/tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](../../CVPR2026/llm_evaluation/pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)

</div>

<!-- RELATED:END -->
