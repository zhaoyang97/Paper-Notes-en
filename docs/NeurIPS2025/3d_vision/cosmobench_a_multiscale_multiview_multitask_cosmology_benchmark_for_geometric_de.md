---
title: >-
  [Paper Note] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning
description: >-
  [NeurIPS 2025][3D Vision][geometric deep learning] This paper introduces CosmoBench—the largest cosmological geometric deep learning benchmark to date—comprising 34,752 point clouds and 24…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "geometric deep learning"
  - "cosmology benchmark"
  - "point cloud"
  - "graph neural network"
  - "merger tree"
date: 2026-05-08
content_hash: 5d77dd7f40ffe5e4
---

# CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

**Conference**: NeurIPS 2025
**arXiv**: [2507.03707](https://arxiv.org/abs/2507.03707)
**Code**: [GitHub](https://github.com/TeresaHuang/CosmoBench)
**Area**: 3D Vision
**Keywords**: geometric deep learning, cosmology benchmark, point cloud, graph neural network, merger tree

## TL;DR

This paper introduces CosmoBench—the largest cosmological geometric deep learning benchmark to date—comprising 34,752 point clouds and 24,996 directed trees across multiple scales, viewpoints, and tasks. A key finding is that simple linear models sometimes outperform large GNNs.

## Background & Motivation

**Rich cosmological simulation data but no unified benchmark**: Cosmological simulations produce vast quantities of point cloud and merger tree data, yet the community lacks a unified benchmark analogous to ShapeNet/ModelNet for systematically evaluating machine learning methods, making fair cross-method comparison difficult.

**Geometric deep learning breakthroughs have yet to arrive in cosmology**: GDL has achieved notable advances in computer vision, structural biology, and climate science, but no analogous breakthrough has been observed in cosmology; large-scale benchmarks are needed to catalyze methodological innovation.

**Insufficient scale and diversity in existing benchmarks**: The benchmark proposed by Balla et al. contains only 3,560 Quijote point clouds and is limited in data scale, physical scale coverage, data modalities, and task diversity.

**The simulation-to-observation gap must be bridged**: Tasks such as inferring cosmological parameters, predicting galaxy velocities, and reconstructing merger trees carry direct scientific value (e.g., inferring the cosmic expansion rate and compensating for hardware limitations), yet existing methods remain immature.

**The trade-off between simple and complex models is underexplored**: The community tends toward increasingly complex deep learning models, but physically motivated simple methods may be equally or more effective on cosmological tasks; systematic comparison is needed.

**Scale-dependent behavioral differences remain unclear**: Cosmological physics exhibits markedly different behavior in the linear (large-scale) and nonlinear (small-scale) regimes, and the performance patterns of different methods across these regimes are not yet well understood.

## Method

### Overall Architecture

CosmoBench curates data from three major cosmological simulation suites—Quijote, CAMELS-SAM, and CAMELS—to construct a multiscale, multiview benchmark comprising **point cloud datasets** (3 datasets, 34,752 point clouds in total) and **directed tree datasets** (CS-Trees, 24,996 trees). Four task categories are covered: cosmological parameter inference from point clouds (graph-level regression), galaxy/halo velocity prediction from positions (node regression), cosmological parameter inference from merger trees (graph-level regression), and fine-grained merger tree reconstruction (node classification / graph super-resolution). Physical baselines, linear model baselines, and deep learning baselines are provided for each task category.

### Key Design 1: Multiscale Point Cloud Dataset Construction

- **Function**: Extracts point cloud datasets at different spatial scales from three simulation suites.
- **Mechanism**: Quijote (1000 cMpc/h, large-scale linear regime) contains 32,752 dark matter halo point clouds; CAMELS-SAM (100 cMpc/h, intermediate nonlinear scale) contains 1,000 galaxy point clouds with associated merger trees; CAMELS (25 cMpc/h, deeply nonlinear scale) contains 1,000 hydrodynamical simulation galaxy point clouds. Each point cloud includes 3D positions and velocities, annotated with the corresponding cosmological parameters $(\Omega_m, \sigma_8)$.
- **Design Motivation**: Covering the full physical scale spectrum from linear to deeply nonlinear regimes enables researchers to systematically analyze how different methods behave under different physical conditions.

### Key Design 2: Invariant-Feature-Based GNN Message Passing

- **Function**: Designs graph neural networks that maintain E(3) invariance while supporting higher-order message passing.
- **Mechanism**: After constructing a radius graph, edge features use the normalized distance $d_{ij}/R_c$ and two dot-product invariants. Edge neighbors are identified via Delaunay triangulation, and E(3)-invariant features $\text{Inv}(\cdot,\cdot)$ are extracted for node–node, node–edge, and edge–edge pairs using Euclidean and Hausdorff distances. Message passing is performed simultaneously on node and edge embeddings using learnable nonlinear update functions.
- **Design Motivation**: Cosmological point clouds possess translational and reflection symmetries that GNNs must respect. Higher-order message passing (edge–edge interactions) aims to capture clustering information beyond two-point correlation functions.

### Key Design 3: Linear Least Squares Baseline (LLS)

- **Function**: Serves as a strong baseline using a linear model with only 49 parameters, predicting cosmological parameters from pairwise distance statistics.
- **Mechanism**: For each point cloud, the mean, standard deviation, and $(1/3, 2/3)$ quantiles of the pairwise distance distribution are computed at 12 different cutoff radii $R_c$, yielding 48 features. A greedy strategy selects the cutoff radius on the validation set, and a biased least-squares fit is then applied to predict the target parameters.
- **Design Motivation**: A physically motivated simple model serves as a sanity check for complex methods—if a GNN cannot substantially outperform a 49-parameter linear model, the deep model likely fails to exploit higher-order information effectively. This also provides a reference baseline at negligible computational cost.

### Key Design 4: Merger Tree Dataset and Super-Resolution Task

- **Function**: Extracts directed merger trees from CAMELS-SAM and designs a temporal super-resolution task.
- **Mechanism**: Trees with root node mass greater than $10^{13} M_\odot/h$ are selected; low-mass subtrees are pruned to eliminate information leakage risk, yielding 25 trees per simulation and 24,996 trees in total. The super-resolution task coarsens each tree by masking even-numbered time steps, then inserts virtual nodes at each merger node; a classifier is trained to determine whether each masked merger node truly exists.
- **Design Motivation**: Merger trees record the formation history of dark matter halos and represent an important data modality beyond point clouds. The temporal super-resolution task simulates reduced temporal resolution caused by storage constraints, directly serving upcoming large-scale survey projects such as Euclid and LSST.

## Loss & Training

- Both point cloud cosmological parameter prediction and velocity prediction use MSE loss; the evaluation metric is the coefficient of determination $R^2$, with uncertainty reported as bootstrap standard deviation on the test set.
- Merger tree node classification uses binary cross-entropy loss; the evaluation metric is accuracy.
- GNNs and DeepSets are trained with the Adam optimizer; datasets are split 60/20/20 into training/validation/test sets.

## Key Experimental Results

### Table 1: Point Cloud Cosmological Parameter Prediction ($R^2$ ↑)

| Method | Quijote $\Omega_m$ | Quijote $\sigma_8$ | Params | CAMELS-SAM $\Omega_m$ | CAMELS-SAM $\sigma_8$ | CAMELS $\Omega_m$ | CAMELS $\sigma_8$ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2PCF+MLP | 0.85 | 0.84 | 11K | 0.73 | 0.82 | 0.84 | 0.30 |
| LLS (49 params) | 0.83 | 0.80 | 49 | 0.77 | 0.82 | 0.78 | 0.28 |
| GNN | 0.80 | 0.77 | 671K | 0.75 | 0.83 | 0.78 | 0.24 |
| GNN (w/o edgeMP) | 0.80 | 0.79 | 128K | 0.72 | 0.84 | 0.80 | 0.27 |

**Key Findings**: The 49-parameter LLS model achieves performance comparable to or better than GNNs with hundreds of thousands of parameters. Removing edge–edge message passing has no significant impact on GNN performance. Prediction of $\sigma_8$ degrades severely on the small-volume CAMELS dataset.

### Table 2: Point Cloud Velocity Prediction ($R^2$ ↑)

| Method | Quijote v | CAMELS-SAM v | CAMELS v |
|------|:---:|:---:|:---:|
| Linear Theory (oracle*) | 0.377 | 0.237 | 0.297 |
| LLS (60 params) | **0.435** | 0.211 | 0.249 |
| GNN (126K params) | 0.410 | **0.287** | 0.253 |

**Key Findings**: At large scales, LLS outperforms both GNN and the linear theory oracle. GNN achieves the best performance at intermediate scales (CAMELS-SAM). ML methods can surpass linear theory—which requires prior knowledge of cosmological parameters—without any such prior.

## Highlights & Insights

1. **Unprecedented scale for a cosmological ML benchmark**: 34K point clouds + 25K merger trees, derived from simulations totaling over 41 million CPU-hours, spanning three spatial scales.
2. **"Less is more" as a key finding**: A 49-parameter linear model can match or exceed GNNs with hundreds of thousands of parameters, revealing a bottleneck for current deep models on cosmological tasks.
3. **Unified interface for multimodal, multitask evaluation**: A unified PyTorch interface is provided, covering two data modalities (point clouds and directed trees) and four task categories.
4. **An initial bridge from simulation to observation**: A redshift-space velocity prediction variant is introduced, moving toward realistic observational settings.

## Limitations & Future Work

1. The CAMELS and CAMELS-SAM datasets contain only 1,000 samples each, which remains insufficient for data-driven methods.
2. Only two cosmological parameters, $\Omega_m$ and $\sigma_8$, are considered (Quijote additionally includes 3 parameters); important parameters such as dark energy are not covered.
3. Current GNNs do not significantly outperform simple baselines on most tasks, indicating that the benchmark's upper bound is far from being reached.
4. The merger tree super-resolution task uses only the 200 largest trees, resulting in limited sample size.
5. Redshift-space treatment is simplified (displacement along the z-axis only) and remains distant from real survey observations.

## Related Work & Insights

- **GDL in cosmology**: Prior work has applied GNNs to infer cosmological parameters from galaxy distributions (Villanueva-Domingo et al., Makinen et al.), but off-the-shelf methods perform poorly on position-only tasks.
- **Point cloud benchmarks**: Computer vision has ShapeNet and ModelNet; biology has AlphaFold DB and MoleculeNet; cosmology lacks a unified benchmark. The pioneering work of Balla et al. is limited in scale (3,560 point clouds).
- **Graph benchmarks**: OGB and TU Datasets primarily cover biological and social networks; cosmological graph data is not included. CosmoBench responds to the call by Dwivedi et al. for improved graph benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First large-scale cosmological geometric learning benchmark with a unique multiscale, multimodal design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Provides physical, linear, and deep learning baselines with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated physical background.
- Value: ⭐⭐⭐⭐ — Fills a gap in cosmological ML benchmarking and reveals the competitiveness of simple models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[CVPR 2026\] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](../../CVPR2026/3d_vision/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](../../ICCV2025/3d_vision/sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](../../ICCV2025/3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)

</div>

<!-- RELATED:END -->
