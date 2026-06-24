---
title: >-
  [Paper Note] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection
description: >-
  [NeurIPS 2025 (Workshop: AI for Science)][Object Detection][maritime anomaly detection] This paper proposes the first graph anomaly detection benchmark for non-grid spatio-temporal systems in the maritime domain. It extends the OMTAD dataset to support node/edge/graph-level anomaly detection, and plans to employ LLM agents for trajectory synthesis and anomaly injection.
tags:
  - "NeurIPS 2025 (Workshop: AI for Science)"
  - "Object Detection"
  - "maritime anomaly detection"
  - "spatio-temporal graphs"
  - "non-grid environments"
  - "LLM agents"
  - "AIS data"
date: 2026-05-08
content_hash: 17b5bed26ce0d6ff
---

# Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection

**Conference**: NeurIPS 2025 (Workshop: AI for Science)  
**arXiv**: [2512.20086](https://arxiv.org/abs/2512.20086)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: maritime anomaly detection, spatio-temporal graphs, non-grid environments, LLM agents, AIS data

## TL;DR
This paper proposes the first graph anomaly detection benchmark for non-grid spatio-temporal systems in the maritime domain. It extends the OMTAD dataset to support node/edge/graph-level anomaly detection, and plans to employ LLM agents for trajectory synthesis and anomaly injection.

## Background & Motivation
- Spatio-temporal graph neural networks (ST-GNNs) have achieved success in structured domains (road traffic, public transit) where nodes correspond to fixed spatial anchors (intersections, stations).
- **Fundamental challenge in the maritime domain**: Open seas lack natural fixed nodes; shipping routes are irregular and sparse, making graph construction itself a non-trivial problem.
- Anomalies may manifest at multiple granularities: individual behavioral anomalies (node-level), abnormal interactions (edge-level), and collective anomalies (graph-level).
- Existing maritime datasets are not designed for anomaly detection and lack systematic anomaly annotations.
- Non-grid spatio-temporal systems are expected to become increasingly prevalent (drone swarms, air traffic management).

## Method

### Overall Architecture
A two-stage extension pipeline based on the OMTAD dataset:
1. **Trajectory Synthesizer**: Enriches vessel-to-vessel interactions in sparse regions.
2. **Anomaly Injector**: Semantically generates anomalies via LLM-based prompting.

### Key Designs

#### Data Foundation: OMTAD
- Coverage: Western Australian waters (105–116°E, 36–15°S), 2018–2020.
- 19,124 trajectories: Cargo (14,384), Tanker (4,020), Fishing (466), Passenger (254).
- AIS records include: vessel ID, geographic position, course over ground (COG), speed over ground (SOG), and UTC timestamps.

#### Two Limitations of OMTAD and Solutions
1. **No neighbors in sparse regions**: Generates synthetic yet physically plausible companion trajectories via bounded perturbations on SOG, COG, and geographic position.
2. **No anomaly labels**: Introduces anomalies through a controlled injection process.

#### Two-Agent Architecture
**Coordinator**:
- Constructs standardized perception packets (AIS + derived features + environmental data + provenance information).
- Sequentially schedules the Trajectory Synthesizer and Anomaly Injector.

**Trajectory Synthesizer**:
- Proximity augmentation: directly incorporates physically nearby vessels.
- Synthetic augmentation: generates "virtual neighbors" in sparse regions with perturbations on SOG, COG, and coordinates.

**Anomaly Injector (prompt-driven)**:
- Prompt parsing: translates natural language descriptions (e.g., "abnormal speed change," "dangerous encounter," "collective loitering") into structured intents.
- Scene realization: maps intents to spatio-temporal graph edits (modifying single-node kinematics, vessel-to-vessel interactions, or collective patterns).
- Label generation: attaches anomaly labels (node/edge/graph-level) along with interpretable provenance text.

#### Preliminary Anomaly Injection Method (Pre-experiments)
- For a trajectory of length $w$, selects a contiguous anomalous block of size $m = r_{node} \cdot w$.
- Perturbs the rate of change of SOG and COG: $a_i^* = \mu_a + k \cdot \sigma_a$, with $k > 3$ (exceeding the 99.7% confidence interval).
- Two-level control: $r_{node}$ governs intra-trajectory anomaly density; $r_{traj}$ governs dataset-level class balance.

### Graph Construction
- Applies the OPTICS clustering algorithm to spatial snapshots at each timestamp.
- Samples a fixed number $k$ of trajectories from each cluster.
- Constructs directed temporal graphs within a window of $w$ hours, yielding $k \times w$ nodes per graph.

## Key Experimental Results

### Preliminary Experiment: Graph-Level Anomaly Detection

| Model | $r_{traj}=0.1$ | $r_{traj}=0.5$ |
|---|---|---|
| LSTM | Baseline | Baseline |
| LSTM + GNN | Outperforms LSTM ✓ | Outperforms LSTM ✓ |
| Transformer | Baseline | Baseline |
| Transformer + GNN | Outperforms Transformer ✓ | Outperforms Transformer ✓ |

### Experimental Settings

| Parameter | Setting |
|---|---|
| Node anomaly ratio $r_{node}$ | {0.1, 0.3, 0.5} |
| Trajectory anomaly ratio $r_{traj}$ | {0.1, 0.5} |
| Fixed $r_{node}$ | 0.5 (preliminary experiments) |
| Perturbation strength $k$ | > 3 (beyond 3σ) |

### Key Findings
- GNN-augmented models consistently outperform pure sequential baselines across all anomaly ratios.
- Graph modeling more naturally captures maritime dynamics by jointly considering vessel states and inter-vessel interactions.
- Graph-structural signals are informative even under relatively naive anomaly injection settings.
- Current injections cover only the simplest kinematic anomalies — real maritime anomalies are far more diverse.

## Highlights & Insights
1. **Fills an important gap**: The first graph anomaly detection benchmark targeting non-grid spatio-temporal systems.
2. **Three-level anomaly support**: Unified evaluation of node/edge/graph-level anomaly detection.
3. **LLM-assisted data generation**: Leverages LLM agents to generate semantically rich anomalies beyond rule-driven injection.
4. **Extensibility**: The framework generalizes to other non-grid spatio-temporal systems such as drone swarms and air traffic management.

## Limitations & Future Work
- Current scope is limited to kinematic anomalies; extension to illegal rendezvous, AIS spoofing, and environmental anomalies is needed.
- The LLM agent pipeline remains at the planning stage and is not yet fully implemented.
- The dataset covers only a single region in Western Australia.
- Task-specific labeling strategies require refinement — defining consistent and interpretable labels across anomaly levels is non-trivial.
- The finalized benchmark dataset has not yet been released.
- Future plans include: deterministic reproducible pipelines, multi-baseline benchmarking, and semantically complex anomaly types.

## Related Work & Insights
- Success of ST-GNNs in structured domains (STGCN, DCRNN, ASTGCN).
- Maritime trajectory prediction methods such as GeoTrackNet and TrAISformer.
- AD-LLM: a comprehensive benchmark for LLM-assisted anomaly detection.
- BotSim: LLM-driven generation of malicious social bots.
- This work represents the first systematic application of LLMs to anomaly injection in the maritime domain.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel direction combining non-grid spatio-temporal anomaly detection with LLM-based injection)
- Technical Depth: ⭐⭐⭐ (Preliminary validation stage; core methods not yet fully implemented)
- Experimental Thoroughness: ⭐⭐⭐ (Only preliminary experiments; comprehensive baseline comparisons are lacking)
- Writing Quality: ⭐⭐⭐⭐ (Problem formulation is clear; future directions are well-defined)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](../../ECCV2024/object_detection/walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)

</div>

<!-- RELATED:END -->
