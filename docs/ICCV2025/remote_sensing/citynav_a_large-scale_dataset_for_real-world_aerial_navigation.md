---
title: >-
  [Paper Note] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation
description: >-
  [ICCV 2025][Remote Sensing][Vision-and-Language Navigation] This paper introduces CityNav, the first large-scale aerial vision-and-language navigation dataset for real-world urban environments, comprising 32…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "Vision-and-Language Navigation"
  - "UAV"
  - "Real World"
  - "Geo-Semantic Map"
  - "Large-Scale Dataset"
date: 2026-05-08
content_hash: dfe962e05e3430b2
---

# CityNav: A Large-Scale Dataset for Real-World Aerial Navigation

**Conference**: ICCV 2025
**arXiv**: [2406.14240](https://arxiv.org/abs/2406.14240)
**Code**: [Project Page](https://water-cookie.github.io/city-nav-proj/)
**Area**: Remote Sensing / Aerial Navigation
**Keywords**: Vision-and-Language Navigation, UAV, Real World, Geo-Semantic Map, Large-Scale Dataset

## TL;DR

This paper introduces CityNav, the first large-scale aerial vision-and-language navigation dataset for real-world urban environments, comprising 32,637 human demonstration trajectories covering 4.65 km². A Geo-Semantic Map (GSM) auxiliary representation is proposed and shown to significantly improve baseline navigation performance.

## Background & Motivation

Vision-and-Language Navigation (VLN) aims to enable agents to navigate in real environments based on natural language instructions. While indoor and ground-level outdoor VLN have been extensively studied, **aerial navigation—especially in real urban environments—remains severely underexplored**:

Limitations of existing aerial VLN datasets:
- **LANI** (6K trajectories): Virtual small-scale environments, 2D action space
- **AVDN** (3K trajectories): Operates only on 2D satellite imagery
- **AerialVLN** (8.4K trajectories): 3D but synthetic cities, lacking real-world complexity

The core issue is that in **real urban environments**, agents must understand spatial relationships among real landmarks and integrate visual and geographic information for navigation—a capability that existing synthetic datasets cannot effectively train.

## Method

### Overall Architecture

CityNav consists of three major components:
1. CityFlight simulation environment (built on real-world 3D scan data)
2. Large-scale human demonstration trajectory collection
3. Geo-Semantic Map (GSM) auxiliary representation

### Key Designs

1. **CityFlight Simulation Environment**:

    - Built from 3D point cloud data of the SensatUrban dataset, covering the real cities of Cambridge and Birmingham
    - Uses Potree (a WebGL point cloud renderer) for 3D scene visualization in the browser, enabling crowdsourced data collection
    - Synchronized with OpenStreetMap in real time, supporting 3D-to-2D coordinate conversion and landmark name retrieval
    - **Action space**: 5D pose $\bm{p} = (x, y, z, \theta, \psi)$, with 6 actions (forward 5 m, turn left/right 30°, ascend/descend 2 m, stop)
    - Maximum flight altitude of 200 m; starting points are randomly sampled within a 500 m radius of the target at altitudes of 100–150 m

2. **Human Demonstration Trajectory Collection**:

    - Crowdsourced via Amazon MTurk; 171 annotators contributed 32,637 high-quality trajectories
    - Three-stage quality control: qualification test (filtering unqualified annotators) → initial collection (rejecting 18.4% of substandard trajectories) → re-collection (7.2% of remaining trajectories fully removed)
    - Target descriptions are drawn from the CityRefer dataset, each containing at least one landmark and describing the spatial relationship between the target and surrounding objects
    - Success criterion: agent stops within a 20 m spherical radius of the target
    - Target type distribution: 48.3% buildings, 40.7% vehicles, 7.4% ground surfaces, 3.6% parking lots

3. **Geo-Semantic Map (GSM)**:

    - Five semantic categories: current field of view, explored region, landmarks, potential destination, surrounding objects
    - Current field of view and explored region are derived from GNSS coordinates
    - Landmarks are retrieved from OpenStreetMap
    - Potential destination and surrounding objects are detected using Grounding DINO
    - Landmark and object names are extracted using GPT-3.5
    - Encoding: binary masks aligned to a 2D map → 5-layer CNN encoder $E$ → feature $\bm{z}^{(t)}_{map}$
    - Integration into VLN model: $\bm{z}^{(t)}_{map}$ is appended to the sequence $[\bm{z}^{(t)}_{RGB}, \bm{z}^{(t)}_{depth}]$ as input to a GRU

### Evaluation Metrics

- **NE (Navigation Error)**: Euclidean distance from stopping point to target
- **SR (Success Rate)**: Proportion of episodes satisfying the success criterion
- **OSR (Oracle Success Rate)**: Proportion of episodes where at least one waypoint satisfies the success criterion
- **SPL (Success weighted by Path Length)**: Success rate weighted by path efficiency

## Key Experimental Results

### Main Results: Three Baseline Models on CityNav

| Method | GNSS | Val-seen NE↓ | Val-seen SR↑ | Test-unseen NE↓ | Test-unseen SR↑ | Test-unseen SPL↑ |
|------|------|-------------|-------------|----------------|----------------|-----------------|
| Seq2Seq | ✗ | 257.1 | 1.81% | 245.3 | 1.50% | 1.30% |
| Seq2Seq + GSM | ✓ | 58.5 | 8.43% | 98.1 | 3.81% | 2.79% |
| CMA | ✗ | 240.8 | 0.95% | 252.6 | 0.82% | 0.79% |
| CMA + GSM | ✓ | 68.0 | 6.25% | 94.6 | 4.68% | 4.05% |
| AerialVLN | ✗ | 185.2 | 1.73% | 187.7 | 1.79% | 0.62% |
| **AerialVLN + GSM** | ✓ | **56.6** | **10.16%** | **85.1** | **6.72%** | **5.16%** |
| Human | - | 9.1 | 89.31% | 9.8 | 87.86% | 57.04% |

GSM reduces navigation error by over 50% and multiplies success rates across all models.

### Ablation Study

| Configuration | NE↓ | SR↑ | OSR↑ | SPL↑ | Note |
|------|-----|-----|------|------|------|
| AerialVLN + GSM (full) | 85.1 | 6.72% | 18.21% | 5.16% | Baseline |
| w/o landmarks | 190.7 | 0.60% | 6.94% | 0.56% | Landmarks are the most critical |
| w/o potential destination | 92.8 | 3.97% | 13.08% | 3.86% | Potential destination matters |
| w/o surrounding objects | 87.5 | 5.17% | 15.16% | 5.10% | Surrounding objects provide auxiliary cues |

### Training Data Comparison

| Training Data | Size | NE↓ | SR↑ |
|---------|--------|-----|-----|
| Shortest path | 22k | 95.1 | 4.96% |
| **Human demonstration** | **22k** | **85.1** | **6.72%** |
| Shortest path + noise | 22k | 123.1 | 2.37% |
| **Human demonstration + noise** | **22k** | **95.0** | **4.92%** |

### Key Findings

- **Large human–machine performance gap**: Human success rate is 87–90%, while the best model achieves only 6–10%, indicating that real-world aerial VLN remains a highly challenging open problem
- **Landmarks are the most critical GSM component**: Removing landmarks causes success rate to drop from 6.72% to 0.60%
- **Human demonstrations outperform shortest-path training**: Human trajectories more frequently pass near landmarks, providing richer visual–geographic associations
- **Human-demonstration-trained models are more robust**: The performance gap widens further under positional noise (NE gap increases from 10.0 to 28.1)
- Longer descriptions and more landmark references consistently improve success rates across all models

## Highlights & Insights

- The first aerial VLN dataset constructed from real-world 3D scans, filling an important research gap
- With 32,637 trajectories, CityNav is the largest aerial VLN dataset to date; the collection pipeline is rigorously designed with three-stage quality control
- The GSM design philosophy—unifying OpenStreetMap geographic information with visual observations into a learnable auxiliary modality—is simple yet highly effective
- The comparison between human demonstrations and shortest paths reveals a fundamental distinction between "cognitive navigation" and "geometric navigation"

## Limitations & Future Work

- Coverage is limited to two cities (Cambridge and Birmingham), constraining global generalizability
- The absolute performance of current models remains low (best SR ~10%), motivating the need for stronger baselines
- Multi-agent collaborative large-area search scenarios have not been explored
- Real physical constraints of UAVs (battery life, obstacle avoidance) are not considered
- The visual fidelity of 3D point cloud rendering still lags behind actual UAV footage

## Related Work & Insights

- **AerialVLN** is the most direct predecessor; CityNav advances it from synthetic to real-world environments
- Experience from ground-level outdoor VLN (**TouchDown**, **Talk2Nav**) in leveraging street-view landmarks complements CityNav's use of aerial landmarks
- The GSM design offers a transferable insight: other multimodal navigation tasks may benefit from incorporating structured geographic knowledge as auxiliary input
- The substantial human–machine gap suggests that future work will require stronger spatial reasoning, 3D scene understanding, and multi-step planning capabilities

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale real-world aerial VLN dataset with high data contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Three baselines × w/ and w/o GSM + training data ablation + robustness evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Comprehensive coverage expected of a dataset paper: task definition, collection pipeline, statistical analysis, and baseline comparisons
- Value: ⭐⭐⭐⭐ Provides critical infrastructure for the aerial VLN community; the human–machine gap clearly defines future research directions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](../../CVPR2026/remote_sensing/olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)

</div>

<!-- RELATED:END -->
