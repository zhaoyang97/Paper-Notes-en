---
title: >-
  [Paper Note] WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?
description: >-
  [CVPR 2026][3D Vision][stereo matching] This paper systematically investigates the design space of synthetic stereo datasets by individually varying six key parameters (floating object density, background objects, object types, materials, camera baseline, lighting augmentation) within the Infinigen procedural generator, and quantifies their impact on zero-shot stereo matching. The study finds that the combination of **realistic indoor scenes + floating objects** is most effective, leading to the construction of the WMGStereo-150k dataset. Training on this single dataset surpasses the combination of SceneFlow + CREStereo + TartanAir + IRS (28% reduction on Middlebury, 25% on Booster), with performance competitive with FoundationStereo.
tags:
  - CVPR 2026
  - 3D Vision
  - stereo matching
  - synthetic data
  - procedural generation
  - zero-shot
  - dataset design
date: 2026-05-08
content_hash: ac7b492735c923a8
---

# WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?

**Conference**: CVPR 2026
**arXiv**: [2504.16930](https://arxiv.org/abs/2504.16930)
**Code**: [GitHub](https://github.com/princeton-vl/InfinigenStereo)
**Area**: 3D Vision / Stereo Matching
**Keywords**: stereo matching, synthetic data, procedural generation, zero-shot, dataset design

## TL;DR

This paper systematically investigates the design space of synthetic stereo datasets by individually varying six key parameters (floating object density, background objects, object types, materials, camera baseline, lighting augmentation) within the Infinigen procedural generator, and quantifies their impact on zero-shot stereo matching. The study finds that the combination of **realistic indoor scenes + floating objects** is most effective, leading to the construction of the WMGStereo-150k dataset. Training on this single dataset surpasses the combination of SceneFlow + CREStereo + TartanAir + IRS (28% reduction on Middlebury, 25% on Booster), with performance competitive with FoundationStereo.

## Background & Motivation

- **Existing Problem**: Synthetic data is the cornerstone of training stereo matching networks, yet the question of **"what makes a dataset effective?"** remains unanswered systematically. SceneFlow uses flying objects, TartanAir uses realistic indoors, and FoundationStereo uses a hybrid approach—each simultaneously changes multiple factors, making it impossible to isolate individual contributions.
- **Research Gap**: Most datasets do not release generation code, precluding reproduction, modification, or ablation. The only related parameter study (Mayer et al. 2018) targets 2D warp optical flow with the conclusion that "realism is overrated," and whether this extends to 3D stereo matching remains unknown.
- **Mechanism**: The Infinigen procedural generator is used as a platform to control each parameter independently, generating 5,000 stereo pairs per configuration → training RAFT-Stereo → evaluating on 7 benchmarks → identifying the optimal parameter combination → generating a large-scale dataset.
- **Key Value**: Beyond providing a new dataset, this work delivers **interpretable parameter analysis** and **open-source generation code**, enabling the community to customize data on demand.

## Method

### Overall Architecture

A controllable stereo data generation system is built on top of Infinigen (a procedural generator for indoor and natural scenes) and the Blender Python API. Three scene types are supported:

1. **Indoor Floating**: Infinigen Indoors generates realistic room layouts, with floating objects randomly placed inside.
2. **Dense Floating**: 200 floating objects densely placed against an empty sky background.
3. **Nature**: Infinigen Nature generates natural environments.

For each parameter configuration, an independent dataset is generated → RAFT-Stereo is trained for 75k steps → zero-shot performance is evaluated on Middlebury/ETH3D/KITTI/Booster → the optimal configuration is determined → WMGStereo-150k (163,396 pairs) is generated using this configuration.

### Key Designs

#### 1. Floating Object Placement System

- **Function**: Randomly generates and places floating objects within a given scene, supporting ray-cast-based placement within the camera's field of view and within bounding boxes.
- **Core Idea**: Floating objects provide extensive geometric diversity—objects of varying shapes distributed randomly in space create rich occlusions, depth discontinuities, and complex geometric structures.
- **Design Motivation**: SceneFlow's FlyingThings3D established the effectiveness of flying objects, but this strategy had never been tested against realistic scene backgrounds. This work combines floating objects with realistic indoor layouts.
- **Ablation Validation**: No floating objects → 0–10 → 10–30; Middlebury(H) decreases continuously from 12.52 → 7.78 → 6.60, confirming that floating objects are critical for zero-shot performance.

#### 2. Background Realism Preservation Strategy

- **Function**: Retains realistic background objects (furniture, tables, chairs, etc.) generated by Infinigen Indoors.
- **Core Idea**: Contrary to the "realism doesn't matter" conclusion in optical flow, realistic background geometry provides useful training signal—regular planes and plausible spatial layouts help networks learn real-world depth distributions.
- **Design Motivation**: FlyingThings3D contains no background at all, while TartanAir relies entirely on realistic scenes but lacks floating objects. This work combines the advantages of both.
- **Ablation Validation**: With vs. without background; consistent improvement across all benchmarks (Middlebury(H): 8.35 → 6.60).

#### 3. Object Type Diversity and Material Filtering

- **Function**: Uses all Infinigen object generators (chairs, shelves, bushes, etc.) while removing high-error objects (cacti, sea urchins, and similar spiky structures) and extreme materials (fully transparent glass, fully reflective metal).
- **Core Idea**: Object type diversity ensures robustness across benchmarks; material diversity is important but extreme cases must be filtered, as current network architectures cannot learn extreme non-Lambertian surfaces without degrading performance on diffuse regions.
- **Design Motivation**: Using only chairs yields good Middlebury results (5.29) but poor KITTI-15 (7.02); using only bushes yields good ETH3D (3.13) but poor Booster (12.19). Bias toward any single type introduces benchmark-specific bias.
- **Filtering Method**: Per-pixel EPE is aggregated by object/material group (pixels accounting for ≥0.1% of the total), and top-error objects and materials are identified and removed.

#### 4. Camera Baseline Randomization

- **Function**: Samples the stereo camera separation uniformly over a wide range of [0.04, 0.4] m.
- **Core Idea**: Baseline determines the disparity distribution—narrow baselines [0.04, 0.1] m produce small disparities, while large baselines [0.2, 0.3] m produce large disparities. Wide-range sampling covers all downstream scenarios.
- **Design Motivation**: A narrow-baseline model achieves 9.60 on Middlebury(H) but degrades to 32.47 on Middlebury(F); a large-baseline model fails on ETH3D (14.05). Baseline diversity is a "free" yet critical improvement.
- **Ablation Validation**: [0.04, 0.1] → [0.2, 0.3] → [0.04, 0.4]; the wide range is the most robust across all benchmarks.

#### 5. Generation Cost Optimization

- **Function**: Maximizes data volume within a fixed computational budget by reducing cost approximately 6× through three measures.
- **Core Idea**:
    - **Reducing indoor solver steps** (550 → 60): The constraint solver greedily adds objects without moving or deleting them, reducing generation time from 50.85 minutes to 13 minutes.
    - **Reducing ray-tracing samples** (8192 → 1024 samples + OptiX denoising): Rendering time reduced to 27 seconds per frame.
    - **Scene reuse**: Each indoor scene uses 20 camera placements; each dense floating scene randomizes object positions, orientations, lighting, and baselines 200 times.
- **Design Motivation**: Under a fixed computational budget, lower quality but higher quantity (30k pairs) outperforms high quality but lower quantity (5k pairs)—Middlebury(H) decreases from 6.60 to 5.63.

## Key Experimental Results

### Main Results: WMGStereo-150k vs. Existing Datasets (DLNR, 200k training steps)

| Training Data | Midd-14(F) | Midd-14(H) | Midd-21 | ETH3D | KITTI-12 | KITTI-15 | Booster(Q) |
|---|---|---|---|---|---|---|---|
| SceneFlow | 10.96 | 6.20 | 8.44 | 23.12 | 9.45 | 15.74 | 18.17 |
| CREStereo | 14.45 | 11.53 | 10.60 | 5.18 | 4.95 | 5.90 | 14.61 |
| TartanAir | 12.56 | 7.27 | 14.47 | 4.35 | 3.98 | 5.33 | 18.14 |
| IRS | 7.81 | 6.13 | 8.49 | 3.91 | 4.56 | 5.60 | 10.32 |
| FSD | 5.80 | 3.27 | 6.93 | 2.13 | 3.56 | 4.18 | 7.51 |
| **WMGStereo-150k** | **5.10** | 3.76 | **6.72** | 2.50 | **3.30** | 4.54 | 9.09 |
| FSD+WMGStereo | 5.24 | **3.24** | 6.88 | **2.08** | 3.59 | **4.26** | **7.42** |

### Cross-Architecture Validation: Single Dataset vs. Four-Dataset Mixed (SF+CRE+Tartan+IRS, 600k pairs)

| Model–Data | Midd-14(H) | Midd-21 | ETH3D | KITTI-12 | KITTI-15 | Booster(Q) |
|---|---|---|---|---|---|---|
| RAFT-Mixed | 5.50 | 8.97 | 2.58 | 3.64 | 4.95 | 11.46 |
| **RAFT-WMGStereo** | **4.48** | **8.17** | 2.93 | **3.25** | **4.25** | **9.17** |
| Sel-IGEV-Mixed | 5.24 | 8.24 | 2.37 | 3.97 | 5.31 | 11.00 |
| **Sel-IGEV-WMGStereo** | **3.61** | **7.62** | 2.47 | **3.26** | **4.55** | **8.84** |
| DLNR-Mixed | 5.21 | 9.30 | 2.50 | 3.68 | 4.95 | 12.17 |
| **DLNR-WMGStereo** | **3.76** | **6.72** | 2.50 | **3.30** | **4.54** | **9.09** |

### Ablation Study: Key Parameter Configurations (RAFT-Stereo, 5k pairs, 75k steps)

| Parameter | Setting | Midd-14(H) | ETH3D | KITTI-15 | Booster(Q) |
|---|---|---|---|---|---|
| Floating Density | No floating objects | 12.52 | 4.47 | 6.19 | 16.40 |
| | 0–10 objects | 7.78 | 3.62 | 6.09 | 12.21 |
| | 10–30 objects | **6.60** | 3.92 | **5.11** | **10.60** |
| Background Objects | No background | 8.35 | 4.39 | 6.28 | 12.72 |
| | With background | **6.60** | **3.92** | **5.11** | **10.60** |
| Materials | No materials | 9.02 | 3.48 | 6.07 | 14.07 |
| | Diffuse only | 7.21 | **2.77** | 5.41 | 12.73 |
| | Metal+glass only | 8.37 | 4.95 | **4.97** | **9.80** |
| | All materials | **6.60** | 3.92 | 5.11 | 10.60 |
| Baseline Range | [0.04, 0.1] m | 9.60 | **2.89** | 6.64 | 17.03 |
| | [0.2, 0.3] m | 7.01 | 14.05 | 5.37 | **8.96** |
| | [0.04, 0.4] m | **6.60** | 3.92 | **5.11** | 10.60 |

### Key Findings

- **Remarkable sample efficiency**: Only 500 WMGStereo samples achieve lower EPE on Middlebury than 100K CREStereo samples, demonstrating that dataset design matters more than dataset size.
- **Optimal scene mixing ratio**: A 33% Indoor + 33% Dense Floating + 33% Nature split is the most robust across all benchmarks; Indoor Floating is the best single scene type.
- **Quantity over quality under fixed compute**: Reducing rendering fidelity by 6× but increasing data volume by 6× (5k → 30k pairs) lowers Middlebury(H) from 6.60 to 5.63.
- **Generalization to unseen benchmarks**: On DrivingStereo, which was not involved in parameter tuning, WMGStereo achieves a 3px error of 1.89, 27% lower than FSD's 2.59.

## Highlights & Insights

- **"Realistic scenes + random objects" outperforms either alone**: This overturns the classic optical flow finding that "realism is overrated." For stereo matching, realistic background geometry does provide useful training signal—but realism alone is insufficient; geometric diversity from floating objects is also required.
- **Camera baseline diversity is a critically underestimated factor**: Simply expanding the baseline sampling range yields substantial gains, yet this has never been systematically studied before.
- **Data design >> data scale**: 500 WMGStereo samples outperform 100K CREStereo samples, demonstrating that parameter choices far outweigh data volume.
- **Unique value of open-source generation code**: Unlike FSD (a static dataset), users can adjust parameters on demand—e.g., generating only glass-material data for non-Lambertian scenes—enabling joint design of data and architecture.

## Limitations & Future Work

- Non-Lambertian surfaces remain a bottleneck: the current approach only removes extreme transparent/reflective materials as a compromise, rather than truly addressing the network's difficulty in learning such surfaces.
- Natural scenes (Nature) alone perform worst (Midd-14(H): 12.27), potentially requiring better natural object generation and camera placement strategies.
- Temporal/video stereo data generation is not addressed.
- Part of the gap with FoundationStereo stems from architectural differences (FS introduces a new architecture), not purely from data.
- The parameter study fixes RAFT-Stereo; whether optimal parameters are architecture-dependent is not fully validated (only the final configuration is evaluated across architectures).

## Related Work & Insights

- **vs. FoundationStereo Dataset (FSD)**: FSD simultaneously introduces multiple new features and a new architecture; this work isolates and analyzes the contribution of each factor. The two are complementary—combined use yields the best results (ETH3D: 2.08, Booster: 7.42).
- **vs. SceneFlow/FlyingThings3D**: The classic flying-object dataset lacks scene realism and material diversity. WMGStereo substantially surpasses it with approximately one-quarter of the data volume.
- **vs. Mayer et al. (2018) flow data study**: That study focused on 2D warp + optical flow and concluded "realism is overrated." This work finds that background realism does help in 3D stereo—an important domain-level distinction.
- **vs. original Infinigen stereo attempts**: Raistrick et al. used Infinigen Nature for stereo but did not achieve competitive results. The breakthroughs here required careful data engineering: adding floating objects, material filtering, and baseline expansion.

## Rating

| Dimension | Score (1–10) | Notes |
|---|---|---|
| Novelty | 7 | Not a new method or architecture, but a systematic study of the dataset design space; the angle is novel though the technical barrier is modest. |
| Experimental Thoroughness | 9 | Parameter study covers six factors, validated across three architectures, with cost analysis and sample efficiency curves—highly comprehensive. |
| Value | 9 | Open-source generation code + parameter guidelines enable direct community use or customization; WMGStereo-150k is itself a high-value dataset. |
| Writing Quality | 8 | Well-structured; the controlled variable experimental design in Tab. 1 is textbook-quality, with rich figures and tables. |
| **Overall** | **8.0** | A data-driven systematic study with rigorous experimental design that offers long-term guidance for the stereo data generation community. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [\[CVPR 2026\] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)
- [\[CVPR 2026\] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)

</div>

<!-- RELATED:END -->
