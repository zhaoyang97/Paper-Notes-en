---
title: >-
  [Paper Note] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos
description: >-
  [ICLR 2026][Robotics][urban simulation] UrbanVerse is a data-driven real-to-sim system that converts crowdsourced city-tour videos into physically-aware, interactive simulation environments. It comprises a 100K+ annotated 3D asset library and an automated scene construction pipeline, generating 160 high-quality scenes in IsaacSim. A PPO navigation policy trained on these scenes achieves an 89.7% success rate in zero-shot real-world transfer, completing a 337 m long-range task with only 2 human interventions.
tags:
  - ICLR 2026
  - Robotics
  - urban simulation
  - real-to-sim
  - embodied AI
  - 3D asset library
  - navigation policy
date: 2026-05-08
content_hash: 2147bd1b504cdb51
---

# UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos

**Conference**: ICLR 2026
**arXiv**: [2510.15018](https://arxiv.org/abs/2510.15018)
**Code**: [urbanverseproject.github.io](https://urbanverseproject.github.io/)
**Area**: Robotics / Simulation
**Keywords**: urban simulation, real-to-sim, embodied AI, 3D asset library, navigation policy

## TL;DR
UrbanVerse is a data-driven real-to-sim system that converts crowdsourced city-tour videos into physically-aware, interactive simulation environments. It comprises a 100K+ annotated 3D asset library and an automated scene construction pipeline, generating 160 high-quality scenes in IsaacSim. A PPO navigation policy trained on these scenes achieves an 89.7% success rate in zero-shot real-world transfer, completing a 337 m long-range task with only 2 human interventions.

## Background & Motivation
Embodied AI agents operating in urban environments—such as delivery robots and quadrupedal robots—are advancing rapidly. Training such agents requires large quantities of diverse, high-fidelity urban environments, yet existing simulation approaches face a fundamental tension:

- **Manually crafted scenes**: e.g., CARLA provides only 15 scenes, are not scalable, and incur high labor costs.
- **Procedurally generated scenes**: e.g., MetaUrban/UrbanSim rely on hard-coded rules, producing distributions that deviate from the real world (e.g., randomly placed scooters that do not reflect realistic parking patterns).
- **Passive real-world data**: City-tour videos offer rich diversity but lack action labels and interactivity.
- **3D reconstruction methods**: e.g., 3DGS can reconstruct scenes from video, but the resulting static textured meshes lack semantic and physical attributes.

Root Cause: a fundamental trade-off between scale and fidelity. Simply increasing quantity through procedural generation does not yield generalization—if scenes fail to faithfully reflect real-world distributions, sheer quantity provides little benefit.

Core Idea of UrbanVerse: **extract scene semantics and layout from real city-tour videos, then instantiate them into physically interactive simulation scenes using high-quality 3D assets**—the "digital cousin" paradigm. This approach combines the diversity of real-world data with the interactivity of simulation.

## Method

### Overall Architecture
UrbanVerse consists of two main pillars:
1. **UrbanVerse-100K**: A 100K+ urban 3D asset library annotated with physical attributes.
2. **UrbanVerse-Gen**: An automated video-to-simulation pipeline.

Input: YouTube city-tour videos → Output: physically interactive simulation scenes in IsaacSim.

### Key Designs

1. **UrbanVerse-100K Asset Library (Section 3.1)**:
   Starting from 800K noisy 3D assets in Objaverse, a three-stage semi-automated pipeline filters and annotates assets:

   - **Asset Filtering**: Ten annotators used a Three.js viewer over three weeks to label and filter assets, removing eight categories of quality issues (broken meshes, missing textures, planar geometry, abnormal scale, etc.), retaining 158K usable assets.
   - **Urban Ontology Construction**: Building on OpenStreetMap tag structures and extending categories from ADE20K, Cityscapes, and related datasets, a three-level urban semantic ontology with 667 leaf categories is established.
   - **Attribute Annotation**: GPT-4.1 annotates 33 attributes (semantics, affordances, physical properties such as mass and friction) per asset using thumbnails and four rotated snapshots; total API cost: \$1,334.

   The final library contains: 102,530 GLB objects + 288 PBR ground materials + 306 HDRI sky maps.

2. **UrbanVerse-Gen Scene Construction Pipeline (Section 3.2)**:
   A unified 3D urban scene graph $\mathcal{V} = \langle\mathcal{O}, \mathcal{G}, \mathcal{S}\rangle$ (object / ground / sky nodes) is defined.

   Three processing stages:
   - **Scene Distillation**: Semantic and 3D layout information is extracted from video. MASt3R estimates metric depth and camera poses; YoloWorld+SAM2 performs open-vocabulary object parsing; Mask2Former segments road surfaces and sidewalks. Cross-frame fusion yields persistent object nodes containing category, centroid, 3D bounding box, orientation, and appearance crops.
   - **Asset Matching and Diversification**: For each node, $k_{cousin}$ matching assets are retrieved from UrbanVerse-100K via three steps: CLIP semantic matching → bounding-box geometric filtering (mBBD) → DINOv2 appearance ranking. Ground materials are matched to PBR textures via pixel-level MSE; sky conditions are matched to HDRI maps via HSV histograms.
   - **Simulation Scene Generation**: Scenes are instantiated in UrbanSim (IsaacSim): ground plane fitting and material application → HDRI sky assignment → object placement (centroid alignment + collision detection + physical property assignment).

3. **Digital Cousin Diversification Strategy**:
   Each video-derived scene layout can spawn $k_{cousin}=5$ digital cousin variants with distinct appearances but identical layouts, achieved by substituting different matched assets. This intra-layout diversity complements inter-layout diversity, enhancing policy generalization.

4. **PPO Navigation Policy Training**:

   - Actor-Critic architecture with continuous action space.
   - Observations: RGB image (135×240) + goal relative position.
   - Three-layer CNN encoder [16, 32, 64] + three-layer MLP (128).
   - Reward design: arrival reward (+2000) + collision penalty (−200) + position tracking (coarse + fine) + velocity reward.
   - Training loads 16 scenes per batch, rotating to a new batch every 100 episodes.

### Loss & Training
PPO optimization; learning rate 1e-4 (adaptive); $\gamma=0.99$; GAE $\tau=0.95$; PPO clip $\varepsilon=0.2$; KL threshold 0.01; entropy coefficient 0.002; 1,500 epochs; mixed-precision training on a single L40S GPU.

## Key Experimental Results

### Main Results
**Scene Construction Fidelity** (KITTI-360, 45 sequences, avg. 198.7 m):

| SfM | Scene Parser | Category (%) | Asset (%) | Distance (m) | Orientation (°) | Volume (m³) | mAP25 |
|-----|-------------|--------------|-----------|--------------|-----------------|-------------|-------|
| MASt3R | YoWorldSAM2 | **93.1** | **75.1** | **1.4** | 19.8 | **0.8** | **28.2** |
| VGGT | YoWorldSAM2 | 91.5 | 70.6 | 2.1 | 20.1 | 1.3 | 9.4 |

**CraftBench Generalization** (10 artist-designed scenes):

| Method | SR (%) | CT | RC (%) |
|--------|--------|----|--------|
| MBRA | 35.6 | 25.6 | 52.9 |
| S2E | 33.1 | 27.7 | 55.7 |
| PPO-UrbanSim | 9.1 | 31.5 | 19.4 |
| **PPO-UrbanVerse** | **41.9** | 35.5 | **62.4** |

**Zero-Shot Sim-to-Real** (16 real urban scenes):

| Method | Wheeled SR (%) | Quadruped SR (%) |
|--------|---------------|-----------------|
| NoMad | 33.3 | 37.5 |
| S2E | 47.9 | 58.6 |
| PPO-UrbanSim | 18.8 | 18.8 |
| **PPO-UrbanVerse** | **77.1** | **89.7** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| 1 layout → 32 layouts | SR: low → 41.9% | Scene quantity follows a scaling power law |
| 1 cousin → 5 cousins | SR: low → higher | Intra-layout diversity also matters |
| UrbanVerse vs. PG scenes | Human score: 3.58 vs. 2.9/5 | >70% of users prefer UrbanVerse |
| Pretrain + target-scene fine-tuning | SR: 0% → 80% | Real-to-sim-to-real loop is effective |

### Key Findings
- **Scaling power law holds**: A power-law relationship exists between the number of scenes / digital cousins and policy performance, with high linear-fit $R^2$.
- **Real-world distribution is critical**: An equivalent number of procedurally generated scenes yields almost no generalization improvement (flat PG curve).
- **PPO-UrbanVerse surpasses navigation foundation models**: A simple PPO policy trained on UrbanVerse scenes outperforms large-scale pretrained visual navigation foundation models such as NoMad and CityWalker.
- **Zero-shot transfer is exceptionally strong**: Quadruped Go2 achieves 89.7% success rate in real-world settings, surpassing S2E by +31.1%.
- **337 m long-range task**: Completed on public streets with only 2 human interventions.
- **Human evaluation**: Automatically generated UrbanVerse scenes are rated 3.58/5, compared to 4.08/5 for artist-crafted scenes—a small gap.

## Highlights & Insights
- **End-to-end pipeline**: A complete closed loop from video collection → asset library construction → scene generation → policy training → real-world deployment.
- **100K-scale asset library**: Addresses quality and scale challenges in 3D asset acquisition, constituting an independent and significant contribution.
- **Discovery of scaling laws**: Quantitative evidence for data scaling laws in embodied AI, supporting the claim that "more scenes = better policies."
- **Validation across two robot morphologies**: The same policy generalizes to both wheeled and quadrupedal robots, suggesting that environment understanding rather than specific kinematics is being learned.
- **160 scenes across 24 countries**: Cross-cultural and cross-geographic diversity is key to real-world generalization.
- **Real-to-sim-to-real closed loop**: For known deployment environments, the workflow—capture a video → generate simulation → fine-tune policy → deploy—offers substantial practical value.

## Limitations & Future Work
- The digital cousin approach inevitably introduces a gap relative to the real scene, as asset substitution cannot perfectly replicate original objects.
- Orientation error (19.8°) remains large and may impact precise navigation.
- Only PPO is explored; more advanced reinforcement learning algorithms are not investigated.
- Only navigation tasks are evaluated; manipulation tasks are not addressed.
- Scene dynamics are absent—pedestrians, vehicles, and other moving obstacles are not modeled.
- HDRI sky maps provide static lighting that cannot simulate time-of-day variation.
- Data sourcing is constrained by the Creative Commons licensing of YouTube videos.

## Related Work & Insights
- **Digital Cousins (Dai et al., 2024)**: Multi-variant generation for indoor scenes; UrbanVerse extends this paradigm to large-scale outdoor urban environments.
- **MetaUrban / UrbanSim (Wu et al., 2025)**: The simulation platform underpinning UrbanVerse; this work addresses the limitations of procedural generation therein.
- **ViNT / NoMad (Shah et al., 2023; Sridhar et al., 2024)**: Visual navigation foundation models trained on passive data, lacking interactive learning.
- **S2E (He et al., 2025)**: Trains obstacle-avoidance policies in simulation and transfers them to the real world, but at a smaller scene scale and diversity.
- **Data Scaling Laws (Lin et al., 2025)**: Scaling laws for imitation learning data; UrbanVerse identifies analogous trends in the RL + simulation scene setting.
- Insight: Crowdsourced video represents a nearly unlimited source of simulation material, a paradigm worth extending to other domains (e.g., indoor environments, factories).

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](../../NeurIPS2025/robotics/spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)

<!-- RELATED:END -->
