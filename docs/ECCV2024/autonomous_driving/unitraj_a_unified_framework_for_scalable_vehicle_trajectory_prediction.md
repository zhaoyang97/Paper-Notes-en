---
title: >-
  [Paper Note] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction
description: >-
  [ECCV 2024][Autonomous Driving][Trajectory Prediction] UniTraj establishes a unified framework for vehicle trajectory prediction by standardizing multiple datasets (nuScenes, Argoverse 2, WOMD), models (AutoBot, MTR, Wayformer), and evaluation strategies. The study reveals a significant drop in cross-dataset generalization for individual models, but demonstrates that scaling up training data volume and diversity substantially boosts performance…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Trajectory Prediction"
  - "Unified Framework"
  - "Cross-Dataset Generalization"
  - "Data Scaling"
  - "Multi-Dataset Training"
date: 2026-05-08
content_hash: f1c151001a26a605
---

# UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction

**Conference**: ECCV 2024  
**arXiv**: [2403.15098](https://arxiv.org/abs/2403.15098)  
**Code**: [https://github.com/vita-epfl/UniTraj](https://github.com/vita-epfl/UniTraj)  
**Area**: Autonomous Driving  
**Keywords**: Trajectory Prediction, Unified Framework, Cross-Dataset Generalization, Data Scaling, Multi-Dataset Training

## TL;DR
UniTraj establishes a unified framework for vehicle trajectory prediction by standardizing multiple datasets (nuScenes, Argoverse 2, WOMD), models (AutoBot, MTR, Wayformer), and evaluation strategies. The study reveals a significant drop in cross-dataset generalization for individual models, but demonstrates that scaling up training data volume and diversity substantially boosts performance, achieving 1st place on the nuScenes leaderboard via joint training.

## Background & Motivation

Vehicle trajectory prediction is a core task for safety and collision avoidance in autonomous driving. Although deep-learning-driven prediction models have achieved high accuracy, they heavily rely on specific training domains. Autonomous driving systems inevitably encounter diverse scenarios featuring different geographical locations, traffic rules, and road layouts, where domain shifts significantly degrade model performance.

**Three Major Dataset Compatibility Barriers**:
- **Different Data Formats**: WOMD uses TFRecord, while Argoverse 2 uses Apache Parquet, preventing direct mixed utilization.
- **Large Differences in Data Features**: Varied temporal durations (8-20s), map resolutions (0.2-2m), sampling rates, and agent-type annotations.
- **Inconsistent Evaluation Metrics**: WOMD uses mAP, whereas Argoverse 2 relies on brier-minFDE, which hinders direct comparison.

**Two Core Research Questions**:
- **RQ1**: How well do trajectory prediction models generalize across datasets/cities? How much does performance degrade?
- **RQ2**: Does scaling up training data volume and diversity improve prediction performance? How much headroom for improvement exists?

**Key Insight**: Build the unified UniTraj framework to comprehensively standardize data formats, features, models, and evaluations, enabling multi-dataset joint training and systematic generalization studies for the first time.

## Method

### Overall Architecture

UniTraj consists of three core components: (1) **Unified Data**—standardizing formats and features across multiple datasets; (2) **Unified Models**—adapting multiple SOTA prediction models to the unified data format; (3) **Unified Evaluation**—providing general and fine-grained evaluation metrics. The framework builds the largest public vehicle trajectory prediction dataset, supporting cross-dataset training, evaluation, and analysis.

### Key Designs

1. **Unified Data Format**: Based on ScenarioNet, different formats (such as TFRecord and Apache Parquet) are converted into a unified scenario description format. Originally designed for traffic simulation, ScenarioNet is repurposed here for trajectory prediction tasks and extended with support for Argoverse 2. This eliminates the need to write separate preprocessing code for each dataset.

2. **Unified Data Features**: Systematically addresses four types of discrepancies:

   | Feature | Argoverse 2 | WOMD | nuScenes | UniTraj Unified Scheme |
   |------|-------------|------|----------|-----------------|
   | Coordinate System | Scene-centric | Scene-centric | Scene-centric | **Agent-centric** |
   | History Duration | 5s | 1s | 2s | [0-8]s configurable |
   | Future Duration | 6s | 8s | 6s | [1-8]s configurable |
   | Map Resolution | 0.2-2m | ~0.5m | ~1m | 0.5m (normalized by linear interpolation) |
   | Map Range | ~200m | ~200m | ~500m | [0-500]m configurable |

   Core processing steps include:
    - **Coordinate Transformation**: Transforms original scene-level data into an agent-centric vectorized format.
    - **Temporal Alignment**: Truncates all trajectories to a uniform 8-second length, offering flexible history-future split configurations.
    - **Agent Features**: Unified into 2D coordinates + velocity + heading, augmented with acceleration and agent type one-hot encoding.
    - **Map Features**: Standardizes the spacing between continuous points to 0.5m through linear interpolation, supporting downsampling while enriching lane heading and type encoding.

3. **Unified Model Platform**: Integrates three representative SOTA models covering different parameter scales:

    - **AutoBot** (1.5M params): Transformer-based, leveraging equivariant feature learning + multi-head attention.
    - **MTR** (60.1M params): 2022 WOMD Challenge winner, utilizing global intention priors + local motion refinement.
    - **Wayformer** (16.5M params): Multi-axis encoder + latent queries (re-implemented in this study).

   By standardizing the output format, new models can seamlessly interface with UniTraj's evaluation and logging tools.

4. **Unified Evaluation Strategy**: Comprises two levels of evaluation:

    - **General Metrics**: minADE/minFDE (Minimum Average/Final Displacement Error), Miss Rate (ratio of minFDE > 2m), brier-minFDE (incorporating a probability penalty term $(1-p)^2$).
    - **Fine-grained Evaluation**:
        - *Trajectory Type Stratification*: Categorizes trajectories into 8 types (stationary, straight, left-turn, etc.) following the WOMD classification, specifically targeting rare yet safety-critical scenarios.
        - *Kalman Difficulty Stratification*: Quantifies scenario difficulty using Kalman filter FDE, specifically designed to evaluate model performance on challenging scenarios.

### Loss & Training

- All models maintain their original configuration and hyperparameters.
- Map range is standardized to a 100m radius with a spatial resolution of 0.5m.
- 2-second history trajectory, predicting a 6-second future.
- For multi-dataset training, training sets from all datasets are directly merged.
- Supports multi-process processing and caching mechanisms for efficient data handling.

## Key Experimental Results

### Cross-Dataset Generalization Experiment (RQ1)

| Training Data | #trajs | Eval nuScenes↓ | Eval Argoverse 2↓ | Eval WOMD↓ |
|---------|--------|---------------|-------------------|-----------|
| nuScenes | 32k | **2.86** | 4.50 | 7.38 |
| Argoverse 2 | 180k | 3.72 | **2.08** | 4.68 |
| WOMD | 1800k | 3.10 | 3.63 | **2.13** |
| **All** | **2012k** | **2.27** | **1.99** | **2.13** |

(Taking the MTR model's brier-minFDE as an example; all models show consistent trends)

### Cross-City Generalization Experiment

| Training City | Eval Pittsburgh↓ | Eval Boston↓ | Eval Singapore↓ | Average↓ |
|---------|-----------------|-------------|----------------|------|
| Pittsburgh | **2.4** | 2.7 | 3.5 | 2.8 |
| Boston | 4.1 | **2.2** | 3.4 | 3.2 |
| Singapore | 4.9 | 3.5 | **2.1** | 3.5 |

### nuScenes Leaderboard

| Method | Rank | minADE5↓ |
|------|------|---------|
| **MTR-UniTraj** | **1** | **0.96** |
| Goal-LBP | 2 | 1.02 |
| CASPNet++ | 3 | 1.16 |
| AutoBot-UniTraj | 11 | 1.26 |
| AutoBot | 19 | 1.37 |

### Key Findings

- **Poor Generalization**: All models experience a substantial drop in performance during cross-dataset evaluation. For instance, when MTR is trained on nuScenes and evaluated on WOMD, the brier-minFDE deteriorates from 2.13 to 7.38.
- **WOMD Generalizes Best**: The cross-dataset generalization ability consistently follows the order WOMD > Argoverse 2 > nuScenes, even when controlling for identical dataset sizes. This suggests that diversity is more critical than scale.
- **Significant Gains from Data Scaling**: Joint training reduces MTR's brier-minFDE on nuScenes from 2.86 to 2.27 (a 20.6% improvement), placing it at the top of the leaderboard.
- **Model Capacity Affects Gains**: MTR (60.1M parameters) benefits much more from larger quantities of data compared to AutoBot (1.5M).
- **Diversity is Key**: WOMD exhibits the most balanced distribution of trajectory types (e.g., featuring twice as many left/right turns as other datasets), which explains its superior generalization.
- **Singapore Generalizes Worst**: Models trained on left-hand traffic cities generalize poorly to right-hand traffic cities, stressing the importance of geographic diversity.

## Highlights & Insights

- **Systematic Contribution**: The first open-source unified framework for vehicle trajectory prediction, resolving long-standing dataset compatibility issues in the community.
- **Data Diversity > Data Scale**: Controlled dataset-size experiments demonstrate that WOMD's superior generalization stems not only from larger data size but also from highly diverse scenarios.
- **Insights into Kalman Difficulty**: Distinguishing easy/hard samples using a Kalman filter baseline reveals that joint training yields the greatest improvement in moderately difficult samples (due to the increased prevalence of moderately challenging samples after merging).
- **Trajectory Type Analysis**: Joint training yields improvements across all trajectory categories, particularly right U-turns (dropping from 8.13 to 2.98), underscoring the vital role of data augmentation for rare scenarios.
- **Practical Engineering Value**: The framework supports one-click integration of new datasets and models, with multi-processing and caching mechanisms ensuring high efficiency.

## Limitations & Future Work

- Currently supports only vehicle trajectory prediction, leaving out pedestrians and cyclists.
- Due to the lack of official train/val splits for prediction tasks, nuPlan is restricted to cross-city experiments.
- Residual discrepancies between datasets (e.g., annotation noise) might affect fair comparison.
- Joint training simply merges datasets without exploring more sophisticated data mixing strategies (such as upsampling rare types).
- Only three models are evaluated, excluding recent LLM-based or diffusion-based approaches.
- Future research can explore domain adaptation or continual learning methods to further improve generalization.

## Related Work & Insights

- **ScenarioNet**: Provides a unified scenario description format, which this work extends from simulation to trajectory prediction.
- **Trajnet++**: A unified benchmark for pedestrian trajectory prediction, whereas this study presents the first unified framework for vehicles.
- **trajdata**: Standardizes the interface for human trajectory datasets, inspiring the design philosophy of UniTraj.
- **Insight**: A unified framework is not just an engineering tool, but can also foster new research questions like "data diversity vs. scale." Even 22M samples have not yet hit the performance ceiling, showing the immense value of constructing even larger datasets.

## Rating

- Novelty: ⭐⭐⭐ Primarily focused on framework construction with limited methodological innovation, but possesses high system-level engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally comprehensive experiments spanning cross-dataset, cross-city, data scaling, fine-grained analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation centered around the two RQs with rigorous experimental analysis.
- Value: ⭐⭐⭐⭐⭐ The open-source framework offers tremendous value to the community, and the insights on data diversity provide strong guidance for future dataset construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RAG-TP: A General Framework for Vehicle Trajectory Prediction via Retrieval-Augmented Generation](../../CVPR2026/autonomous_driving/rag-tp_a_general_framework_for_vehicle_trajectory_prediction_via_retrieval-augme.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[ECCV 2024\] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)

</div>

<!-- RELATED:END -->
