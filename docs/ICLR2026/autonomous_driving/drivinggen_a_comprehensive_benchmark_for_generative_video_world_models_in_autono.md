---
title: >-
  [Paper Note] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving
description: >-
  [ICLR 2026][Autonomous Driving][Video world models] DrivingGen introduces the first comprehensive benchmark for autonomous driving video world models, comprising a diverse evaluation dataset spanning weather/geography/time/complex scenarios and a four-dimensional metric framework (distribution, quality, temporal consistency, trajectory alignment). Evaluation of 14 SOTA models reveals a fundamental trade-off between general-purpose and driving-specific models.
tags:
  - ICLR 2026
  - Autonomous Driving
  - Video world models
  - benchmark
  - driving scene generation
  - trajectory evaluation
  - temporal consistency
date: 2026-05-08
content_hash: 4ebbf922b0c68e7b
---

# DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving

**Conference**: ICLR 2026
**arXiv**: [2601.01528](https://arxiv.org/abs/2601.01528)
**Code**: [https://drivinggen-bench.github.io/](https://drivinggen-bench.github.io/)
**Area**: Autonomous Driving / World Models
**Keywords**: Video world models, benchmark, driving scene generation, trajectory evaluation, temporal consistency

## TL;DR
DrivingGen introduces the first comprehensive benchmark for autonomous driving video world models, comprising a diverse evaluation dataset spanning weather/geography/time/complex scenarios and a four-dimensional metric framework (distribution, quality, temporal consistency, trajectory alignment). Evaluation of 14 SOTA models reveals a fundamental trade-off between general-purpose and driving-specific models.

## Background & Motivation

**State of the Field**: Video generative models as world models are rapidly advancing in autonomous driving, serving future scene prediction, scalable simulation, and synthetic data generation. Both general-purpose models (Kling, Sora, etc.) and driving-specific models (Vista, GEM, etc.) are iterating quickly.

**Limitations of Prior Work**: Existing evaluations suffer from four critical deficiencies: (a) general video metrics (FVD) overlook driving-specific imaging artifacts (e.g., PWM flickering); (b) the physical plausibility of trajectories is rarely quantified; (c) temporal consistency evaluation ignores agent-level anomalies (e.g., vehicles vanishing abruptly); (d) trajectory controllability is almost never assessed.

**Root Cause**: Existing datasets are heavily biased toward clear/daytime/single-city scenarios (nuScenes >80% clear daytime), making it impossible to evaluate model robustness under diverse real-world conditions. The absence of a unified benchmark renders cross-method comparisons unfair.

**Paper Goals**: Establish a unified evaluation framework covering data diversity, visual quality, physical plausibility, temporal consistency, and controllability.

**Starting Point**: Evaluate from both a visual perspective and a robotics perspective — visually appealing generation is insufficient; the underlying trajectory must also be physically plausible.

**Core Idea**: The first benchmark to comprehensively evaluate driving video world models across four dimensions from both visual and robotics perspectives.

## Method

### Overall Architecture
DrivingGen comprises two components: (1) a diverse evaluation dataset (400 samples, split into an open-domain track and an ego-conditioned track); (2) a four-dimensional evaluation metric suite (distribution, quality, temporal consistency, trajectory alignment) with 11 concrete metrics in total. Fourteen video generation models are evaluated under a unified protocol.

### Key Designs

1. **Diverse Evaluation Dataset**:

    - Function: Constructs a 400-sample dataset covering extreme weather (snow, fog, sandstorm, flood), multiple time periods (dawn, night), global geography (North America/Europe/Asia/Africa/Middle East, etc.), and complex interactions (dense traffic, pedestrian crossings, aggressive lane changes).
    - Mechanism: Two tracks — an open-domain track (testing generalization, data sourced from the internet) and an ego-conditioned track (testing trajectory controllability, data sourced from 5 open-source driving datasets). The proportion of clear/daytime samples is capped at <60%, with rare scenarios explicitly oversampled.
    - Design Motivation: Existing datasets contain >80% clear/daytime samples, leading to severely biased evaluations. 400 samples strike a balance between efficiency and coverage.

2. **Fréchet Trajectory Distance (FTD)**:

    - Function: A newly proposed distribution-level trajectory evaluation metric, analogous to FVD but operating in trajectory space.
    - Mechanism: Trajectories are mapped to a latent space using the encoder of Motion Transformer (MTR), and the Fréchet distance is computed in that space. This is the first transfer of the FVD paradigm to trajectory evaluation.
    - Design Motivation: Evaluating visual appearance via FVD alone is insufficient — a visually appealing video with physically implausible trajectories is dangerous in the context of autonomous driving.

3. **Agent Abnormal Disappearance Detection**:

    - Function: Detects non-physical disappearance of vehicles/pedestrians in generated videos (i.e., vanishing without leaving the field of view or being occluded).
    - Mechanism: YOLOv10 is used for detection and SAM2 for tracking each agent. When an agent disappears, a VLM (Cosmos-Reason1) judges whether the disappearance is physically justified (leaving the field of view or being occluded). The proportion of videos free of abnormal disappearances is reported.
    - Design Motivation: Such anomalies are prevalent in existing generative models yet have been entirely overlooked by prior evaluations.

4. **Adaptive Temporal Consistency Evaluation**:

    - Function: Prevents static videos from obtaining inflated consistency scores.
    - Mechanism: Optical flow is first used to estimate per-frame motion magnitude; adaptive temporal downsampling is then applied — videos with low motion are sampled more sparsely so that inter-frame displacement becomes comparable — followed by computing inter-frame similarity of DINOv3 features.
    - Design Motivation: Naively computing inter-frame similarity can be gamed by nearly static videos, which naturally achieve high consistency scores.

### Loss & Training
DrivingGen is an evaluation benchmark and does not involve model training. Trajectories are extracted from generated videos via a classical SLAM pipeline using SIFT + RANSAC + PnP.

## Key Experimental Results

### Main Results
Rankings of 14 models on the open-domain track (by average rank):

| Model | Params | FVD↓ | FTD↓ | Subjective Quality↑ | Objective Quality↑ | Trajectory Quality↑ | Video Consistency↑ | Agent Consistency↑ | Avg Rank |
|-------|--------|------|------|--------------------|--------------------|--------------------|--------------------|-------------------|----------|
| Kling 2.1 | - | 693.4 | 26.73 | 0.554 | 0.802 | 0.644 | 0.895 | 0.798 | **1** |
| Gen-3 Alpha | - | 801.0 | 93.50 | 0.546 | 0.838 | 0.654 | 0.890 | 0.817 | 2 |
| LTX-Video | 13B | 648.2 | 31.29 | 0.522 | 0.829 | 0.556 | 0.885 | 0.745 | 3 |
| Vista (driving-specific) | 2.5B | 675.7 | 54.66 | 0.434 | **0.847** | 0.603 | 0.857 | 0.636 | 6 |
| VaViM (driving-specific) | 1.2B | 1446.6 | 449.2 | 0.469 | **0.847** | 0.312 | **0.916** | 0.772 | 9 |

### Ablation Study

| Evaluation Dimension | Key Finding |
|----------------------|-------------|
| General vs. driving-specific models | General-purpose models achieve higher visual quality but weaker physical consistency; driving-specific models produce more realistic trajectories but inferior visual quality |
| Objective quality (PWM flicker) | Driving-specific models perform better on the IEEE P2020 standard (less flickering), attributed to training data containing real sensor characteristics |
| Agent disappearance | General-purpose models perform better (fewer abnormal disappearances), likely due to larger training data scale and better object persistence |
| Trajectory alignment (ego-conditioned) | Cosmos-Predict2 achieves the best performance (ADE=22.38), suggesting that embedded physics engines improve controllability |

### Key Findings
- **Core trade-off**: General-purpose models "look good but violate physics," while driving-specific models "align with physics but look poor" — the two directions have yet to converge.
- Kling 2.1 ranks first on both tracks, benefiting from commercial-scale data and training resources.
- FTD (trajectory distribution distance) and FVD (video distribution distance) rankings are inconsistent, confirming that visual quality and trajectory quality are independent dimensions.
- VaViM achieves the worst FVD yet the best agent consistency and agent disappearance scores, demonstrating that different evaluation dimensions genuinely reveal different model properties.

## Highlights & Insights
- **FTD fills a critical gap**: Transferring the FID/FVD paradigm to trajectory space by computing Fréchet distance over MTR encoder features provides a distribution-level, physics-grounded evaluation for driving video generation. This metric is directly reusable by future work.
- **Adaptive temporal consistency is an elegant anti-gaming design**: Optical flow-guided adaptive downsampling resolves the problem of static videos receiving inflated consistency scores — a trick applicable to any video generation task requiring temporal consistency evaluation.
- **The "four-dimensional evaluation" framework** offers a cognitive template: evaluation of any generative model should consider distribution, quality, consistency, and controllability, avoiding the misleading nature of any single metric.

## Limitations & Future Work
- The 400-sample dataset size is modest for statistical confidence.
- Trajectories are extracted from generated videos via SLAM, introducing additional noise — ideally, trajectories should be obtained directly from the model.
- Downstream task evaluation is absent (e.g., closed-loop planning performance after training a planner on generated videos).
- Fair efficiency comparisons cannot be made for commercial closed-source models (Kling, Gen-3).

## Related Work & Insights
- **vs. VBench**: VBench is a general video evaluation framework lacking trajectory assessment and driving-specific metrics; DrivingGen designs a complete evaluation system tailored to driving scenarios.
- **vs. WorldScore (Duan et al., 2025)**: WorldScore focuses on general scene consistency, whereas DrivingGen additionally covers agent-level consistency, trajectory quality, and controllability.
- **vs. DrivingDojo/Driverse**: These works cover only a subset of evaluation dimensions; DrivingGen is the first benchmark to provide comprehensive coverage.

## Rating
- Novelty: ⭐⭐⭐⭐ The evaluation system design is novel (FTD, adaptive consistency, agent disappearance detection), though as a benchmark paper it lacks algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of 14 models including commercial and open-source systems, across multiple tracks and dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis is clear, metric designs are well-motivated, and results are presented systematically.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in evaluation of driving video world models and reveals the key insight that "visually appealing ≠ physically plausible."

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](../../CVPR2026/autonomous_driving/vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](../../ICCV2025/autonomous_driving/recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)
- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](../../CVPR2026/autonomous_driving/dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](../../CVPR2026/autonomous_driving/an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)

<!-- RELATED:END -->
