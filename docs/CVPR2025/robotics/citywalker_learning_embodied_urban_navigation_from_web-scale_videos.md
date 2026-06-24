---
title: >-
  [Paper Note] CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos
description: >-
  [CVPR 2025][Robotics][Urban Navigation] Utilizing over 2,000 hours of city walking and driving videos from the internet, action labels are automatically extracted via Visual Odometry (VO) for large-scale imitation learning. This trains embodied agents capable of navigating complex, dynamic urban environments, achieving a 77.3% success rate in real-world deployment, significantly outperforming existing methods.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Urban Navigation"
  - "Imitation Learning"
  - "Visual Odometry"
  - "Web Video Training"
  - "Embodied AI"
date: 2026-05-08
content_hash: 4d8659634113168e
---

# CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos

**Conference**: CVPR 2025  
**arXiv**: [2411.17820](https://arxiv.org/abs/2411.17820)  
**Code**: [https://ai4ce.github.io/CityWalker/](https://ai4ce.github.io/CityWalker/)  
**Area**: Embodied Navigation / Reinforcement Learning  
**Keywords**: Urban Navigation, Imitation Learning, Visual Odometry, Web Video Training, Embodied AI

## TL;DR
Utilizing over 2,000 hours of city walking and driving videos from the internet, action labels are automatically extracted via Visual Odometry (VO) for large-scale imitation learning. This trains embodied agents capable of navigating complex, dynamic urban environments, achieving a 77.3% success rate in real-world deployment, significantly outperforming existing methods.

## Background & Motivation

**Background**: While visual navigation has achieved near-perfect performance in indoor simulators (point-goal navigation is considered a "solved" problem), it remains an unsolved challenge in urban outdoor scenarios. Existing methods primarily operate in static or simple environments.

**Limitations of Prior Work**: Urban navigation faces complex constraints such as pedestrian interaction, traffic lights, obstacle avoidance, and sidewalk norms, which are difficult to model in simulators. Collecting expert data via teleoperation is expensive, small-scale, and lacks diversity. Some works rely on Large Language Models/VLMs to generate action labels, which is costly and difficult to scale.

**Key Challenge**: Learning to navigate in real-world urban environments requires large-scale and diverse training data, which conventional methods (teleoperation/simulators) struggle to provide. Although massive amounts of urban walking videos exist on the internet, they lack action labels.

**Goal**: How to automatically extract action supervision signals from unlabeled web videos to enable large-scale imitation learning.

**Key Insight**: The authors observe that although off-the-shelf Visual Odometry (VO) tools yield imprecise global trajectories, their relative poses within short-time windows are reliable enough to serve as action pseudo-labels for imitation learning.

**Core Idea**: Using VO to extract action pseudo-labels from web city-walking videos, training an urban navigation policy via large-scale imitation learning.

## Method

### Overall Architecture
Input: RGB images from the past $k=5$ frames + trajectory coordinates of the past $k$ steps + target position. Images are encoded using a frozen DINOv2 encoder to extract features, while coordinates are embedded via a learnable encoder. A Transformer processes the sequence of temporal tokens, and the output is fed into an action head to predict the future 5 steps of actions (2D displacement) and an arrival head to predict whether the sub-goal has been reached. The training data comes from over 2,000 hours of web city videos.

### Key Designs

1. **VO-based Action Label Extraction**:

    - **Function**: Automatically generating action supervision signals from unlabeled videos.
    - **Mechanism**: Using DPVO to extract relative inter-frame poses from videos as action labels. Although VO suffers from accumulated global drift and scale ambiguity, the model only needs to predict relative actions within a short-time window (5 steps), making the impact of drift minimal. Scale ambiguity is resolved by normalizing each trajectory by its average step size—this simultaneously addresses the step-size inconsistency across different video sources (walking vs. driving) and different robot platforms.
    - **Design Motivation**: Compared to VLM prompting schemes (such as LeLaN), the VO approach is fully parallelizable, making the processing cost of 2,000 hours of video almost negligible.

2. **Feature Hallucination Loss**:

    - **Function**: Auxiliary training objective that forces the model to learn to predict future visual features.
    - **Mechanism**: Computing the MSE loss between the image tokens output by the Transformer and the ground-truth features of future frames. This guides the model to generate informative tokens that simulate future observations, indirectly improving action prediction quality. Note: during zero-shot inference, this loss actually turns out to be detrimental (since the model tends to predict future frames from a human perspective), but this issue disappears after fine-tuning.
    - **Design Motivation**: Inspired by feature learning, predicting future features forces the model to model environmental dynamics.

3. **Cross-Domain and Cross-Embodiment Training**:

    - **Function**: Improving generalization capabilities by combining walking and driving videos.
    - **Mechanism**: Although driving videos come from different domains and embodiments, they can be unified into the same abstract action space through step-size normalization. Experiments show that using only 250 hours of mixed data yields performance close to that of 1,000 hours of pure walking data, demonstrating the complementary value of cross-domain data.
    - **Design Motivation**: To fully leverage the more abundant driving video resources available on the internet.

### Loss & Training
The total loss is a weighted sum of four components: L1 action loss + orientation loss (negative cosine similarity between predicted and GT action) + BCE loss for arrival state + MSE loss for feature hallucination. The weight for orientation loss is set to 5.0, and the others to 1.0. Pre-training uses 2,000 hours of web videos, while fine-tuning uses 6 hours of teleoperated data (New York City scenes).

## Key Experimental Results

### Main Results

| Method | MAOE↓ (Scene Avg.) | Real Deployment Success Rate | Forward | Left Turn | Right Turn |
|------|-----------------|--------------|------|------|------|
| ViNT (zero-shot) | 17.5° | 37.7% | 62.5% | 0.0% | 50.0% |
| ViNT (fine-tuned) | 16.5° | 57.1% | 100% | 25.0% | 25.0% |
| NoMaD (fine-tuned) | 19.1° | 42.9% | 75.0% | 16.7% | 28.6% |
| CityWalker (zero-shot) | 16.5° | - | - | - | - |
| **CityWalker (fine-tuned)** | **15.2°** | **77.3%** | **100%** | **62.5%** | **66.7%** |

### Ablation Study

| Configuration | MAOE (Scene Avg.) |
|------|---------------|
| Baseline (no ori loss / no feat hall / no fine-tuning) | 17.03° |
| + Orientation loss | 17.00° |
| + Orientation loss + Feature hallucination | 17.02° |
| + Fine-tuning | 15.23° |
| + Orientation loss + Fine-tuning | 15.21° |
| + All components | **15.16°** |

### Key Findings
- Fine-tuning is the largest source of performance improvement (17.03° $\rightarrow$ 15.23°), whereas the marginal contributions of orientation loss and feature hallucination are relatively small.
- Data scaling effect is significant: with more than 1,000 hours of training data, the zero-shot model outperforms the fine-tuned ViNT.
- Cross-domain training (mixed walking and driving) yields remarkable results: 250 hours of mixed data is roughly equivalent to 1,000 hours of pure walking data.
- CityWalker far outperforms baselines in turning scenarios (62.5% for left turns and 66.7% for right turns, compared to at most 25–50% for baselines), demonstrating that large-scale data helps the model learn complex maneuvering strategies.

## Highlights & Insights
- **Replacing VLMs with VO for Labeling**: Utilizing a simple and efficient VO tool instead of expensive VLM prompting to obtain action labels is a highly practical engineering decision. The processing cost for 2,000 hours of video is almost zero.
- **Scaling Law of Data**: Clearly demonstrates the trend of navigation performance scale with data volume, with 1,000 hours representing a key inflection point. This finding can guide future data collection strategies.
- **Step-Size Normalization Unifying Heterogeneous Data**: A simple normalization trick eliminates the discrepancy across domains (walking/driving) and embodiments (humans/quadrupedal robots), which is elegant and practical.

## Limitations & Future Work
- Sensitive to iPhone GPS location noise; actual deployment relies heavily on GPS accuracy.
- Fine-tuning still requires teleoperated data (6 hours); fully zero-shot real-world deployment is not yet achieved.
- Tested only in New York City; cross-city generalization has not been validated.
- Weak performance in "detour" scenarios, as such data is scarce in the training videos.
- Does not incorporate semantic maps or high-level planning, focusing only on local navigation between waypoints.

## Related Work & Insights
- **vs ViNT**: ViNT is a foundational model for visual navigation, primarily trained in suburban/off-road environments, yielding poor generalization in urban environments. CityWalker focuses on urban scenes, leveraging web videos to acquire a larger and more diverse training set.
- **vs NoMaD**: NoMaD uses a diffusion model for action prediction but performs poorly in complex urban scenes (only 42.9% success rate).
- **vs LeLaN**: Concurrent work that relies on VLM prompting and pre-trained navigation models to generate labels, which is expensive and difficult to scale.
- Directly applicable as a reference for applications such as autonomous delivery robots and last-mile autonomous driving navigation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of extracting action labels from web videos using VO is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Accomplished both real-world deployment experiments and scaling law analysis, though the test scale remains somewhat small.
- Writing Quality: ⭐⭐⭐⭐ Question-driven writing style, clear and logical.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for leveraging web videos to train urban navigation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](../../ICLR2026/robotics/urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[CVPR 2025\] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)
- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
