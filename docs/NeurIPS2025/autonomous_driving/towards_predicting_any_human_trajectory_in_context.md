---
title: >-
  [Paper Note] Towards Predicting Any Human Trajectory in Context
description: >-
  [NeurIPS 2025][Autonomous Driving][pedestrian trajectory prediction] This paper proposes TrajICL, an in-context learning (ICL) framework for pedestrian trajectory prediction that achieves cross-scene adaptive prediction without fine-tuning through spatiotemporal similarity-based example selection and prediction-guided example selection, surpassing even fine-tuned baselines.
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "pedestrian trajectory prediction"
  - "in-context learning"
  - "cross-domain adaptation"
  - "synthetic data"
  - "example selection"
date: 2026-05-08
content_hash: c916ba7fed5de58f
---

# Towards Predicting Any Human Trajectory in Context

**Conference**: NeurIPS 2025
**arXiv**: [2506.00871](https://arxiv.org/abs/2506.00871)  
**Code**: Available (Project Page)  
**Area**: Autonomous Driving
**Keywords**: pedestrian trajectory prediction, in-context learning, cross-domain adaptation, synthetic data, example selection

## TL;DR

This paper proposes TrajICL, an in-context learning (ICL) framework for pedestrian trajectory prediction that achieves cross-scene adaptive prediction without fine-tuning through spatiotemporal similarity-based example selection and prediction-guided example selection, surpassing even fine-tuned baselines.

## Background & Motivation

Pedestrian trajectory prediction is a critical task in autonomous driving, robot navigation, and surveillance systems. While existing methods perform well in specific environments, they face two core challenges:

**Poor scene adaptability**: Most models are trained and evaluated in specific environments/domains and struggle to generalize to new scenes (different map layouts, camera positions, sensor types, etc.).

**High fine-tuning cost**: Traditional adaptation methods require on-device backpropagation, which imposes prohibitive computational and memory costs on edge devices; moreover, separate models must be maintained for different scenes.

In-Context Learning (ICL) offers an attractive alternative: by leveraging a small number of examples through forward passes alone, models can adapt to new tasks without updating their weights. However, applying ICL to trajectory prediction presents three challenges:

- **Randomly selected examples provide almost no ICL benefit**: If examples do not match the query in spatial location or motion pattern, the model cannot effectively exploit the context.
- **Example selection based solely on historical trajectories is suboptimal**: Short historical trajectories fail to capture long-term intent, and pedestrian motion is inherently multimodal (similar histories may lead to divergent futures).
- **Insufficient scene diversity in training data**: The limited scenes in existing real-world datasets constrain ICL generalization.

## Method

### Overall Architecture

TrajICL consists of an embedding layer, a trajectory encoder, a context-aware trajectory predictor, and a multimodal decoder. The model is trained exclusively on a large-scale synthetic dataset (MOTSynth) and adapts to target domains at inference time by selecting relevant in-scene examples, without any parameter updates.

### Key Designs

1. **Spatiotemporal Example Selection (STES)**: A composite similarity score $S(X_1, \tilde{X}_1^i) = \sigma(S_p) + \sigma(S_v)$ is defined, where $S_p = \frac{1}{1+d_p}$ (positional MSE similarity) and $S_v = \frac{1}{1+d_v}$ (velocity MSE similarity), with $\sigma$ denoting a normalization function. The Top-M most similar examples are selected. The core motivation is that, in trajectory prediction, historical trajectories that are spatially close and share similar motion patterns provide the most informative context.

2. **Prediction-Guided Example Selection (PG-ES)**: A two-stage selection strategy. In the first stage, STES is applied to select examples and predict future trajectories $\hat{Y}_1$. In the second stage, the historical and predicted future trajectories are concatenated to recompute similarity $S([X_1, \hat{Y}_1^k], [\tilde{X}_1^i, \tilde{Y}_1^i])$, taking the minimum similarity across all $K$ predictions. The motivation is to incorporate predicted future trajectories to account for long-term dynamics, thereby retrieving more relevant examples.

3. **Relative Context Positional Encoding (RCPE) + Similarity-Ranked Positional Encoding (SRPE)**: RCPE uses an MLP to encode the relative position $(x_{rel}, y_{rel})$ of each example agent with respect to the target agent; SRPE uses sinusoidal positional encoding to represent the similarity rank of each example relative to the target. Both encodings are added to the example features, enabling the predictor to be aware of both the relative relevance of each example and its spatial origin.

4. **Synthetic Data Training**: The model is trained on MOTSynth (700+ 90-second videos with diverse outdoor environments), using 424 scenes for training and 107 scenes for evaluation, addressing the limited scene diversity in real-world datasets.

### Loss & Training

Two-stage training:
- **VTP stage**: Standard trajectory prediction training, 100 epochs, AdamW with cosine annealing.
- **ICL stage**: Training with STES-selected examples, 400 epochs.
- **Loss function**: MSE with Winner-Take-All (optimizing only the most accurate prediction): $\mathcal{L} = \min_k \|\hat{Y}_1^{(k)} - Y_1\|^2$

## Key Experimental Results

### Main Results

| Method | Training-free | MOTSynth | JRDB-Image | WildTrack | SDD | JTA |
|--------|--------------|----------|------------|-----------|-----|-----|
| Social-Transmotion | ✓ | 17.6/23.0 | 2.88/3.32 | 24.7/36.3 | 10.2/18.9 | 1.18/1.97 |
| +Full FT | ✗ | 16.0/20.9 | 2.56/2.87 | 22.9/34.5 | 7.96/13.6 | 0.52/0.76 |
| +LoRA (r=64) | ✗ | 16.8/22.2 | 2.65/2.98 | 23.6/35.6 | 9.11/16.8 | 0.60/0.93 |
| **+TrajICL** | **✓** | **15.3/17.5** | **2.61/2.68** | **21.1/28.3** | **8.40/14.8** | 0.59/0.85 |
| Δ vs. Baseline | | -14.2%/-23.9% | -7.6%/-19.2% | -14.6%/-22.0% | -17.6%/-21.7% | -41.5%/-56.9% |

minADE/minFDE (K=20). TrajICL surpasses full fine-tuning on minFDE in 4 out of 6 datasets.

### Ablation Study

| Spatial | Temporal | Pred.-Guided | MOTSynth | WildTrack | JTA | SDD |
|---------|----------|--------------|----------|-----------|-----|-----|
| | | | 21.8 | 35.5 | 1.08 | 16.4 |
| ✓ | ✓ | | 18.0 (-17.4%) | 28.9 (-18.6%) | 0.85 (-21.3%) | 15.1 (-7.9%) |
| ✓ | ✓ | ✓ | **17.5 (-19.7%)** | **28.3 (-20.3%)** | **0.85 (-21.3%)** | **14.8 (-9.8%)** |

Ablation of RCPE+SRPE: joint use yields the most significant improvements on cross-domain datasets (WildTrack minFDE: 31.0→28.3; JRDB-World: 0.23→0.21).

### Key Findings

- Randomly selected examples provide almost no ICL benefit (adding more examples without STES does not improve performance on MOTSynth), whereas STES yields consistent improvement as the number of examples increases.
- PG-ES further reduces minFDE by 6.6%–8.6% over STES across four datasets.
- Under extreme conditions with only 10% labeled data, TrajICL still outperforms the best fine-tuning method.
- The relative importance of spatial and temporal similarity varies by dataset: temporal similarity is more important in JTA/SDD, while spatial similarity dominates in MOTSynth.
- The TrajICL framework generalizes across different backbones (10–12% improvement observed on ForecastMAE as well).

## Highlights & Insights

- **First successful application of ICL to trajectory prediction**: The work demonstrates that appropriate example selection strategies can endow trajectory prediction models with genuine ICL capability.
- **Synthetic-data training, real-data inference**: Decoupling the training and deployment environments is of significant practical value.
- **Training-free adaptation surpasses fine-tuning**: TrajICL outperforms full fine-tuning on minFDE in 4 out of 6 datasets, challenging the assumption that fine-tuning is the optimal adaptation strategy.
- Qualitative analysis shows that TrajICL can perceive 3D structures (e.g., elevators), respect map constraints (e.g., avoiding fences), and capture behavioral tendencies (e.g., following sidewalks).

## Limitations & Future Work

- Increasing the number of in-context examples linearly increases inference computational cost.
- Improvements on world-coordinate datasets (JRDB-World, JTA) are relatively modest.
- The model relies solely on trajectory data as input, without incorporating map or scene information.
- A synthetic-to-real domain gap remains.

## Related Work & Insights

TrajICL introduces a new paradigm for on-device trajectory prediction: training a general-purpose model on synthetic data, then adapting it at deployment time using only a small number of in-scene observations as examples, with no backpropagation required. This approach is also generalizable to other time-series forecasting tasks (vehicle trajectories, robot trajectories, etc.). The idea behind Prediction-Guided Example Selection (PG-ES)—using the model's preliminary predictions to refine example retrieval—can similarly be applied to other ICL scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel application of ICL to trajectory prediction; PG-ES is a creative contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 datasets, comparisons with multiple fine-tuning methods, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated)
- Value: ⭐⭐⭐⭐ (Practically significant for edge deployment scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Certified Human Trajectory Prediction](../../CVPR2025/autonomous_driving/certified_human_trajectory_prediction.md)
- [\[NeurIPS 2025\] OpenBox: Annotate Any Bounding Boxes in 3D](openbox_annotate_any_bounding_boxes_in_3d.md)
- [\[NeurIPS 2025\] LabelAny3D: Label Any Object 3D in the Wild](labelany3d_label_any_object_3d_in_the_wild.md)
- [\[ECCV 2024\] Adaptive Human Trajectory Prediction via Latent Corridors](../../ECCV2024/autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)
- [\[NeurIPS 2025\] Aha: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead](aha_predicting_what_matters_next_online_highlight_detection.md)

</div>

<!-- RELATED:END -->
