---
title: >-
  [Paper Note] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction
description: >-
  [CVPR 2026][Autonomous Driving][trajectory prediction] Den-TP is a data-centric framework that addresses the long-tail density imbalance in trajectory prediction datasets through density-aware data curation and evaluatio…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "trajectory prediction"
  - "data-centric"
  - "density balancing"
  - "submodular optimization"
  - "long-tail distribution"
date: 2026-05-08
content_hash: 338bb28a547c8bbb
---

# Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction

**Conference**: CVPR 2026
**arXiv**: [2409.17385](https://arxiv.org/abs/2409.17385)  
**Code**: None  
**Area**: Autonomous Driving / Trajectory Prediction
**Keywords**: trajectory prediction, data-centric, density balancing, submodular optimization, long-tail distribution

## TL;DR

Den-TP is a data-centric framework that addresses the long-tail density imbalance in trajectory prediction datasets through density-aware data curation and evaluation protocols. Using only 50% of the training data, it maintains overall performance while significantly improving robustness in high-density scenarios.

## Background & Motivation

Trajectory prediction is a critical task in autonomous driving, and Transformer- and GNN-based methods have achieved strong performance on standard benchmarks. However, examining the problem from a data perspective reveals a serious yet overlooked issue: existing trajectory prediction datasets exhibit significant long-tail density imbalance.

In mainstream datasets such as Argoverse 1 and 2, low-density scenes (with few interacting agents) dominate the sample distribution, while safety-critical high-density scenes (involving 60+ agents in complex multi-body interactions) account for less than 2.5% of all samples. This leads to two core limitations: (1) training signals are dominated by low-density samples, causing severe under-training on high-density scenarios; and (2) standard evaluation protocols report average errors over the entire dataset, masking performance degradation in high-density scenes—models may appear competitive on aggregate metrics while catastrophically failing in the most dangerous dense interaction scenarios.

The root cause of this problem is that high-density scenes are the most safety-critical (involving complex multi-body interactions where prediction errors can directly threaten driving safety), yet they carry the least influence during both training and evaluation. Existing countermeasures—resampling, reweighting, and data augmentation—alter sampling frequency but do not reshape the data distribution in a principled manner.

**Core Idea**: Treat scene density as a conditioning variable, construct a compact yet balanced subset via a density-aware partition-and-selection strategy, apply gradient-based submodular optimization to select representative samples within each density bin, and enforce a high-density-biased dynamic budget allocation across bins.

## Method

### Overall Architecture

Den-TP consists of two main stages: an extraction stage and a selection stage. In the extraction stage, stable gradient estimates are obtained via lightweight pre-training, and the dataset is partitioned into density bins according to agent count. In the selection stage, submodular optimization selects representative samples within each bin, while biased sampling explicitly upweights rare high-density scenes across bins. The output is a compact, density-balanced training subset.

### Key Designs

1. **Data Partitioning**:

    - **Function**: Partition the dataset into interpretable subsets according to scene complexity.
    - **Mechanism**: Compute the density level $\rho(S_j)$ of each sample (based on the number of agents in the scene), and partition the dataset into $K$ disjoint subsets $\mathcal{D}_k$ using a fixed interval $\tau$, where $S \in \mathcal{D}_k$ if and only if $\rho(S) \in [\rho_{\min}+(k-1)\tau,\, \rho_{\min}+k\tau)$. Although agent count does not fully capture scene complexity, it serves as a dataset-agnostic proxy that enables consistent cross-dataset analysis.
    - **Design Motivation**: Without density partitioning, gradient updates are dominated by the large number of low-density samples, leading to systematic under-training on high-density scenes. Explicit partitioning is a prerequisite for the subsequent balanced selection.

2. **Gradient Extraction & Submodular Selection**:

    - **Function**: Select the most representative and least redundant samples within each density bin.
    - **Mechanism**: For each sample, gradient features are extracted via backpropagation as $\mathbf{G} = \nabla_{\hat{\mathbf{Y}}} \mathcal{L}$, then fused with decoder embeddings via element-wise multiplication to yield $\mathbf{g} = \phi(\mathbf{G}) \odot \phi(\mathbf{E})$. A submodular scoring function is defined as $P(S_j) = \sum_{S_i \in \mathcal{C}_k} \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\|\|\mathbf{g}_j\|} - \sum_{S_i \in \mathcal{D}_k \setminus \mathcal{C}_k} \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\|\|\mathbf{g}_j\|}$, and samples minimizing $P$ are greedily selected—those that are least redundant with the already-selected set while most representative of the unselected portion.
    - **Design Motivation**: Gradient features capture both loss sensitivity and decoder representation information, reflecting a sample's actual contribution to training more faithfully than clustering in pure feature space.

3. **Dynamic Allocation**:

    - **Function**: Fairly distribute the selection budget across density bins while prioritizing high-density scenes.
    - **Mechanism**: Given a total budget $B = \lfloor \alpha |\mathcal{D}| \rfloor$, bins are processed in reverse order from highest to lowest density. The budget allocated to bin $\mathcal{D}_k$ is $n_k = \min(|\mathcal{D}_k|, \lfloor B/k \rfloor)$. If a bin contains fewer samples than its budget, all samples are retained without gradient-based selection. Remaining budget is then redistributed to lower-density bins.
    - **Design Motivation**: High-density bins are naturally scarce; reverse-order processing ensures they are not eliminated early by insufficient budgets. A key finding is that capabilities learned from high-density scenes transfer to low-density scenes, whereas the reverse transfer is weak.

### Loss & Training

The training loss used for gradient extraction is $\mathcal{L} = \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{cls}}$, where $\mathcal{L}_{\text{reg}}$ is a negative log-likelihood regression loss over the best-matching predicted mode, and $\mathcal{L}_{\text{cls}}$ is a cross-entropy loss for optimizing predicted mode probabilities. Gradient extraction requires a lightweight pre-training step to obtain stable gradient estimates.

## Key Experimental Results

### Main Results

| Dataset | Method | Retention Ratio | minADE↓ | minFDE↓ | MR↓ |
|--------|------|------|----------|------|------|
| Argoverse 1 | Full (HiVT-64) | 100% | 0.695 | 1.037 | 0.109 |
| Argoverse 1 | Random | 50% | 0.750 | 1.175 | 0.137 |
| Argoverse 1 | Herding | 50% | 0.728 | 1.107 | 0.126 |
| Argoverse 1 | **Den-TP** | **50%** | **0.706** | **1.074** | **0.110** |
| Argoverse 1 | Full (HPNet) | 100% | 0.647 | 0.871 | 0.070 |
| Argoverse 1 | **Den-TP** (HPNet) | **50%** | **0.661** | **0.913** | **0.074** |

### Ablation Study

| Strategy | Data Size | minADE↓ | minFDE↓ | MR↓ | Notes |
|------|---------|------|------|------|------|
| Augmenting | 220k | 0.718 | 1.106 | 0.115 | Duplicate high-density samples |
| Weighting | 190k | 0.715 | 1.108 | 0.114 | Reweight high-density |
| Epoch-wise | 95k | 0.752 | 1.189 | 0.130 | Resample each epoch |
| High-density+Random | 95k | 0.724 | 1.111 | 0.117 | Retain high-density + random fill |
| **Den-TP** | **95k** | **0.706** | **1.074** | **0.110** | Density-balanced selection |

### Key Findings

- **50% of data suffices to match or exceed full-data performance**: Den-TP with HiVT-64 at 50% data (minADE 0.706) surpasses the full-data baseline (0.695) on the MR metric, with only a 1.6% gap in minADE.
- **Transferability of high-density capabilities**: Interaction patterns learned from dense scenes generalize to simpler scenarios, while the reverse transfer is weak—providing the theoretical justification for the density-priority allocation strategy.
- **Cross-model generalization**: Subsets selected using HiVT-64 retain their advantage when trained with HiVT-128 and HPNet, indicating that the selection strategy is not architecture-dependent.
- **Naïve strategies are insufficient**: Reweighting and data augmentation increase redundancy rather than coverage, and epoch-wise resampling introduces training instability.

## Highlights & Insights

- **The data-centric perspective on trajectory prediction** is the paper's most significant contribution—prior work in this domain has been almost entirely model-centric. This paper is the first to systematically expose the severe impact of density imbalance on both training and evaluation. A notable insight is that using simple agent counts as a density proxy proves sufficient for meaningful analysis.
- **The density-conditioned evaluation protocol** is equally valuable: standard aggregate metrics obscure long-tail failure modes, and reporting performance stratified by density bin exposes genuine safety risks.
- **The combination of submodular optimization and gradient features** is transferable to other domains with long-tail distributions, such as small-object detection or rare lesion segmentation in medical imaging.

## Limitations & Future Work

- Using agent count as a density proxy is overly simplistic; it cannot distinguish between interaction patterns of equal density but different complexity (e.g., 20 agents moving independently vs. 20 agents converging at a bottleneck).
- Gradient extraction requires a pre-training step, introducing additional computational overhead.
- Validation is limited to Argoverse 1 and 2; generalization to other datasets such as nuScenes and INTERACTION remains untested.
- The design of $\lfloor B/k \rfloor$ in the dynamic allocation strategy lacks theoretical proof of optimality.

## Related Work & Insights

- **vs. Random Selection**: At a 50% budget, Den-TP achieves a minADE of 0.706 versus 0.750 for random selection—a 5.9% improvement.
- **vs. Herding**: Herding performs greedy selection based on feature means without accounting for density distribution; Den-TP outperforms it across all metrics.
- **vs. K-Means Clustering**: Trajectory-feature-based clustering ignores gradient information and density balance, yielding performance between random selection and Den-TP.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic data-centric analysis of density imbalance in trajectory prediction; the density-conditioned evaluation protocol is a methodological contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, models, strategies, and retention ratios.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem formulation, rich visualizations (density distribution plots, performance curves), and rigorous algorithmic description.
- **Value**: ⭐⭐⭐⭐ Reveals a neglected data-level issue in the field with high practical utility—directly applicable to reducing training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[CVPR 2026\] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction](recover_to_predict_progressive_retrospective_learning_for_variable-length_trajec.md)
- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)

</div>

<!-- RELATED:END -->
