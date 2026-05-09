---
title: >-
  [Paper Note] Learnability-Driven Submodular Optimization for Active Roadside 3D Detection
description: >-
  [CVPR2026][Autonomous Driving][Active Learning] This paper proposes LH3D, an active learning framework that employs a three-stage hierarchical submodular optimization strategy—depth confidence → semantic balancing → geometric diversity—to suppress the selection of inherently ambiguous samples in roadside monocular 3D detection. With only 20% of the annotation budget, LH3D significantly outperforms conventional uncertainty- and diversity-based AL methods.
tags:
  - CVPR2026
  - Autonomous Driving
  - Active Learning
  - Roadside Perception
  - Monocular 3D Detection
  - BEV
  - Submodular Optimization
  - Learnability
  - Data Selection
date: 2026-05-08
content_hash: 812d6a0820856e33
---

# Learnability-Driven Submodular Optimization for Active Roadside 3D Detection

**Conference**: CVPR2026
**arXiv**: [2601.01695](https://arxiv.org/abs/2601.01695)
**Code**: Not released
**Area**: Autonomous Driving
**Keywords**: Active Learning, Roadside Perception, Monocular 3D Detection, BEV, Submodular Optimization, Learnability, Data Selection

## TL;DR

This paper proposes LH3D, an active learning framework that employs a three-stage hierarchical submodular optimization strategy—depth confidence → semantic balancing → geometric diversity—to suppress the selection of inherently ambiguous samples in roadside monocular 3D detection. With only 20% of the annotation budget, LH3D significantly outperforms conventional uncertainty- and diversity-based AL methods.

## Background & Motivation

1. **Roadside perception is a critical complement to L5 autonomous driving**: On-vehicle sensors suffer from occlusion, blind spots at intersections, and limited long-range perception; fixed roadside cameras can substantially extend field-of-view coverage.
2. **Annotation cost is a deployment bottleneck**: Roadside scenes contain densely packed objects, numerous small distant targets, and severe occlusion. Annotating 3D attributes per frame requires inferring depth and occlusion relationships, making manual annotation extremely costly.
3. **Traditional active learning fails in roadside scenarios**: Uncertainty-based AL methods prioritize high-uncertainty samples, which in roadside settings coincide with "inherently ambiguous samples"—distant blurry or heavily occluded objects whose 3D properties cannot be reliably annotated from a single viewpoint.
4. **Discovery of inherently ambiguous samples**: Even human experts struggle to accurately annotate 3D attributes of distant roadside objects without paired vehicle-side data, revealing a fundamental learnability problem.
5. **Human study validation**: Under identical annotation budgets and class distributions, detectors trained on ambiguous samples achieve substantially lower AP on Vehicle and Pedestrian than those trained on learnable samples, confirming that ambiguous samples provide weaker supervisory signals.
6. **Need for a new paradigm**: The core objective of active learning should shift from "select the uncertain" to "select the learnable," reorienting from uncertainty maximization to learnability maximization.

## Method

### Overall Architecture: LH3D (Learnable Hierarchical 3D)

Built upon an LSS-style BEV detector (e.g., BEVHeight), LH3D employs a three-stage hierarchical submodular selector. Each stage is modeled as a concave-over-modular submodular function, supporting greedy optimization with a $(1-1/e)$ approximation guarantee.

**Unified objective**:

$$F(S_q) = [\Phi_A(S_q) - \Phi_A(\mathcal{U})] + [\Phi_B(\mathcal{L}_q \cup S_q) - \Phi_B(\mathcal{L}_q)] + [\Phi_C(\mathcal{L}_q \cup S_q) - \Phi_C(\mathcal{L}_q)]$$

### Key Designs

#### Stage 1: Depth-Confident Sample Selection

- The normalized Shannon entropy $h_i$ is computed over the predicted depth distribution of each image and mapped to a confidence weight $r_i = e^{-\tau h_i}$.
- The argmax depth-bin histogram $m_i$ of each image is computed and aggregated into a depth coverage vector via weighted summation.
- Submodular objective: $\Phi_A(S) = \sum_{d=1}^{D} \log(\epsilon + Z_d(S))$; the logarithm ensures balanced coverage across near, mid, and far distance bins.
- Effect: Filters out ambiguous scenes with unreliable depth estimates and prioritizes samples with confident depth predictions.

#### Stage 2: Rare-Common Class Balancing

- The current detector predicts per-class object counts for each image, normalized to a class distribution $p_i(c)$.
- A semantic diversity entropy $\delta_i$ is computed for each image and mapped to a weight $\alpha_i = 1 + \gamma \delta_i$.
- Submodular objective: $\Phi_B(S) = \sum_{c \in \mathcal{C}} \log(\epsilon + N_c(S))$; the logarithmic saturation mechanism causes marginal gains to diminish sharply for already well-covered classes.
- Effect: Prevents Vehicle-dominated long-tail bias and increases exposure of rare categories such as Pedestrian and Cyclist.

#### Stage 3: Geometric Variant Selection

- Gaussian models $\mathcal{N}(\mu_c, \Sigma_c)$ are fitted to BEV centers and heights of each category from the labeled set.
- The negative log-likelihood (NLL) under the Gaussian serves as a geometric novelty score $s_{i,c}$ for candidate images.
- Submodular objective: $\Phi_C(S) = \sum_{c \in \mathcal{C}} \log(\epsilon + U_c(S))$.
- Effect: Encourages selection of samples with moderate deviation from learned spatial patterns while filtering extreme outliers.

### Loss & Training

- The detector is trained with the standard BEV pipeline detection loss.
- In each AL round, training continues from the previous checkpoint for 5 epochs using AdamW (lr=2e-4), batch size 8.
- The initial labeled set contains 500 images; 100 images are selected per round; total annotation budget covers 32K objects.

## Key Experimental Results

### Main Results

#### DAIR-V2X-I Validation Set (BEVHeight backbone, 20% budget, Hard)

| Method | Vehicle | Pedestrian | Cyclist | Average |
|--------|---------|------------|---------|---------|
| RANDOM | 51.41 | 13.42 | 39.38 | 34.74 |
| ENTROPY | 54.51 | 16.72 | 38.57 | 36.53 |
| BADGE | 51.33 | 14.98 | 35.35 | 33.89 |
| PPAL | 51.44 | 18.07 | 39.71 | 36.41 |
| HUA | 51.48 | 13.33 | 34.48 | 33.10 |
| **LH3D (Ours)** | **56.03** | **17.67** | **41.79** | **38.50** |

LH3D achieves +2.09 average AP over PPAL and +5.40 over HUA under the Hard setting.

#### Rope3D Validation Set (BEVHeight backbone, 20% budget, Hard)

| Method | Vehicle | Pedestrian | Cyclist | Average |
|--------|---------|------------|---------|---------|
| RANDOM | 19.65 | 1.50 | 14.80 | 11.99 |
| PPAL | 24.12 | 1.73 | 14.80 | 13.55 |
| **LH3D (Ours)** | **26.12** | **2.04** | **16.69** | **14.95** |

### Ablation Study

#### Stage Ordering (BEVHeight, Hard)

| Order | Car | Ped | Cyc | Avg |
|-------|-----|-----|-----|-----|
| DC→GV→SB | 50.62 | 16.83 | 37.10 | 34.85 |
| SB→DC→GV | 55.90 | 12.46 | 35.95 | 34.77 |
| GV→DC→SB | 40.04 | 13.02 | 32.67 | 28.58 |
| **DC→SB→GV (Ours)** | **56.03** | **17.67** | **41.79** | **38.50** |

Depth-Confident selection must appear first, as depth is the foundation of BEV perception. Placing Geometric Variant selection first yields the worst performance, since geometric selection without prior depth filtering introduces a large number of ambiguous samples.

#### Cross-Backbone Generalization (DAIR-V2X-I, Hard, Average AP)

- BEVHeight: 38.50 (best)
- BEVSpread: 36.07 (best)
- BEVDet: 33.01 (best)

LH3D achieves the best results across all three BEV detectors, validating the generality of the proposed method.

## Highlights & Insights

1. **Novel problem formulation**: The paper is the first to explicitly define "inherently ambiguous samples" in roadside BEV perception and validate their negative impact on training through a human study.
2. **Paradigm shift toward learnability**: The proposed framework redirects active learning from conventional uncertainty maximization to learnability maximization, which is more practically meaningful in roadside scenarios.
3. **Complete theoretical guarantees**: All three submodular objectives are proven to satisfy monotonicity and submodularity, with a $(1-1/e)$ approximation ratio guaranteed for greedy optimization.
4. **Well-motivated hierarchical design**: The depth → semantic → geometric priority ordering is validated through ablation to be optimal and is consistent with the dependency structure of BEV pipelines.
5. **Thorough multi-dataset and multi-backbone evaluation**: LH3D consistently outperforms seven AL baselines across two datasets and three BEV detectors.

## Limitations & Future Work

1. **Limited to monocular BEV detection**: The method is not extended to LiDAR point clouds or multimodal fusion settings, restricting its applicability.
2. **Very few object categories**: Only three classes (Car/Ped/Cyc) are evaluated; finer-grained categories (e.g., different vehicle types) may require more sophisticated semantic balancing strategies.
3. **Long-range and heavily occluded objects remain bottlenecks**: Failure case analysis shows that distant vehicles and occluded pedestrians/cyclists are still frequently missed.
4. **Depth confidence depends on the initial model**: The quality of Stage 1 depth filtering is influenced by the model trained on the small initial labeled set, potentially limiting effectiveness during cold-start.
5. **Computational overhead not thoroughly compared**: Although the authors claim selection overhead is negligible, the actual runtime of the three-stage cascade on large candidate pools is not fully reported.
6. **Temporal information not exploited**: The selection strategy does not leverage inter-frame correlations to remove redundancy in video streams from roadside cameras.

## Related Work & Insights

- **vs. BADGE/CORESET**: These classical AL methods focus on uncertainty or embedding-space diversity and are misled by ambiguous samples in roadside settings. LH3D explicitly filters ambiguity via depth confidence.
- **vs. PPAL**: PPAL is a recent state-of-the-art detection AL method combining difficulty-calibrated uncertainty with category-matching similarity, but does not address the depth ambiguity inherent to roadside scenes.
- **vs. HUA**: HUA employs Bayesian deep learning for hierarchical uncertainty estimation; however, high-uncertainty samples in roadside settings tend to be unlearnable, causing performance degradation.
- **vs. BEVHeight/BEVSpread**: These are detector backbones rather than AL methods. LH3D is a plug-and-play data selection module compatible with any BEV detector.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce the learnability concept into roadside AL; the definition of inherently ambiguous samples and the human study design are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 2 datasets × 3 backbones × 7 baselines with multiple ablation groups; computational overhead comparison is missing.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical structure, rigorous theoretical treatment, and detailed supplementary material.
- Value: ⭐⭐⭐⭐ — Practically meaningful for data efficiency in roadside perception, though scope is limited to monocular BEV detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](../../ICCV2025/autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_predictionasperception_framework_for_3d_object_d.md)
- [\[CVPR 2026\] On the Feasibility and Opportunity of Autoregressive 3D Object Detection](on_the_feasibility_and_opportunity_of_autoregressive_3d_object_detection.md)
- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](../../AAAI2026/autonomous_driving/roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)

</div>

<!-- RELATED:END -->
