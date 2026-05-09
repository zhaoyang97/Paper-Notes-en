---
title: >-
  [Paper Note] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference
description: >-
  [NeurIPS 2025][Adaptive Random Forest] This paper proposes DAFI (Drift-Aware Feature Importance), an algorithm that leverages distribution drift detection to dynamically switch between SHAP and MDI feature importance methods. Combined with Adaptive Random Forest (ARF), DAFI enables flexible training and efficient explainable inference in communication network scenarios where features are dynamically introduced over time.
tags:
  - NeurIPS 2025
  - Adaptive Random Forest
  - Feature Importance
  - Explainable AI
  - Data Stream
  - 6G Networks
  - Drift Detection
  - SHAP
  - MDI
date: 2026-05-08
content_hash: 8e50996f41e9fabd
---

# Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference

**Conference**: NeurIPS 2025
**arXiv**: [2510.08303](https://arxiv.org/abs/2510.08303)
**Code**: Not released
**Area**: Explainability
**Keywords**: Adaptive Random Forest, Feature Importance, Explainable AI, Data Stream, 6G Networks, Drift Detection, SHAP, MDI

## TL;DR

This paper proposes DAFI (Drift-Aware Feature Importance), an algorithm that leverages distribution drift detection to dynamically switch between SHAP and MDI feature importance methods. Combined with Adaptive Random Forest (ARF), DAFI enables flexible training and efficient explainable inference in communication network scenarios where features are dynamically introduced over time.

## Background & Motivation

**AI-native 6G network requirements**: Future communication networks are envisioned as AI-native architectures, where AI must be embedded in base stations and network nodes for mobility pattern recognition and KPI optimization. However, heterogeneous vendors and hardware generations lead to inconsistent feature sets exposed across nodes.

**Feature heterogeneity challenge**: In multi-vendor deployments, different base stations may provide different KPI measurement indicators, making it difficult to train a unified global model. Hardware upgrades introduce new features, requiring models to dynamically adapt to evolving feature spaces.

**Data distribution drift**: Traffic patterns, user behavior, and wireless channel conditions evolve over time. Static models cannot cope with continuous drift, necessitating incremental or stream-based learning approaches.

**Critical role of explainability**: When AI controls critical infrastructure, operator trust and regulatory compliance require models to provide transparent decision explanations. Feature Importance (FI) is the primary mechanism for achieving this.

**High computational overhead of SHAP**: SHAP is the most widely used FI method, but its computational cost grows exponentially with the number of features and model size, making it infeasible for real-time, frequent computation at base stations.

**Inconsistency between MDI and SHAP**: MDI, as a lightweight built-in FI method for tree models, is extremely fast but exhibits significant ranking inconsistencies with SHAP when data drift occurs and the model has not yet been updated, limiting its reliability.

## Method

### Overall Architecture

The paper proposes two complementary components: (1) iterative stream-based training using ARF to handle dynamically expanding feature sets in communication network scenarios; and (2) the DAFI algorithm, which intelligently switches between SHAP and MDI during ARF training based on distribution drift signals, balancing FI accuracy with computational efficiency.

### Key Design 1: Iterative Training with Adaptive Random Forest (ARF)

- **Function**: Incrementally trains ARF on data streams, progressively introducing new features as training proceeds (adding 1 feature at epoch 10, and 2 features at epoch 20).
- **Mechanism**: ARF incorporates built-in drift detectors (based on Hoeffding Trees) that automatically replace degraded subtrees upon detecting concept drift, maintaining stability as the feature space expands. A prequential 80:20 split is adopted, iteratively training and evaluating over 40–50 batches in sequence.
- **Design Motivation**: Feature sets in communication networks expand as new hardware and configurations are introduced. ARF's adaptive mechanism naturally suits this dynamic environment and incurs lower cost than retraining a global model from scratch.

### Key Design 2: DAFI (Drift-Aware Feature Importance)

- **Function**: For each sample, DAFI determines whether to use SHAP (when drift is detected) or MDI (when no drift is present) to compute FI, based on the distribution drift status between the current and previous training batch.
- **Mechanism**: The Kolmogorov-Smirnov (KS) test is applied per feature to detect distributional differences between consecutive batches. If the KS statistic exceeds threshold $\eta$, drift is declared and SHAP is triggered; otherwise, the lightweight MDI is used. The threshold $\eta$ must be tuned to the dataset batch size ($\eta=0.125$ for the Network dataset; $\eta=1$ for others).
- **Design Motivation**: MDI closely aligns with SHAP when the model has adapted to the current data distribution; discrepancies arise mainly during drift before model adaptation. Therefore, the expensive SHAP computation is invoked only when drift occurs, while MDI conserves resources during stable periods.

### Key Design 3: Dynamic Top-$k$ Evaluation Metric

- **Function**: An adaptive Top-$k$ feature evaluation approach is designed, determining $k$ by accumulating SHAP weights up to a threshold of 0.8, rather than using a fixed $k$.
- **Mechanism**: In streaming scenarios with dynamically varying feature counts, a fixed $k$ is inappropriate. Features are ranked in descending order of SHAP weight and accumulated until 80% of the cumulative weight is reached, defining the Top-$k$ set for comparison. Set Match, Exact Match, and Spearman rank correlation are then used to evaluate the consistency of MDI/DAFI against SHAP.
- **Design Motivation**: Normalized FI scores depend on the current number of features. Dynamic $k$ more fairly reflects ranking accuracy across methods under a changing feature space.

## Loss & Training

- ARF is implemented using the `river` library, based on incremental learning with Hoeffding Trees; no explicit loss function optimization is required. Node splitting is performed via information gain or the Gini index.
- The built-in ADWIN drift detector resets affected subtrees upon drift detection.
- Training follows a prequential (test-then-train) streaming setup with $\text{ntrees}=10$; each FI computation samples $\text{nsamples}=50$ test instances.

## Key Experimental Results

### Experiment 1: ARF Performance under Dynamic Features (Network Dataset)

| Phase | # Features | Test Accuracy |
|-------|------------|--------------|
| Epoch 0–9 | 3 features | ~0.64 |
| Epoch 10–19 | 4 features (+1) | ~0.64 → gradual improvement |
| Epoch 20–40 | 6 features (+2) | ~0.72 |

The introduction of new features does not harm model performance; accuracy improves consistently from 0.64 to 0.72, demonstrating ARF's stability under a dynamically expanding feature space.

### Experiment 2: DAFI Runtime Efficiency and FI Accuracy (Network Dataset)

| FI Method | Runtime (s) | Savings (%) | Top-$k$ Set | Top-$k$ Exact | Spearman |
|-----------|------------|-------------|-------------|--------------|----------|
| SHAP | 1488.32 | 0.00 | 1.00 | 1.00 | 1.00 |
| MDI | 9.37 | 99.37 | 0.27 | 0.09 | 0.53 |
| **DAFI** | **674.49** | **54.68** | **0.55** | **0.40** | **0.67** |

On the Network dataset, DAFI reduces runtime by approximately 55% while outperforming MDI across all FI accuracy metrics. On the Electricity dataset, Top-$k$ Exact reaches 0.76 and Spearman 0.85; on the Weather dataset, Spearman reaches 0.83, demonstrating cross-domain generalization.

## Highlights & Insights

- **Precise problem framing**: The paper accurately identifies practical requirements in 6G network AI deployment, including feature heterogeneity, dynamic feature growth, and real-time explainability.
- **Simple yet effective DAFI design**: By using only KS testing to detect drift and switch between SHAP and MDI, a significant efficiency–accuracy trade-off is achieved without complex architectural modifications.
- **Substantial runtime reduction**: SHAP computational overhead is reduced by approximately 50–65% while retaining high FI consistency.
- **Comprehensive experimental validation**: Evaluation is conducted across three datasets from different domains, supported by runtime growth analysis of SHAP with respect to feature count and model scale.

## Limitations & Future Work

- **Limited technical novelty as a vision paper**: The paper self-positions as a vision paper; both ARF and the KS test are existing methods, and the overall contribution is primarily a combination of established techniques.
- **Manual tuning of threshold $\eta$**: The KS drift threshold varies substantially across datasets (0.125 vs. 1.0), with no automated selection strategy provided.
- **Small evaluation scale**: Experiments use only $\text{ntrees}=10$ and $\text{nsamples}=50$; effectiveness in large-scale real-world deployments remains unvalidated.
- **Only feature addition is evaluated**: More complex dynamic feature change scenarios, such as feature removal or replacement, are not addressed.
- **Questionable classification as object detection**: The paper's core focus is network traffic/KPI classification, which bears little relation to the object detection domain.

## Related Work & Insights

- **SHAP / TreeSHAP**: Proposed by Lundberg & Lee (2017); a model-agnostic FI method that is accurate but computationally expensive.
- **MDI (Mean Decrease in Impurity)**: Formalized by Scornet (2021); a built-in tree model FI method that is fast but unstable.
- **Incremental XAI**: iPFI (Fumagalli et al., 2023) and Muschalik et al. (2022) explore incremental explanation methods for streaming scenarios.
- **ARF**: Proposed by Gomes et al. (2017); an adaptive random forest with built-in drift detection for stream-based ensemble learning.
- **XAI in networking**: Brik et al. (2024) survey explainable AI methods for 6G O-RAN.

## Rating

- Novelty: ⭐⭐⭐ — The method combines existing techniques, though the problem formulation and drift-driven switching idea carry independent value.
- Experimental Thoroughness: ⭐⭐⭐ — Validation across three datasets is adequate, but scale is limited and ablation studies are absent.
- Writing Quality: ⭐⭐⭐⭐ — Definitions are clear, structure is well-organized, and motivation is thoroughly articulated.
- Value: ⭐⭐⭐⭐ — Offers practical reference value for real-world AI deployment in 6G network scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[NeurIPS 2025\] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)

<!-- RELATED:END -->
