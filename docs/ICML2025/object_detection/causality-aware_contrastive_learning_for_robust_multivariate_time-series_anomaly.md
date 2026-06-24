---
title: >-
  [Paper Note] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection
description: >-
  [ICML 2025][Object Detection][Multivariate Anomaly Detection] This paper proposes CAROTS—a multivariate time-series anomaly detection framework that integrates causal relationships into contrastive learning. It utilizes causality-preserving augmentation as positive samples (normal variations) and causality-violating augmentation as negative samples (simulated anomalies) to train encoders to distinguish normal from abnormal patterns based on causal structures.
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Multivariate Anomaly Detection"
  - "Causality"
  - "Contrastive Learning"
  - "Data Augmentation"
  - "Causal Discovery"
date: 2026-05-08
content_hash: b77dfac0968e8888
---

# Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection

**Conference**: ICML 2025  
**arXiv**: [2506.03964](https://arxiv.org/abs/2506.03964)  
**Code**: [https://github.com/kimanki/CAROTS](https://github.com/kimanki/CAROTS)  
**Area**: Time Series  
**Keywords**: Multivariate Anomaly Detection, Causality, Contrastive Learning, Data Augmentation, Causal Discovery

## TL;DR
This paper proposes CAROTS—a multivariate time-series anomaly detection framework that integrates causal relationships into contrastive learning. It utilizes causality-preserving augmentation as positive samples (normal variations) and causality-violating augmentation as negative samples (simulated anomalies) to train encoders to distinguish normal from abnormal patterns based on causal structures.

## Background & Motivation

**Background**: Multivariate Time-Series Anomaly Detection (MTSAD) is a core requirement in critical fields such as cybersecurity and medical monitoring.

**Limitations of Prior Work**: Existing unsupervised methods (reconstruction/contrastive learning) focus on superficial differences in data values or distributions while ignoring causal relationships between variables. This leads to normal causal changes being falsely reported as anomalies (e.g., an air conditioner running longer due to high temperature is a normal causal change, not an anomaly).

**Key Challenge**: Distinguishing between "causality-preserving normal variations" and "causality-violating true anomalies."

**Goal**: Integrating causal structures into anomaly detection.

**Key Insight**: Extracting causal graphs between variables using a causal discovery model, and then designing causality-preserving and causality-violating data augmentation strategies.

**Core Idea**: Causality-preserving augmentation = positive samples, causality-violating augmentation = negative samples $\rightarrow$ Contrastive learning to train a causality-aware encoder.

## Method

### Overall Architecture
1. Extract causal relationships between variables using a predictive causal discovery model.
2. Causality-preserving augmenter: modifies data while keeping the causal structure intact.
3. Causality-violating augmenter: breaks causal relationships to simulate anomalies.
4. Contrastive learning: positive samples (causality-preserving) vs. negative samples (causality-violating).
5. Anomaly detection: distance to normal cluster center + causal prediction error.

### Key Designs

1. **Causality-Preserving Augmentation**:

    - **Function**: Generates diverse normal samples that adhere to the original causal structure.
    - **Mechanism**: Applies changes to "cause" variables in the causal graph, with "effect" variables adjusted accordingly based on causal relationships.
    - **Design Motivation**: Expands the diversity of normal patterns to prevent misclassifying normal changes as anomalies.

2. **Causality-Violating Augmentation**:

    - **Function**: Generates synthetic anomalies that break causal relationships.
    - **Mechanism**: Randomly disrupts the causal chain—such as changing an "effect" variable without modifying its corresponding "cause" variable.
    - **Design Motivation**: Simulates real-world anomalies (such as causal breaks caused by sensor failures).

3. **Similarity-Filtered One-Class Contrastive Loss**:

    - **Function**: Progressively introduces more diverse positive samples.
    - **Mechanism**: Initially considers only high-similarity samples as positive samples, gradually relaxing the threshold as training progresses.
    - **Design Motivation**: Prevents high noise from causing clustering instability during the early stages of training.

### Loss & Training
- Integration of contrastive loss + causal prediction loss.
- Weighted combination of two anomaly scores.

## Key Experimental Results

### Main Results
F1 scores on five real-world datasets:

| Method | SWaT | WADI | SMD | MSL | PSM |
|------|------|------|-----|-----|-----|
| USAD | 79.2 | 56.3 | 82.4 | 73.4 | 96.2 |
| AnomalyTransformer | 83.5 | 58.1 | 86.2 | 78.8 | 97.1 |
| **CAROTS** | **86.7** | **63.4** | **88.9** | **82.1** | **97.8** |

### Ablation Study

| Configuration | F1 (SWaT) | Description |
|------|-----------|------|
| No Causal Augmentation | 81.2 | Degenerates to standard contrastive learning |
| Causality-Preserving Only | 83.5 | Lacks anomaly simulation |
| Causality-Violating Only | 84.1 | Lacks normal diversity |
| **Full CAROTS** | **86.7** | Optimal |

### Key Findings
- The largest improvement is achieved on datasets with distinct causal relationships (SWaT: +3.2%).
- The performance advantage is more pronounced on synthetic datasets (where causal relationships are explicitly controllable).
- Similarity filtering stabilizes training, leading to a +1.5% improvement.

## Highlights & Insights
- The combination of **causality $\times$ contrastive learning** is natural and effective—the concept of "anomaly = causal violation" is highly intuitive.
- The data augmentation strategies balance positive sample diversity and negative sample realism.
- This approach can be generalized to any multivariate monitoring scenarios featuring causal relationships between variables.

## Limitations & Future Work
- The accuracy of the causal discovery model directly affects downstream performance.
- It is assumed that causal relationships remain stable under normal states; time-varying causal scenarios require future extension.
- Only linear and simple non-linear causal relations were tested.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of causality and contrastive learning is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 real + 2 synthetic datasets, thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and natural methodology.
- **Value**: ⭐⭐⭐⭐ Advances causality-aware anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](../../NeurIPS2025/object_detection/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[ICML 2025\] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/object_detection/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/object_detection/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[ICLR 2026\] PGRF-Net: A Prototype-Guided Relational Fusion Network for Diagnostic Multivariate Time-Series Anomaly Detection](../../ICLR2026/object_detection/pgrf-net_a_prototype-guided_relational_fusion_network_for_diagnostic_multivariat.md)

</div>

<!-- RELATED:END -->
