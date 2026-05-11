---
title: >-
  [Paper Note] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning
description: >-
  [NeurIPS 2025][graph anomaly detection] This paper proposes RHO (Robust Homophily Learning), which addresses the homophily diversity of normal nodes in semi-supervised graph anomaly detection via an adaptive frequency re…
tags:
  - "NeurIPS 2025"
  - "graph anomaly detection"
  - "homophily learning"
  - "adaptive frequency filter"
  - "semi-supervised"
  - "contrastive learning"
date: 2026-05-08
content_hash: 9b83ad47fcbf1097
---

# Semi-supervised Graph Anomaly Detection via Robust Homophily Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.15448](https://arxiv.org/abs/2506.15448)
**Code**: [GitHub](https://github.com/mala-lab/RHO)
**Area**: Others
**Keywords**: graph anomaly detection, homophily learning, adaptive frequency filter, semi-supervised, contrastive learning

## TL;DR
This paper proposes RHO (Robust Homophily Learning), which addresses the homophily diversity of normal nodes in semi-supervised graph anomaly detection via an adaptive frequency response filter (AdaFreq) and a Graph Normality Alignment (GNA) module, outperforming existing methods on 8 real-world datasets.

## Background & Motivation
- Semi-supervised graph anomaly detection (GAD) leverages a small number of labeled normal nodes to identify anomalies among a large pool of unlabeled nodes.
- Existing methods assume: (1) normal nodes exhibit similar levels of homophily; and (2) labeled normal nodes are representative of the overall homophily pattern.
- **Practical issue**: Homophily varies substantially among normal nodes — on the Amazon and Elliptic datasets, a portion of normal nodes exhibit very low homophily.
- Both conventional GCN filters (low-frequency assumption) and BWGNN filters (predefined frequency response) fail to accommodate normal nodes with heterogeneous homophily distributions.
- This causes low-homophily normal nodes to be misclassified as anomalies.

## Method

### Overall Architecture
RHO consists of three core components:
1. **AdaFreq**: An adaptive frequency response filter that learns jointly from cross-channel and within-channel views.
2. **GNA**: Graph Normality Alignment, which enforces consistency between the normality representations of the two views.
3. **One-class classification loss**: Projects normal nodes toward the center of a hypersphere.

### Key Designs

#### AdaFreq: Adaptive Frequency Response Filter
Core filter function: $g(\lambda) = 1 - k\lambda$, where $k$ is a learnable parameter.
- $k > 0$: suppresses high frequencies, preserves low frequencies (high-homophily nodes).
- $k < 0$: emphasizes high frequencies (low-homophily nodes).
- $k = 0$: all-pass filter.
- Stacking $K$ layers yields: $g(\lambda) = \prod_{i=1}^{K}(1 - k_i\lambda)$, enabling complex frequency responses.

**Cross-channel view**: A single shared $k$ parameter across all channels.
$$H_{ccr}^{(t)} = \sigma((I - k\hat{L})H_{ccr}^{(t-1)}W_{ccr}^{(t)})$$

**Within-channel view**: A distinct $k_j$ per channel, implemented via the Hadamard product.
$$H_{cwr}^{(t)} = \sigma((I - \hat{L})(H_{cwr}^{(t-1)} \odot K)W_{cwr}^{(t)})$$

#### GNA: Graph Normality Alignment
- Constructs positive pairs: representations of the same node across the two views form a positive pair.
- Employs a contrastive learning objective to maximize similarity of positive pairs and minimize that of negative pairs.
- Dual-anchor strategy: contrastive losses are computed with each view serving as the anchor in turn.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \frac{1}{2}(\mathcal{L}_{ccr} + \mathcal{L}_{cwr}) + \alpha \mathcal{L}_{GNA}$

- One-class loss $\mathcal{L}_{ccr/cwr}$: minimizes the distance from normal node representations to the hypersphere center.
- Alignment loss $\mathcal{L}_{GNA}$: cross-view contrastive learning.
- At inference, the anomaly score is the average distance of a node to the centers of both views.

## Key Experimental Results

### Main Results (AUROC, 15% labeled normal nodes)

| Method | Reddit | Tolokers | Photo | Amazon | Elliptic | Question | T-Finance | DGraph |
|---|---|---|---|---|---|---|---|---|
| GGAD | 0.6354 | 0.5340 | 0.6476 | **0.9443** | 0.7290 | 0.5122 | 0.8228 | 0.5943 |
| BWGNN | 0.5580 | 0.5821 | 0.6861 | 0.8312 | 0.7241 | 0.5740 | 0.7683 | 0.4958 |
| CONSISGAD | 0.5347 | 0.5974 | 0.5859 | 0.8715 | 0.7354 | 0.5737 | 0.8277 | 0.5735 |
| **RHO** | 0.6207 | **0.6255** | **0.7129** | 0.9302 | **0.8509** | **0.5833** | **0.8623** | **0.6033** |

### Ablation Study
- Cross-channel view alone: some anomalous nodes on Amazon are misidentified as normal (appearing near the center).
- Within-channel view alone: some normal nodes deviate from the center, producing false positives.
- **Joint use of both views (RHO)**: normal nodes cluster more tightly, and camouflaged anomalies are successfully detected.

### Key Findings
- RHO surpasses the best competing method, GGAD, on 6 out of 8 datasets, with a maximum AUROC gain of 12.19% and an AUPRC gain of 30.68%.
- The improvement is most pronounced on the Elliptic dataset: AUROC increases from 0.7354 (CONSISGAD) to 0.8509.
- AdaFreq demonstrates robustness across three distinct homophily distributions, whereas GCN and BWGNN filters fail under certain distributions.
- Anomaly-generative methods (e.g., GGAD) perform best on Amazon but generalize less effectively than RHO.

## Highlights & Insights
1. **Uncovering an overlooked problem**: The paper is the first to systematically identify the homophily diversity of normal nodes in semi-supervised GAD.
2. **Theoretical guarantee**: Theorem 1 proves that the adaptive filter can automatically amplify spectrally consistent components of normal nodes while suppressing inconsistent ones.
3. **Complementary dual-view design**: The cross-channel and within-channel views capture complementary normality patterns, and their joint use substantially improves detection performance.
4. **No anomaly labels required**: The method does not rely on any labeled anomalous data or anomaly generation.

## Limitations & Future Work
- The hyperparameter $\alpha$ requires dataset-specific tuning (1.0 for large datasets, 0.1 for small ones).
- Computational complexity scales linearly with the number of edges, which may pose efficiency challenges for large-scale graphs.
- At least 5% labeled normal nodes are required.
- Extensions to dynamic or temporal graphs have not been explored.
- The impact of the temperature parameter $\tau$ in GNA contrastive learning on performance is not discussed in detail.
- Validation is limited to node-level anomaly detection; extensions to edge-level or subgraph-level anomaly detection have not been pursued.

## Related Work & Insights
- Spectral GAD methods (AMNet, BWGNN, GHRN) employ predefined frequency responses, lacking adaptability.
- Graph homophily modeling methods primarily handle homophily variation through neighbor selection (edge addition/removal).
- GGAD is the first method specifically designed for semi-supervised GAD (via anomaly generation), but its generalizability is limited.
- RHO is the first to adaptively learn heterogeneous normality patterns from a frequency-domain perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of adaptive frequency filtering and dual-view alignment is novel.)
- Technical Depth: ⭐⭐⭐⭐⭐ (Theoretical analysis, method design, and extensive experiments.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 datasets, multiple baselines, detailed ablation study.)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, rich figures and tables.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/others/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)

</div>

<!-- RELATED:END -->
