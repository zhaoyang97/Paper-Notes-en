---
title: >-
  [Paper Note] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection
description: >-
  [NeurIPS 2025][AI Safety][Graph OOD Detection] This paper proposes RedOUT, a framework that constructs coding trees via structural entropy minimization to eliminate redundant information in graph structures. Combined with the Redundancy-aware Graph Information Bottleneck (ReGIB) principle, RedOUT effectively distinguishes in-distribution (ID) from out-of-distribution (OOD) graph samples at test time without modifying pretrained model parameters…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Graph OOD Detection"
  - "Structural Entropy"
  - "Information Bottleneck"
  - "Test-Time Detection"
  - "Coding Tree"
date: 2026-05-08
content_hash: 9047e3491fd6ba5e
---

# Redundancy-Aware Test-Time Graph Out-of-Distribution Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.14562](https://arxiv.org/abs/2510.14562)  
**Code**: [Available](https://github.com/name-is-what/RedOUT)  
**Area**: Other
**Keywords**: Graph OOD Detection, Structural Entropy, Information Bottleneck, Test-Time Detection, Coding Tree

## TL;DR

This paper proposes RedOUT, a framework that constructs coding trees via structural entropy minimization to eliminate redundant information in graph structures. Combined with the Redundancy-aware Graph Information Bottleneck (ReGIB) principle, RedOUT effectively distinguishes in-distribution (ID) from out-of-distribution (OOD) graph samples at test time without modifying pretrained model parameters, achieving an average AUC of 87.46% across 10 dataset pairs.

## Background & Motivation

OOD detection on graph-structured data poses unique challenges: due to the non-Euclidean geometry and complex topology of graphs, models tend to produce overconfident erroneous predictions when encountering out-of-distribution data. Existing methods fall into two main categories:

**End-to-end methods**: Train OOD-specific GNNs from scratch (e.g., GOOD-D), leveraging graph contrastive learning to extract discriminative representations.

**Post-hoc methods**: Employ pretrained GNNs and fine-tune OOD detectors at inference time (e.g., GOODAT optimizes learnable graph masks on test samples).

However, a core issue persists in these data-driven paradigms: **structural redundancy leads to semantic drift**. Test graphs contain both key structures that distinguish ID from OOD samples (e.g., the distinctive components highlighted by dashed ellipses in Figure 1(b)) and a large number of irrelevant structural elements. Such redundant structures interfere with the model's ability to capture critical patterns. Although GOODAT attempts to address this via graph masking, its learnable graph augmentation may alter the semantic information of substructures or cause information loss.

A key observation by the authors: on the AIDS/DHFR dataset pair, applying structural entropy minimization to graph representations reduces the variance of OOD scores and significantly decreases the distributional overlap between ID and OOD samples (Figure 1(c)). This intuitively demonstrates that the discriminative components retained after redundancy removal more effectively separate distributions.

## Method

### Overall Architecture

RedOUT is an unsupervised test-time graph OOD detection framework built on the following pipeline:
1. **ReGIB Principle**: Decompose graph information into essential information and irrelevant redundancy.
2. **Structural Entropy Minimization**: Construct coding trees as instantiations of the essential view.
3. **Hierarchical Representation Learning**: Learn redundancy-free graph representations based on coding trees.
4. **OOD Detection**: Perform detection using calibrated OOD scores without modifying pretrained model parameters.

### Key Designs

#### 1. **Redundancy-Aware Graph Information Bottleneck (ReGIB)**

The standard GIB objective is $\min -I(f(G);Y) + \beta I(G;f(G))$. The authors introduce an essential view $G^*$ and replace unknown labels with pseudo-labels $\tilde{Y}$, extending the formulation to ReGIB:

$$\min \text{ReGIB} \triangleq -I(f(G^*);f(G)) + I(f(G^*);f(G)|\tilde{Y}) + \beta I(G^*;f(G^*))$$

Intuition behind each term:
- **First term** $I(f(G^*);f(G))$: Prediction term — ensures essential information is preserved.
- **Second term** $I(f(G^*);f(G)|\tilde{Y})$: Compression term — encourages discarding redundant information conditioned on pseudo-labels.
- **Third term** $I(G^*;f(G^*))$: Compression term — corresponds to minimizing the structural complexity of the essential view.

**Design Motivation**: Pretrained models are trained solely on ID data, making their predictions on OOD data more random and their identification of discriminative structures unreliable. ReGIB provides a calibration signal to the model through the discriminative patterns encoded in $G^*$.

#### 2. **Structural Entropy Minimization and Coding Tree Construction**

Via Proposition 4, the authors prove that $\min I(G^*;f(G^*)) \leq \min I(G^*;G) = \min \mathcal{H}(G^*)$, so minimizing structural entropy suffices to obtain the redundancy-free essential view.

Structural entropy is defined as: $\mathcal{H}^T(G) = -\sum_{v_\tau \in T} \frac{g_{v_\tau}}{vol(\mathcal{V})} \log \frac{vol(v_\tau)}{vol(v_\tau^+)}$

Coding tree construction proceeds in two steps: (1) build a full-height binary coding tree; (2) compress it to a fixed height $k$. Two operators, MERGE and DROP, are used for efficient construction, with time complexity $O(h(|\mathcal{E}|\log|\mathcal{V}|+|\mathcal{V}|))$, which is approximately linear.

#### 3. **Hierarchical Representation Learning**

The coding tree encoder is designed as bottom-up message passing: $\mathbf{x}_v^{(l)} = \text{MLP}^{(l)}(\sum_{u \in \mathcal{C}(v)} \mathbf{x}_u^{(l-1)})$

Features are aggregated from leaf nodes to the root, and a readout function produces the tree embedding $\mathbf{Z}_T$.

### Loss & Training

The final optimization objective is: $\mathcal{L} = \mathcal{L}_{Cl} + \lambda \mathcal{L}_{CRI}$

- $\mathcal{L}_{Cl}(G^*, G)$: Contrastive learning loss that maximizes mutual information between the essential view and the original graph representation.
- $\mathcal{L}_{CRI}(G, G^*)$: Conditional redundancy elimination loss that minimizes redundant mutual information conditioned on pseudo-labels.

During OOD detection, the overall loss $\mathcal{L}$ is directly used as the OOD score. Structural entropy minimization serves as a preprocessing step (requiring no additional learning), and the pretrained model parameters are fully frozen.

## Key Experimental Results

### Main Results

OOD detection results (AUC%) across 10 TUDataset and OGB dataset pairs:

| ID/OOD Dataset Pair | RedOUT | GOOD-D (2nd Best) | GOODAT | Gain |
|---|---|---|---|---|
| BZR/COX2 | **95.06** | 94.99 | 82.16 | +0.07 |
| PTC-MR/MUTAG | **94.45** | 81.21 | 81.84 | +12.61 |
| AIDS/DHFR | **99.98** | 99.07 | 96.43 | +0.91 |
| ClinTox/LIPO | **86.56** | 69.18 | 62.46 | **+17.38** |
| Esol/MUV | **95.00** | 91.52 | 85.91 | +3.48 |
| Average AUC | **87.46** | 80.73 | 76.89 | **+6.73** |
| Average Rank | **1.3** | 2.4 | 3.9 | - |

### Ablation Study

Anomaly detection results (AUC%):

| Dataset | RedOUT | GOOD-D | GLocalKD | Note |
|---|---|---|---|---|
| ENZYMES | **69.14** | 65.64 | 57.18 | Protein graphs |
| DHFR | **54.45** | 53.08 | 52.11 | Molecular graphs |
| BZR | **66.86** | 61.24 | 55.32 | Molecular graphs |
| IMDB-B | 53.12 | 54.89 | **56.42** | Social networks |

### Key Findings

1. **Large advantage on molecular graphs**: RedOUT achieves an average improvement of approximately 10% on molecular graph datasets, as semantic information in molecular graphs is directly encoded in structural composition (e.g., functional groups).
2. **Weakness on social networks**: RedOUT does not achieve the best results on IMDB-B/IMDB-M, as these datasets originate from the same source and differ only in labels, making structural-only discrimination difficult.
3. **Efficient and scalable**: Experiments on Erdős-Rényi graphs demonstrate approximately linear growth in runtime and memory usage with respect to node count.

## Highlights & Insights

1. **First introduction of structural entropy into test-time OOD detection**: This work organically combines an information-theoretic tool (structural entropy) with the graph information bottleneck, establishing a new paradigm for graph OOD detection.
2. **Solid theoretical guarantees**: Upper and lower bounds on the ReGIB objective are derived, and it is proven that the optimal encoder retains more information relevant to true labels (Theorem 3.4).
3. **No modification of pretrained models**: Coding tree construction is a preprocessing step that leaves pretrained model parameters entirely unchanged, making the approach highly practical.
4. **Information-theoretic perspective on redundancy elimination**: The work formalizes the intuition of "removing redundancy" through mutual information decomposition.

## Limitations & Future Work

1. The method shows limited effectiveness on graph data where structural features are not salient, such as social networks.
2. The coding tree height $k$ must be specified in advance and may require tuning for different graph types.
3. The method relies on the quality of pseudo-labels $\tilde{Y}$; poor pretrained model performance on ID data may degrade detection effectiveness.
4. Extending the framework to node-level or subgraph-level OOD detection is a promising future direction.

## Related Work & Insights

- Compared to GOODAT, RedOUT avoids the semantic drift that learnable augmentation may introduce, instead directly eliminating redundancy from an information-theoretic perspective.
- Structural entropy minimization, as a hierarchical abstraction tool for graph structure, has potential applications in other graph learning tasks such as graph classification and community detection.
- The ReGIB principle is generalizable to other graph learning tasks that require distinguishing essential information from redundancy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The combination of structural entropy and GIB is novel, though each individual component is not entirely new.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (18 baselines, 10 dataset pairs, anomaly detection experiments, and runtime analysis.)
- **Writing Quality**: ⭐⭐⭐⭐ (Theoretical derivations are clear, though the density of equations presents a non-trivial reading barrier.)
- **Value**: ⭐⭐⭐⭐ (Establishes a new paradigm for graph OOD detection with strong practical utility.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/ai_safety/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[NeurIPS 2025\] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/ai_safety/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](../../ICCV2025/ai_safety/fret_feature_redundancy_elimination_for_test_time_adaptation.md)

</div>

<!-- RELATED:END -->
