---
title: >-
  [Paper Note] MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)
description: >-
  [AAAI 2026][Object Detection][trajectory similarity] This paper proposes MovSemCL, a framework that transforms GPS trajectories into movement-semantic features (displacement vectors + heading angles + Node2Vec spatial gr…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "trajectory similarity"
  - "contrastive learning"
  - "movement semantics"
  - "hierarchical encoding"
  - "curvature-guided augmentation"
date: 2026-05-08
content_hash: bfe31c657a60ad18
---

# MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.12061](https://arxiv.org/abs/2511.12061)  
**Code**: [https://github.com/ryanlaics/MovSemCL](https://github.com/ryanlaics/MovSemCL)  
**Area**: Self-Supervised Learning / Trajectory Analysis
**Keywords**: trajectory similarity, contrastive learning, movement semantics, hierarchical encoding, curvature-guided augmentation

## TL;DR

This paper proposes MovSemCL, a framework that transforms GPS trajectories into movement-semantic features (displacement vectors + heading angles + Node2Vec spatial graph embeddings), achieves hierarchical encoding via patch-level two-stage attention (reducing complexity from $O(L^2)$ to near-linear), and designs Curvature-Guided Augmentation (CGA) to preserve behaviorally critical segments such as turns and intersections. The framework achieves a mean rank approaching the ideal value of 1 on trajectory retrieval tasks while reducing inference latency by 43.4%.

## Background & Motivation

**Background**: Trajectory similarity computation is a fundamental function for ride-sharing, logistics optimization, and urban analytics. Traditional methods (Hausdorff, Fréchet distance) are computationally expensive and semantics-agnostic; learning-based methods (RNN/CNN/Transformer) embed trajectories into vector spaces for efficient cosine-similarity retrieval.

**Three Core Limitations**:
- **(L1) Insufficient semantic and hierarchical modeling**: Existing methods treat trajectories as flat coordinate sequences, neither extracting movement dynamics (velocity changes, direction shifts) nor modeling the hierarchical structure of points → maneuvers → trips.
- **(L2) Poor computational efficiency**: Real-world trajectories often contain hundreds of points. RNNs cannot be parallelized, and the $O(L^2)$ attention of Transformers forces lossy downsampling, degrading motion fidelity.
- **(L3) Semantics-agnostic augmentation**: Random masking in contrastive learning causes spatial discontinuities, while uniform sampling discards turn/intersection information, producing physically implausible trajectory views.

**Key Insight**: The three limitations are addressed by dedicated designs—movement-semantics encoding (L1), patch-based hierarchical encoding (L1+L2), and curvature-guided augmentation (L3)—which together constitute a unified framework.

## Method

### Overall Architecture

MovSemCL consists of three stages: (1) Movement-Semantics Encoding, which converts raw GPS into movement-semantic features; (2) Hierarchical Semantics Encoding, which applies two-stage attention over patches; and (3) Semantics-Aware Contrastive Learning, which trains with CGA augmentation and a MoCo-style contrastive loss.

### Stage 1: Movement-Semantics Encoding

- **Coordinate normalization**: WGS84 latitude/longitude is first projected to a planar coordinate system via Mercator projection, then normalized to $[0,1]$ by region width and height.
- **Movement dynamics features**: Displacement vectors $(dx_i, dy_i)$ and heading angles $\theta_i = \arctan2(dy_i, dx_i)/\pi$ are computed between consecutive points to capture directional flow and instantaneous changes.
- **Trajectory-induced spatial graph**: The map is partitioned into an $N_x \times N_y$ grid; a directed weighted graph is constructed from transition frequencies between adjacent cells in historical trajectories, and Node2Vec is applied to learn structural embeddings $\mathbf{ST}_i \in \mathbb{R}^{d_{se}}$ per cell.
- **Feature concatenation**: The final representation of each point is $\mathbf{f}_i = [dx_i, dy_i, \theta_i, \mathbf{ST}_i] \in \mathbb{R}^{d_{in}}$, where the first three dimensions encode local motion and the spatial embedding encodes global context.

### Stage 2: Hierarchical Semantics Encoding

- **Patch construction**: A sequence of length $L$ is divided into $M = \lceil L/P \rceil$ patches ($P=4$), each representing a locally coherent motion unit.
- **Intra-Patch Attention**: Self-attention is applied within each patch to capture local motion patterns, followed by masked average pooling to compress each patch into a fixed-length vector $\mathbf{h}_j$.
- **Inter-Patch Attention**: Self-attention over the sequence of patch embeddings captures global long-range dependencies and overall trajectory intent.
- **Complexity advantage**: Complexity is reduced from $O(L^2)$ to $O(L \cdot P + M^2)$; since $M \ll L$ for typical trajectory lengths, this achieves near-linear scaling.

### Stage 3: Semantics-Aware Contrastive Learning

- **Curvature-Guided Augmentation (CGA)**: The local turning angle is computed at each trajectory point; points with large angles (turns/intersections) receive high retention weights, while points with small angles (straight-line redundancy) are more likely to be masked. Start and end points are always retained. A multinomial sampling procedure selects the mask set using weights proportional to $(1 - p_i)$.
- **Distinction from naive augmentation**: Random masking produces spatial discontinuities; uniform sampling loses critical turn information; block masking may remove entire key maneuvers. CGA retains segments with the highest behavioral information density.
- **Contrastive objective**: The MoCo framework is adopted. Two CGA-augmented views of each trajectory serve as a positive pair, with embeddings of other trajectories as negatives. Temperature $\tau = 0.05$; the query encoder is updated via backpropagation and the key encoder via exponential moving average.

### CGA Algorithm Details

1. Compute the turning angle $\alpha_i$ at each interior point (arccos of the angle between adjacent displacement vectors).
2. Normalize to $[0,1]$ and assign retention weights: endpoints receive $w_{\text{endpoint}}$; interior points receive $w_{\text{base}} + \hat{\alpha}_i \cdot w_{\text{direction}}$.
3. After normalization to a probability distribution, sample $\lfloor L \cdot r_{\text{mask}} \rfloor$ points for masking via multinomial sampling with weights $(1-p_i)$.
4. Time complexity is $O(L)$; controllable parameters include the mask ratio and the weight triple.

## Key Experimental Results

### Datasets

| Dataset | Source | Avg. Points | Avg. Length | Characteristics |
|---------|--------|-------------|-------------|-----------------|
| Porto | Portuguese taxi trajectories (2013–2014) | 48 | 6.37 km | Dense urban short trips |
| Germany | Cross-city German trajectories (2006–2013) | 72 | 252.49 km | Sparse long-distance routes |

### Main Results

#### Trajectory Retrieval (RQ1)

On a 100K database, MovSemCL achieves a mean rank of **1.005** (Porto) and **1.008** (Germany), approaching the ideal value of 1. TrajCL achieves 1.010 and 1.045, respectively, while the traditional method EDR reaches as high as 28.75 and 1370.

#### Robustness (RQ2)

- **Downsampling**: At a mask probability of 0.5, TrajCL degrades to a mean rank of 36.35 on Porto, while MovSemCL reaches only 9.95.
- **Coordinate distortion**: At a distortion rate of 0.5, MovSemCL maintains a mean rank of approximately 1.004 (Porto), demonstrating strong noise robustness.

#### Heuristic Distance Approximation (RQ3)

After fine-tuning to approximate traditional distances (EDR, Hausdorff, Fréchet), MovSemCL achieves the best average rank across all metrics. HR@5 for EDR improves by **20.3%** over TrajCL (0.172 → 0.207).

#### Efficiency (RQ4)

| Metric | TrajCL | MovSemCL | Gain |
|--------|--------|----------|------|
| FLOPs (M) | 158.69 | 93.34 | 41.2% |
| Latency (ms) | 6.08 | 3.44 | **43.4%** |
| Throughput (samples/s) | 164.46 | 290.41 | **76.6%** |

### Ablation Study (RQ5)

On Porto 100K: removing Movement-Semantics Encoding (MSE) degrades mean rank from 1.005 to 3.045 (largest impact); removing CGA degrades it to 1.098; removing Hierarchical Semantics Encoding (HSE) degrades it to 1.012. MSE is the most critical component.

### Hyperparameter Analysis (RQ6)

- Convergence is achieved within 10 epochs; training stabilizes at 20 epochs without overfitting.
- As few as 20K training trajectories suffice for convergence under standard conditions.
- Embedding dimension 256 is optimal; further increases yield no significant gain.
- **Patch size $P=4$ is optimal**—smaller patches lack local context, while larger patches dilute movement semantics.

## Highlights & Insights

- **Curvature-Guided Augmentation (CGA)** is the most creative design in this work: rather than random masking, it selectively retains critical segments based on motion complexity, balancing physical plausibility with semantic richness.
- Movement-semantic features (displacement + heading angle + spatial graph embeddings) carry far greater information density than raw coordinates; ablation studies confirm they are the primary performance driver.
- Patch-based hierarchical encoding elegantly reduces $O(L^2)$ attention to $O(L \cdot P + M^2)$, balancing efficiency and expressiveness for long trajectories.
- The paper is exceptionally well-structured: three identified limitations correspond one-to-one to three design components, forming a clear logical chain.
- The MoCo contrastive framework with a dynamic negative queue ensures training stability.

## Limitations & Future Work

- Node2Vec embeddings for the spatial graph depend on trajectory coverage in the training set—effective embeddings may not be obtainable for new or data-sparse regions.
- Patch size $P$ is a fixed hyperparameter; different scenarios (urban vs. highway) may require different settings.
- Evaluation is conducted on only two datasets; non-vehicular trajectories (pedestrian, cycling) are not assessed.
- The CGA weight triple $(w_{\text{endpoint}}, w_{\text{base}}, w_{\text{direction}})$ requires manual tuning.

## Related Work & Insights

- **Traditional methods**: EDR (edit distance with spatial threshold), Hausdorff (maximum nearest-neighbor distance), Fréchet (order-aware spatiotemporal distance) → computationally expensive and semantics-agnostic.
- **RNN-based methods**: t2vec (seq2seq autoencoder) → TrjSR / E2DTC (recurrent + attention) → non-parallelizable with weak long-range dependency modeling.
- **Transformer + contrastive learning**: TrajCL (trajectory augmentation + dual-feature attention), CLEAR (multi-positive contrastive learning) → $O(L^2)$ complexity with semantics-agnostic augmentation.
- **Positioning of MovSemCL**: The first work to unify movement-semantic feature extraction, hierarchical patch encoding, and semantics-aware augmentation within a contrastive learning framework.

## Rating

- Novelty: ⭐⭐⭐⭐ — CGA is novel and practical; the combination of movement semantics and hierarchical encoding reflects thoughtful design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six research questions, multi-dataset evaluation, efficiency analysis, complete ablation and hyperparameter studies.
- Writing Quality: ⭐⭐⭐⭐ — Three limitations mapped to three design components; clear and coherent structure.
- Value: ⭐⭐⭐⭐ — Provides an efficient and semantically rich complete solution for trajectory representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](../../NeurIPS2025/object_detection/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)
- [\[CVPR 2026\] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](../../CVPR2026/object_detection/unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](../../ICLR2026/object_detection/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
