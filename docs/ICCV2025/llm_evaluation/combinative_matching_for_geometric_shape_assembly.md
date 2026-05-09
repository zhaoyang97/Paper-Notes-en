---
title: >-
  [Paper Note] Combinative Matching for Geometric Shape Assembly
description: >-
  [ICCV 2025][LLM Evaluation][shape assembly] This paper proposes Combinative Matching (CMNet), which jointly models two fundamental properties of interlocking parts — *surface shape consistency* and *volumetric occupancy complementarity* — via an equivariant network trained with three objectives: orientation alignment, shape matching, and occupancy matching, substantially reducing local ambiguity in geometric assembly.
tags:
  - ICCV 2025
  - LLM Evaluation
  - shape assembly
  - point cloud matching
  - equivariant networks
  - volumetric complementarity
  - optimal transport
date: 2026-05-08
content_hash: 481c4f8fc2f28d57
---

# Combinative Matching for Geometric Shape Assembly

**Conference**: ICCV 2025
**arXiv**: [2508.09780](https://arxiv.org/abs/2508.09780)
**Code**: [https://nahyuklee.github.io/cmnet](https://nahyuklee.github.io/cmnet)
**Area**: Human/Shape Understanding
**Keywords**: shape assembly, point cloud matching, equivariant networks, volumetric complementarity, optimal transport

## TL;DR

This paper proposes Combinative Matching (CMNet), which jointly models two fundamental properties of interlocking parts — *surface shape consistency* and *volumetric occupancy complementarity* — via an equivariant network trained with three objectives: orientation alignment, shape matching, and occupancy matching, substantially reducing local ambiguity in geometric assembly.

## Background & Motivation

Geometric shape assembly requires reconstructing a target object from multiple fragments, with broad applications in archaeology, medical imaging, robotics, and industrial manufacturing. Existing methods typically follow the point cloud registration paradigm — aligning parts by identifying visually similar surface regions. However, this approach suffers from **local ambiguity**: when surface appearances at different locations are similar, the model is prone to incorrect correspondences.

Inspired by architectural joinery (e.g., mortise-and-tenon joints, dovetail joints), the authors observe that stable assembly depends not only on surface visual similarity but, more critically, on **volumetric complementarity** between parts — where one part protrudes, the mating part must recess. Specifically, two corresponding points on an interlocking interface exhibit two properties: (1) **surface shape consistency** — identical local surface geometry; and (2) **volumetric occupancy complementarity** — the occupied space around one point is precisely the unoccupied space around the other. Prior methods exploit only property (1) while ignoring property (2), limiting matching accuracy.

## Method

### Overall Architecture

CMNet (Combinative Matching Network) consists of five components: (a) feature extraction and orientation alignment, (b) surface shape matching branch, (c) volume occupancy matching branch, (d) transformation estimation, and (e) training objectives. The input is a set of part point clouds subjected to random rigid-body transformations; the output is the transformation parameters for each part to reconstruct the target object.

### Key Designs

1. **Orientation Alignment**: An equivariant network VN-EdgeConv $f_d$ extracts equivariant features $\mathbf{F}_{\text{eqv}} \in \mathbb{R}^{K \times D \times 3}$, followed by VN-Linear and Gram-Schmidt orthogonalization to predict a per-point orientation matrix $\mathbf{F}_d \in \mathbb{R}^{K \times 3 \times 3}$ ($\in SO(3)$). Rotation-invariant features are obtained via $\mathbf{F}_{\text{inv}} = \mathbf{F}_{\text{eqv}} \cdot \mathbf{F}_d^\top$. The orientation loss is defined as: $\mathcal{L}_d = \frac{1}{|\mathcal{C}|}\sum_{(i,j)\in\mathcal{C}} \|(\mathbf{F}_d^P)_i \mathbf{R}^P - (\mathbf{F}_d^Q)_j \mathbf{R}^Q\|_F$, enforcing directional consistency at corresponding points. The key motivation is that the subsequent shape descriptor requires rotation invariance, while the occupancy descriptor requires directional consistency to enable complementary alignment.

2. **Surface Shape Matching Branch**: Taking rotation-invariant features as input, a three-layer MLP with LeakyReLU extracts shape descriptors $\mathbf{F}_s \in \mathbb{R}^{K \times d_s}$. A standard Circle Loss is employed so that positive pairs have $L_2$ distance below threshold $\Delta_p$ and negative pairs above $\Delta_n$. This branch directly models property (1), ensuring correct alignment of visually similar matching surfaces.

3. **Volume Occupancy Matching Branch**: Starting from the same rotation-invariant features, a separate three-layer MLP with Tanh extracts occupancy descriptors $\mathbf{F}_o \in \mathbb{R}^{K \times d_o}$. The key innovation is a Circle Loss variant using **cosine similarity**: for positive pairs, the occupancy descriptors are encouraged to be **opposite** ($s_{ij}^p = \|\hat{\mathbf{F}}_{o,i}^P + \hat{\mathbf{F}}_{o,j}^Q\|_2 \approx \cos(\mathbf{F}_{o,i}^P, \mathbf{F}_{o,j}^Q)$), reflecting that interlocking parts occupy complementary spaces. This directly models property (2) — the inverse of the similarity-maximization logic in conventional matching.

4. **Transformation Estimation**: A unified cost matrix is constructed as $\mathbf{C} = (\mathbf{F}_s^P \cdot \mathbf{F}_s^{Q\top} - \mathbf{F}_o^P \cdot \mathbf{F}_o^{Q\top}) / Z$, where the shape descriptor inner product measures similarity and the negated occupancy inner product also contributes as a similarity measure. One-to-one correspondences are obtained via an Optimal Transport (OT) layer; the top-128 correspondences are selected and a weighted SVD is applied to estimate the rigid transformation.

### Loss & Training

The total objective is $\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_s \mathcal{L}_s + \lambda_o \mathcal{L}_o + \mathcal{L}_p$, where $\lambda_d=0.1, \lambda_s=0.5, \lambda_o=0.5$, and $\mathcal{L}_p$ is a cross-entropy loss for point correspondence. The AdamW optimizer is used with an initial learning rate of $10^{-2}$, cosine scheduling, and training for 90/120 epochs on four RTX 3090 GPUs.

## Key Experimental Results

### Main Results

| Dataset/Subset | Method | CRD↓($10^{-2}$) | CD↓($10^{-3}$) | RMSE(R)↓(°) | RMSE(T)↓($10^{-2}$) |
|---|---|---|---|---|---|
| everyday | PMTR | 0.39 | 0.25 | 17.14 | 5.53 |
| everyday | **CMNet** | **0.28** | **0.17** | **12.88** | **3.78** |
| artifact | PMTR | 0.60 | 0.42 | 23.28 | 7.27 |
| artifact | **CMNet** | **0.49** | **0.34** | **18.77** | **5.57** |

CMNet consistently outperforms the previous state-of-the-art PMTR across all metrics, reducing CRD by 28% and rotation error by 25%.

### Ablation Study

| Configuration | CRD↓($10^{-2}$) | CD↓($10^{-3}$) | RMSE(R)↓(°) | Note |
|---|---|---|---|---|
| w/o equivariant network | 0.74 | 0.53 | 38.74 | Replaced by DGCNN; severe degradation |
| w/o shape matching | 0.38 | 0.28 | 13.17 | CRD increases by 35% |
| w/o occupancy matching | 0.35 | 0.25 | 14.01 | CRD increases by 25% |
| Full model | **0.28** | **0.17** | **12.88** | All three branches synergize optimally |
| L2 distance + no orientation loss | 0.42 | 0.31 | 14.88 | Cosine similarity + orientation loss is superior |
| Cosine + orientation loss | **0.28** | **0.17** | **12.88** | Best combination |

### Key Findings

- The learned orientation vectors automatically capture meaningful geometric properties: $\mathbf{x}_i$ points toward the center of the matching surface and is parallel to it; the direction of $\mathbf{y}_i$ reflects surface convexity/concavity and its magnitude reflects curvature — all without explicit supervision.
- t-SNE visualizations show that shape descriptors cluster tightly on matching surfaces (drawn together by $\mathcal{L}_s$), while occupancy descriptors are more dispersed (pushed apart by $\mathcal{L}_o$), validating the complementary roles of the two learning objectives.
- Correlation distribution analysis reveals that shape distributions alone exhibit local ambiguity regions; occupancy distributions alone lack precise localization; their combination produces significantly higher scores at true matching points, effectively eliminating ambiguity.
- CMNet maintains superior performance in cross-domain transfer experiments (everyday↔artifact), demonstrating strong generalization.

## Highlights & Insights

- Formalizing the engineering intuition of "male-female interlocking joints" as two learnable mathematical objectives is a concise and profound insight.
- The combined design of equivariant networks and invariant features is particularly elegant: orientation information and rotation-invariant features are jointly extracted from equivariant representations, satisfying the distinct requirements of shape matching (invariance) and occupancy matching (directional consistency).
- The unified metric in the cost matrix $\mathbf{C}$ — "similarity minus dissimilarity" — is a mathematically elegant formulation.

## Limitations & Future Work

- Evaluation is currently conducted on the Breaking Bad dataset; robustness to the greater noise and incompleteness present in real-world fragment scenarios requires further validation.
- Multi-part assembly inherits the scheme from PMTR, with global consistency constraints remaining underexplored.
- For severely asymmetric fragments (e.g., thin slabs), volumetric occupancy differences may not be sufficiently discriminative.

## Related Work & Insights

- This work extends the primal-dual descriptor idea from Jigsaw, but more explicitly disentangles "surface shape" and "volumetric occupancy" with clear physical semantics.
- The application of equivariant networks (VN-DGCNN) is well established, but the design for jointly learning orientation and invariant dual features is novel.
- The use of optimal transport for correspondence estimation has become standard practice.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of combinative matching is clearly articulated and physically grounded
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies, visualizations, and cross-domain experiments are comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear, with excellent figures and naturally motivated problem formulation
- Value: ⭐⭐⭐⭐ Offers a substantive contribution to the shape assembly field

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](../../AAAI2026/llm_evaluation/dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](../../NeurIPS2025/llm_evaluation/learning_generalizable_shape_completion_with_sim3_equivariance.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](../../CVPR2026/llm_evaluation/unified_primitive_proxies_for_structured_shape_completion.md)
- [\[ICCV 2025\] Generative Zoo](generative_zoo.md)

<!-- RELATED:END -->
