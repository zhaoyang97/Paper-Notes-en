---
title: >-
  [Paper Note] A Unified Interpretation of Training-Time Out-of-Distribution Detection
description: >-
  [ICCV 2025][3D Vision][OOD detection] This paper proposes a novel perspective based on inter-variable "interactions" to provide a unified explanation for why different training-time OOD detection methods are effective —…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "OOD detection"
  - "interaction complexity"
  - "training-time methods"
  - "high-order interactions"
  - "interpretability"
date: 2026-05-08
content_hash: de3b6d414526b6cc
---

# A Unified Interpretation of Training-Time Out-of-Distribution Detection

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: OOD detection, interaction complexity, training-time methods, high-order interactions, interpretability

## TL;DR

This paper proposes a novel perspective based on inter-variable "interactions" to provide a unified explanation for why different training-time OOD detection methods are effective — they all encourage the model to encode more high-order interactions. The paper further validates the dominant role of high-order interactions in OOD detection and explains, through interaction distribution analysis, why near-OOD samples are harder to detect.

## Background & Motivation

**Why is a unified understanding needed?** Existing training-time OOD detection methods (LogitNorm, T2FNorm, CSI, DAL, etc.) are designed based on different intuitions: some apply logit normalization, some apply feature normalization, and some incorporate data augmentation with distribution shifts. Although all are effective, **no prior work has explained whether these design-diverse methods share a common underlying mechanism**. Understanding this shared mechanism is crucial for designing better OOD detection approaches.

**Why adopt the "interaction" perspective?** Prior theoretical work by Ren et al. proved that the output of a DNN can be faithfully decomposed into a sum of interaction effects over different subsets $S$ of input variables: $v(x) = \sum_S I(S|x)$. This decomposition satisfies **sparsity** (only a few interactions are significant), **universal matching** (it can match the outputs of all $2^n$ masked samples), and **generalization** (interaction patterns are similar across samples of the same class). These mathematical guarantees make interactions a reliable tool for explaining DNN reasoning logic.

**Why are near-OOD samples harder to detect?** This is an important but largely unexplored question. While the notion of domain shift is intuitive, it lacks a rigorous quantitative explanation.

## Method

### Overall Architecture

The technical contributions of this paper can be organized into three levels:
1. **Unified interpretation**: All training-time methods are found to encode more high-order interactions.
2. **Causal validation**: Loss functions are designed to force models to learn interactions of specific orders, validating the dominant role of high-order interactions.
3. **Near-OOD explanation**: Interaction distribution similarity is used to explain why near-OOD samples are harder to detect.

### Key Designs

#### Mathematical Definition of Interactions

Given a DNN $v: \mathbb{R}^n \to \mathbb{R}$ with input $x$ consisting of $n$ variables, the network output is set to the logit of the ground-truth class:

$$v(x) = \log \frac{p(y=y_{truth}|x)}{1 - p(y=y_{truth}|x)}$$

The interaction effect of a variable subset $S \subseteq N$ is defined via the Harsanyi Dividend:

$$I(S|x) = \sum_{T \subseteq S} (-1)^{|S|-|T|} v(x_T)$$

where $x_T$ denotes the sample with variables in $N \setminus T$ masked to baseline values. The **order** (complexity) of an interaction is defined as $|S|$.

**Intuition**: Low-order interactions encode simple features (e.g., small patches of blue sky background) that appear in both ID and OOD samples, thus offering weak discriminative power. High-order interactions encode complex AND-relationships (activated only when all constituent variables co-occur simultaneously) and thus carry stronger discriminative power.

#### Unified Understanding of Training-Time OOD Detection Methods

The difference in $m$-th order interaction strength between an enhanced model $v_{enhance}$ (trained with an OOD detection method) and a baseline model $v_{baseline}$ (trained with cross-entropy only) is computed as:

$$\Delta R^{(m)} = R^{(m)}_{enhance} - R^{(m)}_{baseline}$$

where the relative interaction strength $R^{(m)}$ measures the proportion of $m$-th order interactions relative to total interaction strength.

**Key finding**: Across all tested enhancement methods (CSI, LogitNorm, T2FNorm, DAL) and all tested architectures (ResNet-18, ResNet-34, WideResNet-40-2), it is consistently observed that $\Delta R^{(m)} > 0$ when $m > 0.75n$ and $\Delta R^{(m)} < 0$ when $m < 0.25n$. That is, **enhanced models consistently encode more high-order and fewer low-order interactions**.

#### Validating the Dominant Role of High-Order Interactions

**Why not directly remove high-order interactions and measure the effect?** Direct removal is infeasible. Instead, the authors devise an alternative approach — controlling the order of interactions encoded by the model indirectly via a loss function.

Based on Theorem 2, the network output variation $\Delta v^{(m_1, m_2)}$ primarily encodes interactions of orders in $[0, m_2 n]$. A penalty loss is thus designed:

$$L^{(m_1, m_2)}_{inter} = -\mathbb{E}_x \left[ \sum_{c=1}^C p(\hat{y}=c | \Delta v^{(m_1, m_2)}_c(x)) \log p(\hat{y}=c | \Delta v^{(m_1, m_2)}_c(x)) \right]$$

The total loss is: $L = L_{ce} - \alpha L^{(m_1, m_2)}_{inter}$

Setting $[m_1=0.7, m_2=1.0]$ penalizes high-order interactions, yielding a "low-order model"; setting $[m_1=0, m_2=0.3]$ penalizes low-order interactions, yielding a "high-order model."

#### Explaining Near-OOD via Interaction Distribution

The Jaccard similarity is used to compare the interaction distributions between ID and near-OOD/far-OOD samples:

$$SIM_{near} = \frac{\| \min(\tilde{I}_{ID}(v), \tilde{I}_{near\text{-}OOD}(v)) \|_1}{\| \max(\tilde{I}_{ID}(v), \tilde{I}_{near\text{-}OOD}(v)) \|_1}$$

### Loss & Training

The training strategy itself is not the primary contribution of this paper; rather, the aforementioned loss functions serve as tools for **analytical experiments**. Three types of models are trained:
- Baseline model: trained with $L_{ce}$ only ($\alpha = 0$)
- Low-order model: $[m_1=0.7, m_2=1.0]$, $\alpha = 0.1$
- High-order model: $[m_1=0, m_2=0.3]$, $\alpha = 0.1$

## Key Experimental Results

### Main Results

**Effect of high-order interactions on OOD detection (Table 1, averaged over 4 OOD datasets):**

| ID Dataset | Model Type | ResNet-18 FPR95↓ | ResNet-18 AUROC↑ | ResNet-34 FPR95↓ | ResNet-34 AUROC↑ | WRN-40-2 FPR95↓ | WRN-40-2 AUROC↑ |
|-----------|-----------|------------------|------------------|------------------|------------------|-----------------|-----------------|
| CIFAR-10 | Baseline | 62.03 | 88.48 | 50.09 | 89.12 | 56.99 | 89.02 |
| CIFAR-10 | Low-order | 91.45 (+29.4) | 73.07 (-15.4) | 88.63 (+38.5) | 69.29 (-19.8) | 85.13 (+28.1) | 70.16 (-18.9) |
| CIFAR-10 | High-order | 53.05 (-9.0) | 89.53 (+1.1) | 51.32 (+1.2) | 88.97 (-0.2) | 61.64 (+4.7) | 86.63 (-2.4) |
| CIFAR-100 | Baseline | 79.70 | 78.15 | 78.95 | 78.30 | 78.01 | 76.90 |
| CIFAR-100 | Low-order | 92.69 (+13.0) | 51.52 (-26.6) | 89.98 (+11.0) | 58.77 (-19.5) | 90.33 (+12.3) | 54.46 (-22.4) |
| CIFAR-100 | High-order | 75.82 (-3.9) | 79.45 (+1.3) | 81.51 (+2.6) | 77.28 (-1.0) | 82.51 (+4.5) | 73.92 (-3.0) |

**Core conclusion**: The OOD detection performance of low-order models **degrades catastrophically** (FPR95 increases by 13–38 points), whereas high-order models show only marginal degradation or even slight improvement. This strongly demonstrates the dominant role of high-order interactions in OOD detection.

### Ablation Study

**Interaction distribution similarity analysis (Figure 5):**

| Comparison | Similarity Range | Conclusion |
|-----------|-----------------|------------|
| $SIM_{near}$ (near-OOD vs. ID) | 0.4–0.8 | Relatively high similarity |
| $SIM_{far}$ (far-OOD vs. ID) | 0.0–0.2 | Very low similarity |
| $SIM_{near,enhance}$ | < $SIM_{near}$ | Enhancement methods reduce similarity |
| $SIM_{far,enhance}$ | < $SIM_{far}$ | Enhancement methods reduce similarity |

These conclusions are **entirely consistent** across three architectures (ResNet-18/34/WRN-40-2), two ID datasets (CIFAR-10/100), and four enhancement methods (CSI/LogitNorm/T2FNorm/DAL).

### Key Findings

1. **Different training-time methods share the same underlying mechanism**: Despite different design motivations, all methods encourage the model to encode more high-order interactions.
2. **High-order interactions are the core factor in OOD detection**: Removing high-order interactions causes catastrophic performance degradation.
3. **The fundamental reason near-OOD is harder to detect**: Its interaction distribution is more similar to that of ID samples.
4. **Enhancement methods reduce interaction distribution similarity**: This explains why they improve near-OOD detection.

## Highlights & Insights

- **From correlation to causation**: The paper not only observes the association between high-order interactions and OOD performance, but also performs causal validation by designing targeted loss functions.
- **Solid theoretical foundation**: The approach is grounded in the Harsanyi Dividend and proven theorems on interaction sparsity and universal matching.
- **Consistency across methods, architectures, and datasets**: The consistent conclusions across 4 methods × 3 architectures × 5 datasets (60 experimental groups) substantially enhance credibility.
- **Practical implication**: Future OOD detection methods can directly target "increasing high-order interactions" as a design objective.

## Limitations & Future Work

- Validation is limited to the ResNet family; modern architectures such as ViTs are not covered.
- Only training-time methods are examined; post-hoc OOD detection methods (MSP, ODIN, etc.) are not considered.
- Interaction computation is costly (exponential subset enumeration), limiting practical applicability.
- The improvement in OOD performance from high-order models is modest, suggesting that increasing high-order interactions alone is insufficient for substantial gains.
- The underlying reasons why training-time methods encode more high-order interactions are not deeply analyzed.

## Related Work & Insights

This paper follows a line of interaction-analysis work (Li and Zhang 2023, Ren et al.), but is the first to apply the interaction perspective to the interpretation of OOD detection. Unlike explanations offered by Kirichenko et al. (normalizing flows fail to learn semantic representations) and Du et al. (label information facilitates OOD detection), the interaction perspective provides a more unified and quantitative explanatory framework. The key insight is that **identifying shared underlying mechanisms across methods** is more conducive to advancing the field than improving individual methods in isolation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](../../NeurIPS2025/3d_vision/metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](4d_visual_pretraining_for_robot_learning.md)
- [\[ICCV 2025\] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)

</div>

<!-- RELATED:END -->
