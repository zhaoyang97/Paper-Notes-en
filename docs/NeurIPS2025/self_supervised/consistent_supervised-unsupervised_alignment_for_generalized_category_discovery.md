---
title: >-
  [Paper Note] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Generalized Category Discovery] Proposes the NC-GCD framework, which establishes a unified optimization objective for both known and novel classes by pre-allocating fixed Equiangular Tight Frame (ETF) prototypes, and stabilizes pseudo-label assignment across iterations using a Semantic Consistency Matcher (SCM), significantly improving novel class discovery accuracy on 6 GCD benchmarks.
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Generalized Category Discovery"
  - "Neural Collapse"
  - "ETF Prototype"
  - "Clustering Alignment"
  - "Pseudo-label Consistency"
date: 2026-05-08
content_hash: 9fa0f37d9d9632ca
---

# Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery

**Conference**: NeurIPS 2025  
**arXiv**: [2507.04725](https://arxiv.org/abs/2507.04725)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Generalized Category Discovery, Neural Collapse, ETF Prototype, Clustering Alignment, Pseudo-label Consistency

## TL;DR
Proposes the NC-GCD framework, which establishes a unified optimization objective for both known and novel classes by pre-allocating fixed Equiangular Tight Frame (ETF) prototypes, and stabilizes pseudo-label assignment across iterations using a Semantic Consistency Matcher (SCM), significantly improving novel class discovery accuracy on 6 GCD benchmarks.

## Background & Motivation

**Background**: Generalized Category Discovery (GCD) aims to classify known categories and discover novel categories simultaneously. Existing methods, such as SimGCD and CMS, organize the feature space by dynamically learning clustering prototypes or classifier weights.

**Limitations of Prior Work**: Dynamically optimizing prototypes leads to two key issues: (a) inconsistent optimization objectives for known and novel classes, causing the model to bias toward labeled known classes while ignoring the decision boundaries of novel classes; (b) a lack of geometric constraints on feature distribution, making novel classes entirely dependent on unsupervised optimization and highly prone to overlapping with feature-similar categories (class confusion).

**Key Challenge**: The inequality in optimization objectives between novel and known classes prevents the model from equally separating all categories under a unified geometric structure.

**Goal**: Can an optimal geometric structure be pre-allocated so that known and novel classes are equidistantly separated in the feature space, and consistent learning is achieved through a unified alignment loss?

**Key Insight**: Neural Collapse theory indicates that the class feature means of a well-trained classification network converge to a Simplex ETF structure, maximizing inter-class separation and minimizing intra-class variance. The authors transition this theory from a "post-training emergence phenomenon" to a "pre-training prior constraint."

**Core Idea**: Pre-fix ETF prototypes as anchors for all categories, providing a consistent optimization direction for GCD through unified supervised and unsupervised alignment losses.

## Method

### Overall Architecture
NC-GCD consists of four core components: (1) a pre-trained vision encoder $f(\cdot)$ (DINO ViT-B/16), (2) a periodic clustering module $g(\cdot)$, (3) a pre-allocated ETF prototype set $P$, and (4) a Semantic Consistency Matcher $\phi_{\text{SCM}}(\cdot)$. Input images are processed by the encoder to extract embeddings, periodic clustering groups all samples, and high-confidence samples are pulled toward their corresponding ETF prototypes.

### Key Designs

1. **Pre-allocated ETF Prototypes**:

    - Function: Generate fixed equiangular tight frame prototypes $\mathbf{P} = \{p_1, \dots, p_K\}$ before training
    - Mechanism: ETF is constructed via $P = \sqrt{\frac{K}{K-1}} U (I_K - \frac{1}{K} \mathbf{1}_K \mathbf{1}_K^\top)$, satisfying $p_k^\top p_j = \frac{K}{K-1}\delta_{k,j} - \frac{1}{K-1}$ to guarantee maximal uniform separation of all categories
    - Design Motivation: Fixed ETF provides a globally optimal geometric configuration, eliminating optimization inconsistency between known and novel classes

2. **Unsupervised ETF Alignment**:

    - Function: Perform clustering every $T$ epochs, selecting the top-$\alpha\%$ high-confidence samples in each cluster to align with ETF
    - Mechanism: High-confidence samples are aligned with prototypes via Dot-Regression Loss: $\mathcal{L}_{\text{ETF}}^u = \frac{1}{|\tilde{D}_k|} \sum_{e_i \in \tilde{D}_k} \|e_i - p_k\|^2$
    - Design Motivation: Using only high-confidence samples avoids interference from noisy pseudo-labels

3. **Supervised ETF Alignment**:

    - Function: Align labeled sample features with SCM-mapped ETF prototypes
    - Mechanism: $\mathcal{L}_{\text{ETF}}^s = \frac{1}{|\mathcal{D}^l|} \sum \|e_i^l - p_a\|^2$, where $a = \phi_{\text{SCM}}(y_i^l)$
    - Design Motivation: Requires SCM to guarantee the correctness of the mapping from ground-truth labels to ETF

4. **Semantic Consistency Matcher (SCM)**:

    - Function: Ensure pseudo-label consistency across clustering iterations
    - Mechanism: Optimal permutation $\sigma^* = \arg\max_{\sigma \in S_K} \sum_{k} \sum_{i} \mathbb{I}(\hat{y}_i^t = k)\mathbb{I}(\hat{y}_i^{t-1} = \sigma(k))$ is solved, achieving a one-to-one label mapping using the Hungarian algorithm
    - Design Motivation: Periodic clustering is unstable; SCM eliminates fluctuations by enforcing a one-to-one mapping

### Loss & Training

Unified ETF Loss: $\mathcal{L}_{\text{ETF}} = (1-\gamma)\mathcal{L}_{\text{ETF}}^u + \gamma\mathcal{L}_{\text{ETF}}^s$

Representation Learning: $\mathcal{L}_{\text{REP}} = (1-\lambda)\mathcal{L}_{\text{REP}}^u + \lambda\mathcal{L}_{\text{REP}}^s$

Final Loss: $\mathcal{L} = \beta\mathcal{L}_{\text{ETF}} + \mathcal{L}_{\text{REP}}$

## Key Experimental Results

### Main Results (DINOv1, GT K Known)

| Dataset | NC-GCD (All/Old/New) | SPT (All/Old/New) | CMS (All/Old/New) |
|--------|---------------------|-------------------|-------------------|
| CUB-200 | 74.8/76.8/73.8 | 65.8/68.8/65.1 | 68.2/76.5/64.0 |
| Stanford Cars | 59.9/77.8/51.2 | 59.0/79.2/49.3 | 56.9/76.1/47.6 |
| FGVC Aircraft | 60.0/57.6/61.2 | 59.3/61.8/58.1 | 56.0/63.4/52.3 |
| ImageNet-100 | 88.4/94.1/85.5 | 85.4/93.2/81.4 | 84.7/95.6/79.2 |
| CIFAR-100 | 82.7/85.5/77.3 | 81.3/84.3/75.6 | 82.3/85.7/75.5 |

### Overall Average (GT K Known)

| Method | Fine-grained All/New | Overall All/New |
|------|---------------|-------------|
| SPT | 56.9/51.9 | 65.7/60.8 |
| CMS | 54.4/47.6 | 64.1/57.5 |
| **NC-GCD** | **60.3/56.7** | **68.7/64.9** |

### Key Findings
- Most significant accuracy gain in novel classes: Novel class accuracy on fine-grained datasets increases by an average of +4.8% (vs. SPT), proving that fixed ETF effectively alleviates under-separation of novel classes.
- Robust even without GT K, demonstrating that the ETF framework is tolerant to K estimation errors.
- Achieves 88.4% on all classes for ImageNet-100, which is 3.0% higher than the runner-up SPT.

## Highlights & Insights
- **Neural Collapse: From Phenomenon to Prior**: Transitioning NC from a post-training emergence to a pre-training structural constraint, which can be extended to incremental learning and federated learning.
- **SCM Hungarian Matching**: Solves pseudo-label drift across iterations via optimal assignment, being simple yet effective.

## Limitations & Future Work
- Requires estimating the number of classes K; a bias in K affects the geometric quality of the ETF.
- Validated only on DINO ViT-B/16, leaving DINOv2 or larger models unexplored.
- The high-confidence threshold $\alpha$ requires manual tuning.

## Related Work & Insights
- **vs. SimGCD**: SimGCD dynamically learns prototypes, leading to inconsistent objectives between known and novel classes; NC-GCD eliminates this issue via fixed ETF.
- **vs. TRAILER**: TRAILER also uses a fixed classifier, but its cross-entropy ETF loss might introduce bias; NC-GCD separates supervised and unsupervised alignment.
- **vs. CMS**: CMS focuses on contrastive mean-shift, while NC-GCD simultaneously optimizes feature geometry.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Neural Collapse into GCD is a fresh perspective, though ETF has already been applied in other fields.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks, two K settings, and comparison with 10+ methods.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational derivation, with unified mathematical symbols and notation.
- Value: ⭐⭐⭐⭐ Provides a structured geometric prior for GCD, achieving a significant accuracy gain in novel classes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [\[ICLR 2026\] Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery](../../ICLR2026/self_supervised/bures-isotropy_alignment_manifold_learning_of_generalized_category_discovery.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/self_supervised/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](../../CVPR2025/self_supervised/mos_modeling_object-scene_associations_in_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
