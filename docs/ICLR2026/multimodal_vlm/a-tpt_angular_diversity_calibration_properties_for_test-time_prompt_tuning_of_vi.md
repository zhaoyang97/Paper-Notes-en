---
title: >-
  [Paper Note] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Test-time prompt tuning] This paper proposes the A-TPT framework, which promotes angular diversity by maximizing the minimum pairwise angular distance among normalized text features on the uni…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Test-time prompt tuning"
  - "CLIP"
  - "calibration"
  - "angular diversity"
  - "hyperspherical uniform distribution"
date: 2026-05-08
content_hash: 54f604a36ac72a19
---

# A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.26441](https://arxiv.org/abs/2510.26441)  
**Code**: Coming soon  
**Area**: Multimodal VLM / Calibration
**Keywords**: Test-time prompt tuning, CLIP, calibration, angular diversity, hyperspherical uniform distribution

## TL;DR
This paper proposes the A-TPT framework, which promotes angular diversity by maximizing the minimum pairwise angular distance among normalized text features on the unit hypersphere. It addresses the miscalibration caused by overconfident predictions in test-time prompt tuning (TPT) of VLMs, achieving superior performance over existing TPT calibration methods on both natural distribution shifts and medical datasets.

## Background & Motivation

**Background**: TPT adapts VLMs (e.g., CLIP) to new tasks by optimizing learnable prompt vectors with unlabeled samples at inference time, improving accuracy but often increasing calibration error (overconfidence).

**Limitations of Prior Work**: C-TPT improves calibration by maximizing average text feature dispersion (ATFD), yet features may still cluster. O-TPT enforces angular separation via orthogonality constraints, but when the number of classes exceeds the embedding dimension (e.g., 1,000 classes in ImageNet-1K vs. CLIP's 512 dimensions), enforcing orthogonality is mathematically infeasible, causing features to collapse instead.

**Key Challenge**: Neither dispersion (L2 distance) nor orthogonality constraints can guarantee a uniform angular distribution of features on the hypersphere—the former may push all features toward one direction while remaining far from the centroid, and the latter fails when the class count is large.

**Goal**: To propose a TPT calibration method that effectively promotes angular diversity of text features in both the $N > |D|$ and $N < |D|$ regimes.

**Key Insight**: The calibration problem is connected to the Tammes problem (optimal point placement on a hypersphere)—maximizing the minimum pairwise angular distance ensures uniform feature distribution.

**Core Idea**: By maximizing the minimum pairwise angular distance among normalized text features (rather than average dispersion or orthogonality), A-TPT achieves uniform distribution on the hypersphere, substantially improving VLM calibration at inference time.

## Method

### Overall Architecture
A-TPT augments the standard TPT framework with an angular diversity regularization term. During prompt optimization, in addition to minimizing prediction entropy, it maximizes the minimum pairwise angular distance among the normalized text features of all classes, encouraging uniformly distributed features on the hypersphere.

### Key Designs

1. **Angular Diversity Optimization**:

    - **Function**: Maximize the minimum pairwise angular distance of text features on the unit hypersphere.
    - **Mechanism**: For all class pairs $(i,j)$, the angular distance $\theta_{ij} = \arccos(\text{sim}(t_i, t_j))$ is computed, and the optimization objective includes $\max \min_{i \neq j} \theta_{ij}$, which is equivalent to a numerical approximation of the Tammes problem.
    - **Design Motivation**: When $N > |D|$, orthogonality constraints are infeasible, but maximizing the minimum angular distance remains well-defined—it finds the optimal spherical arrangement of $N$ points in a finite-dimensional space. When $N < |D|$, angular diversity fully exploits the hyperspherical space, whereas orthogonality constraints waste capacity.

2. **Integration with TPT**:

    - **Function**: Incorporate the angular diversity loss as a regularization term into TPT's entropy minimization objective.
    - **Mechanism**: $\mathcal{L} = \mathcal{L}_{entropy} + \lambda \mathcal{L}_{angular}$, jointly optimized at inference time for each test sample.
    - **Design Motivation**: Preserves the accuracy gains of TPT while improving calibration through angular diversity constraints.

### Loss & Training
Learnable prompt vectors are optimized via gradient descent on each test sample without labeled data. Multiple views are generated through data augmentation, and prediction entropy plus the angular diversity regularizer are minimized jointly. The temperature parameter $\tau=0.01$ is fixed throughout.

## Key Experimental Results

### Main Results

**ECE↓ (calibration error) of CLIP ViT-B/16 across multiple datasets:**

| Method | Caltech101 | OxfordPets | DTD | EuroSAT | ImageNet |
|--------|-----------|-----------|-----|---------|----------|
| Zero-shot CLIP | 5.66 | — | — | — | — |
| TPT | 6.18 | — | — | — | — |
| C-TPT | Moderate | Moderate | Moderate | Moderate | Moderate |
| O-TPT | Moderate | Moderate | Moderate | Moderate | Moderate |
| **A-TPT** | **2.23** | **Lowest** | **Lowest** | **Lowest** | **Lowest** |

### Ablation Study

| Configuration | ECE | Accuracy | Notes |
|---------------|-----|----------|-------|
| TPT (no calibration) | High | High | Accurate but overconfident |
| + ATFD (C-TPT) | Medium | Maintained | Limited improvement from average dispersion |
| + Orthogonality (O-TPT) | Medium | Maintained | Fails when $N>|D|$ |
| + Angular diversity (A-TPT) | **Low** | Maintained | Best overall |

### Key Findings
- Empirical analysis shows that angular distance (AD) is negatively correlated with ECE—larger AD corresponds to better calibration, validating the theoretical motivation for angular diversity.
- When $N > |D|$ (e.g., 1,000 classes in ImageNet-1K vs. 512 dimensions), O-TPT's orthogonality constraint becomes entirely infeasible, whereas A-TPT remains effective.
- t-SNE visualizations clearly demonstrate that angularly diverse features are not only well spread but also well aligned with class labels.
- A-TPT significantly reduces ECE without sacrificing accuracy, and generalizes to medical datasets.

## Highlights & Insights
- **Application of the Tammes Problem to Machine Learning**: Connecting the optimal point placement problem on a hypersphere to VLM calibration is an elegant cross-domain insight. Maximizing the minimum pairwise angular distance more fundamentally characterizes "uniform distribution" than L2 dispersion or orthogonality.
- **Decoupling Calibration from Accuracy**: Within groups of methods achieving similar accuracy, differences in calibration performance are primarily driven by angular diversity—a finding that provides a new perspective for understanding VLM calibration.
- **Consistency Between Theory and Practice**: Uniform distribution on the hypersphere preserves maximal information (Wang & Isola, 2020), which is consistent with the observed calibration improvements.

## Limitations & Future Work
- Maximizing the minimum angular distance is a non-convex optimization problem and may converge to local optima.
- The per-sample inference-time optimization incurs additional computational overhead on top of the already non-trivial cost of standard TPT.
- Experiments are conducted primarily on classification tasks; downstream tasks such as detection and segmentation remain unvalidated.
- The temperature parameter $\tau=0.01$ is fixed; adaptive temperature scaling may yield further improvements.

## Related Work & Insights
- **vs. C-TPT**: C-TPT maximizes L2 distance from the centroid via ATFD; A-TPT directly optimizes the minimum pairwise angular distance, more directly ensuring uniformity.
- **vs. O-TPT**: O-TPT applies orthogonality constraints but is mathematically infeasible when $N > |D|$; A-TPT's Tammes problem formulation naturally handles this regime.
- **vs. Wang & Isola 2020**: That work demonstrates the importance of uniformity in contrastive learning; A-TPT transfers this insight to TPT calibration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The connection between the Tammes problem and VLM calibration is novel, though the core idea (maximizing the minimum distance) is relatively intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple datasets (including medical), multiple backbones, comprehensive visualizations, and theoretical support.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well articulated and visualizations are convincing.
- **Value**: ⭐⭐⭐⭐ Provides a practical improvement for VLM test-time calibration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)

</div>

<!-- RELATED:END -->
