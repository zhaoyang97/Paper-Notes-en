---
title: >-
  [Paper Note] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection
description: >-
  [NeurIPS 2025][AI Safety][Spurious correlations] SPROD is a post-hoc OOD detection method designed to handle spurious correlations in training data. It subdivides each class prototype into "correctly classified" and "misclassified" subgroups (the latter sharing spurious features), combined with K-means-style refinement and distance-based (generative) scoring. Across 5 spurious-correlation OOD benchmarks, it achieves an average AUROC of 85.1% (+4.8% vs. runner-up KNN) and FPR@…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Spurious correlations"
  - "prototype refinement"
  - "generative scoring"
  - "out-of-distribution detection"
  - "K-means"
date: 2026-05-08
content_hash: 11a068ab5596507d
---

# SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection

**Conference**: NeurIPS 2025
**arXiv**: [2506.23881](https://arxiv.org/abs/2506.23881)  
**Code**: To be confirmed  
**Area**: LLM Evaluation
**Keywords**: Spurious correlations, prototype refinement, generative scoring, out-of-distribution detection, K-means

## TL;DR
SPROD is a post-hoc OOD detection method designed to handle spurious correlations in training data. It subdivides each class prototype into "correctly classified" and "misclassified" subgroups (the latter sharing spurious features), combined with K-means-style refinement and distance-based (generative) scoring. Across 5 spurious-correlation OOD benchmarks, it achieves an average AUROC of 85.1% (+4.8% vs. runner-up KNN) and FPR@95 of 49.0% (−9.3% vs. runner-up).

## Background & Motivation

**Background**: OOD detection methods (MSP, Energy, KNN, MDS, etc.) perform well on standard benchmarks, but spurious correlations in training data (e.g., "waterbirds always appear on water backgrounds") severely degrade detection performance—models may incorrectly classify OOD samples sharing spurious features as in-distribution.

**Limitations of Prior Work**: (a) SP-OOD samples (sharing spurious features but lacking core features) are particularly difficult to detect, as models "recognize" them via spurious features; (b) existing post-hoc methods are unaware of spurious correlations in training data; (c) fine-tuning the backbone further entangles the feature space with spurious features.

**Key Challenge**: Effective OOD detection requires reliance on core (causal) features, yet spurious features are learned equally or more strongly during training. Distinguishing the influence of core vs. spurious features is needed without modifying the model.

**Goal**: A post-hoc method that automatically discovers spurious subgroups in training data and constructs spurious-aware prototypes for OOD detection.

**Key Insight**: Misclassified training samples may share spurious features (e.g., "landbird" samples misclassified as "waterbird" may be misclassified due to water-background spurious features). Misclassification patterns are exploited to automatically discover spurious subgroups.

**Core Idea**: Misclassification analysis reveals spurious subgroups → multiple intra-class prototypes (one per correct/misclassified subgroup) → K-means refinement → distance-based (generative) OOD scoring.

## Method

### Overall Architecture
Frozen backbone → **Stage 1**: Normalize embeddings → compute mean prototype per class → **Stage 2**: Classify training set with prototypes → correctly classified group $\mathcal{S}_c^{corr}$ + misclassification-direction groups $\mathcal{S}_{c \to m}^{misc}$ → one prototype per group (expanded to at most $C^2$ prototypes) → **Stage 3**: K-means-style reassignment to refine prototypes → nearest prototype distance as OOD score.

### Key Designs

1. **Misclassification-Driven Subgroup Discovery (Stage 2)**:

    - Function: Automatically identify sample subgroups sharing spurious features.
    - Mechanism: Training samples are classified using Stage 1 prototypes. Samples in class $c$ misclassified as class $m$, denoted $\mathcal{S}_{c \to m}^{misc}$, likely share spurious features of class $m$. One prototype is computed per $(c, m)$ pair, expanding each class from 1 prototype to at most $C$ (one per misclassification direction plus one for correct classifications).
    - Design Motivation: Misclassification patterns serve as natural signals for spurious correlations—no domain labels or prior knowledge are required.

2. **K-means Refinement (Stage 3)**:

    - Function: Reassign samples to the nearest subgroup prototype.
    - Mechanism: Within each class, samples are reassigned by distance to either the majority prototype (near correctly classified prototype) or minority prototypes (near misclassified prototypes), followed by prototype recomputation—analogous to one step of K-means.
    - Design Motivation: Stage 2 misclassifications may be noisy (some misclassifications are not due to spurious features); K-means refinement purifies the subgroups.

3. **Generative (Distance-Based) vs. Discriminative (Softmax) Scoring**:

    - Function: Use nearest prototype distance rather than softmax probability for OOD judgment.
    - Mechanism: OOD score = minimum distance to all prototypes. Theoretical motivation: generative scoring $p(z|y)$ is more robust under distribution shift than discriminative scoring $p(y|z)$—spurious correlations cause $p_{train}(y|z) \neq p_{test}(y|z)$, while $p(z|y)$ may remain more stable.
    - Design Motivation: Ablations show generative AUROC of 85.1% vs. discriminative 71.8%—a substantial gap.

### Loss & Training
- Fully post-hoc; no training is required.
- Only backbone features and class labels are needed.
- ResNet-50 backbone (frozen).

## Key Experimental Results

### Main Results (5 Spurious-Correlation OOD Benchmarks)

| Dataset | SPROD AUROC | Runner-up (KNN) AUROC | SPROD FPR@95 |
|--------|------------|-----------------|-------------|
| Waterbirds | **98.8%** | 97.2% | 4.7% |
| CelebA | 61.6% | 56.3% | 93.7% |
| UrbanCars | **97.4%** | 92.1% | 19.0% |
| Animals MetaCoCo | **82.4%** | 78.5% | 69.5% |
| Spurious ImageNet100 | **85.3%** | 77.4% | 58.0% |
| **Average** | **85.1%** | 80.3% | **49.0%** |

Compared to 19 baseline methods: average AUROC +4.8%, FPR@95 −9.3%.

### Ablation Study

| Configuration | AUROC | FPR@95 |
|------|-------|--------|
| Generative scoring | **85.1%** | **49.0%** |
| Discriminative scoring | 71.8% | — |
| Fine-tuned backbone | Degraded | — |
| Spurious correlation rate 50% vs. 90% | All degrade; SPROD degrades least | — |
| Zero-shot (no text, vision-only) | Waterbirds **99.01%** | — |

### Key Findings
- The gap between generative and discriminative scoring is large (85.1% vs. 71.8%)—discriminative scoring is unreliable under spurious correlations.
- SPROD is the only method ranked in the top two across all 5 spurious-correlation benchmarks.
- Fine-tuning the backbone is harmful—it increases reliance on spurious features.
- SPROD matches KNN on standard OOD benchmarks (CIFAR-10/100, ImageNet)—no trade-off in conventional performance.
- Higher spurious correlation rates degrade all methods, but SPROD degrades the least.

## Highlights & Insights
- **Misclassifications as spurious signals** is a simple yet profound insight: the errors the model makes directly expose the spurious features it relies upon.
- **Robustness of generative scoring** is theoretically grounded: $p(z|y)$ is more stable under distribution shift than $p(y|z)$.
- **Fully post-hoc with no additional annotation**: only a backbone and class labels are required—no domain labels or prior knowledge of spurious correlations—making the method highly practical.

## Limitations & Future Work
- Class labels are required (a standard assumption, but not fully unsupervised).
- Performance on CelebA is relatively poor (61.6%)—noisy labels and subtle spurious features are harder to handle.
- Not consistently optimal on standard settings—a trade-off between worst-group robustness and average performance exists.
- A rigorous theoretical proof of generative scoring robustness remains to be completed.

## Related Work & Insights
- **vs. KNN**: KNN is the strongest baseline (80.3%); SPROD surpasses it by 4.8% under spurious correlations—the key is that multiple prototypes provide subgroup awareness.
- **vs. MDS (Mahalanobis)**: MDS assumes intra-class Gaussian distributions, an assumption violated by spurious subgroups.
- **vs. fine-tuning methods (CMA, etc.)**: Fine-tuning deepens spurious dependence; post-hoc methods are safer.

## Rating
- Novelty: ⭐⭐⭐⭐ Misclassification-driven subgroup discovery combined with generative scoring is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five spurious benchmarks + standard benchmarks + 19 baselines + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The three-stage method is described clearly.
- Value: ⭐⭐⭐⭐⭐ Addresses the overlooked problem of spurious correlations in OOD detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[ICLR 2026\] GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection](../../ICLR2026/ai_safety/gradpca_leveraging_ntk_alignment_for_reliable_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)
- [\[NeurIPS 2025\] Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)

</div>

<!-- RELATED:END -->
