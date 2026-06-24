---
title: >-
  [Paper Note] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery
description: >-
  [AAAI 2026][Self-Supervised Learning][continual generalized category discovery] Grounded in Neural Collapse theory, this paper replaces dynamic classifiers with a fixed Equiangular Tight Frame (ETF) classifier and achieves continual generalized category discovery via supervised alignment and confidence-guided unsupervised alignment, reducing forgetting by 16.1% and improving novel category discovery by 3.2% across four benchmarks.
tags:
  - "AAAI 2026"
  - "Self-Supervised Learning"
  - "continual generalized category discovery"
  - "Neural Collapse"
  - "ETF classifier"
  - "forgetting mitigation"
  - "confidence-guided alignment"
date: 2026-05-08
content_hash: 8257f24429979287
---

# GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery

**Conference**: AAAI 2026
**arXiv**: [2602.19872](https://arxiv.org/abs/2602.19872)  
**Code**: None  
**Area**: LLM Evaluation
**Keywords**: continual generalized category discovery, Neural Collapse, ETF classifier, forgetting mitigation, confidence-guided alignment

## TL;DR
Grounded in Neural Collapse theory, this paper replaces dynamic classifiers with a fixed Equiangular Tight Frame (ETF) classifier and achieves continual generalized category discovery via supervised alignment and confidence-guided unsupervised alignment, reducing forgetting by 16.1% and improving novel category discovery by 3.2% across four benchmarks.

## Background & Motivation

**Background**: Continual Generalized Category Discovery (C-GCD) requires models to recognize novel categories from continuously arriving unlabeled data while retaining performance on old categories. Existing methods such as Happy employ entropy regularization and prototype replay, while MetaGCD adopts meta-learning.

**Limitations of Prior Work**: Existing C-GCD frameworks rely on dynamically optimized classifier heads or class prototypes, giving rise to two core issues: (1) catastrophic forgetting—new-class training overwrites prior knowledge; and (2) category confusion—the absence of geometric constraints leads to ambiguous decision boundaries among similar categories.

**Key Challenge**: Dynamic classifier updates cause inconsistent optimization objectives across sessions, as the model must simultaneously learn new categories and retain old ones while the optimization target itself keeps shifting.

**Goal**: Can a geometrically optimal, unified structure be predefined so that all categories are always aligned toward fixed directions?

**Key Insight**: The Neural Collapse phenomenon indicates that well-trained classifiers ultimately organize features into an ETF structure, where class means are maximally separated and intra-class variance approaches zero.

**Core Idea**: Use a fixed ETF classifier as a global geometric anchor, aligning features from all sessions to predefined optimal directions.

## Method

### Overall Architecture

GOAL consists of a base session and incremental sessions. In the base session, a feature encoder is trained on labeled data and aligned to fixed ETF prototypes. In incremental sessions, the model receives unlabeled data and maps novel-class samples to unused ETF directions via confidence filtering and clustering. The entire process requires neither a sample replay buffer nor prototype updates.

### Key Designs

1. **Predefined ETF Prototypes**

    - **Function**: Predefine $K$ equiangular tight frame prototypes $\mathbf{P} = \{p_1, \ldots, p_K\}$ as fixed alignment targets for all categories.
    - **Mechanism**: Construct unit vectors satisfying $p_k^\top p_j = -\frac{1}{K-1}$ ($k \neq j$), ensuring equal angular separation and maximum inter-class distance.
    - **Design Motivation**: Neural Collapse (NC2) shows that the optimal classifier structure is a simplex ETF; directly using it as a fixed target eliminates the inconsistency caused by dynamic optimization.

2. **Supervised Alignment (Base Session)**

    - **Function**: Align features to their corresponding ETF directions during the labeled base stage.
    - **Mechanism**: The alignment loss $\mathcal{L}_{\text{Align}}^s = -\frac{1}{N} \sum_{i} \langle \hat{e}_i, p_{y_i} \rangle$ is combined with a supervised contrastive loss $\mathcal{L}_{\text{rep}}^s$ and standard cross-entropy $\mathcal{L}_{\text{cls}}$.
    - **Design Motivation**: The three losses act synergistically—ETF alignment enforces global structure, contrastive learning promotes intra-class compactness, and cross-entropy trains a parametric classifier.

3. **Confidence-Guided Unsupervised ETF Alignment (Incremental Sessions)**

    - **Function**: Discover novel categories from unlabeled data and align them to unused ETF directions.
    - **Mechanism**: The procedure consists of three steps—(1) initialize novel class weights via KMeans clustering, selecting cluster centers with the lowest similarity to the old classifier; (2) rank samples by prediction entropy and select the top-$\alpha\%$ high-confidence samples; (3) map high-confidence samples to unassigned ETF prototypes via cluster matching and optimize with $\mathcal{L}_{\text{Align}}^u$.
    - **Design Motivation**: The ETF structure contains a large number of reserved directions; novel classes simply "occupy" new directions without affecting the alignment targets of old classes.

### Loss & Training

- Base session: $\mathcal{L}_{Base} = \mathcal{L}_{\text{Align}}^s + \mathcal{L}_{\text{rep}}^{Base} + \mathcal{L}_{\text{cls}}$
- Incremental sessions: $\mathcal{L}_{Inc} = \mathcal{L}_{\text{Align}}^u + \mathcal{L}_{\text{rep}}^{Inc} + \mathcal{L}_{\text{cls}}^u$, where $\mathcal{L}_{\text{cls}}^u$ is cross-entropy with pseudo-labels plus entropy regularization over mean predictions.
- No old sample storage is required, and ETF prototypes are never updated.

## Key Experimental Results

### Main Results (5-Session C-GCD, 4 Benchmarks)

| Dataset | Metric | GOAL | Happy (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| CIFAR-100 | Avg All Acc | 72.1% | 69.0% | +3.1% |
| TinyImageNet | Avg All Acc | 67.1% | 65.6% | +1.5% |
| CUB-200 | Avg All Acc | 69.9% | 68.9% | +1.0% |
| ImageNet-100 | Avg All Acc | 85.9% | 85.0% | +0.9% |

### Forgetting & Discovery Metrics

| Dataset | $\mathcal{M}_f \downarrow$ Ours / Prev. SOTA | $\mathcal{M}_d \uparrow$ Ours / Prev. SOTA |
|--------|------|------|
| CIFAR-100 | 11.82 / 29.40 | 53.97 / 51.36 |
| TinyImageNet | 11.66 / 29.20 | 47.88 / 43.38 |
| CUB-200 | 14.36 / 29.77 | 58.57 / 53.13 |
| ImageNet-100 | 3.24 / 17.09 | 73.08 / 72.88 |
| **Average** | **10.27 / 26.37** | **58.38 / 55.19** |

### Key Findings
- Forgetting is substantially reduced (average $\mathcal{M}_f$: 10.27 vs. 26.37, an improvement of 16.1 percentage points), which represents the primary advantage of the fixed ETF structure.
- The advantage becomes more pronounced in long-horizon 10-session experiments (+5.25% on CIFAR-100), indicating that geometric consistency is increasingly critical over longer horizons.
- The model demonstrates strong retention of old-class accuracy (Old Acc in 10-session: 82.63% vs. 71.89%), though gains in novel category discovery are comparatively modest.

## Highlights & Insights
- **Fixed geometric structure over dynamic optimization**: ETF prototypes require no updates; novel classes simply occupy reserved directions, fundamentally preventing the overwriting of old-class representations. This principle is transferable to other continual learning scenarios.
- **Confidence-guided progressive alignment**: Restricting ETF alignment to high-confidence samples prevents noisy pseudo-labels from corrupting the global geometric structure—a simple yet effective quality control strategy.
- **Engineering application of Neural Collapse**: The paper translates a theoretical phenomenon into a practical architectural design, demonstrating a transition of NC theory from "explaining observations" to "guiding design."

## Limitations & Future Work
- **ETF requires the total number of classes $K$ to be known in advance**: In practice, the final number of categories may not be available beforehand, necessitating more flexible structures.
- **Gains in novel category discovery are relatively limited** (+3.2%): The primary advantage lies in forgetting mitigation rather than novel class discovery.
- **Dependence on clustering quality**: The effectiveness of KMeans initialization and confidence filtering is contingent on feature space quality, which may be unstable in early sessions when features are poorly learned.
- **Evaluated only on image classification**: The applicability of C-GCD to NLP or multimodal settings remains unexplored.

## Related Work & Insights
- **vs. Happy**: Happy mitigates classifier bias via entropy regularization and prototype replay, but its reliance on dynamic prototypes results in a high forgetting rate of 26.37%; GOAL reduces this to 10.27% through the fixed ETF structure.
- **vs. MetaGCD**: MetaGCD improves cross-session transfer via meta-learning, but the forgetting problem is not fundamentally resolved ($\mathcal{M}_f$ 33.07%).
- **vs. NCGCD**: NCGCD also explores the application of ETF in GCD but does not support incremental learning; GOAL extends the paradigm to the continual setting.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Neural Collapse and C-GCD is innovative; using a fixed ETF as a global anchor is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 4 datasets, 5-session and 10-session settings, and both forgetting and discovery dimensions.
- Writing Quality: ⭐⭐⭐⭐ Geometric intuition is clearly articulated; the method is presented in a well-organized manner.
- Value: ⭐⭐⭐⭐ A 16% reduction in forgetting is significant, though gains in novel category discovery remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[NeurIPS 2025\] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../NeurIPS2025/self_supervised/consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)
- [\[ICLR 2026\] Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery](../../ICLR2026/self_supervised/bures-isotropy_alignment_manifold_learning_of_generalized_category_discovery.md)
- [\[CVPR 2026\] TAR: Token-Aware Refinement for Fine-grained Generalized Category Discovery](../../CVPR2026/self_supervised/tar_token-aware_refinement_for_fine-grained_generalized_category_discovery.md)
- [\[CVPR 2026\] Seeing Through the Shift: Causality-Inspired Robust Generalized Category Discovery](../../CVPR2026/self_supervised/seeing_through_the_shift_causality-inspired_robust_generalized_category_discover.md)

</div>

<!-- RELATED:END -->
