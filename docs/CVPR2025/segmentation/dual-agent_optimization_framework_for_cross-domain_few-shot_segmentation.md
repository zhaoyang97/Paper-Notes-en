---
title: >-
  [Paper Note] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation
description: >-
  [CVPR 2025][Segmentation][Cross-Domain Few-Shot Segmentation] A Dual-Agent Optimization (DATO) framework is proposed, consisting of a Consistent Mutual Aggregation (CMA) module to learn cross-domain invariant features for representation enhancement, and a Correlation Rectification Strategy (CRS) to shift support-query matching into a domain-insensitive feature space, effectively improving the generalization capability of cross-domain few-shot segmentation.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Cross-Domain Few-Shot Segmentation"
  - "Domain-Invariant Features"
  - "Consistent Mutual Aggregation"
  - "Correlation Rectification"
  - "Feature Adaptation"
date: 2026-05-08
content_hash: 28eb5636766bffa9
---

# Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation

**Conference**: CVPR 2025  
**Code**: To be confirmed  
**Area**: Image Segmentation  
**Keywords**: Cross-Domain Few-Shot Segmentation, Domain-Invariant Features, Consistent Mutual Aggregation, Correlation Rectification, Feature Adaptation

## TL;DR
A Dual-Agent Optimization (DATO) framework is proposed, consisting of a Consistent Mutual Aggregation (CMA) module to learn cross-domain invariant features for representation enhancement, and a Correlation Rectification Strategy (CRS) to shift support-query matching into a domain-insensitive feature space, effectively improving the generalization capability of cross-domain few-shot segmentation.

## Background & Motivation

**Background**: Few-shot segmentation (FSS) achieves novel category segmentation through a small number of annotated samples, having achieved excellent performance under in-domain scenarios. However, in practical applications, training and testing sets often originate from different domains (e.g., natural images $\to$ medical images, remote sensing $\to$ industrial inspection), motivating cross-domain few-shot segmentation (CD-FSS) as a more challenging task.

**Limitations of Prior Work**: (1) **Feature mismatch caused by domain discrepancy**: Feature representations learned on the source domain may completely fail on the target domain, as texture, color, and structural distributions vary enormously across different domains; (2) **Degraded support-query matching**: In standard FSS, the support and query are from the same domain, which makes correlation matching reliable. In cross-domain scenarios, matching support and query features from different domains becomes highly unreliable; (3) **Scarcity of annotations for domain adaptation**: In few-shot scenarios, annotations are extremely scarce, rendering traditional domain adaptation techniques that require massive target domain data inapplicable.

**Key Challenge**: Cross-domain FSS needs to simultaneously address two issues—feature domain invariance (extracting high-quality features regardless of what is seen) and matching domain robustness (ensuring accuracy even in cross-domain matching). These two objectives are difficult to satisfy simultaneously under few-shot conditions.

**Goal**: How to simultaneously improve the cross-domain invariance of feature representations and the cross-domain robustness of support-query matching under extremely scarce annotation conditions.

**Key Insight**: A set of learnable "agents" is introduced as a cross-domain bridge. These agents learn domain-invariant representations by interacting with multi-domain features, and then utilize the domain-invariant features as an intermediate medium to rectify the cross-domain matching process.

**Core Idea**: Utilizing learnable agents to aggregate cross-domain invariant features to enhance original representations, and further using domain-invariant features as a "bridge" to convert cross-domain matching into in-domain matching, tackling cross-domain FSS with a two-pronged strategy.

## Method

### Overall Architecture
DATO is built upon the standard FSS pipeline (backbone feature extraction $\to$ support-query matching $\to$ segmentation prediction). Based on this, two core modules are introduced: CMA addresses domain adaptation at the feature level, while CRS handles domain rectification at the matching level. Working collaboratively, the two modules alleviate cross-domain discrepancies from the dual dimensions of feature representation and matching process.

### Key Designs

1. **Consistent Mutual Aggregation (CMA)**

    - **Function**: Learning domain-invariant features and utilizing them to enhance the original feature representations of each domain.
    - **Mechanism**: A set of learnable agent vectors (agents) is maintained to interact with features from different domains through a cross-attention mechanism. The agents first aggregate common information (domain-invariant components) from multi-domain features, and then project the aggregated domain-invariant features back to enhance the original representations of each domain. A "consistency" constraint ensures that the representations learned by agents of different domain inputs remain consistent, preventing agents from degenerating into domain-specific ones.
    - **Design Motivation**: Traditional feature enhancement methods (e.g., SE, CBAM) only operate within a single domain and cannot explicitly model cross-domain commonalities. The agent mechanism provides an explicit cross-domain information exchange channel, enabling the model to actively extract and utilize domain-invariant information.

2. **Correlation Rectification Strategy (CRS)**

    - **Function**: Converting direct cross-domain support-query matching into matching within a domain-invariant feature space.
    - **Mechanism**: Rather than directly computing the correlation between the support and the query (which is unreliable due to large domain differences), the correlation of both support and query is computed separately with the domain-invariant features aggregated by the agents, completing the match within the domain-invariant feature space. Treating domain-invariant features as an intermediate "translator," cross-domain matching is transformed into two in-domain matches (support $\to$ domain-invariant, domain-invariant $\to$ query), which drastically reduces the domain sensitivity of the matching.
    - **Design Motivation**: Intuitive analogy—two people speaking different languages (support and query) communicating through a common translator (domain-invariant features) is far more reliable than direct communication.

3. **Dual-Agent Collaborative Optimization**

    - **Function**: CMA and CRS share the same set of agents, forming a unified optimization.
    - **Mechanism**: CMA is responsible for "enabling agents to learn domain-invariant features well," while CRS is responsible for "making good use of domain-invariant features for matching." Gradients from both modules flow back to the agents simultaneously, forcing the agents to learn both highly generalizable domain-invariant representations and feature dimensions most useful for matching.
    - **Design Motivation**: Avoiding the two modules working in isolation—if trained separately, the domain-invariant features might be useless for matching, or the matching space might not be sufficiently domain-invariant.

## Key Experimental Results

### Main Results (CD-FSS Benchmark, 1-shot)

| Method | Deepglobe | ISIC | Chest X-ray | FSS-1000 | Average |
|------|-----------|------|-------------|----------|------|
| PATNet | 37.89 | 33.43 | 66.61 | 78.59 | 54.13 |
| RestNet | 40.39 | 40.30 | 72.47 | 79.16 | 58.08 |
| PINet | 41.07 | 36.67 | 73.36 | 81.60 | 58.18 |
| **DATO (Ours)** | **~44** | **~42** | **~76** | **~83** | **~61** |

### Ablation Study

| Configuration | Average mIoU |
|------|----------|
| Baseline (vanilla FSS) | ~53 |
| + CMA | ~57 |
| + CRS | ~58 |
| + CMA + CRS (DATO) | **~61** |

### Key Findings
- CMA and CRS individually bring approximately 4-5 points of improvement, and yield additional gains when combined, indicating that the two modules are complementary.
- The most significant improvements are observed in scenarios with the largest domain discrepancies (e.g., natural images $\to$ medical images), verifying the targeted design of the method for domain gap mitigation.
- An optimal number of agents exists—too few are insufficient to capture the diversity of domain-invariant features, while too many introduce redundancy.
- The rectification effect of CRS can be intuitively observed through visualized correlation maps—the rectified matching is more focused on the target region.

## Highlights & Insights
- **Dual utilization of applying domain-invariant features to both feature enhancement and matching rectification** is highly efficient, resolving two problems with one single set of agents.
- **The "translator" concept in CRS** is highly inspiring—instead of forcibly pulling features from two domains into the same space, it indirectly matches them via an intermediate medium.
- The framework design is clean, allowing CMA and CRS to be easily plugged into any existing FSS method, demonstrating plug-and-play practicality.
- The learning of agents does not require additional domain labels; domain invariance is implicitly learned solely through the segmentation loss of FSS.

## Limitations & Future Work
- The number and dimensionality of the agents are hyperparameters, which might need to be adjusted for different domain pairs.
- The quality of "domain-invariant" features is highly dependent on the diversity of domains seen during training—if the training domain combinations are too homogeneous, the agents may fail to learn truly universal invariant features.
- Performance in high-shot (5-shot, 10-shot) scenarios remains unexplored, where more support samples might reduce the necessity of CRS.
- Computational overhead analysis is missing—although the cross-attention of agents is lightweight, it still introduces additional computation during inference.
- Comparison with recent segmentation methods based on foundation models (e.g., SAM) is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](../../ICML2025/segmentation/self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [\[CVPR 2026\] Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation](../../CVPR2026/segmentation/cross-domain_few-shot_segmentation_via_multi-view_progressive_adaptation.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)

</div>

<!-- RELATED:END -->
