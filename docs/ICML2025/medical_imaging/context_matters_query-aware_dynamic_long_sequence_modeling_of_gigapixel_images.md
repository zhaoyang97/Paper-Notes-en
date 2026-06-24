---
title: >-
  [Paper Note] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images
description: >-
  [ICML 2025][Medical Imaging][Whole Slide Images] This paper proposes the Querent framework, which achieves efficient long-range context modeling in gigapixel Whole Slide Images (WSIs) through query-aware dynamic region importance evaluation. It theoretically achieves a bounded approximation of full self-attention and outperforms state-of-the-art (SOTA) methods in biomarker prediction, gene mutation prediction, cancer subtyping, and survival analysis across 10+ WSI datasets.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Whole Slide Images"
  - "Multiple Instance Learning"
  - "Dynamic Attention"
  - "Computational Pathology"
  - "Long Sequence Modeling"
date: 2026-05-08
content_hash: cc2c74e364200b38
---

# Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images

**Conference**: ICML 2025  
**arXiv**: [2501.18984](https://arxiv.org/abs/2501.18984)  
**Code**: Yes (GitHub)  
**Area**: Medical Imaging  
**Keywords**: Whole Slide Images, Multiple Instance Learning, Dynamic Attention, Computational Pathology, Long Sequence Modeling

## TL;DR
This paper proposes the Querent framework, which achieves efficient long-range context modeling in gigapixel Whole Slide Images (WSIs) through query-aware dynamic region importance evaluation. It theoretically achieves a bounded approximation of full self-attention and outperforms state-of-the-art (SOTA) methods in biomarker prediction, gene mutation prediction, cancer subtyping, and survival analysis across 10+ WSI datasets.

## Background & Motivation

**Background**: Whole Slide Images (WSIs) in computational pathology contain $10000^2 \sim 100000^2$ pixels, requiring the identification of scattered diagnostic features from thousands to tens of thousands of patches—akin to finding a "needle in a haystack." Multiple Instance Learning (MIL) has become the dominant weakly-supervised framework.

**Limitations of Prior Work**:
   - The full self-attention of Transformers has $O(n^2)$ complexity, making it infeasible for tens of thousands of patches.
   - Linear attention (e.g., TransMIL, Nyströmformer) reduces complexity but sacrifices modeling capability, as linear approximations create information bottlenecks.
   - Local-global attention (e.g., HIPT, LongMIL) uses fixed windows, failing to adapt to the highly variable nature of "which regions are relevant to the current patch."

**Key Challenge**: A crucial observation is that the correlation among patches in WSIs highly depends on the context. Tumor boundary regions are highly correlated with distant, similar infiltrative patterns, but are uncorrelated with adjacent normal tissues. Fixed attention patterns cannot capture this context-dependent heterogeneous relationship.

**Goal**: Achieve computational efficiency while maintaining standard full attention modeling capabilities.

**Key Insight**: Each query patch dynamically determines "which distant regions are relevant to me" by efficiently estimating importance using region-level metadata, applying full attention only to high-importance regions.

**Core Idea**: Region-level metadata (min/max feature compression) $\rightarrow$ importance scoring $\rightarrow$ selection of top-K regions $\rightarrow$ sparse but precise attention.

## Method

### Overall Architecture
Querent processes WSIs in 4 steps:
1. **Region Partitioning & Metadata Summation**: Partition the WSI patches into regions (each with $K$ patches) and use a min-max network to compute a compact metadata representation for each region.
2. **Region Importance Evaluation**: For a given query patch, efficiently evaluate the importance scores of all regions using the metadata.
3. **Selective Self-Attention**: Compute full self-attention only between the query patch and the patches of the top-K most relevant regions.
4. **Attention Pooling**: Aggregate features to perform slide-level prediction.

### Key Designs

1. **Region-Level Min-Max Metadata**:

    - **Function**: Compress the features of $K$ patches in each region into two vectors (min and max).
    - **Mechanism**:
        - For all patches $\{x_{i1}, ..., x_{iK}\}$ in region $R_i$, calculate the element-wise minimum $m_i^{\min}$ and maximum $m_i^{\max}$.
        - Map these to a shared embedding space via learnable projections $f_{\min}$ and $f_{\max}$.
    - **Design Motivation**: The min-max range implicitly encodes the "span" of patch features in a region. If a query's projection falls within the min-max range of a certain region, it suggests that patches highly relevant to the query may exist in that region.
    - **Complexity**: $O(M)$ (where $M$ is the number of regions), which is much smaller than $O(N)$ (where $N$ is the number of patches).

2. **Query-Aware Importance Scoring**:

    - **Function**: Dynamically evaluate the relevance of all regions for each query patch.
    - **Mechanism**: $s_i = \max(|\langle \hat{q}, \hat{m}_i^{\min} \rangle|, |\langle \hat{q}, \hat{m}_i^{\max} \rangle|)$.
    - Select the top-K regions with the highest scores for full attention.
    - **Design Motivation**: Each query has a unique attention pattern—tumor patches focus on distant tumor regions, while normal patches focus on the local context.
    - **Theoretical Guarantee**: It is mathematically proven that the difference between Querent's attention output and full self-attention is bounded by a constant (Theorem 1).

3. **Selective Full Attention**:

    - **Function**: Process standard self-attention solely between the query and the selected regions.
    - **Mechanism**: For each query, only use the Key/Value (K/V) matrices from the selected regions for attention computations.
    - **Complexity**: $O(N \cdot K_{\text{sel}} \cdot K)$, where $K_{\text{sel}}$ is the number of selected regions (far smaller than the total number of regions $M$).
    - **Design Motivation**: Maintain full modeling capability within the regions where precise attention is computed, making approximations only at the "which regions to select" level.

### Loss & Training
- Classification task: Cross-entropy loss
- Survival analysis: Cox proportional hazards loss
- End-to-end training (including metadata projection and importance scoring networks)
- Patch features are extracted by a pre-trained computational pathology (CPath) foundation model (PLIP)

## Key Experimental Results

### Main Results
Comprehensive evaluation across 10+ WSI datasets:

| Task | Dataset | Querent (AUC) | Best Baseline (AUC) | Gain |
|------|--------|-------------|-------------|------|
| Biomarker Prediction | TCGA-BRCA | 0.847 | 0.812 (TransMIL) | +3.5% |
| Gene Mutation Prediction | TCGA-LUNG | 0.721 | 0.693 (ABMIL) | +2.8% |
| Cancer Subtyping | TCGA-NSCLC | 0.966 | 0.951 (LongMIL) | +1.5% |
| Survival Analysis | TCGA-COAD | 0.672 | 0.641 (DSMIL) | +3.1% |
| Survival Analysis | TCGA-UCEC | 0.718 | 0.689 (WiKG) | +2.9% |

### Efficiency Comparison

| Method | Memory (GB) | Latency (ms) | AUC (BRCA) |
|------|----------|----------|-----------|
| Full Self-Attention | OOM | - | - |
| TransMIL (Linear) | 2.1 | 45 | 0.812 |
| LongMIL (Local-Global) | 3.5 | 82 | 0.831 |
| **Querent** | **2.8** | **65** | **0.847** |

### Ablation Study

| Configuration | AUC (BRCA) | Description |
|------|-----------|------|
| Uniform Attention (Fixed Regions) | 0.823 | No selection based on query |
| Random Region Selection | 0.815 | Misses importance information |
| Max-only Metadata | 0.838 | Lacks lower bound information |
| **Min-Max Metadata + Query-Aware** | **0.847** | Full Method |
| Top-3 Regions | 0.840 | Slightly less context |
| **Top-5 Regions** | **0.847** | Optimal number of selections |
| Top-10 Regions | 0.846 | Diminishing marginal returns |

### Key Findings
- Consistently outperforms SOTA across all 10+ datasets and 4 tasks—demonstrating the generalizability of the method.
- Query-aware selection increases AUC by 2-3% compared to fixed/random selection—validating the core hypothesis of "context-dependent correlation."
- Memory and latency lie between full attention and linear attention—achieving an optimal balance between efficiency and effectiveness.
- Top-5 regions already cover most valuable long-range dependencies—indicating that context correlations in WSIs are inherently sparse.
- The theoretical upper bound is indeed tight in practice—empirically validating the promise of approximating full attention.

## Highlights & Insights
- **"Context determines correlation"** is a highly precise observation—the same tumor patch might be most correlated with similar regions located more than 100 patches away, which cannot be captured by fixed windows.
- The design of Min-Max metadata is remarkably elegant—compressing regions into two vectors (min and max) to effectively estimate the maximum potential interaction between a region and a query.
- Dual support from theoretical guarantees and empirical validation enhances the credibility of the methodology.
- The methodology is generalizable—applicable not only to WSIs but also to any scenario involving ultra-long sequences with sparse correlations.
- End-to-end differentiable—the importance scoring network is trained jointly, allowing region selection to adaptively fit different tasks.

## Limitations & Future Work
- The region size $K$ is fixed; an adaptive region partitioning mechanism could be more effective.
- The metadata only utilizes min/max values; incorporating richer statistical indicators (e.g., quantiles, variance) might provide more information.
- Evaluated only on 2D WSIs; extending to 3D volumetric data (e.g., CT scans) is a future direction.
- The choice of the pre-trained encoder significantly impacts performance—comparisons between PLIP and other CPath foundation models could be more comprehensively analyzed.
- The "Top-K hard truncation" for attention selection might discard marginally important regions on boundaries—exploration of soft selection mechanisms is warranted.

## Related Work & Insights
- **vs TransMIL**: Linear attention, which sacrifices modeling capacity for efficiency; Querent maintains full attention but restricts its application to high-importance regions.
- **vs HIPT/LongMIL**: Fixed local-global windows, unable to adapt to contextual changes; Querent dynamically adapts to each unique query.
- **vs MambaMIL**: SSM-based sequence modeling maintains linear complexity but introduces strong sequential dependencies; Querent supports non-local skip attention across arbitrary positions.
- **vs WiKG**: Models patch relationships via graph structures, but the graph construction relies on predefined adjacency; Querent dynamically constructs query-specific adjacencies.
- **Insights**: The framework of region-level metadata coupled with dynamic selection can be generalized to other long-sequence scenarios, such as document understanding (cross-paragraph attention) and video analysis (cross-frame attention).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Query-aware dynamic sparse attention represents a significant breakthrough in WSI analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation with 10+ datasets, 4 tasks, and dual theoretical/empirical justification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Highly clear motivations, exquisite diagrams, and thorough analysis.
- **Value**: ⭐⭐⭐⭐⭐ Provides an important driving force for both computational pathology and long-sequence modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](../../CVPR2025/medical_imaging/reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](../../NeurIPS2025/medical_imaging/dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[CVPR 2025\] SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](../../CVPR2025/medical_imaging/salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)

</div>

<!-- RELATED:END -->
