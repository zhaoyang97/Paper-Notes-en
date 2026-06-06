---
title: >-
  [Paper Note] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP
description: >-
  [CVPR 2026][Interpretability][Sparse Autoencoders] This paper proposes "information scope" as a novel dimension for SAE feature interpretability. By introducing the Contextual Dependency Score (CDS)…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "CLIP"
  - "Information Scope"
  - "Contextual Dependency"
  - "Outlier Token"
date: 2026-05-08
content_hash: 8aa8815e627c11a5
---

# Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP

**Conference**: CVPR 2026
**arXiv**: [2604.05724](https://arxiv.org/abs/2604.05724)  
**Code**: None  
**Area**: Model Compression / Interpretability
**Keywords**: Sparse Autoencoders, CLIP, Information Scope, Contextual Dependency, Outlier Token

## TL;DR
This paper proposes "information scope" as a novel dimension for SAE feature interpretability. By introducing the Contextual Dependency Score (CDS), it partitions CLIP's SAE features into local features (low CDS) and global features (high CDS), revealing their differentiated functional roles in classification, segmentation, and depth estimation.

## Background & Motivation
**Background**: Sparse Autoencoders (SAEs) have become a core tool for interpreting internal representations of vision models such as CLIP, decomposing dense polysemantic representations into sparse monosemantic features.

**Limitations of Prior Work**: Existing SAE interpretability research focuses almost exclusively on the **semantic identity** of features ("what concept does this feature represent"). However, a feature labeled "dog" may encode the entire object globally or respond only to local texture (e.g., fur)—post-semantic analysis cannot distinguish between the two.

**Key Observation**: **Outlier tokens** (patch tokens with abnormally high norms) in Vision Transformers exhibit strong **spatial instability** under minor contextual shifts (Shifted Context Cropping, SCC)—their positions vary dramatically with context. This suggests that global signals are highly sensitive to context, while local signals remain stably anchored to visual content.

**Core Idea**: The spatial stability difference under contextual shifts is leveraged to quantify each SAE feature's "information scope"—whether it aggregates local or global evidence.

## Method

### Overall Architecture
Input image → Shifted Context Cropping generates two overlapping crops → ViT encoding → SAE decoding yields spatial activation maps per feature → Earth Mover's Distance (EMD) computed between activation maps of the two crops → averaged to obtain CDS

### Key Designs
1. **Shifted Context Cropping (SCC)**:

    - The image is resized to $(p+s)n \times (p+s)n$, and two $pn \times pn$ crops are extracted with a shift of $sn$ pixels.
    - The two crops share $(p-s) \times (p-s)$ patches with identical pixel content.
    - **Design Motivation**: Isolates purely contextual factors (positional encoding differences + attention context variation) while eliminating content differences.

2. **Contextual Dependency Score (CDS)**:
   For each SAE feature $f_j$:
   - Select the $k_{CDS}$ images with the strongest activations for that feature.
   - Apply SCC to each image and extract feature activation maps $M_{j,1}^{(m)}$ and $M_{j,2}^{(m)}$ over the overlapping region.
   - Normalize to probability distributions and compute Earth Mover's Distance (EMD).
   - Normalize by grid diagonal distance and average:
   $$CDS_j = \frac{1}{k_{CDS} \cdot D_{grid}} \sum_{m=1}^{k_{CDS}} \text{EMD}(\mathcal{N}(M_{j,1}^{(m)}), \mathcal{N}(M_{j,2}^{(m)}))$$
   - **Low CDS** → spatially stable → local-scope feature; **High CDS** → large spatial variation → global-scope feature.

3. **Feature Partitioning and Validation**:
   The CDS histogram exhibits a multimodal distribution, naturally partitioned into low-CDS and high-CDS groups by a threshold $\gamma$. Validation is performed by examining the activation patterns of each group on outlier vs. non-outlier tokens: high-CDS features activate predominantly on outlier tokens, while low-CDS features activate predominantly on normal tokens.

### Downstream Analysis
- Feature group removal experiments with frozen CLIP backbone and linear probe evaluation.
- Three tasks: ImageNet classification, ADE20K semantic segmentation, NYUd depth estimation.

## Key Experimental Results

### Main Results (Feature Group Removal → Linear Probe Performance)

| Model | Embedding Type | ImageNet Top-1↑ | ADE20K mIoU↑ | NYUd RMSE↓ |
|-------|---------------|----------------|-------------|-----------|
| CLIP-B/16 | Original | 74.82 | 25.87 | 0.8841 |
| | Remove High-CDS | **75.54** | **26.02** | **0.8616** |
| | Remove Low-CDS | 64.86 | 11.65 | 0.9481 |
| CLIP-L/14 | Original | 80.82 | 26.66 | 0.8029 |
| | Remove High-CDS | **81.28** | 26.44 | **0.7994** |
| | Remove Low-CDS | 78.30 | 13.89 | 0.8878 |

### Ablation Study

| Analysis | Key Finding | Remarks |
|----------|------------|---------|
| Outlier vs. Non-outlier EMD | Outlier EMD >> Non-outlier EMD | Outlier tokens are spatially highly unstable |
| High-CDS activation on outlier tokens | 83.45 vs. 1.66 (B/16) | High-CDS features selectively respond to outlier tokens |
| Low-CDS activation on normal tokens | 31.51 vs. 10.39 (B/16) | Low-CDS features encode local information |
| Removing high-CDS improves classification | +0.72 ~ +1.42% | Removal of global noise proves beneficial |

### Key Findings
- **Removing low-CDS features** severely degrades segmentation and depth estimation (mIoU drops from 26 to 12), demonstrating that spatially fine-grained information is predominantly carried by low-CDS features.
- **Removing high-CDS features** slightly improves classification performance, suggesting that global features may contain redundant information.
- The CDS partition closely aligns with the outlier token phenomenon, providing a **feature-level mechanistic explanation** for outlier tokens.

## Highlights & Insights
- Information scope is proposed as an orthogonal interpretability dimension beyond semantics, addressing a blind spot in existing SAE analysis.
- The CDS metric cleverly isolates contextual factors via SCC, with a precise and physically meaningful design.
- A clear chain linking SAE features → outlier tokens → global/local information is established.

## Limitations & Future Work
- CDS computation requires selecting top-$k$ images per feature and performing bidirectional inference, with computational cost scaling with dictionary size.
- Analysis is limited to CLIP; the information scope characteristics of self-supervised ViTs such as DINOv2 remain unexplored.
- The binary partition (local vs. global) may be overly coarse; finer-grained analysis along a continuous spectrum would be valuable.
- The application of CDS-guided feature selection in practical downstream tasks has not been explored.

## Related Work & Insights
- Complements ViT outlier token research (Darcet et al.) by moving from phenomenological description to feature-level mechanistic understanding.
- The CDS framework is extensible to NLP for analyzing SAE features in LLMs (local tokens vs. global context).
- Provides a quantitative tool for "quality control" of SAE features (i.e., which features are trustworthy).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Both the information scope dimension and the CDS metric are entirely novel contributions with a distinctive perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three CLIP model variants and three downstream tasks are evaluated, though validation on non-CLIP models is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from observation → hypothesis → metric → validation is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for model interpretability research, though practical application scenarios warrant further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](../../NeurIPS2025/interpretability/transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[CVPR 2026\] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)
- [\[CVPR 2026\] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)

</div>

<!-- RELATED:END -->
