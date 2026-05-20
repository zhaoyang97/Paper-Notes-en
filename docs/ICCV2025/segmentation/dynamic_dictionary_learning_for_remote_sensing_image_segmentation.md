---
title: >-
  [Paper Note] Dynamic Dictionary Learning for Remote Sensing Image Segmentation
description: >-
  [ICCV 2025][Segmentation][Dynamic Dictionary Learning] This paper proposes D2LS, a dynamic dictionary learning framework that iteratively updates category-aware semantic embeddings (the dictionary) via multi-stage altern…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Dynamic Dictionary Learning"
  - "Remote Sensing Image Segmentation"
  - "Category Embeddings"
  - "Cross-Attention"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: a6f5859743332122
---

# Dynamic Dictionary Learning for Remote Sensing Image Segmentation

**Conference**: ICCV 2025
**arXiv**: [2503.06683](https://arxiv.org/abs/2503.06683)  
**Code**: [D2LS](https://anonymous.4open.science/r/D2LS-8267/)  
**Area**: Remote Sensing / Semantic Segmentation
**Keywords**: Dynamic Dictionary Learning, Remote Sensing Image Segmentation, Category Embeddings, Cross-Attention, Contrastive Learning

## TL;DR

This paper proposes D2LS, a dynamic dictionary learning framework that iteratively updates category-aware semantic embeddings (the dictionary) via multi-stage alternating cross-attention, and incorporates contrastive constraints to enhance inter-class separability. D2LS surpasses the state of the art on both coarse-grained and fine-grained remote sensing image segmentation benchmarks.

## Background & Motivation

**Background**: Semantic segmentation of remote sensing imagery is a core task in the remote sensing community, requiring per-pixel category assignment (e.g., buildings, roads, vegetation, water bodies) in satellite or aerial images. Existing methods predominantly rely on implicit representation learning paradigms, directly predicting segmentation results from image features through end-to-end encoder–decoder architectures.

**Limitations of Prior Work**: Remote sensing images exhibit severe intra-class heterogeneity (large appearance variation within the same category across different scenes, e.g., clouds of varying thickness) and inter-class homogeneity (visually similar appearance across different categories, e.g., low-rise buildings and roads). Existing methods employ fixed semantic embeddings that cannot dynamically adjust category representations based on the contextual features of the input image, leading to suboptimal performance in fine-grained classification scenarios.

**Key Challenge**: In conventional semantic segmentation methods, category representations are "one-size-fits-all"—all images share the same category prototypes. However, the same category in remote sensing imagery can vary enormously across geographic regions, lighting conditions, and seasons, necessitating input-adaptive dynamic category representations.

**Goal**: To design a framework that explicitly models category semantic embeddings and can dynamically adjust category representations conditioned on the input image, thereby resolving fine-grained category confusion in remote sensing segmentation.

**Key Insight**: The authors draw inspiration from dictionary learning—treating each category as an "entry" in a dictionary and iteratively updating these entries through interaction with image features so that they adapt to the current input.

**Core Idea**: Replace static category prototypes with a dynamic dictionary. Multi-stage alternating cross-attention is employed to iteratively query between image features and dictionary embeddings, progressively updating category representations to achieve input-adaptive semantic segmentation.

## Method

### Overall Architecture

The D2LS pipeline proceeds as follows: input remote sensing image → backbone network extracts multi-scale features → initialization of category dictionary embeddings (one learnable embedding vector per category) → multi-stage alternating cross-attention modules iteratively update dictionary embeddings → updated dictionary embeddings classify pixel features → output segmentation results. During training, contrastive constraints are additionally imposed in the dictionary space to enhance inter-class separability.

### Key Designs

1. **Dynamic Dictionary Construction**:

    - *Function*: Maintains a learnable semantic embedding vector for each category, serving as an "entry" in the dictionary.
    - *Mechanism*: Dictionary embeddings are initialized as learnable parameters with dimensionality aligned to image features. During inference, these embeddings are dynamically updated via cross-attention based on input image features. The key innovation lies in the multi-stage nature of the dictionary update—each stage first queries dictionary embeddings using image features (Image-to-Dict), then queries image features using the updated dictionary embeddings (Dict-to-Image), alternating over multiple rounds.
    - *Design Motivation*: A single round of cross-attention may be insufficient to fully capture the associations between image features and category representations. Multi-stage alternating queries allow dictionary embeddings to progressively absorb contextual information from the input image, becoming increasingly adapted to the current input.

2. **Multi-stage Alternating Cross-Attention**:

    - *Function*: Establishes bidirectional information flow between image features and dictionary embeddings, enabling progressive refinement of category representations.
    - *Mechanism*: Each stage comprises two steps—(1) cross-attention with dictionary embeddings as Query and image features as Key/Value, allowing dictionary embeddings to aggregate relevant information from the image; (2) cross-attention with image features as Query and updated dictionary embeddings as Key/Value, endowing image features with category-aware enhancement. Multiple stages are stacked to achieve progressive refinement.
    - *Design Motivation*: In remote sensing images, category boundaries are often ambiguous and the context is complex, requiring multiple rounds of interaction to establish accurate category–pixel correspondences. Alternating queries ensure bidirectional information flow and avoid single-direction bottlenecks.

3. **Contrastive Constraint on Dictionary Space**:

    - *Function*: Encourages intra-class compactness and inter-class separation among dictionary embeddings.
    - *Mechanism*: After the dictionary embeddings are updated, a pulling constraint is applied to same-category embeddings (minimizing intra-class distance) and a pushing constraint is applied to different-category embeddings (maximizing inter-class distance). Concretely, a contrastive loss is used, where positive pairs consist of dictionary embeddings and corresponding pixel features of the same category, and negative pairs are drawn from across categories.
    - *Design Motivation*: Relying solely on cross-attention to update dictionary embeddings may result in embeddings of different categories clustering too closely in the feature space, particularly for visually similar categories. Contrastive constraints explicitly enhance inter-class discriminability and are critical for addressing the inter-class homogeneity problem.

### Loss & Training

The total loss comprises three components: (1) a standard cross-entropy segmentation loss as the primary supervision signal; (2) a contrastive loss in the dictionary space to enhance inter-class separability; and (3) an auxiliary pixel-level loss for intermediate-stage supervision. Training employs the AdamW optimizer with a multi-scale training strategy to accommodate the wide range of scale variations in remote sensing imagery.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on coarse-grained and fine-grained remote sensing segmentation datasets:

| Dataset | Metric | D2LS | Prev. SOTA | Gain |
|--------|------|------|-----------|------|
| LoveDA (online test) | mIoU | Best | SegFormer/UPerNet | Significant |
| UAVid (online test) | mIoU | Best | Prior best | Significant |
| iSAID | mIoU | SOTA | Prior best | Consistent |
| Cloud thickness fine-grained | mIoU | SOTA | — | Particularly prominent on fine-grained tasks |

D2LS achieves top results on two online test benchmarks (LoveDA and UAVid), demonstrating the effectiveness of the method under fair evaluation conditions.

### Ablation Study

| Configuration | mIoU Change | Note |
|------|----------|------|
| Full D2LS | Best | Complete model |
| w/o multi-stage update | −~2% | Single-stage cross-attention only |
| w/o alternating queries | −~1.5% | Unidirectional (Image-to-Dict) query only |
| w/o contrastive constraint | −~1.8% | Dictionary-space contrastive loss removed |
| 1-stage vs. 3-stage vs. 5-stage | 3-stage optimal | Excessive stages risk overfitting |

### Key Findings

- The multi-stage alternating mechanism contributes the most: dictionary embeddings require sufficient interaction with image features to achieve optimal representations.
- Contrastive constraints are particularly critical for fine-grained tasks: removing them in cloud thickness classification leads to a notable increase in inter-class confusion.
- D2LS ranks among the top entries on the LoveDA and UAVid online leaderboards, validating the generalization capability of the method.
- The computational overhead of the dynamic dictionary is manageable, with inference speed close to that of standard segmentation methods.

## Highlights & Insights

- **Dictionary Learning Paradigm for Semantic Segmentation**: The combination of classical dictionary learning with Transformer cross-attention achieves input-adaptive category representations. This "dynamic prototype" paradigm can be transferred to other tasks requiring handling of intra-class variation, such as medical image segmentation and autonomous driving scene understanding.
- **Bidirectional Refinement via Alternating Queries**: Rather than updating the dictionary solely from image features, the dictionary and image features mutually query and reinforce each other. This bidirectional interaction is more effective than unidirectional querying.
- **Validation on Online Test Benchmarks**: Online evaluation on LoveDA and UAVid mitigates overfitting to the test set, yielding more credible results.

## Limitations & Future Work

- The number of dictionary categories must be specified in advance, precluding application to open-vocabulary semantic segmentation.
- Multi-stage alternating attention introduces additional computational cost; although the authors claim this overhead is manageable, scalability in large-scale deployment scenarios warrants further consideration.
- Validation is currently limited to remote sensing scenarios; performance on natural image segmentation remains unknown, as the intra-class variation characteristics of remote sensing may not fully generalize to natural scene settings.
- Future work could explore integration with vision–language models, using textual descriptions to assist dictionary initialization.

## Related Work & Insights

- **vs. SegFormer**: SegFormer employs a simple MLP decoder with implicit category representations. D2LS explicitly models category embeddings and updates them dynamically, offering a clear advantage in fine-grained segmentation.
- **vs. Mask2Former**: Mask2Former also uses learnable queries to predict segmentation masks, but these queries are fixed and category-agnostic. D2LS's dictionary embeddings are category-aware and are adapted to the input through multi-stage updates.
- **vs. Prototype Learning Methods**: Traditional prototype learning methods (e.g., PANet) use the mean of support set features as category prototypes, which are static. D2LS's dictionary is dynamically updated, enabling better handling of intra-class variation.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of dictionary learning and alternating cross-attention is novel, though the core components (cross-attention, contrastive learning) are individually well-established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both coarse-grained and fine-grained datasets, online leaderboard validation, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the method is described in a systematic manner.
- Value: ⭐⭐⭐⭐ Offers direct value to the remote sensing segmentation community; the dynamic prototype paradigm has strong transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[ICCV 2025\] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation](learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)

</div>

<!-- RELATED:END -->
