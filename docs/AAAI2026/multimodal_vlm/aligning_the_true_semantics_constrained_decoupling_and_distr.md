---
title: >-
  [Paper Note] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment
description: >-
  [AAAI 2026][Multimodal VLM][Cross-modal alignment] This paper proposes the CDDS algorithm, which decouples embeddings into semantic and modality components via a dual-path UNet…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Cross-modal alignment"
  - "embedding decoupling"
  - "distribution sampling"
  - "image-text retrieval"
  - "contrastive learning"
date: 2026-05-08
content_hash: f811371dc4db0817
---

# Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment

**Conference**: AAAI 2026
**arXiv**: [2603.05566v1](https://arxiv.org/abs/2603.05566v1)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Cross-modal alignment, embedding decoupling, distribution sampling, image-text retrieval, contrastive learning

## TL;DR
This paper proposes the CDDS algorithm, which decouples embeddings into semantic and modality components via a dual-path UNet, and employs a distribution sampling method to achieve cross-modal semantic alignment indirectly, avoiding distribution distortion caused by directly adjusting embeddings. CDDS surpasses the state of the art by 6.6%–14.2% on Flickr30K and MS-COCO.

## Background & Motivation
Most existing cross-modal alignment methods achieve semantic consistency by directly pulling image and text embeddings closer through contrastive learning. However, embeddings contain not only semantic information but also modality-specific information (e.g., color distributions in images, syntactic structures in text, and training noise). Such non-semantic information cannot be matched across modalities, and directly aligning embeddings introduces semantic bias, leading to the situation where "embedding consistency ≠ semantic consistency." Intuitively, one could decouple embeddings into semantic and modality components and align only the semantic part. Yet this faces two challenges: (1) semantic and modality information are intricately entangled, lacking clear decoupling criteria; and (2) embeddings from different modalities are constructed differently, making cosine-similarity-based cross-modal interaction theoretically unjustified, while forcibly adjusting embeddings distorts the original distribution.

## Core Problem
How to, in cross-modal alignment: (1) effectively decouple embeddings into semantic and modality components while ensuring decoupling validity and information completeness; and (2) achieve true semantic alignment without distorting the original distribution, thereby avoiding the bias and information loss introduced by the modality gap.

## Method

### Overall Architecture
CDDS adopts a fine-grained approach consisting of three components: constrained decoupling, semantic component constraint (semantic alignment via distribution sampling), modality component constraint, and information completeness constraint. After decoupling, only the semantic components participate in cross-modal alignment.

### Key Designs
1. **Dual-path UNet decoupling architecture**: A shared encoder (ViT) maps embeddings into a high-dimensional space. $z$ groups of Gaussian noise perturbations are then introduced to obtain multiple perturbed representations, enhancing robustness. A semantic decoder and a modality decoder separately extract the semantic component and the modality component from the perturbed representations using UNet-style skip connections to preserve features at each level. The final robust semantic components $V^s$/$T^s$ and modality components $V^m$/$T^m$ are obtained by averaging across the $z$ decoded results.

2. **Related Semantics Identification**: Distributions $C^v$ and $C^t$ are constructed from the column-wise features of the semantic components. KL divergence is used to measure cross-modal distributional correlation, yielding a matrix $S$. An adaptive soft-threshold sparsification algorithm is proposed: the threshold $k_i^v$ is determined using the mean and standard deviation of the conditional probability weighted by a learnable parameter $\alpha_i$, filtering weakly correlated distributions and retaining strongly correlated distribution pairs that describe related semantics.

3. **Distribution Sampling**: After identifying related semantics, rather than directly pulling semantic components closer via contrastive learning, the method samples positionally from image distributions conditioned on strongly correlated text distributions to construct cross-modal semantic components (x-semantic components) $V^x$/$T^x$. $V^x$ expresses the semantics of an image in the descriptive form of the text modality, effectively bridging the modality gap. Cross-modal semantic alignment is achieved indirectly by constraining the consistency between $V^x$ and $V^s$, without adjusting the original distribution.

### Loss & Training
The total loss is $\mathcal{L} = \alpha_s \mathcal{L}_s + \alpha_m \mathcal{L}_m + \alpha_f \mathcal{L}_f + (1 - \alpha_f) \mathcal{L}_x$, comprising four terms:
- **Semantic consistency $\mathcal{L}_s$**: A contrastive loss that pulls matched pairs of semantic component $V^s$ and the corresponding x-semantic $V^x$ closer while pushing non-matched pairs apart; symmetrically applied to the text side.
- **Modality consistency $\mathcal{L}_m$**: KL divergence is used to constrain the modality component distributions of all patches/words within the same modality to remain consistent.
- **Information completeness $\mathcal{L}_f$**: The semantic component plus the modality component should reconstruct the original embedding (L2 loss), with $w_m$ and $w_s$ as learnable weights.
- **X-semantic completeness $\mathcal{L}_x$**: The modality component plus the x-semantic component should also reconstruct the original embedding, complementing $\mathcal{L}_f$, with $\alpha_f$ controlling the balance between the two.

Training is conducted for 25 epochs using the AdamW optimizer with a learning rate of 2e-4 and batch size of 64. The encoder and decoder each have 2 layers, the feature dimension is 512, and training is performed on NVIDIA L40 GPUs.

## Key Experimental Results

| Dataset | Metric | Ours (CDDS) | Prev. SOTA (LAPS) | Gain |
|--------|------|------------|------------------|------|
| Flickr30K (Swin-384) | rSum | 552.5 | 545.3 | +7.2 |
| MS-COCO 1K (Swin-384) | rSum | 548.6 | 544.1 | +4.5 |
| MS-COCO 5K (Swin-384) | rSum | 472.1 | 470.1 | +2.0 |
| Flickr30K (ViT-224) | rSum | 510.6 | 507.3 | +3.3 |
| MS-COCO 5K (ViT-224) | rSum | 437.8 | 434.4 | +3.4 |
| Flickr30K (CLIP ViT-L) | I→T R@1 | 95.2 | 94.6 | +0.6 |

CDDS consistently outperforms the state of the art across all four backbone configurations (ViT-224/384, Swin-224/384) and when extended to CLIP, also significantly surpasses VLP models (VILT, SOHO, ALBEF, BLIP, etc.).

### Ablation Study
- **Removing the decoupling architecture (w/o Dec.)**: Performance drops by 4.6%, confirming that decoupling is the core contribution.
- **Removing the modality constraint (w/o Mod.)**: Performance drops by 0.9%, indicating that the modality consistency constraint provides auxiliary benefit.
- **Removing information completeness (w/o Int.)**: Performance drops by 6.7%, demonstrating that the information completeness constraint is the most critical component.
- **Removing Gaussian noise (w/o Gau.)**: A moderate performance drop is observed, indicating that noise perturbation enhances decoupling robustness.
- **Removing distribution sampling (w/o Sam.)**: Replacing distribution sampling with contrastive learning leads to a performance drop, validating the superiority of the distribution sampling approach.
- Applying the distribution sampling module to other models (VSE++, SCAN, SGR, CHAN, LAPS) yields consistent gains of 0.4%–1.1%, demonstrating its generalizability.

## Highlights & Insights
- The embedding decoupling perspective offers a principled approach to cross-modal alignment, transforming "aligning embeddings" into "aligning semantic components."
- The distribution sampling method is particularly elegant: by sampling positionally from the counterpart modality's distribution to construct x-semantics, alignment is achieved indirectly without distorting the original distribution, which is theoretically more principled than direct contrastive learning.
- The three-way constraint (semantic consistency, modality consistency, information completeness) forms a closed loop, ensuring both decoupling validity and lossless information preservation.
- The adaptive soft-threshold sparsification avoids the crude truncation of fixed top-$k$ selection.
- The method is plug-and-play: the distribution sampling module can be applied to other cross-modal methods and consistently yields improvements.

## Limitations & Future Work
- **High computational cost**: Related Semantics Identification (Eq. 5) must be executed per batch with complexity $O(N^2)$; precomputing over the full dataset or using random sampling both lead to notable performance degradation, as reported by the authors.
- Validation is limited to image-text retrieval; other cross-modal tasks (e.g., image captioning, VQA, text-to-image generation) remain unexplored.
- The decoupled semantic/modality components lack interpretability analysis (only shallow t-SNE visualizations are provided).
- No comparison is made with recent large-scale pretrained models (e.g., BLIP-2, CoCa) under equivalent conditions.

## Related Work & Insights
- Compared to coarse-grained methods (VSE++, GPO, DIAS): CDDS does not merely align global embeddings but aligns semantic components after decoupling, thereby avoiding interference from modality-specific noise.
- Compared to fine-grained methods (SCAN, CAAN, NAAF, CHAN): These methods still assume that corresponding columns of embeddings from different modalities describe the same semantics, whereas CDDS identifies related semantics via distributional correlation before alignment.
- Compared to LAPS (previous state of the art): LAPS enhances robustness through spatial relationship modeling but still performs direct embedding alignment; CDDS improves along both the decoupling and indirect alignment dimensions, outperforming LAPS across all configurations.

The decoupling paradigm is transferable to other multimodal tasks: decomposing embeddings into "content" and "style/modality" components is a general paradigm extensible to audio-visual, 3D-language, and other cross-modal settings. The indirect alignment idea from distribution sampling is also worth borrowing: when aligning representations from two different spaces, "expressing one's semantics in the descriptive form of the counterpart space" is more gentle than directly pulling them closer. The adaptive soft-threshold sparsification can be applied to other tasks requiring cross-domain correspondence discovery.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of decoupling and distribution sampling is innovative, though embedding decoupling itself is not an entirely new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbone configurations, two datasets, CLIP extension, detailed ablations, and generalizability validation are provided, though coverage of downstream tasks is limited.
- Writing Quality: ⭐⭐⭐⭐ The overall logic is clear and the mathematical derivations are complete, though the notation is occasionally dense.
- Value: ⭐⭐⭐⭐ The plug-and-play distribution sampling module offers practical reference value for the cross-modal alignment community.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value to Me: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](bridging_modalities_via_progressive_re-alignment_for_multimo.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](../../NeurIPS2025/multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

</div>

<!-- RELATED:END -->
