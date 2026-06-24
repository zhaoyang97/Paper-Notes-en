---
title: >-
  [Paper Note] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation
description: >-
  [CVPR 2025][Segmentation][Open-Vocabulary Segmentation] PartCATSeg improves h-IoU by over 10% on multiple open-vocabulary part segmentation benchmarks by disentangling and aggregating object-level and part-level image-text cost volumes, introducing a compositional loss to constrain part-whole relationships, and leveraging DINO features for structural guidance.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-Vocabulary Segmentation"
  - "Part Segmentation"
  - "Cost Aggregation"
  - "Vision-Language Models"
  - "DINO Structural Guidance"
date: 2026-05-08
content_hash: e28f1015456114d0
---

# Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2501.09688](https://arxiv.org/abs/2501.09688)  
**Code**: [https://github.com/jihochoi/PartCATSeg](https://github.com/jihochoi/PartCATSeg)  
**Area**: Segmentation  
**Keywords**: Open-Vocabulary Segmentation, Part Segmentation, Cost Aggregation, Vision-Language Models, DINO Structural Guidance

## TL;DR

PartCATSeg improves h-IoU by over 10% on multiple open-vocabulary part segmentation benchmarks by disentangling and aggregating object-level and part-level image-text cost volumes, introducing a compositional loss to constrain part-whole relationships, and leveraging DINO features for structural guidance.

## Background & Motivation

**Background**: Open-Vocabulary Part Segmentation (OVPS) is an emerging direction aiming to perform fine-grained part-level segmentation on classes unseen during training. Existing methods are primarily based on pre-trained vision-language models (VLMs), such as VLPart, which uses DINO features to establish part correspondences between base and novel classes, OV-PARTS, which utilizes object-level context to enhance part segmentation, and PartCLIPSeg, which jointly trains objects and parts via attention mechanisms.

**Limitations of Prior Work**: OVPS faces two core challenges. First, aligning part-level text with visual features is significantly more difficult than object-level alignment—the proportion of part-level image-text pairs in VLM pre-training data is much lower than that of object-level pairs, leading to weaker and noisier supervision for part-level features. Visualizations from CLIP clearly show that image-text similarities for part texts like "head" and "wing" are much weaker than those for object-level texts like "bird". Second, existing methods lack an understanding of the structural relationships between parts, often leading to absurd segmentation errors, such as misclassifying "leg" as "tail" or placing "beak" on the "tail" region.

**Key Challenge**: There is a fundamental conflict between the scarcity of part-level supervision signals and the high requirement of part segmentation for fine-grained alignment; simultaneously, methods relying purely on visual feature matching ignore the structural constraints between parts within an object, leading to the "part-whole illusion" issue.

**Goal**: (1) How to enhance part-level image-text correspondence without being dominated by object-level information? (2) How to capture part-object compositional relationships under limited part annotations? (3) How to introduce spatial structure prior to distinguish parts with similar appearances but different locations?

**Key Insight**: The authors approach this from the perspective of Cost Aggregation, modeling object and part image-text correspondences as independent cost volumes, which are then optimized through a disentangled aggregation strategy. This perspective is promising because cost aggregation has been proven effective in dense matching tasks to reduce matching errors and enhance generalization.

**Core Idea**: By disentangling object-level and part-level cost volumes and aggregating them independently, combined with part-object constraints from a compositional loss and structural guidance from DINO, the proposed method significantly improves image-text alignment accuracy for open-vocabulary part segmentation.

## Method

### Overall Architecture

PartCATSeg is extended based on the CAT-Seg architecture. Given an image and class text, the visual and text encoders of CLIP are first used to extract dense visual embeddings and class text embeddings, respectively. The class names are then split into object-level names (e.g., "cat") and part-level names (e.g., "paw") to compute two sets of cost volumes. These two sets of cost volumes are independently refined through spatial aggregation and category aggregation Transformers, and then fused into a joint object-part cost volume via linear projection, which finally generates the segmentation mask through a decoder. Disentangled loss, joint loss, and compositional loss are simultaneously applied during training.

### Key Designs

1. **Disentangled Cost Aggregation**:

    - **Function**: Processes object-level and part-level image-text correspondences separately to prevent part-level signals from being dominated by object-level ones.
    - **Mechanism**: Parses class names into object classes $\mathbf{C}_{\text{Obj}}$ and part classes $\mathbf{C}_{\text{Part}}$, and computes cosine similarity cost volumes $\mathbb{C}_{\text{Obj}}$ and $\mathbb{C}_{\text{Part}}$ respectively. Each cost volume is independently refined through a spatial aggregation Transformer (capturing local continuity based on Swin Transformer blocks) and a category aggregation Transformer (modeling inter-class relationships) to produce refined features $F''_{\text{Obj}}$ and $F''_{\text{Part}}$. The two sets of predictions are supervised separately using BCE loss.
    - **Design Motivation**: The proportion of part-level image-text pairs in CLIP pre-training is much lower than that of object-level pairs. Directly operating on a unified cost volume would cause part signals to be overwhelmed. Disentangled processing allows each level to have an independent representation space to refine correspondences.

2. **Object-aware Part Cost Aggregation**:

    - **Function**: Injects object-level semantic context into part-level predictions to form object-specific part representations like "cat's paw".
    - **Mechanism**: The refined object and part features are fused into a joint feature $F_{\text{Obj-Part}}(i) = \text{Linear}([F''_{\text{Obj}}(i); F''_{\text{Part}}(i)])$ via linear projection. This feature is then compared with object-specific part text embeddings (e.g., "cat's paw") to compute similarity, yielding a third set of cost volumes $\mathbb{C}_{\text{Obj-Part}}$. This cost volume is further refined through spatial and category aggregation Transformers before being fed into the decoder.
    - **Design Motivation**: The visual appearance of a part (e.g., "leg") can vary drastically across different objects, necessitating object-level context for disambiguation.

3. **Compositional Loss**:

    - **Function**: Employs the inductive bias that "parts compose objects" to compensate for the lack of part-level annotations.
    - **Mechanism**: At each spatial position, softmax normalization is applied to the object cost volume and the object-part cost volume to obtain distributions $\mathbb{P}_{\text{Obj}}$ and $\mathbb{P}_{\text{Obj-Part}}$. Through a predefined part-to-object mapping $M$, the part distribution is aggregated back into an object-level distribution $\tilde{\mathbb{P}}_{\text{Obj}}$. The Jensen-Shannon divergence is then used to constrain the aggregated distribution to be consistent with the directly predicted object distribution:
    
    $$\mathcal{L}_{\text{comp}} = \frac{1}{2}(D_{\text{KL}}(\mathbb{P}_{\text{Obj}} \| \tilde{\mathbb{P}}_{\text{Obj}}) + D_{\text{KL}}(\tilde{\mathbb{P}}_{\text{Obj}} \| \mathbb{P}_{\text{Obj}}))$$
    
    - **Design Motivation**: Part annotations are scarce, but object annotations are abundant. Through compositional constraints, object-level information can be utilized to indirectly supervise part predictions, ensuring that part assignments cover the entire object region, which is particularly helpful for recognizing small parts.

### Loss & Training

The total loss is the sum of three parts: $\mathcal{L} = \mathcal{L}_{\text{Obj-Part}} + \mathcal{L}_{\text{disen}} + \lambda_{\text{comp}} \mathcal{L}_{\text{comp}}$, where $\mathcal{L}_{\text{disen}}$ contains BCE losses at both object and part levels, and $\mathcal{L}_{\text{Obj-Part}}$ is the BCE loss of the joint cost volume. The training follows the strategy of CAT-Seg, fine-tuning the Query and Value heads of the CLIP encoder.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | PartCATSeg | Prev. SOTA | Gain |
|--------|------|------|------------|----------|------|
| Pascal-Part-116 | Pred-All | h-IoU | 45.77 | 30.67 (PartCLIPSeg) | +15.10 |
| Pascal-Part-116 | Oracle-Obj | h-IoU | 50.41 | 38.79 (PartCLIPSeg) | +11.62 |
| ADE20K-Part-234 | Pred-All | h-IoU | 24.19 | 11.38 (PartCLIPSeg) | +12.81 |
| ADE20K-Part-234 | Oracle-Obj | h-IoU | 49.96 | 41.83 (PartGLEE) | +8.13 |
| PartImageNet | Pred-All | h-IoU | 55.12 | 25.94 (PartCLIPSeg) | +27.79 |
| PartImageNet | Oracle-Obj | h-IoU | 72.66 | 53.85 (PartCLIPSeg) | +18.81 |

### Ablation Study

| Configuration | Pred-All h-IoU | Oracle-Obj h-IoU | Description |
|------|----------------|------------------|------|
| Cost Agg (baseline) | 31.94 | 37.66 | Cost Aggregation only |
| + DINO | 43.28 | 48.55 | Add structural guidance, +11.34 |
| + DINO + L_comp (L1) | 43.36 | 49.57 | Compositional loss with L1 normalization |
| + DINO + L_comp (SM) | 45.77 | 50.41 | Softmax normalization performs best |

| Structural Guidance Position | Pred-All h-IoU | Oracle-Obj h-IoU |
|------------|----------------|------------------|
| None | 33.65 | 37.60 |
| Object-level only | 38.61 | 42.25 |
| Part-level only | 44.41 | 51.35 |
| Both levels | 45.77 | 50.41 |

### Key Findings
- DINO structural guidance contributes the most, improving h-IoU by 11.34% from baseline to adding DINO.
- The compositional loss performs better with Softmax normalization than L1 normalization, as Softmax encourages each position to primarily belong to a single part category.
- Structural guidance is more effective at the part level than the object level, indicating that the value of DINO lies in capturing fine-grained structural information within objects rather than merely distinguishing foreground from background.
- A significant advantage is also maintained in cross-dataset evaluation, where the mIoU of unsupervised classes on PartImageNet OOD reaches 40.17%, a 20.34% improvement over PartCLIPSeg.

## Highlights & Insights
- **The idea of disentangled cost volumes** is highly ingenious. Instead of having parts and objects compete for attention in a unified space, they are modeled separately and then fused. This idea can be transferred to any task requiring multi-granularity matching (e.g., fine-grained retrieval, hierarchical classification).
- **The compositional loss acts as a self-supervised signal requiring no extra annotations**. By leveraging the "sum of parts equals the object" prior, abundant object-level annotations are indirectly converted into part-level supervision. This trick is elegant and highly generalizable.
- **The usage of DINO as a structural prior** is inspiring. Instead of using it for feature matching (as in VLPart), its pixel-level features are utilized as Query/Key guidance for spatial aggregation, making the aggregation process aware of spatial structures.

## Limitations & Future Work
- It depends on the feature quality of pre-trained CLIP and DINO. For domains not fully covered by VLMs (such as part segmentation in medical imaging), the performance might be limited.
- The compositional loss assumes that the part-to-object mapping is predefined, making it unable to handle scenarios of dynamically discovering new parts.
- Computing three sets of cost volumes and multiple Transformer aggregations during inference incurs relatively high computational overhead; the paper does not discuss efficiency issues.
- The validation is limited to semantic segmentation scenarios, without exploring more complex settings like instance-level or panoptic segmentation.

## Related Work & Insights
- **vs CAT-Seg**: CAT-Seg is the direct baseline framework. However, it only handles object-level OVSS. This work extends it to the part level via disentangled cost volumes and compositional loss.
- **vs PartCLIPSeg**: PartCLIPSeg jointly trains objects and parts using attention mechanisms but lacks explicit cost disentanglement and structural guidance. PartCATSeg outperforms it by a large margin on all datasets.
- **vs VLPart**: VLPart uses DINO for image-to-image matching between base and novel classes. In contrast, this work utilizes DINO for spatial aggregation guidance in image-text cost volumes, which is a more direct and effective way of utilization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The ideas of disentangled cost aggregation and compositional loss are simple and effective, though the core framework remains an extension of CAT-Seg.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Convincing results with evaluations on three datasets, cross-dataset settings, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich visualizations and clear motivation.
- **Value**: ⭐⭐⭐⭐⭐ Achieves over 10%+ h-IoU improvements on all benchmarks, establishing a strong new baseline for the OVPS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](../../CVPR2026/segmentation/pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

</div>

<!-- RELATED:END -->
