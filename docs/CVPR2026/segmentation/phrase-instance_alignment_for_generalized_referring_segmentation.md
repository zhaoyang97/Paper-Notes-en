---
title: >-
  [Paper Note] Phrase-Instance Alignment for Generalized Referring Segmentation
description: >-
  [CVPR 2026][Segmentation][Generalized Referring Segmentation] This paper proposes InstAlign, which reformulates Generalized Referring Expression Segmentation (GRES) as an instance-level reasoning problem. By introducing…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Generalized Referring Segmentation"
  - "Phrase-Instance Alignment"
  - "Instance-Level Reasoning"
  - "Multi-Target Segmentation"
  - "No-Target Detection"
date: 2026-05-08
content_hash: 436f4759d122354c
---

# Phrase-Instance Alignment for Generalized Referring Segmentation

**Conference**: CVPR 2026
**arXiv**: [2411.15087](https://arxiv.org/abs/2411.15087)  
**Code**: [https://eronguyen.github.io/InstAlign](https://eronguyen.github.io/InstAlign)  
**Area**: Image Segmentation
**Keywords**: Generalized Referring Segmentation, Phrase-Instance Alignment, Instance-Level Reasoning, Multi-Target Segmentation, No-Target Detection

## TL;DR

This paper proposes InstAlign, which reformulates Generalized Referring Expression Segmentation (GRES) as an instance-level reasoning problem. By introducing a Phrase-Object Alignment (POA) loss to establish fine-grained correspondences between linguistic phrases and visual instances, and employing a relevance-weighted aggregation mechanism to handle both multi-target and no-target scenarios in a unified manner, InstAlign achieves +3.22% cIoU and +12.25% N-acc improvements on gRefCOCO.

## Background & Motivation

1. **Background**: Generalized Referring Expression Segmentation (GRES) extends classical RES by requiring models to handle expressions such as "the two people on the left," "all the cars," or even "the elephant on the sofa" (where no elephant exists in the image)—descriptions that may correspond to multiple objects or none at all. Existing GRES methods (e.g., ReLA, LQMFormer, MABP) still follow region-based strategies, directly predicting a single binary foreground mask for the entire expression.

2. **Limitations of Prior Work**: This paradigm of predicting a single mask in one shot flattens the rich linguistic structure into an undifferentiated region—the model cannot distinguish which visual instances correspond to individual phrases within the same expression, leading to over- or under-segmentation of relevant instances. For example, given the expression "the two dogs on the left," existing methods tend to merge both dogs into a single blob or segment only one of them.

3. **Key Challenge**: The fundamental issue is the absence of **instance-level supervision**. Although query-based architectures employ multiple object queries, supervision is applied only to the final merged mask. Individual queries are never forced to specialize to distinct instances, resulting in entangled and semantically ambiguous query representations.

4. **Goal**: (a) How can each object query be made to automatically correspond to a unique visual instance? (b) How can explicit alignment be established between queries and individual phrases in the expression? (c) How can multi-target and no-target scenarios be handled within a unified inference framework?

5. **Key Insight**: The authors observe that referring expressions naturally decompose into structured phrases (e.g., "left dog" vs. "right dog"). If a model first performs instance-aware segmentation and then conducts phrase alignment, interpretable and accurate segmentation can be achieved.

6. **Core Idea**: Reformulate GRES from "directly predicting a merged mask" to "phrase-conditioned instance segmentation followed by relevance-weighted aggregation," with explicit POA loss providing fine-grained query-to-phrase supervision.

## Method

### Overall Architecture

InstAlign takes an image and a referring expression as input, and outputs a final segmentation mask along with a prediction of whether a target exists. The overall pipeline consists of four steps: (1) a visual encoder extracts multi-scale features while BERT encodes text tokens; (2) $N$ learnable object queries interact with visual and textual features through $K$ transformer decoder layers; (3) each query predicts an instance mask and a relevance score, while the POA loss aligns it to its corresponding linguistic phrase; (4) all instance masks are aggregated via relevance-score weighting to produce the final segmentation, with a no-target predictor determining whether the expression has a referent in the image.

### Key Designs

1. **Instance-Aware Segmentation Framework**:

    - **Function**: Explicitly supervises each object query to correspond to a distinct visual instance.
    - **Mechanism**: Mask2Former serves as the backbone, augmented with text conditioning—bidirectional cross-attention among queries, visual features, and text features is performed in each decoder layer. The $N$ queries ultimately produce instance masks $\hat{s}_i$ and relevance scores $\hat{p}_i$. Hungarian matching assigns predicted instances to ground-truth instances one-to-one, with matching cost $\mathcal{L}_{\text{match}}(i,j) = \lambda_{\text{score}}\mathcal{L}_{\text{score}}(\hat{p}_i,1) + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}(\hat{s}_i, s_j)$. Matched queries are trained on both mask and score objectives; unmatched queries are trained only to predict a score of zero.
    - **Design Motivation**: This is the first introduction of instance-level supervision in GRES, breaking the query entanglement caused by supervising only the final merged mask.

2. **Phrase-Object Alignment (POA) Loss**:

    - **Function**: Establishes explicit semantic correspondence between each object query and the most relevant phrase in the expression.
    - **Mechanism**: The process proceeds in three steps. First, a scaled dot-product attention computes a relevance matrix from each query to every text token: $R_k = \text{softmax}(Q_k T_k^\top / \sqrt{C})$. Second, this matrix performs a weighted sum over text features to obtain a "soft phrase embedding" for each query: $P_k = R_k T_k$. Third, a cosine similarity loss $\mathcal{L}_{\text{phrase}}(i) = 1 - \text{sim}(Q_k^i, P_k^i)$ forces the query embedding to align with its corresponding phrase embedding. This loss is incorporated into the Hungarian matching cost with weight $\lambda_{\text{phrase}}$.
    - **Design Motivation**: Unlike implicit cross-modal attention in prior work, POA provides direct phrase-instance correspondence supervision, yielding notable gains in disambiguation (e.g., distinguishing two dogs) and compositional expressions (attribute + relation). Visualizations confirm that queries automatically "claim" their corresponding phrases.

3. **Instance Aggregation (IA) Module**:

    - **Function**: Soft-aggregates multiple instance masks into a final prediction mask weighted by relevance scores.
    - **Mechanism**: The final mask is computed as $\mathcal{M}_{\text{merged}} = \text{Sigmoid}(\sum_{i=1}^N \hat{p}_i \cdot \sigma(\hat{s}_i))$, where $\sigma(\cdot)$ is a PReLU activation that acts as a learnable dynamic threshold to suppress background noise. This continuous, differentiable weighting allows the model to elegantly handle multi-target and compositional expressions.
    - **Design Motivation**: Compared to hard selection strategies (i.e., selecting the top-scoring queries), soft aggregation avoids the risk of missing relevant instances or including irrelevant ones. Ablation experiments show that PReLU contributes +0.8% cIoU and +1.5% N-acc.

4. **No-Target Predictor**:

    - **Function**: Determines whether the referring expression has a corresponding object in the image.
    - **Mechanism**: A relevance-weighted global query feature $Q_{\text{global}} = \sum_i \hat{p}_i \cdot Q^i$ is concatenated with the sentence-level text embedding $T_{\text{sen}} = \text{Average}(T_K)$ and fed into an MLP classifier. When the relevance scores of all queries are collectively low, the model infers a no-target scenario.
    - **Design Motivation**: The predictor reuses the same relevance representations used for mask inference, making the design unified and lightweight. Ablation results show that both $Q_{\text{global}}$ and $T_{\text{sen}}$ are indispensable.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \lambda_{\text{merged}}\mathcal{L}_{\text{merged}} + \lambda_{\text{inst}}\mathcal{L}_{\text{inst}} + \lambda_{\text{nt}}\mathcal{L}_{\text{nt}}$. Swin-B (pretrained on ImageNet-22K) is used as the visual encoder, BERT as the text encoder, with a 9-layer transformer decoder, 100 object queries, input resolution 480×480, batch size 32, AdamW optimizer, 20 training epochs, and approximately 24 hours of training on 4 NVIDIA A5000 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | InstAlign | Prev. SOTA | Gain |
|--------|------|-----------|-----------|------|
| gRefCOCO val | cIoU | 68.94% | 65.72% (MABP) | +3.22% |
| gRefCOCO val | gIoU | 74.34% | 70.94% (LQMFormer) | +3.40% |
| gRefCOCO val | N-acc | 79.72% | 67.47% (LQMFormer) | +12.25% |
| gRefCOCO testA | cIoU | 73.22% | 71.85% (CoHD) | +1.37% |
| Ref-ZOM test | mIoU | 70.81% | 69.81% (CoHD) | +1.00% |
| Ref-ZOM test | Acc | 94.23% | 93.34% (CoHD) | +0.89% |

Notably, InstAlign uses only a Swin-B backbone—far smaller in scale than LLM-based methods (e.g., SAM4MLLM-8B)—yet comprehensively surpasses them on cIoU/gIoU, with a N-acc lead exceeding 13 percentage points.

### Ablation Study

| Configuration | cIoU | gIoU | N-acc | Notes |
|------|------|------|-------|------|
| No instance supervision | 63.33 | 66.95 | 70.56 | Degenerates to ReLA-style method |
| Mask2Former supervision | 66.26 | 70.32 | 76.19 | +2.93% cIoU |
| + POA (full model) | 68.94 | 74.34 | 79.72 | POA adds further +2.68% cIoU |
| Hard-selection aggregation | 66.67 | 69.25 | 72.96 | −2.27% vs. IA |
| IA without PReLU | 68.13 | 72.35 | 78.22 | PReLU contributes +0.81% |
| N=20 queries | 67.64 | 72.67 | 77.25 | Too few queries |
| N=100 queries | 68.94 | 74.34 | 79.72 | Optimal |
| N=200 queries | 68.01 | 73.24 | 78.12 | Excess queries hurt performance |

### Key Findings

- **POA is the largest contributor**: From no instance supervision to the full model with POA, cumulative gains reach +5.6% cIoU and +9.16% N-acc. POA is especially beneficial for no-target detection.
- **Instance-level supervision is a necessary prerequisite**: Even without POA, introducing Mask2Former-style matching supervision alone yields substantial improvement (+2.93%), confirming that GRES genuinely requires query specialization at the instance level.
- **100 queries is the optimal trade-off**: More queries lead to a performance drop, likely due to noise introduced by redundant queries.

## Highlights & Insights

- **Redefining GRES from a region-level problem to an instance-level reasoning problem**—the conceptual reframing is more significant than any individual technical contribution. This reformulation naturally unifies the handling of multi-target and no-target scenarios.
- **The "soft phrase embedding" design in POA is elegant**—it requires no external parser to segment phrases; instead, attention weights automatically discover query-to-word correspondences in an end-to-end learnable manner.
- **The relevance-weighted aggregation paradigm is generalizable** to other tasks that require selecting and merging from multiple candidates, such as visual grounding in multi-turn dialogue.

## Limitations & Future Work

- The authors acknowledge that the model still struggles with hierarchical or compositional attribute relations, such as cases where an appended attribute conflicts with the main description (e.g., "the bowl with white soup on the left").
- Generalization to open-vocabulary settings or large-scale data has not been evaluated.
- POA performs soft alignment without leveraging explicit phrase parsing information, which may limit its precision on long and complex expressions.

## Related Work & Insights

- **vs. ReLA**: ReLA employs region-level relational attention without instance-level supervision. The instance-aware design of InstAlign represents a fundamental architectural difference, lifting N-acc from 56.37% to 79.72%.
- **vs. LLM-based methods (GSVA, SAM4MLLM)**: These methods rely on large foundation models and external data, operating at more than 10× the scale of InstAlign. Yet InstAlign with Swin-B surpasses them, demonstrating that task-specific structural design is more effective than brute-force scaling.
- **vs. MABP**: MABP also injects linguistic features into query initialization but supervises each query only via fixed patch assignments, without any phrase-level alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Redefining GRES as instance-level reasoning is a strong conceptual contribution, though the specific techniques (Hungarian matching + contrastive alignment) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two benchmarks, comprehensive ablation studies, visualization analyses, and comparisons against diverse baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, informative figures and tables, and well-motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ — A 12%+ improvement in N-acc is a significant advance; instance-level reasoning represents the right direction for GRES.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment](lod-loc_v3_generalized_aerial_localization_in_dense_cities_using_instance_silhou.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)
- [\[CVPR 2026\] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)
- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](wsrvos_weakly_supervised_rvos.md)

</div>

<!-- RELATED:END -->
