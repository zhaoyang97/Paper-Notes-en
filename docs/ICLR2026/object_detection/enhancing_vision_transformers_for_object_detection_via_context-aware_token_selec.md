---
title: >-
  [Paper Note] Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing
description: >-
  [ICLR 2026][Object Detection][Vision Transformer] The paper proposes Select and Pack Attention (SPA): it uses a lightweight gating layer supervised by **dynamic** multi-scale object labels to select informative tokens for each image, then **packs** varying numbers of tokens into fixed-length containers to restore batch parallelism. This achieves a +0.5~2.7 AP precisio
tags:
  - ICLR 2026
  - Object Detection
  - Vision Transformer
  - Token Packing
date: 2026-05-08
content_hash: 9e10fe98ebf06761
---
# Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q1LVcZ1PWc](https://openreview.net/forum?id=Q1LVcZ1PWc)  
**Code**: To be confirmed  
**Area**: Efficient Transformer / Sparse Attention / Object Detection  
**Keywords**: Sparse Attention, Token Selection, Object Detection, Vision Transformer, Multi-scale Supervision, Token Packing  

## TL;DR
The paper proposes Select and Pack Attention (SPA): it uses a lightweight gating layer supervised by **dynamic** multi-scale object labels to select informative tokens for each image, then **packs** varying numbers of tokens into fixed-length containers to restore batch parallelism. This achieves a +0.5~2.7 AP precision improvement and 10.9%~24.9% reduction in computational cost on object detection.

## Background & Motivation
**Background**: ViT has surpassed CNNs in detection/segmentation tasks via global self-attention. However, attention complexity grows quadratically with the number of tokens, while background tokens often constitute the majority of an image—especially in sparse scenarios like small object detection, where most pixels are non-informative yet still included in attention calculations. Consequently, several sparse attention methods have been proposed to focus only on "important tokens."

**Limitations of Prior Work**: This paper categorizes the failures of existing methods into two types. Regarding **efficiency**, GPU batch training requires token alignment within a batch, but the number of valid tokens varies across images. SparseViT uses padding to the maximum length in a batch, re-introducing background tokens. DynamicViT/EViT only discard fixed numbers of tokens during inference, while during training they calculate attention for all tokens plus a mask prediction module, exceeding the original ViT's cost. DAT only narrows the query's receptive field without reducing tokens. Regarding **performance**, these methods are mainly effective for classification; when moved to tasks requiring rich semantics like detection/instance segmentation, DynamicSwin significantly drops in performance due to imprecise token selection that loses entire objects.

**Key Challenge**: Sparse attention aims to save computation by dropping tokens, but determining "which tokens to drop" lacks **context-awareness** for downstream detection targets (often using heuristics/fixed ratios). Furthermore, inconsistent token counts across images **break batch parallelism**—saving FLOPs but failing to increase actual throughput.

**Goal**: Design a sparse attention mechanism that adaptively and accurately selects tokens based on input content while maintaining GPU batch parallelism to achieve a win-win in precision and efficiency for complex detection tasks.

**Core Idea**: **(1) Dynamic Selection + Explicit Supervision**—using a linear gating layer to score tokens, supervised directly by **multi-scale selection labels** derived from object labels (bbox/mask), making "token selection" no longer an implicit byproduct. **(2) Token Packing**—packing inconsistent positive tokens from different images into fixed-length containers to form a new batch, using attention masks to isolate images and restore batch parallelism to truly reduce computation.

## Method

### Overall Architecture
SPT (Select and Pack Transformer) is a four-stage hierarchical backbone producing 4 scales of features $r_1,\dots,r_4$. The first two stages use standard Swin blocks, while the latter two stages use SPA blocks—a division inspired by DAT’s observation that shallow features lack discriminative power, and premature token selection leads to severe information loss. Each SPA block outputs to the next layer and passes the score map down, forming multi-scale supervision with the scale’s own selection labels. Since SPA is more efficient than Swin, the authors add 4 extra blocks in the third stage to improve accuracy while maintaining lower overall computation.

```mermaid
flowchart LR
    X[Input Image] --> S1[Stage1<br/>Swin Block]
    S1 --> S2[Stage2<br/>Swin Block]
    S2 -->|Gated s0| S3[Stage3<br/>SPA Block ×N+4]
    S3 -->|score s1 upload| S4[Stage4<br/>SPA Block]
    S2 -. Multi-scale Selection Labels .-> S3
    S3 -. Multi-scale Selection Labels .-> S4
    S4 --> OUT[Multi-scale Features → Detection Head]
```

### Key Designs

**1. Token Selection with Multi-scale Supervision: Turning "token selection" from an implicit byproduct into an explicitly taught task.** The authors found that relying solely on final task loss leads the gating layer to give high scores to almost all tokens, failing to sparsify. Thus, they introduce selection labels derived from object tags (aggregating bboxes into a binary mask where object regions are 1). To avoid over-restriction, they lower Gumbel-Softmax thresholds and fuse multi-scale labels—each SPA block uses the current matching scale and the **element-wise maximum** of the upsampled scores from the previous stage to retain more informative tokens. Specifically, the gating $f_{\theta_g}$ scores the flattened input $r\in\mathbb{R}^{B\times N\times C}$, combines it with the upsampled score, applies a sigmoid gate, and uses Gumbel-Softmax to isolate positive tokens $r_p$:

$$s = \mathrm{Max}(f_{\theta_g}(r),\, s_{up}),\quad r_g = \mathrm{Sigmoid}(s)\odot r,\quad r_p = \text{Gumbel-Softmax}(s)\odot r_g$$

The selection loss uses binary cross-entropy summed across all SPA blocks: $\mathcal{L}_{select} = -\sum_{block}\big(y\log s + (1-y)\log(1-s)\big)$, with total loss $\mathcal{L}_{SPT}=\mathcal{L}_{task}+\alpha\mathcal{L}_{select}$.

**2. Token Packing: Re-establishing batch parallelism for "inconsistent numbers of positive tokens" using fixed-length containers.** After dynamic selection, the number of tokens varies per image. SparseViT's padding approach wastefully re-introduces background tokens. Inspired by sequence packing, the authors pre-define a series of containers with length $L$ and fill them with selected positive tokens sequentially, padding only the final container. This results in $p\in\mathbb{R}^{B'\times L\times C}$, where $B'$ is much smaller than the original $B$, and total tokens $B'\times L \ll B\times N$. Attention masks within containers ensure tokens only attend to those from the **same original image**. $L$ is set to the square of the Swin window size $M^2$. Complexity comparison:

$$\Omega(\text{MSA}) = B(4NC^2+2N^2C),\quad \Omega(\text{W-MSA}) = B(4NC^2+2M^2NC)$$
$$\Omega(\text{SPA}) = B(NC+NC^2) + B'(3LC^2+2L^2C)$$

W-MSA reduces dependency on $N$ from quadratic to linear, while SPA is also linear with respect to $N$ and further reduces computation as $B'\ll B$.

**3. Integration with Swin's Shifted Window: Preventing cross-container information loss.** Packing may split spatially adjacent tokens into different containers. The authors embed SPA into Swin blocks, reusing the shifted window mechanism—shifting feature maps every two transformer blocks to rotate token pairings within containers, ensuring attention covers all tokens over multiple blocks and compensating for the locality of single packing.

## Key Experimental Results

### Main Results: COCO2017 Object Detection (Cascade Mask RCNN, FLOPs for training)

| Method | Attention | AP | AP50 | AP75 | overall FLOPs(G) | FPS |
|------|--------|----|----|----|----|----|
| Swin-T (dense) | Dense | 46 | 68.1 | 50.3 | 267 | 50 |
| DAT-T | Sparse | 44.4 | 67.6 | 48.5 | 272 | 46 |
| DynamicSwin-T | Sparse | 44.3 | 65.9 | 48.5 | 272 | 46 |
| **SPT-T (ours)** | Sparse | **47.1 (+2.7)** | 68.9 | 51.6 | **261 (-4.0%)** | 54 |
| Swin-S (dense) | Dense | 48.5 | 70.2 | 53.5 | 359 | 32 |
| **SPT-S (ours)** | Sparse | **49.3 (+2.1)** | 71 | 55.2 | **342 (-5.8%)** | 33 |
| Swin-B (dense) | Dense | 51.9 | 70.5 | 56.4 | 982 | 11 |
| **SPT-B (ours)** | Sparse | **53.2 (+2.7)** | 71.3 | 58.9 | **944 (-3.9%)** | 12 |

SPT consistently outperforms all baselines including dense Swin across three scales, gaining +2.1~2.7 AP over the strongest sparse baseline; backbone computation is reduced by 10.9%~11.4%. On BDD100K, it gains +0.6~0.7 AP with up to 24.9% backbone computation reduction; on BDD-S (early/small objects), SPT-T/-S improves relative performance by 19.1%/9.6% with 20.8%~22.4% reduction in backbone computation.

### Ablation Study: Selection Design and SPA Start Stage (PASCAL VOC Multi-label / BDD100K)

| Setting | Mean Select Ratio(%) | mAP |
|------|------|------|
| Uniform top-50 selection (like SparseViT) | 50 | 44.42 |
| SPA dynamic selection (w/o $\mathcal{L}_{select}$) | 59.77 | 44.49 |
| SPA + $\mathcal{L}_{select}$ | **29.60** | **44.60** |

| SPA Start Stage (BDD100K) | AP | AP50 | AP75 |
|------|----|----|----|
| Stage 4 only | 21.9 | 32.7 | 24.2 |
| Stages 3-4 | **22.6** | **33.1** | **24.6** |
| Stages 2-4 | 20.5 | 31.3 | 22.3 |
| Stages 1-4 | 18.3 | 29.4 | 20.6 |

### Key Findings
- **Necessity of Dynamic + Explicit Supervision**: Dynamic selection is superior to fixed ratios; adding $\mathcal{L}_{select}$ compresses the selection ratio from ~60% to 29.6% while increasing mAP—supervision makes selection both accurate and efficient.
- **Optimal Start at Stage 3**: Starting too early (where shallow features lack discriminative power) causes mis-deletion of objects, dropping AP from 22.6 to 18.3, confirming DAT's observations.
- **Higher Gains with Higher Sparsity**: The computation reduction is more significant in BDD-S small object scenarios (20.8%) than in full BDD100K (16.8%), aligning with expectations of calculating only object tokens.
- **Generalizability**: Instance segmentation (COCO, SPT-S 39.6→40.9 AP) and multi-label classification (VOC, 44.12→44.60 mAP) also show benefits.

## Highlights & Insights
- **Batch Parallelism as a First-class Citizen**: Many sparse methods report FLOPs reduction but fail to achieve real throughput because inconsistent token counts lead to padding. Token Packing addresses this engineering bottleneck, and translates theoretical gains into actual FPS (SPT-T 54 FPS vs Swin-T 50).
- **Using Existing Object Labels for Supervision**: Since detection/segmentation tasks already provide bbox/mask data, aggregating them into selection labels cost almost zero extra labeling effort. This converts "token selection" from a hard-to-train implicit task into a supervised one, solving gating collapse (scoring all tokens high).
- **Multi-scale Max Fusion for Recall**: Single-scale labels are too aggressive; element-wise maximum across scales is a simple but effective "recall fallback" that prevents losing small objects while lowering the selection ratio.

## Limitations & Future Work
- **Efficiency Capped by Early Stages**: SPA is only used in the last two stages. Downsampling means late-stage tokens are only 1/16 of those in early stages. Even if SPA drops 78% of tokens, overall computation only drops ~16.8%—the bulk of the computation remains in the non-sparsified early stages.
- **Reliance on Object-level Labels**: Since supervision comes from bboxes/masks, it is unclear how to construct selection labels for tasks without dense annotations (e.g., self-supervised pre-training or pure classification).
- **Packing context via indirect compensation**: The "same-image attention" within containers and the 2-block shifted windows mean cross-container dependency is only an approximation. Potential loss from this has not been quantified.
- Experiments focused on detection/segmentation/multi-label classification and haven't covered semantic segmentation or video, where sequences are longer or more dense.

## Related Work & Insights
- **Sparse Attention Lineage**: Compared to SparseViT (l2 norm selection + max padding), DynamicViT/EViT (drop during inference/full training calculation), and DAT (reducing field without reducing tokens), this work distinguishes itself through **explicit multi-scale supervision + Packing-based parallelism**.
- **Borrowed Ideas**: Gated selection is inspired by MoE/Heterogeneous Federated Learning calculation paths; fixed-length container packing is inspired by sequence packing (Dehghani et al. 2024). This suggests a general trend: migrating "routing + packing" mechanisms from NLP/MoE to visual sparse attention can solve both efficiency and parallelism.
- **Future Insights**: Moving sparsification to earlier stages (or designing reliable shallow selection labels) to break the 16.8% acceleration ceiling is a natural extension. Additionally, the paradigm of "using task-inherent labels to supervise internal sparse decisions" could be generalized to token merging or KV-cache compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of multi-scale object label supervision and Token Packing to restore batch parallelism is novel. Treating "engineering batch parallelism" as a core design point rather than a post-optimization addresses a real pain point in deploying sparse attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers four datasets (COCO/BDD100K/BDD-S/VOC), three tasks (detection/segmentation/multi-label), and three scales (T/S/B), reporting AP, FLOPs, FPS, parameters, and selection ratios. Ablations break down "dynamic vs fixed," "supervised vs unsupervised," and "start stage." It lacks head-to-head comparison with more recent sparse SOTA and multi-run variance reports.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear categorization of failures, well-explained complexity formulas and diagrams, and sufficient motivation for design choices (gating collapse, stage selection). Tables and formulas are somewhat dense.
- **Value**: ⭐⭐⭐⭐ — Provides a plug-and-play backbone with win-win accuracy and efficiency for practical scenarios like small/sparse object detection with real FPS gains. Potential is limited by hanya sparsifying the latter stages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [\[AAAI 2026\] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](../../AAAI2026/object_detection/lampq_towards_accurate_layer-wise_mixed_precision_quantization_for_vision_transf.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](../../AAAI2026/object_detection/temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[CVPR 2026\] Tri-Modal Fusion Transformers for UAV-based Object Detection](../../CVPR2026/object_detection/tri-modal_fusion_transformers_for_uav-based_object_detection.md)
- [\[ICLR 2026\] DiffuDETR: Rethinking Detection Transformers with Denoising Diffusion Process](diffudetr_rethinking_detection_transformers_with_denoising_diffusion_process.md)

</div>

<!-- RELATED:END -->
