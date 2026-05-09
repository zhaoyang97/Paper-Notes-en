---
title: >-
  [Paper Note] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
description: >-
  [CVPR 2026][Segmentation][panoptic scene graph generation] DSFlash combines a unified segmentation/relation backbone, a gated bidirectional relation head, and mask-based dynamic patch pruning to deliver SOTA panoptic scene graph generation on PSG at mR@50=30.9 with only 18 ms latency (56 FPS).
tags:
  - CVPR 2026
  - Segmentation
  - panoptic scene graph generation
  - real-time inference
  - bidirectional relation prediction
  - token pruning
  - EoMT
date: 2026-05-08
content_hash: 9d1abbba39752a1a
---

# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**Conference**: CVPR 2026  
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)  
**Code**: not yet released (authors will release upon acceptance)  
**Area**: Scene Graph Generation / Visual Scene Understanding  
**Keywords**: panoptic scene graph generation, real-time inference, bidirectional relation prediction, token pruning, EoMT

## TL;DR

DSFlash combines a unified segmentation/relation backbone, a gated bidirectional relation head, and mask-based dynamic patch pruning to deliver SOTA panoptic scene graph generation on PSG at mR@50=30.9 with only 18 ms latency (56 FPS).

## Background & Motivation

**State of the Field**: Scene graph generation (SGG) structures an image into nodes (instances) and edges (relations), forming (subject, predicate, object) triplets that have proven valuable for downstream tasks such as VQA, image captioning and embodied reasoning. Panoptic scene graph generation (PSGG) further replaces bounding boxes with segmentation masks for more precise spatial localization.

**Limitations of Prior Work**: Existing PSGG methods almost entirely ignore inference efficiency. DSFormer reaches SOTA accuracy (mR@50=30.7) but takes 458 ms per frame and uses two independent backbones (MaskDINO for segmentation and ResNet for relation prediction), wasting compute. One-stage methods such as HiLo claim to be more efficient but still incur 427 ms latency with poor accuracy. The only speed-oriented work, REACT, reduces latency to 19 ms but performs bbox detection with YOLOv8 instead of panoptic segmentation, achieving only mR@50=19.0 — a large quality gap.

**Root Cause**: There is a sharp tension between high-quality panoptic scene graph generation and real-time inference: existing methods are either accurate but extremely slow, or fast but solve a simplified task (bbox detection rather than segmentation, or only salient-relation prediction rather than a comprehensive scene graph).

**Paper Goals**: Make panoptic scene graph generation real-time without sacrificing graph quality, and compute a comprehensive scene graph (all relations between all instances) rather than only a few salient ones.

**Starting Point**: Build on top of the two-stage paradigm and exploit a modern efficient segmentation backbone (EoMT) that simultaneously yields segmentation masks and feature representations, eliminating the redundant backbone forward pass; halve the number of relation-classification forwards via a gating mechanism for bidirectional prediction; and prune irrelevant patch tokens using mask-coverage as a task prior.

**Core Idea**: Reuse a frozen efficient segmentation backbone's features + gated bidirectional relation head + task-prior-driven token pruning = real-time comprehensive panoptic scene graph generation.

## Method

### Overall Architecture

DSFlash adopts a two-stage design. Stage one uses a frozen EoMT (Encoder-only Mask Transformer) as the segmentation backbone, producing panoptic masks and intermediate features. Concretely, patch tokens from EoMT blocks 2/5/8/11 (S/B variants) or 5/11/17/23 (L variant) are concatenated into a 768×40×40 feature tensor. Stage two iterates over each mask pair (S₀, S₁): mask embedding encodes subject/object positions into the feature patches, ViT patch embedding produces 13×13 patch tokens (384-dim), a lightweight Transformer neck processes them, and finally the gated bidirectional relation head outputs predictions for both directions in one forward. Ground-truth masks are used during training and EoMT-predicted masks at inference.

### Key Designs

1. **Merged Backbone (Unified Backbone)**:

    - Function: eliminate the redundancy of separate backbones for segmentation and relation prediction, cutting inference latency by an order of magnitude.
    - Mechanism: directly tap multi-scale patch tokens from EoMT's intermediate layers as the input features for relation prediction, removing the extra ResNet backbone. EoMT remains frozen throughout; only the neck and head are updated during training.
    - Design Motivation: DSFormer's two independent backbones (MaskDINO + ResNet) require two full forward passes and dominate latency (the segmentation share alone consumes most of the 445 ms). EoMT, being encoder-only, drops the feature adapter, pixel decoder and transformer decoder, achieving 4× faster inference than Mask2Former while preserving feature quality through DINO/EVA-02 large-scale self-supervised pretraining.

2. **Gated Bidirectional Relation Prediction**:

    - Function: predict both S₀→S₁ and S₁→S₀ relations in a single forward pass, halving the number of forwards.
    - Mechanism: from the encoded feature $x$, a sigmoid-gated MLP produces a gate vector $g = \sigma(\text{gate\_mlp}(x))$, splitting $x$ into $t_\rightarrow = g \odot x$ and $t_\leftarrow = (1-g) \odot x$; a shared MLP relation head outputs predictions for both directions. During training each mask pair is processed twice (with S₀/S₁ swapped) and an MSE consistency loss enforces that the intermediate features swap when the inputs are swapped ($t_\rightarrow \approx t'_\leftarrow$, $t_\leftarrow \approx t'_\rightarrow$), guaranteeing direction equivariance.
    - Design Motivation: the authors observe that positive labels in PSG appear 3× more often in the forward direction than in the reverse, and the model exploits this statistical bias as a shortcut. The shared MLP head plus the consistency loss force balanced treatment of both directions; the additional bidirectional supervision also boosts mR@50 from 25.0 to 28.8.

3. **Mask-based Dynamic Patch Pruning**:

    - Function: discard patch tokens that overlap with neither the subject nor object mask, reducing the model neck's compute.
    - Mechanism: mask embedding already requires computing the overlap ratio between each patch and the subject/object masks. Patches with zero overlap to both produce a pure-background mask embedding that contains no useful localization signal and can be dropped directly. Since the final prediction depends only on the classification token, the model naturally supports variable-length inputs.
    - Design Motivation: the overlap ratio is computed regardless, so the pruning decision is essentially free. The benefit is most pronounced on low-end GPUs (GTX 1080), where latency drops from 230 ms to 205 ms.

### Loss & Training

- **Relation classification loss**: Binary Cross Entropy applied independently to both directions: BCE($z_\rightarrow$, $y_\rightarrow$) and BCE($z_\leftarrow$, $y_\leftarrow$).
- **Feature consistency loss**: an MSE term enforcing that the intermediate features swap when the inputs are swapped: Consistency = (1/D)Σ[($t_\rightarrow^i$ − $t'^i_\leftarrow$)² + ($t_\leftarrow^i$ − $t'^i_\rightarrow$)²].
- **Negative sampling**: 1 negative per 5 positives.
- **Data augmentation**: DeiT III style — random horizontal flip + colour jitter + one of {grayscale, solarization, Gaussian blur}.
- **Optimizer**: AdamW, lr=1e-5, cosine schedule with warmup, gradient clipping at norm ≤ 1, 20 epochs.
- **Training efficiency**: backbone is frozen throughout; only the neck and head are trained, finishing in under 24 hours on a single GTX 1080.

## Key Experimental Results

### Main Results

Evaluated on PSG with the SGDet protocol, batch size 1, RTX 3090:

| Method | mR@50 ↑ | Latency (ms) ↓ | Params |
|--------|---------|----------------|--------|
| MotifNet-R50 | 9.56 | 100 | 109M |
| VCTree-R50 | 10.14 | 116 | 105M |
| MotifNet-MD | 16.32 | 504 | 332M |
| VCTree-MD | 17.58 | 520 | 327M |
| HiLo-R50 | 16.34 | 277 | 59M |
| HiLo-L | 19.08 | 427 | 230M |
| REACT | 19.00 | 19 | 43M |
| DSFormer | 30.70 | 458 | 330M |
| **DSFlash-S*** | 25.05 | **18** | **40M** |
| **DSFlash-B*** | 28.50 | 23 | 116M |
| **DSFlash-L** | **30.90** | 50 | 340M |

### Ablation Study

Cumulative effect of the proposed optimizations (RTX 3090, batch size 1):

| Step | mR@50 ↑ | Latency (ms) ↓ | RPS ↑ |
|------|---------|----------------|-------|
| Baseline (DSFormer) | 30.7 | 445 | 435 |
| + Unified Backbone | 25.0 | 41 (-91%) | 5,745 |
| + Efficient Mask Embedding | 25.0 | 37 (-10%) | 7,132 |
| + Gated Bidirectional Prediction | 28.8 | 29 (-22%) | 11,491 |
| + Skip Mask Upsampling | 28.5 | 23 (-21%) | 12,928 |
| + Switch to EoMT-S | 25.1 | 18 (-22%) | 17,897 |
| + Switch to EoMT-L (replaces previous row) | 30.9 | 50 (+72%) | 5,996 |

Effect of pruning and token merging across GPUs:

| Prune | ToMe | H100 | RTX 3090 | GTX 1080 | mR@50 |
|-------|------|------|----------|----------|-------|
| ✗ | 0% | 19 ms | 29 ms | 230 ms | 28.80 |
| ✓ | 0% | 20 ms | 29 ms | 205 ms | 26.67 |
| ✓ | 30% | 20 ms | 30 ms | 173 ms | 26.51 |
| ✗ | 50% | 20 ms | 29 ms | 167 ms | 24.87 |
| ✗ | 60% | 21 ms | 29 ms | 155 ms | 21.93 |

### Key Findings

- The unified backbone is the largest source of speedup, dropping latency from 445 ms to 41 ms (-91%), at a cost of 5.7 mR@50 points — primarily because EoMT segmentation quality is slightly below MaskDINO.
- Gated bidirectional prediction not only halves the forward count (RPS from 7,132 to 11,491) but also lifts mR@50 from 25.0 to 28.8 thanks to the additional bidirectional supervision signal.
- The correlation between mR@50 and the segmentation model's Panoptic Quality is as high as 0.99, indicating that segmentation quality is the decisive factor for scene graph performance.
- Pruning and token merging yield almost no latency benefit on high-end GPUs (already saturated) but are very effective on the GTX 1080, where stacking both reduces latency from 230 ms to 173 ms.
- EoMT-B with low-resolution masks beats EoMT-S with high-resolution masks (faster and more accurate), suggesting that backbone capacity matters more than mask resolution.

## Highlights & Insights

- **First real-time comprehensive panoptic scene graph generation system**: DSFlash is not only fast (56 FPS) but also computes all relations between all instances (a comprehensive scene graph) rather than predicting only a few salient ones. This unique combination of speed and completeness fills a gap for edge deployment and real-time applications in PSGG.
- **Elegant gated bidirectional prediction**: the sigmoid-gated split of features into two directional branches with a shared MLP head produces both-direction predictions in a single forward. Even more elegantly, the consistency loss not only fixes the dataset's forward/backward labelling imbalance but also acts as additional supervision that *improves* accuracy (25.0 → 28.8 mR@50) — a true win-win of "less compute and better accuracy".
- **Zero-overhead pruning by exploiting task priors**: mask-based patch pruning leverages the fact that the overlap ratio is already needed for mask embedding, so the prune decision adds essentially no compute. This pattern of turning task-specific priors into acceleration is broadly transferable.

## Limitations & Future Work

- **Frozen backbone caps the ceiling**: the fully frozen EoMT means relation prediction cannot back-propagate into the features, potentially limiting peak performance. End-to-end fine-tuning or partially unfreezing the later layers may yield further gains.
- **Small dataset scale**: PSG only has 49k images and 56 predicate classes; behaviour at larger scale and with more diverse scenes is unknown.
- **Subject/object confusion**: the authors note that subject/object confusion is a common failure mode; the gating mechanism alleviates but does not eliminate it. Instance-level contrastive learning could be explored to sharpen the distinction.

## Related Work & Insights

- **vs DSFormer**: inherits the mask embedding and strictly decoupled two-stage philosophy, but the merged backbone and bidirectional prediction cut latency from 458 ms to 50 ms (9× speedup) while slightly improving mR@50 (30.7 → 30.9).
- **vs REACT**: previously the only speed-oriented SGG method, REACT uses YOLOv8 + bbox detection; DSFlash reaches comparable speed (19 ms vs 18 ms) but is 6–12 mR@50 points higher in the PSGG setting.
- **vs HiLo**: a one-stage method that claims to be efficient but is actually 427 ms with only 19.08 mR@50, echoing recent scepticism about the "one-stage is better" narrative.
- EoMT's encoder-only design and frozen-reuse pattern can be transferred to other two-stage vision tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ gated bidirectional prediction and zero-overhead mask-based pruning are clever; first real-time comprehensive PSGG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ three backbone variants × three GPUs, step-by-step ablations, pruning/merging cross-experiments — analysis is very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ clear structure; the rigorous discussion of evaluation protocols (SingleMPO) is a notable strength; complete derivations.
- Value: ⭐⭐⭐⭐ fills the real-time PSGG gap; the 40M-param / 18-ms configuration is highly attractive for edge deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](../../ICCV2025/segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[CVPR 2026\] GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)

<!-- RELATED:END -->
