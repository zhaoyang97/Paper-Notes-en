---
title: >-
  [Paper Note] DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation
description: >-
  [AAAI 2026][Model Compression][Self-Supervised Learning] DOS is a framework that distills semantic softmaps exclusively over observable (unmasked) points, combined with Zipf-Sinkhorn regularization based on a Zipfian prior to handle the long-tail distribution of 3D semantics. It achieves state-of-the-art self-supervised learning performance on six 3D benchmarks, reaching 95% of supervised performance under linear probing.
tags:
  - AAAI 2026
  - Model Compression
  - Self-Supervised Learning
  - Point Cloud Representation
  - Self-Distillation
  - Long-Tail Distribution
  - Semantic Softmap
date: 2026-05-08
content_hash: 22ca459ec1aa110c
---

# DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation

**Conference**: AAAI 2026
**arXiv**: [2512.11465](https://arxiv.org/abs/2512.11465)
**Code**: None (pretrained weights released)
**Area**: Model Compression
**Keywords**: Self-Supervised Learning, Point Cloud Representation, Self-Distillation, Long-Tail Distribution, Semantic Softmap

## TL;DR
DOS is a framework that distills semantic softmaps exclusively over observable (unmasked) points, combined with Zipf-Sinkhorn regularization based on a Zipfian prior to handle the long-tail distribution of 3D semantics. It achieves state-of-the-art self-supervised learning performance on six 3D benchmarks, reaching 95% of supervised performance under linear probing.

## Background & Motivation

Self-supervised learning (SSL) for 3D point clouds has made notable progress in recent years, yet three core challenges remain:

**Challenge 1: Shortcut learning caused by positional information leakage.** Existing masked self-distillation methods (e.g., Sonata, MSM) use the positional embeddings of masked tokens to infer features, allowing the model to take a "shortcut"—recovering features via positional cues rather than genuine semantic understanding. This results in learning low-level geometric features rather than high-level semantics.

**Challenge 2: Insufficient supervisory signal.** Feature regression-based methods align representations point-by-point independently, ignoring the relative importance of different points with respect to a given semantic concept. Clustering-based methods assume uniform prototype usage, which contradicts the semantic imbalance inherent in 3D scenes.

**Challenge 3: Long-tail distribution of 3D semantics.** In real-world scenes, certain semantic categories (e.g., "road") contain an abundance of points, while others (e.g., "pedestrian," "traffic cone") are sparsely represented. Standard Sinkhorn-Knopp regularization enforces uniform prototype usage, which is misaligned with the natural frequency distribution and causes frequent structures to be over-segmented.

**Core Idea**: Three innovations are introduced to simultaneously address the above challenges:

**Observable point distillation**: Masked tokens are discarded entirely; supervision is applied only to visible points, fundamentally eliminating positional leakage.

**Semantic Softmap distillation**: Prototype activations are normalized across points rather than per point, promoting inter-point competition and providing richer gradients than clustering.

**Zipf-Sinkhorn regularization**: A power-law prior replaces the uniform prior, aligning prototype usage with the long-tail distribution of real-world 3D semantics.

## Method

### Overall Architecture

DOS adopts a standard student-teacher masked self-distillation framework (as illustrated in Figure 2):

1. The point cloud is cropped to generate two views, each subjected to different augmentations and random masking.
2. The **student** processes only visible points $\mathcal{P}_v$; the **teacher** processes the complete point cloud $\mathcal{P}$.
3. Teacher weights are updated via EMA.
4. Cosine similarities with learnable prototypes $\mathcal{Q}$ are computed and normalized into softmaps.
5. The teacher softmap is regularized via Zipf-Sinkhorn, and the student is trained to match it.

### Key Designs

#### 1. Observable Point Distillation

Standard masked distillation supervises the student on masked points, but recovering these points often exploits positional embeddings as a shortcut. DOS adopts a straightforward solution:

- **Masked tokens are discarded entirely**; supervision is applied only at visible indices $\mathcal{I}_v$.
- The teacher observes the full input and produces targets over the visible subset.
- The student observes only partial input and is implicitly encouraged to reason about missing regions.

This design appears counterintuitive—how can learning occur without supervising masked points? The key insight is that the teacher's output reflects complete context, whereas the student observes only partial input; thus, matching the teacher at visible points already implicitly requires reasoning about missing regions.

Ablation experiments confirm its effectiveness: transitioning from the naive setting (54.7 mIoU) to observable distillation (69.3 mIoU) yields a **gain of +14.6 mIoU**.

#### 2. Semantic Softmap Distillation

Unlike conventional clustering methods that normalize independently at each point, the Softmap normalizes **across points** for each prototype:

$$S_T(i,k) = \frac{s^T_{ik}}{\sum_{j \in \mathcal{I}_v} s^T_{jk}}, \quad S_S(i,k) = \frac{s^S_{ik}}{\sum_{j \in \mathcal{I}_v} s^S_{jk}}$$

where $s^T_{ik} = \exp(\cos(\phi_T(\mathcal{P})_i, q_k) / \tau_T)$ denotes the cosine similarity between visible points and prototypes.

**Key distinction** (as illustrated in Figure 3):
- **Clustering**: for each point, normalization is performed over all prototypes (point → prototype assignment).
- **Softmap**: for each prototype, normalization is performed over all points (spatial activation map of a prototype).

This reformulation yields several benefits:
- Each prototype induces inter-point competition, treating points as soft positive/negative samples with respect to a semantic concept.
- It encourages spatially-grounded representation learning.
- Even weakly-activated prototypes contribute to learning, avoiding information loss.

The loss is formulated as KL divergence:

$$\mathcal{L}_\sigma(\mathcal{P}, \mathcal{P}_v, \mathcal{Q}) = \frac{1}{K}\sum_{k=1}^K \text{KL}(\tilde{S}_T(:,k) \| S_S(:,k))$$

Same-view and cross-view distillation each contribute half the weight, with total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \mathcal{L}_2$.

#### 3. Zipf-Sinkhorn Regularization

While the Softmap formulation is stable, it may still lead to semantic prototype collapse or point-level collapse. Standard Sinkhorn enforces uniform prototype usage, yet 3D semantics naturally follow a Zipf distribution (as shown in Figure 4, where ScanNet200 category frequencies exhibit a power-law distribution).

**Zipf-Sinkhorn Algorithm** (Algorithm 1):

1. Compute Zipf prior: $w_k \propto 1/k^\alpha$, normalized.
2. Initialize: normalize the similarity matrix $F$.
3. Iterate Sinkhorn for $T$ rounds:
   - Row normalization: ensure equal total contribution per point.
   - Column normalization to $\mathbf{w}$: align prototype usage with the Zipf distribution.
4. Column normalization to obtain the final softmap $\tilde{S}_T$.

**Effect of Zipf exponent $\alpha$**:
- $\alpha = 0$: degenerates to uniform prior (standard Sinkhorn).
- $\alpha \approx 1.3$–$1.6$: optimal range, balancing semantic coverage and specialization.
- $\alpha \geq 2$: overly concentrated, leading to training instability.

### Loss & Training

- PTv3 is used as the default encoder; training takes approximately 20 hours on 2×A100 GPUs.
- Masking ratio: 70% for semantic segmentation, 60% for object detection.
- Mask block size: 40 cm for indoor scenes, 1 m for outdoor scenes.
- Voxel size: 0.08 m for indoor, 0.2 m for outdoor.
- 1024 prototypes.
- A lightweight PTv3 decoder is appended for segmentation tasks; CenterPoint is used for detection tasks.

## Key Experimental Results

### Main Results

**Linear probing (LP) results for semantic segmentation**:

| Method | nuScenes mIoU | Waymo mIoU | SemKITTI mIoU | ScanNet mIoU |
|--------|--------------|-----------|--------------|-------------|
| PTv3 (Supervised) | 80.4 | 71.3 | 69.1 | 77.6 |
| NOMAE | 65.1 | 59.2 | - | - |
| Sonata* | 66.1 | 60.5 | 62.0 | 72.5 |
| **DOS*** | **74.8** | **67.0** | **68.3** | **73.9** |

DOS* LP achieves **93–99% of supervised PTv3 performance**, establishing a new state of the art among SSL methods.

**Fine-tuning (FT) results**:

| Method | nuScenes mIoU | Waymo mIoU | SemKITTI mIoU | ScanNet mIoU |
|--------|--------------|-----------|--------------|-------------|
| PTv3 (Supervised) | 80.4 | 71.3 | 69.1 | 77.6 |
| Sonata* | 81.7 | 72.9 | 72.6 | 79.4 |
| **DOS*** | **81.8** | **73.9** | **73.5** | **79.7** |

After fine-tuning, DOS* **surpasses the supervised baseline** on all datasets.

**Object detection (nuScenes)**:

| Method | NDS | mAP | Setting |
|--------|-----|-----|---------|
| CenterPoint (no pretraining) | 65.4 | 57.6 | 100% labels |
| NOMAE | 60.9 | 54.4 | 20% labels |
| **DOS** | **62.1** | **57.1** | 20% labels |
| **CenterPoint + DOS** | **69.7** | **65.5** | 100% labels |

DOS pretraining yields gains of +4.3 NDS and +7.9 mAP for CenterPoint.

### Ablation Study

**Incremental component ablation** (nuScenes LP):

| Component | mIoU | Incremental Gain |
|-----------|------|-----------------|
| Naive masked self-distillation | 54.7 | Baseline |
| + Token jitter (Sonata) | 55.1 | +0.4 |
| + Observable point distillation | 69.3 | **+14.6** |
| + Softmap (replacing clustering) | 72.3 | +3.0 |
| + Feature regression (replacing clustering) | 63.0 | −6.3 (inferior to clustering) |
| + Zipf-Sinkhorn ($\alpha$=1.3) | **74.1** | +1.8 |

**Ablation on Zipf exponent $\alpha$** (ScanNet200):

| α | ScanNet mIoU | ScanNet200 mIoU | Head (66) | Common (68) | Tail (66) |
|---|-------------|----------------|-----------|------------|-----------|
| 0.0 | 71.9 | 27.9 | 50.4 | 20.5 | 10.6 |
| 1.3 | 72.8 | 29.1 | 50.8 | **23.5** | **13.2** |
| 3.0 | 69.8 | 27.7 | - | - | - |

The Zipf prior primarily improves **medium- and low-frequency categories** (Common +3.0, Tail +2.6), while its effect on high-frequency categories is minimal.

### Key Findings

1. **Observable point distillation is the most impactful contribution**: +14.6 mIoU, far exceeding token jitter's +0.4.
2. **Softmap outperforms both clustering and feature regression**: 72.3 vs. 69.3 vs. 63.0; inter-point competition provides richer gradients.
3. **Zipf prior primarily benefits medium- and low-frequency categories**: Tail categories gain +2.6 mIoU on ScanNet200.
4. **Strong cross-domain transfer**: DOS trained solely on Waymo outperforms all methods on nuScenes and SemKITTI.
5. **Excellent few-shot capability**: only 5 annotated scenes suffice to reach 85.9 mIoU on ParisLuco.
6. **High data efficiency**: performance with 0.1% labeled data far surpasses supervised training.

## Highlights & Insights

1. **Observable point distillation is elegant yet highly effective**: withholding supervision from masked points paradoxically yields large performance gains—counterintuitive but logically sound, as it eliminates positional leakage at its root.
2. **Conceptual innovation in Softmap**: reversing the normalization direction from "assigning each point over prototypes" to "distributing each prototype over points" establishes a unified perspective bridging InfoNCE and clustering (Figure 3).
3. **Well-grounded theoretical basis for the Zipf prior**: frequency distributions in nature widely follow power laws, and 3D semantics are no exception; replacing the uniform distribution with a Zipf distribution is a principled and natural design choice.
4. **Releasing general-purpose pretrained weights holds practical value**: cross-dataset pretrained LiDAR backbones can be directly applied to new domains.
5. **Extremely comprehensive experimental evaluation**: six benchmarks, three evaluation protocols (LP/FT/detection), and diverse settings including cross-domain transfer, few-shot, and data efficiency.

## Limitations & Future Work

1. Only PTv3 is used as the backbone; although alternative architecture experiments are included in the appendix, multi-architecture validation is absent from the main experiments.
2. The Zipf exponent $\alpha$ requires manual selection; adaptive adjustment of $\alpha$ may further improve performance.
3. The number of prototypes is fixed at 1024; the optimal relationship between prototype count and semantic granularity warrants further exploration.
4. LP improvements on indoor scenes are less pronounced than on outdoor scenes (ScanNet200: 30.7 vs. supervised 35.3).
5. Integration with 2D-3D cross-modal distillation (e.g., D-DITR) has not been explored.

## Related Work & Insights

- **Sonata**: the closest predecessor, employing masked self-distillation with token jitter; DOS builds upon it by addressing positional leakage and long-tail distribution.
- **SwAV** (Caron et al., 2020): first applied Sinkhorn to prototype balancing in SSL; DOS extends this to a Zipf-aware variant.
- **NOMAE**: avoids leakage by reconstructing local neighborhoods, but remains susceptible to geometric shortcuts.
- **PTv3**: Point Transformer V3, the default backbone architecture in DOS.
- Insights: explicitly modeling data distribution priors (e.g., long-tail) within SSL may be an underexplored research direction; the "less is more" principle (supervising only visible points) is worth investigating in other masked learning methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (three innovative components, each independently motivated and mutually reinforcing)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 benchmarks, multiple evaluation protocols, extensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear logical flow, well-crafted figures, rigorous mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (new state of the art in 3D SSL; released pretrained weights offer direct practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](../../NeurIPS2025/model_compression/vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/model_compression/representation-guided_parameter-efficient_llm_unlearning.md)

</div>

<!-- RELATED:END -->
