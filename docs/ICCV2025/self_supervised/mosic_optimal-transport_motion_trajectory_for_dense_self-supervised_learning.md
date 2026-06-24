---
title: >-
  [Paper Note] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning
description: >-
  [ICCV 2025][Self-Supervised Learning][Dense self-supervised learning] MoSiC extracts long-range motion trajectories via an offline point tracker and propagates cluster assignments along the temporal dimension through an Optimal Transport (Sinkhorn-Knopp)-based clustering mechanism. This enables learning spatially and temporally consistent dense representations from video data, improving DINOv2 by 1%–6% across multiple image and video benchmarks using only video for training.
tags:
  - "ICCV 2025"
  - "Self-Supervised Learning"
  - "Dense self-supervised learning"
  - "optimal transport"
  - "motion trajectories"
  - "spatiotemporal consistency"
  - "video segmentation"
date: 2026-05-08
content_hash: 80c04e6d6cd23c11
---

# MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning

**Conference**: ICCV 2025
**arXiv**: [2506.08694](https://arxiv.org/abs/2506.08694)  
**Code**: [github.com/SMSD75/MoSiC](https://github.com/SMSD75/MoSiC)  
**Area**: Self-Supervised Learning / Video Representation Learning
**Keywords**: Dense self-supervised learning, optimal transport, motion trajectories, spatiotemporal consistency, video segmentation

## TL;DR

MoSiC extracts long-range motion trajectories via an offline point tracker and propagates cluster assignments along the temporal dimension through an Optimal Transport (Sinkhorn-Knopp)-based clustering mechanism. This enables learning spatially and temporally consistent dense representations from video data, improving DINOv2 by 1%–6% across multiple image and video benchmarks using only video for training.

## Background & Motivation

Dense self-supervised learning has achieved significant progress in pixel/patch-level representation learning, yet extending it to video remains challenging. The core difficulties are:

**Failure of static augmentations**: Color transformations in the image domain implicitly preserve pixel correspondences, but object motion, camera displacement, and deformation in videos break this assumption.

**Occlusion issues**: Prior methods such as TimeTuning establish inter-frame correspondences via mask propagation, but temporary occlusions introduce propagation errors.

**Drift accumulation**: Errors accumulate frame by frame during long-range tracking, leading to degraded feature representations.

The core insight of MoSiC draws from the Gestalt psychology principle — "points that move together belong together" — and extends this principle to a finer-grained patch level.

## Method

### Overall Architecture

MoSiC adopts a teacher-student framework with the following pipeline:
1. Patchify frames from a video clip $X \in \mathbb{R}^{h \times w \times c \times T}$;
2. The student network processes randomly masked patches; the teacher network processes the original patches;
3. Long-range motion trajectories are extracted using an offline point tracker (CoTracker-v3);
4. Clustering is performed on the first frame via the Sinkhorn-Knopp optimal transport algorithm;
5. Cluster assignments are propagated to subsequent frames along the motion trajectories;
6. A cross-entropy loss aligns the cluster assignments between the student and teacher.

### Key Designs

1. **Motion Trajectory Extraction**: $N$ points are sampled from the first frame to form a uniform grid, and the frozen CoTracker-v3 tracks these points throughout the video clip, yielding trajectories $\text{Traj}_{t,i} \in \mathbb{R}^{T \times N \times 2}$. A key advantage of CoTracker is its long-range tracking capability and robustness to object re-appearance after occlusion (object permanence).

2. **OT-based Clustering**: Given cluster prototypes $P \in \mathbb{R}^{K \times d}$, the transport cost between patch features and prototypes (negative cosine similarity) is computed, and the entropy-regularized optimal transport problem is solved via the Sinkhorn-Knopp algorithm:

$$M^* = \arg\min_{M \in \mathcal{M}} \langle M, C \rangle - \epsilon \frac{1}{\lambda} H(M)$$

where $H(M)$ is the entropy regularization term and $\epsilon$ controls the smoothness of the assignments. Uniform marginal constraints are imposed to prevent collapse. This procedure is applied independently to both the student and teacher networks.

3. **Cluster Propagation**: After obtaining cluster assignments on the first frame $t_0$ via OT, the teacher's cluster assignments are propagated to subsequent frames along the motion trajectories: $\mathcal{Q}_t^{\text{teach},i} = \mathcal{Q}_{t_0}^{\text{teach},i}$. This ensures that points moving along the same trajectory maintain the same cluster identity — even when their appearance changes due to viewpoint variation. The teacher side uses bilinear interpolation for feature sampling at continuous coordinates, while the student side uses nearest-neighbor interpolation.

### Loss & Training

The training loss takes a cross-entropy form, computed between the student's cluster scores at the first frame $S_{t_0}^{\text{stu},k,i}$ (after softmax) and the teacher's propagated one-hot cluster assignments:

$$\mathcal{L}_{\text{clust}}(i) = -\sum_{t=1}^{T} \sum_{k=1}^{K} v_{t,i} \cdot \delta(\mathcal{Q}_t^{\text{teach},i} = k) \cdot \log(S_{t_0}^{\text{stu},k,i})$$

Key details:
- $v_{t,i}$ is a visibility flag; the loss is computed only over visible trajectory points, enhancing robustness to occlusion;
- Only simple augmentations (cropping + masking) are used, without complex augmentations such as color jitter or grayscale conversion;
- The model is initialized from DINOv2 pretrained weights and trained on YouTube-VOS.

## Key Experimental Results

### Main Results

| Benchmark | Dataset | Metric | MoSiC-S14 | DINOv2-S14 | Gain |
|-----------|---------|--------|-----------|------------|------|
| In-context Scene Understanding | Pascal VOC (1/128) | mIoU | 62.5 | 56.0 | +6.5 |
| In-context Scene Understanding | Pascal VOC (1/1) | mIoU | 78.2 | 77.0 | +1.2 |
| In-context Scene Understanding | ADE20K (1/1) | mIoU | 40.7 | 38.8 | +1.9 |
| Unsupervised Video Semantic Seg. | DAVIS (F-Clustering) | mIoU | 58.9 | 57.4 | +1.5 |
| Unsupervised Video Semantic Seg. | YTVOS (F-Clustering) | mIoU | 60.6 | 56.3 | +4.3 |
| Frozen Clustering | Pascal VOC (K=500) | mIoU | 60.2 | 58.6 | +1.6 |
| Linear Segmentation | Pascal VOC | mIoU | 79.7 | 78.9 | +0.8 |
| Linear Segmentation | ADE20K | mIoU | 39.6 | 37.9 | +1.7 |
| Semantic Segmentation | Pascal VOC | mIoU | 51.2 | 37.5 | +13.7 |

MoSiC-B14 (85M parameters) further improves results: Pascal VOC in-context (1/128) reaches 65.5, and (1/1) reaches 80.5.

### Ablation Study

| Configuration | Pascal VOC (mIoU) | ADE20K (mIoU) | Notes |
|---------------|-------------------|---------------|-------|
| No masking | 51.1 | 18.2 | Baseline |
| 10% masking ratio | 51.5 | 18.6 | Best |
| 40% masking ratio | 49.9 | 18.5 | Excessive masking is harmful |
| Without EMA teacher | 50.5 | 18.2 | EMA teacher is beneficial |
| With EMA teacher | 51.5 | 18.6 | Default setting |
| 8×8 grid | 49.2 | 17.5 | Sparse grid insufficient |
| 16×16 grid | 51.5 | 18.6 | Default setting |

### Key Findings

- MoSiC generalizes to various visual foundation models (DINO, EVA-CLIP, DINOv2-R), consistently delivering 2%–7% improvements;
- Gains are more pronounced in low-data regimes (+6.5% at 1/128 data);
- Compared to TimeTuning, MoSiC achieves +8.7% clustering improvement on DAVIS and +9.4% on YTVOS.

## Highlights & Insights

1. **Motion trajectories as implicit supervision**: The Gestalt psychology principle — "points that move together belong together" — is elegantly translated into an optimizable objective at the patch level;
2. **Improving image representations using only video**: Without image annotations, temporal signals from video can enhance dense representation quality for static images;
3. **Visibility mask mechanism**: Computing the loss only over visible trajectories gracefully addresses the pseudo-label noise caused by occlusions, offering greater robustness than TimeTuning's mask propagation;
4. **Plug-and-play**: The method can be applied to various visual foundation models as a post-pretraining fine-tuning stage.

## Limitations & Future Work

- The approach relies on an offline point tracker (CoTracker-v3); tracker quality directly affects the accuracy of cluster propagation;
- Training requires video data (YouTube-VOS) and cannot be applied in purely image-based settings;
- The tracker itself may fail under extremely fast motion or severe camera shake;
- The number of clusters $K$ must be set manually, with no adaptive mechanism.

## Related Work & Insights

- The key distinction from TimeTuning lies in replacing mask propagation with a robust long-range point tracker, avoiding the accumulation of propagation errors under occlusion;
- The method is complementary to image-domain approaches such as NeCo and CrIBo, which exploit cross-image consistency, whereas MoSiC exploits cross-frame temporal consistency;
- The OT clustering mechanism is inherited from DINO/SwAV, with the novel contribution of propagating cluster assignments along motion trajectories.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of motion trajectories and OT-based cluster propagation is a first in dense self-supervised learning
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, four evaluation benchmarks, generalization across multiple backbones, and comprehensive ablations
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical flow, well-formulated equations, and intuitive illustrations
- **Value**: ⭐⭐⭐⭐ Provides an effective solution for leveraging video data to enhance image representations

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](../../CVPR2026/self_supervised/shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)
- [\[CVPR 2025\] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](../../CVPR2025/self_supervised/smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)

</div>

<!-- RELATED:END -->
