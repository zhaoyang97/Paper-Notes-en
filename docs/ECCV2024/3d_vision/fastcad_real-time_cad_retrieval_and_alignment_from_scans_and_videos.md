---
title: >-
  [Paper Note] FastCAD: Real-Time CAD Retrieval and Alignment from Scans and Videos
description: >-
  [ECCV 2024][3D Vision][CAD model retrieval] This work proposes FastCAD to achieve CAD model retrieval and alignment for all objects in a scene within 50ms through contrastive learning embedding space distillation and direct parameter prediction, which is 50 times faster than existing methods with superior accuracy.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "CAD model retrieval"
  - "3D alignment"
  - "embedding distillation"
  - "real-time 3D reconstruction"
  - "contrastive learning"
date: 2026-05-08
content_hash: 28193d898b524d68
---

# FastCAD: Real-Time CAD Retrieval and Alignment from Scans and Videos

**Conference**: ECCV 2024  
**arXiv**: [2403.15161](https://arxiv.org/abs/2403.15161)  
**Code**: No public code  
**Area**: 3D Vision  
**Keywords**: CAD model retrieval, 3D alignment, embedding distillation, real-time 3D reconstruction, contrastive learning

## TL;DR

This work proposes FastCAD to achieve CAD model retrieval and alignment for all objects in a scene within 50ms through contrastive learning embedding space distillation and direct parameter prediction, which is 50 times faster than existing methods with superior accuracy.

## Background & Motivation

**Background**: Representing 3D environments with aligned CAD models is crucial for downstream tasks like AR and robotics; compared to noisy point clouds/meshes, CAD representations offer advantages such as being hole-free, having clean geometry, and providing object-level annotations.

**Limitations of Prior Work**: Current SOTA methods are computationally heavy, requiring sequential encoding of detected objects followed by optimization-based CAD alignment in a second stage, with running times ranging from 2.6 seconds to 20 minutes.

**Key Challenge**: High-precision CAD retrieval and alignment require intensive computation, failing to meet the real-time requirements of downstream applications (AR/robotics).

**Goal**: To elevate CAD retrieval and alignment speeds to real-time levels while maintaining or even improving retrieval accuracy.

**Key Insight**: Single-stage direct prediction—simultaneously outputting alignment parameters and shape embeddings to bypass the latency of two-stage sequential pipelines, combined with embedding distillation to avoid using an encoder during inference.

**Core Idea**: Learning a high-quality CAD embedding space under a contrastive learning framework and distilling it into a single-stage detection network to achieve real-time CAD retrieval and alignment.

## Method

### Overall Architecture

FastCAD takes a point cloud (from an RGB-D scan or online 3D reconstruction output) as input, encodes it into a feature volume via sparse 3D convolutions, and then predicts for each sampled location $(\hat{x}, \hat{y}, \hat{z})$: classification probability $\hat{\boldsymbol{p}}$, oriented bounding box (OBB) parameters $\hat{\boldsymbol{b}}$, front-facing side classification $\hat{\boldsymbol{f}}$, and shape embedding vector $\hat{\boldsymbol{w}}$. During inference, $\hat{\boldsymbol{w}}$ is used to retrieve the nearest neighbor CAD model, while $\hat{\boldsymbol{b}}$ and $\hat{\boldsymbol{f}}$ are utilized for alignment.

### Key Designs

1. **Embedding Distillation**:

    - **Function**: Distills the high-quality shape embeddings obtained from contrastive learning into the detection network, avoiding the need for an independent encoder during inference.
    - **Mechanism**: First, an independent encoder network is trained using a contrastive learning framework to construct a unified scan-to-CAD embedding space. Then, during the training of FastCAD, the embedding vectors of ground-truth (GT) CAD models are used as supervision signals: $\mathcal{L}_{\text{emb}}(\hat{\boldsymbol{w}}_i, \boldsymbol{w}_i)$ is formulated as an MSE loss.
    - **Design Motivation**: Two-stage retrieval (detecting bbox then encoding) suffers from distribution shift—the encoder is trained with cropped GT bboxes but evaluated with cropped predicted bboxes, leading to a severe performance drop (shape accuracy drops from 83.1% to 51.0%). Direct distillation allows the network to better utilize surrounding context and neighborhood information.

2. **Contrastive Embedding Space**:

    - **Function**: Learns a unified embedding space where embedding vectors of noisy scanned objects and clean CAD models can be directly compared.
    - **Mechanism**: Employs a triplet loss: $\mathcal{L}_{\text{Contrastive}} = \max(0, d^2(\mathbf{A}, \mathbf{P}) + m - d^2(\mathbf{A}, \mathbf{N}))$, where $\mathbf{A}$ is the anchor (scanned object), $\mathbf{P}$ is the positive sample (corresponding CAD), and $\mathbf{N}$ is the negative sample (different CAD in the same class).
    - **Two Auxiliary Tasks**:
        - Foreground/Background Segmentation: Supervised with binary cross-entropy, forcing the encoder to learn to distinguish the target object from background noise.
        - Chamfer Distance Prediction: A shallow MLP is trained to predict the Chamfer distance between positive and negative CAD models from their embeddings: $\mathcal{L}_{\text{Chamfer}} = \|d_\theta(\text{cat}(\mathbf{P}, \mathbf{N})) - d_{\text{Chamfer}}(X_{\text{pos}}, X_{\text{neg}})\|_1$, which helps the network learn embeddings that encapsulate shape similarity information.
    - **Design Motivation**: Contrastive loss alone is insufficient, as negative samples can sometimes be highly similar to positive ones. The auxiliary tasks enforce the embedding to retain fine-grained shape information. Ablation studies show that the two auxiliary tasks improve shape accuracy from 81.1% to 83.1%.

3. **Front-Facing Side Prediction**:

    - **Function**: Predicts the orientation of the CAD model inside the bounding box (which of the four sides is the front).
    - **Mechanism**: Predicts a classification probability $\hat{\boldsymbol{f}} \in \mathbb{R}^4$ for each oriented bounding box, using the cross-entropy loss $\mathcal{L}_{\text{ff}}$. Target labels are modified for symmetric objects; for example, a 2-fold symmetry is modified to $(\frac{1}{2}, 0, \frac{1}{2}, 0)$, and a 4-fold symmetry to $(\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4})$.
    - **Design Motivation**: Compared to encoding orientation inside the embedding (which would require storing 4 embeddings per CAD), predicting the front orientation independently is not only more accurate (61.7% vs 56.2%) but also reduces the number of stored/searched embeddings by four times. Leveraging symmetry annotations yields an additional 1.6% improvement.

### Loss & Training

Total loss:
$$\mathcal{L}_{\text{tot}} = \frac{1}{N_{\text{mat}}} \sum_{i=1}^{N_{\text{det}}} \mathcal{L}_{\text{cls}}(\hat{\boldsymbol{p}}_i, \boldsymbol{p}_i) + \mathbb{1}_i \left( \mathcal{L}_{\text{bb}}(\hat{\boldsymbol{b}}_i, \boldsymbol{b}_i) + \mathcal{L}_{\text{ff}}(\hat{\boldsymbol{f}}_i, \boldsymbol{f}_i) + \mathcal{L}_{\text{emb}}(\hat{\boldsymbol{w}}_i, \boldsymbol{w}_i) \right)$$

- Classification loss $\mathcal{L}_{\text{cls}}$: focal loss
- Bounding box loss $\mathcal{L}_{\text{bb}}$: DIoU loss
- Front-facing side $\mathcal{L}_{\text{ff}}$: cross-entropy
- Shape embedding $\mathcal{L}_{\text{emb}}$: MSE loss

Training details: Optimized with AdamW, learning rate 1e-3, for 225 epochs. The encoder employs the Perceiver architecture with a 256-dimensional embedding, trained for 750 epochs. For video settings, a separate version of FastCAD is trained on reconstruction outputs.

## Key Experimental Results

### Main Results

| Method | Input | Class Acc | Instance Acc | Inference Time |
|------|------|-----------|-------------|---------|
| Scan2CAD | RGB-D | 35.6% | 31.7% | 740s |
| SceneCAD | RGB-D | 52.3% | 61.2% | 2.6s |
| **FastCAD (Scan)** | **RGB-D** | **52.8%** | **61.7%** | **50ms** |
| RayTran | Video | 36.2% | 43.0% | - |
| **FastCAD (Video)** | **Video** | **39.3%** | **48.2%** | **100ms** |

### Ablation Study

| Configuration | Align Acc | Recon Acc | Shape Acc | Explanation |
|------|-----------|-----------|-----------|------|
| Two-stage retrieval (pred bbox) | 61.7% | 15.6% | 51.0% | Severe distribution shift |
| Two-stage retrieval (GT bbox) | 61.7% | 30.6% | 78.1% | Even GT bbox is worse than distillation |
| Embedding distillation (final) | 61.7% | 41.7% | 83.1% | Distillation significantly outperforms two-stage |
| Contrastive learning only | 62.3% | 38.3% | 81.1% | Baseline |
| +Chamfer + Segmentation | 61.7% | 41.7% | 83.1% | Auxiliary tasks improve by 3.4%/2.0% |
| PointNet++ encoder | 61.5% | 29.6% | 74.0% | Weak encoder |
| Perceiver encoder | 62.3% | 38.3% | 81.1% | Strong encoder is significantly better |

### Key Findings

- FastCAD achieves a **50x** speedup (50ms vs. 2.6s) under scan input with slightly better accuracy (61.7% vs. 61.2%).
- Under video input, accuracy dramatically improves from 43.0% to 48.2%, while latency drops from ~3200ms to 100ms (10 FPS real-time support).
- Reconstruction accuracy improves from 22.9% to 29.6% (compared against Vid2CAD using the same retrieval settings).
- High-quality embedding space: Even when retrieving the 10th nearest neighbor CAD model, shape accuracy remains strong.
- Color information contributes minimally; key information is encoded in the geometric structures.

## Highlights & Insights

- **Embedding distillation** is an elegant design choice—it avoids dual-network overhead during inference while solving the distribution shift issue in two-stage pipelines.
- **Symmetry-aware front-facing side prediction** is an overlooked but highly effective technique that leverages object symmetry annotations to reduce ambiguity.
- The design intuition behind auxiliary tasks (segmentation + Chamfer distance prediction) is stellar, forcing the embedding space to not only separate positive/negative samples but also encode fine-grained distances between shapes.
- The plug-and-play integration with online 3D reconstruction methods (e.g., DG Recon) highlights the advantages of modular design.
- The newly proposed Scan2CAD reconstruction accuracy and shape accuracy metrics successfully fill an evaluation gap.

## Limitations & Future Work

- Poor performance on the "display" class (only 24.1%), caused by inconsistent CAD model orientations in ShapeNet (28 out of 149 have opposite orientations).
- Lack of temporal consistency mechanism under video settings, which may cause discontinuous CAD predictions across subsequent frames.
- Dependence on the CAD model database—retrieval fails if no structurally similar models are present in the library.
- Evaluated only on ScanNet/Scan2CAD; scene scale and diversity remain limited.
- Generalization performance under open-vocabulary or zero-shot scenarios is left unexplored.

## Related Work & Insights

- **vs SceneCAD**: SceneCAD utilizes scene graphs and support relations in post-processing, yielding comparable accuracy but running 50x slower. FastCAD achieves a superior speed-accuracy trade-off via an end-to-end single-stage design.
- **vs ScanNotate**: ScanNotate exhaustively renders all CADs for matching; its shape accuracy is close to FastCAD (83.5% vs. 83.1%), but it is four orders of magnitude slower.
- **vs RayTran**: RayTran directly predicts on 3D volumetric features, resulting in extremely high computational intensity that prevents online deployment. FastCAD enables plug-and-play operation by choosing explicit point clouds as intermediate representations.
- **vs Vid2CAD**: Vid2CAD's frame-by-frame detection-plus-tracking pipeline is fragile and error-prone, whereas FastCAD is much more robust by adopting a reconstruct-then-detect strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of embedding distillation, auxiliary tasks, and symmetry-aware orientation prediction is cleverly designed, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely complete, featuring new metrics, dual scan/video settings, rigorous ablations, and online incremental evaluations.
- Writing Quality: ⭐⭐⭐⭐ Structured and clear, with rich diagrams and well-justified motivations and design decisions.
- Value: ⭐⭐⭐⭐ Achieving 50x speedup with superior accuracy holds strong potential for real-time CAD reconstruction, and the newly proposed evaluation metrics offer long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](../../NeurIPS2025/3d_vision/micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
