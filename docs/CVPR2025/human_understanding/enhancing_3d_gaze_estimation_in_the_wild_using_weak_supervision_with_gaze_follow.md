---
title: >-
  [Paper Note] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels
description: >-
  [CVPR 2025][Human Understanding][Gaze estimation] Proposes a two-stage self-training weakly supervised framework, ST-WSGE, which leverages 2D gaze-following datasets (such as GazeFollow) to generate 3D pseudo-labels to enhance the generalization capability of 3D gaze estimation in the wild. Concurrently, a modality-agnostic Gaze Transformer (GaT) is designed to uniformly process both image and video inputs, achieving SOTA results on Gaze360, GFIE, MPIIFaceGaze…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Gaze estimation"
  - "weakly supervised learning"
  - "self-training"
  - "gaze following"
  - "video understanding"
date: 2026-05-08
content_hash: fd6acfd1938151de
---

# Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels

**Conference**: CVPR 2025  
**arXiv**: [2502.20249](https://arxiv.org/abs/2502.20249)  
**Code**: Coming soon  
**Area**: LLM Evaluation  
**Keywords**: Gaze estimation, weakly supervised learning, self-training, gaze following, video understanding

## TL;DR

Proposes a two-stage self-training weakly supervised framework, ST-WSGE, which leverages 2D gaze-following datasets (such as GazeFollow) to generate 3D pseudo-labels to enhance the generalization capability of 3D gaze estimation in the wild. Concurrently, a modality-agnostic Gaze Transformer (GaT) is designed to uniformly process both image and video inputs, achieving SOTA results on Gaze360, GFIE, MPIIFaceGaze, and other datasets.

## Background & Motivation

**Background**: 3D gaze estimation is an important signal for understanding human behavior, applied in scenarios like AR/VR, human-computer interaction, and psychological analysis. Current mainstream methods have achieved high accuracy in controlled laboratory environments (such as frontal faces + screen gaze targets) but perform poorly in real-world "physically unconstrained" scenes.

**Limitations of Prior Work**: Gaze estimation in the wild faces massive challenges—severe head pose variations, eye occlusions, low resolution, and diverse appearances. The fundamental cause is the lack of diverse 3D gaze annotations in wild environments: collecting high-quality 3D gaze data requires complex laser setups, which are costly and not scalable. Although existing in-the-wild datasets like Gaze360 and GFIE have made contributions, the diversity of training data remains heavily insufficient.

**Key Challenge**: 3D gaze annotations are hard to collect at scale (requiring specialized equipment), whereas 2D gaze following annotations ("where is this person looking at in pixel coordinates") are relatively easy to obtain and highly diverse. The key problem is how to leverage these abundant 2D weak annotations to compensate for the lack of 3D annotations. Previous works, such as Kothari et al., attempted to use LAEO (Looking At Each Other) datasets to generate pseudo 3D labels, but the gaze distribution in LAEO is mostly limited to horizontal directions and requires at least two interacting people in the image.

**Goal**: (1) How to utilize more diverse 2D gaze following data to generate reliable 3D pseudo labels? (2) How to design a unified model that can handle both image and video modalities simultaneously to maximize the utilization of existing training data?

**Key Insight**: The authors observe that gaze following datasets (e.g., GazeFollow) display a much wider gaze distribution and richer natural scene diversity. Based on a key assumption—that a model pre-trained on unconstrained 3D gaze data can provide reasonable depth-direction ($z$) estimations—the predicted $z$ component can be combined with the 2D annotated $x$ and $y$ directions through geometric projection to generate robust 3D pseudo-labels.

**Core Idea**: By employing a two-stage self-training process (first learning a 3D prior, then projecting to generate pseudo-labels for joint training with 3D data) and a modality-agnostic Transformer to unify image/video training, the generalization of in-the-wild gaze estimation is significantly improved.

## Method

### Overall Architecture

ST-WSGE is a two-stage training workflow. The inputs consist of 3D gaze datasets (such as Gaze360) and 2D gaze following datasets (such as GazeFollow), and the final output is a 3D gaze estimation model that can process both images and videos. In the first stage, the GaT model is trained supervisely on 3D data. In the second stage, the stage-one model is used to infer 3D gaze on GazeFollow data, which is combined with 2D labels via geometric transformations to generate pseudo 3D labels for joint training of a new GaT.

### Key Designs

1. **Gaze Transformer (GaT) — Modality-Agnostic Gaze Architecture**:

    - **Function**: Uniformly processes image ($T=1$) and video ($T>1$) inputs to output 3D gaze vectors.
    - **Mechanism**: Adopts a tiny Swin3D hierarchical spatiotemporal encoder to represent images and videos uniformly as 4D tensors $X \in \mathbb{R}^{T \times H \times W \times 3}$. A patchifier splits the input into spatiotemporal patches (temporal stride $t=2$), where image inputs are replicated to simulate videos. The encoder captures local and global features using shifted window self-attention. The output is spatially pooled and temporally interpolated, then passed through a shared MLP to predict gaze vectors.
    - **Design Motivation**: CNNs excel at extracting local eye features but lack global reasoning capabilities. Standard ViT patches are too large ($16\times16$) and might segment the eye region awkwardly. The small patch size ($4\times4$) and shifted window of Swin3D preserve detail while aggregating global context, naturally supporting spatiotemporal expansion.

2. **Pseudo 3D Gaze Label Generation (Geometric Projection)**:

    - **Function**: Fuses predicted 3D gaze with 2D annotations to generate reliable pseudo 3D labels.
    - **Mechanism**: Let the predicted 3D gaze be $\hat{g} = (\hat{g}_x, \hat{g}_y, \hat{g}_z)$ and the 2D annotated direction be $v = (v_x, v_y)$. The pseudo label is defined as $g^{ps} = (v_x \| (\hat{g}_x, \hat{g}_y)\|_2, v_y \| (\hat{g}_x, \hat{g}_y)\|_2, \hat{g}_z)$. Essentially, the predicted 3D vector is rotated around the z-axis so that its projection on the image plane aligns with the 2D annotation, while retaining the model-estimated z-component.
    - **Design Motivation**: 2D gaze annotations only lack depth direction, while the pre-trained model's z-estimation is already relatively reasonable. Directly replacing the predicted projection components with the 2D annotated $x$ and $y$ leverages the precise direction of large-scale annotations while preserving the 3D prior.

3. **Multi-Dataset Training Strategy**:

    - **Function**: Balances training across multiple datasets of different scales and modalities.
    - **Mechanism**: Employs an alternating batch strategy (each batch comes from a single dataset) with oversampling for small datasets and undersampling for large datasets to ensure balanced contributions. Video datasets can be used as both image and video sets (labeled as I&V).
    - **Design Motivation**: Mixed sampling struggles to handle inconsistent dimensions between images and videos. The alternating strategy has been proven effective in multi-dataset training.

### Loss & Training

The loss function is a temporally weighted angular loss: $\mathcal{L}_{gaze} = \frac{1}{T}\sum_{t=1}^{T} \frac{180}{\pi} \arccos(\frac{\hat{g}_t^T g_t}{\|\hat{g}_t\| \|g_t\|})$, which directly measures the angular error (in degrees) between the predicted and ground-truth gaze vectors. In the two-stage training, the second stage jointly trains the 3D data and GazeFollow pseudo-labeled data.

## Key Experimental Results

### Main Results

| Dataset | Metric (Angular Error °↓) | ST-WSGE (Img) | ST-WSGE (Vid) | Prev. SOTA (Supervised) |
|--------|---------|-------|-------|---------|
| Gaze360 Full | MAE | 13.2 | 12.2 | 13.6 / 12.6 |
| GFIE | MAE (Img/Vid) | 15.9 / 15.5 | - | 21.9 / 20.9 |
| MPIIFaceGaze | MAE | 6.4 | - | 7.4 |

Cross-domain generalization improvement is particularly significant on GFIE (~27% relative improvement), proving that the diversity of GazeFollow data effectively bridges the domain gap.

### Ablation Study

| Configuration | Gaze360 Img | GFIE Img | MPIIFaceGaze | Description |
|------|-----------|---------|-------------|------|
| Supervised (w/o GF) | 13.6 | 21.9 | 7.4 | Baseline |
| WS (Directly using 2D labels)| 13.1 | 16.1 | 6.5 | 2D labels are effective |
| ST (3D prediction only) | 13.6 | 20.2 | 7.4 | Pure self-training has limited effect |
| ST-WSGE (Pseudo 3D labels) | 13.2 | 15.9 | 6.4 | Optimal |

### Key Findings

- GaT outperforms both Swin(2D)-LSTM and Swin(2D)-Tr baselines across all training modality configurations, especially when trained with combined images and videos (I&V).
- Training with image datasets can improve video inference performance (cross-modality gain), validating the benefit of a unified architecture.
- The addition of GazeFollow data brings the most significant improvement to cross-domain performance (GFIE error reduces from 21.9° to 15.9°), showing that data diversity is the core bottleneck.

## Highlights & Insights

- **The geometric projection design for pseudo-label generation is highly elegant**: It does not rely on heuristic rules or depth estimation networks; instead, it lifts 2D annotations to 3D with a simple vector rotation. This is because 2D annotations already contain precise $x$ and $y$ spatial direction information, lacking only $z$. Meanwhile, the pre-trained model's $z$ estimation is inherently more robust than its $x$ and $y$ estimation (since the range of $z$ variation is relatively small).
- **Practical value of the modality-agnostic design**: A single model handles both images and videos simultaneously, allowing different datasets to mutually benefit each other and avoiding the resource waste of training two separate models.
- **Transferability to other 3D estimation tasks**: Any task with abundant 2D annotations but lacking 3D annotations (such as 3D human pose or 3D hand pose estimation) can benefit from a similar self-training pseudo-label strategy.

## Limitations & Future Work

- The quality of pseudo-labels depends on the accuracy of the $z$-direction estimation from the stage-one model. For extreme head poses (e.g., completely facing away), the $z$ estimation may be unreliable.
- Only validated on a single 2D dataset (GazeFollow); other gaze following datasets (such as VideoAttentionTarget) have not been tested.
- Self-training was only conducted for one iteration; multi-iteration or curriculum learning strategies might yield further improvements.
- GaT utilizes a tiny Swin3D with limited model capacity; larger models may provide stronger generalization.

## Related Work & Insights

- **vs Kothari et al. (LAEO)**: Uses 2D labels from the LAEO dataset + head-fitting heuristics to generate pseudo 3D labels, but LAEO's gaze distribution is restricted to the horizontal direction and requires mutual gaze between at least two people. This work uses GazeFollow to cover a broader range of gaze distributions and scenes without relying on heuristics.
- **vs MCGaze**: MCGaze uses multi-scale spatiotemporal interaction modules but focuses only on in-domain performance, whereas this work optimizes both in-domain and cross-domain generalization.
- **vs ViT-based methods**: The large patch size of standard ViT is sub-optimal for gaze estimation (where the eye region is too small). Swin3D's small patch size + hierarchical structure provides a more suitable design choice.

## Rating

- Novelty: ⭐⭐⭐⭐ The pseudo-label generation method and modality-agnostic design are novel, though the self-training framework itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-dataset cross-domain evaluation with thorough ablation studies, though lacking qualitative/visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, coherent logic, and high-quality figures/tables.
- Value: ⭐⭐⭐⭐ Highly practical for wild gaze estimation; the pseudo-label methodology is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](../../ECCV2024/human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)
- [\[CVPR 2025\] WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)

</div>

<!-- RELATED:END -->
