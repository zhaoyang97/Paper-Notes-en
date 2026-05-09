---
title: >-
  [Paper Note] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs
description: >-
  [CVPR 2026][3D Vision][image-to-point-cloud registration] Hg-I2P introduces a Heterogeneous Graph to jointly model relationships between 2D image regions and 3D point cloud regions. Through multi-path adjacency mining for learning cross-modal edges, heterogeneous-edge-guided feature adaptation, and graph-based projection consistency pruning, it achieves state-of-the-art generalization and accuracy across six indoor and outdoor cross-domain benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - image-to-point-cloud registration
  - heterogeneous graph
  - cross-modal feature adaptation
  - correspondence pruning
  - cross-domain generalization
date: 2026-05-08
content_hash: 72d5ed3abdd93143
---

# Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs

**Conference**: CVPR 2026
**arXiv**: [2603.27969](https://arxiv.org/abs/2603.27969)
**Code**: [https://github.com/anpei96/hg-i2p-demo](https://github.com/anpei96/hg-i2p-demo)
**Area**: 3D Vision
**Keywords**: image-to-point-cloud registration, heterogeneous graph, cross-modal feature adaptation, correspondence pruning, cross-domain generalization

## TL;DR

Hg-I2P introduces a Heterogeneous Graph to jointly model relationships between 2D image regions and 3D point cloud regions. Through multi-path adjacency mining for learning cross-modal edges, heterogeneous-edge-guided feature adaptation, and graph-based projection consistency pruning, it achieves state-of-the-art generalization and accuracy across six indoor and outdoor cross-domain benchmarks.

## Background & Motivation

1. **Background**: Image-to-point-cloud (I2P) registration aims to establish correspondences between 2D pixels and 3D points, serving as a cornerstone for visual localization, navigation, and 3D reconstruction. Recent learning-based methods have advanced the field by improving backbone networks, matching strategies, and loss functions; for example, MATR adopts coarse-to-fine matching, while CoFiI2P introduces patch-level matching.

2. **Limitations of Prior Work**: Existing methods perform well within their training domain but suffer significant performance degradation in unseen scenes. The fundamental reason is the large distributional gap between 2D image features (appearance-based) and 3D point cloud features (geometry-based)—even correct correspondences may exhibit low feature similarity, making it difficult for neural networks to discriminate valid matches.

3. **Key Challenge**: Existing improvements either focus solely on feature refinement (lacking explicit cross-modal reasoning) or solely on correspondence pruning (relying on depth prediction or hand-crafted heuristics). Handling these two aspects in isolation fails to systematically address the generalization problem. Although visual foundation models (SAM, DepthAnything, etc.) can help bridge the modality gap, a unified framework that simultaneously exploits feature refinement and correspondence pruning has been lacking.

4. **Goal**: (1) How to construct a unified structure that supports both cross-modal feature refinement and correspondence pruning? (2) How to effectively learn the cross-modal mapping between 2D and 3D regions? (3) How to exploit consistency information within the graph structure to filter erroneous matches?

5. **Key Insight**: 2D/3D SAM is used to segment images and point clouds into regions, and a heterogeneous graph is constructed to model inter-region relationships. The heterogeneous edges (I2P edges) define a 2D–3D region mapping that can guide both feature refinement (cross-modal message passing along edges) and correspondence pruning (projection consistency checking within the graph).

6. **Core Idea**: A heterogeneous graph is used to jointly model 2D–3D region relationships, enabling cross-modal feature adaptation and correspondence pruning simultaneously within the same graph structure, thereby achieving robust and generalizable I2P registration.

## Method

### Overall Architecture

Given an RGB image and a colored point cloud, 2D/3D SAM is first applied to segment them into $M$ 2D regions and $N$ 3D regions, forming a heterogeneous graph $\mathcal{G}_H = (\mathcal{V}_H, \mathcal{E}_H)$. Three core modules are then applied: (1) **MP-mining** to learn heterogeneous edges $\mathcal{E}_{I2P}$; (2) **HE-adapting** to perform cross-modal feature message passing along heterogeneous edges for feature refinement; and (3) **HC-pruning** to filter erroneous correspondences via graph-based projection consistency. Finally, refined features are used for feature-level matching to obtain 2D–3D correspondences, and pose is estimated via RANSAC-PnP.

### Key Designs

1. **Heterogeneous Graph Definition and Construction**:

    - **Function**: Establishes a unified relational structure representing both 2D image and 3D point cloud regions.
    - **Mechanism**: Vertices $\mathcal{V}_H = \mathcal{V}_I \cup \mathcal{V}_P$ correspond to $M$ 2D regions and $N$ 3D regions respectively, each represented by a $c$-dimensional feature vector (average pooling of features within the region). Edges are categorized into three types: (a) homogeneous 2D–2D edges $\mathbf{E}_{I2I}$, defined by 2D region feature distance $e^{-\alpha\|\mathbf{v}_i^I - \mathbf{v}_j^I\|_2^2}$; (b) homogeneous 3D–3D edges $\mathbf{E}_{P2P}$, defined by 3D region feature distance; (c) heterogeneous 2D–3D edges $\mathbf{E}_{I2P}$, which theoretically should be defined by the IoU of 3D regions projected onto 2D under the GT pose, but since the GT pose is unavailable at inference time, these edges must be learned.
    - **Design Motivation**: Unlike prior methods that process 2D and 3D features in isolation, the heterogeneous graph provides a unified framework for jointly modeling intra-modal and inter-modal relationships, enabling both feature refinement and correspondence pruning to be performed within the same structure.

2. **MP-mining (Multi-Path Adjacency Mining)**:

    - **Function**: Learns heterogeneous edges $\mathcal{E}_{I2P}$ at inference time (without GT pose).
    - **Mechanism**: Leverages the known homogeneous edges $\mathbf{E}_{I2I}$ and $\mathbf{E}_{P2P}$ to mine 2D–3D adjacency relationships through three paths: $\mathbf{E}_{I2P}^1 = \mathbf{E}_{I2I}\tilde{\mathbf{E}}_{I2P}$ (relayed through 2D neighbors), $\mathbf{E}_{I2P}^2 = \tilde{\mathbf{E}}_{I2P}\mathbf{E}_{P2P}$ (relayed through 3D neighbors), and $\mathbf{E}_{I2P}^3 = \mathbf{E}_{I2I}\tilde{\mathbf{E}}_{I2P}\mathbf{E}_{P2P}$ (relayed through two hops). The three matrices are concatenated and passed through an attention layer to predict the final $\hat{\mathbf{E}}_{I2P}$.
    - **Design Motivation**: From a Bayesian inference perspective, multi-path adjacency relationships capture indirect causal connections between 2D and 3D regions—even when the initial direct matching $\tilde{\mathbf{E}}_{I2P}$ is inaccurate, propagation through homogeneous region neighbors can correct the estimate.

3. **HE-adapting (Heterogeneous-Edge-Guided Feature Adaptation)**:

    - **Function**: Performs cross-modal message passing along learned heterogeneous edges to refine 2D and 3D features, enhancing cross-modal matching capability.
    - **Mechanism**: Proceeds in two steps: (a) **Message generation**: for each 2D region $\mathcal{I}_i$, cross-modal messages $\bar{\mathbf{m}}_i^I$ are obtained by weighted aggregation of features from connected 3D region neighbors via $\mathcal{E}_{I2P}$; cross-attention is used to learn the correlation between 2D region features and cross-modal messages. (b) **Message interaction**: original region features and message features are concatenated channel-wise, fused via self-attention, and combined with original features using a blending ratio $\beta$ to produce adapted features. The same operation is applied symmetrically on the 3D side.
    - **Design Motivation**: HE-adapting enables graph-structured cross-modal information flow—2D features are informed by matched 3D geometric information, and 3D features are informed by matched 2D appearance information—thereby narrowing the modality gap and improving cross-domain generalization.

4. **HC-pruning (Graph-Based Projection Consistency Pruning)**:

    - **Function**: Filters erroneous correspondences produced by feature matching.
    - **Mechanism**: An initial pose $\tilde{\mathbf{T}}$ is first estimated via RANSAC-PnP from the refined feature matches. Two complementary pruning criteria are then applied: (a) based on $\mathcal{E}_{I2P}$ adjacency and reprojection distance $\delta_{\text{rej}}$; (b) based on the cosine similarity of relative position vectors derived from graph projections. Correspondences satisfying at least one criterion are retained as inliers.
    - **Design Motivation**: The dual-criterion design effectively handles false matches caused by noisy pose estimation or imperfect edge learning—the two criteria are complementary, one enforcing local distance constraints and the other enforcing global directional consistency.

### Loss & Training

$$L_{\text{Hg-I2P}} = L_{\text{corr}} + \lambda_1 \|\hat{\mathbf{E}}_{I2P}[\text{mask}] - \mathbf{E}_{I2P}[\text{mask}]\|_2^2$$

where $L_{\text{corr}}$ is the standard circle loss for correspondence supervision, and the second term supervises heterogeneous edge learning (computed only at valid non-zero positions).

## Key Experimental Results

### Main Results

Cross-scene I2P registration on the 7-Scenes dataset (trained on one scene, tested on others):

| Method | IR (C→) AVG | RR (C→) AVG | IR (K→) AVG | RR (K→) AVG |
|--------|------------|------------|------------|------------|
| MATR | 0.387 | 0.478 | 0.537 | 0.706 |
| Top-I2P | 0.433 | 0.628 | 0.596 | 0.785 |
| MinCD | 0.445 | 0.592 | 0.568 | 0.814 |
| Hg-I2P† (w/o HC-pruning) | 0.472 | 0.642 | 0.618 | 0.802 |
| **Hg-I2P (Ours)** | **0.581** | **0.667** | **0.688** | **0.853** |

Significant improvements are also observed under cross-dataset settings on RGBD-V2, ScanNet, and others.

### Ablation Study

| Configuration | IR AVG | RR AVG | Note |
|---------------|--------|--------|------|
| Hg-I2P (full) | 0.581 | 0.667 | Full model |
| Hg-I2P† (w/o HC-pruning) | 0.472 | 0.642 | Removing HC-pruning drops IR by 18.8% |
| Baseline MATR | 0.387 | 0.478 | Baseline method |

The inclusion of HC-pruning yields approximately 23% improvement in IR and 4% in RR, demonstrating that graph-based correspondence pruning is critical for accurate registration.

### Key Findings

- The unified heterogeneous graph framework significantly outperforms methods that only perform feature refinement or only perform correspondence pruning.
- The advantage is more pronounced under cross-domain settings (training and testing on different datasets), validating the generalization capability of the proposed approach.
- The heterogeneous edges learned by MP-mining are critical for both HE-adapting and HC-pruning—accurate 2D–3D region mapping forms the foundation of the entire system.
- Compared to prior work that also uses SAM (e.g., An et al.), Hg-I2P more systematically leverages edge information and projection constraints through the graph structure.

## Highlights & Insights

- **Heterogeneous graph as a unified framework**: Feature refinement and correspondence pruning, traditionally treated as separate problems, are unified within a single graph structure, elegantly avoiding the pitfalls of fragmented processing. Graph edges simultaneously serve feature propagation (HE-adapting) and geometric verification (HC-pruning), achieving dual objectives within a single structure.
- **Bayesian interpretation of multi-path adjacency mining**: Treating homogeneous edges as conditional probabilities, the multi-path product corresponds to marginalization in Bayesian inference, providing an elegant theoretical grounding for relational learning on graphs.
- **Coarse-to-fine cross-modal message passing**: Messages are first aggregated at the region level (coarse) and then interacted with original features at the pixel/point level (fine), balancing efficiency and accuracy.

## Limitations & Future Work

- The method depends on the segmentation quality of 2D/3D SAM—poor segmentation in certain scenes (e.g., texture-less regions) may degrade the quality of the heterogeneous graph construction.
- HC-pruning requires an initial pose estimate from RANSAC-PnP; if the initial match quality is too low, the resulting erroneous pose may adversely affect subsequent pruning.
- The number of graph vertices ($M + N$) depends on the segmentation granularity of SAM and may need to be adjusted for different scenes.
- Runtime is not reported; whether the combined overhead of SAM, graph construction, and message passing is suitable for real-time applications remains an open question.

## Related Work & Insights

- **vs. MinCD (Bie et al.)**: MinCD converts I2P registration into 3D–3D registration using DepthAnything, but predicted depth lacks real scale and requires additional alignment. Hg-I2P operates directly in the 2D–3D space, avoiding inaccuracies due to depth scale ambiguity.
- **vs. An et al. (2024)**: Also uses SAM, but only for aligning object pairs to extract correspondences. Hg-I2P goes further by defining a heterogeneous graph structure that systematically exploits edge information for feature adaptation and pruning.
- **vs. MATR**: MATR employs coarse-to-fine matching but lacks cross-modal reasoning; Hg-I2P explicitly introduces cross-modal information flow through graph message passing.

## Rating

- Novelty: ⭐⭐⭐⭐ — Heterogeneous graphs for I2P registration offer a novel perspective; MP-mining and HE-adapting are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Cross-domain experiments across six datasets provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ — Figures and tables are clear and derivations are detailed, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐ — Provides a systematic solution to the generalization problem in I2P registration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](../../ICCV2025/3d_vision/turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[CVPR 2026\] APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition](apc_adversarial_point_counterattack.md)

</div>

<!-- RELATED:END -->
