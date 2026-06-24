---
title: >-
  [Paper Note] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs
description: >-
  [ECCV 2024][Object Detection][Self-Supervised Multiple Object Tracking] This paper proposes Walker, the first self-supervised multiple object tracker. By constructing a quasi-dense temporal object appearance graph, designing a multi-positive contrastive loss to optimize random walks on the graph for instance similarity learning, and introducing mutually-exclusive connectivity constraints and a motion-constrained bidirectional walk inference strategy…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Self-Supervised Multiple Object Tracking"
  - "Random Walk"
  - "Temporal Appearance Graph"
  - "Contrastive Learning"
  - "Sparse Annotation"
date: 2026-05-08
content_hash: 156a21036d437c33
---

# WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs

**Conference**: ECCV 2024  
**arXiv**: [2409.17221](https://arxiv.org/abs/2409.17221)  
**Code**: None  
**Area**: Object Detection / Multiple Object Tracking  
**Keywords**: Self-Supervised Multiple Object Tracking, Random Walk, Temporal Appearance Graph, Contrastive Learning, Sparse Annotation

## TL;DR

This paper proposes Walker, the first self-supervised multiple object tracker. By constructing a quasi-dense temporal object appearance graph, designing a multi-positive contrastive loss to optimize random walks on the graph for instance similarity learning, and introducing mutually-exclusive connectivity constraints and a motion-constrained bidirectional walk inference strategy, Walker achieves competitive self-supervised tracking performance on MOT17, DanceTrack, and BDD100K, outperforming prior self-supervised methods even with 400 times fewer annotations.

## Background & Motivation

**Background**: Multiple Object Tracking (MOT) is one of the core tasks in computer vision, requiring simultaneous location (detection) and association of multiple objects in video. State-of-the-art MOT methods rely on massive annotated datasets containing bounding-box annotations alongside unique instance IDs to establish temporal association across frames. In the MOT17 dataset, for instance, this translates into frame-by-frame annotations of tens of thousands of frames along with identity correspondences across frames, making the annotation cost extremely prohibitive.

**Limitations of Prior Work**: (1) **Excessive Annotation Costs**: Dense frame-by-frame bounding-box and instance ID annotations require massive human labor, severely limiting the scale and diversity of training datasets. (2) **Lagging Self-Supervised Tracking Performance**: Existing self-supervised or unsupervised tracking methods underperform full-supervision counterparts by a wide margin, offering limited practical value. (3) **Lack of Self-Supervised Association Learning**: Most self-supervised methods only acquire detection capabilities, while the multi-frame association required for tracking remains heavily reliant on supervised signals.

**Key Challenge**: The core challenge in MOT is temporal association—how to match detections of the same object across different frames. Supervised methods directly learn association through instance ID labels, whereas self-supervised methods lack such explicit cross-frame correspondence signals. How to learn effective instance association without tracking labels (instance IDs) remains a core challenge.

**Goal**: (1) Design a self-supervised learning framework to learn instance association from sparsely annotated videos; (2) Eliminate the requirement for tracking labels (instance IDs) entirely, utilizing only sparse bounding box annotations; (3) Achieve tracking performance competitive with supervised methods.

**Key Insight**: Modeling instance association in MOT as a random walk problem over a graph. By building a temporal graph where nodes are bounding boxes and edge weights represent appearance similarities, the cross-frame correspondence of objects can be automatically discovered by optimizing the random walk paths without explicit tracking labels.

**Core Idea**: Graph-based self-supervised learning of instance similarity by optimizing random walks on a temporal appearance graph, incorporating mutually-exclusive connectivity constraints and bidirectional walk inference to achieve high-quality multiple object tracking.

## Method

### Overall Architecture

Walker's pipeline consists of three phases: (1) **Graph Construction**: Sparse bounding boxes are extracted from video frames to construct a quasi-dense temporal object appearance graph, featuring bounding boxes as nodes and appearance similarity as edge weights. (2) **Self-Supervised Learning**: A multi-positive contrastive loss is designed to optimize random walk paths on the graph to learn instance-level appearance similarity, alongside mutually-exclusive connectivity constraints to optimize graph topology. (3) **Inference**: On a new video, a detector extracts bounding boxes, which are then integrated into trajectories using bidirectional random walks joint with motion constraints.

### Key Designs

1. **Quasi-Dense Temporal Object Appearance Graph**:

    - **Function**: Models the multi-object tracking problem in videos as a graph structure.
    - **Mechanism**: For every frame in a video, bounding boxes of objects are acquired using a detector. Each bounding box is considered a graph node, and its appearance feature is extracted using a CNN or ViT encoder. Directed edges are established between nodes across adjacent frames (as well as frame leaps) and are weighted by the cosine similarity of their appearance features. "Quasi-dense" implies that edges are constructed across all frames within a temporal window, rather than solely adjacent frames, thereby capturing longer-term temporal dependencies. This produces a weighted directed graph $G = (V, E, W)$.
    - **Design Motivation**: Graph structures naturally represent the many-to-many cross-frame affinity relationships in MOT. Quasi-dense connectivity facilitates information propagation across multiple frames, which helps address challenging scenarios like short-term occlusions.

2. **Multi-Positive Contrastive Loss and Random Walk Optimization**:

    - **Function**: Learns instance-level appearance similarity in a self-supervised manner.
    - **Mechanism**: Performs random walks on the constructed temporal graph. A walk starts at a node, randomly picking the next node according to transition probabilities (based on similarity) to simulate tracking. The core innovation is **multi-positive contrastive learning**: starting from an anchor node $x$, multiple random walks are performed, and nodes visited along the paths (belonging to the same instance) are treated as positive samples. Lacking instance labels, positive samples are implicitly discovered through the paths; if the walk transition probability from node $A$ to node $B$ is high, $A$ and $B$ are likely the same instance. The contrastive loss pulls the features of nodes within the same walk path closer, while pushing different instances further apart. The optimization objective is to guide random walks to prefer transitions among nodes of the identical instance.
    - **Design Motivation**: Traditional contrastive learning utilizes only a single positive pair, whereas an object instance in MOT appears across many frames, implying multiple positives. Thus, the multi-positive contrastive loss is more suited for tracking. Random walks provide an intuitive, label-free mechanism to discover positive pairs.

3. **Mutually-Exclusive Connective Properties**:

    - **Function**: Optimizes the learned graph topology to satisfy MOT constraints.
    - **Mechanism**: In MOT, different bounding boxes in the same frame must belong to different instances—representing a critical prior. The mutually-exclusive connectivity constraint utilizes this prior: for two nodes $i$ and $j$ in the identical frame ($i \neq j$), if node $k$ (from another frame) is highly similar to $i$, then the similarity between $k$ and $j$ must be low. Formally, applying softmax normalization over detection results of each frame enforces that the sum of connection weights from any external node to all nodes within the same frame equals 1, creating a mutually-exclusive competitive effect. This makes the graph topology conform better to the one-to-one matching constraints in MOT.
    - **Design Motivation**: Unconstrained random walks might yield high similarities between multiple nodes within the same frame (e.g., different objects with similar appearance), violating the one-to-one matching rule in MOT. Mutually-exclusive constraints act as a vital mechanism to incorporate MOT domain knowledge.

### Loss & Training

The total loss consists of two parts: (1) a multi-positive contrastive loss (a variant of InfoNCE) where positive samples are defined by random walk paths, and negative samples are drawn from different instances in the same batch; (2) a mutually-exclusive connectivity regularization term, imposing competitive constraints on co-frame nodes. Training only requires sparse bounding box annotations (no instance IDs needed), automatically discovering cross-frame correspondences through random walks. Inference leverages motion constraints (e.g., Kalman filter prediction) to bound the search space of bidirectional random walks.

## Key Experimental Results

### Main Results

Performance comparison on three major MOT benchmarks:

| Dataset | Metric | Walker (Ours) | Prev. SOTA | Gain |
|--------|------|-------------|---------------|------|
| MOT17 | HOTA | Competitive | Significantly Lower | Substantial Gain |
| MOT17 | IDF1 | Competitive | Significantly Lower | Substantial Gain |
| DanceTrack | HOTA | Competitive | Significantly Lower | Substantial Gain |
| BDD100K | MOTA | Competitive | Significantly Lower | Substantial Gain |

Walker represents the first self-supervised tracker achieving competitive performance simultaneously on MOT17, DanceTrack, and BDD100K. Even reducing the annotation demand by $400\times$, it still outperforms preceding self-supervised methods.

### Ablation Study

| Configuration | HOTA | IDF1 | Description |
|------|------|------|------|
| Baseline (w/o Random Walk) | Lower | Lower | Appearance matching only |
| + Multi-positive contrastive learning | Improved | Improved | Learns better similarity |
| + Mutually-exclusive connectivity constraints | Significant improvement | Significant improvement | Satisfies MOT one-to-one constraints |
| + Motion-constrained bidirectional walk | **Best** | **Best** | Introduces motion prior at inference |
| 1/400 Annotation scale | Still outperforms prev. SOTA | Still outperforms prev. SOTA | Extremely low annotation requirement |

### Key Findings

- Walker outperforms previous self-supervised trackers even under the extreme condition of $400\times$ fewer annotations, demonstrating the effectiveness of the random walk framework in low-annotation regimes.
- The mutually-exclusive connectivity constraint contributes significantly to performance, indicating that incorporating structural priors of MOT into self-supervised learning is critical.
- Multi-positive contrastive learning is more suitable for MOT scenarios than single-positive contrastive learning.
- Motion-constrained bidirectional walk is more robust than unidirectional walk, particularly in scenes with occlusions and objects of similar appearance.
- On DanceTrack, where appearances are similar and motions are complex, Walker performs exceptionally well, demonstrating that the learned appearance representations are highly discriminative.

## Highlights & Insights

- **First self-supervised MOT to achieve competitive performance**: Signals a shift in self-supervised tracking from "conceptually interesting" to "practically applicable."
- **Natural connection between random walks and MOT**: Since MOT essentially seeks optimal paths on temporal graphs, utilizing random walks to learn this process is highly intuitive.
- **Elegant integration of the mutually-exclusive constraint**: Formulating the "one-to-one matching" structural prior of MOT into a differentiable regularization term successfully introduces domain knowledge while preserving the self-supervised training framework.
- **Extremely low annotation overhead**: A $400\times$ reduction in annotation workload is of immense significance for practical applications.

## Limitations & Future Work

- It still requires sparse bounding box annotations and is not fully unsupervised.
- Inference still relies on external detectors (tracking-by-detection paradigm), making detection quality the performance bottleneck.
- The walk length and sampling frequency of random walks are hyperparameters requiring fine-tuning.
- There remains a performance gap compared to fully supervised SOTA, particularly in highly occluded and long-range tracking scenarios.
- Future research can explore extending random walks to longer temporal horizons or incorporating Transformers for global inference.

## Related Work & Insights

- **CenterTrack / ByteTrack**: Fully supervised MOT methods, serving as the performance benchmarks for this work.
- **DINO / DINOv2**: Self-supervised visual representation learning methods that provide robust appearance features.
- **Random Walk Networks**: Pioneering works that leverage random walks in semantic correspondence and video understanding.
- **QDTrack**: A quasi-dense matching tracking method that inspired the graph construction approach.
- **Insight**: The annotation bottleneck is the primary constraint on the advancement of MOT. Breakthroughs in self-supervised methods will fundamentally drive practical applications.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of random walks and MOT is novel, and the design of the mutually-exclusive constraint is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation across three datasets, in-depth analysis on annotation efficiency)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐ (Significantly reduces annotation requirements for MOT, holding high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[ECCV 2024\] Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)
- [\[CVPR 2025\] Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)
- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)
- [\[ECCV 2024\] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)

</div>

<!-- RELATED:END -->
