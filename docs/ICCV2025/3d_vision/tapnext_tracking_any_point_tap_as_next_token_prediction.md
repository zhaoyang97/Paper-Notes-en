---
title: >-
  [Paper Note] TAPNext: Tracking Any Point (TAP) as Next Token Prediction
description: >-
  [ICCV 2025][3D Vision][Point Tracking] TAPNext reformulates Tracking Any Point (TAP) in video as a sequential masked token decoding task…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Point Tracking"
  - "Next Token Prediction"
  - "Online Tracking"
  - "Occlusion Handling"
  - "TAP"
date: 2026-05-08
content_hash: 3caf03f7fc42ec4d
---

# TAPNext: Tracking Any Point (TAP) as Next Token Prediction

**Conference**: ICCV 2025
**arXiv**: [2504.05579](https://arxiv.org/abs/2504.05579)  
**Code**: [https://tap-next.github.io/](https://tap-next.github.io/)  
**Area**: 3D Vision / Video Understanding
**Keywords**: Point Tracking, Next Token Prediction, Online Tracking, Occlusion Handling, TAP

## TL;DR

TAPNext reformulates Tracking Any Point (TAP) in video as a sequential masked token decoding task, eliminating the tracking-specific inductive biases and heuristics prevalent in conventional approaches. It achieves causal online tracking and establishes new state-of-the-art results among both online and offline trackers, with remarkably low inference latency.

## Background & Motivation

**Background**: Tracking Any Point (TAP) in video is a fundamental computer vision task with broad applications in robotics, video editing, and 3D reconstruction. Dominant methods such as TAPIR, CoTracker, and BootsTAP rely on carefully engineered tracking-specific components, including correlation pyramids, temporal windowing, and iterative refinement.

**Limitations of Prior Work**: Existing methods suffer from several critical issues: (1) heavy reliance on tracking-specific inductive biases (e.g., correlation volume construction, heuristic occlusion estimation) constrains generality and scalability; (2) many methods require temporal windowing—processing multiple frames simultaneously—which introduces latency and additional complexity; (3) iterative optimization (e.g., iterative updates in TAPIR) incurs substantial inference cost.

**Key Challenge**: There is an inherent tension between tracking performance and architectural simplicity. To improve accuracy, existing methods continuously stack specialized components, leading to increasingly complex systems. In contrast, experience from NLP demonstrates that simple, unified architectures (e.g., Transformer with next token prediction) combined with large-scale training can consistently outperform carefully tuned task-specific designs.

**Goal**: The paper aims to design a minimalist point tracking framework that removes all tracking-specific inductive biases, and to verify whether a simple sequence prediction paradigm can match or surpass the state of the art on TAP benchmarks.

**Key Insight**: Inspired by autoregressive models in natural language processing, the authors observe that a point trajectory is intrinsically a sequence—the point's location at each frame constitutes a token. Framing tracking as "predict the next token given preceding tokens" allows the task to be solved with a standard sequence model.

**Core Idea**: TAP is reformulated as sequential masked token decoding. Image features from video frames and existing trajectory points serve as known tokens, while the unknown position in the next frame is represented as a masked token. A Transformer directly predicts the masked token, enabling fully causal, frame-by-frame online tracking.

## Method

### Overall Architecture

TAPNext takes a video frame sequence and query points as input, and outputs the position and visibility (occlusion flag) of each query point at every frame. Processing is fully online: upon receiving a new frame, the model accepts that frame's image tokens alongside accumulated trajectory tokens, and predicts the position and occlusion state of all query points for the current frame in a single forward pass. The core network adopts a hybrid architecture combining ViT-based feature extraction with state space model (SSM)-based sequence modeling.

### Key Designs

1. **Image–Trajectory Hybrid Token Sequence**:

    - Function: Provides a unified token sequence representation for video frames and point trajectories.
    - Mechanism: Each video frame is encoded into a set of image tokens via a ViT encoder. The historical trajectory (position + visibility) of each query point is encoded as a trajectory token. When processing each frame, the current frame's image tokens and all query points' trajectory tokens are concatenated into a single sequence and fed into Transformer layers for self-attention interaction. Crucially, the attention mask is causal—trajectory tokens can only attend to information from the current and preceding frames—ensuring the online processing property.
    - Design Motivation: Unifying heterogeneous information (image features and trajectory signals) into a homogeneous token sequence allows the Transformer's expressive power to be fully leveraged, without requiring hand-crafted correlation computation or matching strategies.

2. **Masked Token Prediction Mechanism**:

    - Function: Implements the core "next position prediction" decoding paradigm.
    - Mechanism: For query points whose positions in the new frame have not yet been determined, a learnable masked token $[M]$ is used as a placeholder. The Transformer aggregates information from image tokens and historical trajectory tokens via attention to fill in the masked token—outputting the $(x, y)$ coordinates and occlusion probability of the query point in the current frame. The prediction head is a simple MLP that regresses position and classifies occlusion from the final hidden state of the masked token.
    - Design Motivation: This design directly draws on the masked language modeling idea from BERT and the autoregressive generation paradigm from GPT. Compared to template matching or iterative refinement in conventional trackers, a single forward pass prediction is more efficient, and end-to-end training allows the model to automatically learn the optimal information aggregation strategy.

3. **State Space Model (SSM) for Temporal Modeling**:

    - Function: Efficiently models long-horizon trajectory states.
    - Mechanism: In addition to Transformer layers, a Mamba-style SSM is introduced to handle the temporal evolution of trajectory tokens. The SSM maintains a hidden state with linear complexity, continuously compressing historical trajectory information. Compared to the quadratic complexity of a pure Transformer, the SSM is significantly more efficient for long videos. The Transformer handles intra-frame image–trajectory interaction, while the SSM manages inter-frame temporal state propagation.
    - Design Motivation: The computational cost of pure Transformers for long sequences is prohibitive. The linear complexity of the SSM enables TAPNext to process long videos efficiently, and the recurrent structure of the SSM is naturally suited to online processing—only a single hidden state update is required per frame.

### Loss & Training

Training supervises position prediction with a Huber loss: $L_{pos} = \text{Huber}(\hat{p}_t - p_t)$, and occlusion classification with a binary cross-entropy loss: $L_{occ} = \text{BCE}(\hat{o}_t, o_t)$. The total loss is $L = L_{pos} + \lambda L_{occ}$. Training data comes from the Kubric synthetic dataset and pseudo-labels on real TAP-Vid videos generated by BootsTAP.

## Key Experimental Results

### Main Results

TAP-Vid Query First benchmark (Average Jaccard, higher is better):

| Method | DAVIS | Kinetics | Type | Online |
|--------|-------|----------|------|--------|
| TAPIR | 58.5 | 52.3 | Specialized | No |
| BootsTAP | 62.4 | 55.8 | Specialized | No |
| TAPTRv3 | 63.2 | 54.5 | Specialized | No |
| CoTracker3 | 63.8 | 55.8 | Specialized | No |
| TAPNext | **65.2** | **57.3** | Token Prediction | **Yes** |

Computational efficiency comparison (latency for tracking 1024 points):

| Method | H100 Latency (ms) | V100 Latency (ms) | Temporal Windowing |
|--------|-------------------|-------------------|--------------------|
| TAPIR | 28.5 | 89.3 | Yes |
| CoTracker3 | 15.2 | 42.1 | Yes |
| TAPNext (256 pts) | **5.05** | **14.2** | **No** |
| TAPNext (512 pts) | **5.26** | **18.0** | **No** |
| TAPNext (1024 pts) | **5.33** | **23.0** | **No** |

### Ablation Study

| Configuration | DAVIS AJ↑ | Note |
|---------------|-----------|------|
| Full TAPNext | 65.2 | Complete model |
| w/o SSM (pure Transformer) | 63.1 | Long-video performance drops without SSM |
| w/o masked token (correlation-based) | 62.8 | Replacing with conventional correlation computation degrades performance |
| w/o pseudo-label training | 61.5 | Training on synthetic data only |
| w/ temporal windowing (offline) | 65.5 | Marginal gain with windowing but online property is lost |

### Key Findings

- TAPNext surpasses all offline trackers in an online setting, demonstrating that carefully engineered inductive biases are not strictly necessary—end-to-end training can automatically learn equivalent or superior strategies.
- SSM contributes substantially to long-sequence tracking (AJ improvement of 2.1), validating the necessity of efficient temporal modeling.
- Visualizations reveal that TAPNext spontaneously learns behaviors analogous to hand-designed occlusion detection and motion segmentation in traditional methods—PCA visualizations show that the model's attention naturally distinguishes foreground from background.
- Inference latency is extremely low (5 ms/frame on H100), 3–5× faster than offline methods, making TAPNext highly practical for real-world deployment.
- A notable performance degradation is observed on videos exceeding 150 frames, attributable to the maximum sequence length of 48 frames used during training.

## Highlights & Insights

- **"Less is More" Design Philosophy**: Removing all tracking-specific components and adopting the most general sequence prediction framework still achieves state-of-the-art results, which is a significant insight for the field. This suggests that when model capacity and training data are sufficient, general-purpose architectures can learn better strategies than hand-crafted designs.
- **Emergent Behaviors**: Visualizations reveal that TAPNext autonomously learns motion segmentation (foreground–background separation) and occlusion detection—capabilities that are explicitly engineered in traditional methods. This parallels emergent abilities observed in large language models and represents an intriguing property of end-to-end sequence model training.
- **Speed Advantage of Online Tracking**: The causal design eliminates the need for frame buffer windows; each frame requires only a single forward pass, resulting in very low latency that is well-suited to real-time applications such as visual feedback in robotic manipulation.

## Limitations & Future Work

- Tracking performance degrades significantly on long videos (>150 frames) due to the 48-frame sequence length limit during training, which represents the most critical current bottleneck.
- The choice of SSM variant has a non-negligible effect on final performance, but this aspect is not sufficiently explored in the paper.
- The scaling behavior with respect to model size and training data volume remains understudied, yet may be a key direction for further improvement.
- The paradigm is validated only for point tracking; extending it to more complex visual tracking tasks (e.g., multi-object tracking, segmentation tracking) constitutes a valuable future direction.

## Related Work & Insights

- **vs. TAPIR**: TAPIR employs a correlation pyramid with iterative updates, representing the canonical template-matching paradigm. TAPNext entirely abandons the matching paradigm in favor of sequence prediction and achieves superior performance.
- **vs. CoTracker3**: CoTracker performs joint multi-point tracking with temporal windowing, leveraging inter-point relationships but requiring offline processing. TAPNext operates in a purely online manner yet remains stronger.
- **vs. BootsTAP**: BootsTAP employs bootstrapped training to generate pseudo-labels for expanding training data. Notably, TAPNext also leverages BootsTAP pseudo-labels during training, highlighting the critical importance of data quality and scale for model performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reformulating TAP as next token prediction is an elegant and profound paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive TAP-Vid benchmark results and detailed efficiency analysis, though scaling law analysis is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, experimental design is rigorous, and visualization analyses are persuasive.
- Value: ⭐⭐⭐⭐⭐ Advances the TAP state of the art while providing a new methodological direction for visual tracking, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[NeurIPS 2025\] Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention](../../NeurIPS2025/3d_vision/dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[NeurIPS 2025\] TAPIP3D: Tracking Any Point in Persistent 3D Geometry](../../NeurIPS2025/3d_vision/tapip3d_tracking_any_point_in_persistent_3d_geometry.md)
- [\[ICCV 2025\] TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction](tar3d_creating_high-quality_3d_assets_via_next-part_prediction.md)

</div>

<!-- RELATED:END -->
