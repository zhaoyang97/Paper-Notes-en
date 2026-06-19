---
title: >-
  [Paper Note] Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation
description: >-
  [CVPR 2026][Human Understanding][Vision Transformer] Without modifying the plain ViT backbone and lightweight decoder of ViTPose, TAR-ViTPose employs "Joint-centered Temporal Aggregation (JTA) + Global Restore Attention (GRA)" to align, aggregate, and inject joint features from adjacent frames back into the current frame. This plug-and-play approach improves 2D video pos
tags:
  - CVPR 2026
  - Human Understanding
  - Vision Transformer
  - ViTPose
date: 2026-05-08
content_hash: bd2379c55a17d54e
---
# Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_Beyond_Static_Frames_Temporal_Aggregate-and-Restore_Vision_Transformer_for_Human_Pose_CVPR_2026_paper.html)  
**Code**: https://github.com/zgspose/TARViTPose  
**Area**: Video Understanding / Human Pose Estimation  
**Keywords**: Video Pose Estimation, Vision Transformer, Temporal Aggregation, Joint-level Attention, ViTPose  

## TL;DR
Without modifying the plain ViT backbone and lightweight decoder of ViTPose, TAR-ViTPose employs "Joint-centered Temporal Aggregation (JTA) + Global Restore Attention (GRA)" to align, aggregate, and inject joint features from adjacent frames back into the current frame. This plug-and-play approach improves 2D video pose estimation by +2.3 mAP on PoseTrack2017 compared to single-frame ViTPose, while achieving higher speeds (413 fps for ViT-S).

## Background & Motivation
**Background**: Although 2D Human Pose Estimation (HPE) is primarily deployed in video scenarios, current mainstream practices still rely on single-frame methods that process frames independently. Recently, Vision Transformers (especially ViTPose) have pushed single-frame HPE to SOTA on static images due to their simple design of a plain ViT backbone and lightweight decoder.

**Limitations of Prior Work**: Single-frame ViTPose treats each frame as an isolated image, completely ignoring the inherent temporal continuity in videos. When encountering video-specific degradations such as motion blur, occlusion, or defocus, single-frame predictions become unstable, jittery, or even fail—precisely the most common challenging scenarios in real-world videos.

**Key Challenge**: Existing methods attempting to adapt ViTPose for video follow an "external attachment" route—using pre-trained ViTPose solely as a single-frame feature extractor, followed by an additional suite of specially designed temporal fusion modules (Transformer-based like DSTA/CM-Pose/MTPose, or Mamba-based like GLSMamba) and matching decoders. This increases pipeline complexity and inference costs, deviating from the "simplicity" of plain ViT. Even works like Poseidon, which reuse the ViTPose lightweight decoder, only perform a coarse fusion of multiple frames via simple cross-attention, failing to precisely align joint features corresponding across frames.

**Goal**: Can temporal modeling be **directly embedded** within the ViTPose framework rather than attached externally—leveraging multi-frame temporal cues while preserving the plain ViT design and lightweight decoding pipeline?

**Key Insight**: The authors observe that different joints often have **relatively independent temporal trajectories** during movement. For example, while running, wrists swing back and forth while the head remains mostly forward. Therefore, treating all feature tokens equally for global attention is inappropriate; temporal dependencies must be modeled **per joint** to ensure accurate alignment of corresponding keypoint features across frames.

**Core Idea**: Assign a learnable query token to each joint, use "mask-aware attention" to aggregate (Aggregate) temporal features only from **its corresponding region** in adjacent frames, and then use cross-attention to inject (Restore) the aggregated temporal information back into the current frame's token sequence. Finally, reuse the original ViTPose decoder to regress heatmaps. This is Temporal **A**ggregate-and-**R**estore.

## Method

### Overall Architecture
TAR-ViTPose follows the two-stage top-down paradigm of "detection followed by estimation." In the first stage, a human detector locates individuals in the current frame $X(t)$. After expanding the bounding box by 25%, a specific video clip $S_i$ is cropped from a continuous sequence $S=\langle X(t{-}T),\dots,X(t),\dots,X(t{+}T)\rangle$ centered at the current frame with a span of $T$ (the paper uses $T=2$, i.e., 2 frames before and after, totaling 4 auxiliary frames). The second stage is the focus of this work: a **shared ViT encoder** extracts features $F^{out}_i(\tau)$ for each frame in the clip. The temporal modeling modules are **inserted after the encoder and before the decoder** to aggregate joint features from adjacent frames into the current frame, which are then fed into the original ViTPose lightweight decoder to regress heatmaps $H_i(t)$.

The entire pipeline adds only two lightweight modules: **JTA** is responsible for "joint-level alignment and aggregation of cross-frame features," and **GRA** is responsible for "restoring the aggregated results to the current frame." All other components (ViT encoder, decoder) remain inherited from ViTPose, making it plug-and-play.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Human Clip Si<br/>Current frame + T frames before/after"] --> B["Shared ViT encoder<br/>Frame-wise features F_out(τ)"]
    B --> C["Lightweight Decoder (Shared)<br/>Frame-wise Heatmaps H(τ)"]
    C -->|Threshold φ Binarization| D["Joint-centered Temporal Aggregation JTA<br/>Per-joint query + mask-aware attention<br/>Cross-frame alignment & aggregation → eQ"]
    B --> D
    D --> E["Global Restore Attention GRA<br/>eQ injected back to current frame tokens"]
    B -->|Current frame F_out(t) as query| E
    E --> F["Enhanced Features → Original Decoder<br/>Regress current frame heatmap Hi(t)"]
```

### Key Designs

**1. Joint-centered Temporal Aggregation (JTA): Each joint tracks its own trajectory**

JTA addresses the problem of how to aggregate cross-frame features without interference. A direct approach would be self-attention across all frame features or cross-attention using the current frame as the query and adjacent frames as keys/values (as in Poseidon), but this treats all tokens equally and ignores independent joint trajectories. JTA assigns a learnable query token $Q\in\mathbb{R}^{N\times C}$ to each of the $N$ joints ($N=15$ in PoseTrack). These queries perform cross-attention with feature tokens from all frames to aggregate temporal information **per joint**:

$$\tilde{Q} = \mathrm{JTA}\big(Q,\ \{F^{out}_i(\tau)\}_{\tau=t-T}^{t+T}\big)$$

Specifically, 6 layers of identical structures are stacked: each layer performs "feature-to-joint" cross-attention (joint queries retrieve features from adjacent frames) followed by "joint-to-joint" self-attention (interaction between joint queries), iterating from $Q_0=Q$ to $\tilde{Q}=Q_6$. Each resulting query token aggregates the "temporal features of the same joint across multiple frames."

**2. Mask-aware Feature-to-Joint Attention: Heatmaps as spatial gating**

To ensure the "right wrist query" only focuses on the right wrist region in adjacent frames without being distracted by the background or other joints, JTA uses a mask-aware mechanism. The frame-wise ViT features $F^{out}_i(\tau)$ are first passed through the **shared lightweight decoder** to obtain heatmaps $H(\tau)$. These are used to construct a binary spatial mask for each joint in each frame:

$$M^{j}_{x,y}(\tau)=\begin{cases}0 & H^{j}_{x,y}(\tau)\ge \phi\\ -\infty & \text{otherwise}\end{cases}$$

The threshold $\phi$ defaults to 0.2. Locations with heatmap responses above the threshold are set to 0 (allowing attention), otherwise to $-\infty$ (forcing attention to 0 after softmax). The mask is resized to the encoder resolution and added to the attention logits for residual masked attention:

$$Q_l=\mathrm{softmax}\big(f_q(Q_{l-1})f_k(F^{out}_i)^\top+M\big)f_v(F^{out}_i)+Q_{l-1}$$

Here, $F^{out}_i$ is the concatenated matrix of tokens from all frames. This forces each joint query to "focus only near its own keypoint," precisely aligning temporal features of the same joint.

**3. Global Restore Attention (GRA): Injecting temporal cues without losing global context**

While $\tilde{Q}$ is rich in cross-frame temporal information, it consists of only $N$ joint tokens and lacks global spatial context between keypoints. Directly regressing heatmaps from these queries (as in DSTA) results in a significant performance drop (to 70.3 mAP in ablations, 11.4 lower than the single-frame baseline). GRA uses cross-attention with the current frame tokens $F^{out}_i(t)$ as the query and $\tilde{Q}$ as both key and value to inject the aggregated temporal semantics back into the current frame spatial features:

$$\hat{F}^{out}_i(t)=\mathrm{CrossAttn}\big(F^{out}_i(t),\ \tilde{Q},\ \tilde{Q}\big)$$

The spatio-temporally enhanced features $\hat{F}^{out}_i(t)$ are then fed into the **original** ViTPose lightweight decoder. This step ensures that enhancement occurs at the "full token sequence" level, supplementing temporal cues while **fully preserving** the current frame's inherent global context.

### Loss & Training
The entire model (ViT encoder, JTA, GRA, decoder) is trained end-to-end using the Mean Squared Error (MSE) between predicted and GT heatmaps:

$$L=\sum_i\sum_{j=1}^{N}\big\lVert H^{j}_i(t)-G^{j}_i(t)\big\rVert_2^2$$

The encoder/decoder are initialized with ViTPose weights pre-trained on COCO, while temporal modules (JTA/GRA) are randomly initialized. Training uses a single RTX A6000 for 30 epochs with $\phi=0.2$. Inference uses simple IoU tracking to ensure each person crop passes through the backbone only once, with features reused across frames to increase speed.

## Key Experimental Results

### Main Results
On the PoseTrack2017 val set, TAR-ViTPose consistently outperforms single-frame ViTPose across ViT-S/B/L/H backbones, with notable gains for difficult joints like wrists and ankles:

| Backbone | Method | Wrist | Ankle | Mean mAP |
|------|------|-------|-------|----------|
| ViT-S | ViTPose | 75.0 | 70.4 | 80.1 |
| ViT-S | **TAR-ViTPose** | 77.8 | 74.2 (+3.8) | **81.9 (+1.8)** |
| ViT-B | ViTPose | 77.7 | 73.9 | 81.7 |
| ViT-B | **TAR-ViTPose** | 80.3 | 77.3 (+3.4) | **84.0 (+2.3)** |
| ViT-H | ViTPose | 81.6 | 77.8 | 84.7 |
| ViT-H | **TAR-ViTPose** | 83.8 | 80.2 | **86.8 (+2.1)** |

Compared to SOTA video methods (PoseTrack2017 val, Faster R-CNN boxes): The ViT-H version reaches 86.8 mAP, exceeding DSTA (85.6) by 1.2 points. Even the smallest ViT-S version (81.9) is 4.6 points higher than HRNet-W48 (77.3). When using GT boxes (marked with *), the ViT-H version reaches 90.3 mAP, surpassing Poseidon (88.9) by 1.4 points.

Efficiency (A6000, 2 auxiliary frames, batch=16):

| Method | Backbone | #Params | FPS | mAP |
|------|------|---------|-----|-----|
| PoseWarper | HRNet-W48 | 71.1M | 52 | 81.0 |
| DSTA | ViT-H | 422.2M | 25 | 84.3 |
| **TAR-ViTPose** | ViT-S | 35.6M | **413** | 81.5 |
| **TAR-ViTPose** | ViT-H | 672.5M | 28 | **86.3** |

### Ablation Study
Conducted on PoseTrack2017 val, ViT-B, $T=2$:

| Configuration | Key Metrics | Description |
|------|---------|------|
| (a) Full-frame self-attention | 82.2 mAP / 38.22 GFLOPs | All tokens aggregated together |
| (b) Current-aux cross-attention | 82.6 mAP / 12.74 GFLOPs | Current frame as query |
| (c) Aux self-attn + (b) | 82.8 mAP / 16.99 GFLOPs | Poseidon-style |
| (d) **Ours (Joint-level)** | **84.0 mAP / 3.89 GFLOPs** | JTA+GRA, efficient & accurate |
| JTA only (No GRA) | 70.3 mAP | Regression from joint queries loses context |
| JTA + GRA (Full) | 84.0 mAP | +2.3 over single-frame baseline 81.7 |
| Without mask | 82.6 mAP | Attention scatters into background |
| With mask (Full) | 84.0 mAP | Mask provides +1.4 mAP |

### Key Findings
- **GRA is critical**: Removing GRA and regressing directly from joint queries (like DSTA) drops performance to 70.3 mAP, worse than the single-frame baseline (81.7). This proves joint tokens lack global context and must be restored to the full spatial sequence.
- **Joint-level modeling is efficient**: Compared to full-frame self-attention (82.2 mAP / 38.22 GFLOPs), the proposed scheme (84.0 mAP / 3.89 GFLOPs) is 1.8 points more accurate with less than 1/10th of the computation.
- **Mask-aware attention contributes +1.4 mAP**: Visualization shows that without the mask, joint query attention scatters to the background; with the mask, it adheres tightly to corresponding keypoints.
- **Difficult joints benefit most**: Joints with heavy motion or occlusion (e.g., ankles, wrists) see the most significant gains.

## Highlights & Insights
- **Plug-and-play Lightweightness**: JTA + GRA add only ~16.6M parameters and minimal inference overhead without modifying the ViTPose backbone or decoder, essentially adding video capabilities to any pre-trained ViTPose "for free."
- **Heatmaps as Attention Masks**: Using the lightweight decoder's heatmaps from auxiliary frames as spatial gating provides spatial priors for cross-attention "for free," avoiding explicit optical flow or alignment networks.
- **Importance of Aggregate-Restore Decoupling**: Concentrating temporal information into a few joint tokens (low-dimensional aggregation) first, then injecting it back into the full spatial sequence (preserving global context), decouples "temporal alignment" from "global localization."

## Limitations & Future Work
- The work does not handle **temporal pose tracking**; it focuses strictly on estimating the pose for the current frame, relying on simple IoU for tracking.
- The ViT-H version has 672.5M parameters, which involves high memory and deployment costs despite reasonable throughput.
- Dependence on heatmap quality: If auxiliary frame heatmaps fail due to extreme degradation, the mask might block correct regions, leading to cascaded failures.
- The temporal span $T=2$ is fixed; adaptive spans for varying action speeds warrant further exploration.

## Related Work & Insights
- **vs ViTPose (Single-frame baseline)**: ViTPose is independent per frame; this work adds JTA+GRA to introduce temporal awareness while reusing the backbone/decoder, gaining +2.3 mAP.
- **vs DSTA / CM-Pose / MTPose / GLSMamba (External fusion)**: These use ViTPose only as an extractor and add complex fusion modules/decoders. This work embeds temporal modeling internally, remaining simpler and faster (ViT-H 28 fps vs DSTA 25 fps).
- **vs Poseidon**: Poseidon reuses the ViTPose decoder but uses coarse cross-attention for fusion. This work's joint-level modeling (84.0) significantly outperforms the Poseidon-style configuration (82.8).

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[CVPR 2026\] Egocentric Visibility-Aware Human Pose Estimation](egocentric_visibility-aware_human_pose_estimation.md)
- [\[CVPR 2026\] Differentially Private 2D Human Pose Estimation](differentially_private_2d_human_pose_estimation.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](../../ECCV2024/human_understanding/repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[CVPR 2026\] Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes](beyond_scanpaths_graph-based_gaze_simulation_in_dynamic_scenes.md)

</div>

<!-- RELATED:END -->
