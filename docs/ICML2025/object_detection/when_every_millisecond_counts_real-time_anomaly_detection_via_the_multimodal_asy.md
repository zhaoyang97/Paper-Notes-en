---
title: >-
  [Paper Note] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network
description: >-
  [ICML 2025 Spotlight][Object Detection][Real-Time Anomaly Detection] A multimodal asynchronous hybrid network is proposed, which combines the high temporal resolution of event cameras (processed via asynchronous GNN) with the rich spatial features of RGB cameras (processed via CNN). This achieves an inference speed of 579 FPS and an average response time of 1.17s in traffic anomaly detection, introducing event streams to the field of autonomous driving anomaly detection for t…
tags:
  - "ICML 2025 Spotlight"
  - "Object Detection"
  - "Real-Time Anomaly Detection"
  - "Event Camera"
  - "Multimodal Fusion"
  - "Asynchronous Graph Neural Network"
  - "Autonomous Driving Safety"
date: 2026-05-08
content_hash: 967a1958032e8e1a
---

# When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2506.17457](https://arxiv.org/abs/2506.17457)  
**Code**: [Yes](https://github.com/PKU-XD/EventAD)  
**Area**: Autonomous Driving  
**Keywords**: Real-Time Anomaly Detection, Event Camera, Multimodal Fusion, Asynchronous Graph Neural Network, Autonomous Driving Safety

## TL;DR

A multimodal asynchronous hybrid network is proposed, which combines the high temporal resolution of event cameras (processed via asynchronous GNN) with the rich spatial features of RGB cameras (processed via CNN). This achieves an inference speed of 579 FPS and an average response time of 1.17s in traffic anomaly detection, introducing event streams to the field of autonomous driving anomaly detection for the first time.

## Background & Motivation

Autonomous driving anomaly detection (e.g., sudden appearance of pedestrians, abnormal vehicle behavior) is crucial for driving safety. Existing approaches mainly face the following challenges:

**Accuracy vs. Speed Trade-off**: SOTA methods (such as TTHF) use complex deep networks to pursue higher-accuracy detections, but suffer from high inference latency (only 3 FPS). In actual driving scenarios, a latency of several hundred milliseconds can mean the difference between safe braking and a collision.

**Temporal Blind Spots in Frame-Level Detection**: Traditional RGB cameras capture images at a fixed frame rate (e.g., 30 FPS). During the "temporal blind spot" of approximately 33ms between frames, high-speed targets may have already traveled a significant distance.

**Prior Methods Ignoring Response Time Metrics**: Most existing methods focus solely on detection accuracy metrics such as AUC and AP, without incorporating response time into their evaluation systems.

**Key Challenge**: How to achieve millisecond-level response times while maintaining high detection accuracy?

**Key Insight**: Introduce a novel sensor, the event camera. Operating at microsecond-level temporal resolution, event cameras asynchronously detect luminance changes, generating sparse event streams that are naturally suited for capturing rapid physical dynamics. By fusing event streams complementarily with RGB images, an asynchronous GNN is used to process the event streams to maintain low latency, while a CNN is employed to extract spatial features from the RGB frames.

**Core Idea**: Design a multimodal asynchronous hybrid network that processes event streams asynchronously (without waiting for frame synchronization), enabling inter-frame anomaly detection and pushing the inference speed to ~600 FPS.

## Method

### Overall Architecture

The network consists of two parallel branches and an anomaly detection head:
- **RGB Branch**: ResNet50 is used to extract spatial appearance features + YOLOX for object detection.
- **Event Branch**: An asynchronous GNN (DAGr) processes the event stream to extract temporal dynamic features.
- **Fusion + Detection**: Unidirectional feature fusion (CNN $\rightarrow$ GNN) + GRU temporal modeling + attention mechanism + risk score prediction.

### Key Designs

1. **Asynchronous Event Graph Construction and Processing**: An event camera outputs an event stream $E = \{e_i = (x_i, t_i, p_i)\}$, where $x_i$ represents pixel coordinates, $t_i$ represents the timestamp, and $p_i \in \{-1, 1\}$ represents the polarity. An event is triggered when the change in log intensity exceeds a threshold $C$: $|\Delta L| > C$. Events are modeled as graph nodes, with normalized spatio-temporal coordinates $\hat{x}_i = (u_i/W, y_i/H)$, $\hat{t}_i = \beta t_i$, and edges are constructed based on spatio-temporal proximity (within radius $R$, with up to 16 neighbors per node). A Deep Asynchronous GNN (DAGr) with residual graph convolution layers and spline convolutions is used to process the nodes:

    $$f_i' = W_c f_i + \sum_{j \in \mathcal{N}(i)} W(e_{ij}) f_j$$

   Spline convolution accelerates inference through lookup tables, maintaining a much lower time complexity than standard attention mechanisms. **Design Motivation**: To preserve the asynchronous and sparse nature of event streams, avoiding information loss and latency introduced by aggregating events into frame-like representations.

2. **Unidirectional Multimodal Fusion (CNN $\rightarrow$ GNN)**: CNN intermediate feature maps $G_I = \{g_I^l\}_{l=1}^L$ are used to enhance GNN node features via spatial sampling:

    $$f_i' = [f_i, g_I(\hat{x}_i)]$$

   This is formulated as concatenating corresponding CNN features sampled at the spatial coordinates of each event node. The key design is **unidirectional sharing**—features flow only from the CNN to the GNN, and the GNN does not feed back into the CNN. **Design Motivation**: In event-sparse scenarios (such as static or slow-motion scenes where few events are generated), RGB features can compensate for the lack of event information, while avoiding bidirectional communication that would increase computational latency.

3. **Spatio-Temporal Relation Learning and Attention-based Anomaly Detection**: For each detected object $i$, event features $o_{t,i} = \text{AsyncGNN}(E_{t,i}; \theta_{\text{GNN}})$ and CNN features $g_{t,i}$ are extracted and concatenated, followed by dimensionality reduction to obtain the fused feature $f_{t,i}$. Two separate GRUs model the temporal dependencies:

    $$h_{b,t,i} = \text{GRU}(b_{t,i}, h_{b,t-1,i}; \theta_1)$$
    $$h_{f,t,i} = \text{GRU}(f_{t,i}, h_{f,t-1,i}; \theta_2)$$

   where $b_{t,i}$ represents bounding box features and $f_{t,i}$ represents the fused features. An attention mechanism dynamically assigns weights to targets: $\alpha_{b,t} = \text{softmax}(\tanh(H_{b,t}^\top w_b))$, forcing the model to focus on potential anomalous targets. The final risk score is formulated as: $s_{t,i} = \text{softmax}(\phi(\hat{h}_{t,i}; \theta_3))$.

### Loss & Training

- **Object Detection Loss**: IoU loss + class loss + regression loss (YOLOX framework).
- **Anomaly Detection Loss**: Weighted cross-entropy loss, with a negative class weight of 0.27 and a positive class weight of 1.0 in the ROL dataset (to handle class imbalance).
- **Optimizer**: GRU-attention module is optimized using Adam ($lr=0.001$), while GNN-ResNet is optimized using AdamW ($lr=2\times 10^{-4}$).
- **Learning Rate Scheduler**: ReduceLROnPlateau.
- **Training Scale**: RGB component for 30 epochs (batch size 64), event component for 150,000 iterations (batch size 32, approximately 2500 epochs over the data).
- **Event Data Generation**: The v2e tool is used to convert RGB videos into simulated event streams (since existing real-world datasets lack the event modality).

## Key Experimental Results

### Main Results

Evaluated on two traffic anomaly detection benchmarks: ROL and DoTA.

| Method | AUC (ROL) | AUC (DoTA) | mTTA (ROL) | FPS | mResponse (ROL) |
|--------|------|------|------|------|------|
| FOL-Ensemble | 0.849 | 0.866 | 2.05s | 33 | 2.16s |
| MAMTCF | 0.841 | 0.862 | 2.01s | 98 | 1.88s |
| AM-Net | 0.855 | 0.874 | 2.18s | 61 | 1.96s |
| STFE | 0.862 | 0.881 | 2.23s | 77 | 2.04s |
| TTHF | 0.871 | 0.891 | 2.35s | **3** | 2.46s |
| **Ours** | **0.879** | **0.896** | **2.80s** | **579** | **1.17s** |

Core Advantages:
- Achieves the best AUC (0.879 on ROL, 0.896 on DoTA), while running **193$\times$ faster** than TTHF in terms of FPS.
- Achieves an mResponse of only 1.17s (on ROL), which is **more than twice as fast** as TTHF (2.46s).
- Reaches an mTTA of 2.80s, providing earlier warnings than all comparison methods.

### Ablation Study

| Configuration | AUC | AP | mTTA | mAP | Description |
|------|-----|-----|------|-----|------|
| RGB+Event Only | 0.805 | 0.479 | 1.44 | 41.66 | Base Multimodal |
| +GRU | 0.817 | 0.508 | 1.98 | 43.59 | Crucial for temporal modeling |
| +Attention | 0.823 | 0.518 | 2.06 | 35.76 | Focuses on anomalous targets |
| +BBox | 0.839 | 0.531 | 2.11 | 43.29 | Improves spatial localization |
| +Object | 0.845 | 0.539 | 1.96 | 42.94 | Object-level features |
| +Two-stage | 0.868 | 0.561 | 2.51 | 43.82 | Two-stage optimization |
| Full Model | **0.879** | **0.570** | **2.80** | **45.15** | All components |

Network depth and backbone ablation:

| Configuration | AUC | FPS | mResponse | Description |
|------|-----|-----|-----------|------|
| 4 layers (Default) | 0.879 | 579 | 1.17s | Optimal speed |
| 5 layers | 0.885 | 312 | 1.31s | Slight accuracy boost, speed halved |
| 6 layers | 0.892 | 166 | 1.56s | Accuracy-speed trade-off |
| CNN $\rightarrow$ Swin | 0.881 | 278 | 1.44s | Slight accuracy boost via Transformer |
| CNN $\rightarrow$ ViT-B | 0.886 | 213 | 1.51s | Stronger global modeling |

### Key Findings

1. **GRU is the most critical component**: Integrating GRU improves the AUC from 0.805 to 0.817 and mTTA from 1.44s to 1.98s, demonstrating that temporal accumulation is vital for anomaly anticipation.
2. **Unique Value of the Event Modality**: The event stream provides inter-frame anomaly detection capabilities. In scenarios where high-speed targets suddenly appear, anomalies can be detected before the next RGB frame arrives.
3. **Overwhelming Speed Advantage**: An inference speed of 579 FPS means each inference step takes only about 1.7ms, which is far below the RGB frame interval (33ms), making real-time response highly practical.
4. **Scalability of Network Depth**: Increasing the number of layers slightly improves accuracy (+1.3% AUC) at the cost of a linear increase in latency, offering flexibility in accuracy-speed trade-offs.

## Highlights & Insights

- **First to introduce event cameras to traffic anomaly detection**: Pioneering the use of high temporal resolution and asynchronous characteristics of event streams to address the response time bottleneck.
- **Inter-frame detection capability**: Traditional methods can only detect anomalies when a new image frame arrives. In contrast, this approach utilizes event streams to detect fast-moving anomalous targets between frames (as demonstrated by the inter-frame detection of a sudden pedestrian in Figure 5).
- **Elegant unidirectional fusion design**: The CNN $\rightarrow$ GNN unidirectional feature sharing compensates for lack of information when events are sparse, while avoiding the latency overhead associated with backward communication.
- **Key contribution of the mResponse metric**: An average detection latency metric, mResponse, is proposed across multiple thresholds, offering a more comprehensive assessment of real-time performance than conventional single-threshold evaluations.
- **Practically oriented**: Firmly integrates response time into the evaluation framework, emphasizing that "detection without sufficient time to react" is practically equivalent to a failure in detection.

## Limitations & Future Work

1. **Simulated Event Data**: The event streams are converted from RGB videos using v2e, which introduces a domain gap relative to real event cameras. Future work could validate the method on real-world event driving datasets such as DSEC.
2. **Object Detection Reliance on RGB Frames**: YOLOX bounding boxes still rely on RGB frames. The event stream is only utilized for feature enhancement, which does not fully exploit the low-latency potential of event cameras. Future research can explore pure event-based detection to further reduce end-to-end latency.
3. **Limited Scenario Coverage**: Both ROL and DoTA datasets focus on front-view dashcam scenarios. Multi-view scenarios and extreme weather conditions have not been studied.
4. **Simple Attention Mechanism**: The model currently utilizes a basic tanh+softmax attention mechanism. Cross-attention or Transformers could be investigated to further improve the focus on anomalies, though speed trade-offs must be carefully weighed.
5. **Complex Training Strategy**: The two modalities are trained separately, leading to intricate hyperparameter tuning. End-to-end joint training might be more optimal.

## Related Work & Insights

- **vs. TTHF**: TTHF achieves a higher frame-level AUC through textual information fusion (0.847 vs. 0.736) but runs at only 3 FPS, making it undeployable in practice. This work outperforms TTHF in global AUC while running 193$\times$ faster.
- **vs. MOVAD**: MOVAD first proposed the concept of online traffic anomaly detection. This work pushes the boundaries further to millisecond-level real-time detection.
- **vs. DAGr**: Inherits the asynchronous GNN architecture from DAGr to process event streams, extending its application from object detection to anomaly detection.
- **Inspiration**: The paradigm of event cameras combined with asynchronous processing can be extended to other safety-critical perception tasks (e.g., pedestrian intent prediction, collision warning), particularly in time-sensitive scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce event cameras to traffic anomaly detection; the concept of inter-frame detection is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, comprehensive ablation studies, and backbone comparisons are provided, although the event data is simulated.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear (response time is formulated), the architecture diagram is intuitive, and the evaluation metrics are well-designed.
- Value: ⭐⭐⭐⭐⭐ Improves response time to an entirely new order of magnitude, is highly practically oriented, and has direct significance for autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When Transformers Meet Mamba: A Hybrid Transformer-Mamba Network for Video Object Detection](../../CVPR2026/object_detection/when_transformers_meet_mamba_a_hybrid_transformer-mamba_network_for_video_object.md)
- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](../../CVPR2025/object_detection/efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[CVPR 2025\] TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision](../../CVPR2025/object_detection/tornadonet_real-time_building_damage_detection_with_ordinal_supervision.md)
- [\[ICML 2025\] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)

</div>

<!-- RELATED:END -->
