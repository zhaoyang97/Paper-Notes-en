---
title: >-
  [Paper Note] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
description: >-
  [ICLR 2026][Autonomous Driving][egocentric video] Apple collected 829 hours of egocentric video paired with 3D hand joint tracking data (EgoDex) using Vision Pro, covering 194 tabletop manipulation tasks and 338K trajectories. The dataset is used to systematically benchmark imitation learning policies (BC/DDPM/FM + Transformer), providing the largest-scale data foundation to date for scaling dexterous manipulation training.
tags:
  - ICLR 2026
  - Autonomous Driving
  - egocentric video
  - dexterous manipulation
  - imitation learning
  - hand pose
  - dataset
date: 2026-05-08
content_hash: 170a29d2562348af
---

# EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video

**Conference**: ICLR 2026
**arXiv**: [2505.11709](https://arxiv.org/abs/2505.11709)
**Code**: [https://github.com/apple/ml-egodex](https://github.com/apple/ml-egodex)
**Area**: Autonomous Driving / Robotics
**Keywords**: egocentric video, dexterous manipulation, imitation learning, hand pose, dataset

## TL;DR
Apple collected 829 hours of egocentric video paired with 3D hand joint tracking data (EgoDex) using Vision Pro, covering 194 tabletop manipulation tasks and 338K trajectories. The dataset is used to systematically benchmark imitation learning policies (BC/DDPM/FM + Transformer), providing the largest-scale data foundation to date for scaling dexterous manipulation training.

## Background & Motivation

**Background**: Robot imitation learning suffers from severe data scarcity. Unlike NLP and 2D vision, which benefit from internet-scale corpora, dexterous manipulation lacks large-scale datasets. The dominant data collection paradigm is teleoperation, as exemplified by Open X-Embodiment and DROID.

**Limitations of Prior Work**: Teleoperation is bottlenecked by physical hardware constraints and is difficult to scale further; collected data is tied to specific robot hardware and generalizes poorly. In-the-wild internet videos (e.g., Ego4D) offer scale but lack precise 3D hand pose annotations, making them unsuitable for training dexterous manipulation policies.

**Key Challenge**: A fundamental tension exists between scalability and annotation precision — teleoperation provides accurate action labels but does not scale, while in-the-wild videos scale but lack the fine-grained dexterous annotations required.

**Goal**: To construct a large-scale dataset that is both passively scalable and equipped with precise 3D hand joint annotations, while establishing a standardized benchmark for evaluating dexterous manipulation capabilities.

**Key Insight**: Apple Vision Pro's multi-camera array, on-device SLAM, and ARKit are leveraged to track the 3D positions and orientations of 25 joints per hand in real time, enabling simultaneous data collection and annotation during natural user interactions.

**Core Idea**: Replace the unscalable teleoperation paradigm with large-scale egocentric video and accurate hand pose data passively collected via wearable XR devices.

## Method

### Overall Architecture

EgoDex's core contributions fall into two parts:

1. **Dataset Construction**: 829 hours and 90M frames of egocentric video with synchronized 3D hand skeleton data are collected using Vision Pro, covering 194 tabletop manipulation tasks (338K trajectories).
2. **Benchmark Evaluation**: Two evaluation tasks are defined — trajectory prediction and inverse dynamics — with Transformer models trained under the X-IL framework and systematically evaluated.

Data pipeline: Vision Pro recording → 1080p@30Hz video + 30Hz skeletal joints + camera intrinsics/extrinsics → compressed storage (2TB) → train/eval split (99%/1%).

### Key Designs

1. **Data Collection System**:

    - **Function**: Collects natural manipulation data using Vision Pro + visionOS 2 + ARKit.
    - **Mechanism**: The wearer requires no additional equipment and manipulates objects naturally. ARKit leverages the device's on-board multi-camera array and SLAM to track in real time the 3D positions and orientations of the head, arms, wrists, and 25 joints per hand. Recordings are organized into sessions (10–15 minutes), with episode boundaries marked by pause/resume events. Video is compressed with modern codecs (raw 500TB → 2TB).
    - **Design Motivation**: Unlike teleoperation, which requires a robot and active human control, Vision Pro data collection is *passively scalable* — as XR glasses proliferate, large volumes of data can accumulate naturally. Compared to post-hoc methods such as HaMeR, on-device real-time tracking achieves higher accuracy by leveraging known intrinsics/extrinsics and multiple viewpoints.

2. **Action Representation**:

    - **Function**: Encodes the action at each timestep as a 48-dimensional vector.
    - **Mechanism**: $\mathbf{a}_t$ = 2 hands × (3D wrist position + 6D wrist orientation + 5 fingertips × 3D position) = 48 dimensions. Actions are expressed in the current camera coordinate frame using relative trajectories.
    - **Design Motivation**: Compared to representing only wrist positions (as in EgoMimic), including the 3D positions of each fingertip captures the fine-grained information necessary for dexterous manipulation.

3. **Task Design and Diversity**:

    - **Function**: 194 tasks are categorized into three types — reversible tasks (paired inverse operations), reset-free tasks (terminal state lies within the initial state distribution), and reset tasks.
    - **Mechanism**: Reversible and reset-free tasks eliminate time-consuming environment resets, improving collection efficiency. GPT-4 is used to integrate collector-provided metadata (task name, description, environment, objects) into structured natural language annotations.
    - **Design Motivation**: Compared to DROID, EgoDex exhibits a broader distribution of action verbs — DROID has many verbs appearing fewer than 10 times, whereas most verbs in EgoDex appear more than 1,000 times.

4. **Benchmark Design**:

    - **Function**: Defines two standardized evaluation protocols — trajectory prediction and inverse dynamics.
    - **Trajectory Prediction**: $f_\theta(\mathbf{o}_{0..t}, \mathbf{s}_{0..t}, l) = \hat{\mathbf{a}}_{t:t+H}$, predicting the next $H$ actions given an image sequence, skeleton sequence, and language description.
    - **Inverse Dynamics**: Additionally conditions on a goal image $\mathbf{o}_{t+H}$ to predict intermediate trajectories. The goal image reduces multimodality.
    - **Evaluation Metric**: Best-of-K — $K$ predictions are sampled and the one closest to ground truth is selected; the average 3D Euclidean distance across 12 keypoints (2 wrists + 10 fingertips) is reported.

### Loss & Training

- The X-IL framework is used to train 2 architectures × 3 policies = 6 model variants:
    - **Architectures**: Encoder-Decoder Transformer and Decoder-only Transformer.
    - **Policies**: Behavior Cloning (BC, deterministic), Denoising Diffusion Policy (DDPM, stochastic), Flow Matching (FM, stochastic).
- Trained for 50K steps, batch size 2048, on 8×A100 GPUs.
- A total of 14 model variants are trained and evaluated (varying horizon, goal conditioning, data scale, and model size).

## Key Experimental Results

### Main Results

Trajectory prediction results at a 2-second prediction horizon ($H=60$):

| Model | Avg Dist (K=1) | Avg Dist (K=10) | Final Dist (K=1) | Final Dist (K=10) |
|------|---------------|-----------------|-------------------|-------------------|
| Dec + BC | 0.045 | 0.045 | 0.062 | 0.062 |
| Dec + DDPM | 0.053 | 0.041 | 0.071 | 0.044 |
| Dec + FM | 0.052 | 0.040 | 0.071 | 0.043 |
| EncDec + BC | **0.044** | 0.044 | **0.060** | 0.060 |
| EncDec + DDPM | 0.052 | 0.039 | 0.071 | 0.043 |
| EncDec + FM | 0.051 | **0.038** | 0.070 | **0.041** |

### Ablation Study

| Configuration | Avg Dist (m) | Final Dist (m) | Notes |
|------|-------------|----------------|------|
| H=30 (1s) | 0.031 | 0.049 | Short horizon, highest accuracy |
| H=60 (2s) | 0.045 | 0.062 | Default horizon |
| H=90 (3s) | 0.053 | 0.069 | Long horizon, increased error |
| w/o goal image | 0.045 | 0.062 | No goal conditioning |
| w/ goal image | 0.035 | 0.029 | Final dist reduced by 53% |
| 200M params | 0.045 | 0.062 | Default model size |
| 500M params | 0.045 | 0.062 | No gain from scaling model |

### Key Findings
- **EncDec > Dec-only**: The encoder-decoder architecture consistently outperforms the decoder-only architecture across all policies, though the margin is small.
- **BC vs. Stochastic Policies**: BC achieves the best performance at $K=1$ (deterministic predictions are on average more accurate), while FM/DDPM are superior at $K=5$/$K=10$ (capable of sampling better modes); FM outperforms BC by 34% at $K=10$.
- **Goal Images Substantially Reduce Final-Point Error**: Visual goal conditioning reduces the final distance from 0.062 to 0.029 (↓53%), as the goal image provides an "anchor" for the trajectory endpoint, alleviating multimodality.
- **Performance Scales with Data Volume**: Performance improves consistently as data volume increases (log-linear relationship), validating the scaling hypothesis.
- **No Benefit from 500M Model**: This indicates the current 200M model is sufficient and that the bottleneck lies in data rather than model capacity.

## Highlights & Insights
- **Passively Scalable Data Paradigm**: Collecting manipulation data with consumer XR devices proposes a path toward an "ImageNet moment" for robotics datasets — data can accumulate naturally as XR glasses become widespread. This paradigm is transferable to any domain requiring large-scale human behavior data (e.g., gesture recognition, sign language translation).
- **Reversible Task Design Eliminates Reset Overhead**: By designing paired inverse tasks (e.g., plugging and unplugging a charger), the terminal state of one task becomes the initial state of the other, greatly improving collection efficiency. This trick is applicable to any data collection scenario.
- **Best-of-K Evaluation Metric**: This elegantly addresses the inherent multimodality of human motion — given the same initial state, multiple reasonable trajectories exist, and a single ground-truth metric would penalize correct but different predictions.

## Limitations & Future Work
- **Limited Scene Diversity**: All data is collected in tabletop environments, lacking diverse settings such as kitchens or outdoor scenes. The authors suggest using image generation models for visual augmentation.
- **Reduced Annotation Accuracy Under Occlusion**: In heavily occluded scenarios such as folding towels, ARKit's hand tracking accuracy degrades (as the tracker is itself model-based).
- **Embodiment Transfer Gap Not Validated**: The paper does not present experiments demonstrating transfer from human data to robot policies, only discussing potential approaches (co-training, pretraining + fine-tuning, etc.). This is the most critical missing link.
- **Object Interaction Modeling Absent**: Only hand pose is tracked; object pose and contact point annotations are absent, limiting the ability to learn hand-object interaction dynamics.

## Related Work & Insights
- **vs. DROID (teleoperation)**: DROID contains 76K trajectories / 86 tasks / 19M frames; EgoDex contains 338K trajectories / 194 tasks / 90M frames, surpassing DROID comprehensively in scale. However, DROID data can be used directly for robot training, whereas EgoDex requires cross-embodiment transfer.
- **vs. EgoMimic**: The most closely related prior work, also using egocentric video and hand tracking. EgoMimic covers only 4 hours of data and tracks only wrist positions, whereas EgoDex provides 829 hours with full finger joint tracking — a substantial improvement in both scale and precision.
- **vs. Ego4D**: Ego4D contains 3,000 hours of video but lacks 3D hand pose annotations and does not focus on manipulation tasks, making it unsuitable for direct use in dexterous manipulation training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using Vision Pro for large-scale dexterous manipulation data collection is a first in terms of scale and quality, though the paradigm of collecting human behavior data with wearable devices is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Fourteen model variants and multiple ablations are systematically evaluated, but the critical robot transfer experiment is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Structure is clear, figures and tables are abundant, and the dataset comparison table is immediately interpretable.
- **Value**: ⭐⭐⭐⭐⭐ As a dataset paper, the potential impact is substantial — 829 hours of open-source data could advance the entire field of dexterous manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](../../CVPR2026/autonomous_driving/learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [\[CVPR 2026\] Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal](../../CVPR2026/autonomous_driving/ghost-fwl_a_large-scale_full-waveform_lidar_dataset_for_ghost_detection_and_remo.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](../../CVPR2026/autonomous_driving/searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)

</div>

<!-- RELATED:END -->
