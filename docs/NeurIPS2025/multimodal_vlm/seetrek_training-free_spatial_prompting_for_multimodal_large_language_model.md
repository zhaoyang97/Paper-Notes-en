---
title: >-
  [Paper Note] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial Understanding] This paper proposes See&Trek, a training-free and GPU-free spatial prompting framework that enhances spatial understanding in MLLMs through maximum semantic richness…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Spatial Understanding"
  - "Multimodal Large Language Model"
  - "Visual Prompting"
  - "Visual Odometry"
  - "Training-Free"
date: 2026-05-08
content_hash: c79b5147d1d7388e
---

# See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model

**Conference**: NeurIPS 2025
**arXiv**: [2509.16087](https://arxiv.org/abs/2509.16087)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Spatial Understanding, Multimodal Large Language Model, Visual Prompting, Visual Odometry, Training-Free

## TL;DR
This paper proposes See&Trek, a training-free and GPU-free spatial prompting framework that enhances spatial understanding in MLLMs through maximum semantic richness sampling and motion reconstruction, achieving up to 3.5% improvement on VSI-Bench.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have achieved remarkable progress on image understanding and VQA tasks, yet spatial reasoning remains a critical weakness, particularly in scenarios involving object localization, motion prediction, and physical interaction.

**Limitations of Prior Work**: Current MLLMs commonly adopt uniform temporal sampling strategies (e.g., extracting 8 or 32 frames) when processing video, which introduces two fundamental problems:
   - **Visual Homogeneity**: Uniform sampling tends to select frames with few salient features (e.g., walls, ceilings), reducing the signal-to-noise ratio of input frames.
   - **Unknown Motion**: Relying solely on sampled frames without explicit ego-motion information prevents the model from inferring object motion and displacement, forcing it to rely on commonsense priors acquired during pretraining.

**Key Challenge**: Spatial reasoning requires rich visual semantics and explicit motion information, yet existing pipelines lack both.

**Key Insight**: Leveraging off-the-shelf perception models (YOLO) and visual odometry (VO) to inject spatial cues into MLLMs **without any training**.

**Core Idea**: Visual diversity is increased through maximum semantic richness sampling, while camera trajectory is recovered via motion reconstruction and encoded onto keyframes.

## Method

### Overall Architecture
Given a video sequence, See&Trek proceeds in three steps: (1) detecting objects with YOLO and selecting semantically richest keyframes via Maximum Semantic Richness Sampling; (2) estimating the camera motion trajectory using ORB features and the essential matrix (Motion Reconstruction); (3) encoding motion information onto keyframes as spatiotemporal tokens (Spatiotemporal Encoding), which are then combined with text prompts and fed into the MLLM.

### Key Designs

1. **Maximum Semantic Richness Sampling (Balanced-TopK)**:

    - **Function**: Select $K$ keyframes from the video that are both semantically rich and temporally well-distributed.
    - **Mechanism**: YOLO is first applied to detect the object category set $\mathcal{C}_t$ in each frame; the frame with the most distinct object categories is selected as the anchor frame. The valid interval is then divided into $K-1$ temporal segments, and within each segment the frame with the fewest category overlaps with already-selected frames and the most objects is chosen.
    - **Design Motivation**: Naive TopK selection is biased toward temporally clustered high-object-count frames. Balanced-TopK addresses temporal locality bias by combining temporal segmentation with a category-deduplication strategy, balancing semantic richness and temporal diversity.

2. **Motion Reconstruction**:

    - **Function**: Estimate the camera motion trajectory from monocular video.
    - **Mechanism**: ORB features are extracted and matched between adjacent frames; RANSAC is applied to estimate the essential matrix $\mathbf{E}$, which is decomposed via SVD to obtain relative rotation $\mathbf{R}_t$ and translation $\mathbf{T}_t$. The global trajectory is accumulated recursively as $\mathbf{T}_t^{world} = \mathbf{R}_{t-1}^{world}\mathbf{T}_t + \mathbf{T}_{t-1}^{world}$.
    - **Design Motivation**: Explicit camera motion information enables MLLMs to reason about spatial relationships based on evidence rather than speculation.

3. **Spatiotemporal Encoding**:

    - **Function**: Visually encode trajectory information onto keyframes.
    - **Mechanism**: Each keyframe is assigned a color marker (from a continuous colormap reflecting temporal order) and a frame index, overlaid directly on the upper-right corner of the image. BEV and 3D trajectory visualizations are additionally generated.
    - **Design Motivation**: This addresses the association problem between keyframes and motion trajectories — MLLMs cannot link independent frames to standalone trajectory plots. Spatiotemporal encoding bridges this gap through visual markers.

### Loss & Training
Completely training-free. Only CPU-level computation is required for ORB matching and YOLO inference, with a single forward pass through the MLLM.

## Key Experimental Results

### Main Results — VSI-Bench

| Model | Baseline Avg. | +See&Trek | Gain |
|-------|--------------|-----------|------|
| InternVL3-1B | 29.5 | 32.0 | +3.5% |
| InternVL3-8B | 40.2 | 43.2 | +3.0% |
| InternVL3-14B | 44.2 | 45.6 | +1.4% |
| Qwen2.5-VL-7B | 27.3 | 29.0 | +2.6% |
| LLaVA-OneVision-7B | 31.4 | 33.0 | +1.6% |
| Kimi-VL-A3B | 33.4 | 35.1 | +1.7% |

### Ablation Study — STI-Bench

| Model | Baseline | +See&Trek | Static Gain | Dynamic Gain |
|-------|----------|-----------|-------------|--------------|
| InternVL3-8B | 35.2 | 36.5 | +0.5% | +3.4% |
| Qwen2.5-VL-7B | 30.3 | 33.2 | +1.8% | +4.3% |

### Key Findings
- See&Trek yields the largest gains on **Relative Direction (Rel. Dir.)** and **Approach Order (Appr. Order)**, the two subtasks most dependent on motion information.
- Minor performance drops are occasionally observed on **Object Count (Obj. Count)**, as semantic sampling may miss frames in which certain object categories appear.
- Smaller models (1B/3B) benefit more, indicating that the method effectively compensates for limited spatial reasoning capacity in compact models.
- Fully training- and GPU-free; plug-and-play compatible with both open-source and proprietary models.

## Highlights & Insights
- **Zero-cost enhancement**: Spatial understanding is improved purely through input augmentation without modifying model parameters or requiring additional training — a highly practical engineering approach.
- **Elegant design of Balanced-TopK**: By combining temporal segmentation with category deduplication, the strategy maximizes information content within a limited frame budget, and is transferable to other video understanding scenarios.
- **Motion reconstruction as "free features"**: Classical CV visual odometry pipelines provide MLLMs with a structured spatial prior that is more reliable than end-to-end learned representations.

## Limitations & Future Work
- VO relies on feature matching and may fail in texture-poor or fast-motion scenarios.
- Balanced-TopK depends on YOLO's detection capabilities and is ineffective for object categories outside YOLO's vocabulary.
- The method is applicable only to video inputs and cannot handle single-image spatial reasoning.
- Occasional negative effects on Object Count tasks suggest the need for task-adaptive activation of individual components.

## Related Work & Insights
- **vs. SpatialRGPT/LLaVA-3D**: These methods require additional modalities such as depth maps or point clouds, whereas See&Trek operates on RGB video alone.
- **vs. VideoRAG/VideoTree**: These approaches require fine-tuning the MLLM or rely on VLMs for retrieval, incurring significant computational overhead; See&Trek is entirely training-free.
- Future work could integrate the See&Trek framework with depth estimation models to obtain more precise spatial information.

## Rating
- Novelty: ⭐⭐⭐⭐ First zero-training spatial prompting framework with a concise and effective design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10+ models and two benchmarks with comprehensive results.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure with highly informative figures and tables.
- Value: ⭐⭐⭐⭐ Strong engineering practicality; plug-and-play deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
