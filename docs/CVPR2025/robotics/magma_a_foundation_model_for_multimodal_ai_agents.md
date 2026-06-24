---
title: >-
  [Paper Note] Magma: A Foundation Model for Multimodal AI Agents
description: >-
  [CVPR 2025][Robotics][Multimodal Agent] Magma unifies UI screenshots, robot data, and human manipulation videos into a single pre-training framework by labeling interactive regions on images (Set-of-Mark) and tracking motion trajectories in videos (Trace-of-Mark). This enables a single model to possess both multimodal understanding and cross-domain action prediction capabilities, achieving SOTA performance in both UI navigation and robotic manipulation.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Multimodal Agent"
  - "Vision-Language-Action Model"
  - "UI Navigation"
  - "Robotic Manipulation"
  - "Spatial Intelligence"
date: 2026-05-08
content_hash: e25d56541b08e547
---

# Magma: A Foundation Model for Multimodal AI Agents

**Conference**: CVPR 2025  
**arXiv**: [2502.13130](https://arxiv.org/abs/2502.13130)  
**Code**: [https://microsoft.github.io/Magma](https://microsoft.github.io/Magma) (Open Source)  
**Area**: Agent / Robotics / Multimodal VLM  
**Keywords**: Multimodal Agent, Vision-Language-Action Model, UI Navigation, Robotic Manipulation, Spatial Intelligence

## TL;DR
Magma unifies UI screenshots, robot data, and human manipulation videos into a single pre-training framework by labeling interactive regions on images (Set-of-Mark) and tracking motion trajectories in videos (Trace-of-Mark). This enables a single model to possess both multimodal understanding and cross-domain action prediction capabilities, achieving SOTA performance in both UI navigation and robotic manipulation.

## Background & Motivation

**Background**: Current AI Agents based on Vision-Language-Action (VLA) models are typically trained separately for specific domains—using Pix2ACT, WebGUM, or Ferret-UI for UI navigation in the digital world, and RT-2 or OpenVLA for robotic manipulation in the physical world. These models operate in isolation and cannot generalize across domains.

**Limitations of Prior Work**: (1) Most models sacrifice general multimodal understanding to learn task-specific action policies, leading to poor generalization; (2) The inputs (2D screenshots vs. 3D scenes) and action spaces (2D coordinates vs. 7-DoF) of the digital and physical worlds differ vastly, and simple joint training leads to mutual interference; (3) Existing vision-language-action data is limited, whereas massive unlabelled videos and image-text data cannot be directly utilized for action pre-training.

**Key Challenge**: There is an inherent gap between verbal intelligence (semantic understanding) and spatio-temporal intelligence (spatio-temporal reasoning). Semantic annotations are in textual format while action annotations are in spatial coordinate format, causing conflicts in their optimization objectives; simple joint training actually degrades performance.

**Goal**: How to use a unified framework to enable a model to simultaneously learn to "understand" (multimodal understanding) and "act" (action planning), while leveraging massive unlabelled videos for action supervision signals?

**Key Insight**: The authors observe that for both UI and robotics, the essence of an action is "doing something at a certain location." By highlighting all interactive objects on an image with a "Mark", action prediction is simplified to "selecting a mark" rather than "predicting coordinates", significantly reducing the search space and unifying the output format across different domains.

**Core Idea**: Utilizing Set-of-Mark (labeling interactive regions on static images) and Trace-of-Mark (tracking movement trajectories in videos) as proxy tasks to unify heterogeneous image, video, and robotic data under a single "mark prediction" interface for joint pre-training.

## Method

### Overall Architecture
Magma adopts a standard VLM architecture: ConvNeXt-XXLarge as the visual encoder to handle arbitrary resolution images/video frames, and LLaMA-3-8B as the language backbone for autoregressive decoding. The input consists of a sequence of visual observations and a textual task description, while the output is a text token (semantic answer) or a spatial token (action coordinate/mark index). The core innovation lies not in the architecture, but in the annotation methods for the pre-training data and the design of the proxy tasks.

The pre-training data comprises four categories: UI screenshots (2.7 million, SeeClick + Vision2UI), robot trajectories (9.4 million, Open-X-Embodiment), human manipulation videos (25 million, Epic-Kitchen + Ego4d + SomethingV2, etc.), and image-text pairs (1.2 million, ShareGPT4V + LLaVA-1.5), totaling approximately 39 million samples.

### Key Designs

1. **Set-of-Mark (SoM) Action Localization**:

    - **Function**: Highlights all interactive regions on a single image frame, converting "pixel coordinate prediction" into "mark index selection".
    - **Mechanism**: Given an image, domain-specific detectors (DOM trees/Android View Hierarchy for UI, and motion points extracted by CoTracker for video/robotics) are used to extract $K$ candidate regions. Bounding boxes are drawn for each region with numbered labels. The model only needs to predict "which mark to select" instead of exact coordinates. This is formulated as $o_t^{mark} = \pi(\mathcal{I}_t^M, \text{task}, \text{ctx})$, where the output is a subset of the mark set.
    - **Design Motivation**: Directly predicting pixel coordinates involves completely different formats between UI and robotics (2D vs. 7-DoF) and entails a vast search space. By unifying actions with mark indexes, action prediction across different domains is reduced to the same "selection" task, enabling cross-domain joint training.

2. **Trace-of-Mark (ToM) Action Planning**:

    - **Function**: Tracks the future motion trajectories of marked points over a video, converting unlabelled videos into usable "action" training data.
    - **Mechanism**: Given a video segment, CoTracker is used to set $s^2$ grid points on the first frame and track their positions in subsequent frames. Static points with motion smaller than a threshold $\epsilon$ are filtered out (treated as background), retaining foreground trajectories with significant motion. For videos with camera motion (especially ego-centric videos), homography transfromation is applied first to eliminate global motion before extracting the foreground. Finally, K-Means is used to cluster foreground/background trajectories, and representative points are randomly sampled as marks.
    - **Design Motivation**: Compared with predicting the next video frame, predicting trajectories captures action-relevant object dynamics over longer horizons with minimal tokens while ignoring irrelevant environmental content. More importantly, this allows massive videos without action annotations to generate useful training signals.

3. **ConvNeXt Global Encoding + Unified Resolution Processing**:

    - **Function**: Supports arbitrary resolution inputs ranging from standard natural images to high-resolution UI screenshots (up to 2000px).
    - **Mechanism**: ConvNeXt-XXLarge is utilized as the visual encoder, which naturally supports arbitrary resolutions without cropping or patching. High-resolution images are directly encoded globally, skipping local patch cropping or global-local fusion.
    - **Design Motivation**: UI screenshots have very high resolutions with scattered information, whereas robot/video frames have lower resolutions. Leveraging the inherent resolution flexibility of CNNs for unified processing is simpler and more efficient than ViT-patching schemes.

### Loss & Training
Standard autoregressive next-token prediction loss is employed. All outputs (semantic text, UI coordinates, robotic 7-DoF actions, mark indexes, trajectory coordinates) are unified into language tokens: 2D coordinates are normalized and quantized into a 256-bin text vocabulary, while robotic actions are represented using the last 256 rarely-used tokens in the LLM's vocabulary. Full-parameter fine-tuning is performed (including both the visual encoder and the LLM), with a learning rate of 1e-5 and a maximum of 3 epochs.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | Magma-8B | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ScreenSpot-Mobile | Text/Icon Acc | 60.4/58.5 | 78.0/52.0 (SeeClick) | Icon +6.5% |
| ScreenSpot-Desktop | Text/Icon Acc | 75.3/52.9 | 72.2/30.0 (SeeClick) | Icon +22.9% |
| ScreenSpot-Web | Text/Icon Acc | 69.1/52.0 | 55.7/32.5 (SeeClick) | Icon +19.5% |
| VisualWebBench Ele-G | Accuracy | 96.3 | 67.5 (GPT-4V) | +28.8% |
| SimplerEnv-Google Robot | Success Rate | 52.3 | 34.2 (RT-1-X) | +18.1% |
| SimplerEnv-Bridge | Success Rate | 35.4 | 15.9 (Octo) | +19.5% |
| Mind2Web Cross-Website | Step SR | 45.4 | 36.5 (GPT-4V-OmniParser) | +8.9% |
| AITW Overall | Accuracy | 67.3 | 59.3 (SeeClick) | +8.0% |

### Ablation Study

| Configuration | SS-Overall | VWB-Ele-G | SE-Bridge | SE-Google |
|------|---------|---------|---------|---------|
| Magma-8B (UI only) | 57.7 | 68.5 | - | - |
| Magma-8B (OXE only) | - | - | 22.2 | 35.7 |
| Magma-8B (UI+OXE, w/o SoM+ToM) | 56.2 | 89.1 | 17.5 | 31.5 |
| Magma-8B (Full, w/o SoM+ToM) | 57.4 | 90.1 | 17.7 | 37.5 |
| Magma-8B (Full, w/ SoM+ToM) | **61.4** | **96.3** | **35.4** | **52.3** |

### Key Findings
- **SoM+ToM is Critical**: Without SoM/ToM, simply mixing UI and robot data leads to performance degradation on both ends (due to mutual interference). Adding SoM/ToM dramatically boosts performance: UI increases by +4pp, and robotics increases by +17.8pp/+14.8pp.
- **Video Data Requires ToM to Be Effective**: Adding video data without ToM yields only minimal improvement (since text descriptions of videos only enhance verbal intelligence). ToM allows video data to serve as a strong source for spatial intelligence training.
- **Significant Boost in Spatial Reasoning**: Magma substantially outperforms similarly-sized VLMs on VSR (65.1%) and SpatialEval, and approaches GPT-4o performance levels on BLINK (41.0%).
- **Strong Few-Shot Transfer to Robotics**: On LIBERO, fine-tuning with only 10 trajectories significantly outperforms OpenVLA, doubling the success rate.

## Highlights & Insights
- **Ingenious Unified Interface Design with SoM/ToM**: By simply "drawing marks on images", the gap between semantic understanding and action prediction is bridged. Marks serve both as visual grounding and action anchors, facilitating positive transfer across different domains rather than causing conflicts.
- **Turning Unlabelled Videos into Treasure**: ToM extracts action-level supervision signals from any video using point tracking and motion filtering. This implies that massive YouTube videos can be leveraged as Agent pre-training data, significantly elevating the scaling law ceiling.
- **Using ConvNeXt over ViT**: To address the ultra-high resolution needs of UI screenshots, the model bypasses complex cropping and stitching, instead leveraging the resolution-agnostic nature of CNNs directly. This simple and effective idea is transferable to any multimodal task requiring varying input resolutions.

## Limitations & Future Work
- Currently only supports single-step action prediction; long-sequence multi-step planning depends on an external execution loop, lacking intrinsic planning capabilities.
- ToM depends heavily on the quality of CoTracker, which may fail in scenarios with severe occlusions or rapid movements (although the paper evaluated on YouCook2 showing precision=0.89, it might not be robust in more complex settings).
- Representing robotic actions with the last 256 tokens in the vocabulary acts as a hack and cannot capture fine-grained control in continuous action spaces.
- UI and robotic data account for a relatively small portion (~15%) of the pre-training data; scaling up these two data categories further may yield greater benefits.
- The model has not been evaluated on 3D navigation tasks (e.g., Habitat, AI2Thor), leaving the boundaries of its cross-domain generalization unclear.

## Related Work & Insights
- **vs. OpenVLA**: OpenVLA is trained solely on OXE robot data, making it highly domain-specific with poor generalization. In contrast, Magma learns more general spatial intelligence using heterogeneous data coupled with SoM/ToM, outperforming OpenVLA on SimplerEnv zero-shot (doubling the success rate).
- **vs. SeeClick**: SeeClick is based on Qwen-VL and focuses specifically on UI grounding. Magma substantially outperforms SeeClick in the icon category of ScreenSpot (+19.5%), demonstrating that spatial reasoning learned from videos via ToM also transfers beneficially to UI understanding.
- **vs. GPT-4V-OmniParser**: While GPT-4V paired with OmniParser performs well on ScreenSpot, Magma dominates on VisualWebBench element localization with 96.3% vs. 67.5%, proving that pre-trained spatial perception is vastly superior to prompt engineering.
- This work provides an elegant scheme for "how to extract action signals from videos", which can inspire future cross-disciplinary research between video understanding and embodied AI.

## Rating
- Novelty: ⭐⭐⭐⭐ While SoM is built on prior work, ToM is a natural extension; however, the framework design unifying both into cross-domain pre-training is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four major task categories including UI, robotics, VL understanding, and spatial reasoning, featuring comprehensive ablations and real-world robot validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, with a seamless narrative flow running from motivation through method to experiments.
- Value: ⭐⭐⭐⭐⭐ An open-source unified Agent foundation model by Microsoft, holding substantial practical value and potential subsequent impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](../../NeurIPS2025/robotics/benchmarking_egocentric_multimodal_goal_inference_for_assist.md)
- [\[ICLR 2026\] Embodied Navigation Foundation Model](../../ICLR2026/robotics/embodied_navigation_foundation_model.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](../../ICCV2025/robotics/adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)

</div>

<!-- RELATED:END -->
