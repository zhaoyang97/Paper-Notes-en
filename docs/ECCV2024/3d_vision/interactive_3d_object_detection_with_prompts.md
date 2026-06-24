---
title: >-
  [Paper Note] Interactive 3D Object Detection with Prompts
description: >-
  [ECCV 2024][3D Vision][3D Object Detection] This work proposes a multi-modal interactive 3D object detection framework named "Prompt in 2D, Detect in 3D" + "Detect in 3D, Refine in 3D". By bridging the 2D-3D complexity gap with simple 2D interaction prompts (clicks or bounding boxes) and supporting iterative refinement, it significantly reduces 3D annotation costs. Its effectiveness and outstanding open-set adaptation capabilities are validated on nuScenes.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Object Detection"
  - "Interactive Labeling"
  - "2D Prompts"
  - "Multi-modal Fusion"
  - "Open-set Detection"
date: 2026-05-08
content_hash: ad953583e6226d66
---

# Interactive 3D Object Detection with Prompts

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Object Detection, Interactive Labeling, 2D Prompts, Multi-modal Fusion, Open-set Detection

## TL;DR

This work proposes a multi-modal interactive 3D object detection framework named "Prompt in 2D, Detect in 3D" + "Detect in 3D, Refine in 3D". By bridging the 2D-3D complexity gap with simple 2D interaction prompts (clicks or bounding boxes) and supporting iterative refinement, it significantly reduces 3D annotation costs. Its effectiveness and outstanding open-set adaptation capabilities are validated on nuScenes.

## Background & Motivation

**Background**: 3D object detection is a core task in autonomous driving and robotics. With the development of LiDAR sensors and multi-modal perception, the accuracy of 3D detection models has steadily improved. However, training high-precision models heavily relies on large-scale, high-quality 3D annotated data, including precise 3D bounding boxes (position, size, and orientation).

**Limitations of Prior Work**: 3D object annotation is an extremely time-consuming and labor-intensive process. Labeling a 3D bounding box requires precisely calibrating 7 degrees of freedom (x, y, z, l, w, h, θ) in the point cloud, which typically requires expert annotators to repeatedly adjust inside 3D visualization tools. Compared to 2D image annotation (where drawing a box requires only about 4 clicks), the cost of 3D annotation is higher by an order of magnitude, severely hindering the development and application of 3D detection technologies.

**Key Challenge**: High-quality 3D detection requires massive amounts of precise 3D annotations, yet the high cost of 3D labeling limits data scale. Existing automatic labeling methods (such as generating pseudo-labels using pretrained models) lack sufficient accuracy, while fully manual labeling is hard to scale. Consequently, a solution is needed that sits between fully automatic and fully manual methods—achieving the highest quality of 3D annotations with minimal human interaction.

**Goal**: (1) How can simple 2D interaction prompts (clicks/boxes on images or bird's-eye-view maps) be utilized to generate 3D detection results? (2) How can the dimensional gap between 2D prompts and 3D detection be bridged? (3) How can the framework support annotators in iteratively refining the initial results?

**Key Insight**: The authors note two key observations: first, human interaction on 2D images is far simpler and more intuitive than on 3D point clouds; second, while 2D prompts lack depth information, they contain rich semantic and spatial cues that can recover 3D info through multi-modal fusion. Inspired by the success of SAM (Segment Anything Model) in the 2D domain, the authors introduce the interactive prompting paradigm to 3D detection.

**Core Idea**: Drive 3D object detection using simple prompts (clicking/boxing) in 2D space, while supporting iterative refinement in 3D space, thus achieving high-quality 3D annotation at a low cost.

## Method

### Overall Architecture

The framework consists of two stages: (1) **Prompt in 2D, Detect in 3D**—the user provides 2D prompts (clicks or bounding boxes) on the camera image or the Bird's-Eye View (BEV), and the system transforms these 2D prompts into 3D detection results via multi-modal fusion. (2) **Detect in 3D, Refine in 3D**—the user can interactively refine the detection results in 3D space (e.g., dragging vertices to fine-tune 3D boxes). The entire process supports iterative cycles: detection $\rightarrow$ refinement $\rightarrow$ re-refinement, until the annotator is satisfied.

### Key Designs

1. **Multi-modal Prompt Encoder**:

    - **Function**: Unified encoding of different forms of 2D prompts into feature vectors.
    - **Mechanism**: Support multiple prompt modalities—clicks on camera images, 2D bounding boxes (boxes), and clicks/boxes on the bird's-eye view. For click prompts, sinusoidal positional encodings map 2D coordinates into high-dimensional features. For box prompts, the center and size of the box are encoded. Prompts from different modalities (camera view vs. BEV view) are projected into a unified prompt feature space through their respective projection layers. The prompt encoder also contains a type embedding to distinguish between clicks and boxes.
    - **Design Motivation**: Supporting multiple prompt types allows the system to adapt to different annotation scenarios—labeling on camera images is suitable for occluded objects (since appearance is visible), while labeling on BEV is suitable for determining precise coordinates. Unified encoding allows subsequent modules to process prompts seamlessly regardless of their source.

2. **Prompt-guided 3D Detection Head**:

    - **Function**: Locating and regressing targets in 3D space based on 2D prompt features.
    - **Mechanism**: The 2D prompt features are used as queries to perform cross-attention with 3D voxel/point cloud features. First, 2D prompts are back-projected into 3D space using camera intrinsic and extrinsic parameters to determine candidate 3D regions. Then, point cloud features are aggregated within these candidate regions and fused with prompt features, which are decoded into 3D bounding box parameters (center coordinates, size, orientation) via a Transformer decoder. A key technical detail is using deformable attention for efficient feature sampling in 3D space to avoid the high computational cost of global attention.
    - **Design Motivation**: Since 2D prompts naturally lack depth information, it must be recovered from 3D features (point clouds/voxels). The cross-attention mechanism allows 2D prompts to "query" the 3D space and identify the 3D regions that best match their semantics. Back-projection provides a rough spatial prior, upon which deformable attention performs fine-grained feature sampling.

3. **Iterative 3D Refinement**:

    - **Function**: Allowing annotators to progressively refine initial 3D detection results.
    - **Mechanism**: The current 3D detection results (which might be imprecise) are encoded as "3D refinement prompts" containing the center, size, and orientation of the current box. These 3D refinement prompts, along with the original 2D prompts, are fed into the detection head for a new round of prediction. Annotators can fine-tune the detection results in a 3D visualization interface (e.g., shifting the box position or adjusting the box size), and the system takes the modified box as a new refinement prompt for iterative loops until satisfied. The step size of modifications decreases in each round, usually requiring only 2–3 rounds to reach high-quality annotations.
    - **Design Motivation**: Because 2D prompts inherently lack depth precision, the initially detected 3D box may contain errors in the depth direction. Iterative refinement allows human annotators to provide corrective signals in 3D space, gradually aligning the detection results to a satisfactory level. This human-in-the-loop collaboration balances efficiency (primarily automated detection) and accuracy (supplementary human fine-tuning).

### Loss & Training

Standard 3D detection losses are utilized during training: (1) classification loss $L_{cls}$ (focal loss); (2) 3D bounding box regression loss $L_{reg}$ (L1 loss + IoU loss), which constrain center offset, size, and orientation, respectively; (3) computing and accumulating losses at each refinement stage to guide the model to learn to employ refinement prompts for improved predictions. Training data is generated by simulating user prompts—sampling 2D prompts with added noise from the ground truth (simulating imprecise clicks/boxes by users).

## Key Experimental Results

### Main Results

| Prompt Type | mAP | NDS | Compared to Fully Supervised | Description |
|---------|-----|-----|-----------|------|
| BEV Center Click | 48.3 | 56.2 | -8.5 mAP | Simplest prompt, requiring only a single click |
| Camera Image Box | 51.7 | 59.1 | -5.1 mAP | 2D box provides more spatial information |
| BEV Box | 53.9 | 60.8 | -2.9 mAP | BEV box contains the richest information |
| BEV Box + 1-round Refinement | 55.2 | 62.0 | -1.6 mAP | Significant improvement achieved with just one round of refinement |
| BEV Box + 2-round Refinement | 56.1 | 62.8 | -0.7 mAP | Close to the fully supervised performance level |
| Fully Supervised (CenterPoint) | 56.8 | 63.5 | - | Baseline upper bound |

### Ablation Study

| Configuration | mAP | Description |
|------|-----|------|
| Full model (BEV Box) | 53.9 | Full model |
| w/o Cross-attention | 48.1 | Removes prompt-to-3D feature interaction, drops by 5.8 |
| w/o Back-projection Prior | 50.6 | Does not utilize camera parameters to narrow search range |
| w/o Deformable attention | 51.2 | Switched to global attention, slight decline |
| w/o Iterative Refinement Training | 52.7 | Trained only on single-step detection |

### Key Findings

- Cross-attention is the most critical module (contributing 5.8 mAP), indicating that the interaction and fusion of 2D prompts with 3D features is the core of the method's success.
- Richer prompt information leads to higher detection accuracy: BEV Box > Camera Box > BEV Click, which aligns with intuition.
- Iterative refinement improves mAP by approximately 1.0 on average per round with diminishing returns—2 rounds of refinement already yield performance very close to the fully supervised baseline.
- **Outstanding open-set capacity**: On categories unseen during training, the method can still produce reasonable 3D boxes because prompt-driven detection does not rely on category priors. The mAP is only about 3–5% lower than that of seen categories.

## Highlights & Insights

- **The 2D $\rightarrow$ 3D interaction paradigm revolutionizes the 3D annotation pipeline**: Traditional 3D annotation requires annotators to operate 7 degrees of freedom in 3D space. This method offloads most of the workload to the model, requiring annotators to perform only simple 2D actions (drawing boxes/clicking), which increases efficiency multi-fold. This paradigm can be directly integrated into 3D annotation pipelines.
- **Natural emergence of open-set capabilities**: Since the model relies on prompts rather than category priors for detection, it naturally possesses open-set capabilities—a property that traditional fully supervised detectors lack. This means the method can be used to annotate any novel category without retraining.
- **Human-in-the-loop collaborative design through iterative refinement**: Instead of striving for perfect one-step detection, it fosters cooperation between humans and the model to iteratively improve results. This design philosophy is highly practical for annotation tool development.

## Limitations & Future Work

- The method is validated only on the nuScenes dataset, lacking experiments on other mainstream 3D datasets like Waymo or KITTI.
- It relies on accurate calibration (intrinsic and extrinsic parameters) of cameras and LiDAR; calibration errors directly affect the accuracy of the 2D $\rightarrow$ 3D back-projection.
- Currently, natural language prompts (e.g., "detect the truck ahead") are not supported. Incorporating language prompts could further enhance the flexibility and open-set capabilities of annotation.
- The interactive interface design and user experience (UX) for iterative refinement are not detailed in the paper, which may present engineering challenges during real-world deployment.
- For highly occluded or overlapping objects, 2D prompts may fail to provide sufficient discriminative information.

## Related Work & Insights

- **vs. SAM (Segment Anything Model)**: SAM realized the interactive prompting paradigm in the 2D segmentation domain. This work extends a similar philosophy to 3D detection. The core challenge is the 2D $\rightarrow$ 3D dimension lifting, which SAM does not need to handle.
- **vs. PointPrompt / 3D-BoNet**: These methods perform prompt-based segmentation/detection in 3D space, requiring 3D prompt inputs. The innovation of this work lies in accepting 2D prompts, which significantly reduces the difficulty of user interaction.
- **vs. WeakSup3D**: Weakly supervised 3D detection uses 2D boxes as supervision signals but does not support interactive refinement. This method supports both efficient initial detection and iterative improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework concept of driving 3D detection with 2D prompts is highly novel, though some technical components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are conducted only on nuScenes; dataset coverage is insufficient.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the framework description is well-structured.
- Value: ⭐⭐⭐⭐⭐ Direct industrial application value for improving 3D annotation efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](../../CVPR2025/3d_vision/sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](../../AAAI2026/3d_vision/multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[ECCV 2024\] TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection](tcc-det_temporarily_consistent_cues_for_weakly-supervised_3d_detection.md)

</div>

<!-- RELATED:END -->
