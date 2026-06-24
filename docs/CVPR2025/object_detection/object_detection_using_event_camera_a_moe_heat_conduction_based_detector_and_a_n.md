---
title: >-
  [Paper Note] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset
description: >-
  [CVPR 2025][Object Detection][Event camera] This paper proposes MvHeat-DET, which models visual features as a 2D heat diffusion process and dynamically routes among three frequency domain transforms (DFT/DCT/Haar) using MoE. Combined with IoU-aware query selection, it performs object detection on event streams. Additionally, the paper releases a high-definition event camera detection dataset, EvDET200K (10,054 videos / 200K bboxes / 10 classes).
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Event camera"
  - "Heat conduction"
  - "MoE"
  - "DETR"
  - "benchmark"
date: 2026-05-08
content_hash: 567d384fc218018b
---

# Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset

**Conference**: CVPR 2025  
**arXiv**: [2412.06647](https://arxiv.org/abs/2412.06647)  
**Code**: [https://github.com/Event-AHU/OpenEvDET](https://github.com/Event-AHU/OpenEvDET)  
**Area**: Object Detection / Event Camera / Vision Backbone  
**Keywords**: Event camera, Heat conduction, MoE, DETR, benchmark

## TL;DR
This paper proposes MvHeat-DET, which models visual features as a 2D heat diffusion process and dynamically routes among three frequency domain transforms (DFT/DCT/Haar) using MoE. Combined with IoU-aware query selection, it performs object detection on event streams. Additionally, the paper releases a high-definition event camera detection dataset, EvDET200K (10,054 videos / 200K bboxes / 10 classes).

## Background & Motivation
1. **Background**: Event cameras (DVS), with high dynamic range, high temporal resolution, and low power consumption, are more robust than RGB cameras in low-light/fast-motion scenarios. Event detection is a critical downstream task.
2. **Limitations of Prior Work**:
    - CNN-based event detectors (RED, Faster R-CNN) suffer from limited receptive fields.
    - Transformer-based detectors (RVT, SAST) have $\mathcal{O}(N^2)$ complexity and poor interpretability.
    - SNN-based detectors (SpikeYOLO) are energy-efficient but lag behind ANNs in accuracy.
3. **Key Challenge**: The lack of an event detection backbone that simultaneously addresses accuracy, efficiency, and interpretability.
4. **Goal**:
    - Design a mathematically interpretable event detection backbone with complexity lower than $\mathcal{O}(N^2)$.
    - Address the limitations of low resolution, few categories, and small scale in existing event detection datasets.
5. **Key Insight**: The 2D heat conduction equation in physics, $\partial_t u = k(u_{xx}+u_{yy})$, can be interpreted as "feature diffusion in space," and its frequency domain solution $\hat u(t)=\hat f \cdot e^{-k(v_x^2+v_y^2)t}$ is closed-form, with a complexity of only $\mathcal{O}(N^{1.5})$.
6. **Core Idea**: Upgrade the Heat Conduction Operator (HCO) to MoE-HCO, which dynamically selects among three "expert transforms" (DFT, DCT, and Haar) for different event scenarios (dense/sparse). This aligns with the sparsity of events while retaining global information exchange capabilities.

## Method

### Overall Architecture
Input event stream $\rightarrow$ stacked into event frames $\rightarrow$ Stem network generates patch embeddings $\rightarrow$ 4 stages of **MoE-HCO** blocks (each stage containing several MHCO Layers + downsampling) $\rightarrow$ IoU-based Query Selection selects top-K tokens $\rightarrow$ DETR-style decoding head outputs detection boxes. The overall architecture maintains a ViT-style design, replacing self-attention with MHCO.

### Key Designs

1. **MHCO: MoE Heat Conduction Operator**

    - **Function**: Simulates feature transmission between patches using closed-form frequency domain heat diffusion.
    - **Mechanism**: First, depthwise convolution is used to expand the single-channel temperature map to multi-channel to obtain $U_0$. Then, the forward transform is performed in one of three expert branches (DFT, DCT, Haar), multiplied by the diffusion kernel $e^{-k(v_x^2+v_y^2)t}$, and finally mapped back to the spatial domain via inverse transform: $U_t = \mathcal{F}^{-1}\big(\mathcal{F}(U_0)\,e^{-k(v_x^2+v_y^2)t}\big)$. Among these:
        - **DFT/IDFT**: Processes global interaction across patches (suitable for dense and highly dynamic scenarios).
        - **DCT/IDCT** and **Haar/IHaar**: Naturally satisfy the Neumann boundary condition (derivative is 0), suitable for independent patch-level processing (perfect for sparse, low-motion event scenarios since there is no info to exchange between patches when events are mostly blank).
    - **Design Motivation**: A single transform cannot easily accommodate both sparse and dense events. MoE allows the model to adaptively route based on the scenario. The closed-form frequency domain solution reduces complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N^{1.5})$.

2. **Learnable Heat Diffusivity $k$ (Frequency Embeddings)**

    - **Function**: Ensures "semantically critical regions receive more heat", making the heat diffusion content-adaptive.
    - **Mechanism**: Randomly initialize Frequency Embeddings (FEs) with the same shape as the frequency domain feature $\hat x$, predict $k$ via a linear layer, and multiply it with the diffusion kernel. $t$ is kept as a constant.
    - **Design Motivation**: Physically, $k$ reflects the thermal conductivity of a material. Visually, different image regions should have distinct "diffusion rates," so that the features of key regions become more prominent after multiple diffusion steps.

3. **MoE Routing (Policy Network + Gumbel-Softmax)**

    - **Function**: Performs a differentiable hard selection among the three experts (DFT / DCT / Haar).
    - **Mechanism**: A lightweight policy network scores the current feature map, and Gumbel-Softmax sampling is used to select a single expert branch, avoiding the extra computational overhead of running all three branches during inference.
    - **Design Motivation**: Since spatial distributions of event scenarios vary drastically (some have only a single moving point while others are highly dense), hard routing is more efficient than soft weighting and aligns better with the semantics of "receptive field shifting based on the scene."

4. **IQS: IoU-based Query Selection**

    - **Function**: Resolves the bias of "high classification score but poor localization" in DETR query selection.
    - **Mechanism**: Integrates IoU into the classification loss: $\mathcal{L}(y,\hat y) = \mathcal{L}_{bbox}(b,\hat b) + \mathcal{L}_{cls}(IoU, c, \hat c)$, implicitly encoding localization quality into the classification score, and then selecting queries based on the top-K classification scores.
    - **Design Motivation**: The original DETR query selection mistakenly selects boxes with high scores but low IoUs. IQS aligns the selection better with the final detection objective.

### Loss & Training
- Total loss = bbox L1 + GIoU + IoU-aware classification (Equation 10).
- Standard DETR training pipeline with Hungarian matching.
- Gumbel-Softmax temperature is gradually annealed to transition the MoE routing from soft to hard.

## Key Experimental Results

### Main Results
**N-Caltech101** (Classic event classification $\rightarrow$ detection benchmark):

| Method | Input | mAP |
|------|------|-----|
| YOLE | Event frames | 39.8 |
| EAS-SNN | Event points | 53.8 |
| Jeziorek et al. | Event frames | 53.4 |
| **MvHeat-DET (Ours)** | Event frames | **55.7** |

**EvDET200K** (The new dataset proposed in this paper):

| Method | mAP@50:95 | mAP@50 | Params | FLOPs | FPS |
|------|-----------|--------|--------|-------|-----|
| Faster R-CNN | 46.0 | 73.3 | 40.9M | 71.2G | 23 |
| Swin-T | 49.0 | 78.4 | 160M | 1043G | 26 |
| DetectoRS | 49.1 | 78.8 | 123M | 117G | 32 |
| YOLOv10-B | 44.1 | 77.9 | 19.1M | 92G | 30 |
| RVT | 40.7 | 73.1 | 9.9M | 8.4G | 88 |
| SpikeYOLO | 41.2 | 74.8 | 68.8M | 78.1G | 77 |
| S5-ViT | 42.9 | 76.3 | 18.2M | 5.6G | 84 |
| **MvHeat-DET** | **Leading** | **Leading** | Medium | Low | High |

### EvDET200K Dataset Statistics
- 10,054 videos (2-5s), 200,260 bboxes, 10 classes (people / car / bicycle / electric bicycle / basketball / ping pong / goose / cat / bird / UAV).
- Authentically captured by Prophesee EVK4-HD, with a resolution of 1280×720 (much higher than Gen1's 304×240).
- Partitioned in 6:1:3 ratio, containing 2,949 dense scene videos and 51% small objects.
- Covers 6 challenging factors: sunny, rainy, daytime, nighttime, multi-view, and multi-motion.

### Key Findings
- The three transform experts are complementary: sparse scenes favor DCT/Haar, while dense scenes favor DFT; routing outputs are highly non-uniform.
- Replacing the original DETR query selection with IQS significantly improves mAP@75, indicating that localization quality benefits the most.
- Compared block-by-block with vHeat (which uses only DCT), MoE-HCO performs more stably on EvDET200K, validating that "a single transform is insufficient."

## Highlights & Insights
- **Physical priors as backbones**: Using the closed-form frequency domain solution of PDEs like heat conduction as a token mixer is a natural extension of the vHeat approach. This work further demonstrates that "different transforms correspond to different boundary conditions / signal assumptions," combining them using MoE. This is the first time the "frequency domain operator selection" problem has been introduced into MoE.
- **Sparsity of event camera data serves as a natural MoE routing signal**: Scenarios vary much more significantly between dense and sparse in event streams compared to RGB images, leading to larger gains from MoE. This concept can be transferred to any sensing modal with highly varying input distributions (e.g., LiDAR, point clouds, medical ultrasound).
- **A new high-definition event detection benchmark**: It fills the gap in resolution and category diversity left by Gen1/1Mpx, serving as a standard evaluation platform for subsequent event detection algorithms.
- **IQS is a plug-and-play improvement for DETR**, which can be independently applied to RGB DETR models.

## Limitations & Future Work
- The three experts are handcrafted, and whether the expert set can be automatically learned based on the dataset remains unexplored.
- $t$ is fixed as a constant. Making $t$ learnable or position-dependent might further improve the representation capacity.
- Evaluated only on event detection; the scalability of MoE-HCO has not been verified on other event-based tasks like event tracking, recognition, or flow estimation.
- EvDET200K is still annotated manually at 5 frames per video, and the annotation density could be further improved.
- Future improvement: Combine MHCO with SNNs to form a hybrid encoder of "frequency-domain heat diffusion + spike firing", balancing energy efficiency and accuracy.

## Related Work & Insights
- **vs vHeat (NeurIPS 2024)**: vHeat uses only DCT to simulate heat diffusion. This paper argues that the Neumann boundary condition assumed by DCT might not hold for all patches, thus introducing DFT and Haar and using MoE to adaptively combine them. The advantage is increased generality, while the disadvantage is a more complex architecture.
- **vs RVT / SAST**: They utilize transformers to process events, resulting in high complexity. The $\mathcal{O}(N^{1.5})$ frequency-domain operator in this work offers significant efficiency advantages when dealing with large numbers of tokens.
- **vs SpikeYOLO**: SNNs are energy-efficient but lack accuracy. This paper adopts the ANN paradigm but retains efficiency with frequency-domain operators.
- Can serve as a baseline: Any subsequent "event + DETR" work can directly compare against MvHeat-DET.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of routing frequency domain transforms in MoE is introduced for the first time, and the physics-vision alignment is clearly explained.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features a self-built benchmark + 15 baseline evaluations, providing a highly comprehensive assessment.
- Writing Quality: ⭐⭐⭐ Complete mathematical derivations, though sections are slightly verbose.
- Value: ⭐⭐⭐⭐ Outstanding contribution with the new dataset (as HD event detection datasets are rare), and the method can be directly borrowed by future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [\[ECCV 2024\] Plain-Det: A Plain Multi-Dataset Object Detector](../../ECCV2024/object_detection/plain-det_a_plain_multi-dataset_object_detector.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](../../NeurIPS2025/object_detection/burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)

</div>

<!-- RELATED:END -->
