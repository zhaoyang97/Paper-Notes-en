---
title: >-
  [Paper Note] Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models
description: >-
  [CVPR 2025][VLM Reasoning][Multimodal Large Language Models] This paper proposes Coarse Correspondences, a lightweight, training-free visual prompting method. By overlaying coarse-grained instance correspondence markers obtained from object tracking onto image frames, it significantly enhances the spatial-temporal reasoning capabilities of MLLMs, achieving improvements of +20.5% on ScanQA, +9.7% on OpenEQA, +6.0% on EgoSchema, and +11% on R2R navigation.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Multimodal Large Language Models"
  - "Spatial-Temporal Reasoning"
  - "Visual Prompting"
  - "Object Tracking"
  - "3D Scene Understanding"
date: 2026-05-08
content_hash: c78777153c206437
---

# Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models

**Conference**: CVPR 2025  
**arXiv**: [2408.00754](https://arxiv.org/abs/2408.00754)  
**Code**: [GitHub](https://coarse-correspondence.github.io)  
**Area**: Video Understanding  
**Keywords**: Multimodal Large Language Models, Spatial-Temporal Reasoning, Visual Prompting, Object Tracking, 3D Scene Understanding

## TL;DR
This paper proposes Coarse Correspondences, a lightweight, training-free visual prompting method. By overlaying coarse-grained instance correspondence markers obtained from object tracking onto image frames, it significantly enhances the spatial-temporal reasoning capabilities of MLLMs, achieving improvements of +20.5% on ScanQA, +9.7% on OpenEQA, +6.0% on EgoSchema, and +11% on R2R navigation.

## Background & Motivation
Multimodal Large Language Models (MLLMs) excel at vision-language tasks but still underperform in 3D spatial understanding and long-video temporal reasoning, often performing only slightly better than text-only baselines. Existing solutions typically require specialized 3D architecture designs, task-specific fine-tuning, or foreign 3D data inputs (such as point clouds), which are costly and lack generalizability.

Key Challenge: General MLLMs inherently possess some potential for spatial-temporal reasoning, but lack cross-frame object correspondence information—when multiple image frames are input, the model struggles to determine which regions in different frames correspond to the same object.

Core Idea: Leveraging existing lightweight video tracking models to extract object-level cross-frame correspondences, and then directly visualizing these correspondences on the images through simple visual markers (such as numbered dots). This allows MLLMs to "see" cross-frame object associations, thereby substantially improving spatial-temporal reasoning without modifying model architectures or requiring task-specific fine-tuning.

## Method

### Overall Architecture
Coarse Correspondences consists of four steps: (1) tracking correspondences → (2) sparse frame sampling → (3) selecting salient correspondence instances → (4) visualizing correspondences. The processed, marked images are directly fed into the general MLLM for reasoning.

### Key Designs
1. **Tracking Correspondences**:

    - Function: Use existing video object tracking models (e.g., Tracking Anything, SAMv2) to extract category-agnostic instance segmentation masks and cross-frame IDs from high-frame-rate input image sequences.
    - Mechanism: For each frame $I_i$, obtain the instance segmentation $M_i$ ($H \times W$ matrix), where each pixel is labeled with its corresponding instance ID. The same object shares the same ID across different frames.
    - Design Motivation: Video tracking is a relatively lightweight operation, which is far more efficient than processing a large number of frames in MLLMs.

2. **Sparsify Frames & Select Correspondences**:

    - Function: Uniformly sample a small number of frames $m \ll n$ from the high-frame-rate tracking results, and then select the Top-K most salient object instances.
    - Mechanism: Saliency is sorted by two criteria: (1) cross-frame occurrence frequency $\mathcal{F}req(\text{ID}) = \sum_{i=s_1}^{s_m} \mathbf{1}_{\{\text{ID} \in M_i\}}$; (2) total area $\mathcal{A}rea(\text{ID})$. Objects occurring most frequently are prioritized.
    - Design Motivation: Ablation studies reveal that labeling all correspondences actually degrades performance (information overload), and keeping only a few salient instances yields the best results.

3. **Visualizing Correspondences**:

    - Function: Overlay fixed-size numbered markers (dots + numbers) on each image frame for selected instances. The marker position is at the centroid of the instance mask $(\\bar{x}_{ij}, \\bar{y}_{ij})$.
    - Mechanism: The centroid position is calculated by the weighted average of the mask pixel coordinates. The same object uses the identical numbered marker across different frames, allowing MLLMs to intuitively understand "which regions represent the same entity".
    - Design Motivation: Visual markers are a form of information delivery natively supported by MLLMs, requiring no modifications to any model components.

### Loss & Training
For closed-source models (GPT-4V/O), no training is needed at all; the images processed by Coarse Correspondences are used directly during inference. For open-source models (such as LLaVA), marked images can also be used during instruction tuning, and the same marked images are used at inference. Crucially, the performance gains can generalize to unseen datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (GPT-4O+CC) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ScanQA | CIDEr | 87.0 | 72.2 (GPT-4O) | +14.8 |
| ScanQA | BLEU-2 | 25.5 | 19.8 (GPT-4O) | +5.7 |
| OpenEQA (EM-EQA) | Accuracy | 59.1 | 49.4 (GPT-4O) | +9.7 |
| EgoSchema | Accuracy | 73.2 | 67.2 (GPT-4O) | +6.0 |
| R2R Navigation | Success Rate | 23.0 | 12.0 (GPT-4O) | +11.0 |

| Dataset | Metric | LLaVA+CC | LLaVA(Fine-tuned) | Gain |
|--------|------|------|----------|------|
| ScanQA | CIDEr | 74.2 | 67.3 | +6.9 |
| SQA3D (unseen) | - | +3.1 | - | Validation of generalization capability |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No correspondence visualization | baseline | Original images are used |
| Label all instances | Below baseline | Information overload harms reasoning |
| Segmentation contours only | Slightly below CC | Dense mask boundaries increase visual noise |
| Numbered markers only | Optimal | Concise visual markers perform best |
| K=3-5 (number of markers) | Optimal range | Too many or too few are suboptimal |

### Key Findings
- General MLLMs (like GPT-4O) achieve zero-shot performance surpassing many specialized models that use 3D point clouds or massive frames, using only 8 frames with CC markers.
- On EgoSchema, using 8 frames outperforms methods like LongViViT (using 256 frames) and MC-ViT-L (using 128+ frames).
- Incorporating CC during open-source model (LLaVA) training yields a +6.9% improvement and generalizes to the unseen dataset SQA3D (+3.1%).
- The success rate in navigation tasks doubles (12% → 23%), validating the value of spatial-temporal reasoning for embodied tasks.

## Highlights & Insights
- **Simple yet Effective**: It requires no training, no architectural modifications, and no 3D data, dramatically enhancing 3D and temporal reasoning using only visual markers on 2D images.
- **Few-Frame Efficiency**: By tracking across dense frames first and then sparse-sampling, it maintains or even exceeds the performance of dense-frame methods using very few frames.
- **Cross-Task Generalizability**: The same framework yields significant improvements across three completely distinct tasks: 3D QA, Video QA, and navigation.
- **Paradigm Insights into Visual Prompting**: Correspondence essentially helps MLLMs establish cross-frame "referring" capabilities, a paradigm that can migrate to other scenarios requiring cross-image/cross-view understanding.
- **Echoes the Classical Role of Visual Correspondence in 3D Reconstruction**: The paper highlights that the value of visual correspondence in semantic tasks has been underappreciated in the deep learning era.

## Limitations & Future Work
- Dependence on Tracking Model Quality: If the tracker fails in scenes with rapid motion or severe occlusion, the effectiveness of CC will be compromised.
- Utilizing only coarse instance-level correspondences without leveraging finer-grained (pixel-level, semantic-part-level) correspondence information.
- The design of marker visualization (size, color, shape) may require tuning for different models.
- The interaction with the model's internal attention mechanism remains unexplored—effects could potentially be enhanced if correspondence is directly injected into the model rather than visually overlayed.
- Navigation experiments were evaluated on only 100 samples due to computational cost constraints.

## Camera Motion Invariance Analysis

This is a highly intriguing diagnostic experiment in the paper. The authors construct a dedicated test suite: 10 scenes captured from left to right, with 5 spatial relationship questions per scene, each tested 20 times, totaling 1,000 trials. Then, the frame sequence is reversed (simulating a right-to-left capture) and tested for another 1,000 trials.

Key Findings: Without CC, the accuracy of the left-to-right sequence is 58%, but plummets to 50.4% upon reversal, indicating that the model heavily relies on frame order to "guess" spatial relationships. With CC, both forward and backward accuracies reach 71.2%, and the harmonic mean increases from 53.9% to 71.2% (+17.3%). This demonstrates that correspondence markers enable the model to truly establish spatial understanding rather than memorize frame-order patterns.

## Related Work & Insights
- **vs 3D-LLM / ScanRefer+MCAN**: These methods require specialized 3D architectures and data; CC, as a plug-and-play solution, outperforms them.
- **vs LLoVi / LangRepo (Socratic Methods)**: These methods convert video to text before using LLMs for reasoning, losing visual details; CC enhances MLLMs directly at the visual level.
- **vs VideoAgent**: Agent methods require multi-step reasoning and multiple model calls; CC needs only a single forward pass.
- **vs Set-of-Mark Prompting**: SoM annotates on a single image, while CC extends this to the dimension of cross-frame correspondence.
- Insight: When we assume the model "understands" space, it might actually just be exploiting shortcuts like frame order; CC provides a means of verification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The approach is extremely simple yet deeply insightful, elegantly chaining tracking, marking, and MLLM reasoning. It is the first to demonstrate that coarse-grained correspondence is sufficient to greatly boost spatial-temporal reasoning in MLLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of 3D QA, Video QA, navigation, open-source/closed-source models, training/inference, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, simple and easy-to-understand method, and compelling experimental demonstrations.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value; it can be plugged and played to enhance any MLLM supporting multi-image input, opening up a new direction for visual prompting to enhance spatial-temporal reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](../../AAAI2026/vlm_reasoning/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[ICML 2026\] Temporal-Aware Reasoning Optimization for Video Temporal Grounding](../../ICML2026/vlm_reasoning/temporal-aware_reasoning_optimization_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
