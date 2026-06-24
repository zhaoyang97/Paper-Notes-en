---
title: >-
  [Paper Note] Geometry-Guided Camera Motion Understanding in VideoLLMs
description: >-
  [CVPR 2025][Interpretability][camera motion] Proposes a complete framework spanning benchmark construction, diagnosis, and injection. By extracting camera motion cues from a 3D foundation model (VGGT) and injecting them into the VideoLLM via structured prompting, training-free camera motion perception enhancement is achieved.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "camera motion"
  - "VideoLLM"
  - "3D foundation model"
  - "structured prompting"
  - "VGGT"
date: 2026-05-08
content_hash: 465bb473bd0a08c4
---

# Geometry-Guided Camera Motion Understanding in VideoLLMs

**Conference**: CVPR 2025  
**arXiv**: [2603.13119](https://arxiv.org/abs/2603.13119)  
**Code**: To be confirmed  
**Area**: Interpretability  
**Keywords**: camera motion, VideoLLM, 3D foundation model, structured prompting, VGGT

## TL;DR

Proposes a complete framework spanning benchmark construction, diagnosis, and injection. By extracting camera motion cues from a 3D foundation model (VGGT) and injecting them into the VideoLLM via structured prompting, training-free camera motion perception enhancement is achieved.

## Background & Motivation

**Background**: VideoLLMs perform well on high-level video semantics (objects, actions, narratives), but their fine-grained recognition of camera motion (pan, tilt, dolly, etc.) is severely insufficient.

**Limitations of Prior Work**: Camera motion is a spatio-temporal geometric signal that cannot be localized to a single frame and is easily interfered with by object motion, cuts, and motion blur. Furthermore, large-scale video datasets lack explicit supervision for camera motion.

**Key Challenge**: The vision encoder of VideoLLMs performs token compression in deep layers to optimize semantic alignment, which weakens motion-sensitive cues, whereas camera motion understanding requires precise geometric information.

**Goal**: To equip VideoLLMs with reliable, fine-grained camera motion recognition capabilities and enable them to generate camera-aware video descriptions.

**Key Insight**: Instead of modifying the VideoLLM weights, a frozen 3D foundation model is leveraged to extract geometric camera cues. A lightweight classifier then predicts motion primitives, which are injected via structured prompts.

**Core Idea**: Compensate for the missing camera motion representation in VideoLLMs using geometric priors from 3DFMs, achieving zero-training camera motion enhancement through a plug-and-play structured prompting mechanism.

## Method

### Overall Architecture

1. Videos are segmented by shots, and each shot is divided into 1-second non-overlapping segments.
2. The frozen VGGT (1.2B parameters) extracts a camera token $\mathbf{c}_t \in \mathbb{R}^{2048}$ for each frame.
3. A lightweight Transformer classifier predicts constrained multi-label motion primitives.
4. The prediction results are serialized into structured prompts and injected into the VideoLLM during inference.

### Key Designs

**1. Construction of CameraMotionDataset and CameraMotionVQA**
- **Function**: Build a dataset of 12,274 1-second segments from MultiCamVideo (rendered with Unreal Engine 5, featuring precise extrinsic camera parameters), containing 15 atomic camera motion labels.
- **Mechanism**: Compute yaw/pitch/roll changes and translation changes from frame-by-frame camera extrinsics, then map them to motion primitives via threshold matching. An incompatibility matrix $\mathbf{M}$ is defined to constrain incompatible combinations (e.g., pan-left and pan-right).
- **Design Motivation**: Synthetic data provides deterministic annotations (with 93% human-verified consistency), avoiding the subjectivity of real-world annotations, while balanced sampling addresses class imbalance.

**2. Constrained Regularized Motion Classifier**
- **Function**: Linearly project camera tokens to 512 dimensions, add positional encodings and a [CLS] token, and predict $K=15$ class logits through a 4-layer Transformer encoder.
- **Mechanism**: BCE loss + two regularization terms:
    - Incompatibility loss $\mathcal{L}_{\text{inc}} = \sum_{i<j} \mathbf{M}_{ij} p_i p_j$ (penalizing co-occurrence of mutually exclusive primitives)
    - Cardinality loss $\mathcal{L}_{\text{card}}$ (constraining the number of activated primitives between 1 and 3)
- **Design Motivation**: Physical constraints ensure the semantic plausibility of the predicted results, preventing anomalies like predicting both pan-left and pan-right simultaneously.

**3. Vision Encoder Probing Experiment**
- **Function**: Train a Q-Former-style probe on the intermediate layer features of the frozen vision encoder in Qwen2.5-VL to diagnose the retention of camera motion information.
- **Mechanism**: Extract features at full-attention layers of the ViT (layers 7/15/23/31), revealing that performance is best in shallow layers and gradually degenerates in deeper layers.
- **Design Motivation**: Confirm that the vision encoder of VideoLLMs loses camera motion information in deep layers, providing a theoretical foundation for injecting external geometric cues.

**4. VGGT-Q-Former Distillation**
- **Function**: Distill the camera-aware capabilities of VGGT using a lightweight Q-Former student model to reduce inference overhead.
- **Mechanism**: Alternate between local-frame attention and global attention, utilizing a 3-stage progressive training scheme (classifier $\rightarrow$ distillation regression $\rightarrow$ joint fine-tuning).
- **Design Motivation**: VGGT has a heavy inference footprint (1.2B parameters). Post-distillation achieves a 5.3$\times$ throughput increase and uses only 39% of the peak memory.

### Loss & Training

- Classifier Training: $\mathcal{L} = \mathcal{L}_{\text{bce}} + \lambda_{\text{inc}} \mathcal{L}_{\text{inc}} + \lambda_{\text{card}} \mathcal{L}_{\text{card}}$, where $\lambda_{\text{inc}}=\lambda_{\text{card}}=1.0$
- Distillation Regression: $\mathcal{L}_{\text{reg}} = \sum_{t=1}^{T} |\tilde{\mathbf{c}}_t - \mathbf{c}'_t|_2^2$
- At inference time, threshold $\tau=0.5$, and post-processing enforces mutual exclusion constraints and normalization.

## Key Experimental Results

### Main Results

| Method | Inst. Acc↑ | Macro-F1↑ | Weighted-F1↑ |
|---|---|---|---|
| VGGT w. constraints | **0.738** | **0.87** | **0.92** |
| VGGT w/o. constraints | 0.572 | 0.79 | 0.84 |
| VGGT-Q-Former (distilled) | 0.638 | 0.83 | 0.87 |
| Q-Former probing | 0.450 | 0.69 | 0.74 |
| Off-the-shelf VideoLLMs | ~25% VQA acc (close to random guess) | - | - |

### Ablation Study

| Pipeline | Params(M) | Peak Mem(MB) | Throughput(samples/s) |
|---|---|---|---|
| VGGT classifier | 9.47 | 23649 | 4.39 |
| VGGT-Q-Former | 9.15 | 9203 | 23.36 |
| Q-Former probing | 15.18 | 9232 | 25.12 |

Distillation sacrifices 8.13% in accuracy but gains 5.3$\times$ throughput and 61% memory savings.

### Key Findings

1. **Blind Spot of VideoLLMs in Camera Motion**: Existing off-the-shelf VideoLLMs perform close to random guess (~25%) on CameraMotionVQA, and even models fine-tuned on CameraBench perform worse than their base models.
2. **Importance of Constrained Modeling**: Removing the mutual exclusion constraints drops the instance accuracy from 73.8% to 57.2%.
3. **Loss of Camera Motion Information with Depth**: Probing experiments show that shallow full-attention blocks perform best, and performance decays in deeper layers, supporting the hypothesis that token compression attenuates motion-sensitive cues.
4. **Structured Prompting Significantly Enhances Description Quality**: After injecting the motion header, the VideoLLM generates explicit motion directions (e.g., pan-left/right), framing composition descriptions, and spatio-temporal reasoning.

## Highlights & Insights

- A complete "benchmark-diagnosis-injection" closed loop: it not only uncovers the limitations of prior work but also proposes a practical plug-and-play solution.
- Reveals the inner mechanism of VideoLLMs losing camera motion information via vision encoder probing.
- The training-free structured prompting strategy is highly practical and generalizable.
- The technical pipeline combining synthetic data, constrained modeling, and distillation provides valuable insights.

## Limitations & Future Work

- Domain gap exists between synthetic data and real-world videos.
- Focuses only on camera extrinsic changes, not covering intrinsic variations (e.g., zooming).
- Only explores a single 3DFM backbone (VGGT).
- Static primitives might be out-of-distribution samples for VGGT, requiring specialized handling.
- Lacks quantitative evaluation for the improvement of description quality brought by structured prompting.

## Related Work & Insights

- CameraBench defines a classification taxonomy for camera motion but lacks geometric annotations; this work complements it with geometrically deterministic labels.
- Unlike SpatialVID, which provides frame-by-frame depth and camera poses, this work focuses on 1-second segment motion primitive recognition.
- Insights: Utilizing 3DFMs as an external source of geometric priors can enhance the spatial understanding capability of various VideoLLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The pipeline design of 3DFM cue extraction + constrained classification + structured injection is novel, and the probing analysis is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple evaluation dimensions, including benchmark evaluation, probing diagnosis, distillation efficiency, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The logical flow is clear, progressing systematically from problem discovery and mechanical analysis to solution design.
- **Value**: ⭐⭐⭐⭐ Highly practical; the plug-and-play solution can be directly applied to various VideoLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2025\] KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)
- [\[NeurIPS 2025\] MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition](../../NeurIPS2025/interpretability/mopformer_motion-primitive_transformer_for_wearable-sensor_activity_recognition.md)
- [\[ACL 2026\] Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs](../../ACL2026/interpretability/flattery_in_motion_benchmarking_and_analyzing_sycophancy_in_video-llms.md)
- [\[ACL 2025\] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](../../ACL2025/interpretability/probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)

</div>

<!-- RELATED:END -->
