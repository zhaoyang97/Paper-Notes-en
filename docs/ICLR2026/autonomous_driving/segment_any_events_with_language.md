---
title: >-
  [Paper Note] SEAL: Segment Any Events with Language
description: >-
  [ICLR 2026][Autonomous Driving][Event camera] This paper introduces the open-vocabulary event instance segmentation (OV-EIS) task for the first time…
tags:
  - "ICLR 2026"
  - "Autonomous Driving"
  - "Event camera"
  - "open-vocabulary instance segmentation"
  - "SAM"
  - "CLIP"
  - "multimodal fusion"
  - "annotation-free training"
date: 2026-05-08
content_hash: 99d16517771a5b23
---

# SEAL: Segment Any Events with Language

**Conference**: ICLR 2026
**arXiv**: [2601.23159](https://arxiv.org/abs/2601.23159)  
**Code**: [https://0nandon.github.io/SEAL](https://0nandon.github.io/SEAL) (coming soon)  
**Area**: Autonomous Driving
**Keywords**: Event camera, open-vocabulary instance segmentation, SAM, CLIP, multimodal fusion, annotation-free training

## TL;DR

This paper introduces the open-vocabulary event instance segmentation (OV-EIS) task for the first time, and proposes the SEAL framework. Through multimodal hierarchical semantic guidance (MHSG) and a lightweight multimodal fusion network, SEAL achieves multi-granularity (instance-level + part-level) semantic segmentation of event streams using only event–image pairs (without dense annotations), substantially outperforming all baselines while achieving the fastest inference speed.

## Background & Motivation

**Advantages of event cameras**: Event cameras offer extremely high temporal resolution, ultra-low latency, high dynamic range, and low power consumption, remaining effective in scenarios where conventional cameras fail, such as low-light and overexposed conditions.

**Limitations of existing event segmentation**: Existing event semantic segmentation (ESS) methods are confined to closed-set vocabularies and cannot recognize objects outside the training categories, nor can they distinguish between different instances of the same class.

**Open-vocabulary event understanding is nascent**: OpenESS achieves only open-vocabulary semantic segmentation without instance-level recognition; EventSAM supports event instance segmentation but lacks semantic recognition capability.

**Absence of evaluation benchmarks**: No multi-semantic benchmark dataset for event instance segmentation has previously existed.

**Efficiency requirements**: Event cameras are commonly deployed on edge devices, necessitating parameter-efficient and fast-inference model designs.

**Domain gap**: Directly applying image-domain pretrained models to event streams introduces a large domain gap due to noise and artifacts, even when images are reconstructed via E2VID.

## Method

### Overall Architecture

SEAL belongs to the annotation-free domain adaptation (AF-DA) category:
- **During training**: Only event–image pairs $(I^{evt}, I^{img})$ are used; no dense event annotations are required.
- **During inference**: Only event embeddings $I^{evt}$ are input; masks and category predictions are generated given user-provided visual prompts (points/boxes).
- The framework consists of two core components: the **MHSG module** (providing multimodal hierarchical semantic supervision) and the **multimodal fusion network** (a lightweight mask classifier).

### Key Designs 1: Multimodal Hierarchical Semantic Guidance (MHSG)

**Hierarchical visual guidance**:
- SAM is applied to paired images to generate segmentation maps at three granularities: semantic-level $M_s^{img}$, instance-level $M_i^{img}$, and part-level $M_p^{img}$.
- Pixel-level features are extracted via the CLIP visual encoder and then RoI-pooled to obtain visual features for each level of masks.

**Hierarchical textual guidance**:
- A LLaMA-based MLLM generates rich textual descriptions for each mask.
- These descriptions are encoded via the CLIP text encoder to form hierarchical textual guidance signals.
- Unlike OpenESS, this approach does not rely on predefined category names but instead leverages the MLLM to produce diverse and rich vocabulary.

### Key Designs 2: Multimodal Fusion Network (Three Components)

**① Backbone Feature Enhancer**:
- Six multimodal fusion modules (self-attention + cross-attention + FFN) are stacked on top of the EventSAM backbone features.
- During training, textual guidance $M_l^{text}$ serves as the key/value in cross-attention.
- During inference, dataset class names or user-defined language inputs are used instead.
- Mask features are pooled from language-fused features via RoI-Align.

**② Spatial Encoding**:
- Addresses two problems: *dead masks* (masks of small objects vanish after downsampling, yielding zero vectors) and *semantic conflict* (masks of different semantics project onto the same region on low-resolution feature maps).
- SAM mask decoder mask tokens are used to encode spatial priors such as shape and location.
- Spatial features $G_l^{evt}$ and semantic features $S_l^{evt}$ are concatenated and projected: $M_l^{evt} = \text{proj}(\text{concat}(G_l^{evt}, S_l^{evt}))$

**③ Mask Feature Enhancer**:
- Further enhances semantic and spatial priors in mask features through masked cross-attention layers.
- Language-fused backbone features (with positional encoding) serve as key/value, constraining attention to foreground regions.

### Loss & Training

- **Two-stage training**: Stage 1 trains EventSAM following its original protocol; Stage 2 freezes EventSAM and trains only the fusion network.
- **Training data**: Mixed-24K (merging DDD17-Seg and DSEC-Semantic training sets, totaling 24,032 pairs).
- **Loss function**: Cosine similarity distillation loss that aligns event mask features with both visual and textual guidance simultaneously:

$$\mathcal{L}_{distill} = \sum_{l \in \{s,i,p\}} \frac{1}{K_l}(1 - \cos(\hat{M}_l^{evt}, M_l^{img})) + \sum_{l \in \{s,i,p\}} \frac{1}{K_l}(1 - \cos(\hat{M}_l^{evt}, M_l^{text}))$$

## Key Experimental Results

### Four Evaluation Benchmarks

| Benchmark | Source | Test Size | Resolution | # Classes | Evaluation Dimension |
|-----------|--------|-----------|------------|-----------|---------------------|
| DDD17-Ins | DDD17-Seg | 3,890 | 352×200 | 6 | Coarse-grained instance segmentation |
| DSEC11-Ins | DSEC-Semantic | 2,809 | 640×440 | 11 | Medium-grained instance segmentation |
| DSEC19-Ins | DSEC-Semantic | 2,809 | 640×440 | 19 | Fine-grained instance segmentation |
| DSEC-Part | DSEC-Semantic | 2,809 | 640×440 | 9 (5+4) | Part-level segmentation |

### Main Results (Table 1: Closed-Set Instance Segmentation, Box Prompt AP)

| Method | Category | DDD17-Ins AP | DSEC11-Ins AP | DSEC19-Ins AP | Inference Time (ms) | Params (M) |
|--------|----------|-------------|--------------|--------------|-------------------|-----------|
| OVSAM | AR-CDG | 21.6 | 22.2 | 11.6 | 102.27 | 314.7 |
| OpenSeg | Hybrid | 35.0 | 23.6 | 13.0 | 427.01 | 228.4 |
| MaskCLIP++ | Hybrid | 32.8 | 25.4 | 14.1 | 394.61 | 301.7 |
| frame2recon | AF-DA | 34.8 | 21.2 | 10.5 | 278.35 | 141.7 |
| frame2voxel | AF-DA | 33.6 | 21.3 | 11.3 | 88.19 | 109.1 |
| **SEAL (Ours)** | **AF-DA** | **38.2** | **28.8** | **14.8** | **22.28** | **99.1** |
| Gain | - | +3.2 | +3.4 | +0.7 | - | - |

### Part Segmentation Results (Table 2: DSEC-Part)

| Method | Point AP | Box AP |
|--------|---------|--------|
| VLPart | 12.9 | 16.1 |
| **SEAL** | **13.6** | **18.3** |
| Gain | +0.7 | +2.2 |

### Ablation Study — Hierarchical Semantic Guidance (Table 3)

- Removing part-level guidance → part segmentation AP decreases (DSEC-Part Box: 14.4–15.4 vs. 18.3).
- Removing instance/semantic-level guidance → instance segmentation AP decreases.
- **Using all three granularities yields the best performance**, validating the necessity of hierarchical guidance.

### Ablation Study — Model Architecture (Table 5)

| Fusion | SE | MFE | DDD17 Box AP | DSEC-Part Box AP |
|--------|-----|-----|-------------|-----------------|
| ✓ | | | 35.5 | 14.9 |
| ✓ | ✓ | | 35.7 | 15.7 |
| ✓ | | ✓ | 38.1 | 16.6 |
| ✓ | ✓ | ✓ | **38.2** | **18.3** |

### Efficiency Advantage

- SEAL achieves an inference time of **22.28 ms**, far below all baselines (the second-best, frame2voxel, takes 88.19 ms — approximately **4× slower**).
- With **99.1M** parameters, SEAL is the most parameter-efficient solution (the second-best, frame2spike, has 95.9M parameters but much lower performance).
- The single-backbone architecture avoids the redundancy of baseline methods that require two separate backbones for mask generation and classification.

## Highlights & Insights

1. **First definition of the OV-EIS task**: Advances open-vocabulary event understanding from the semantic level to the instance level, filling a research gap.
2. **Elegant hierarchical semantic guidance design**: Leverages SAM's inherent three-level mask mechanism to construct part/instance/semantic three-granularity supervision in a natural and effective manner.
3. **Annotation-free training framework**: Requires only event–image pairs without any manual dense annotations; supervision signals are automatically generated via CLIP and MLLM.
4. **Dual advantage in efficiency and performance**: Inference speed is 4× faster than the fastest baseline, parameter count is the lowest, and AP is comprehensively highest — well-suited for low-power edge deployment of event cameras.
5. **Spatial encoding module resolves dead masks and semantic conflicts**: Spatial priors from SAM mask tokens compensate for semantic features, with UMAP visualizations clearly demonstrating improvements in the feature space.
6. **Four self-constructed evaluation benchmarks**: Cover label granularity (6/11/19 classes) and semantic granularity (instance/part), providing a complete evaluation framework for future research.

## Limitations & Future Work

1. **Dependency on event–image paired data**: Training still requires temporally synchronized event–image pairs, limiting applicability to purely event-based data.
2. **Visual prompts still required**: Inference requires user-provided point/box prompts; the prompt-free SEAL++ variant is only briefly mentioned in the appendix.
3. **Benchmark limitations**: All four benchmarks are drawn from driving scenarios (DDD17/DSEC), lacking validation across diverse settings such as indoor and industrial environments.
4. **Limited number of categories**: The closed-set evaluation covers at most 19 classes, and truly large-scale open-vocabulary capability has not been demonstrated.
5. **E2VID reconstruction quality**: The MHSG hierarchical guidance depends on paired image quality, which may degrade under extreme event conditions.
6. **Two-stage training**: Requires training EventSAM before training the fusion network, resulting in a relatively complex training pipeline.

## Related Work & Insights

| Direction | Representative Works | Relation to This Paper |
|-----------|---------------------|----------------------|
| Event semantic segmentation | EV-SegNet, ESS, HALSIE, HMNet | Prior work; performs semantic segmentation only |
| Event instance segmentation | EventSAM | Base model of this work; performs class-agnostic segmentation only |
| Open-vocabulary event understanding | OpenESS, EventCLIP, EventBind | Semantic-level only; this paper advances to the instance level |
| Image open-vocabulary segmentation | CLIP, MaskCLIP, OpenSeg, OVSeg | Used as mask classifiers in baselines |
| SAM and variants | SAM, OVSAM, Mask-Adapter | Provide spatial priors and baseline comparisons |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to define the OV-EIS task; MHSG hierarchical guidance design is original and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 benchmarks, 11 baseline comparisons, 3 ablation studies, with thorough visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous problem definition, and well-motivated exposition.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for open-world understanding in event vision; the framework is efficient and practically applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](../../CVPR2026/autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [\[ICCV 2025\] SAM4D: Segment Anything in Camera and LiDAR Streams](../../ICCV2025/autonomous_driving/sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [\[ICLR 2026\] ST4VLA: Spatially Guided Training for Vision-Language-Action Models](st4vla_spatially_guided_training_for_vision-language-action_models.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](../../CVPR2026/autonomous_driving/terraseg_self-supervised_ground_segmentation_for_any_lidar.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](../../NeurIPS2025/autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)

</div>

<!-- RELATED:END -->
