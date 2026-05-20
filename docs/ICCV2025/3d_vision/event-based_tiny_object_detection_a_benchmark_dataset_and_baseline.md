---
title: >-
  [Paper Note] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline
description: >-
  [ICCV 2025][3D Vision][Event Camera] This paper introduces EV-UAV, the first large-scale event camera benchmark for anti-UAV tiny object detection (147 sequences / 23M+ event-level annotations / average target size only…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Event Camera"
  - "Small Object Detection"
  - "Anti-UAV"
  - "Sparse Point Cloud"
  - "Spatiotemporal Correlation"
  - "benchmark"
date: 2026-05-08
content_hash: 57adc193e851d254
---

# Event-based Tiny Object Detection: A Benchmark Dataset and Baseline

**Conference**: ICCV 2025
**arXiv**: [2506.23575](https://arxiv.org/abs/2506.23575)  
**Code**: [https://github.com/ChenYichen9527/Ev-UAV](https://github.com/ChenYichen9527/Ev-UAV)  
**Area**: 3D Vision / Event Camera / Small Object Detection
**Keywords**: Event Camera, Small Object Detection, Anti-UAV, Sparse Point Cloud, Spatiotemporal Correlation, benchmark

## TL;DR
This paper introduces EV-UAV, the first large-scale event camera benchmark for anti-UAV tiny object detection (147 sequences / 23M+ event-level annotations / average target size only 6.8×5.4 pixels), and proposes EV-SpSegNet — a detection framework based on sparse 3D point cloud segmentation. The method exploits the observation that tiny moving targets form continuous elongated curves in spatiotemporal event point clouds, and incorporates a Spatiotemporal Correlation loss (STC loss) to guide the network in retaining target events. It outperforms 13 state-of-the-art methods across IoU/ACC/detection probability metrics while achieving 10–100× faster inference.

## Background & Motivation

### Problem Definition
Small object detection (SOD) in anti-UAV scenarios: real-time and accurate localization of extremely small UAV targets against complex backgrounds. Targets are tiny (average 6.8×5.4 pixels) and lack texture and contour features.

### Limitations of Prior Work

**Limitations of conventional frame cameras**:
- Low frame rate (30–60 Hz), unable to capture high-speed UAVs
- Limited dynamic range (~60 dB), failing under extreme illumination
- Data redundancy, with background occupying the vast majority of pixels

**Potential of event cameras**:
- Microsecond-level temporal resolution (≥10⁶ Hz)
- High dynamic range (120 dB) — functional in both bright and dim conditions
- Asynchronous sparse output, naturally avoiding data redundancy

**Insufficiency of existing event datasets**:
- VisEvent/EventVOT: UAV is only a sub-category, with large targets (84×66 / 129×100 pixels)
- F-UAV-D: indoor environments only
- NeRDD: large targets only with simple backgrounds
- Lack of event-level annotations — existing datasets provide only frame-level bounding boxes

**Limitations of existing event-based detection methods**:
- Frame-conversion methods (RVT/RED): convert events to image representations, compressing temporal information and processing redundant backgrounds
- Graph-based / SNN methods: require specialized hardware or deliver insufficient performance
- None are designed for the unique geometric structure of tiny objects

### Core Observation
Small moving targets form **continuous elongated curves** in spatiotemporal event point clouds, which are fundamentally distinct from the planar structures of the background and the discrete point-like distribution of noise. This geometric characteristic serves as a key discriminative cue between targets and non-targets.

## Method

### Overall Architecture
EV-SpSegNet is an event point cloud segmentation network based on a U-shaped sparse convolution architecture:
- **Input**: 8-second raw event stream, voxelized into a 3D sparse point cloud at 1 pixel × 1 pixel × 1 ms resolution
- **Encoder–Decoder**: symmetric structure with GDSCA modules at each level
- **Output**: per-event binary classification (target / non-target)
- **Loss**: BCE + STC spatiotemporal correlation loss

### Key Design 1: Grouped Dilated Sparse Convolution Attention (GDSCA)

The GDSCA module comprises three components:

1. **Grouped Dilated Sparse Convolution (GDSC) Block**:
    - Splits input features into groups along the channel dimension (4 groups by default)
    - Each group applies sparse convolution with a different dilation rate (1, 2, 3, 4)
    - Extracts multi-scale local temporal features — adapting to target curves of varying motion speeds

2. **Sp-SE Block** (Sparse Squeeze-and-Excitation):
    - Fuses features from groups with different dilation rates
    - Re-weights channels via channel attention

3. **Patch Attention Block**:
    - Partitions the point cloud into larger sub-regions
    - Applies self-attention across sub-regions to enable global context interaction
    - Downsamples before attention to reduce quadratic complexity

Design rationale: GDSC first captures local multi-scale features → Patch Attention then performs global interaction. Experiments show that using Patch Attention alone actually degrades performance, as global attention without sufficient local features leads to feature confusion.

### Key Design 2: Spatiotemporal Correlation Loss (STC Loss)

Standard BCE loss treats each event independently, ignoring neighborhood structure. STC loss introduces the prior that *the more high-confidence supporting events surround an event, the more likely it is to be a target*:

$$L_{stc}(p, y) = \begin{cases} -w_{stc}^\gamma \log(p), & y=1 \\ -(1-w_{stc})^\gamma \log(1-p), & y=0 \end{cases}$$

The spatiotemporal correlation weight is:

$$w_{stc}(p) = \text{sigmoid}\left(\sum_{e^j \in V^{k\tau}} p^j\right)$$

where $V^{k\tau}$ denotes the $k \times k \times \tau$ neighborhood. Default settings: $k=3, \tau=5, \gamma=2$.

Effect:
- For positive samples (target events): more supporting neighbors → larger $w_{stc}$ → larger weight → encourages retention of spatiotemporally correlated targets
- For negative samples (noise): isolated events → small $w_{stc}$ → large $(1-w_{stc})$ → higher penalty → encourages removal of isolated noise

### Key Design 3: Event-Level Annotation Method

To address the efficiency challenge of per-event annotation:
1. Events are first accumulated into frames, and 2D bounding boxes are annotated on frames
2. Each 2D bbox is extended into a 3D cuboid in XYT space according to the temporal interval Δt of the corresponding frame
3. All events within the 3D cuboid are taken as target annotations

This approach achieves microsecond-level precision while maintaining annotation efficiency.

## Key Experimental Results

### EV-UAV Dataset Statistics
- 147 event sequences (99 train / 24 val / 24 test)
- 23M+ target event annotations
- Average target size: 6.8×5.4 pixels (**extremely small**, roughly 1/50 of existing datasets)
- 45% ultra-small targets (<8×8); covers strong/normal/low illumination, multiple scenes, and multiple targets
- DAVIS346 camera, 346×260 resolution

### Main Results (Comparison with 13 SOTA Methods)

| Method | Type | IoU(%)↑ | ACC(%)↑ | Pd(%)↑ | Fa(10⁻⁴)↓ | Params | Infer.(ms) |
|--------|------|---------|---------|--------|-----------|--------|------------|
| YOLOV10-S | Frame | 32.55 | 33.39 | 32.18 | 589.67 | 7.3M | 1627 |
| RVT | Event+Voxel | 43.21 | 51.38 | 60.35 | 55.68 | 9.9M | 1737 |
| Spike-YOLO | SNN | 43.94 | 48.26 | 59.62 | 55.38 | 69.0M | 1883 |
| COSeg | Point Cloud | 51.89 | 60.93 | 71.32 | 9.21 | 23.4M | 364 |
| **EV-SpSegNet** | **Point Cloud** | **55.18** | **65.02** | **77.53** | **1.63** | **4.0M** | **35.9** |

Key findings:
- IoU surpasses the second-best method by 3.29 percentage points (55.18 vs. 51.89 by COSeg)
- False alarm rate of only 1.63×10⁻⁴, 34× lower than RVT
- 35.9 ms to process 8 seconds of event data, 10× faster than the fastest point cloud method (RandLA-Net at 353 ms)
- Only 4.0M parameters, smaller than most compared methods

### Ablation Study: Component Contributions

| GDSC | PA | STC Loss | IoU↑ | ACC↑ | Pd↑ | Fa↓ | Params |
|:---:|:---:|:---:|------|------|------|------|--------|
| - | - | - | 51.36 | 60.21 | 71.94 | 4.81 | 5.6M |
| ✓ | - | - | 52.73 | 62.64 | 73.12 | 4.28 | 3.8M |
| - | ✓ | - | 51.26 | 59.53 | 71.67 | 4.31 | 5.8M |
| ✓ | ✓ | - | 53.62 | 64.02 | 76.76 | 1.93 | 4.0M |
| ✓ | ✓ | ✓ | **55.18** | **65.02** | **77.53** | **1.63** | 4.0M |

Key findings:
1. PA alone slightly degrades performance — GDSC must first provide local features
2. GDSC+PA combination yields a significant jump (IoU +2.36, Fa drops from 4.28 to 1.93)
3. STC loss further improves upon GDSC+PA (IoU +1.56, Pd +0.77)

### Generalization of STC Loss

| Method | Original IoU | +STC IoU | Gain |
|--------|-------------|----------|------|
| KPConv | 48.19 | 50.32 | +2.13 |
| RandLA-Net | 50.32 | 51.15 | +0.83 |
| COSeg | 51.89 | 53.12 | +1.23 |

STC loss yields consistent improvements across all point cloud segmentation methods, demonstrating its generality.

### Ablation on Dilation Rate Combinations

| Dilation Rates | IoU↑ | Fa↓ |
|----------------|------|------|
| (1,2,3,4) | **55.18** | **1.63** |
| (1,2,3,5) | 53.21 | 2.41 |
| (1,3,5,7) | 52.75 | 4.61 |
| (1,3,5,9) | 51.35 | 4.76 |

Small-increment dilation rates (1,2,3,4) achieve the best performance; excessively large dilation rates cause corresponding events to be too far apart, undermining effective temporal correlation modeling.

## Highlights & Insights

1. **Physical intuition behind the problem formulation**: tiny targets form "continuous curves" in spatiotemporal point clouds, backgrounds form "surfaces," and noise forms "discrete points" — this geometric distinction is the core design inspiration.
2. **Elegant annotation scheme**: lifting 2D bounding boxes into 3D XYT cuboids automatically converts frame-level annotations to event-level annotations, while also supporting degradation to per-timestamp bbox annotations.
3. **Generality of STC loss**: applicable not only within EV-SpSegNet but also to existing point cloud segmentation methods with consistent gains — making it a general-purpose loss for event-based data.
4. **Extreme efficiency**: 4M parameters + 35.9 ms to process 8 seconds of data (compared to 1–4 seconds per frame for frame-based methods), leveraging sparse convolution to naturally avoid redundant computation.
5. **Compelling qualitative results**: the method detects targets that were missed by human annotators due to heavy background clutter.

## Limitations & Future Work

1. **Inherent limitation of event cameras**: stationary or slow-moving targets do not generate events, leading to detection failures — complementary integration with frame cameras is necessary.
2. **Annotation approximation**: events within the 3D cuboid do not necessarily all belong to the target (background events may be included), introducing annotation noise.
3. **Single-category detection only**: the current framework detects only UAV targets and has not been extended to multi-category scenarios.
4. **Resolution constraint**: DAVIS346 provides only 346×260 resolution; scalability to higher-resolution event cameras remains unverified.
5. **Insufficient per-condition evaluation**: although the dataset covers diverse illumination conditions, per-condition performance breakdowns are not reported.

## Related Work & Insights

- **Paradigm shift from 2D detection to 3D point cloud segmentation**: conventional event-based detection first converts events to frames and then applies YOLO/DETR — this work directly segments on 3D event point clouds, avoiding temporal information loss caused by frame conversion.
- **Design philosophy of STC loss** resembles Focal Loss (reweighting hard samples), but the weighting criterion shifts from "prediction difficulty" to "spatial neighborhood consistency."
- The grouped dilated convolution design in GDSC is inspired by image-domain techniques (Dilated Conv/Atrous Conv), but adapting it to 3D sparse convolution and combining it with Patch Attention is novel.
- The annotation methodology of EV-UAV is generalizable to other event camera tasks (e.g., event-based semantic segmentation, object tracking).

## Rating ⭐⭐⭐⭐
- **Novelty**: ⭐⭐⭐⭐ (geometric insight of "curve vs. surface vs. point" + STC loss + complete benchmark — solid contributions)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (comparison with 13 methods + multi-dimensional ablations + STC generalization + qualitative analysis — very comprehensive)
- **Writing Quality**: ⭐⭐⭐⭐ (problem motivation is clear, dataset description is thorough, method intuition is well communicated)
- **Value**: ⭐⭐⭐⭐⭐ (35.9 ms ultra-fast inference + open-source dataset/code, directly applicable to anti-UAV systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark](../../AAAI2026/3d_vision/griffin_aerial-ground_cooperative_detection_and_tracking_dataset_and_benchmark.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
