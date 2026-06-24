---
title: >-
  [Paper Note] LiDAR-Event Stereo Fusion with Hallucinations
description: >-
  [ECCV 2024][Hallucination Detection][Event camera] This paper proposes the first framework to fuse sparse LiDAR depth points with event stereo cameras. By "hallucinating" (inserting fictitious events) within the event stack representations (VSH) or the raw event stream (BTH), the framework compensates for the missing information of event cameras in motion-free or textureless regions, significantly improving event stereo matching accuracy.
tags:
  - "ECCV 2024"
  - "Hallucination Detection"
  - "Event camera"
  - "LiDAR fusion"
  - "stereo matching"
  - "depth estimation"
  - "event hallucination"
date: 2026-05-08
content_hash: b79dd94533f67703
---

# LiDAR-Event Stereo Fusion with Hallucinations

**Conference**: ECCV 2024  
**arXiv**: [2408.04633](https://arxiv.org/abs/2408.04633)  
**Code**: [Available](https://eventvppstereo.github.io/)  
**Area**: Hallucination Detection  
**Keywords**: Event camera, LiDAR fusion, stereo matching, depth estimation, event hallucination

## TL;DR

This paper proposes the first framework to fuse sparse LiDAR depth points with event stereo cameras. By "hallucinating" (inserting fictitious events) within the event stack representations (VSH) or the raw event stream (BTH), the framework compensates for the missing information of event cameras in motion-free or textureless regions, significantly improving event stereo matching accuracy.

## Background & Motivation

### Background
Event cameras (neuromorphic cameras) asynchronously report pixel brightness changes with microsecond-level temporal resolution and extremely high dynamic range, making them highly suitable for depth estimation under rapid motion and extreme lighting. Event stereo matching encodes event streams into structured representations (e.g., Histograms, Voxel Grids, MDES, etc.) and then estimates disparity maps using deep networks.

### Limitations of Prior Work
Event cameras only trigger upon brightness changes, leading to **catastrophic failures** in the following scenarios:

1. **Motionless scenes**: Absolutely no events are generated when the camera or objects are static.
2. **Large textureless regions**: Broad regions with uniform brightness, such as the sky, walls, or roads, do not trigger events.
3. The semi-dense nature of event data makes matching keypoint correspondences in stereo matching extremely difficult.

### Existing Methods and Limitations
- In RGB stereo matching, LiDAR fusion methods (concatenating inputs, modulating cost volumes, Virtual Pattern Projection) have been widely researched.
- However, the field of **Event Stereo + LiDAR Fusion** remains completely unexplored.
- Directly applying RGB fusion methods causes issues: the fixed frame rate of LiDAR (typically 10Hz) is inherently incompatible with the asynchronous acquisition of event cameras. This forces either only using depth when LiDAR is available (wasting most temporal data) or degrading the processing rate to the LiDAR frequency (losing the advantages of the microsecond temporal resolution of event cameras).

### Key Insight
Event cameras and LiDAR are inherently complementary:
- **Event Cameras**: Provide rich information at object boundaries (where brightness changes sharply), whereas LiDAR is sparse in these areas.
- **LiDAR**: Offers reliable range measurements in textureless and motion-free regions where event cameras provide no information.

### Core Idea
Inspired by Virtual Pattern Projection (VPP) in the RGB domain, a **"hallucination" mechanism** is designed: fictitious matching cues are inserted into the event data using LiDAR depth points. Given the known depth (i.e., disparity) of a pixel, identical distinctive patterns are injected into corresponding locations of the left and right views, facilitating the stereo network to find correct correspondences more easily.

## Method

### Overall Architecture

Depending on the level of access to the event stereo network, three frameworks are defined:
- **White box**: Access to the network and the stack representation implementation is available.
- **Gray box**: Access to the stack representation is available, but the internal network is inaccessible.
- **Black box**: Both the stack representation and the network are inaccessible.

Two hallucination strategies are proposed: VSH (for gray box) and BTH (for black box), neither of which requires modifying the stereo network itself.

### Key Designs

#### 1. Virtual Stack Hallucination (VSH)

**Function**: Directly inject virtual patterns onto the constructed event stack representations to enhance matching distinctiveness.

**Mechanism**: Given the left and right event stacks $\mathcal{S}_L, \mathcal{S}_R$ (of size $W \times H \times C$) and a set of LiDAR depth measurements $Z$, for each depth point $z(x,y)$:

1. Convert depth to disparity: $d(x,y) = \frac{bf}{z(x,y)}$
2. Compute the corresponding position in the right image: $x' = x - d(x,y)$
3. Inject identical virtual patterns into corresponding locations of the left and right stacks:

$$\mathcal{S}_L(x,y,c) \leftarrow \mathcal{A}(x,y,x',c), \quad \mathcal{S}_R(x',y,c) \leftarrow \mathcal{A}(x,y,x',c)$$

The virtual pattern $\mathcal{A}$ is randomly sampled from a uniform distribution:

$$\mathcal{A}(x,y,x',c) \sim \mathcal{U}(\mathcal{S}^-, \mathcal{S}^+)$$

where $\mathcal{S}^-, \mathcal{S}^+$ are the minimum and maximum values in the stack. Single pixels or local windows (with 3×3 showing the best performance) can be selected, supporting alpha blending.

**Design Motivation**: Event stacks are completely empty (semi-dense) in event-free regions; injecting matching-consistent random patterns significantly enhances local distinctiveness. Injecting the identical pattern into corresponding left and right locations directly provides matching cues for the correct disparity. Its effect on event stacks is more prominent than on RGB images because it operates on much sparser data.

#### 2. Back-in-Time Hallucination (BTH)

**Function**: Directly insert fictitious events into the raw event stream without accessing the stack representation.

**Mechanism**: Within the event histories $\mathcal{E}_L, \mathcal{E}_R$ sampled backward from time $t_d$, for each depth point $d(\hat{x},\hat{y})$, a pair of fictitious events is injected:

$$\hat{e}^L = (\hat{x}, \hat{y}, \hat{p}, \hat{t}), \quad \hat{e}^R = (\hat{x}', \hat{y}, \hat{p}, \hat{t})$$

satisfying three constraints:
- **Temporal Ordering**: $\hat{t}$ lies within the temporal range of the event history.
- **Geometric Constraint**: $\hat{x}' = \hat{x} - d(\hat{x},\hat{y})$.
- **Consistency Constraint**: The polarities $\hat{p}$ and timestamps $\hat{t}$ of the left and right fictitious events are identical.

**Single Timestamp Injection**: Inject $K_{\hat{x},\hat{y}}$ pairs of events with random polarities at a fixed timestamp $t_z$. A key advantage: even if $t_z < t_d$ (i.e., LiDAR data is outdated), it can still be effectively utilized as long as it falls within the event history time range.

**Repeated Injection**: A more advanced strategy that divides the event history into $B$ temporal bins and performs injections independently within each bin. Each depth point is injected only into one randomly allocated bin, enhancing distinctiveness in the temporal dimension. This uses $B=12$ injection points, with 2 fictitious events injected per point.

**Design Motivation**: BTH does not require access to stack representations (black-box compatible) and can leverage the advantages of the temporal dimension of event data. Repeated injection specifically enhances robustness against temporal misalignment of LiDAR data (misaligned, $t_z < t_d$).

#### 3. Framework-Level Adaptation

- Comprehensive support for 8 stack representations: Histogram, Voxel Grid, MDES, Concentration, TORE, Time Surface, ERGO-12, and Tencode.
- Supports both direct application on pre-trained models (without retraining) and training from scratch.
- Occlusion handling, uniform/non-uniform patches, and other details are inherited from VPP.

### Loss & Training

- Backbone network: AANet variant based on SE-CFF.
- Training: 25 epochs, batch size 4, maximum disparity of 192.
- Optimizer: Adam, lr=$5 \times 10^{-4}$ with cosine decay.
- Data augmentation: Random cropping and vertical flipping.
- Overhead: VSH introduces an additional 2-15ms of CPU overhead, while BTH introduces 10ms.

## Key Experimental Results

### Main Results

**DSEC Dataset - Pre-trained Models (Average Rank across 8 Representations)**:

| Fusion Method | 1PE↓ Avg Rank | 2PE↓ Avg Rank | MAE↓ Avg Rank | Description |
|---------|--------------|--------------|--------------|------|
| Baseline (No Fusion) | 3.00 | 3.00 | 3.00 | Pure Event Stereo |
| Guided [Poggi] | - | - | - | Cost volume modulation, limited improvement |
| **VSH (Ours)** | **1.75** | **1.38** | **1.50** | Gray box strategy |
| **BTH (Ours)** | **1.25** | **1.63** | **1.13** | Black box strategy, optimal |

**DSEC Dataset - Retrained Models (Average Rank across 8 Representations)**:

| Fusion Method | 1PE↓ Avg Rank | 2PE↓ Avg Rank | MAE↓ Avg Rank |
|---------|--------------|--------------|--------------|
| Concat [LidarStereoNet] | 3.38 | 3.00 | 3.13 |
| Guided+Concat [CCVNorm] | 3.63 | 3.50 | 3.38 |
| Guided [Poggi] | 5.00 | 5.00 | 5.00 |
| **VSH (Ours)** | **1.38** | **1.88** | **1.13** |
| **BTH (Ours)** | 1.63 | 1.38 | 1.88 |

During retraining, VSH achieves the best performance, with 1PE often dropping below 10% (e.g., ERGO-12: 9.25%).

**M3ED Dataset - Cross-Domain Generalization (Pre-trained)**:

| Representation | Baseline 1PE | VSH 1PE | BTH 1PE | Relative Improvement |
|------|-------------|---------|---------|---------|
| Histogram | 37.70 | 20.19 | 22.32 | ~46% |
| ERGO-12 | 36.33 | 22.53 | 20.41 | ~44% |
| Tencode | 43.56 | 28.24 | 22.61 | ~48% |

The improvement on M3ED is even more remarkable, with 1PE dropping from over 30-40% to around 20%.

### Ablation Study

**Hyperparameter Ablation on DSEC Search Set (1PE, Average of 8 Representations)**:

| Configuration | Performance | Description |
|------|------|------|
| VSH: Single pixel vs. 3×3 patch vs. 5×5 | 3×3 optimal | Appropriate patch size enhances distinctiveness |
| VSH: Random patterns vs. Uniform patterns | Uniform is better | Unified patterns are more effective |
| VSH: alpha=0 vs. 0.5 vs. 1.0 | 0.5 optimal | Balance between original content and pattern |
| BTH: Single injection vs. Repeated injection | Repeated injection is better | Exploits temporal dimension |
| BTH: 1 vs. 2 vs. 4 fictitious events | Saturated at 2 | A small number of events is sufficient |
| BTH: Random polarity vs. Uniform polarity | Uniform is better | Enhances consistency |

### Key Findings

1. **Guided method has limited effectiveness in event stereo**: 16-line LiDAR is too sparse, and cost volume modulation provides little help in event-free regions.
2. **VSH and BTH significantly outperform all adaptations of existing RGB fusion methods**: 1PE is reduced by 2-3% (pre-trained) or more (retrained).
3. **BTH is optimal for pre-training scenarios, while VSH is optimal for retraining scenarios**: BTH is more flexible (black-box), whereas VSH is more direct (suitable for trainable optimization).
4. **Outdated LiDAR data can still be effectively utilized**: The repeated injection strategy of BTH leads to only a minor drop in accuracy when $t_z < t_d$, preserving the microsecond-level temporal resolution advantage of event cameras.
5. **The method is generalized to all 8 event representations**: It is not tailored to a single specific representation.

## Highlights & Insights

1. **Pioneering Problem Definition**: First to explore event stereo + LiDAR fusion, identifying the inherent complementarity of the two sensors.
2. **Elegant Design of "Hallucination"**: Achieves significant improvements solely through data-level injections, without modifying network architectures or stack representation formats.
3. **Black-box Compatibility**: BTH operates without even needing access to stack representations, offering high versatility.
4. **Elegant Handling of Asynchronous Sensors**: Fully integrates outdated LiDAR data seamlessly by leveraging the temporal range of event history.

## Limitations & Future Work

1. VSH requires access to the stack representation (a gray-box limitation), and the occlusion handling in BTH is less sophisticated than in VSH.
2. In experiments, the alignment of sparse LiDAR points depends on external odometry and ICP registration.
3. The patterns of fictitious events are simplistic (random or uniform); exploring and learning optimized injection patterns is an area for future work.
4. Online calibration errors between the event camera and LiDAR are currently ignored.
5. The methodology can be extended to monocular event depth estimation + LiDAR fusion.

## Related Work & Insights

- **VPP (Virtual Pattern Projection)** [Bartolomei, CVPR 2024]: A pioneer in projecting virtual patterns for RGB stereo, whose concept is adapted to the event domain in this work.
- **SE-CFF** [Nam et al., CVPR 2024]: State-of-the-art event stereo matching framework, upon which this work builds.
- **DSEC** [Gehrig et al., RA-L 2021]: A large-scale outdoor event stereo dataset.
- **Insight**: The key to sensor fusion is not merely concatenating data, but identifying the failure modes of each sensor and using the other to compensate dynamically.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Opens a new direction in event stereo + LiDAR fusion with an elegantly designed hallucination mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive testing on 2 datasets, 8 stack representations, multiple fusion baselines, both pre-trained and retrained modes, and thorough hyperparameter ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly defined problem, detailed methodological description, and well-organized experimental structure.
- **Value**: ⭐⭐⭐⭐ — Highly practical (aligns with the primary sensor suite of autonomous driving) with excellent versatility (applicable across all 8 representations).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](../../ICML2026/hallucination/capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)
- [\[ECCV 2024\] BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models](beaf_observing_beforeafter_changes_to_evaluate_hallucination.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/hallucination/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[CVPR 2026\] Evaluating and Easing Hallucinations for GUI Grounding](../../CVPR2026/hallucination/exposing_and_evaluating_hallucinations_for_gui_grounding.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](../../ACL2026/hallucination/meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)

</div>

<!-- RELATED:END -->
