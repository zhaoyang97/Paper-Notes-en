---
title: >-
  [Paper Note] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions
description: >-
  [AAAI 2026][Autonomous Driving][Multimodal Gait Recognition] This paper introduces LRGait — the first LiDAR-Camera multimodal gait dataset targeting long-range (10–50m) cross-distance scenarios — and proposes EMGaitNet, an end-to-end framework that achieves 2D-3D cross-modal feature fusion via Semantic Mining (SeMi), Semantic-Guided Alignment (SGA), and Symmetric Cross-Attention Fusion (SCAF) modules, reaching state-of-the-art performance on multiple benchmarks.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Multimodal Gait Recognition
  - Long-Range Recognition
  - LiDAR-Camera Fusion
  - CLIP Semantic Guidance
  - Cross-Distance Retrieval
date: 2026-05-08
content_hash: 8b2eca9783cf1596
---

# Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions

**Conference**: AAAI 2026
**arXiv**: [2603.14189](https://arxiv.org/abs/2603.14189)
**Code**: [github.com/O-VIGIA/LRGait](https://github.com/O-VIGIA/LRGait)
**Area**: Autonomous Driving / Gait Recognition
**Keywords**: Multimodal Gait Recognition, Long-Range Recognition, LiDAR-Camera Fusion, CLIP Semantic Guidance, Cross-Distance Retrieval

## TL;DR

This paper introduces LRGait — the first LiDAR-Camera multimodal gait dataset targeting long-range (10–50m) cross-distance scenarios — and proposes EMGaitNet, an end-to-end framework that achieves 2D-3D cross-modal feature fusion via Semantic Mining (SeMi), Semantic-Guided Alignment (SGA), and Symmetric Cross-Attention Fusion (SCAF) modules, reaching state-of-the-art performance on multiple benchmarks.

## Background & Motivation

### State of the Field

Gait recognition is a non-intrusive and difficult-to-spoof biometric technology with important applications in intelligent surveillance and remote identity verification. While recent methods have achieved strong results in controlled environments, significant challenges remain under **long-range** and **multimodal** conditions.

### Limitations of Prior Work

| Dataset | Sensor | Max Distance | Cross-Distance | Day/Night |
|---------|--------|--------------|----------------|-----------|
| CASIA-B | Camera | 2–4m | ✗ | ✗ |
| SUSTech1K | LiDAR+Camera | 8–12m | ✗ | ✗ |
| FreeGait | LiDAR+Camera | 25m | ✗ | ✗ |
| **LRGait (Ours)** | **LiDAR+Camera** | **10–50m** | **✓** | **✓** |

Core issues:
1. Existing datasets extend to at most 25m, insufficient to cover real-world surveillance needs (e.g., 50m).
2. No cross-distance samples for the same identity at varying distances (e.g., 50m→10m retrieval).
3. Most methods support only a single modality, failing to exploit the complementary advantages of LiDAR and RGB.

### Root Cause

**Modality Gap**: The representation spaces of RGB images and LiDAR point clouds differ substantially, making direct fusion ineffective.

**Preprocessing Loss**: Existing methods typically use depth maps (projected from point clouds) or silhouette maps (extracted from RGB) as inputs, discarding fine-grained geometric/textural details.

**Long-Range Degradation**: At long distances, point clouds become extremely sparse and RGB images become blurry, greatly exacerbating the degradation of preprocessing-based methods.

## Method

### Overall Architecture

EMGaitNet is an end-to-end framework that directly processes **raw RGB video** and **raw point cloud sequences**:

1. **Feature Extraction**: ResNet9 extracts 2D visual features; PointGNN extracts 3D geometric features.
2. **SeMi Module**: CLIP-based semantic mining that extracts body-part-aware semantic cues.
3. **SGA Module**: Semantic-Guided Alignment that bridges the 2D-3D modality gap using semantic features.
4. **SCAF Module**: Symmetric Cross-Attention Fusion that hierarchically integrates 2D-3D features.
5. **ST Module**: Spatio-temporal module that captures global gait dynamics.

### Key Designs

#### 1. Dual-Stream Feature Extraction

**2D Branch**: Uses OpenGait's lightweight ResNet9 backbone to extract per-frame visual features $F_{i,j}^{2d} \in \mathbb{R}^{h \times w \times d}$.

**3D Branch**: Employs the PointGNN backbone to process raw point clouds, progressively capturing local and global geometric patterns via graph convolution layers:

- A local neighborhood graph is constructed based on feature cosine similarity (TopK most similar points):
$$\mathcal{N}_{P_i^j}(P_i^j[k]) = \underset{u \neq k}{\text{TopK}}(\cos(F_{i,j}^{3d}[k], F_{i,j}^{3d}[u]))$$

- Edge features are computed and aggregated:
$$F_{i,j}^{3d}[k] = \text{Maxpool}_{u \in \mathcal{N}}(\text{MLP}(e_{k,u}))$$

**Design Motivation**: Graph-based representations in PointGNN are better suited for sparse point clouds (long-range scenarios) than PointNet++. Constructing graphs based on feature similarity rather than pure spatial distance mitigates the impact of point cloud sparsity.

#### 2. CLIP Semantic Mining Module (SeMi)

**Function**: Leverages CLIP to extract body-part-level semantic cues, serving as an intermediate bridge for cross-modal alignment.

**Core Idea**:
- Constructs body-part prompts: "A photo of the [PART] of a [X] person", where [PART] ∈ {"head", "arms", "torso", "legs", "feet"}.
- Extracts global visual embeddings $v = \text{CLIP}_v(I_i^j)$ using the CLIP visual encoder.
- Maps visual features $v$ to the text space via an **Inversion Network** to obtain $v^*$, replacing the [X] placeholder.
- Feeds the modified prompts into the CLIP text encoder to produce identity-aware semantic features $t^* \in \mathbb{R}^{5 \times d}$.

**Design Motivation**:
- Class-level semantics (e.g., "a person's legs") are insufficient; gait recognition requires **instance-level** fine-grained semantics.
- The Inversion Network injects individual-specific visual information into the prompts, personalizing the semantic cues.
- Body-part decomposition provides a natural intermediate representation for subsequent cross-modal alignment.

#### 3. Semantic-Guided Alignment Module (SGA)

**Function**: Uses the multi-granularity semantic cues produced by SeMi as an intermediate bridge to align RGB and point cloud features.

**Core Idea**: Applies cross-attention with 2D/3D features as Query and semantic features $t^*$ as Key/Value:

$$\text{CA}(\bar{F}_{i,j}^{2d}, t^*) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

Features are refined via residual connection and FFN:

$$\hat{F}_{i,j}^{2d} = \text{LayerNorm}(\tilde{F}_{i,j}^{2d} + \text{FFN}(\tilde{F}_{i,j}^{2d}))$$

The same operation is applied to 3D features $F_{i,j}^{3d}$, yielding aligned $\hat{F}_{i,j}^{3d}$.

**Design Motivation**: Semantic features serve as shared alignment anchors, aligning features from different modalities in a common semantic space while suppressing background noise (e.g., spurious associations between LiDAR point clouds and irrelevant RGB background regions) via the attention mechanism.

#### 4. Symmetric Cross-Attention Fusion Module (SCAF)

**Function**: Hierarchically integrates complementary information in the aligned shared space through symmetric dual-stream cross-attention.

**Core Idea**: 2D and 3D features alternately serve as Query while attending to the other as Key/Value, enabling bidirectional alignment and mutual information fusion:

$$F_{i,j}^{2d'} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where each attention head is:
$$\text{head}_k = \text{Softmax}\left(\frac{Q_k K_k^\top}{\sqrt{d_h}}\right) V_k$$

$Q_k$ is derived from 2D features and $K_k, V_k$ from 3D features (and vice versa).

**Design Motivation**: The symmetric structure ensures both modalities contribute equally; the multi-head design captures complementary information across different subspaces; hierarchical fusion progressively refines cross-modal features.

#### 5. Spatio-Temporal Module (ST)

**Function**: Aggregates global gait dynamics.

- **Temporal Pooling**: MaxPool over the 2D feature sequence along the temporal dimension → $F_i^{tp} \in \mathbb{R}^{h \times w \times d}$
- **Spatial Pooling**: AvgPool over 3D features along the point dimension → $F_i^{sp} \in \mathbb{R}^{n \times d}$
- **Spatio-Temporal Cross-Attention Fusion**:

$$\tilde{F}_i^{sp} = \text{CA}(F_i^{sp}, F_i^{tp}) + F_i^{sp}$$
$$F_i^{fusion} = \text{MLP}(\text{CA}(F_i^{tp}, \tilde{F}_i^{sp}) + F_i^{tp})$$

Horizontal Pyramid Pooling (HPP) is then applied for part-level matching.

### Loss & Training

$$\mathcal{L} = \alpha \mathcal{L}_{tri} + \beta \mathcal{L}_{ce}$$

- $\mathcal{L}_{tri}$: Triplet loss ($\alpha=1.0$) for learning discriminative embedding spaces.
- $\mathcal{L}_{ce}$: Cross-entropy loss ($\beta=2.0$) for classification supervision.
- L2 distance is used at inference to measure similarity between probe and gallery samples.
- Adam optimizer with weight decay 0.0005; MultiStepLR decay at 15K and 30K epochs.
- 10 RGB frames and corresponding point clouds are randomly sampled per epoch; trained on 2× RTX 3090 GPUs.

## Key Experimental Results

### Main Results

**SUSTech1K (Overall Rank-1 Accuracy)**:

| Method | Modality | NM | BG | CL | OC | Overall R-1 | Overall R-5 |
|--------|----------|-----|-----|-----|-----|------------|------------|
| GaitBase | sil | 81.3 | 77.3 | 49.6 | 81.4 | 76.0 | 89.1 |
| LidarGait++ | pc | 94.2 | 93.9 | 79.7 | 91.9 | 92.7 | 98.2 |
| LiCAF | depth+sil | 95.8 | 95.7 | 82.7 | 96.6 | 93.9 | 98.8 |
| **EMGaitNet** | **pc+rgb** | **98.2** | **96.4** | **81.7** | **99.6** | **96.0** | **99.0** |

**LRGait (Cross-Distance Cross-View Rank-1, gallery=D-10)**:

| Method | Modality | D-20 | D-30 | D-40 | D-50 | N-20 | N-30 | Overall R-1 |
|--------|----------|------|------|------|------|------|------|------------|
| GaitBase | sil | 67.9 | 53.9 | 48.5 | 33.8 | 41.6 | 33.4 | 46.8 |
| LiCAF | depth+sil | 74.8 | 71.6 | 60.4 | 65.3 | 42.5 | 27.8 | 59.6 |
| **EMGaitNet** | **pc+rgb** | **88.5** | **82.4** | **80.8** | **74.4** | **38.2** | **31.7** | **68.9** |

**FreeGait**:

| Method | R-1 | R-5 | mAP |
|--------|-----|-----|-----|
| LidarGait++ | 82.0 | 93.6 | 87.2 |
| HMRNet | 80.8 | 93.6 | 86.5 |
| **EMGaitNet** | **85.2** | **96.8** | **89.0** |

### Ablation Study

**Contribution of Each Module (LRGait dataset)**:

| Configuration | Baseline | +SGA | +SGA+SeMi | +SGA+SeMi+ST |
|---------------|----------|------|----------|-------------|
| Overall R-1 | 52.3 | 58.5 | 64.2 | **68.9** |
| Overall R-5 | 70.2 | 75.9 | 80.7 | **85.8** |

Incremental contribution of each module:
- SGA: +6.2% (bridging the modality gap)
- SeMi: +5.7% (semantic guidance)
- ST: +4.7% (spatio-temporal modeling)

### Key Findings

1. **End-to-end outperforms preprocessing**: EMGaitNet using raw pc+rgb surpasses LiCAF using depth+sil by 2.1% on SUSTech1K.
2. **Clear advantage at long range**: At D-50, EMGaitNet outperforms the second-best method by 14.0% (74.4% vs. 65.3%), demonstrating the superiority of end-to-end approaches at far distances.
3. **Strong occlusion robustness**: Achieves 99.6% accuracy under OC conditions, indicating that multimodal priors effectively compensate for missing occluded information.
4. **Nighttime remains challenging**: All methods degrade substantially at night; EMGaitNet achieves only 21.9% at N-40.
5. **SeMi contributes most critically**: Body-part-level semantic cues provide the essential intermediate representation for cross-modal alignment.

## Highlights & Insights

1. **First 50m long-range multimodal gait dataset**: Fills the gap in long-range gait recognition data, covering cross-distance, day/night, and multi-weather conditions.
2. **CLIP-guided cross-modal fusion**: An inversion network injects visual information into the text space to generate instance-level semantic anchors — a novel cross-modal alignment strategy.
3. **End-to-end processing of raw inputs**: Avoids information loss caused by feature preprocessing (depth mapping, silhouette extraction), especially in long-range and nighttime scenarios.
4. **PointGNN for gait recognition**: First application of a graph-based 3D backbone to gait tasks; constructing graphs via feature similarity rather than spatial distance adapts well to sparse point clouds.
5. **LiDAR long-range pedestrian detection benchmark**: Additional annotation of 4,500 frames for long-range pedestrian detection provides independent research value.

## Limitations & Future Work

1. **Severe day-to-night domain shift**: Nighttime performance drops substantially; dedicated multimodal domain adaptation methods are needed.
2. **Frozen CLIP**: The possibility of fine-tuning CLIP on gait data remains unexplored.
3. **Computational overhead**: Graph construction in PointGNN and CLIP encoding increase inference latency.
4. **Only 101 identities**: Dataset scale remains small compared to GREW (26K identities).
5. **Single-person assumption**: Multi-person scenarios that arise in real-world surveillance are not addressed.

## Related Work & Insights

- **Comparison with LiCAF**: LiCAF performs asymmetric fusion via simple cross-attention, whereas EMGaitNet employs semantically guided symmetric fusion, offering a more principled design.
- **Relationship to SUSTech1K**: LRGait extends the distance range from 12m to 50m, representing a major advancement in the distance coverage of gait datasets.
- **Insight**: The semantic capabilities of VLMs such as CLIP can effectively bridge heterogeneous modalities; this strategy is generalizable to other multimodal fusion tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Outstanding dataset contribution; CLIP-guided fusion is creative, though individual modules are relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, detailed ablations, and comparisons across diverse modalities and conditions.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with thorough dataset description.
- Value: ⭐⭐⭐⭐ — The dataset and baselines provide an important foundation for long-range gait recognition research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](../../CVPR2026/autonomous_driving/foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)
- [\[AAAI 2026\] TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)

</div>

<!-- RELATED:END -->
