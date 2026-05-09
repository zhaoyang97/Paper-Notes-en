---
title: >-
  [Paper Note] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors
description: >-
  [AAAI 2026][Autonomous Driving][HD Map] This paper proposes PriorDrive, a framework that encodes multiple types of vectorized prior maps (SD maps, outdated HD maps, historical prediction maps) into a unified representation via a Unified Vector Encoder (UVE) and Hybrid Prior Representation (HPQuery), and integrates them into various online mapping models. It achieves a +14.3 mAP improvement on nuScenes and is compatible with both query-based and non-query-based mapping architectures.
tags:
  - AAAI 2026
  - Autonomous Driving
  - HD Map
  - Prior Map
  - Unified Vector Encoding
  - Plug-and-Play
  - Online Mapping
date: 2026-05-08
content_hash: 4581fd49e2146f11
---

# PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors

**Conference**: AAAI 2026
**arXiv**: [2409.05352](https://arxiv.org/abs/2409.05352)
**Code**: [https://github.com/MIV-XJTU/PriorDrive](https://github.com/MIV-XJTU/PriorDrive)
**Area**: Autonomous Driving / HD Map Construction
**Keywords**: HD Map, Prior Map, Unified Vector Encoding, Plug-and-Play, Online Mapping

## TL;DR

This paper proposes PriorDrive, a framework that encodes multiple types of vectorized prior maps (SD maps, outdated HD maps, historical prediction maps) into a unified representation via a Unified Vector Encoder (UVE) and Hybrid Prior Representation (HPQuery), and integrates them into various online mapping models. It achieves a +14.3 mAP improvement on nuScenes and is compatible with both query-based and non-query-based mapping architectures.

## Background & Motivation

HD maps are critical for autonomous driving, but traditional creation and maintenance pipelines are costly and suffer from update latency. Online mapping constructs maps in real time using onboard sensors, yet remains bottlenecked by occlusion, adverse weather, and degraded performance in distant regions. Leveraging prior maps (SD maps, outdated HD maps, historical predictions) can compensate for the limitations of single-source perception data.

However, existing prior map utilization methods exhibit systematic deficiencies:
1. P-MapNet/NMP/HRMapNet employ **rasterized encoding**, which suffers from resolution-limited information loss and redundancy, and requires complex post-processing for format conversion.
2. MapEX exploits only a single prior source (outdated HD maps), whose low update frequency makes it difficult to reflect real-time road changes.
3. NavMap discards variable-length segment information when integrating HD and SD maps, causing loss of valuable data.
4. Vector encoding faces three unique challenges: different maps contain different vector types (points/lines/polygons), vector lengths are variable, and alignment and consistency across maps must be addressed.
5. Existing methods typically encode only position and category attributes, lacking the ability to capture fine-grained information such as direction, geometry, and topology.
6. The potential of historical prediction maps as a prior source is largely overlooked.

## Method

### Overall Architecture

Multi-source prior maps $M_{\text{prior}} = \{M_1, M_2, \ldots, M_t\}$ are fed into UVE to produce a unified encoding $f_{\text{prior}}$, which then interacts with the online mapping model: for non-query-based models, it is fused into BEV features via DeConv+Conv; for query-based models, it interacts with queries at both instance and point levels via HPQuery. The enhanced map prediction is $P = D_{\text{map}}(f_{\text{bev}}, \text{HPQuery})$. Predictions can be uploaded to the cloud as prior maps for other vehicles, forming a positive feedback loop.

### Key Designs

1. **Unified Vector Encoder (UVE)**: Analogous to BERT encoding text — vector points ≈ words, vector elements ≈ sentences. The Fused Prior Embedding (FPE) integrates five types of information: positional embedding (Fourier-encoded $x, y$), directional embedding (encoding $v_x, v_y$), a [VEC] token (an anchor at the head of each vector for extracting instance-level features), learnable instance and type embeddings (distinguishing different vector instances and types), and 2D positional embedding to preserve point order. A Dual Encoding mechanism applies $M$ layers of intra-vector attention (masked to the same instance, learning point interactions to enhance perception) followed by $N$ layers of inter-vector attention (fully open mask, learning global cross-instance context), extracting fixed-length instance-level and point-level features from variable-length vectors. An attention masking mechanism adaptively focuses on higher-quality elements across different prior maps.

2. **Vector Data Pre-training (Position Modeling)**: A novel segment-level and point-level pre-training paradigm. A Noise & Mask Generator operates in two modes: (a) Noise — applying Gaussian noise $\varepsilon \sim \mathcal{N}(0,1)$ to horizontal and vertical coordinates of 10% of segment-level (all points within selected elements) and 5% of point-level selections; (b) Mask — setting selected point coordinates to $-1$. After UVE encoding, an MLP reconstructs all point coordinates with loss $L = \text{RMSE}(P, \text{mlp}(E_{\text{uve}}(M_{\text{org}}^*)))$. Pre-training runs for 24 epochs (~12 hours) as a one-time process, improving UVE's ability to denoise and encode noisy historical maps.

3. **HPQuery Integration**: For query-based models (MapTR series) with $Q = \{q_{ij}\} = \{q_i^{\text{ins}} + q_j^{\text{pt}}\}$, three fusion operations are provided: addition ($q^* = q.\text{add}(f)$, prior features added to corresponding queries), replacement ($q^* = q.\text{replace}(f)$, prior features directly replace part of the query), and concatenation ($q^* = \text{concat}[q, f]$, prior features appended to queries). These three operations interact at both instance and point levels. For non-query-based models, $f_{\text{prior}}$ is upsampled via DeConv, then concatenated and convolved into the BEV features.

### Loss & Training

- Pre-training: $L = \text{RMSE}(P, \text{mlp}(E_{\text{uve}}(M_{\text{org}}^*)))$, 24 epochs, ~12 hours, performed once.
- Main training: The original loss functions and hyperparameters of each baseline model are retained unchanged (e.g., Hungarian matching loss in MapTR). PriorDrive introduces no additional loss terms.
- Compute: 8× RTX A6000.

## Key Experimental Results

### Main Results: Cross-Dataset / Cross-Model Validation

| Dataset | Baseline | Metric | +PriorDrive | Baseline | Gain |
|---------|----------|--------|-------------|----------|------|
| nuScenes | MapTRv2 R50 24ep | mAP | **75.8** | 61.5 | +14.3 |
| nuScenes | MapTRv2 R50 110ep | mAP | **80.5** | — | New SOTA |
| nuScenes | HDMapNet Effi-B0 | mIoU | **40.1** | 32.9 | +7.2 |
| nuScenes | PivotNet R50 24ep | mAP | **65.3** | 56.5 | +8.8 |
| Argoverse 2 | MapTRv2 | mAP | **72.8** | 64.7 | +8.1 |
| OpenLane-V2 | TopoLogic | OLS | **46.2** | 41.6 | +4.6 |

### Ablation Study

| Configuration | mAP | Change | Notes |
|---------------|-----|--------|-------|
| Full model (SD+HD+Local) | **75.8** | — | Three complementary priors optimal |
| w/o UVE (separate MLPs) | 69.7 | −6.1 | Unified encoder is critical |
| w/o pre-training | 71.5 | −4.3 | Pre-training improves encoding quality |
| SD map prior only | 72.8 | −3.0 | Single source inferior to multi-source |
| Outdated HD map prior only | 73.3 | −2.5 | HD map precise but partially outdated |
| Historical local prior only | 72.3 | −3.5 | Single prediction insufficient |
| No prior (original MapTRv2) | 61.5 | −14.3 | Baseline reference |
| Search range 5m | **75.8** | — | Optimal search range |
| Search range 10m | 74.2 | −1.6 | Wider range introduces noise |

### Key Findings

- The three prior map types are complementary: each independently improves performance, and their combination achieves the best result (SD+3.0, HD+2.5, Local+3.5 → combined +14.3).
- More online local priors yield better results (1 source: +3.0 mAP; multiple sources: +5.2 mAP), demonstrating information accumulation benefits.
- Minimal computational overhead: FPS decreases from 10.3 to 9.9 (−3.9%), parameters increase by only 3.1 MB, and UVE accounts for only 15.3% of runtime.
- Effectiveness is maintained on a newly split dataset with no geographic overlap, demonstrating that the model genuinely learns to leverage prior information rather than memorizing maps.
- Gains are more significant at larger perception ranges (60m×30m), confirming the advantage of prior maps in distant regions.
- Validation on topological reasoning (OpenLane-V2 OLS +4.6) confirms cross-task generalization.

## Highlights & Insights

- Truly plug-and-play: compatible with HDMapNet, MapTR, MapTRv2, PivotNet, TopoLogic, and other mapping models.
- First framework to unify encoding of three types of prior maps, maximizing complementary information utilization.
- The analogy between vector data and NLP text (points ≈ words, elements ≈ sentences) is a novel and effective modeling perspective.
- The vector pre-training paradigm (segment-level + point-level noise/masking → coordinate reconstruction) fills a gap in the field.
- Positive feedback loop design: current predictions are uploaded to the cloud, become priors for other vehicles, and continuously improve overall accuracy.

## Limitations & Future Work

- Prior maps must be obtained from external sources (SD maps, historical data), making cold-start deployment in new regions challenging.
- Pre-training depends on the vector map distribution within the training dataset; cross-domain transfer requires retraining.
- Alignment and consistency across different priors are partially mitigated by the attention mechanism but not fully resolved.
- FPS drops from 10.3 to 9.9; while marginal, further optimization may be needed for scenarios with strict real-time requirements.

## Related Work & Insights

- vs. P-MapNet/NMP: Vector-based vs. rasterized encoding, avoiding resolution limitations and information loss.
- vs. MapEX: Supports multiple prior types vs. only outdated HD maps, with finer-grained encoding.
- vs. HRMapNet: Surpasses its SOTA at 110ep (80.5 vs. 73.6); the technical approach is more general.
- The unified vector encoder design is transferable to other heterogeneous vector data fusion scenarios (e.g., joint encoding of point clouds, trajectories, and maps).

## Rating

⭐⭐⭐⭐⭐ (5/5)
The problem is precisely formulated, the methodology is comprehensive (encoder + pre-training + integration), and experiments across 3 datasets × multiple baselines × detailed ablations are highly thorough. The work has direct engineering value for HD map construction in autonomous driving.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2026/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)

<!-- RELATED:END -->
