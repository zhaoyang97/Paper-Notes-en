---
title: >-
  [Paper Note] Enhancing Vectorized Map Perception with Historical Rasterized Maps
description: >-
  [ECCV2024][Autonomous Driving][vectorized map perception] This paper proposes HRMapNet, which maintains a low-cost global historical rasterized map to provide complementary prior information for online vectorized map perception. It enhances existing methods at two levels—BEV feature aggregation and query initialization—achieving significant improvements on nuScenes and Argoverse 2.
tags:
  - "ECCV2024"
  - "Autonomous Driving"
  - "vectorized map perception"
  - "historical rasterized map"
  - "BEV"
  - "bird's-eye-view"
  - "HD map"
date: 2026-05-08
content_hash: 808399296d567e04
---

# Enhancing Vectorized Map Perception with Historical Rasterized Maps

**Conference**: ECCV2024  
**arXiv**: [2409.00620](https://arxiv.org/abs/2409.00620)  
**Code**: [HXMap/HRMapNet](https://github.com/HXMap/HRMapNet)  
**Area**: Autonomous Driving  
**Keywords**: vectorized map perception, historical rasterized map, BEV, bird's-eye-view, HD map, autonomous driving

## TL;DR

This paper proposes HRMapNet, which maintains a low-cost global historical rasterized map to provide complementary prior information for online vectorized map perception. It enhances existing methods at two levels—BEV feature aggregation and query initialization—achieving significant improvements on nuScenes and Argoverse 2.

## Background & Motivation

- High-definition maps (HD maps) are crucial for autonomous driving, but traditional offline construction is extremely costly, prompting researchers to shift toward online map perception based on onboard sensors.
- Online vectorized map perception methods, represented by MapTR, directly predict vectorized map elements in the BEV space. However, relying solely on the current frame's onboard sensors leads to a substantial decline in accuracy and robustness under challenging scenarios such as occlusions, adverse weather, or nighttime.
- Temporal information serves as a feasible completion. However, existing methods (e.g., StreamMapNet) only leverage short-term temporal information from the past few frames, failing to fully exploit the value of historical observations.
- The core insight of this paper is that historical prediction results can be rasterized and accumulated into a global map at a low cost, serving as prior information for online perception. Rasterized maps offer advantages such as ease of merging, easy retrieval, clear semantics, and small memory footprint.

## Core Problem

How can historical map information be maintained and utilized in a cost-effective manner to compensate for the perceptual limitations of single-frame onboard sensors in challenging scenarios?

## Method

### 1. Global Rasterized Map Construction and Maintenance

- The online prediction results (vectorized map) of each frame are rasterized into a local raster map $M_i^l \in \{0,1\}^{H \times W \times N}$, where $N=3$ represents three categories of map elements: lane divider, pedestrian crossing, and road boundary.
- Based on the ego-pose, local coordinates are mapped to global coordinates, and a global map $M^g$ is maintained using update rules similar to an occupancy grid map:
    - If the local prediction indicates the presence of a map element at a position, the global value increases by $S^+$ (default: 30).
    - If not, the global value decreases by $S^-$ (default: 1).
- During retrieval, a local raster map is cropped from the global map based on the current ego-pose and binarized using a threshold $S_{th}$.
- The global map is stored using 8-bit unsigned integers, with a memory cost of approximately 1 MB per kilometer (the Boston map in nuScenes is only about 120 MB, whereas the BEV feature-based map in NMP requires 11 GB).

### 2. BEV Feature Aggregation Module

- Existing methods extract BEV features $F_I \in \mathbb{R}^{H \times W \times C}$ from onboard camera images.
- HRMapNet additionally places BEV queries at positions with map elements in the retrieved local raster map and extracts corresponding features from the images via spatial cross-attention, obtaining the complementary BEV features $F_M$. Positions without map elements are padded with zeros.
- Final feature fusion: $F_{BEV} = \text{Conv}(\text{Concat}(F_I + F_M, M^l))$, where the image BEV features, complementary map features, and the semantic information of the raster map itself are concatenated and then fused via convolution.

### 3. Query Initialization Module

- In the DETR paradigm, learnable queries must search for map elements from random positions. The historical raster map provides a prior for where these elements are likely to exist.
- For each valid position $p$ in the raster map, a position embedding $PE(p)$ and a semantic label embedding $LE(p)$ are encoded and summed to obtain the map prior embedding $ME(p) = PE(p) + LE(p)$.
- Base queries interact with the map prior embeddings via cross-attention before being fed into the original decoder layers. This enables queries to locate target elements more efficiently.
- To control memory overhead, the local raster map is downsampled (default resolution of 0.6 m) before extracting the prior embeddings.

### 4. Training and Inference

- The loss function is completely identical to that of the baseline methods (classification loss, point-to-point loss, edge direction loss, etc.).
- During training, the global map is progressively updated from scratch in each epoch.
- During inference, the global map also starts empty by default and is updated incrementally as the test frames progress chronologically.

## Key Experimental Results

### Main Results on nuScenes

| Method | Extra Info | Epoch | mAP |
|------|----------|-------|-----|
| MapTRv2 | None | 24 | 61.5 |
| **HRMapNet (MapTRv2)** | HRMap | 24 | **67.2 (+5.7)** |
| MapTRv2 | None | 110 | 68.7 |
| **HRMapNet (MapTRv2)** | HRMap | 110 | **73.6 (+4.9)** |
| StreamMapNet | None | 24 | 60.4 |
| **HRMapNet (StreamMapNet)** | HRMap | 24 | **66.3 (+5.9)** |

### Main Results on Argoverse 2

| Method | mAP |
|------|-----|
| MapTRv2 | 64.3 |
| **HRMapNet (MapTRv2)** | **68.3 (+4.0)** |
| StreamMapNet | 61.5 |
| **HRMapNet (StreamMapNet)** | **64.3 (+2.8)** |

### Ablation Study

- BEV Feature Aggregation only: +3.1 mAP ($61.5 \rightarrow 64.6$)
- Plus Query Initialization: +2.6 mAP ($64.6 \rightarrow 67.2$)
- Both modules make significant contributions.

### Impact of Initial Map on Performance

| Initial Map | mAP |
|----------|-----|
| Empty Map (Default) | 67.2 |
| Self-Constructed Map on Val Set (Two-Pass) | 72.6 (+5.4) |
| Train Set Map | 83.7 (+16.5) |

### Robustness to Localization Errors

- Under a translation error of 0.1 m + rotation error of 0.01 rad, the mAP drops by only about 1.2 ($67.2 \rightarrow 66.0$).
- Under the most severe noise setting (0.2 m + 0.02 rad), the performance is still 63.8, consistently outperforming the mapper-free baseline of 61.5.

### Inference Speed

- HRMapNet + MapTRv2: 17.0 FPS (Baseline: 19.6 FPS)
- HRMapNet + StreamMapNet: 21.1 FPS (Baseline: 22.5 FPS)
- The speed drop is acceptable, still satisfying real-time requirements.

## Highlights & Insights

1. **Simple Yet Effective Concept**: Rasterizing historical predictions into a global map and feeding it back into online perception is conceptually simple but practically powerful, boosting the mAP by 4 to 6 points.
2. **Plug-and-play**: The framework can be integrated with most existing vectorized map perception methods. The paper validates its effectiveness with two representative baselines, MapTRv2 and StreamMapNet.
3. **Extremely Low Storage Cost**: The global rasterized map only requires approximately 1 MB/km, which is orders of magnitude lower than BEV feature-based methods (e.g., NMP requires 11 GB).
4. **High Robustness**: It is robust to localization errors, showing almost unaffected performance within typical localization accuracy ranges.
5. **Great Potential for Practical Applications**: It is naturally suited for crowdsourced map perception scenarios, where multiple vehicles collectively maintain a global map.

## Limitations & Future Work

1. The paper primarily focuses on utilizing historical raster maps. However, the quality of the raster map depends on online perception accuracy. When first driving into a new area, no historical information is available, and the system degenerates to pure online perception.
2. The update strategy of the global map is relatively simple (fixed incremental/decremental values), lacking a confidence-based adaptive update mechanism.
3. The training memory overhead of the Query Initialization module reaches 65 GB at a 0.3 m resolution, demanding downsampling to control resource usage, which may lead to a loss of fine-grained information.
4. The evaluation is conducted only on nuScenes and Argoverse 2. Both datasets feature geographical overlaps between their train and validation sets, so the performance on completely unseen areas warrants further verification.
5. Map obsolescence and scene dynamics are not addressed—information in historical maps may no longer be valid due to constructions or other changes.

## Related Work & Insights

| Method | Information Source | Characteristics |
|------|----------|------|
| MapTRv2 | Single-frame images | Baseline method, no extra info |
| StreamMapNet | Short-term temporal | Query propagation + BEV fusion |
| SQD-MapNet | Short-term temporal | Stream query denoising |
| P-MapNet | SD Map (OSM) | Requires external data, limited improvement |
| NMP | Historical BEV features | Huge memory footprint (11 GB) |
| **HRMapNet** | Historical rasterized map | Low-cost, full history, plug-and-play, superior performance |

## Insights & Connections

- The idea of using rasterized maps as lightweight containers for historical information can be extended to other BEV perception tasks, such as 3D object detection and occupancy prediction.
- Crowdsourced map perception is a highly valuable direction for practical application. Multi-vehicle collaborative construction and sharing of global maps can further enhance each individual vehicle's perception capabilities.
- The design approach of Query Initialization (guiding DETR queries with prior info to narrow down the search space) is highly generalizable and can be applied to prior injection in other detection or segmentation tasks.
- The initial map experiment (achieving 83.7 mAP with the train set map) implies that if high-quality historical maps are available, online perception accuracy can leap dramatically, which holds great significance for real-world deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ (Although simple conceptually, the insight is solid; using raster maps as a low-cost historical information carrier is a smart engineering choice)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, two baselines, comprehensive ablation, and rich auxiliary experiments on robustness and initial maps)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, informative figures and tables)
- Value: ⭐⭐⭐⭐ (Direct reference value for actual deployment of autonomous driving map perception)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[CVPR 2026\] OptiMVMap: Offline Vectorized Map Construction via Optimal Multi-vehicle Perspectives](../../CVPR2026/autonomous_driving/optimvmap_offline_vectorized_map_construction_via_optimal_multi-vehicle_perspect.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2025/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)

</div>

<!-- RELATED:END -->
