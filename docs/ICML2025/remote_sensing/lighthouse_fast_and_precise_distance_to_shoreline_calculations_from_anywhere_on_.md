---
title: >-
  [Paper Note] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth
description: >-
  [ICML 2025][Remote Sensing][Shoreline distance calculation] This work introduces Lighthouse, a global shoreline dataset with a 10-meter resolution and a millisecond-level query library. By fusing ESA WorldCover and OpenStreetMap data, and combining a hierarchical BallTree with spherical Voronoi indexing, it enables real-time shoreline distance queries requiring only 1 CPU and 2GB RAM, improving accuracy by over 100 times compared to existing datasets.
tags:
  - "ICML 2025"
  - "Remote Sensing"
  - "Shoreline distance calculation"
  - "Spherical Voronoi tessellation"
  - "BallTree"
  - "High-resolution land cover"
  - "Real-time geographic query"
date: 2026-05-08
content_hash: 727179920ca63bf5
---

# LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth

**Conference**: ICML 2025  
**arXiv**: [2506.18842](https://arxiv.org/abs/2506.18842)  
**Code**: [github.com/allenai/lighthouse](https://github.com/allenai/lighthouse) (Apache 2.0)  
**Area**: Remote Sensing  
**Keywords**: Shoreline distance calculation, Spherical Voronoi tessellation, BallTree, High-resolution land cover, Real-time geographic query

## TL;DR

This work introduces Lighthouse, a global shoreline dataset with a 10-meter resolution and a millisecond-level query library. By fusing ESA WorldCover and OpenStreetMap data, and combining a hierarchical BallTree with spherical Voronoi indexing, it enables real-time shoreline distance queries requiring only 1 CPU and 2GB RAM, improving accuracy by over 100 times compared to existing datasets.

## Background & Motivation

Distance-to-shoreline data is critical for tasks such as coastal erosion monitoring, infrastructure planning, habitat change tracking, and object detection/segmentation in satellite imagery. Existing global shoreline datasets suffer from extremely low resolutions (typically 1-4 km), failing to meet high-precision requirements:

- **GSHHG**: Resolution of approximately 1855 meters, covering only the open ocean.
- **NASA OBPG**: Resolution of 4000 meters (with a 1 km interpolated version), covering only the open ocean.
- **ArcGIS/ESRI**: Resolution of 200 km, covering only land.

The utility of these coarse-resolution datasets is highly limited in coastal and inland areas. Furthermore, even with high-resolution shoreline data, achieving fast queries globally at a 10-meter level of precision poses a massive computational challenge—a simple precomputation scheme would require approximately 100 TB of RAM, which is entirely impractical.

## Method

### Overall Architecture

The design of Lighthouse is divided into two phases: **offline data construction** and **online real-time query**.

**Offline Phase**:
1. Fuse ESA WorldCover V2 (a 10-meter resolution satellite-derived land cover map) and OpenStreetMap crowd-sourced annotation data.
2. Divide the globe into $1^\circ \times 1^\circ$ tiles, extract shoreline points for each tile, and construct a BallTree index.
3. Perform spherical Voronoi tessellation on global shoreline points to locate the nearest shoreline for query points in the open ocean (outside land tiles).

**Online Phase**:
1. Determine the tile that contains the query point based on its coordinates, or use the Voronoi index to find the nearest tile.
2. Load the corresponding BallTree and query the nearest shoreline point using the Haversine distance metric.
3. Query the land cover class (land/water) of the point using the h5 file.

### Key Designs

#### 1. Data Fusion Strategy

ESA WorldCover V2 is currently the highest-precision global 10-meter land cover map for permanent water bodies (based on Sentinel-2), but it has critical omissions:
- Hundreds of islands in **Micronesia** are uncovered.
- **Antarctica** data was unavailable at its release.
- Parts of northern Greenland, Hawaii, and some South Atlantic islands are missing.

Solution: Supplement the missing regions using crowd-sourced shoreline annotations from OpenStreetMap. While OSM's global resolution is non-uniform, it covers all islands and includes the latest annotations for Antarctica. Fusing the two yields a complete global shoreline map.

#### 2. Shoreline Point Extraction (Algorithm 1)

The following pipeline is performed for each $1^\circ \times 1^\circ$ tile:
1. **Binarization**: Convert land cover labels into a binary water vs. non-water mask.
2. **Sobel Edge Detection**: Extract pixels along the water-land boundary.
3. **BallTree Construction**: Build a BallTree over the boundary coordinates using the Haversine metric.
4. **Uncompressed Storage**: Sacrifice a small amount of disk space to achieve the lowest possible read latency.

The Haversine metric is selected over the more precise Vincenty formula because the latter has significantly higher computational complexity, whereas Haversine provides sufficient accuracy for practical applications.

#### 3. Spherical Voronoi Tessellation

For query points located in the open ocean (not within any land tile), the nearest shoreline tile must be determined. Since directly traversing all tiles is infeasible, a spherical Voronoi tessellation of global shoreline points is precalculated.

Downsampling strategy with key constraints:
- Because the time complexity of Voronoi construction is $O(n^2)$, it is impossible to include all shoreline points.
- **Constraint 1**: At least one representative point is kept for each line segment in the original dataset.
- **Constraint 2**: The distance between connected points does not exceed a 1 km threshold.

This ensures that no critical islands or coastal structures are omitted.

#### 4. Single-point Query in h5 Format

Land cover data is stored in the h5 format rather than standard GeoTIFF. The key advantage is the ability to query the class label of a single pixel without loading the entire tile into memory, thereby drastically reducing memory over-head.

### Loss & Training

This work does not involve deep learning training; the core focuses on data engineering and algorithmic optimization. Key engineering decisions include:

- **Latency Priority**: All design choices primary target minimizing query latency (e.g., uncompressed BallTrees, single-point h5 queries).
- **Tile Caching**: Maintaining a tile cache during online queries to avoid redundant loading.
- **Batch Query Optimization**: Supporting batch query modes with vectorized computation and cache reuse.

## Key Experimental Results

### Main Results

| Dataset | Resolution (m) | Coverage | Open Source | Remarks |
|--------|-----------|---------|------|------|
| **Lighthouse (Ours)** | **~10** | **Global** | **Yes** | Includes inland water bodies |
| GSHHG | 1855 | Open ocean | Yes | 1-arc-minute resolution |
| NASA OBPG | 4000 | Open ocean | Yes | Also has 1km interpolated version |
| ArcGIS/ESRI | 200000 | Land only | Yes | Land to open ocean |

Resolution improvement: Lighthouse achieves an **~185x** resolution improvement compared to the previous best dataset, GSHHG.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| ESA WorldCover only | Incomplete coverage | Micronesia, Antarctica, etc., are missing |
| OpenStreetMap only | Coarse resolution | Antarctica median segment spacing of 35 meters |
| ESA + OSM Fusion | 10m global coverage | Complementary advantages, complete coverage |
| Compressed BallTree | Increased latency | Small disk savings, high latency penalty |
| Uncompressed BallTree | Millisecond-level latency | Scheme adopted in ours |
| GeoTIFF format | Requires loading entire tile | High memory overhead |
| h5 format | Single-pixel query | Scheme adopted in ours |

### Key Findings

1. **Highly Resource-Efficient**: Realizes millisecond-level online inference with only 1 CPU and 2 GB of RAM.
2. **Single-Query Latency Well Below 10ms**: Batch queries are further accelerated via caching and vectorization.
3. **Antarctica Median Label Segment Spacing of 35m in OSM**: Serves as the worst-case lower bound of resolution.
4. **Scalability**: The proposed method can scale to the highest resolutions of commercial satellite imagery (10 cm as of June 2025).
5. **Production Validation**: Already deployed and running in Skylight Global (a satellite-based vessel detection platform).

## Highlights & Insights

1. **Precise Problem Selection**: Calculating the distance to the coastline appears simple, but achieving global real-time queries at a 10m resolution is a non-trivial system design problem.
2. **Hierarchical Search Architecture**: The two-tier design of BallTree (local precise query) + Voronoi (global routing) elegantly resolves the contradiction between high resolution and high cost.
3. **Complementary Data Fusion**: The high precision of WorldCover combined with the high coverage of OSM yields mutually beneficial advantages.
4. **Engineering-Oriented Design Philosophy**: Consistently prioritizing latency over storage space, such as utilizing uncompressed BallTrees and h5 single-pixel queries.
5. **Echoing Mandelbrot's Fractal Geometry**: The paper references Benoit Mandelbrot's classic thesis on the infinite complexity of coastlines, highlighting the error amplification issue brought by high resolution.

## Limitations & Future Work

1. **Non-Uniform Resolution**: The ESA portion is 10m, but the OSM-supplemented regions feature inconsistent resolutions (e.g., ~35m in Antarctica).
2. **Temporal Dynamics**: Shorelines continuously change due to sea-level rise, glacier retreat, etc.; the dataset is only a snapshot of a specific point in time.
3. **Ambiguous Shoreline Definitions**: Inherent ambiguity exists in the boundary definitions of complex terrains such as cliffs, sandy beaches, harbors, and wetlands.
4. **Annotation Errors**: Mixes misclassifications from CV models with human errors in crowd-sourced annotations.
5. **Haversine Approximation**: Haversine is less accurate than Vincenty over long distances, but is deemed sufficient for practical applications in this work.

## Related Work & Insights

- **ESA WorldCover V2** (Zanaga et al., 2022): Sentinel-2-based global 10m land cover map, serving as the primary data source of this work.
- **Dynamic World** (Brown et al., 2022): Google's near-real-time global 10m land cover map; not chosen because its precision on permanent water bodies is inferior to WorldCover.
- **NOAA C-CAP**: 1m resolution land cover map for the conterminous United States; high-precision but covers only a specific region.
- **Fractal Geometry** (Mandelbrot, 1982): Discussions on the infinite length of coastlines, indicating the inherent limitations of high-resolution data.
- **Insights for Remote Sensing**: Combining high-precision land cover data with efficient spatial indexing can provide a reference paradigm for real-time global queries of other geographic features (e.g., rivers, road boundaries).

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 7 | The methodology is straightforward, but the problem definition and system design are highly valuable |
| Practicality | 9 | Already deployed in production; open-sourced data and code |
| Technical Depth | 6 | Core focus is on data engineering with no deep learning innovation |
| Writing Quality | 8 | Clear, concise, humorous (library name abbreviation), and highly practical |
| Reproducibility | 10 | Fully open-sourced code and data |
| **Overall** | **7.5** | A typical work epitomizing "doing the right thing is more important than doing the hard thing" |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](../../ICLR2026/remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)
- [\[ICML 2025\] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data](high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[CVPR 2026\] Local Precise Refinement: A Dual-Gated Mixture-of-Experts for Enhancing Foundation Model Generalization against Spectral Shifts](../../CVPR2026/remote_sensing/local_precise_refinement_a_dual-gated_mixture-of-experts_for_enhancing_foundatio.md)
- [\[CVPR 2025\] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](../../CVPR2025/remote_sensing/earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)

</div>

<!-- RELATED:END -->
