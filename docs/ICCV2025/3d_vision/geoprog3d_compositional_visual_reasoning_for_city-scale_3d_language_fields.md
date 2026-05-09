---
title: >-
  [Paper Note] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields
description: >-
  [ICCV 2025][3D Vision][3D language fields] This paper proposes GeoProg3D, the first visual programming framework supporting natural language interaction with city-scale high-fidelity 3D scenes. By combining a Geo-aware City-scale 3D Language Field (GCLF) with Geo-Visual APIs (GV-APIs) and an LLM reasoning engine, the framework enables compositional geospatial reasoning. GeoProg3D comprehensively outperforms existing 3D language field and VLM methods on the newly introduced GeoEval3D benchmark, which contains 952 annotated queries.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D language fields
  - city-scale scenes
  - visual programming
  - compositional reasoning
  - geographic information
date: 2026-05-08
content_hash: f2fddcb6141d1a45
---

# GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields

**Conference**: ICCV 2025
**arXiv**: [2506.23352](https://arxiv.org/abs/2506.23352)
**Code**: [snskysk/GeoProg3D](https://snskysk.github.io/GeoProg3D/)
**Area**: 3D Vision
**Keywords**: 3D language fields, city-scale scenes, visual programming, compositional reasoning, geographic information

## TL;DR

This paper proposes GeoProg3D, the first visual programming framework supporting natural language interaction with city-scale high-fidelity 3D scenes. By combining a Geo-aware City-scale 3D Language Field (GCLF) with Geo-Visual APIs (GV-APIs) and an LLM reasoning engine, the framework enables compositional geospatial reasoning. GeoProg3D comprehensively outperforms existing 3D language field and VLM methods on the newly introduced GeoEval3D benchmark, which contains 952 annotated queries.

## Background & Motivation

Methods such as **3D Language Fields** (e.g., LangSplat, LERF) have enabled natural language interaction with 3D scenes, but face two fundamental challenges when scaling to city-level environments:

**Lack of Scalability**: Existing methods (LangSplat, LERF, etc.) are primarily designed for indoor scenes. Applying them directly to large-scale urban data exceeding $1\ \text{km}^2$ encounters severe memory and computation bottlenecks. For instance, LERF and LangSplat run out of memory (OOM) on UrbanScene3D ($5 \times 10^6\ \text{m}^2$).

**Limited Task Diversity**: Current 3D language fields are largely restricted to word-level object grounding and cannot address the diverse demands of urban applications, such as spatial relation interpretation, object counting and size quantification, and landmark recognition. Queries in urban scenes are inherently compositional (e.g., "red shops within 100 meters of Chase Bank"), requiring multi-step reasoning.

**Core Insight**: Integrate city-scale 3D language fields with a visual programming framework—using hierarchical 3D Gaussians for efficient large-scale scene reconstruction and language embedding, providing rich operation interfaces through geographic APIs, and having an LLM dynamically compose these APIs to handle diverse query requirements.

## Method

### Overall Architecture

GeoProg3D consists of two core components and one reasoning engine:

1. **GCLF (Geo-aware City-scale 3D Language Field)**: Built upon tree-structured hierarchical 3D Gaussians, embedding CLIP language features and geographic coordinate information.
2. **GV-APIs (Geo-Visual APIs)**: Nine specialized image and geographic processing functions.
3. **LLM Reasoning Engine**: Generates Python programs via in-context learning (ICL) to dynamically compose GV-APIs and operate on the GCLF.

The query processing pipeline consists of two steps:
- **Step 1 – Program Generation**: $z = \Pi(q, R)$, where LLM $\Pi$ generates a Python program $z$ given query $q$ and in-context examples $R$.
- **Step 2 – Program Execution**: $a = \Lambda(z; \mathcal{T})$, where Python engine $\Lambda$ executes the program on GCLF $\mathcal{T}$ to obtain answer $a$.

### Key Design 1: GCLF — City-Scale 3D Language Field

**Scene Representation**: The method adopts tree-structured 3D Gaussians [Ren et al.], learning hierarchical nesting relationships among Gaussians. During rendering, the level of detail is dynamically selected—Gaussians at levels where their image-space diameter falls below one pixel are rendered—representing distant scenes at coarse granularity to balance quality and efficiency.

| Scene | LangSplat # Gaussians | GCLF # Gaussians | LangSplat Speed (ms) | GCLF Speed (ms) |
|---|---|---|---|---|
| Center Blvd | 37,212 | 1,136,015 | 2.73 | 14.46 |
| UrbanScene3D | OOM | 37,813,418 | OOM | 20.83 |

Although GCLF requires tens of times more Gaussians than LangSplat, rendering speed is only a few times slower (still real-time), and GCLF operates normally on UrbanScene3D where LangSplat runs OOM.

**Language Alignment**: Following LangSplat, a scene-specific autoencoder compresses CLIP features before embedding them into the Gaussians. At inference time, the cosine similarity between the CLIP text feature $T(q)$ and the decoded language embedding $D(\hat{l}(v))$ is computed per pixel to localize the target.

**Georeferencing**: A semi-automatic procedure aligns 3D Gaussian coordinates to real-world coordinates from OpenStreetMap—four top-down views of small regions are rendered, 20+ landmark points are manually selected, and the coordinate transformation is computed using scikit-image. After alignment, the system supports landmark name queries and real-world metric measurements (distances and heights in meters).

### Key Design 2: GV-APIs — Nine Geo-Visual APIs

| # | Function | Function |
|---|---|---|
| 1 | `GetLandmarkSeg(query)` | Retrieves a segmentation region by landmark name |
| 2 | `GetStructureSeg(query, area)` | Retrieves segmentation by structure name (e.g., "bridge") |
| 3 | `SegAround(area, distance)` | Retrieves the surrounding region within a specified distance |
| 4 | `SegDirection(area, direction)` | Retrieves the region in a specified direction |
| 5 | `SegBetween(seg1, seg2)` | Retrieves the region between two areas |
| 6 | `LargestSeg(segs)` | Retrieves the largest connected segment via clustering |
| 7 | `MeasureDist(from, to)` | Computes real-world distance in meters |
| 8 | `MeasureHeight(area)` | Computes real-world height in meters |
| 9 | `GetObjectSeg(query, area)` | Runs GroundingDINO detection within a specified area |

APIs 1–6 progressively narrow the region of interest within the large urban space of GCLF; APIs 7–8 leverage georeferencing for real-world metric measurements; API 9 enables fine-grained detection by running visual foundation models on GCLF-rendered images. All APIs operate on the trained GCLF without additional training.

### Visual Programming

The method uses GPT-3.5 (gpt-3.5-turbo-instruct) as the LLM $\Pi$. Only 10–15 in-context examples are required for the LLM to effectively use the GV-APIs, with a program success rate exceeding 90%. The LLM is capable of generating novel structured programs not present in the examples, demonstrating structure-level generalization.

**Example**: Query "red-letter billboard within 100 meters of The View" → generates a three-step program: (1) `GetLandmarkSeg("The View")` to localize the landmark → (2) `SegAround(area, 100)` to retrieve the surrounding region → (3) `GetStructureSeg("Red-letter billboard", area)` to retrieve the target.

## Key Experimental Results

### Main Results 1: Grounding Task (GRD)

| Scene | Area (m²) | LSeg | LERF | LangSplat | GCLF | GeoProg3D |
|---|---|---|---|---|---|---|
| GoogleEarth | 2.4×10⁵ | 0.96% | 11.44% | 14.15% | 20.09% | **45.20%** |
| UrbanScene3D | 5.0×10⁶ | 4.65% | OOM | OOM | 6.98% | **30.23%** |

GeoProg3D substantially outperforms baselines in grounding accuracy: 45.2% vs. LangSplat's 14.15% on GoogleEarth (3× improvement); on UrbanScene3D, LERF and LangSplat run OOM while GeoProg3D achieves 30.23%.

### Main Results 2: Multi-Task Comparison (SPR/CMP/CNT/MES)

| Method | SPR Acc↑ | CMP Acc↑ | CNT MAE↓ | MES-H MAE(m)↓ | MES-D MAE(m)↓ |
|---|---|---|---|---|---|
| GPT-4o Vision | 24.77 | 2.63 | 3.02 | 158.16 | 195.29 |
| Llama-3.2 Vision | 54.84 | 28.49 | 2.54 | 88.06 | 133.20 |
| InternVL2.5-8B | 54.27 | 26.95 | 2.79 | 51.30 | 157.14 |
| GeoChat | 57.23 | 41.99 | 2.89 | 84.74 | 89.34 |
| TEOChat | 59.04 | 48.11 | 2.84 | 150.39 | 198.89 |
| **GeoProg3D** | **64.00** | **59.73** | **2.00** | **45.24** | **49.28** |

GeoProg3D comprehensively outperforms nine strong baselines (including GPT-4o Vision and GeoChat) across all five tasks. Notably, on distance measurement (MES-D), the error is only 49.28 m, far below GPT-4o's 195.29 m (4× improvement).

### GeoEval3D Benchmark Statistics

- 952 manually annotated query-answer pairs covering five tasks
- Scene area exceeds $3\ \text{km}^2$, spanning New York and Shenzhen
- Query complexity substantially exceeds prior datasets (10× more words on average)

## Highlights & Insights

1. **First city-scale 3D compositional reasoning framework**: Extends the visual programming paradigm from 2D images to city-scale 3D scenes.
2. **Elegance of modular design**: GCLF handles scene representation and localization, GV-APIs provide operation interfaces, and the LLM performs reasoning—each component has clearly defined responsibilities and can be independently upgraded.
3. **Critical value of georeferencing**: Aligning Gaussian coordinates with OpenStreetMap enables the system to return metric measurements in real-world units (meters).
4. Only 10–15 examples are needed for the LLM to effectively use the APIs, demonstrating the sample efficiency of visual programming.

## Limitations & Future Work

1. **Semi-automatic georeferencing requires manual annotation**: Each scene requires manually selecting 20+ landmark points for coordinate alignment, limiting fully automated deployment.
2. **High storage overhead of GCLF**: Tens of times more Gaussians are required compared to LangSplat (37M vs. 37K).
3. **LLM dependency**: Program generation quality is bounded by LLM capability (currently GPT-3.5), and complex compositional reasoning may fail.
4. The handling of dynamic changes in urban scenes (construction, demolition, temporal variation) is not discussed.

## Related Work & Insights

- **City-scale 3D reconstruction**: NeRF-based (Block-NeRF, etc.), 3DGS-based (CityGaussian, Octree-GS)
- **3D language fields**: LERF (NeRF+CLIP), LangSplat (3DGS+CLIP+SAM), LEGaussians
- **Visual programming**: ViPer/ViperGPT (2D images), CodeVQA, SayPlan (3D point clouds, indoor only)
- **Geo-aware VLMs**: GeoChat, TEOChat, LHRS-BOT

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 5 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] CF³: Compact and Fast 3D Feature Fields](cf3_compact_and_fast_3d_feature_fields.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)

<!-- RELATED:END -->
