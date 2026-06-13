---
title: >-
  [Paper Note] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion
description: >-
  [CVPR 2026][3D Vision][3D city generation] This paper proposes Yo'City, a multi-agent framework that achieves user-personalized, text-driven unbounded 3D city generation through a "City–District–Grid" hierarchical planni…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D city generation"
  - "multi-agent framework"
  - "hierarchical planning"
  - "isometric image synthesis"
  - "scene graph expansion"
  - "LLM-driven"
date: 2026-05-08
content_hash: 47a12686e878b460
---

# Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion

**Conference**: CVPR 2026
**arXiv**: [2511.18734](https://arxiv.org/abs/2511.18734)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: 3D city generation, multi-agent framework, hierarchical planning, isometric image synthesis, scene graph expansion, LLM-driven

## TL;DR

This paper proposes Yo'City, a multi-agent framework that achieves user-personalized, text-driven unbounded 3D city generation through a "City–District–Grid" hierarchical planning strategy, a produce–refine–evaluate isometric image synthesis loop, and a scene graph-guided expansion mechanism. The approach comprehensively outperforms existing methods such as SynCity in semantic consistency and visual quality.

---

## Background & Motivation

**Broad demand for 3D city models**: High-quality 3D city models are essential for virtual reality, gaming, urban planning, digital twins, and robotic simulation, yet manual construction is extremely time-consuming.

**Limitations of Prior Work**: Procedural modeling and image-based reconstruction methods rely on handcrafted rules or street-view data and scale poorly; GAN/diffusion-based approaches require map or satellite data for training and cannot process user text inputs.

**Problems with SynCity**: SynCity employs an autoregressive tile-by-tile pipeline that lacks explicit hierarchical planning, resulting in spatial inconsistencies (some tiles being dense while others are sparse), blurry textures, and geometric simplification at large scales.

**Opportunities from LLM/VLM**: The world knowledge and reasoning capabilities of large language models open new possibilities for urban planning, yet agentic 3D city generation remains largely unexplored.

**Core Challenge**: Cities are open, large-scale, and highly structured spaces with far greater object diversity and spatial organization density than indoor scenes, requiring hierarchical reasoning and expansion mechanisms.

---

## Method

### Overall Architecture

Yo'City consists of four core modules spanning a **"Plan–Generate–Expand"** pipeline:

> User text $p_0$ → **Global Planner** (city-level planning) → **Local Designer** (grid-level descriptions) → **3D Generator** (isometric image → 3D assets) → **Expansion Module** (scene graph-guided expansion)

A key characteristic is **parallel generation** of all tiles, eliminating autoregressive causal dependencies, thereby avoiding error accumulation and accelerating generation.

### Problem Formulation

The urban space is divided into an $H \times W$ grid $\mathcal{T} = \{0, \ldots, H-1\} \times \{0, \ldots, W-1\}$, where each tile $(x, y)$ corresponds to a 3D scene block (e.g., a residential block). Unlike SynCity's autoregressive approach, Yo'City **simultaneously and in parallel** generates attributes for all tiles based on the global prompt $p_0$, breaking the strict causal constraint that tile $(x,y)$ must depend on the already-generated tile set $\mathcal{T}(x,y)$.

### Key Module 1: Global Planner

- **Function**: Transforms the user's abstract text description $p_0$ into a high-level city layout.
- **Three-step planning process**:
  1. **Size Estimation**: The LLM estimates the city size as an $H \times W$ rectangular grid, with each cell as the basic spatial unit.
  2. **District Planning**: Plans $N$ functional districts, generating a blueprint set $\{B_i \mid i=1,2,\ldots,N\}$, where each $B_i$ describes the district's function (e.g., "commercial center") and building types (e.g., "high-rise office buildings").
  3. **Layout Allocation**: Considering inter-district spatial relationships and adjacency constraints, distributes functional districts across the $H \times W$ grid, allowing districts to span multiple cells.
- **RAG Enhancement**: For prompts referencing real cities (e.g., "New York style"), relevant information on urban structure and zoning characteristics is retrieved from a Wikipedia corpus, distilled by GPT-4o-mini, and injected into the planning process, aligning generated results with real urban spatial logic.

### Key Module 2: Local Designer

- **Function**: Refines the coarse-grained blueprints $\{B_i\}$ from the Global Planner into grid-level text descriptions.
- **Mechanism**: The LLM conditions on blueprint $B_i$ and global prompt $p_0$ to generate detailed designs $\{d_i \mid i=1,2,\ldots,H \times W\}$ for each grid cell in the city, covering architectural style, density, landmarks, and surrounding environment.
- **Continuity Guarantee**: Joint planning is performed for all grid cells within the same functional district, ensuring intra-district spatial and stylistic consistency.
- **Design Motivation**: Compared to generating an entire city layout in one step, the coarse-to-fine strategy provides the LLM with an implicit chain of reasoning—global organization before local refinement—yielding more realistic and coherent layouts.

### Key Module 3: 3D Generator

Transforms grid descriptions $\{d_i\}$ into 3D assets in two stages:

#### Stage A: Produce–Refine–Evaluate Isometric Image Synthesis Loop

Naïve text-to-image approaches tend to produce misaligned objects and incomplete buildings. An iterative loop is designed to address this:

1. **Produce**: Generates an initial isometric image for grid $d_i$ on a predefined ground platform. The platform serves as a common anchor, ensuring consistent scale and spatial alignment across all assets.
2. **Refine**: An image editing model removes the platform and corrects geometric artifacts while enhancing visual diversity.
3. **Evaluate**: A dedicated evaluator scores the result on text-image alignment, realism, and layout rationality. If the quality threshold is not met, feedback is provided and the image is regenerated until all quality criteria are satisfied.

#### Stage B: 3D Model Conversion and Scene Assembly

- A pretrained Hunyuan3D model converts high-quality isometric images into 3D models.
- Using a parallel grid alignment pipeline, all 3D models are arranged directly according to the predefined layout from the Global Planner, **eliminating boundary inconsistencies without requiring 3D fusion**.
- Connecting elements such as roads and ground planes are added, with support for user-customized ground materials by theme (e.g., ancient / modern style).

### Key Module 4: Relationship-guided Expansion

Urban spaces follow functional district adjacency principles (residential areas near schools and commercial districts, industrial zones far from residential areas). The expansion mechanism enables unbounded city evolution:

#### Expansion Process

1. **VLM Reasoning**: Given the rendered city and district overview, the VLM generates a text description $d_{\text{new}}$ for a new expansion tile and constructs a **scene graph**—with the new tile as the central node and edges to existing functional districts encoding qualitative distance relationships (near / relatively near / far, etc.).
2. **Candidate Position Search**: Feasible candidate positions $\mathcal{X}$ are identified in the city layout via breadth-first search.
3. **Joint Distance-Semantic Optimization**: Determines the optimal placement position.

#### Optimization Objectives

**Distance-driven spatial objective** (attracting or repelling to satisfy spatial relationships):

$$L_{\text{dist}}(x) = \sum_{g \in \mathcal{G}} \gamma_{r(g)} \|x - g\|_2$$

where $\gamma_{r(g)}$ is a signed weight: positive values indicate attraction (proximity) and negative values indicate repulsion (separation).

**Semantic regularization** (ensuring the new tile is semantically compatible with its neighborhood):

$$L_{\text{sem}}(x) = -\sum_{y \in \mathcal{N}(x)} \text{Embedding\_Sim}(d_{\text{new}}, d_y)$$

Based on Sentence-BERT embedding similarity, this encourages placement at positions with high semantic compatibility.

**Joint objective**:

$$x^* = \arg\min_{x \in \mathcal{X}} \left[ L_{\text{dist}}(x) + \lambda \, L_{\text{sem}}(x) \right]$$

Once the optimal position $x^*$ is determined, the 3D Generator is invoked to produce the new tile model, completing one round of expansion. Users can iteratively interact and expand repeatedly, enabling true open-world city growth.

---

## Experimental Setup

- **Dataset**: 100 city text descriptions (30% human-written + 70% GPT-4o-generated), covering diverse styles.
- **Baselines**: Trellis (text-to-3D), Hunyuan3D API (text-to-3D), CityCraft (layout + asset retrieval), SynCity (autoregressive tile-by-tile).
- **Implementation Details**: GPT-4o as the LLM, GPT-Image-1 for image editing, Hunyuan3D API for image-to-3D.

---

## Key Experimental Results

### Table 1: City-level Quantitative Comparison (VQAScore + GPT-5/Human Win Rate)

| Method | VQAScore | Geometric Fidelity (GPT-5 / Human) | Texture Clarity (GPT-5 / Human) | Layout Coherence (GPT-5 / Human) | Overall Realism (GPT-5 / Human) |
|--------|----------|------------------------------------|---------------------------------|----------------------------------|---------------------------------|
| Trellis | 0.6189 | 6.5% / 7.0% | 4.5% / 6.0% | 6.5% / 3.5% | 9.0% / 5.0% |
| Hunyuan3D | 0.6198 | 12.0% / 7.0% | 12.5% / 9.5% | 7.0% / 5.5% | 12.0% / 6.5% |
| CityCraft | 0.5639 | 9.5% / 8.0% | 6.0% / 6.0% | 15.0% / 16.5% | 12.0% / 13.5% |
| SynCity | 0.6975 | 15.0% / 12.0% | 21.5% / 18.5% | 14.0% / 10.5% | 15.5% / 12.0% |
| **Yo'City** | **0.7151** | **85–93.5%** | **78.5–95.5%** | **85–93.5%** | **84.5–95%** |

Yo'City comprehensively outperforms all baselines across every dimension, achieving the highest VQAScore (0.7151), with win rates consistently above 78.5% against the strongest baseline, SynCity.

### Table 2: Grid-level Comparison (SynCity vs. Yo'City)

| Method | Alignment Score | Aesthetic Score |
|--------|----------------|-----------------|
| SynCity | 0.6572 | 4.95 |
| **Yo'City** | **0.6927** (+0.0355) | **5.52** (+0.57) |

Grid-level evaluation further confirms that Yo'City not only achieves better global consistency but also yields superior semantic alignment and aesthetic quality at the individual tile level.

### Ablation Study: Coarse-to-Fine Planning

| Metric | w/o reason | w/ reason |
|--------|-----------|-----------|
| VQAScore | 0.7034 | **0.7151** |
| Layout Coherence (win rate) | 27.0% | **73.0%** |
| Overall Realism (win rate) | 24.5% | **75.5%** |

Coarse-to-fine planning (Global Planner + Local Designer) yields a +46% win rate improvement in layout coherence, validating the necessity of hierarchical reasoning.

### Expansion Mechanism Stability

Across 4 expansion steps on each of 5 cities, the coefficient of variation for VQAScore is only 3.34%, indicating that semantic consistency remains stable throughout the expansion process.

---

## Highlights & Insights

- **First hierarchical agentic city generation framework**: The "City–District–Grid" three-level planning simulates the organizational logic of real cities, better reflecting urban intrinsic structure than SynCity's flat tile-by-tile approach.
- **Parallel generation overcomes the autoregressive bottleneck**: Eliminating causal dependencies between tiles prevents error accumulation while substantially accelerating generation.
- **RAG enhances realism**: Injecting real urban structural knowledge retrieved from Wikipedia into the planning process grounds instructions such as "New York style" in factual information rather than LLM hallucinations.
- **Produce–refine–evaluate closed loop**: The evaluator feedback mechanism ensures isometric image quality and resolves the spatial alignment issues inherent to naïve text-to-image generation.
- **Relationship-guided expansion supports unbounded evolution**: The scene graph combined with joint distance-semantic optimization automatically ensures that newly added areas satisfy functional district adjacency principles (e.g., shopping centers near residential areas, industry far from residences).
- **Strong personalization capability**: The framework handles fine-grained personalized descriptions such as "Harry Potter theme park" and "Silk Road."

---

## Limitations & Future Work

1. **Reliance on closed-source large models**: The core pipeline depends on GPT-4o, GPT-Image-1, and the Hunyuan3D API, making it costly and difficult to reproduce; open-source alternatives may degrade quality.
2. **Highly subjective evaluation**: Visual quality is assessed primarily through GPT-5 and human pairwise comparisons, lacking objective geometric metrics (e.g., FID, point cloud accuracy).
3. **No comparison against real city data**: Reconstruction accuracy has not been evaluated against real-world urban data sources such as Google Earth or OpenStreetMap.
4. **Risk of semantic drift during expansion**: Long-term maintenance of global stylistic consistency across many expansion rounds has not been thoroughly validated (only 4–8 steps were tested).
5. **Generation speed not reported**: End-to-end generation time for a single city is not provided; latency from multiple API calls may be substantial.
6. **Hard grid-boundary segmentation**: 3D model assembly relies on road and ground-plane infilling, with no support for cross-tile structures (e.g., large stadiums spanning two tiles).

---

## Related Work & Insights

- **SynCity**: Autoregressive tile-by-tile generation without hierarchical planning leads to spatial inconsistencies and texture degradation at scale. Yo'City's hierarchical planning and parallel generation directly address both issues.
- **CityCraft**: Generates a semantic layout and then retrieves predefined assets, limiting generalization to the asset library and performing poorly for non-modern city styles.
- **Indoor Scene LLM Generation (LayoutGPT, I-Design)**: Indoor environments are closed and controllable, whereas cities are open with high object density and category diversity; Yo'City's hierarchical decomposition is the key differentiator.
- **Hunyuan3D / Trellis**: General-purpose text-to-3D models that lack city-level layout planning and tiling capabilities, performing poorly when applied directly to city generation.
- **Insights**: This framework is generalizable to other large-scale scene generation tasks (e.g., highway networks, campus planning), and the RAG + hierarchical planning paradigm is applicable to structured generation tasks beyond the urban domain.

---

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images](from_orbit_to_ground_generative_city_photogrammetry_from_extreme_off-nadir_satel.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](../../ICCV2025/3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)

</div>

<!-- RELATED:END -->
