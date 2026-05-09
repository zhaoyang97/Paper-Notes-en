---
title: >-
  [Paper Note] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh
description: >-
  [AAAI 2026][Multimodal VLM][Accessibility mapping] This paper adapts the Project Sidewalk accessibility annotation platform to Chandigarh, India, through customized interface labels, VLM-driven task guidance (Gemini 2.5 Flash), and a POI-centric analysis framework. Approximately 40 km of sidewalks are audited across three regions of distinct land use, identifying 1,644 locations where accessibility improvements can be made.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Accessibility mapping
  - VLM-assisted annotation
  - sidewalk accessibility
  - POI analysis
  - human-AI collaboration
date: 2026-05-08
content_hash: 55d4d6a79d5323bf
---

# Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh

**Conference**: AAAI 2026
**arXiv**: [2602.09216](https://arxiv.org/abs/2602.09216)
**Code**: None (custom deployment based on the Project Sidewalk platform)
**Area**: Multimodal VLM
**Keywords**: Accessibility mapping, VLM-assisted annotation, sidewalk accessibility, POI analysis, human-AI collaboration

## TL;DR
This paper adapts the Project Sidewalk accessibility annotation platform to Chandigarh, India, through customized interface labels, VLM-driven task guidance (Gemini 2.5 Flash), and a POI-centric analysis framework. Approximately 40 km of sidewalks are audited across three regions of distinct land use, identifying 1,644 locations where accessibility improvements can be made.

## Background & Motivation

### State of the Field
Urban accessibility is a core principle of the UN New Urban Agenda. **Project Sidewalk** is a web-based crowdsourcing platform that allows users to annotate sidewalk accessibility issues by virtually walking through Google Street View. The platform has been deployed in 44 cities worldwide, with over 10,000 users generating 1.4 million annotations covering 26,000 km of urban streets. However, existing deployments are concentrated predominantly in the United States and Europe.

### Limitations of Prior Work

**Difficulty of geographic adaptation**: Sidewalk conditions in Indian cities differ fundamentally from those in Western contexts—pedestrians frequently walk in shared lanes, informal shoulders, or discontinuous pathways; curb ramps are rare; drainage ditches, parked two-wheelers, and street vendors routinely occupy pedestrian space. Applying the American label taxonomy (e.g., "Curb Ramp") and its example images directly to Indian scenes causes annotator confusion.

**High cognitive burden for annotators**: Novice annotators struggle to determine which labels to apply when confronted with streets lacking formal sidewalks or visually ambiguous walking surfaces.

**Lack of a prioritization framework**: Given India's vast scale and population, an effective prioritization framework is needed to identify accessibility gaps—rather than treating all infrastructure equally, focus should be placed on accessibility around **key Points of Interest (POIs)**.

### Starting Point
The paper addresses these issues at two levels: (1) **the tooling level**—reducing annotation ambiguity and cognitive load through interface redesign and VLM-assisted guidance; and (2) **the analytical framework level**—providing targeted improvement recommendations through a POI-centric accessibility analysis rather than a simple exhaustive audit.

## Method

### Overall Architecture
The workflow consists of four steps: (a) selecting representative regions based on population distribution and land use; (b) extracting OSM road types and leading/trailing Google Street View panoramas for each street segment; (c) generating targeted task guidance via a VLM; and (d) having human annotators perform labeling using the adapted Project Sidewalk interface.

### Key Designs

#### 1. **Interface Redesign and Label Localization**
- **Function**: Fully adapts Project Sidewalk's label taxonomy, labels, and example images to the Indian context.
- **Core changes**:
    - **"Curb Ramp" → "Curb Style"**: The American curb ramp label presupposes standardized design, whereas India features highly diverse curb transitions (formal ramps, steps, broken curbs, drainage gaps, etc.). The label is redefined to describe how pedestrian paths transition to the carriageway.
    - **New labels added**: Parked cars, carts, drainage infrastructure, electrical boxes, and other India-specific obstacles.
    - **Labels removed**: Fire hydrants, mailboxes, recycling bins, and other elements rarely encountered in India.
    - **Severity example replacement**: Example images in all hover tooltips are replaced with Chandigarh street-view photographs.
    - **No Sidewalk label refinement**: A distinction is drawn between "no pedestrian space at all" and "space exists but is rendered unusable by clutter or encroachment."
- **Design Motivation**: Directly applying the American label system causes annotator confusion—the very concept of a "curb ramp" is inapplicable in India, where curb transitions are far more diverse than in the US.

#### 2. **VLM-Assisted Mission Guidance**
- **Function**: Provides annotators with brief, context-sensitive guidance at the start of each street segment to clarify what to attend to.
- **Mechanism**:
    - **Gemini 2.5 Flash** is used to generate guidance messages.
    - **Trigger events**: Mission start, entry into a new street segment, use of the jump function.
    - **Inputs**: OSM road type + leading and trailing Google Street View panoramas.
    - **Road-type-adaptive prompting**:
        - Arterial roads: annotators are prompted to focus on curb styles and crosswalks.
        - Residential roads: annotators are prompted to treat the road itself as the pedestrian path and to attend to obstacles and surface conditions.
        - Secondary roads: annotators are prompted to check both sides for sidewalks.
    - **Output**: A brief natural-language message displayed in a pop-up window and a status panel above the mini-map.
    - **No labels are created**: The VLM provides directional guidance only; all annotation decisions are made by humans.
- **Design Motivation**: Pedestrian environments in Indian cities are highly variable—from formal sidewalks to shared lanes. Annotators, especially novices, require immediate contextual information to reduce cognitive load. Road type is a critical signal because expected infrastructure varies substantially across road categories.

#### 3. **POI-Centric Accessibility Analysis Framework**
- **Function**: Computes multi-tier accessibility scores centered on key Points of Interest.
- **Mechanism**:
    - **POI selection**: Google Places API is used to collect POIs within a 400 m radius in each region; after deduplication, 10,128 unique POIs are obtained, categorized into 10 classes (financial, education, healthcare, public services, transportation, food & beverage, religious, utilities, commercial, social).
    - **Path extraction**: Starting from each POI, walking paths within 1 km are extracted via DFS traversal of the OSMnx road network graph.
    - **GSV coverage check**: Path segments must have at least 75% Street View coverage.
    - **Three-tier scoring**:
        - **Segment-level score (SegScore)**: $AS_{segment} = \frac{1}{1 + e^{-w_s \cdot x_a}}$, sigmoid-normalized based on annotation features and severity weights.
        - **POI-level score (POISecScore)**: Length-weighted average of all segment-level scores within 1 km of the POI.
        - **Cross-region POI score (POIScore)**: POI-count-weighted average of POI-level scores across regions.
        - Severity weights: Level 1 → 0.2, Level 2 → 0.6, Level 3 → 1.0.
- **Design Motivation**: Exhaustive audits are infeasible at the scale of a country like India; a POI-centric approach prioritizes areas of greatest impact on residents' daily lives.

### Region Selection
Three regions representing distinct land uses are selected:
- **Sector 45** (Residential): The most densely populated area in Phase II.
- **Sector 34** (Commercial): 171 commercial POIs, the highest count among three candidate commercial areas.
- **Sector 12** (Institutional): Home to PGIMER hospital, attracting a large resident and transient population.

## Key Experimental Results

### VLM Guidance Quality Evaluation

| Dimension | Mean | Std. Dev. | Min | Max | N |
|-----------|------|-----------|-----|-----|---|
| Relevance | **4.97** | 0.26 | 2 | 5 | 150 |
| Accuracy | **4.40** | 0.71 | 2 | 5 | 150 |
| Usefulness | **4.61** | 0.70 | 1 | 5 | 150 |

Mean utility score: **4.66/5**

### Annotator Agreement

| Dimension | Annotator Pair | Spearman ρ | Weighted Cohen's κ |
|-----------|---------------|-----------|-------------------|
| Relevance | R1–R3 | 1.000 | 1.00 |
| Accuracy | R2–R3 | 0.444 | 0.487 |
| Usefulness | R2–R3 | 0.445 | **0.665** |

### POI Accessibility Analysis Results

| Region | Land Use | Audited Roads (km) | POIs | Improvable Locations | Total Annotations |
|--------|----------|--------------------|------|----------------------|-------------------|
| Sector 12 | Institutional | ~13 | ~80 | ~550 | ~970 |
| Sector 34 | Commercial | ~14 | ~85 | ~530 | ~970 |
| Sector 45 | Residential | ~13 | ~65 | ~564 | ~973 |
| **Total** | — | **~40** | **~230** | **1,644** | **2,913** |

### Key Findings
1. **VLM guidance is highly effective**: Three annotators rating 50 street segments yielded a mean utility score of 4.66/5, with relevance approaching a perfect 4.97.
2. **56.4% of annotated locations require improvement**: 1,644 of 2,913 annotated locations could benefit from infrastructure improvements.
3. **Commercial areas exhibit the best overall accessibility**; educational and public service facilities rank lowest and should be prioritized.
4. **Functional accessibility outperforms general accessibility**: The institutional area (Sector 12) achieves the best healthcare facility accessibility, while other facility types (e.g., transit stops, food & beverage) score poorly, indicating that improvements are concentrated around core functions.
5. **Segment-level scores exhibit a long-tail distribution**: A large number of segments have severe issues (long negative tail), necessitating trimming and normalization.
6. Residential areas show the highest accessibility for religious, social, and commercial POIs, while other categories warrant further attention.

## Highlights & Insights
1. **VLM as a "human annotation assistant" is a novel paradigm**: Rather than replacing human annotators, the VLM provides contextual guidance prior to annotation, reducing cognitive load—an underappreciated mode of human-AI collaboration.
2. **Road type as a critical cue for VLM guidance**: Combining simple metadata (OSM road type) with visual information suffices to generate highly relevant guidance, mitigating the uncertainty inherent in purely visual understanding.
3. **POI-centric prioritization is practically valuable**: In resource-constrained developing countries, "improve accessibility around key locations first" is far more feasible than "comprehensively improve all infrastructure."
4. **Cross-cultural adaptation of label taxonomies has methodological value**: The adaptation from the US to India involved not only renaming labels but reconceptualizing the underlying framework (e.g., Curb Ramp → Curb Style).
5. **The three-tier scoring system (segment → POI → cross-region) is well-designed**: It enables localization of specific problem segments while also producing region-level comparative metrics.

## Limitations & Future Work
1. **Only 3 annotators**: The VLM guidance evaluation sample is small (3 annotators × 50 segments = 150 ratings), limiting statistical significance.
2. **Single city**: Chandigarh is one of India's best-planned cities; the generalizability of findings to more complex urban environments such as Mumbai or Delhi remains uncertain.
3. **VLM guidance quality not directly linked to annotation quality**: The perceived quality of guidance is evaluated, but annotator agreement under guidance vs. no-guidance conditions is not compared.
4. **GSV coverage dependency**: Insufficient Street View coverage in some areas limits the scope of analysis.
5. **"No Sidewalk" annotations reflect only localized absence**: They may be misinterpreted as indicating the complete absence of sidewalks along an entire street.
6. The possibility of using VLMs for direct automated annotation (rather than as a guidance tool only) is not explored.

## Related Work & Insights
- **Project Sidewalk**: The core platform, deployed in 40+ cities globally; this paper represents its first Indian deployment.
- **Gemini 2.5 Flash**: The VLM used to generate task guidance; the Flash variant (rather than Pro) was likely selected for cost and latency considerations.
- **LLM-assisted annotation literature**: The most closely related work (Bibal et al. 2025) uses LLMs to guide NLP annotation, improving annotator agreement from 0.593 to 0.84.
- Insight: **The "downgraded application" of VLMs (guidance rather than decision-making) may be more practical than full automation in human-AI collaboration scenarios**, especially for tasks requiring 3D spatial understanding.

## Rating
- Novelty: ⭐⭐⭐ (The VLM-guided annotation concept is novel, but the paper is primarily a systems/applied work)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers VLM evaluation + large-scale field annotation + multi-dimensional analysis, though annotator count is small)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; problem definition and contextual introduction are thorough)
- Value: ⭐⭐⭐⭐ (Directly applicable to urban accessibility mapping in developing countries; methodology is transferable)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](../../CVPR2026/multimodal_vlm/humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[AAAI 2026\] Towards Scalable Web Accessibility Audit with MLLMs as Copilots](towards_scalable_web_accessibility_audit_with_mllms_as_copilots.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](../../ICLR2026/multimodal_vlm/vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](../../NeurIPS2025/multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)

<!-- RELATED:END -->
