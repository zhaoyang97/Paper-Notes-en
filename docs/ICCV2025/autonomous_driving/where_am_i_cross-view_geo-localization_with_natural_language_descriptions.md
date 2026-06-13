---
title: >-
  [Paper Note] Where am I? Cross-View Geo-localization with Natural Language Descriptions
description: >-
  [ICCV 2025][Autonomous Driving][Cross-View Geo-localization] This paper introduces a novel task of cross-view geo-localization via natural language descriptions, constructs the CVG-Text multimodal dataset covering 30…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Cross-View Geo-localization"
  - "Natural Language"
  - "Text-to-Image Retrieval"
  - "Satellite"
  - "OSM"
  - "LMM"
  - "Explainable Retrieval"
date: 2026-05-08
content_hash: bc79bac523cef9ef
---

# Where am I? Cross-View Geo-localization with Natural Language Descriptions

**Conference**: ICCV 2025
**arXiv**: [2412.17007](https://arxiv.org/abs/2412.17007)  
**Code**: [yejy53.github.io/CVG-Text](https://yejy53.github.io/CVG-Text/)  
**Area**: Autonomous Driving / Cross-View Geo-localization
**Keywords**: Cross-View Geo-localization, Natural Language, Text-to-Image Retrieval, Satellite, OSM, LMM, Explainable Retrieval

## TL;DR

This paper introduces a novel task of cross-view geo-localization via natural language descriptions, constructs the CVG-Text multimodal dataset covering 30,000+ coordinates across 3 cities (street-view + satellite + OSM + text), and proposes CrossText2Loc — a method employing Extended Positional Embedding for long-text handling and an Explainable Retrieval Module for localization rationale, achieving over 10% improvement in Top-1 Recall.

## Background & Motivation

### Problem Scenario

In GPS-denied environments (urban canyon occlusion, indoor-outdoor transitions, emergency calls, etc.), users need to determine their location by describing the surrounding environment in natural language. Examples include:
- A taxi passenger verbally communicating their location to the driver
- A pedestrian describing surroundings during an emergency call for rescue localization

### Limitations of Prior Work

**Cross-view localization focuses on image matching**: Methods such as Sample4G and SAFA concentrate on street-view-to-satellite image retrieval, but in practice users may only be able to provide textual descriptions.

**Text-based localization limited to point clouds**: Text2Pose and Text2Loc perform text-based localization in 3D point clouds, but point cloud acquisition is costly, storage overhead is large, and global-scale deployment is impractical.

**Satellite/OSM data are more practical**: Satellite imagery and OpenStreetMap offer global coverage at low storage cost, yet text-to-satellite/OSM cross-view retrieval has not been previously studied.

**Insufficient handling of long text**: Scene descriptions are typically long (averaging 126 tokens), whereas models such as CLIP enforce a maximum sequence length of 77 tokens, truncating critical information.

## Method

### CVG-Text Dataset Construction

#### Data Collection
- Covers 3 cities: New York (urban), Brisbane (suburban), and Tokyo (urban)
- 30,000+ coordinate points, each containing:
    - Panoramic street-view images (2048×1024) and single-view street-view images
    - Satellite images (512×512, zoom level 20, resolution ~0.12 m)
    - OSM raster tiles (512×512, retaining POI annotations)

#### GPT-4o-Driven Text Generation

A progressive scene analysis strategy is adopted:

1. **OCR Preprocessing**: PaddleOCR extracts textual information from street-view images (shop names, bus stop signs, etc.), helping GPT accurately capture key localization cues and reducing hallucinations.
2. **Open-World Segmentation**: Semantic segmentation is applied to street-view images to provide positional and semantic details; OCR results from moving objects (e.g., vehicles) are filtered out.
3. **Systematic Prompting**: GPT-4o is guided to progressively describe scenes in the order of "road features → building landmarks → overall environment," using simple directional terms such as front, back, left, and right.
4. **Quality Control**: Format filtering → GPT self-review → expert human annotation (20% of samples, 10 annotators × 100 hours, pass rate 77.6%).

Text statistics: average length 126 tokens, type-token ratio (TTR) 0.76, low inter-text similarity (0.17), indicating high quality.

### CrossText2Loc Model

#### Image–Text Contrastive Learning

A dual-stream architecture consisting of a text encoder and a visual encoder aligns cross-domain features via contrastive learning:

$$L_{itc} = \sum_{i=1}^n \sum_{j=1}^n -\log\frac{\exp(\text{sim}(v_i, t_j)/\tau)}{\sum_{k=1}^n \exp(\text{sim}(v_i, t_k)/\tau)}$$

where $\tau$ is a learnable temperature parameter.

#### Extended Positional Embedding (EPE)

Scene descriptions average 126 tokens, yet CLIP is limited to 77 tokens. Positional embeddings are extended to $N=300$ tokens via linear interpolation:

$$P^*(x) = (1-(x-\lfloor x\rfloor)) \cdot P(\lfloor x\rfloor) + (x-\lfloor x\rfloor) \cdot P(\lceil x\rceil)$$

Unlike LongCLIP, since GPT-generated text does not contain a salient short title at the beginning, **full-sequence interpolation** is adopted rather than knowledge-preserving stretching.

#### Explainable Retrieval Module (ERM)

An optional inference-time module that enhances retrieval interpretability and trustworthiness:

1. **Attention Heatmap Generation**: Non-negative gradient contributions are iteratively accumulated from starting layer $s$ to output layer $L$:

$$R^{(l)} = R^{(l-1)} + \frac{1}{H}\sum_{h=1}^H \max(0, \nabla A_h^{(l)} \odot A_h^{(l)}) R^{(l-1)}$$

2. **LMM Explanation**: Text and image heatmaps are fed to GPT-4o, which performs key-cue analysis → comparative reasoning → retrieval rationale and confidence output.
3. **Confidence-Based Re-ranking**: ERM confidence scores are normalized and summed with similarity scores to re-rank the Top-5 results.

## Key Experimental Results

### Main Results: Cross-View Text Retrieval Localization

| Method | NYC-Sat R@1 | NYC-OSM R@1 | Brisbane-Sat R@1 | Brisbane-OSM R@1 | Tokyo-Sat R@1 | Tokyo-OSM R@1 |
|--------|-------------|-------------|------------------|------------------|---------------|---------------|
| CLIP-L/14 | 35.08 | 31.50 | 34.08 | 32.50 | 28.08 | 21.00 |
| SigLIP-SO400M | 33.50 | 27.75 | 34.25 | 29.75 | 28.42 | 17.50 |
| BLIP | 34.58 | **52.92** | 34.50 | 43.00 | 29.75 | 30.67 |
| **Ours (w/o ERM)** | 46.25 | 59.08 | 43.58 | 46.08 | 36.83 | 34.33 |
| **Ours (w/ ERM)** | **50.33** | **62.33** | **47.58** | **48.75** | **41.75** | **36.92** |

**Key Findings**:
- CrossText2Loc outperforms the strongest baseline BLIP by 15.75% on satellite retrieval (New York) and 9.41% on OSM retrieval.
- ERM re-ranking further improves R@1 by 4–5%, simulating user decision-making based on provided rationales.
- OSM retrieval performs best in New York (rich POI data such as bus stops and shop names) and worst in Tokyo (insufficient CLIP pretraining on Japanese text).

### Ablation Study: EPE Module

| Method | Sat R@1 | OSM R@1 |
|--------|---------|---------|
| CLIP | 35.08 | 31.50 |
| **CLIP + EPE** | **46.25** | **59.08** |
| SigLIP | 19.67 | 20.17 |
| **SigLIP + EPE** | **29.50** | **45.25** |

EPE yields significant improvements on both encoders, with OSM retrieval showing the most pronounced gain (+27.6%), demonstrating that fine-grained details in long descriptions are critical for POI matching.

### Ablation Study: Text Generation Quality

| Text Source | Len | TTR | Simi. | R@1-OSM | R@1-Sat |
|------------|-----|-----|-------|---------|---------|
| Direct GPT generation | 108 | 0.74 | 0.22 | 25.17 | 38.00 |
| **CVG-Text (OCR + Seg. + Chain Prompting)** | **126** | **0.76** | **0.17** | **59.08** | **46.25** |

OCR-assisted precise capture of street-view text → reduced GPT hallucinations → 33.9% improvement in OSM retrieval.

### Text Augmentation for Cross-View Retrieval

| Query Modality | OSM R@1 | Sat R@1 |
|---------------|---------|---------|
| Street-view image only (Sample4G) | 27.10 | 91.70 |
| Text only | 59.08 | 46.25 |
| **Image + Text fusion** | **67.30** | **98.40** |

The text branch provides complementary information to conventional cross-view retrieval, improving OSM retrieval by 40.2%.

## Highlights & Insights

1. **Novel Task Definition**: This is the first work to formulate cross-view geo-localization via natural language descriptions, filling the gap in text-to-satellite/OSM retrieval research.
2. **High-Quality Dataset**: CVG-Text employs a progressive pipeline of OCR + segmentation + GPT-4o, generating scene descriptions of substantially higher quality than direct GPT generation.
3. **Long-Text Handling**: EPE extends positional embeddings via simple linear interpolation, yet achieves a 27.6% gain on OSM retrieval, confirming that fine-grained details in scene descriptions must not be discarded.
4. **Explainable Retrieval**: ERM provides not only similarity scores but also natural language retrieval rationales and confidence estimates, simulating the decision-making process in real-world applications.
5. **Clear Practical Value**: Text retrieval complements image retrieval; their fusion achieves 98.4% R@1 on satellite retrieval.

## Limitations & Future Work

1. Textual descriptions are generated by GPT-4o, which may diverge from real users' description habits (typically shorter and more ambiguous).
2. Coverage is limited to 3 cities, with insufficient geographic diversity and limited consideration of regional street-view style variation.
3. ERM relies on GPT-4o for inference-time explanation, introducing additional computational cost and latency.
4. Degraded performance in Tokyo exposes CLIP's limitations in non-English scenarios.
5. The retrieval scope is $M=100$ (approximately 10 km²); scalability to large-area retrieval remains to be validated.

## Related Work & Insights

- **Cross-View Geo-localization**: CVUSA, VIGOR, Sample4G — image-to-image retrieval
- **Vision-Language Navigation and Localization**: Text2Pose, Text2Loc — text-based localization in point clouds, with high acquisition cost
- **LMM-Based Data Synthesis**: LatteCLIP — leveraging LMMs to synthesize text for unsupervised CLIP fine-tuning
- **Multimodal Alignment**: CLIP, LongCLIP — text-image contrastive learning with sequence length limitations

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Pioneering definition of text-to-satellite/OSM cross-view geo-localization task)
- Technical Depth: ⭐⭐⭐⭐ (EPE is simple yet effective; ERM enhances interpretability; overall methodology is engineering-oriented)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-city, multi-source, extensive ablations; lacks comparison with other text-based localization methods)
- Practical Value: ⭐⭐⭐⭐⭐ (Clear application scenarios including emergency localization and pedestrian navigation)
- Overall Recommendation: ⭐⭐⭐⭐ (A complete contribution introducing a new task and dataset with the potential to open a new research direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](../../CVPR2026/autonomous_driving/vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[ICCV 2025\] Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations](beyond_one_shot_beyond_one_perspective_cross-view_and_long-horizon_distillation_.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](../../CVPR2026/autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)

</div>

<!-- RELATED:END -->
