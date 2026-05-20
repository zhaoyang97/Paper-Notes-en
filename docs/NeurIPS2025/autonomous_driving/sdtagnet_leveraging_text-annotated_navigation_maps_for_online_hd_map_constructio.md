---
title: >-
  [Paper Note] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction
description: >-
  [NeurIPS 2025][Autonomous Driving][HD map] This paper proposes SDTagNet, the first method to encode OpenStreetMap text annotations (road names, lane counts, one-way indicators…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "HD map"
  - "OpenStreetMap"
  - "NLP tag embedding"
  - "graph transformer"
  - "SD map prior"
date: 2026-05-08
content_hash: 5307c1a145c34b9f
---

# SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction

**Conference**: NeurIPS 2025
**arXiv**: [2506.08997](https://arxiv.org/abs/2506.08997)  
**Code**: [GitHub](https://github.com/immel-f/SDTagNet)  
**Area**: Autonomous Driving / Online HD Map Construction
**Keywords**: HD map, OpenStreetMap, NLP tag embedding, graph transformer, SD map prior

## TL;DR
This paper proposes SDTagNet, the first method to encode OpenStreetMap text annotations (road names, lane counts, one-way indicators, etc.) via BERT and to unify all SD map elements (points, polylines, and relations) through a point-level graph Transformer. On long-range HD map construction, SDTagNet achieves +5.9 mAP (+45%) over prior-free baselines and +3.2 mAP (+20%) over existing SD map prior methods.

## Background & Motivation
**Background**: Online HD map construction is a critical task in autonomous driving. Methods such as MapTR and MapTRv2 predict vectorized map elements (lane dividers, boundaries, pedestrian crossings, etc.) in real time from onboard sensor data.

**Limitations of Prior Work**: The perception range of onboard sensors is limited—particularly at long range—and maintaining HD maps incurs extremely high costs. Existing approaches incorporate SD maps (e.g., OpenStreetMap) as priors, yet suffer from two key deficiencies:
   - **Underutilization of information**: Only polyline geometry and a manually selected set of 7 semantic tag categories are exploited, while OSM's rich textual annotations (road names, lane counts, one-way markers, speed limits, etc.), point features (traffic lights, bus stops), and relation elements are ignored.
   - **Coarse encoding granularity**: Methods such as SMERF encode at the polyline level (one token per polyline), which misaligns with the per-point queries used by downstream detectors.

**Key Challenge**: OSM's global database contains approximately 100,000 distinct keys and 168 million values. This unstructured text cannot be handled by manual feature engineering, yet the information is highly relevant to map construction (e.g., "oneway: yes" directly affects lane topology).

**Core Idea**: Apply an NLP encoder to SD map text annotations and employ a point-level graph Transformer to uniformly encode all element types.

## Method

### Overall Architecture
SDTagNet is a plug-and-play SD map prior encoding module integrated into the MapTRv2 backbone. The pipeline proceeds as follows: raw OSM data → NLP Tag Embedding (BERT encodes text annotations) → SD Map Encoder (graph Transformer fusing geometric, semantic, and relational information) → Map Decoder (via cross-attention).

### Key Designs

1. **NLP Tag Embedding Module**

    - **Function**: Encodes the key-value text annotations of OSM elements (e.g., "highway: residential", "lanes: 2", "oneway: yes") into 144-dimensional embedding vectors.
    - **Mechanism**: A BERT architecture is adopted, using the [CLS] token as the embedding of the entire tagset. Because SD map text differs substantially from natural language (it consists of keyword lists rather than complete sentences, and small variations such as "lanes:2" vs. "lanes:3" should yield markedly different embeddings), the encoder is trained from scratch via self-supervised contrastive pre-training on global OSM data.
    - **Loss & Training**: A custom contrastive objective based on Multiple Negatives Ranking Loss is employed. Positive pairs are tagsets that share the same semantic labels but differ in irrelevant tags (e.g., national survey reference numbers); negatives are other unique tagsets. Batch size is 5,120; 20 positive pairs are sampled per unique tagset; training runs for 4 epochs.
    - **Design Motivation**: Compared to manual feature engineering, NLP encoding handles open vocabularies without requiring a predefined category taxonomy, and BERT is lightweight enough for real-time deployment.

2. **Point-Level SD Map Encoder**

    - **Function**: Elevates SD map element encoding from the polyline level (1 token/polyline) to the point level (1 token/point), unifying the encoding of points, polylines, and relations.
    - **Mechanism**: Each query token is composed of sinusoidal positional encoding, NLP tag embedding, and an ORF element identifier. Polylines are resampled to a fixed 10 points; each point is encoded independently but shares the same ORF identifier.
    - **ORF Element Identifier**: Inspired by Orthogonal Random Features (ORF) from graph Transformers, orthogonal vectors are derived from the QR decomposition of a random Gaussian matrix and used as element identifiers. All points belonging to the same element share the same ORF (concatenated twice); relation elements are represented as edges using the ORF pair of their two member elements.
    - **Design Motivation**: Point-level queries align with the detection queries of downstream MapTRv2; ORF identifiers resolve the loss of element membership information caused by point-level decomposition while naturally supporting relation encoding.

3. **Map Decoder Connection**

    - Cross-attention is used to supply encoded SD map tokens to the Map Decoder (following the PMapNet scheme).
    - Experiments demonstrate this is more effective than BEV feature fusion, as cross-attention better compensates for spatial alignment errors.

### Loss & Training
- NLP encoder pre-training: self-supervised contrastive learning (Multiple Negatives Ranking Loss).
- End-to-end training: standard MapTRv2 losses with NLP encoder fine-tuning.
- Training runs for 24 epochs on Argoverse 2 and 110 epochs on nuScenes, using 4 × H100 GPUs.

## Key Experimental Results

### Main Results (Argoverse 2, Geographically Non-Overlapping Split)

| Method | Short-Range mAP | vs. Baseline | Long-Range mAP | vs. Baseline |
|--------|----------------|-------------|----------------|-------------|
| MapTRv2 (no prior) | 46.5 | — | 13.0 | — |
| + PMapNet | 46.9 | +0.4 | 15.3 | +2.3 |
| + PMapNet (full info) | 47.0 | +0.5 | 15.7 | +2.7 |
| + SMERF | 46.3 | -0.2 | 12.2 | -0.8 |
| + SMERF (full info) | 45.9 | -0.6 | 14.2 | +1.2 |
| **+ SDTagNet** | **48.1** | **+1.6** | **18.9** | **+5.9** |

Comparison with state-of-the-art non-SD-prior methods (short range): SDTagNet achieves 78.0 mAP vs. MapTracker's 76.9, while SDTagNet operates as a single-frame method.

### Ablation Study

| Configuration | Long-Range mAP | Notes |
|---------------|----------------|-------|
| SMERF baseline | 12.2 | Polyline-level encoding |
| + Point-level queries | 11.9 | Performance degrades without ORF |
| + Point-level queries + ORF | 14.2 | ORF is a prerequisite for point-level queries |
| + NLP Tag Embedding | 15.8 | Text annotations provide substantial gains |
| **Full SDTagNet** | **18.9** | All components yield the best synergistic performance |

### Key Findings
- SD map priors provide the greatest benefit at long range (+45%), with limited improvement at short range (+1.6 mAP).
- ORF identifiers are a necessary prerequisite for point-level queries—point-level queries without ORF perform worse than polyline-level encoding.
- The value of text annotations is particularly evident in qualitative analysis: SDTagNet correctly identifies one-way road topology, whereas competing methods incorrectly predict bidirectional intersections.
- Geographically non-overlapping splits are critical: the use of SD map priors exacerbates overfitting under geographically overlapping splits.

## Highlights & Insights
- **Elegance of NLP for map annotation**: Treating OSM key-value pairs as "sentences" and encoding them with BERT is a simple yet effective approach; the contrastive pre-training strategy addresses the distributional gap between map annotations and natural language.
- **ORF as element identifiers**: Borrowing ideas from graph Transformers to resolve element membership in point-level encoding while naturally supporting relation encoding represents an elegant unified representation.
- **Practical value of long-range perception**: A +45% improvement at long range has direct implications for highway driving scenarios.

## Limitations & Future Work
- Short-range improvement is limited (+1.6 mAP); the marginal value of SD map priors diminishes when sensor coverage is sufficient.
- Performance may degrade in regions with sparse OSM annotations due to dependence on OSM data quality and coverage.
- Validation is conducted only on the MapTRv2 architecture; applicability to other online mapping frameworks (e.g., StreamMapNet, MapTracker) requires further investigation.
- Although BERT is lightweight, it still introduces additional computational overhead; real-time latency metrics are not reported.

## Related Work & Insights
- **vs. PMapNet**: Rasterizes OSM data as images, incurring significant information loss and precluding the use of text annotations; SDTagNet retains more information through vectorized encoding.
- **vs. SMERF**: Employs a polyline-level Transformer with coarse granularity and processes only 7 manually selected tag categories; SDTagNet comprehensively outperforms it via point-level encoding and an open vocabulary.
- **vs. TopoSD**: Adopts a hybrid approach (BEV grid + vector tokens); however, SDTagNet demonstrates that cross-attention alone is sufficient.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of NLP encoding to SD map text annotations; ORF element identifier design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, geographically non-overlapping splits, comprehensive ablations, and fair baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, method descriptions are detailed, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Substantial long-range perception improvements; the NLP+map paradigm offers broad inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[CVPR 2026\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2026/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[NeurIPS 2025\] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation](leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)
- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)

</div>

<!-- RELATED:END -->
