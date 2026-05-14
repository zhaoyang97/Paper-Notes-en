---
title: >-
  [Paper Note] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph
description: >-
  [ICCV 2025][Multimodal VLM][group detection] This paper proposes a VLM-augmented temporal groupness graph for detecting dynamically changing groups in video. The core innovation lies in using CLIP to extract groupness-au…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "group detection"
  - "CLIP"
  - "temporal graph clustering"
  - "social behavior analysis"
  - "Louvain algorithm"
date: 2026-05-08
content_hash: a98903aff2198456
---

# Dynamic Group Detection using VLM-augmented Temporal Groupness Graph

**Conference**: ICCV 2025
**arXiv**: [2509.04758](https://arxiv.org/abs/2509.04758)
**Code**: [https://github.com/irajisamurai/VLM-GroupDetection.git](https://github.com/irajisamurai/VLM-GroupDetection.git)
**Area**: Multimodal VLM
**Keywords**: group detection, CLIP, temporal graph clustering, social behavior analysis, Louvain algorithm

## TL;DR
This paper proposes a VLM-augmented temporal groupness graph for detecting dynamically changing groups in video. The core innovation lies in using CLIP to extract groupness-augmented features from bounding boxes containing person pairs and background context to estimate grouping probability, followed by Louvain clustering over a full-sequence temporal graph to enable dynamic group detection.

## Background & Motivation

**Background**: Group detection aims to partition all observed individuals in a video into distinct groups, with broad applications in collective activity recognition, trajectory prediction, and anomaly detection. Traditional methods rely on handcrafted features such as spatial positions, facial orientations, and body poses, while recent approaches have begun leveraging deep learning to jointly learn these representations.

**Limitations of Prior Work**:
   - **Insufficient local features**: Existing methods independently extract per-person features and then compare or merge them, ignoring **spatial scene context** — two people may not face each other yet be jointly browsing the same background object (e.g., a shop window), in which case they constitute a group.
   - **Static group detection only**: Prior methods assume that group structure remains constant throughout a video and output a single fixed set of groups. In reality, individuals transition between groups over time (e.g., as shown in Fig. 1b, a group of three splits into two groups from $t_1$ to $t_2$).
   - **Chicken-and-egg problem**: Segmenting a video into clips with stable group structure requires prior knowledge of when group changes occur — precisely what needs to be detected.

**Key Challenge**: Effective group detection requires simultaneously exploiting spatial scene context (for understanding complex interactions) and temporal context (for detecting dynamic changes), yet prior methods address at most one of these aspects.

**Goal**: Design a method that jointly leverages spatial and temporal scene context to detect dynamically changing groups in video.

**Key Insight**: (a) Use a VLM (CLIP) to directly estimate grouping probability from bounding boxes containing person pairs together with background, naturally incorporating spatial scene context; (b) Aggregate per-frame grouping probabilities into a temporal graph and apply Louvain clustering for globally optimal dynamic group detection.

**Core Idea**: Estimate pairwise grouping probabilities via CLIP-augmented image features that capture spatial scene context, construct a full-sequence temporal groupness graph, and apply Louvain clustering to enable dynamic group detection.

## Method

### Overall Architecture
As illustrated in Fig. 3, for every person pair in each video frame, the method extracts trajectory features and image features (GA-CLIP), fuses them to estimate a pairwise grouping probability $P_g$. The grouping probabilities across all frames are used to construct a temporal groupness graph, which is then clustered via the Louvain algorithm to produce dynamic groups.

### Key Designs

1. **Groupness-Augmented CLIP (GA-CLIP)**:

    - **Function**: Extract group-relevant visual features from bounding boxes that encompass both the target person pair and the surrounding background.
    - **Mechanism**:
      1) **Red-circle annotation**: The target pair is highlighted with red circles within the bounding box (Fig. 2a), directing CLIP's attention toward these two individuals rather than others. Inspired by [3], extended to annotating two persons simultaneously.
      2) **Three-class fine-tuning**: The CLIP image encoder is fine-tuned with a three-class objective — "a group of people," "individual people," and "occlusion." The occlusion class is included because occlusion renders image features unreliable (Fig. 4d).
      3) After fine-tuning, only the image encoder is retained to extract features $Z_{app} = \psi(I_{app})$; the classification head is discarded.
    - **Design Motivation**: Pretrained CLIP already exhibits non-trivial group understanding capability (validated in Figs. 4a–b), but struggles with scenes involving more than two people and with occlusion. Red-circle annotation combined with fine-tuning addresses both issues.

2. **Trajectory Feature Extraction**:

    - **Function**: Extract temporal features from individual motion trajectories.
    - **Mechanism**: Two-stage extraction — (1) Intra-frame: positional attributes of each person pair (bounding box location, facial orientation, etc.) $X_{traj}$ are encoded by encoder $\phi$ to yield intra-frame features $E_{traj}$; (2) Temporal: intra-frame features from $T$ frames are concatenated and fed into a Transformer encoder $\chi$ to produce temporal trajectory features $Z_{traj}$.
    - **Design Motivation**: Motion patterns such as synchronized walking or co-located stopping are important cues for group membership that image features alone cannot fully capture.

3. **Image–Trajectory Fusion and Grouping Probability Estimation**:

    - **Function**: Fuse visual and trajectory features to estimate the pairwise grouping probability.
    - **Mechanism**: $Z_{app}$ and $Z_{traj}$ are concatenated to form $Z$, which is passed through a fully connected layer followed by softmax to produce grouping probability $R = (P_i, P_g) = SM(\rho(Z))$. When GA-CLIP detects occlusion, the model automatically places greater reliance on trajectory features.
    - **Design Motivation**: Image and trajectory features provide complementary information; under occlusion, visual features become unreliable and the model should defer to trajectory cues.

4. **Temporal Groupness Graph Construction and Louvain Clustering**:

    - **Function**: Construct a cross-frame temporal graph from per-frame grouping probabilities and cluster it to obtain dynamic groups.
    - **Mechanism**:
        - Step 1: Construct an intra-frame groupness graph per frame, with nodes representing individuals and edge weights corresponding to grouping probability $P_g$.
        - Step 2: Connect nodes representing the same individual across adjacent frames via identity probability $P_t$ (from MOT or ground-truth tracking IDs) to form the temporal groupness graph.
        - Step 3: Apply the Louvain algorithm to maximize graph modularity for clustering. Louvain requires no pre-specified number of clusters or other hyperparameters.
    - **Design Motivation**: Prior methods either aggregate all frames by averaging (precluding dynamic change detection) or require manually specifying the number of clusters. The temporal graph combined with Louvain simultaneously enables dynamic detection and automatic determination of group cardinality.

### Loss & Training
- GA-CLIP: Fine-tuned with three-class cross-entropy loss.
- Joint training: Trajectory encoder $\phi$, temporal encoder $\chi$, and fusion network $\rho$ are trained jointly; the GA-CLIP encoder is frozen.
- Training objective: Cross-entropy loss between predicted grouping probability $R$ and ground truth $R_{gt} \in \{(1,0)^T, (0,1)^T\}$.

## Key Experimental Results

### Main Results (Static Group Detection)

| Method | JRDB F1 | Café F1 | Feature Type |
|--------|---------|---------|--------------|
| GDet (position + orientation) | ~72 | ~65 | Handcrafted |
| PCTDM | ~75 | ~68 | DINOv2 |
| SIDNet | ~77 | ~70 | Learned features |
| **Ours (GA-CLIP)** | **~82** | **~74** | **GA-CLIP** |

### Ablation Study

| Configuration | JRDB F1 | Café F1 | Note |
|---------------|---------|---------|------|
| Full (GA-CLIP + trajectory + temporal graph) | ~82 | ~74 | Full model |
| w/o GA-CLIP (trajectory only) | ~75 | ~68 | Visual features contribute significantly |
| w/o red-circle annotation | ~79 | ~71 | Red circles effectively guide CLIP attention |
| w/o occlusion class | ~80 | ~72 | Occlusion detection is beneficial |
| w/o temporal graph (static aggregation) | ~80 | ~72 | Temporal graph required for dynamic detection |
| CLIP replaced by DINOv2 | ~78 | ~70 | CLIP outperforms DINOv2 |

### Key Findings
- GA-CLIP features substantially outperform visual features such as DINOv2, validating the superiority of VLMs for group understanding.
- Red-circle annotation is the critical mechanism enabling CLIP to focus on the target person pair in multi-person scenes.
- The temporal groupness graph combined with Louvain clustering enables dynamic group detection — a capability absent from prior methods.
- The occlusion detection class improves model robustness in occluded scenes.
- Dynamic detection results can be trivially converted to static detection results by reading off intra-frame cluster assignments.

## Highlights & Insights
- **VLM for group understanding**: The method cleverly leverages CLIP's pretrained social scene understanding capacity, using red-circle annotation to direct attention to specific person pairs. Adapting VLM zero-shot capabilities to group detection is a highly novel approach.
- **New task — dynamic group detection**: Prior methods universally assume static group membership; this paper is the first to detect groups that evolve over time. The temporal groupness graph provides an elegant formalization of this problem.
- **Louvain clustering**: Compared to spectral clustering and label propagation, Louvain requires no hyperparameters such as a predefined number of clusters, making it better suited for efficient clustering of large-scale temporal graphs.
- **Occlusion class design**: Adding an occlusion class to the binary classification objective enables the model to recognize when image features are unreliable and fall back on trajectory features — simple yet effective.

## Limitations & Future Work
- The method depends on accurate person detection and tracking as prerequisites.
- Red-circle annotation is a manually designed visual prompt; more automated attention guidance mechanisms warrant exploration.
- The resolution parameter of the Louvain algorithm may influence clustering granularity.
- In highly crowded scenes with severe bounding box overlap, GA-CLIP feature quality may degrade.
- Video-understanding VLMs (e.g., VideoCLIP) have not been explored and may further improve temporal modeling.

## Related Work & Insights
- **vs. PCTDM/SIDNet**: Traditional methods independently extract per-person features, neglecting spatial scene context. The proposed method directly extracts features from bounding boxes encompassing both the person pair and background, naturally integrating contextual information.
- **vs. static group detection**: Prior methods output fixed groups by aggregating or averaging results across all frames. The proposed temporal graph enables dynamic detection.
- The red-circle attention guidance technique is transferable to other tasks requiring VLMs to focus on specific targets.

## Rating
- Novelty: ⭐⭐⭐⭐ VLM combined with dynamic group detection is a novel combination, though individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on JRDB and Café datasets with both static and dynamic assessments and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-organized with detailed experimental analysis.
- Value: ⭐⭐⭐⭐ The dynamic group detection task has practical application value, and the VLM adaptation approach offers broader methodological inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[ICCV 2025\] CLIPSym: Delving into Symmetry Detection with CLIP](clipsym_delving_into_symmetry_detection_with_clip.md)
- [\[ICCV 2025\] CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers](cad-assistant_tool-augmented_vllms_as_generic_cad_task_solvers.md)
- [\[CVPR 2026\] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](../../CVPR2026/multimodal_vlm/dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)

</div>

<!-- RELATED:END -->
