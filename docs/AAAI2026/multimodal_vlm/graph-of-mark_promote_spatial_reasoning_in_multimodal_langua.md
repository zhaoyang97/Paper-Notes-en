---
title: >-
  [Paper Note] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting
description: >-
  [AAAI 2026 (Main Track)][Multimodal VLM][Visual Prompting] This paper proposes Graph-of-Mark (GoM), a training-free pixel-level visual prompting method that explicitly encodes inter-object spatial relationships by overla…
tags:
  - "AAAI 2026 (Main Track)"
  - "Multimodal VLM"
  - "Visual Prompting"
  - "Spatial Reasoning"
  - "Scene Graph"
  - "Multimodal Language Models"
  - "Zero-Shot Reasoning"
date: 2026-05-08
content_hash: 94d183ee30a11b00
---

# Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting

**Conference**: AAAI 2026 (Main Track)
**arXiv**: [2603.06663](https://arxiv.org/abs/2603.06663)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Visual Prompting, Spatial Reasoning, Scene Graph, Multimodal Language Models, Zero-Shot Reasoning

## TL;DR

This paper proposes Graph-of-Mark (GoM), a training-free pixel-level visual prompting method that explicitly encodes inter-object spatial relationships by overlaying a depth-aware scene graph (comprising nodes and directed edges) directly onto input images, achieving up to an 11 percentage point improvement in zero-shot spatial reasoning accuracy for multimodal language models on VQA and grounding tasks.

## Background & Motivation

Current multimodal language models (MLMs) tend to treat visual content as "bags of objects" — they can recognize a "cup" and a "table" in an image, yet frequently fail to accurately determine their spatial relationship (to the left? behind? above?). This deficiency stems from two sources: (1) visual encoders lose fine-grained relational information when converting images into tokens; and (2) standard training objectives emphasize global image understanding and label matching, lacking explicit spatial supervision.

Prior solutions fall into two categories: fine-tuning models on spatially annotated data (e.g., SpatialVLM), which incurs high computational cost; and applying visual prompting techniques such as Set-of-Mark (SoM), which annotates object regions with numeric identifiers. As the current de facto standard, SoM has been shown to enhance visual grounding, but its fundamental limitation is that **it treats annotated objects as isolated entities, entirely ignoring spatial relationships between them**. Each object receives a label, yet whether a cup is "above" or "beside" a table remains for the model to infer unaided.

## Core Problem

How can inter-object spatial relationships be explicitly communicated to an MLM — without modifying model parameters (training-free) — by modifying the input image itself, thereby eliciting the model's latent spatial reasoning capabilities?

This question is consequential because spatial reasoning is indispensable for downstream tasks such as robotic manipulation, navigation, and medical image analysis, yet current mainstream MLMs perform poorly in this regard. A plug-and-play, inference-only solution would enhance any existing model at zero additional training cost.

## Method

### Overall Architecture

GoM upgrades SoM's "set annotation" paradigm to a "graph annotation" paradigm. The core idea is to **overlay a scene graph onto the input image, where nodes represent objects and directed edges represent spatial relationships between objects**. The full pipeline consists of four stages:

**Input**: raw image + user text query → **Stage 1**: detection & segmentation → **Stage 2**: relationship estimation → **Stage 3**: query-aware filtering → **Stage 4**: visual rendering → **Output**: scene-graph-augmented image fed into any MLM for inference.

### Key Designs

1. **Multi-Detector Ensemble with Fine-Grained Segmentation**: To maximize scene coverage, GoM employs three complementary detectors: OWL-V2 (open-vocabulary detection), YOLOv8 (high-confidence common objects), and Mask R-CNN (robust region proposals). Overlapping detections are merged via Weighted Boxes Fusion (WBF) to eliminate redundant annotations, followed by SAM-HQ (Segment Anything in High Quality) for precise mask generation. This ensemble strategy achieves broader coverage than any single detector, though at significant computational cost.

2. **Three-Dimensional Spatial Relationship Estimation**: This is the most critical capability GoM adds over SoM. For each pair of objects, relationships are assessed along three dimensions:

    - **2D Directional Relations**: horizontal and vertical displacements between bounding-box centers are computed; a threshold $\tau_{dir}$ determines "above/below/left/right". Formally: $R_{dir}(i,j) = \text{above} \iff \Delta y > |\Delta x| + \tau_{dir}$
    - **3D Depth Relations**: the monocular depth estimator MiDaS assigns a relative depth value to each object; a threshold $\tau_z$ determines "in_front_of / behind"
    - **Proximity Relations**: Euclidean distances between objects are computed to determine "near/touching/very close"

3. **Query-Aware Filtering**: Overlaying all pairwise relationships would clutter the image. GoM applies semantic similarity matching against the text query, retaining only objects mentioned in the query and their immediate spatial neighbors, ensuring the visual prompt remains task-relevant.

4. **Collision-Free Visual Rendering**: Objects are highlighted with colored masks and unique IDs; relationships are drawn as directed arrows from head to tail entity. A multi-step conflict resolution algorithm ensures annotations do not occlude the objects themselves — overlapping labels are iteratively displaced along coordinate axes; if an ID marker is displaced too far, a dashed line reconnects it to the original position; multiple arrows originating from the same object are assigned distinct curvature radii to remain distinguishable.

### Loss & Training

GoM is a **training-free** inference-stage method involving no loss functions or model parameter updates. All components (OWL-V2, YOLOv8, Mask R-CNN, SAM-HQ, MiDaS) use pretrained weights and are chained together solely at inference time. Hyperparameters include the directional threshold $\tau_{dir}$ and the depth threshold $\tau_z$.

## Key Experimental Results

The method is evaluated on 3 open-source MLMs (LlamaV-o1, Qwen-2.5-VL, Gemma-3) across 4 standard benchmarks, including GQA for spatial reasoning, VQAv2 for general VQA, and RefCOCOg for referring expression comprehension.

| Evaluation Dimension | Key Finding | Quantitative Result |
|---|---|---|
| Overall Improvement | GoM vs. raw image / SoM baseline | Up to **11 percentage points** |
| Best-Performing Model | LlamaV-o1 (reasoning-optimized) | Highest absolute performance; GoM further catalyzes its latent reasoning capacity |
| Scene Graph Complexity Sweet Spot | 3–10 objects, 4–16 relations | Beyond ~20 objects, noise begins to degrade performance |
| Visual Graph vs. Text Graph | Drawn on image vs. provided as text triples | Visual graph alone typically **outperforms** text graph alone |
| Supplementary Text Description | Both visual and text graph provided | Further gains, though not consistently additive |

### Ablation Study

- **Visual Graph vs. Text Graph** is among the most valuable findings: MLMs are more adept at extracting spatial relationships from structure embedded in pixel space than from serialized text descriptions. This suggests that visual encoders retain a degree of graph-structure comprehension.
- **Effect of Scene Graph Density**: when the number of objects exceeds ~20, the overlaid annotations themselves become noise, disrupting model performance. This reveals a fundamental tension in visual prompting — balancing information gain against visual clutter.
- **Contribution of Depth Relations (MiDaS)**: incorporating 3D depth information contributes substantially to tasks involving "in front of / behind" judgments, but provides limited benefit for purely 2D directional questions.
- **Necessity of Query-Aware Filtering**: the unfiltered variant shows a marked performance drop, validating the importance of selective over full-scene annotation.

## Highlights & Insights

- **Paradigm Upgrade from "Set" to "Graph"**: extending SoM's object annotation from isolated nodes to a graph structure with relational edges is conceptually elegant and intuitively well-motivated — an "obvious yet overlooked" contribution.
- **Purely Visual Structural Communication**: the finding that MLMs can directly interpret graph structure (arrows, labels) from pixels without textual assistance is empirically illuminating.
- **Plug-and-Play Engineering Value**: training-free, model-agnostic, and applicable to any MLM, the method has a low deployment barrier.
- **Collision-Free Rendering Algorithm**: the practical engineering problem of visual annotations occluding original image content is carefully addressed through label displacement, dashed-line reconnection, and adaptive arrow curvature — demonstrating thorough attention to detail.

## Limitations & Future Work

- **High Inference Latency**: the cascaded pipeline of five pretrained models (3 detectors + SAM-HQ + MiDaS) incurs substantially greater inference overhead than directly feeding images to an MLM. The paper reports no speed comparison, which may be a critical concern in practical deployment.
- **Detector-Quality Dependency**: the chain-like dependency (detection → segmentation → depth estimation → relationship judgment) means errors at any stage are amplified downstream. If the detector misses a key object or MiDaS produces inaccurate depth estimates, the resulting scene graph is fundamentally incorrect.
- **Static Images Only**: temporal spatial relationships in video are not considered, nor are occluded objects handled.
- **Limited Experimental Scope**: only 3 open-source MLMs are tested; closed-source models such as GPT-4V/4o, Claude, and Gemini are excluded. Given that SoM was originally demonstrated on GPT-4V, this omission is notable.
- **Degradation in Dense Scenes**: when more than ~20 objects are present, the scene graph introduces noise rather than signal, limiting applicability to complex real-world scenes such as street views or factory floors.
- **Fixed Relation Types**: only directional, depth, and proximity relations are modeled; functional relations (e.g., "supports," "contains") and semantic relations (e.g., "belongs to," "in use") are not covered, limiting relational expressiveness.

## Related Work & Insights

1. **vs. Set-of-Mark (SoM)**: GoM is a direct extension of SoM — SoM annotates only objects (nodes), while GoM additionally encodes inter-object spatial relationships (edges). GoM explicitly outperforms SoM on spatial reasoning tasks. SoM retains advantages in simplicity, low latency, and broader task applicability, whereas GoM's multi-model pipeline adds considerable complexity.

2. **vs. SpatialVLM**: SpatialVLM enhances spatial understanding through fine-tuning, requiring spatially annotated training data and additional training compute. GoM's training-free nature is its primary advantage, at the cost of running multiple external models at inference time. The two approaches are complementary — GoM-augmented images could also serve as inputs to fine-tuning methods such as SpatialVLM.

3. **vs. Herzig et al. (Structured Representations + Pretrained VLMs)**: that line of work integrates scene graph information via latent-space or textual mechanisms. GoM's distinctive contribution lies in embedding the scene graph directly at the pixel level, enabling the model's visual encoder to "see" structural information directly rather than receiving it through auxiliary modules.

## Inspirations & Connections

- **Connection to VHD-Guided Adaptive Visual Re-injection**: GoM's finding that MLMs can interpret structural information from visual signals could be integrated with the VHD/TVC direction — during long-chain reasoning, when VHD detects visual forgetting, the re-injected content could include not only raw visual tokens but also GoM-augmented structured visual information, helping the model maintain spatial relationship comprehension throughout inference.
- **The trade-off between scene graph density and guidance effectiveness** warrants further investigation — could annotation density be determined adaptively? Combined with attention mechanisms, could the model itself "select" the relational edges it needs to attend to?
- **Cross-Modal Structural Consistency**: the finding that visual graphs outperform text graphs implies an asymmetry in how the visual and text encoders of MLMs process structural information, offering a clue for understanding VLM internal mechanisms.

## Rating

- Novelty: ⭐⭐⭐ The extension from SoM to GoM is well-motivated and natural, but the core idea is incremental (adding edges to a graph) without a fundamental paradigm shift
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation design is comprehensive (visual/text graph comparison, density analysis, component contribution), but only 3 open-source models are tested; closed-source models and inference latency analysis are absent
- Writing Quality: ⭐⭐⭐⭐ Technical details are clearly described; pipeline stage logic is coherent; mathematical formalization is appropriate
- Value: ⭐⭐⭐ Training-free and plug-and-play characteristics are strong selling points, but the cascaded five-model pipeline limits practical deployment; the 11 percentage point gain is meaningful in spatial reasoning scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](../../CVPR2026/multimodal_vlm/graphvlm_benchmark_vlm_graph_learning.md)
- [\[CVPR 2026\] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](../../CVPR2026/multimodal_vlm/dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)
- [\[AAAI 2026\] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation](few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
