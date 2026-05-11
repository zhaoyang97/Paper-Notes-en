---
title: >-
  [Paper Note] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding
description: >-
  [AAAI 2026][3D Vision][Open-Vocabulary 3D] This paper proposes the Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D) task and the corresponding OpenScan benchmark…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Open-Vocabulary 3D"
  - "Attribute Understanding"
  - "3D Scene Segmentation"
  - "Benchmark"
  - "Knowledge Graph"
date: 2026-05-08
content_hash: b08f2141bf9674c5
---

# OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding

**Conference**: AAAI 2026  
**arXiv**: [2408.11030](https://arxiv.org/abs/2408.11030)  
**Code**: [https://youjunzhao.github.io/OpenScan/](https://youjunzhao.github.io/OpenScan/)  
**Area**: 3D Scene Understanding / Open-Vocabulary  
**Keywords**: Open-Vocabulary 3D, Attribute Understanding, 3D Scene Segmentation, Benchmark, Knowledge Graph

## TL;DR

This paper proposes the Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D) task and the corresponding OpenScan benchmark, extending 3D scene understanding beyond object categories to eight linguistic attribute dimensions, revealing critical deficiencies of existing OV-3D methods in understanding abstract object attributes.

## Background & Motivation

### State of the Field

Open-vocabulary 3D scene understanding (OV-3D) aims to localize and classify novel objects beyond training categories. Leveraging vision-language models (VLMs) such as CLIP, OV-3D has achieved remarkable progress in object-category-level recognition. Representative methods including OpenMask3D, SAI3D, MaskClustering, and Open3DIS demonstrate strong performance on ScanNet200.

### Limitations of Prior Work

Existing methods and benchmarks (ScanNet, ScanNet200) **focus exclusively on object-category-level** open-vocabulary problems. However, understanding object-related attributes (e.g., affordance, material, properties) is equally critical for AI systems. For instance, a robot needs to understand "something you can sit on" (functional attribute) rather than merely "chair" (category label).

### Root Cause

The absence of large-scale 3D scene attribute annotation benchmarks prevents systematic evaluation of OV-3D models' generalization ability in object attribute understanding. Existing benchmarks contain only object category annotations, with no attribute annotations.

### Paper Goals

To construct a comprehensive evaluation benchmark **beyond object categories**, assessing OV-3D models' ability to understand abstract object attributes across multiple linguistic dimensions.

### Starting Point

The GOV-3D (Generalized Open-Vocabulary 3D Scene Understanding) task is introduced, extending queries from object categories to abstract object-related attributes. The OpenScan benchmark is built upon ScanNet200, acquiring attribute annotations through a combination of knowledge graphs and human annotation.

### Core Idea

**Object category recognition is merely the tip of the iceberg in 3D scene understanding.** Truly open-vocabulary understanding should encompass abstract concepts spanning multiple linguistic dimensions, including affordance, material, and properties.

## Method

### Overall Architecture

The OpenScan benchmark construction pipeline:
1. **Knowledge Graph Association**: ConceptNet is used to establish associations between the 200 object categories in ScanNet200 and various attributes.
2. **Human Annotation**: Visual attributes (e.g., material) are annotated manually.
3. **Attribute Categorization**: Attributes are organized into eight linguistic dimensions.
4. **Attribute Verification**: Human verification ensures semantic consistency.
5. **Query Generation**: Text queries are generated with object names concealed.

### Key Design 1: Eight-Dimensional Linguistic Attribute System

**Function**: Object attributes are organized into eight representative linguistic dimensions.

**Specific Dimensions**:
- **Affordance**: Functional use of an object, e.g., "sit" for a chair.
- **Property**: Object characteristics, e.g., "soft" for a pillow.
- **Type**: Category membership, e.g., a phone is a "communication device."
- **Manner**: Mode of use, e.g., a hat is "worn on a head."
- **Synonym**: Near-synonymous substitution, e.g., "image" for a picture.
- **Requirement**: Necessary conditions, e.g., a bicycle requires "balance to ride."
- **Element**: Constituent parts, e.g., a bicycle has "two wheels."
- **Material**: Material type, e.g., "plastic" for a bottle.

**Design Motivation**: These eight dimensions span from commonsense knowledge (affordance, requirement) to visual knowledge (material), enabling comprehensive evaluation of a model's deep understanding of objects.

### Key Design 2: Knowledge-Graph-Driven Annotation Generation

**Function**: ConceptNet is leveraged to automatically generate object–attribute association annotations.

**Mechanism**: For each object category $c_i$ in ScanNet200, relevant edges are queried from the knowledge graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
$$\{e\}_i = \{(v_m, r, w, v_n) \in \mathcal{E} | v_m = c_i\}$$

For the same relation $r$, the attribute with the highest weight $w$ is retained, ensuring only the most representative attribute is preserved for each object per dimension.

**Design Motivation**: Knowledge graphs provide a structured and scalable source of commonsense knowledge, enabling attribute annotation generation for a large number of objects at low cost.

### Key Design 3: Query Template Design

**Function**: Text queries with concealed object names are generated to evaluate the GOV-3D task.

**Mechanism**: The object category $v_m$ in the query is replaced with "this term," followed by concatenation of the relation and attribute:
$$q = \text{Concatenate}(t, r, v_n)$$
For example: "this term is made of wood."

**Design Motivation**: By excluding object names from queries, the model is forced to **reason through attributes** to localize objects, rather than relying on simple name matching.

### Loss & Training

The benchmark itself does not involve training losses. Evaluation metrics follow standard OV-3D protocols: AP/AP50/AP25 for instance segmentation, and mIoU/mAcc for semantic segmentation.

## Key Experimental Results

### Main Results: 3D Instance Segmentation

| Method | Affordance | Property | Synonym | Material | Mean | ScanNet200 |
|--------|-----------|----------|---------|----------|------|------------|
| OpenMask3D | 7.2 | 7.5 | 16.9 | 18.8 | 9.9 | 15.4 |
| SAI3D | 5.3 | 5.8 | 10.0 | 11.3 | 7.7 | 12.7 |
| MaskClustering | 6.2 | 7.0 | 16.2 | 12.1 | 8.1 | 12.0 |
| **Open3DIS** | **11.9** | **12.8** | **26.7** | **28.3** | **15.8** | **23.7** |

(AP metric; all methods perform **significantly worse** on OpenScan than on ScanNet200.)

### 3D Semantic Segmentation

| Method | OpenScan mIoU | OpenScan mAcc | ScanNet mIoU |
|--------|--------------|---------------|--------------|
| OpenScene | 0.45 | 1.87 | 47.5 |
| PLA | 0.01 | 2.37 | 66.6 |
| RegionPLC | 0.07 | 2.36 | 68.7 |

Semantic segmentation methods nearly completely fail on OpenScan (mIoU < 1%), indicating **severe lack of generalization from categories to attributes**.

### Ablation Study: Effect of Pre-training Vocabulary Size

Increasing the training vocabulary size ($S = 10 \to 170$) yields **no significant improvement** across most attribute dimensions, with only a marginal gain on the material dimension. This demonstrates that **simply expanding the number of training categories cannot resolve the attribute understanding problem**.

### Key Findings

1. All OV-3D models perform substantially worse on OpenScan than on ScanNet200, confirming that GOV-3D is a more challenging task.
2. **Synonym and Material** achieve relatively higher performance: the former due to semantic proximity to object categories, the latter due to CLIP's visual pattern recognition capability.
3. **Affordance and Property** are the most challenging: they require commonsense reasoning, which is not covered by CLIP's pre-training objective.
4. Using query templates (with relational descriptions) improves AP by approximately 0.2–6 points compared to using bare attribute words.

## Highlights & Insights

1. **Visionary Problem Formulation**: Extending OV-3D from categories to attributes is a natural yet previously overlooked research direction.
2. **Systematic Benchmark Design**: The eight-dimensional attribute system provides comprehensive coverage; the hybrid strategy of knowledge graphs combined with human annotation balances efficiency and quality.
3. **Experimentally Revealed Fundamental Limitations**: The results demonstrate that simply expanding training vocabulary cannot address attribute understanding, pointing toward the need for deeper methodological transformation.
4. **Substantial Scale**: 153,644 attribute annotations, 341 attributes, with an average of 3.15 attribute annotations per object.

## Limitations & Future Work

1. Commonsense attribute annotations rely on ConceptNet, whose coverage and quality may be limited.
2. Only material is annotated among visual attributes; other visual dimensions such as color and shape are not covered.
3. The benchmark is constructed solely on ScanNet200, restricting scene types to indoor environments.
4. No solution is proposed for the GOV-3D task; the paper only exposes the problem.
5. The selection criteria for the eight attribute dimensions are insufficiently justified, with potential omissions or overlaps.
6. Evaluation assumes the target object referenced by the query exists in the scene, which may be inconsistent with the GOV-3D task's claimed requirement to determine object presence.

## Related Work & Insights

1. **OpenScene** (Peng et al. 2023): Supports arbitrary text queries for zero-shot 3D semantic segmentation, but lacks quantitative evaluation on attribute dimensions.
2. **SceneFun3D** (Delitzas et al. 2024): Provides functional annotations for robot interaction scenes, but focuses solely on the affordance dimension.
3. **MMScan** (Lyu et al. 2024): A visual attribute understanding benchmark that lacks commonsense attributes.
4. **Insights**: The deficiencies of vision-language models in commonsense reasoning suggest that image-text alignment alone is insufficient; incorporating structured knowledge or multi-step reasoning may be necessary.

## Rating

⭐⭐⭐⭐ (4/5)

**Strengths**: The problem formulation is valuable, the benchmark construction is rigorous and comprehensive, and the experimental findings offer important guidance for future research.

**Weaknesses**: No solution is proposed, and certain annotation design choices lack sufficient justification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](../../NeurIPS2025/3d_vision/openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)

</div>

<!-- RELATED:END -->
