---
title: >-
  [Paper Note] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding
description: >-
  [AAAI 2026][3D Vision][Open-Vocabulary 3D] This paper proposes the Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D) task and the corresponding OpenScan benchmark, extending 3D scene understanding from object categories to eight linguistic property dimensions, revealing the severe limitations of existing OV-3D methods in understanding abstract object properties.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Open-Vocabulary 3D"
  - "Property Understanding"
  - "3D Scene Segmentation"
  - "benchmark"
  - "Knowledge Graph"
date: 2026-05-08
content_hash: 3e1c010bb7fb1173
---

# OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding

**Conference**: AAAI 2026  
**arXiv**: [2408.11030](https://arxiv.org/abs/2408.11030)  
**Code**: [https://youjunzhao.github.io/OpenScan/](https://youjunzhao.github.io/OpenScan/)  
**Area**: 3D Scene Understanding / Open-Vocabulary  
**Keywords**: Open-Vocabulary 3D, Property Understanding, 3D Scene Segmentation, benchmark, Knowledge Graph

## TL;DR

This paper proposes the Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D) task and the corresponding OpenScan benchmark, extending 3D scene understanding from object categories to eight linguistic property dimensions, revealing the severe limitations of existing OV-3D methods in understanding abstract object properties.

## Background & Motivation

### Background

Open-Vocabulary 3D Scene Understanding (OV-3D) aims to localize and classify novel objects beyond the training set. Recently, with the help of vision-language models (VLMs) like CLIP, OV-3D has made significant progress in object-category-level recognition. Representative methods such as OpenMask3D, SAI3D, MaskClustering, and Open3DIS exhibit excellent performance on ScanNet200.

### Limitations of Prior Work

Existing methods and benchmarks (ScanNet, ScanNet200) **focus only on object categories** in open-vocabulary scenarios. However, the understanding of object-related properties (e.g., affordance, material, characteristics) is also crucial for AI systems. For instance, a robot needs to understand "something to sit on" (affordance property) rather than just "chair" (category label).

### Key Challenge

The lack of large-scale 3D scene property annotation benchmarks prevents systematic evaluation of OV-3D models' generalization capabilities regarding object properties. Existing benchmarks only contain object category annotations, lacking property level labels.

### Goal

Construct a comprehensive evaluation benchmark **beyond object categories** to assess the capabilities of OV-3D models in understanding abstract object properties across multiple linguistic dimensions.

### Key Insight

Introduce the GOV-3D (Generalized Open-Vocabulary 3D Scene Understanding) task to extend queries from object categories to object-related abstract properties. Build the OpenScan benchmark based on ScanNet200, obtaining property annotations through a combination of knowledge graphs and manual annotations.

### Core Idea

**Object category recognition is only the tip of the iceberg in 3D scene understanding**; true open-vocabulary understanding should encompass abstract concepts across multiple linguistic dimensions such as affordance, material, and properties.

## Method

### Overall Architecture

The construction pipeline of the OpenScan benchmark:
1. **Knowledge Graph Association**: Utilize ConceptNet to establish associations between 200 object categories in ScanNet200 and various properties.
2. **Manual Annotation**: Manually annotate visual properties (e.g., materials).
3. **Property Classification**: Categorize properties into eight linguistic dimensions.
4. **Property Verification**: Manually verify to ensure semantic consistency.
5. **Query Generation**: Generate text queries that hide object names.

### Key Design 1: Eight-dimensional Linguistic Property System

**Function**: Segment object properties into eight representative linguistic dimensions.

**Specific Dimensions**:
- **Affordance**: Object functions or usages, such as "sit" for a chair.
- **Property**: Object characteristics, such as "soft" for a pillow.
- **Type**: Corresponding category, such as a phone being a "communication device".
- **Manner**: Usage manner, such as a hat being "worn on a head".
- **Synonym**: Near-synonymous replacements, such as "image" for picture.
- **Requirement**: Necessary conditions, such as a bicycle needing "balance to ride".
- **Element**: Constituent elements, such as a bicycle having "two wheels".
- **Material**: Material types, such as "plastic" for a bottle.

**Design Motivation**: These eight dimensions cover multiple layers of understanding, from commonsense knowledge (affordance, requirements) to visual knowledge (material), comprehensively evaluating the deep understanding capabilities of models regarding objects.

### Key Design 2: Knowledge Graph-driven Annotation Generation

**Function**: Automatically generate association annotations between objects and properties utilizing ConceptNet.

**Mechanism**: For each object category $c_i$ in ScanNet200, query relevant edges from the knowledge graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
$$\{e\}_i = \{(v_m, r, w, v_n) \in \mathcal{E} | v_m = c_i\}$$

For the same relation $r$, keep the property with the highest weight $w$, ensuring each object retains only the most representative properties for each dimension.

**Design Motivation**: The knowledge graph provides a structured and scalable source of commonsense knowledge, allowing property annotations to be generated for many objects at a low cost.

### Key Design 3: Query Template Design

**Function**: Generate text queries with hidden object names to evaluate the GOV-3D task.

**Mechanism**: Replace the object category $v_m$ in the query with "this term", and then concatenate the relation and the property:
$$q = \text{Concatenate}(t, r, v_n)$$
For example, "this term is made of wood".

**Design Motivation**: The absence of object names in the query forces the model to localize objects **through property reasoning** rather than simple name matching.

### Loss & Training

The benchmark itself does not involve training losses. Standard OV-3D metrics are used as evaluation metrics: AP/AP50/AP25 for instance segmentation, and mIoU/mAcc for semantic segmentation.

## Key Experimental Results

### Main Results: 3D Instance Segmentation

| Method | Affordance | Property | Synonym | Material | Mean | ScanNet200 |
|------|-----------|----------|---------|----------|------|------------|
| OpenMask3D | 7.2 | 7.5 | 16.9 | 18.8 | 9.9 | 15.4 |
| SAI3D | 5.3 | 5.8 | 10.0 | 11.3 | 7.7 | 12.7 |
| MaskClustering | 6.2 | 7.0 | 16.2 | 12.1 | 8.1 | 12.0 |
| **Open3DIS** | **11.9** | **12.8** | **26.7** | **28.3** | **15.8** | **23.7** |

(AP metric, the performance of all methods on OpenScan is **significantly lower** than on ScanNet200)

### 3D Semantic Segmentation

| Method | OpenScan mIoU | OpenScan mAcc | ScanNet mIoU |
|------|--------------|---------------|--------------|
| OpenScene | 0.45 | 1.87 | 47.5 |
| PLA | 0.01 | 2.37 | 66.6 |
| RegionPLC | 0.07 | 2.36 | 68.7 |

Semantic segmentation methods fail almost completely on OpenScan (mIoU < 1%), indicating a **severe lack of generalization from categories to properties**.

### Ablation Study: Influence of Pre-trained Vocabulary Size

Increasing the pre-trained vocabulary size ($S = 10 \to 170$) shows **no significant improvement** for most property dimensions, with only slight improvements in the material dimension. This indicates that **simply expanding the number of training categories cannot solve the property understanding problem**.

### Key Findings

1. All OV-3D models perform significantly worse on OpenScan than on ScanNet200, demonstrating that GOV-3D is a much more challenging task.
2. **Synonym and Material** perform relatively better: the former due to its semantic closeness to object categories, and the latter due to CLIP's visual pattern recognition capabilities.
3. **Affordance and Property** are the most challenging: they require commonsense reasoning abilities, which are not covered by CLIP's pre-training objectives.
4. Using query templates (containing relation descriptions) improves AP by approximately 0.2 to 6 points compared to using property words alone.

## Highlights & Insights

1. **Visionary problem formulation**: Extending OV-3D from categories to properties is a natural yet previously overlooked research direction.
2. **Systematic benchmark design**: The eight-dimensional property system provides comprehensive coverage, and the hybrid strategy of knowledge graph and manual annotation balances efficiency and quality.
3. **Experimental findings reveal fundamental limitations**: Demonstrating that simply expanding the training vocabulary cannot solve the property understanding problem points to the need for deeper methodological reforms.
4. **Impressive scale**: 153,644 property annotations, 341 unique properties, with an average of 3.15 property annotations per object.

## Limitations & Future Work

1. Commonsense property annotations rely on ConceptNet, which might have limited coverage and quality.
2. Visual properties only cover the material dimension, while other visual features like color and shape are not covered.
3. Built solely on ScanNet200, limiting the scene types to indoor environments.
4. No proposed solutions are provided for the GOV-3D task; it only highlights existing limitations.
5. The selection criteria for the eight property dimensions are not fully justified and may have omissions or overlaps.
6. The evaluation phase assumes that the queried target object exists in the scene, whereas the GOV-3D task claims to require judging whether it exists.

## Related Work & Insights

1. **OpenScene** (Peng et al. 2023): Supports zero-shot 3D semantic segmentation for arbitrary text queries but lacks quantitative evaluation on property dimensions.
2. **SceneFun3D** (Delitzas et al. 2024): Functional annotation for robotic interaction scenarios, but focuses only on the affordance dimension.
3. **MMScan** (Lyu et al. 2024): A visual property understanding benchmark that lacks commonsense properties.
4. **Insights**: The limitations of vision-language models in commonsense reasoning suggest that image-text alignment alone is insufficient; incorporating structured knowledge or multi-step reasoning capabilities may be necessary.

## Rating

⭐⭐⭐⭐ (4/5)

**Strengths**: The problem definition is valuable, the benchmark construction is solid and comprehensive, and the experimental findings provide important guidance.

**Weaknesses**: No solutions are proposed, and some annotation design choices lack thorough justification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](../../NeurIPS2025/3d_vision/openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/openvoxel_training-free_grouping_and_captioning_voxels_for_open-vocabulary_3d_sc.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)

</div>

<!-- RELATED:END -->
