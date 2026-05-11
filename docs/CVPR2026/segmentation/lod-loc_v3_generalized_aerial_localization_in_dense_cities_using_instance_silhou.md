---
title: >-
  [Paper Note] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment
description: >-
  [CVPR 2026][Segmentation][UAV localization] This paper proposes LoD-Loc v3, which addresses two critical limitations of LoD-based UAV localization — poor cross-scene generalization and pose ambiguity in dense urban areas…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "UAV localization"
  - "LoD city models"
  - "instance segmentation"
  - "synthetic data"
  - "silhouette alignment"
date: 2026-05-08
content_hash: 305185c71b114d86
---

# LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.19609](https://arxiv.org/abs/2603.19609)
**Code**: [Project Page](https://nudt-sawlab.github.io/LoD-Locv3/)
**Area**: Segmentation / UAV Localization
**Keywords**: UAV localization, LoD city models, instance segmentation, synthetic data, silhouette alignment

## TL;DR
This paper proposes LoD-Loc v3, which addresses two critical limitations of LoD-based UAV localization — poor cross-scene generalization and pose ambiguity in dense urban areas — by constructing a large-scale synthetic instance segmentation dataset (InsLoD-Loc, 100K images) and upgrading the localization paradigm from semantic to instance silhouette alignment. On the Tokyo-LoDv3 dense scene benchmark, the method achieves a ~2000% improvement in (2m, 2°) accuracy over the previous state of the art.

## Background & Motivation
1. **Background**: Mainstream UAV visual localization relies on high-fidelity 3D reconstructions (SfM/photogrammetry), which are accurate but costly to build and maintain, data-intensive, and raise privacy and security concerns. Localization based on Level-of-Detail (LoD) city models offers a lightweight alternative — LoD models retain only building geometry following the CityGML standard and have been deployed at scale in the United States, China, Japan, Germany, and other countries.
2. **Limitations of Prior Work**: LoD-Loc v2 localizes by aligning building semantic segmentation silhouettes extracted from images with silhouettes rendered from LoD models, but suffers from two critical problems: (1) **poor generalization** — models trained in one city degrade significantly when deployed in another; (2) **dense-scene ambiguity** — in dense urban areas, semantic masks of multiple buildings merge into a single large region, and rendered semantic masks under different poses become nearly indistinguishable.
3. **Key Challenge**: Semantic segmentation only distinguishes "building" from "background," causing all buildings in dense areas to form a single connected region and eliminating discriminative information. In contrast, the instance-level arrangement of buildings is unique to each pose.
4. **Goal**: (1) Address cross-scene generalization through large-scale synthetic data; (2) resolve pose ambiguity in dense scenes through instance-level alignment.
5. **Key Insight**: Localization in urban environments is fundamentally an instance alignment process — each visible building in the image must be matched to its corresponding instance in the LoD model.
6. **Core Idea**: Replace semantic segmentation with instance segmentation for building silhouette extraction, and use the Dice coefficient for instance-level one-to-one matching to evaluate pose hypotheses.

## Method

### Overall Architecture
A three-stage pipeline: (1) LoD model instantiation — assigning a unique ID/color to each building; (2) building instance segmentation — extracting per-building masks from query images using a SAM-based model; (3) coarse-to-fine localization — evaluating pose hypotheses in a 4-DoF search space via instance silhouette alignment. The coarse stage uniformly samples the pose space; the fine stage iteratively refines via particle filtering.

### Key Designs

1. **InsLoD-Loc Synthetic Dataset (100K images)**:

    - **Function**: Provides large-scale, diverse training data to enable zero-shot cross-scene generalization.
    - **Mechanism**: A two-stage data generation pipeline — (a) UE5 + Cesium plugin streams Google Earth Photorealistic 3D Tilesets, and the AirSim plugin renders photorealistic RGB images; (b) corresponding LoD models are obtained from public sources, coordinate systems are unified, and instance masks are generated using the OSG rendering engine. The dataset covers 40 flight zones across 6 countries, with 3 camera configurations (varying FOV/resolution/sampling strategy) and flight altitudes of 200–500 m.
    - **Design Motivation**: LoD-Loc v2 trained on a single scene fails to generalize. InsLoD-Loc covers commercial, industrial, residential, educational, medical, and suburban land-use types to ensure training data diversity.

2. **LoD Model Instantiation**:

    - **Function**: Assigns a unique identifier to each building to enable instance-level rendering.
    - **Mechanism**: The untextured LoD model is parsed as a graph $G=(V,E,F)$, where each building $B_i$ corresponds to a connected component $G_i$. Graph partitioning divides the model into $M$ disjoint building instances, each assigned a unique 24-bit RGB color ID, enabling direct instance mask generation at render time.
    - **Design Motivation**: Semantic segmentation yields only a binary "building vs. non-building" mask. Instantiation gives each building an independent mask, providing the foundation for subsequent one-to-one matching.

3. **SAM-Based Building Instance Segmentation**:

    - **Function**: Automatically extracts per-building instance masks from query images.
    - **Mechanism**: A learnable Prompter Module is added on top of the SAM architecture. The SAM encoder extracts image features $F_{embed}$; the Prompter Module predicts prompt embeddings from these features; the SAM decoder generates the instance mask set $\mathcal{S}_q = \{M_q^j\}_{j=1}^N$. During training, the SAM encoder is frozen and fine-tuned with LoRA; only the Prompter Module and SAM decoder are updated.
    - **Design Motivation**: SAM has strong zero-shot segmentation capability but cannot automatically produce instance segmentation results. The Prompter Module transforms SAM into an automatic, task-specific instance segmentation pipeline.

### Instance Silhouette Alignment Cost Function

For each predicted instance $M_q^j$ in the query image, the best match (highest Dice coefficient $d_j^*$) is found in the rendered instance set $\mathcal{S}_{hyp}$. The final cost is a weighted sum of all instance matching scores. Two weighting strategies are provided:

- **Confidence-weighted**: $c_{ins}^{(conf)} = \sum_j \frac{s_j}{\sum_i s_i} d_j^*$
- **Area-weighted**: $c_{ins}^{(area)} = \sum_j \frac{A_j}{\sum_i A_i} d_j^*$

### Loss & Training
Multi-task training loss: $L = L_{rpn} + L_{roi}$. The RPN loss supervises proposal generation; the RoI loss supervises final classification, regression, and mask prediction. AdamW optimizer with learning rate $2 \times 10^{-4}$, weight decay 0.05, cosine annealing, trained for 20 epochs. The SAM encoder uses ViT-Huge pretrained weights with LoRA fine-tuning.

## Key Experimental Results

### Main Results (UAVD4L-LoDv2, localization success rate %)

| Method | Training Data | in-Traj 2m-2° | out-Traj 2m-2° | in-Traj 5m-5° | out-Traj 5m-5° |
|--------|--------------|--------------|---------------|--------------|---------------|
| MC-Loc(DINOv2) | — | 1.20 | 2.40 | 17.40 | 26.10 |
| LoD-Loc | In-dist.† | 49.56 | 54.20 | 89.09 | 89.51 |
| LoD-Loc v2 | In-dist.† | 93.70 | 97.90 | 99.50 | 100.00 |
| **LoD-Loc v3** | InsLoD-Loc | **97.60** | **97.40** | **99.70** | **99.40** |

### Tokyo-LoDv3 Dense Scene

| Method | Grid 2m-2° | Grid 5m-5° | Seq 2m-2° | Seq 5m-5° |
|--------|-----------|-----------|----------|----------|
| LoD-Loc v2† | 22.70 | 74.70 | 35.60 | 92.00 |
| **LoD-Loc v3** | **39.30** | **89.90** | **50.30** | **97.30** |

### Ablation Study (Semantic vs. Instance Alignment, same InsLoD-Loc training data)

| Alignment | Grid 2m-2°/3m-3°/5m-5° | Seq 2m-2°/3m-3°/5m-5° |
|-----------|----------------------|---------------------|
| LoD-Loc v2 (semantic) | 19.60/39.40/72.10 | 21.50/47.80/89.00 |
| **LoD-Loc v3 (instance)** | **38.10/65.40/86.40** | **49.80/79.90/95.80** |

### Key Findings
- **Zero-shot generalization surpasses in-distribution training**: LoD-Loc v3, trained exclusively on the synthetic InsLoD-Loc dataset, outperforms in-distribution-trained LoD-Loc v2 on UAVD4L-LoDv2, demonstrating that sufficiently diverse synthetic data can substitute for in-distribution real data.
- **Instance alignment, not data scale, drives gains**: Under the same InsLoD-Loc training data, the semantic variant (19.60%) falls far behind the instance variant (38.10%), confirming that performance improvements stem from the instance-level paradigm rather than data volume.
- **Dramatic improvement in dense scenes**: LoD-Loc v2 nearly fails on the Tokyo-LoDv3 dense benchmark, whereas v3 achieves substantial gains, validating the central role of instance alignment in resolving ambiguity.
- **Area-weighted and confidence-weighted strategies perform comparably**: Each strategy has marginal advantages in specific settings; area-weighted performs slightly better on Swiss-EPFL.
- **Feature matching methods (CAD-Loc, etc.) fail entirely**: All SIFT/SuperPoint/LoFTR-based methods achieve 0% success on LoD models due to the absence of texture.

## Highlights & Insights
- **The paradigm shift from semantic to instance alignment** is conceptually simple yet highly impactful: the same data and the same localization framework, with only the silhouette matching granularity changed from semantic to instance level, yields a twofold performance improvement in dense scenes — underscoring the importance of fine-grained matching in ambiguous environments.
- **The UE5 + Google Earth + OSG data generation pipeline** has strong engineering value: RGB rendering and instance mask rendering use separate engines with precise alignment, and the approach is extensible to any city with available LoD models.
- **LoD city models as localization base maps**: Compared to SfM point clouds, LoD models are extremely lightweight (geometric shells only), have been constructed at scale across many countries, and offer substantial practical application potential.

## Limitations & Future Work
- Instance segmentation may fail under extreme adverse weather conditions.
- LoD model accuracy is inherently limited, with alignment errors in some regions.
- The coarse-to-fine two-stage search is computationally limited in very large search spaces.
- Validation is restricted to urban scenes; non-building areas (e.g., forests, farmland) are not addressed.
- The method relies on a simplified 4-DoF assumption (known gravity direction); full 6-DoF scenarios remain unexplored.

## Related Work & Insights
- **vs. LoD-Loc v2**: The direct predecessor; v3 upgrades v2 from semantic to instance alignment and replaces scene-specific training with large-scale synthetic data to address generalization.
- **vs. LoD-Loc**: The earliest version relies on wireframe alignment and requires high-detail LoD2/3 models; v2 and v3 reduce this requirement to LoD1.
- **vs. CAD-Loc / MC-Loc**: Feature matching and alignment methods based on SIFT/SuperPoint/LoFTR fail entirely on textureless LoD models.
- **vs. SAM**: This work demonstrates an effective adaptation strategy for SAM on domain-specific tasks — adding a Prompter Module with LoRA fine-tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The semantic-to-instance paradigm shift is not a fundamental technical breakthrough, but the insight is sharp and the synthetic data pipeline is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, 7+ baselines, multiple ablations, and dedicated dense-scene testing.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and the technical presentation is complete.
- **Value**: ⭐⭐⭐⭐ Strong practical potential for global-scale UAV navigation; the method is immediately deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segme.md)
- [\[CVPR 2026\] UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)
- [\[CVPR 2026\] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [\[CVPR 2026\] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)

</div>

<!-- RELATED:END -->
