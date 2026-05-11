---
title: >-
  [Paper Note] Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species
description: >-
  [CVPR 2026][Autonomous Driving][plant counting] This paper introduces TPC-268, the first large-scale plant counting dataset integrating plant taxonomy, comprising 10,000 images, 678,050 point annotations…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "plant counting"
  - "class-agnostic counting"
  - "taxonomic hierarchy"
  - "fine-grained dataset"
  - "density estimation"
date: 2026-05-08
content_hash: ec20947011013b98
---

# Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species

**Conference**: CVPR 2026
**arXiv**: [2603.21229](https://arxiv.org/abs/2603.21229)
**Code**: [https://github.com/tiny-smart/TPC-268](https://github.com/tiny-smart/TPC-268)
**Area**: Autonomous Driving
**Keywords**: plant counting, class-agnostic counting, taxonomic hierarchy, fine-grained dataset, density estimation

## TL;DR

This paper introduces TPC-268, the first large-scale plant counting dataset integrating plant taxonomy, comprising 10,000 images, 678,050 point annotations, and 268 countable categories (covering 242 species), with complete Linnaean taxonomic hierarchy annotations, and provides comprehensive benchmarking under the class-agnostic counting (CAC) paradigm.

## Background & Motivation

**Background**: Visual counting has advanced rapidly over the past decade, but has primarily focused on rigid objects such as crowds and vehicles. The emergence of class-agnostic counting (CAC) enables models to generalize to unseen categories, yet existing CAC datasets (e.g., FSC-147) lack fine-grained categorical complexity.

**Limitations of Prior Work**: Plant counting differs fundamentally from general object counting — plants exhibit non-rigid morphology, dramatic morphological changes across growth stages, phenotypic plasticity, and enormous species diversity organized via taxonomic hierarchies. Crowd counting models need only distinguish "person" from "background," whereas plant counting systems must learn to discriminate subtle textural differences across hundreds of species. Existing plant counting datasets from plant science are small in scale, species-limited, and devoid of taxonomic information.

**Key Challenge**: The taxonomic hierarchy of plants provides a natural visual similarity prior (e.g., congeners sharing leaf morphology), yet existing counting methods entirely disregard this structured information. Furthermore, the "one model per species" paradigm is fundamentally unscalable given the vast species space of the plant kingdom.

**Goal**: To construct the first large-scale plant counting benchmark dataset integrating taxonomic hierarchy, providing a biologically grounded evaluation platform for CAC research in the plant domain.

**Key Insight**: Combining hierarchical taxonomic classification (kingdom → species) with instance-level point annotations, and using taxonomic distance to define data splits, ensuring that model generalization is evaluated across genuine cross-species gaps.

**Core Idea**: By constructing a large-scale plant counting dataset with complete Linnaean taxonomic hierarchy annotations, this work elevates CAC generalization evaluation from simple "unseen categories" to biologically meaningful "cross-taxonomic-gap" generalization.

## Method

### Overall Architecture

TPC-268 is a new dataset and benchmark rather than a new method. The dataset construction pipeline consists of: (1) multi-source data collection (Wikipedia 34%, PlantCLEF 29%, Internet 14%, etc.); (2) annotation protocol design (instance point annotations + exemplar bounding boxes + complete taxonomic hierarchy + biological organization categories); (3) taxonomy-based data splitting (MILP-optimized train/val/test partitioning); and (4) benchmarking of state-of-the-art methods under the CAC paradigm.

### Key Designs

1. **Taxonomic Hierarchy Annotation**:

    - Function: Provides each instance with a complete 7-level taxonomic identity from kingdom to species.
    - Mechanism: Pl@ntNet is used for initial genus/species identification, followed by completion of remaining levels via the World Flora Online database; all labels undergo three rounds of manual cross-validation. Each species is encoded as a 7-dimensional vector — e.g., apple (*Malus domestica*) is encoded as $[1,1,1,14,39,113,136]$. This encoding transforms the counting problem into a joint counting and hierarchical inference task.
    - Design Motivation: The taxonomic hierarchy of plants provides a structured visual similarity prior — congeners typically share morphological features. Explicitly encoding this information enables models to leverage taxonomic relatedness for cross-species generalization.

2. **Taxonomy-Distance-Based Data Splitting**:

    - Function: Ensures genuine taxonomic gaps between training and test sets.
    - Mechanism: The minimum indivisible unit is defined as a "species–tissue pair" (e.g., rice–flower vs. rice–stoma), with each species–tissue pair assigned exclusively to one subset. Mixed integer linear programming (MILP) solves the partitioning problem subject to constraints: (1) each subset covers all observed scales; (2) mean density is balanced across subsets (67.81 instances/image); (3) approximate 7:1:2 ratio is maintained.
    - Design Motivation: Conventional CAC datasets partition by category at random, potentially allowing close relatives of test species to appear in training. The taxonomic split guarantees that models face genuine cross-genus/cross-family generalization at test time, yielding a more rigorous and realistic evaluation.

3. **Multi-Scale and Multi-Tissue Coverage**:

    - Function: Covers the full spectrum of observation scales from microscopic tissue to macroscopic canopy.
    - Mechanism: The dataset spans four organizational levels — tissue (stomata 1,096, resin 228), organ (fruit 4,422, flower 2,994, seed 602, etc.), individual (whole plant 214), and population (canopy 56). Resolution ranges from $165 \times 127$ to $6,000 \times 4,000$ pixels.
    - Design Motivation: Real-world plant counting spans multiple scales from microscopy to UAV imagery; models must remain robust under extreme scale variation.

### Loss & Training

As a dataset paper, no new training strategy is proposed. Benchmark methods use their respective training strategies and are evaluated under the standard 3-shot and 1-shot CAC settings.

## Key Experimental Results

### Main Results (3-shot)

| Method | Conference | Type | Val MAE↓ | Val RMSE↓ | Test MAE↓ | Test RMSE↓ | Test R²↑ |
|--------|------------|------|----------|-----------|-----------|-----------|----------|
| FamNet | CVPR'21 | Regression | 28.87 | 52.51 | 30.43 | 65.62 | 0.62 |
| C-DETR | ECCV'22 | Detection | 22.66 | 77.51 | 22.68 | 57.97 | 0.74 |
| LOCA | ICCV'23 | Regression | 17.26 | 53.19 | **17.51** | **38.37** | **0.78** |
| DAVE | CVPR'24 | Regression | 16.47 | 52.87 | 17.61 | 40.06 | 0.75 |
| CACViT | AAAI'24 | Regression | 16.63 | 42.49 | 22.04 | 41.79 | 0.73 |
| CountGD | NeurIPS'24 | Detection | 18.32 | 54.55 | 19.52 | 50.51 | 0.61 |
| TasselNetV4 | ISPRS'26 | Regression | **13.20** | **43.93** | 22.95 | 51.36 | 0.60 |

### Cross-Dataset Transfer

| Direction | CountTR MAE | CACViT MAE | LOCA MAE |
|-----------|------------|------------|----------|
| FSC-147→FSC-147 | 11.90 | 10.83 | 10.72 |
| FSC-147→TPC-268 | 38.62 (+225%) | 26.73 (+147%) | 24.70 (+130%) |
| TPC-268→TPC-268 | 25.19 | 22.04 | 17.51 |
| TPC-268→FSC-147 | 26.53 (+5%) | 17.88 (-19%) | 15.16 (-13%) |

### Key Findings

- **Regression methods consistently outperform detection methods**: The dense spatial arrangement and structural entanglement of plants make explicit instance localization extremely challenging; holistic density estimation is better suited to this domain.
- **Val-to-Test generalization gap**: Global attention models such as CACViT and TasselNetV4 achieve the best validation performance but degrade on the test set (TasselNetV4: val 13.20 → test 22.95), indicating that local structural consistency is critical for cross-species generalization. LOCA, combining local and global cues, is the most robust on the test set.
- **Asymmetric cross-dataset transfer**: Transferring from FSC-147 (general objects) to TPC-268 (plants) causes a performance drop of 130–225%, whereas the reverse transfer from TPC-268 to FSC-147 shows little degradation or even improvement, demonstrating that plant counting is a harder task than general object counting.
- **Value of taxonomic information**: Incorporating species names in CountGD reduces MAE from 19.52 to 17.53, and further incorporating the full taxonomic hierarchy reduces it to 16.90, confirming that taxonomic priors serve as effective inductive biases.

## Highlights & Insights

- **The idea of integrating taxonomic hierarchy with visual counting** is highly creative. Conventional CAC treats categories as flat, independent labels, whereas this work leverages the hierarchical structure of biology to provide prior knowledge for cross-category generalization. This idea is transferable to any domain with hierarchical classification (e.g., model-level hierarchies for industrial parts, zoological taxonomy, etc.).
- **The MILP-optimized data splitting** ensures evaluation rigor — no close relative of any test species appears in the training set. This taxonomy-distance-defined partitioning strategy is worth adopting by other fine-grained dataset efforts.
- **The asymmetry in cross-dataset transfer** reveals a profound insight: models trained in a "hard" domain generalize more readily to an "easy" domain, but not vice versa. This has important implications for the design of "universal" counting models.

## Limitations & Future Work

- Taxonomic coverage remains limited, concentrating on angiosperms and a small number of fungi, with underrepresentation of gymnosperms, mosses, and other major clades.
- The density distribution is highly imbalanced — 72.1% of images contain fewer than 50 instances, while only 3% have more than 500.
- The absolute performance of all benchmarked methods remains low (best MAE 17.51), indicating that plant counting is far from practical deployment.
- t-SNE analysis reveals that features from the state-of-the-art method (LOCA) lack clear category-level separation, suggesting that current models fail to capture deep biological features. Explicitly modeling visual similarity through taxonomic relatedness is a promising direction for further investigation.
- Direct backbone replacement with BioCLIP performs poorly (MAE 34.75), indicating that dedicated adaptation strategies are required.

## Related Work & Insights

- **vs. FSC-147**: FSC-147 is the most widely used CAC benchmark (6,135 images, 147 categories), but lacks structured inter-category relationships. TPC-268 provides a more rigorous generalization evaluation framework through taxonomic hierarchy, and the morphological complexity of plants far exceeds that of the rigid objects in FSC-147.
- **vs. ShanghaiTech/NWPU**: Traditional crowd counting datasets contain a single category with high density but simple category structure. TPC-268 extends counting to 268 fine-grained categories, introducing the joint challenge of "counting + recognition."
- **vs. CountGD**: CountGD supports text-guided open-vocabulary counting; incorporating taxonomic text yields notable performance gains (MAE 19.52→16.90), yet still falls short of the purely visual method LOCA (17.51), suggesting that text encoding cannot fully substitute for visual exemplars.

## Rating

- Novelty: ⭐⭐⭐⭐ First introduction of plant taxonomy into counting dataset design; the evaluation protocol defined by taxonomic distance is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of 10 CAC methods + cross-dataset transfer + taxonomic information ablation + fine-grained error analysis + t-SNE visualization.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction principles are clearly articulated with thorough analysis, though some statistical descriptions are slightly verbose.
- Value: ⭐⭐⭐⭐ Provides a new challenging benchmark for CAC research with tangible application value for precision agriculture and ecological monitoring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Counting Stacked Objects](../../ICCV2025/autonomous_driving/counting_stacked_objects.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [\[CVPR 2026\] Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal](ghost-fwl_a_large-scale_full-waveform_lidar_dataset_for_ghost_detection_and_remo.md)

</div>

<!-- RELATED:END -->
