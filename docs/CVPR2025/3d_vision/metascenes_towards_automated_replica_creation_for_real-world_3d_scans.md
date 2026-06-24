---
title: >-
  [Paper Note] MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans
description: >-
  [CVPR 2025][3D Vision][3D scene reconstruction] MetaScenes constructs a large-scale simulatable 3D scene dataset (15,366 object instances across 831 categories) by automatically replacing object assets from real-world scans to achieve Real-to-Sim transition. It proposes a multimodal alignment model, Scan2Sim, for automated asset selection, validating the dataset's efficacy on scene synthesis and cross-domain VLN transfer tasks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D scene reconstruction"
  - "Real-to-Sim"
  - "asset replacement"
  - "multimodal alignment"
  - "embodied AI"
date: 2026-05-08
content_hash: 87543a09b01ac454
---

# MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans

**Conference**: CVPR 2025  
**arXiv**: [2505.02388](https://arxiv.org/abs/2505.02388)  
**Code**: [https://meta-scenes.github.io/](https://meta-scenes.github.io/)  
**Area**: 3D Vision  
**Keywords**: 3D scene reconstruction, Real-to-Sim, asset replacement, multimodal alignment, embodied AI

## TL;DR
MetaScenes constructs a large-scale simulatable 3D scene dataset (15,366 object instances across 831 categories) by automatically replacing object assets from real-world scans to achieve Real-to-Sim transition. It proposes a multimodal alignment model, Scan2Sim, for automated asset selection, validating the dataset's efficacy on scene synthesis and cross-domain VLN transfer tasks.

## Background & Motivation

1. **Background**: Embodied AI (EAI) research highly relies on high-quality 3D scenes to support skill learning, Sim2Real transfer, and generalization. Existing methods predominantly count on manually designed scene assets by artists, which is labor-intensive and lacks scalability.
2. **Limitations of Prior Work**: Existing datasets (e.g., Scan2CAD, ReplicaCAD) face two core issues: insufficient diversity of available assets (ShapeNet only covers 35-110 categories) and the difficulty of automating the trade-off between geometric/textural accuracy and properties during the replacement process.
3. **Key Challenge**: Daily objects exhibit extremely high diversity (particularly small items), while available CAD asset libraries are restricted, making "imperfect replacement" a common occurrence with no systematic replacement guidelines.
4. **Goal**: (1) How to construct diverse, realistic, and interactive simulatable scenes at scale? (2) How to automatically select the optimal replacement asset? (3) How to validate the utility of these scenes for EAI tasks?
5. **Key Insight**: Leverage foundation models (GPT-4V, SAM) to generate rich descriptions, and then obtain diverse candidate assets through three pathways: Text-to-3D, Image-to-3D, and Text-to-3D retrieval, followed by learning an automatic selection model based on human-annotated rankings.
6. **Core Idea**: Realize an automated pipeline from real-world scans to simulatable scenes through multi-source asset candidates + manual ranking annotation + multimodal alignment learning.

## Method

### Overall Architecture
The input consists of ScanNet real-world 3D scans (706 rooms) processed through three phases: (1) Collection—gathering diverse 3D candidate assets for each scanned object; (2) Annotation—manually ranking candidate assets and placing them into the scene; (3) Optimization—physics-based optimization to ensure interactive plausibility. The final output is simulated 3D scene replicas. Based on this pipeline, the Scan2Sim model is trained to achieve automation.

### Key Designs

1. **Object Asset Curation**:

    - **Function**: Generate highly diverse and high-quality candidate replacement assets for each scanned object.
    - **Mechanism**: First, choose the 2D viewpoint with the least occlusion using depth maps, segment the object with SAM, and generate detailed textual descriptions (textures, colors, physical attributes) via GPT-4V. Then, obtain candidates through three pathways: Text-to-3D (Shape-E), Image-to-3D (TripoSR, InstantMesh, Michelangelo), and Text-to-3D retrieval (retrieved from Objaverse using Uni3D and ULIP). Finally, refine textures using Paint3D. Each object receives at least 6 candidates, totaling 98,423 unique assets.
    - **Design Motivation**: CAD assets from a single source lack diversity. The multi-source strategy (retrieval + generation) significantly boosts candidate diversity and quality while leveraging foundation models to automate description generation, avoiding manual annotation bottlenecks.

2. **Scan2Sim Multimodal Alignment Model (Optimal Asset Retrieval)**:

    - **Function**: Automatically select the best-matching replacement asset from the candidate asset pool.
    - **Mechanism**: Construct a quadruple $\langle I_i, T_i, \mathbb{P}_i, y_i \rangle$ (image, text, candidate point cloud set, optimal label) for each object. Extract features $h^I, h^T$ using frozen image/text encoders, and extract candidate point cloud features $h^P$ with a learnable 3D encoder. Calculate matching scores $q^r = [\langle h^P_{i,k}, h^r_i \rangle]$, and sum the three-way scores for supervision under a softmax cross-entropy loss $\mathcal{L}_{match}$. An auxiliary loss $\mathcal{L}_{aux}$ is added by randomly sampling negative samples from different scenes to enhance cross-scene alignment capabilities.
    - **Design Motivation**: Human-ranked annotations provide supervision signals of "human preference", enabling the model to learn subtle judgments regarding geometric similarity, texture matching, and functional equivalence, which existing general alignment models (CLIP, ULIP-2) fail to achieve.

3. **Physics-based Optimization**:

    - **Function**: Ensure the physical plausibility of object placements after replacement.
    - **Mechanism**: First, construct a hierarchical scene graph from the scene point cloud to encode spatial relationships (support, inclusion, embedding). Then, optimize object poses via MCMC sampling, considering both scene graph constraints and physical collisions. Finally, append physical attributes (mass, material, elasticity) in Blender. The scene graph accuracy reaches 96.3% via manual verification.
    - **Design Motivation**: Simple pose alignment (translation + scale + rotation) cannot guarantee physical realism. Global constraint optimization is required to handle spatial relations between objects, particularly the placement of small items.

### Loss & Training
- Main loss: $\mathcal{L} = \mathcal{L}_{match} + \mathcal{L}_{aux}$, where $\mathcal{L}_{match}$ is the cross-entropy loss for annotated ranking, and $\mathcal{L}_{aux}$ is the auxiliary alignment loss using cross-scene negative sampling.
- Pose alignment uses a heuristic method: center alignment $\rightarrow$ scaling on the longest edge $\rightarrow$ 30-degree interval rotation search for the optimal angle.

## Key Experimental Results

### Main Results

| Method | Modality (Input $\rightarrow$ Candidate) | Top-1 Acc (%) | Top-5 Acc (%) | CD $\downarrow$ | IoU $\uparrow$ |
|------|----------------|-------------|-------------|-----|------|
| ULIP-2 | I+T $\rightarrow$ P | 13.1 | 57.7 | 0.20 | 0.49 |
| CLIP | T $\rightarrow$ I | 14.9 | 66.6 | 0.21 | 0.51 |
| GPT-4V | T $\rightarrow$ I | 16.5 | 59.9 | 0.19 | 0.52 |
| **Scan2Sim** | **I+T $\rightarrow$ P** | **28.4** | **76.0** | **0.17** | **0.60** |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| I $\leftrightarrow$ I (SSIM/LPIPS) | Top-1 only 5.9-6.3%; 2D images struggle to capture 3D geometry |
| P $\leftrightarrow$ P (PointBert/PointNet++) | Top-1 9.5-11.8%; large distribution discrepancy between scan point clouds and asset point clouds |
| T $\rightarrow$ P (ULIP-2) | Top-1 14.3%; large-scale pre-training is helpful but insufficient |
| Scan2Sim (I+T $\rightarrow$ P) | **Top-1 28.4%**; supervision signals from ranking annotations are crucial |

### Key Findings
- Scan2Sim outperforms the strongest baseline (GPT-4V) by 11.9 percentage points in Top-1 accuracy, indicating that the value of domain-specific annotated data far exceeds that of general large models.
- Unimodal alignment methods generally underperform multimodal ones, with the image-to-image approach performing the worst (around 6%), as a single 2D viewpoint cannot fully represent 3D structures.
- The average CD of replaced objects in MetaScenes is 0.25, significantly better than Scan2CAD's 0.35.
- In cross-domain VLN transfer, models trained on MetaScenes show a 6.4% improvement in SPPL on ScanNet++.

## Highlights & Insights
- **Multi-source asset strategy**: Leverages retrieval + Text-to-3D + Image-to-3D simultaneously to maximize candidate diversity. This paradigm can be integrated into any 3D scene generation task.
- **Human preference learning**: Learns "what makes a good replacement" from manual ranking annotations, which aligns closer to practical needs than simple geometric matching. This is conceptually similar to RLHF, utilizing human preferences to define "optimality."
- **Scene graph + MCMC physical optimization**: An spatial relationship accuracy of 96.3% guarantees physical plausibility, serving as a critical step in the Real-to-Sim pipeline.
- **Dataset scale & annotation quality**: Covers 15,366 object instances across 831 categories, totaling 98,423 assets with 6+ candidates per object. The granularity of annotations (ranking instead of binary classification) is unprecedented in the Real-to-Sim domain.

## Limitations & Future Work
- Scan2Sim achieves a Top-1 accuracy of only 28.4%, leaving a large gap to full automation, which indicates that fine-grained object matching in multimodal alignment remains challenging.
- The dataset is built upon ScanNet (706 rooms), in which room types and geographic diversity are limited; future work can scale this up to larger scan repositories such as ScanNet++.
- Physical optimization relies on MCMC sampling, which might lack the efficiency required to support massive-scale scene generation.
- The replacement accuracy for small items has not been evaluated independently, although they are precisely the most challenging part.
- The potential of generative assets (e.g., direct conditional generation using 3D generation models) as replacement candidates has not been explored.
- The cost of manual ranking annotations remains high; active learning or human-in-the-loop annotation paradigms could be explored to alleviate the annotation burden.

## Related Work & Insights
- **vs Scan2CAD**: Scan2CAD relies solely on ShapeNet (35 categories), whereas MetaScenes utilizes Objaverse + generative models (831 categories), elevating asset diversity by over 20-fold. Additionally, Scan2CAD lacks candidate ranking annotations, preventing the training of automated selection models.
- **vs HSSD-200**: HSSD depends on Floorplanner's asset library and artist designs; although the scenes are exquisite, the cost is prohibitive, and they cannot leverage real-world scans. MetaScenes starts from real-world scans, preserving realistic layout information with greater scalability.
- **vs ACDC**: ACDC uses foundation models (Dino-V2) for matching but lacks training signals, yielding only 12.3% Top-1 accuracy in complex scenes; Scan2Sim learns a more precise preference model through ranking annotations, pushing Top-1 accuracy to 28.4%.
- **vs R3DS**: R3DS uses ShapeNet + Wayfair assets on Matterport3D, covering 110 categories but without reconstructed assets; MetaScenes is the first to introduce Image-to-3D reconstructed assets as candidates.

## Rating
- Novelty: ⭐⭐⭐⭐ The pipeline design of multi-source candidates + ranking learning is novel, though individual components are built on existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons and validations on two downstream tasks are provided, but the ablation study could be deeper.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, logical motivations, and excellent charts/visualizations.
- Value: ⭐⭐⭐⭐⭐ Exceptional dataset contribution (15,366 objects + 98,423 candidate assets), directly driving progress in the EAI domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](../../ICCV2025/3d_vision/demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[CVPR 2025\] Seeing A 3D World in A Grain of Sand](seeing_a_3d_world_in_a_grain_of_sand.md)
- [\[ECCV 2024\] FastCAD: Real-Time CAD Retrieval and Alignment from Scans and Videos](../../ECCV2024/3d_vision/fastcad_real-time_cad_retrieval_and_alignment_from_scans_and_videos.md)

</div>

<!-- RELATED:END -->
