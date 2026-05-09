---
title: >-
  [Paper Note] Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes
description: >-
  [ICCV 2025][Segmentation][Salient Object Detection] This paper constructs USC12K, the first unconstrained dataset for salient and camouflaged object detection covering four scene types, proposes USCNet built upon SAM, introduces an Attribute Relationship Modeling (ARM) module to explicitly model the relationship between salient and camouflaged objects, and designs a new metric CSCS to quantify confusion between the two categories, achieving state-of-the-art performance across all scene types.
tags:
  - ICCV 2025
  - Segmentation
  - Salient Object Detection
  - Camouflaged Object Detection
  - Unconstrained Scenes
  - Attribute Relationship Modeling
  - SAM
date: 2026-05-08
content_hash: 9f1579343a9fc884
---

# Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes

**Conference**: ICCV 2025
**arXiv**: [2412.10943](https://arxiv.org/abs/2412.10943)
**Code**: [GitHub](https://github.com/)
**Area**: Image Segmentation
**Keywords**: Salient Object Detection, Camouflaged Object Detection, Unconstrained Scenes, Attribute Relationship Modeling, SAM

## TL;DR
This paper constructs USC12K, the first unconstrained dataset for salient and camouflaged object detection covering four scene types, proposes USCNet built upon SAM, introduces an Attribute Relationship Modeling (ARM) module to explicitly model the relationship between salient and camouflaged objects, and designs a new metric CSCS to quantify confusion between the two categories, achieving state-of-the-art performance across all scene types.

## Background & Motivation

Salient Object Detection (SOD) and Camouflaged Object Detection (COD) are two related yet opposing tasks in computer vision—SOD targets the most visually prominent objects, while COD targets objects that blend into their surroundings. Both have important applications in medical anomaly detection, autonomous driving obstacle recognition, and military reconnaissance.

**Root Cause: Existing models cannot distinguish between salient and camouflaged objects.** The paper identifies a counterintuitive phenomenon: SOD models still achieve relatively high detection scores on COD datasets (e.g., ICON achieves $F_\beta^\omega=0.6384$ on COD10K), and COD models perform similarly on SOD datasets (e.g., SINet-V2 achieves 0.7412 on DUTS). That is, SOD models misclassify camouflaged objects as salient, and COD models misclassify salient objects as camouflaged.

**Two fundamental causes:**

**Flawed dataset annotation paradigm**: Existing SOD/COD datasets impose mutually exclusive constraints—assuming a scene contains either salient or camouflaged objects, but not both. This does not reflect the real world, where a scene may contain both types simultaneously, only one type, or neither. This constrained annotation causes camouflaged objects in COD datasets and salient objects in SOD datasets to be treated as background, leading to annotation conflicts.

**Lack of explicit attribute relationship modeling in models**: Existing SOD/COD models learn the two tasks independently. Even unified models (e.g., VSCode, UJSC) establish connections only indirectly through contrastive learning, without explicitly modeling the intra-sample salient–camouflaged relationship.

**Core Idea**: (1) Construct the USC12K dataset covering four scene types to eliminate data constraints; (2) Design the ARM module to explicitly model attribute relationships from both inter-sample (Inter-SPQ) and intra-sample (Intra-SPQ) perspectives.

## Method

### Overall Architecture
USCNet is built upon SAM and comprises three main components: (1) a SAM image encoder with Adapter layers; (2) an Attribute Relationship Modeling (ARM) module that generates attribute prompts for salient, camouflaged, and background categories; and (3) a frozen SAM mask decoder that predicts three types of masks based on the attribute prompts.

### Key Designs

1. **USC12K Dataset (Four-Scene Unconstrained Dataset)**:

    - Function: Constructs the first dataset without constraints on the presence of salient or camouflaged objects.
    - Core Design: 12,000 images covering four scene types—
        - **Scene A**: Salient objects only (3,000 images, sourced from DUTS and HKU-IS)
        - **Scene B**: Camouflaged objects only (3,000 images, sourced from COD10K and CAMO)
        - **Scene C**: Both salient and camouflaged objects co-existing (3,000 images, including 2,617 collected from the web)
        - **Scene D**: Background scenes containing neither salient nor camouflaged objects (3,000 images)
    - Annotation: Covers 9 super-categories and 179 sub-categories, with coarse annotations by SAM followed by manual refinement. The training set contains 8,400 images and the test set 3,600.
    - Design Motivation: Only by covering all logical combinations of scene types can a model truly learn to distinguish salient from camouflaged objects.

2. **Attribute Relationship Modeling (ARM) Module**:

    - Function: Explicitly models the relationships among salient, camouflaged, and background attributes.
    - Mechanism: Two complementary prompt query mechanisms are designed—
        - **Inter-SPQ (Inter-sample Prompt Query)**: A set of learnable query embeddings $Q_{S_r}, Q_{C_r}, Q_{B_r} \in \mathbb{R}^{N \times C}$ that remain fixed during inference. These capture universal discriminative features across samples (e.g., statistical patterns of size, position, color, and texture).
        - **Intra-SPQ (Intra-sample Prompt Query)**: Dynamically generated from encoder features $F$ and varies with each input sample. Attention maps are generated via an attention head $\Phi_{AH}$ (supervised by ground truth) to extract sample-specific attribute features:
       $$[Q_{S_a}, Q_{C_a}, Q_{B_a}] = \text{Linear}(\sigma(\Phi_{AH}(F)) \otimes F)$$
        - The two are summed and then passed through self-attention, Query-to-Image cross-attention, and MLP to generate the final attribute prompts $P$:
       $$P = \text{MLP}(\text{Q2I}(\text{SA}(\text{Intra-SPQ} + \text{Inter-SPQ}), F))$$
    - Design Motivation: Inter-SPQ learns global universal patterns (e.g., "salient objects tend to have vivid colors"), while Intra-SPQ focuses on specific relationships within a single sample (e.g., "in this image, the flower is salient while the butterfly is camouflaged"), making the two complementary.

3. **Frozen SAM Mask Decoder**:

    - Function: Predicts three masks $M_S, M_C, M_B$ based on the three attribute prompts.
    - Mechanism: $[M_S, M_C, M_B] = \text{MaskDe}([P_S, P_C, P_B], F)$, with softmax applied to generate the final predictions.
    - Design Motivation: Leverages SAM's powerful segmentation capability, requiring only attribute-specific prompts to guide outputs for different categories.

4. **CSCS Metric (Camouflage–Salient Confusion Score)**:

    - Function: Quantifies the degree to which a model confuses salient and camouflaged objects.
    - Formula: $\text{CSCS} = \frac{1}{2}(\frac{\mathcal{P}_{CS}}{\mathcal{P}_{BS}+\mathcal{P}_{SS}+\mathcal{P}_{CS}} + \frac{\mathcal{P}_{SC}}{\mathcal{P}_{BC}+\mathcal{P}_{SC}+\mathcal{P}_{CC}})$
    - Here $\mathcal{P}_{CS}$ denotes the proportion of camouflaged regions predicted as salient, and $\mathcal{P}_{SC}$ the proportion of salient regions predicted as camouflaged. Lower CSCS is better.
    - Design Motivation: Existing metrics (e.g., weighted F-measure) only evaluate foreground–background separation and cannot measure the degree of confusion between the two object categories.

### Loss & Training
- Total loss: $\mathcal{L}_{Total} = \lambda_p \mathcal{L}_{pred.} + \lambda_a \mathcal{L}_{att.}$
- Both prediction loss and attention loss use Focal Loss: $\mathcal{L}_{focal} = -\frac{1}{N}\sum_{i=1}^{N}\alpha_{t_i}(1-p_{t_i})^\gamma \log(p_{t_i})$
- Class weight ratio — background : salient : camouflaged = 1:4:6 (proportional to pixel counts), $\gamma=2$
- $\lambda_p=1, \lambda_a=0.5$
- SAM2 hiera-large backbone, AdamW optimizer, lr=0.0001, batch size=24, max 90 epochs

## Key Experimental Results

### Main Results
Comprehensive evaluation on four scenes of USC12K (compared against 21 related methods):

| Method | Type | IoU_S↑ | IoU_C↑ | mIoU↑ | mAcc↑ | CSCS↓ |
|--------|------|--------|--------|-------|-------|-------|
| PGNet (SOD) | SOD | 74.69 | 57.31 | 71.82 | 80.76 | 7.71 |
| CamoFormer (COD) | COD | 75.88 | 66.19 | 74.81 | 84.17 | 7.57 |
| VSCode (Unified) | Unified | 76.04 | 60.31 | 74.17 | 84.01 | 8.17 |
| SAM2-Adapter | Adapter | 78.75 | 70.28 | 74.98 | 84.74 | 9.12 |
| **USCNet (Ours)** | Unified | **79.70** | **74.99** | **78.03** | **87.92** | **7.49** |

### Ablation Study
Contribution of each component in the ARM module (Overall Scenes):

| Encoder | Decoder | Intra-S | Inter-S | Q2I | I2Q | mIoU | CSCS↓ |
|---------|---------|---------|---------|-----|-----|------|-------|
| Frozen | Tuning | ✗ | ✗ | ✗ | ✗ | 68.78 | 11.58 |
| Tuning | Tuning | ✗ | ✗ | ✗ | ✗ | 74.98 | 9.12 |
| Tuning | Frozen | ✗ | ✔ | ✔ | ✔ | 75.31 | 9.07 |
| Adapter | Frozen | ✔ | ✔ | ✔ | ✔ | **78.03** | **7.49** |

False detection rate improvement (after vs. before USC12K training):

| Model / Test Set | Before $F_\beta^\omega$ | After $F_\beta^\omega$ | Note |
|------------------|------------------------|------------------------|------|
| ICON → COD10K | 0.6384 | 0.0146 | SOD false detection of COD substantially reduced |
| SINet-V2 → DUTS | 0.7412 | 0.0708 | COD false detection of SOD substantially reduced |

### Key Findings
- Scene C (co-existence of both object types) is the most challenging scenario, with all models performing below single-attribute scenes.
- USCNet achieves the best performance with only 4.04M trainable parameters, the smallest among compared methods.
- After training on USC12K, false detection scores drop from 0.6–0.8 to 0.01–0.07, nearly eliminating cross-category confusion.
- USCNet also demonstrates generalization on conventional SOD/COD datasets (DUTS, HKU-IS, NC4K, COD10K).

## Highlights & Insights
- **Deep problem insight**: The paper systematically identifies and validates the cross-task false detection phenomenon in SOD/COD models, revealing that the root cause lies in the data annotation paradigm rather than model capacity.
- **Sound dataset design philosophy**: Moving from "constrained" to "unconstrained" settings, the four scene types cover all logical combinations of salient and camouflaged object presence.
- **Elegant ARM module design**: Inter-SPQ captures universal attribute differences across samples, while Intra-SPQ focuses on sample-specific relationships, with the two being mutually complementary.
- **CSCS fills an evaluation gap**: Existing metrics cannot measure salient–camouflaged confusion; this new metric enables more targeted evaluation.
- Exceptionally high parameter efficiency—only 4.04M trainable parameters.

## Limitations & Future Work
- The 2,617 web-collected images in Scene C may have lower annotation quality compared to professionally curated datasets.
- The current model enforces a strict three-way categorization (salient / camouflaged / background), without accounting for intermediate states such as "semi-camouflaged" objects.
- The fixed class weight ratio (1:4:6) may not be optimal for all data distributions.
- The approach is limited to 2D static images and has not been extended to video scenarios.
- The dataset scale (12K images) is relatively modest, potentially limiting model generalization in more complex scenes.

## Related Work & Insights
- The concept of unconstrained detection can be extended to other opposing task pairs, such as "complete object vs. occluded object" detection.
- The Inter/Intra-SPQ design philosophy of the ARM module is applicable to other multi-attribute classification and segmentation tasks.
- The paradigm of using SAM as a general segmentation backbone with task-specific prompts retains significant potential in specialized domains.
- The design rationale behind CSCS—measuring inter-class confusion rather than foreground–background separation—offers valuable methodological insights.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel problem formulation (unconstrained SOD+COD); dataset, model, and metric are all first of their kind.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison with 21 methods, four-scene evaluation, extensive ablation and generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and persuasive, though some sections are dense.
- Value: ⭐⭐⭐⭐ Advances the unification of SOD and COD; the dataset and benchmark are expected to have long-term impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](../../CVPR2026/segmentation/discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[CVPR 2026\] Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification](../../CVPR2026/segmentation/detecting_ai-generated_forgeries_via_iterative_manifold_deviation_amplification.md)

<!-- RELATED:END -->
