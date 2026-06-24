---
title: >-
  [Paper Note] RelationField: Relate Anything in Radiance Fields
description: >-
  [CVPR 2025][3D Vision][Neural Radiance Fields] RelationField introduces object-to-object relationship modeling to neural radiance fields for the first time. By distilling relationship knowledge from multimodal large language models (such as GPT-4o) into an implicit relationship feature head in NeRF, it enables open-vocabulary 3D scene relationship querying and scene graph generation, significantly outperforming existing methods on the 3DSSG benchmark.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Neural Radiance Fields"
  - "relationship understanding"
  - "3D scene graphs"
  - "open-vocabulary"
  - "knowledge distillation"
date: 2026-05-08
content_hash: cd8cafeec9fb3f32
---

# RelationField: Relate Anything in Radiance Fields

**Conference**: CVPR 2025  
**arXiv**: [2412.13652](https://arxiv.org/abs/2412.13652)  
**Code**: [https://relationfield.github.io/](https://relationfield.github.io/)  
**Area**: 3D Vision  
**Keywords**: Neural Radiance Fields, relationship understanding, 3D scene graphs, open-vocabulary, knowledge distillation

## TL;DR

RelationField introduces object-to-object relationship modeling to neural radiance fields for the first time. By distilling relationship knowledge from multimodal large language models (such as GPT-4o) into an implicit relationship feature head in NeRF, it enables open-vocabulary 3D scene relationship querying and scene graph generation, significantly outperforming existing methods on the 3DSSG benchmark.

## Background & Motivation

**Background**: In recent years, Neural Radiance Fields (NeRFs) have expanded from purely novel view synthesis to semantic scene understanding. Works like LERF and OpenNeRF have achieved open-vocabulary object segmentation and detection by distilling features from vision-language foundation models (e.g., CLIP, DINO, SAM) into 3D. Meanwhile, 3D scene graphs, as a compact scene representation, can simultaneously capture objects and their relationships in a scene.

**Limitations of Prior Work**: Existing methods distilling semantic features into radiance fields (e.g., LERF, LangSplat) focus primarily on object-centric semantic features and only perform object-level segmentation and detection. They fail to understand complex relationships between objects (such as "the light switch controls the lamp" or "the pillow lies on the sofa"). Conversely, existing 3D scene graph methods (e.g., Open3DSG) rely on explicit 3D representations (point clouds or meshes) and depth sensor data, limiting their applicability.

**Key Challenge**: Radiance fields only predict attributes like color and density for each 3D point, which naturally does not support relationship modeling between two objects—as relationships require considering two spatial locations simultaneously. Directly using 2D multimodal LLMs for frame-by-frame relationship reasoning suffers from occlusion and viewpoint changes, lacking 3D consistency.

**Goal**: To model open-vocabulary object-to-object relationships within neural radiance fields, enabling them to answer relationship queries such as "what is standing on the shelf" or "which object is similar to another object," and directly extract 3D scene graphs from the radiance fields.

**Key Insight**: The authors observe that while CLIP is adept at object-level semantics, its relationship understanding is limited, whereas multimodal LLMs (like GPT-4o) possess strong relationship reasoning capabilities but only operate in 2D. Distilling the relationship knowledge of LLMs into 3D representations could combine the strengths of both.

**Core Idea**: By introducing an additional "query position" $\mathbf{z}$ to extend the NeRF input, the radiance field is enabled to predict relationship features between any two locations. Set-of-Mark prompts are then used to extract relationship annotations from GPT-4o to supervise training.

## Method

### Overall Architecture

RelationField is built on the Nerfacto model, taking a set of posed RGB images as input and outputting a rich 3D representation that supports object and relationship queries. The overall pipeline consists of three stages: (1) extracting 2D object-relationship annotations from training views via SoM-prompted GPT-4o; (2) distilling relationship knowledge into the implicit relationship feature head of the radiance field; and (3) interactively querying relationships by specifying query locations and textual relationship descriptions, or automatically constructing 3D scene graphs.

The model adds three extra output heads on top of standard NeRF: a 768-dimensional open-vocabulary semantic head with CLIP/OpenSeg embeddings, a 256-dimensional instance grouping head, and a 512-dimensional relationship feature head embedded in the jina-embeddings-v3 space.

### Key Designs

1. **Implicit Relationship Feature Prediction Head (Relationship Field)**:

    - **Function**: Encodes feature representations for relationships between any two 3D positions in the radiance field.
    - **Mechanism**: Standard NeRF takes $(\mathbf{x}, \mathbf{d})$ (position and direction) as input. RelationField additionally introduces a query position $\mathbf{z} \in \mathbb{R}^3$, changing the model function to $g_\theta(\mathbf{x}, \mathbf{d}, \mathbf{z}) \mapsto (\mathbf{c}, \sigma, \mathbf{o}, \mathbf{r})$, where $\mathbf{r}$ is the relationship feature. Implementation-wise, the ray sampling point and the query position are concatenated and fed into an MLP head to output a 512-dimensional relationship feature vector. This feature resides in a language embedding space, allowing it to be matched with any textual relationship description via cosine similarity.
    - **Design Motivation**: Relationships are inherently attributes between two entities and require two spatial positions as inputs to be defined. By taking the "query position" as an additional input, pairwise relationships are elegantly formulated as a conditional generation problem in the radiance field, preserving the advantages of continuous volume representation.

2. **SoM-based Relationship Knowledge Extraction**:

    - **Function**: Extracts pixel-aligned, dense relationship feature supervision signals from multimodal LLMs.
    - **Mechanism**: First, SAM is used to extract object masks for each training image. Then, Set-of-Mark (SoM) annotations (letters/number tags) are overlaid on the images, which are fed into GPT-4o to prompt it to identify relationships between neighboring object pairs (e.g., "A stands on B"). The textual description $t_{ij}$ output by the LLM is encoded into a high-dimensional feature $\phi_{t_{ij}}$ using jina-embeddings-v3, and then projected to the image plane via the SAM masks to obtain pixel-level relationship feature annotations.
    - **Design Motivation**: Models like CLIP excel at object understanding but struggle with relationship understanding, whereas GPT-4o possesses strong spatial and relationship reasoning capabilities. The SoM prompting technique significantly enhances the LLM's visual grounding ability, enabling it to accurately associate marked objects and describe relationships.

3. **Pairwise Pixel Sampling and Rendering Training**:

    - **Function**: Efficiently trains the relationship feature field.
    - **Mechanism**: During training, a "pairwise pixel sampler" is used to uniformly and randomly sample pairs of rays and query positions from the training views. The density prediction of the radiance field is utilized to estimate the depth of the query positions along the rays, and the ray and query samples are concatenated and sent to the relationship MLP head. The cosine similarity between the rendered relationship features and the ground-truth relationship features is maximized, which means minimizing the loss:
    $$\mathcal{L} = 1 - \frac{\mathbf{r}}{||\mathbf{r}||_2} \cdot \frac{\hat{\mathbf{r}}}{||\hat{\mathbf{r}}||_2}$$
    - **Design Motivation**: The pairwise sampling strategy ensures training efficiency while fully covering relationships between different object pairs in the scene. Using the rendering weights of the radiance field to aggregate 3D relationship features into 2D ensures multi-view consistency.

### Loss & Training

The total loss consists of three parts: (1) color reconstruction loss of standard NeRF; (2) cosine similarity loss of CLIP semantic features for object-level semantic learning; and (3) cosine similarity loss of relationship features. The instance grouping head is trained in a contrastive learning manner, forcing rays of the same instance to cluster in the embedding space. During querying, a pairwise softmax is used to compare with canonical phrases ("and", "next to", "none") to output relationship confidence.

## Key Experimental Results

### Main Results

3D scene graph prediction on the 3DSSG dataset (RIO10 subset):

| Method | Object R@5 | Object R@10 | Predicate R@3 | Predicate R@5 | Relation R@50 | Relation R@100 |
|------|-----------|-------------|--------------|--------------|--------------|----------------|
| GPT-4 (2D+depth) | 0.34 | 0.42 | 0.55 | 0.58 | 0.52 | 0.54 |
| Open3DSG | 0.56 | 0.61 | 0.58 | 0.65 | 0.55 | 0.56 |
| ConceptGraphs | 0.37 | 0.46 | 0.74 | 0.79 | 0.69 | 0.71 |
| **RelationField** | **0.69** | **0.80** | **0.76** | **0.82** | **0.73** | **0.74** |

### Ablation Study

Relationship-guided 3D instance segmentation (ScanNet++):

| Method | IoU | Accuracy |
|------|-----|----------|
| LERF | 0.25 | 0.50 |
| OpenNeRF | 0.45 | 0.83 |
| LangSplat | 0.49 | 0.87 |
| **RelationField** | **0.53** | **0.96** |

3D consistency ablation: directly reasoning relationships on 2D with GPT-4 vs. 3D distillation of RelationField shows that the latter improves Object R@5 from 0.34 to 0.69, and Predicate R@3 from 0.55 to 0.76. Replacing GPT-4o with Llama 3.2 as the relationship extractor results in only a slight drop in relationship prediction recall, proving that the method is insensitive to the LLM backend.

### Key Findings

- 3D distillation significantly outperforms 2D frame-by-frame reasoning because 2D methods are severely affected by occlusions and viewpoint changes (partially visible objects can be missed by GPT-4), whereas 3D representations achieve consistent understanding by aggregating multi-view information.
- RelationField achieves 96% accuracy on the relationship-guided segmentation task, far exceeding methods that only use CLIP features (50-87%), demonstrating that bag-of-words CLIP embeddings cannot distinguish duplicate/identical objects via relationships.
- Replacing closed-source GPT-4o with open-source Llama 3.2 results in minimal performance loss, demonstrating the generalizability of the method.

## Highlights & Insights

- **Modeling relationships as conditional queries in radiance fields**: By extending the NeRF input with an additional query position $\mathbf{z}$, the relationships between object pairs are elegantly encoded as part of the volumetric representation. This concept can be generalized to any implicit representation requiring the modeling of pairwise or multivariate relations.
- **SoM + LLM relationship knowledge extraction pipeline**: Transforming the textual reasoning capability of LLMs into pixel-level dense feature supervision represents a general paradigm for extracting structured knowledge from large models.
- **First to generate 3D scene graphs from radiance fields**: Demonstrates that high-quality, open-vocabulary 3D scene graphs can be constructed using only RGB images (without requiring depth sensors or explicit 3D representations).

## Limitations & Future Work

- The relationship knowledge relies entirely on the quality of the LLM's prompted outputs. If the LLM misunderstands a specific relationship or if the prompt is poorly designed, errors will propagate into the 3D relationship field.
- It requires known camera intrinsics and high-quality multi-view capture, which is not easily satisfied in many practical application scenarios.
- High training cost: Querying GPT-4o for every training image to extract relationships incurs significant API costs and time overhead.
- The quality upper bound of the relationship field is constrained by the reconstruction quality of the underlying radiance field; scenes with poor reconstruction will also suffer from poor relationship understanding.
- Future work could explore extending the relationship field to 3D Gaussian Splatting for real-time performance.

## Related Work & Insights

- **vs Open3DSG**: Open3DSG distills CLIP+InstructBLIP into a 3D graph neural network but relies on pre-given class-agnostic instance segmentation and explicit 3D meshes; RelationField does not require explicit 3D representations and trains directly from RGB images.
- **vs ConceptGraphs**: ConceptGraphs also uses GPT-4 but integrates it with a SLAM pipeline, reconstructing first and then using GPT-4 to label scene-level captions; RelationField directly distills relationship knowledge into a continuous volumetric representation, enabling end-to-end relationship learning.
- **vs LERF/LangSplat**: These methods focus on object-level CLIP feature distillation and cannot handle complex relationship queries; RelationField extends the capability of feature fields through an additional relationship head.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulates open-vocabulary relationships within radiance fields for the first time; the extended input with query locations is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features two quantitative tasks (scene graph prediction and relationship-guided segmentation) and ablations covering 3D consistency and LLM choice, though the newly proposed relationship-guided segmentation benchmark is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning, intuitive illustrations, and logical coherence from motivation through methods to experiments.
- Value: ⭐⭐⭐⭐ Opens up a new direction for relationship modeling in 3D scene understanding, with potential applications in robotics interaction and AR/VR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](../../NeurIPS2025/3d_vision/hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)

</div>

<!-- RELATED:END -->
