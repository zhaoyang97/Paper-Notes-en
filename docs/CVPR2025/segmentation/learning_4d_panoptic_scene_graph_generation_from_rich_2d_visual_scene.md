---
title: >-
  [Paper Note] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene
description: >-
  [CVPR 2025][Segmentation][4D Panoptic Scene Graph] This paper proposes a 4D panoptic scene graph generation framework based on 4D-LLMs and 2D-to-4D transfer learning. By utilizing chained scene graph inference, it leverages the open-vocabulary capabilities of LLMs and transfers dimension-invariant features from abundant 2D scene annotations to 4D scenes, significantly mitigating issues of data scarcity and limited vocabulary.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "4D Panoptic Scene Graph"
  - "Scene Graph Generation"
  - "Transfer Learning"
  - "Large Language Models"
  - "3D Scene Understanding"
date: 2026-05-08
content_hash: a7f5b5d5d763a0c8
---

# Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene

**Conference**: CVPR 2025  
**arXiv**: [2503.15019](https://arxiv.org/abs/2503.15019)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: 4D Panoptic Scene Graph, Scene Graph Generation, Transfer Learning, Large Language Models, 3D Scene Understanding

## TL;DR
This paper proposes a 4D panoptic scene graph generation framework based on 4D-LLMs and 2D-to-4D transfer learning. By utilizing chained scene graph inference, it leverages the open-vocabulary capabilities of LLMs and transfers dimension-invariant features from abundant 2D scene annotations to 4D scenes, significantly mitigating issues of data scarcity and limited vocabulary.

## Background & Motivation

**Background**: The 4D Panoptic Scene Graph (4D-PSG) is a recently proposed high-level representation for modeling the dynamic 4D real world. It encodes objects and inter-object relationships in 3D point cloud sequences into a graph structure, where nodes represent object instances (including 3D masks and semantic labels) and edges represent relationships (e.g., "person sitting on a chair"). 4D-PSG comprehensively describes "what objects exist, where they are, and what happens between them" in space, and their dynamic changes over time.

**Limitations of Prior Work**: Current research on 4D-PSG faces three severe challenges: (1) Data scarcity: annotating 4D scenes is extremely expensive, and existing 4D-PSG datasets are very small, which severely limits model training performance; (2) Limited vocabulary (OOV problem): small datasets restrict the model to recognizing a limited set of object classes and relationship types, leading to failures when encountering concepts not covered in the training set; (3) Pipeline defects: existing benchmark methods employ a multi-step pipeline (first detecting objects, then predicting relationships), where errors from each step accumulate sequentially, leading to sub-optimal performance.

**Key Challenge**: A fundamental conflict exists between the scarcity of 4D scene annotations and the model's demand for large amounts of annotated data. Meanwhile, 2D scene graph annotations are highly abundant (e.g., Visual Genome contains over 100k images), but how to leverage 2D data to assist 4D tasks remains an open question.

**Goal**: To design an end-to-end 4D-PSG generation framework that simultaneously addresses data scarcity, limited vocabulary, and cumulative pipeline errors.

**Key Insight**: The authors' core observation is that semantic attributes of objects and inter-object relationships are dimension-invariant between 2D and 4D scenes. For instance, the semantic meaning of the relationship "person sitting on a chair" is identical in both a 2D image and a 3D point cloud. Therefore, it is possible to transfer these dimension-invariant semantic representations from rich 2D scene graph annotations to 4D scenes.

**Core Idea**: Assisting the training of 4D scene graph models with 2D scene graph data—achieving end-to-end generation via 4D-LLMs, and compensating for 4D data scarcity through 2D-to-4D transfer learning.

## Method

### Overall Architecture
The proposed framework comprises three main components: (1) 4D-LLM—integrating a large language model with a 3D mask decoder to achieve end-to-end 4D-PSG generation; (2) Chained Scene Graph Inference—iteratively reasoning about object and relationship labels by leveraging the open-vocabulary capability of LLMs; (3) 2D-to-4D Transfer Learning—extracting dimension-invariant features from large-scale 2D scene graph annotations and transferring them to 4D scenes via a space-time scene transcending strategy. The input is a 4D point cloud sequence (multi-timestep 3D point clouds), and the output is the complete 4D panoptic scene graph.

### Key Designs

1. **4D-LLM and 3D Mask Decoder Integration**:

    - **Function**: Simultaneously performing 3D object segmentation and scene graph generation in an end-to-end manner, eliminating cumulative errors from multi-step pipelines.
    - **Mechanism**: Multi-scale features are extracted from the 4D point cloud sequence using a 3D encoder (such as PointNet++ or Sparse3D) and projected into token sequences as input to the large language model. The LLM generates structured scene graph descriptions (object names, relationship descriptions, etc.) in an autoregressive manner. Concurrently, the hidden states of the LLM are fed into a parallel 3D mask decoder, yielding instance-level 3D masks for each object by integrating 3D features via a cross-attention mechanism. The key innovation lies in unifying the semantic reasoning capability of LLMs and the spatial segmentation capability of 3D mask decoders within a single framework.
    - **Design Motivation**: Traditional pipeline methods (segmentation followed by classification, and finally relationship reasoning) potentially introduce errors at each step and cannot backpropagate gradients. The end-to-end design allows segmentation, identification, and relationship reasoning to reinforce each other.

2. **Chained SG Inference**:

    - **Function**: Iteratively reasoning accurate and comprehensive object and relationship labels by leveraging the open-vocabulary capability of LLMs.
    - **Mechanism**: Scene graph inference is decomposed into multi-turn chained dialogues. In the first turn, the LLM is asked to describe the objects present in the scene and their attributes; in the second turn, based on the object list from the first turn, the LLM infers the relationships between object pairs. The output of each turn serves as the context input for the next. This iterative approach leverages the in-context learning capability of LLMs—given that "there are people, chairs, and tables in the scene", reasoning that "the person sits next to the chair, and the table is in front of the chair" becomes more accurate. Since a pre-trained LLM is utilized, the model naturally possesses open-vocabulary capabilities, allowing it to predict object categories and relationship types unseen in the training set.
    - **Design Motivation**: Generating a complete scene graph in a single step places overly high demands on the model's capabilities. Chained inference decomposes the complex problem into multiple simple sub-problems, each utilizing prior information to reduce difficulty. This also aligns with the human cognitive process of observing scenes.

3. **2D-to-4D Spatial-Temporal Scene Transcending**:

    - **Function**: Transferring dimension-invariant semantic knowledge from rich 2D scene graph annotations to 4D scenes.
    - **Mechanism**: The strategy consists of two parts: spatial dimension transfer and temporal dimension expansion. For spatial transfer, a scene graph reasoning head (containing parameters for object classification and relationship prediction) trained on 2D images is directly transferred to the 4D model, since the semantics of "what the object is" and "what the relationship is between objects" remain consistent across 2D/4D. For temporal expansion, a temporal aggregation module is designed to fuse single-frame 2D scene knowledge with the temporal dynamic information of 4D sequences. Specifically, a temporal attention mechanism is used to aggregate object features across timesteps to learn "how relationships change over time" (e.g., "a person transitions from standing to sitting"). The scale of 2D SG datasets (such as Visual Genome) is vastly larger than 4D-PSG data (100k+ vs. hundreds), meaning transfer learning significantly mitigates the data scarcity issue.
    - **Design Motivation**: Training from scratch on small-scale 4D datasets leads to small vocabulary and overfitting. 2D scene graphs provide a massive amount of (object, relationship) supervision signals, whose semantics are universal across dimensions.

### Loss & Training
The training strategy is divided into three stages: (1) Pre-training semantic reasoning capabilities on 2D scene graph datasets; (2) Fine-tuning on 4D datasets using a combination of 3D mask loss (BCE + Dice loss), object classification loss, and relationship prediction loss. The LLM component utilizes the standard autoregressive generation loss.

## Key Experimental Results

### Main Results

| Method | R@20 (Predicate) | R@50 (Predicate) | R@20 (Triplet) | R@50 (Triplet) |
|------|-------------------|-------------------|-----------------|-----------------|
| Ours | **38.7** | **47.2** | **22.4** | **31.6** |
| PSGFormer4D | 24.3 | 32.1 | 13.8 | 20.7 |
| 3D-SGFormer | 21.6 | 28.5 | 11.2 | 17.3 |
| PointSG | 18.9 | 25.4 | 9.7 | 14.8 |

### Ablation Study

| Configuration | R@20 (Pred) | R@50 (Pred) | Description |
|------|-------------|-------------|------|
| Full Model | 38.7 | 47.2 | Full framework |
| w/o 2D-to-4D Transfer | 28.9 | 36.4 | Without 2D data transfer, leading to a substantial drop |
| w/o Chained Inference | 33.5 | 41.8 | Changed to single-step scene graph generation |
| w/o 3D Mask Decoder | 35.1 | 43.6 | Removed mask generation, performing relationship prediction only |
| Pipeline baseline | 24.3 | 32.1 | Step-by-step pipeline (non-end-to-end) |

### Key Findings
- The 2D-to-4D transfer learning contributes the most (R@20 drops by 9.8 points without it), validating the core value of transferring semantic knowledge from 2D data.
- Chained inference contributes 5.2 points (R@20), demonstrating that the step-by-step reasoning strategy effectively reduces the complexity of scene graph generation.
- The end-to-end method comprehensively outperforms the pipeline baseline (R@20 is 14.4 points higher), proving the importance of eliminating accumulated errors.
- In OOV (out-of-vocabulary) scenarios, the proposed method performs far better than closed-vocabulary baselines due to two factors: the open-vocabulary capability of LLMs and the extensive coverage of 2D data.

## Highlights & Insights
- The concept of **2D-to-4D transfer learning** possesses strong generalizability. The insight that "dimension-invariant semantics can be transferred across dimensions" is applicable not only to scene graphs but also to tasks like 2D-to-3D object detection and action recognition. The key premise is identifying truly dimension-invariant features.
- **Chained inference** draws inspiration from the Chain-of-Thought concept of LLMs, applying it to structured prediction tasks. This step-by-step "find objects first, then reason relationships" strategy is simple yet effective, and can be transferred to other structured prediction tasks.
- The architectural design integrating LLMs with a 3D mask decoder demonstrates that LLMs are not merely text generation tools, but can also drive spatial-geometric predictions.

## Limitations & Future Work
- The labeled datasets for 4D-PSG remain very small, which cannot be thoroughly compensated for even with 2D transfer.
- The inference speed of LLMs is slow, and multi-turn chained inference further increases latency, making real-time performance a noticeable bottleneck.
- 2D-to-4D transfer assumes "dimension-invariant semantics," but certain spatial relationships (e.g., "above" vs. "in front of") may be expressed differently in 2D and 3D.
- The code is currently not open-sourced, which limits reproducibility.
- Future research can explore transfer pathways from video (2D + time) to 4D, leveraging video data to further mitigate the shortage of 4D annotations.

## Related Work & Insights
- **vs PSGFormer4D**: PSGFormer4D is a pioneering method for 4D-PSG, employing a step-by-step pipeline. The proposed end-to-end scheme comprehensively outperforms it across all metrics, with a particularly pronounced advantage in OOV scenarios.
- **vs 2D Scene Graph Generation (IMP, Neural Motifs)**: Traditional 2D SGG methods cannot handle 3D and temporal information. This work performs dimensional expansion when utilizing semantic knowledge from 2D SGG, rather than simple reuse.
- **vs 3D-LLM/PointLLM**: These methods apply LLMs to 3D understanding but do not address scene graph generation. This work expands the application scope of 3D-LLMs to structured scene graph prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of 2D-to-4D transfer, 4D-LLMs, and chained inference is presented for the first time in the 4D-PSG domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Clear comparisons are provided in both main and ablation experiments, substantially outperforming the baselines.
- Writing Quality: ⭐⭐⭐⭐ The problem formulation is clear, and the description of the methodology is highly logical.
- Value: ⭐⭐⭐⭐ Pushes the frontier of 4D scene understanding, though somewhat constrained by the niche nature of the 4D-PSG task itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scene-Centric Unsupervised Panoptic Segmentation](scene-centric_unsupervised_panoptic_segmentation.md)
- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_panoptic_scene_graph_realtime.md)
- [\[ICCV 2025\] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](../../ICCV2025/segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](../../ECCV2024/segmentation/openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
