---
title: >-
  [Paper Note] ASHiTA: Automatic Scene-grounded Hierarchical Task Analysis
description: >-
  [CVPR 2025][3D Vision][Hierarchical Task Analysis] The first framework, ASHiTA, is proposed to automatically decompose high-level tasks into hierarchies of scene-grounded subtasks. By alternating LLM-assisted hierarchical task analysis with task-driven 3D scene graph construction based on the Information Bottleneck principle, joint reasoning of the task hierarchy and scene representation is achieved.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Hierarchical Task Analysis"
  - "3D Scene Graph"
  - "Information Bottleneck"
  - "LLM Task Decomposition"
  - "Scene Understanding"
date: 2026-05-08
content_hash: 4793bdf345d8a98b
---

# ASHiTA: Automatic Scene-grounded Hierarchical Task Analysis

**Conference**: CVPR 2025  
**arXiv**: [2504.06553](https://arxiv.org/abs/2504.06553)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Hierarchical Task Analysis, 3D Scene Graph, Information Bottleneck, LLM Task Decomposition, Scene Understanding

## TL;DR

The first framework, ASHiTA, is proposed to automatically decompose high-level tasks into hierarchies of scene-grounded subtasks. By alternating LLM-assisted hierarchical task analysis with task-driven 3D scene graph construction based on the Information Bottleneck principle, joint reasoning of the task hierarchy and scene representation is achieved.

## Background & Motivation

**Background**: The field of scene reconstruction and semantic understanding has made significant progress in grounding natural language to 3D environments, with various open-vocabulary 3D scene graph methods such as ConceptGraph, HOV-SG, and CLIO emerging. These methods can associate simple, explicit instructions (e.g., "go to the kitchen", "preheat the oven") with objects in the scene.

**Limitations of Prior Work**: Existing methods cannot handle abstract, high-level instructions (e.g., "prepare dinner") because: (1) high-level instructions do not explicitly mention semantic elements in the scene; (2) the process of decomposing a high-level task into concrete subtasks is environment-dependent—decomposing "clean the office" will result in completely different steps in different offices. (3) Although LLMs can perform task decomposition, a decomposition detached from environment information is often impractical.

**Key Challenge**: Scene representation should depend on the task (retaining only task-relevant objects), but task decomposition in turn depends on what objects exist in the scene—this presents a chicken-and-egg dependency cycle.

**Goal**: Build a framework that can start from a high-level natural language task and automatically generate a complete task analysis that contains a hierarchical structure of subtasks, while being grounded in a 3D scene graph.

**Key Insight**: Generalize the Information Bottleneck (IB) principle to a hierarchical version (H-IB) to compress scene representations at different abstraction levels, while alternating bottom-up scene graph construction and top-down LLM task hierarchy refinement.

**Core Idea**: Address the cyclic dependency between task decomposition and scene representation through iterative, alternating scene hierarchy updates (compressing 3D primitives into task-aligned scene graphs based on H-IB) and task updates (refining the task hierarchy using scene information with LLMs).

## Method

### Overall Architecture

ASHiTA first constructs a base primitive layer (class-agnostic 3D semantic segmentation) from RGB-D inputs. Then, given a high-level task, it iteratively executes two core steps: (1) scene hierarchy update—compressing primitives into a multi-layer scene graph according to the current task hierarchy utilizing H-IB; (2) task update—refining the task hierarchy (adding new subtasks and objects) using an LLM and the current scene graph. The input consists of an RGB-D image sequence and a high-level task description, and the output is a complete task hierarchy grounded in a 3D scene graph.

### Key Designs

1. **Hierarchical Information Bottleneck (H-IB)**:

    - **Function**: Compress low-level primitives into multi-layer scene graph nodes aligned with the task hierarchy.
    - **Mechanism**: The classic IB searches for a compressed representation $\mathcal{S}$ of the input data that retains the maximum information about the task $\mathcal{T}$. H-IB generalizes this to multiple layers—given multi-resolution tasks $\mathcal{T}_1 \dots \mathcal{T}_n$, it solves for multi-layer compressions $\mathcal{S}_1 \dots \mathcal{S}_n$ to minimize $\sum_{k=1}^{n} I(\mathcal{S}_{k-1};\mathcal{S}_k) - \beta \sum_{k=1}^{n} I(\mathcal{T}_k;\mathcal{S}_k)$. An iterative update formula is derived under the Markov chain assumption, where $\beta=10$ controls the compression rate. The scene graph is constructed bottom-up using H-IB, followed by a top-down pruning of redundant nodes based on node confidence.
    - **Design Motivation**: Traditional IB can only perform single-layer compression, but task hierarchies are inherently multi-layer (task $\rightarrow$ subtask $\rightarrow$ object). H-IB retains the most relevant information at each level simultaneously, producing a hierarchical scene graph. Compared to recursively calling standard IB, H-IB maintains the consistency of information transmission across layers (as confirmed by the significant recall drop of recursive IB in ablation studies).

2. **LLM-Assisted Task Hierarchy Refinement**:

    - **Function**: Update the task decomposition using objects discovered in the scene that are not covered by the current task hierarchy.
    - **Mechanism**: Track the primitives assigned to subtasks during the bottom-up H-IB construction but removed during the top-down pruning. Object names are generated for these primitives via a Word Generator (using CLIP similarity to match a household object vocabulary generated by an LLM). Then, GPT-4o-mini is queried to score (0-1) the suggested objects for each subtask. Objects scoring above a threshold $r_s=0.8$ are added to the existing subtasks, while other high-scoring objects trigger the LLM to generate new subtasks.
    - **Design Motivation**: The initial task decomposition is based on the general knowledge of LLMs and lacks environment-specific context. Feeding back discoveries from the scene graph to the LLM establishes a closed-loop reasoning of "see $\rightarrow$ think $\rightarrow$ search again".

3. **Spatial-Aware Conditional Probability Update**:

    - **Function**: Utilize the spatial locations of already grounded nodes in subsequent iterations to improve H-IB input.
    - **Mechanism**: Define a spatial conditional probability $p_s(s_i|t)$: the probability is 1 when a primitive is within the radius $r$ of a task entity, and decays exponentially as $\exp(-(d-r)^2/r^2)$ when outside. The spatial probability is multiplied by the original embedding probability and normalized to serve as the input for the next round of H-IB. The position of the task entity is taken from the centroid of the aligned scene graph node, and the radius is determined by the nearest neighbor distance or the bounding box size.
    - **Design Motivation**: Once certain objects are grounded in the scene, their spatial locations serve as key prior—relevant objects tend to be spatially close. The spatial-aware probability update enables subsequent iterations to focus more accurately on task-relevant regions.

### Loss & Training

ASHiTA is a training-free inference framework. The bottom layer relies on pretrained EfficientViT (class-agnostic segmentation) and MobileCLIP (vision-language encoding), while task reasoning uses GPT-4o-mini. The convergence condition for H-IB iteration is $\mathcal{C}^{\tau} - \mathcal{C}^{\tau+1} < 10^{-8}$ or reaching a maximum of 1000 iterations.

## Key Experimental Results

### Main Results (SG3D HM3DSem Grounding)

| Method | s-acc(%) ↑ | t-acc(%) ↑ | Setup |
|------|-----------|-----------|------|
| 3D-VisTA | 25.3 | 10.3 | GT 3D Instance Segmentation |
| PQ3D | 24.4 | 9.7 | GT 3D Instance Segmentation |
| **ASHiTA** | **28.7** | **12.1** | GT 3D Instance Segmentation |
| ASHiTA + Txt Emb. | 65.4 | 39.3 | GT Segmentation + GT Label |
| Hydra + GPT | 8.2 | 2.4 | Incremental Scene Graph |
| HOV-SG + GPT | 9.0 | 2.0 | Incremental Scene Graph |
| **ASHiTA** | **21.7** | **8.8** | Incremental Scene Graph |

### Ablation Study

| Configuration | s-rec(%) | s-prec(%) | t-acc(%) |
|------|---------|-----------|---------|
| **ASHiTA (Full)** | **10.39** | **20.6** | **9.27** |
| Recursive IB | 1.51 | 24.53 | 1.46 |
| w/o Top Down Pruning | 9.22 | 18.93 | 5.37 |
| w/o Spatial Update | 8.70 | 22.22 | 6.34 |
| w/o Hierarchy Refinement | 7.71 | 23.13 | 6.83 |
| Primitives + GPT | 6.14 | 7.16 | 5.37 |

### Key Findings

- **H-IB significantly outperforms Recursive IB**: Although recursively applying standard IB yields slightly higher precision (24.53% vs 20.6%), its recall is only 1.51% (compared to 10.39% for the full model). This indicates that information fragmentation between layers leads to a large number of missed subtasks.
- **Every module contributes**: Removing any component (pruning, spatial updates, hierarchy refinement) leads to a 30-42% drop in task accuracy.
- **LLM + Scene Graph >> Pure LLM**: The precision of Primitives + GPT (which directly uses raw primitive labels with LLMs) is only 7.16%, significantly lower than ASHiTA's 20.6%. This demonstrates that the information bottleneck framework is crucial for compressing and focusing information.
- **Effective on real robots**: Real-world demonstrations on a Boston Dynamics Spot show that ASHiTA can generate reasonable task decompositions and groundings in complex scenes.

## Highlights & Insights

- **Elegant solution to cyclic dependency**: Task decomposition depends on the environment, and the environment representation depends on the task. ASHiTA optimizes both alternately, inspired by EM-like reasoning. This design pattern can be transferred to any scenario where representation and reasoning are mutually dependent.
- **Innovative generalization of Information Bottleneck**: Generalizing the classic single-layer IB to multi-layer H-IB is non-trivial, requiring handling of the inter-layer Markov assumption and convergence proofs. H-IB can serve as a general tool for any problem requiring multi-level information compression.
- **Zero-training framework**: The entire system requires no labeled data or end-to-end training. It relies entirely on a combination of pretrained vision models and LLMs, demonstrating the power of foundation model composition.

## Limitations & Future Work

- **Lack of spatial relationship modeling**: There are no relational edges like "on top of" or "inside" in the scene graph, limiting the understanding of spatial instructions.
- **Inability to handle multiple instances of the same object**: ASHiTA merges multiple identical chairs into a small number of nodes and assigns different labels, which fails when manipulating multiple identical objects is required.
- **Unguaranteed task hierarchy completeness**: The generated subtasks are not guaranteed to be sufficient for completing high-level tasks, and the tree structure restricts an object from being shared across multiple subtasks.
- **Future directions**: Introduce classic planning methods (TAMP/PDDL) to verify subtask completeness; incorporate spatial relationship reasoning; and explore extending the framework to sequential task execution.

## Related Work & Insights

- **vs CLIO**: CLIO also performs task-driven 3D scene graph generation, but it can only handle simple instructions explicitly mentioning objects and does not support high-level abstract tasks. ASHiTA extends the capability of CLIO to high-level tasks by incorporating LLM-based task decomposition.
- **vs SayCan/LLM+P**: These methods use LLMs for task planning without coupling them with the 3D representation of the environment—the plans generated by LLMs may include objects not present in the environment or impractical steps. ASHiTA grounds the task decomposition through feedback from the scene graph.
- **vs HOV-SG + GPT**: Directly performing grounding with GPT on the HOV-SG scene graph yields an s-prec of only 4.87%, indicating that pure LLMs cannot effectively process complex 3D scene information without an information compression framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to unify hierarchical task analysis and 3D scene graph construction into an alternating optimization framework, with H-IB being a novel tool with theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes quantitative evaluations on the SG3D benchmark, comprehensive ablation studies, and real-world robot demonstrations. However, the relatively low absolute metrics indicate that the problem itself is highly challenging.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations and detailed methodology description. The supplementary material includes the full derivation of H-IB and tutorial examples.
- **Value**: ⭐⭐⭐⭐ Opens up a new direction for high-level task understanding in embodied AI, and H-IB is reusable for other multi-level information compression scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)
- [\[CVPR 2025\] Olympus: A Universal Task Router for Computer Vision Tasks](olympus_a_universal_task_router_for_computer_vision_tasks.md)
- [\[CVPR 2025\] VGGT: Visual Geometry Grounded Transformer](vggt_visual_geometry_grounded_transformer.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis](dynamic_neural_surfaces_for_elastic_4d_shape_representation_and_analysis.md)

</div>

<!-- RELATED:END -->
