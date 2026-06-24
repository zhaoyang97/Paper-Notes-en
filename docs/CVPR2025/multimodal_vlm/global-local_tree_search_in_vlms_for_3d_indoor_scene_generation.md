---
title: >-
  [Paper Note] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation
description: >-
  [CVPR 2025][Multimodal VLM][3D Scene Generation] Proposes a global-local tree search algorithm that leverages the spatial reasoning capabilities of VLMs. Through hierarchical scene representations and visual prompts from an emoji grid, it achieves high-quality 3D indoor scene layout generation, ranking first on average in user studies.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "3D Scene Generation"
  - "Vision-Language Models"
  - "Tree Search"
  - "Indoor Scenes"
  - "Layout Planning"
date: 2026-05-08
content_hash: 48f395a63afee1c5
---

# Global-Local Tree Search in VLMs for 3D Indoor Scene Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.18476](https://arxiv.org/abs/2503.18476)  
**Code**: [https://github.com/dw-dengwei/TreeSearchGen](https://github.com/dw-dengwei/TreeSearchGen)  
**Area**: Multimodal VLM  
**Keywords**: 3D Scene Generation, Vision-Language Models, Tree Search, Indoor Scenes, Layout Planning

## TL;DR
Proposes a global-local tree search algorithm that leverages the spatial reasoning capabilities of VLMs. Through hierarchical scene representations and visual prompts from an emoji grid, it achieves high-quality 3D indoor scene layout generation, ranking first on average in user studies.

## Background & Motivation

1. **Background**: The core challenge of 3D indoor scene generation is modeling appropriate spatial relationships between objects. Early data-driven methods (e.g., based on GANs/VAEs) suffer from insufficient robustness due to the small scale of 3D datasets (3D-FRONT contains only 18K+ scenes). Recent methods have shifted towards leveraging the commonsense reasoning capabilities of VLMs to generate scene layouts.
2. **Limitations of Prior Work**: Direct layout generation methods using VLMs, such as LayoutGPT, adopt left-to-right autoregressive reasoning and cannot correct previous erroneous outputs. Although HoloDeck and AnyHome generate scene graphs and then convert them into layouts, their rule-based conversion algorithms are flawed, leading to low diversity and less realistic results.
3. **Key Challenge**: The chain-like reasoning of VLMs is inherently a greedy strategy. Once an object is misplaced, errors accumulate continuously without backtracking correction, resulting in issues such as objects extending beyond rooms or overlapping with each other.
4. **Goal**: (1) How to equip VLMs with reasoning capabilities for backtracking and correction? (2) How to reduce the search tree depth in complex scenes? (3) How to enable VLMs to accurately perceive spatial locations in 2D top-down views?
5. **Key Insight**: When arranging a room, humans place objects one by one, with multiple candidate positions for each object. If the current placement is inappropriate, previous decisions are adjusted. This process is inherently a tree search problem.
6. **Core Idea**: Formulate 3D scene generation as a constraint satisfaction planning problem, solved via a hierarchical search framework combining "global tree search (object level) + local tree search (parameter level)" with VLM reasoning.

## Method

### Overall Architecture
The input is a textual description (e.g., "a bedroom with a queen-size bed"), and the output is a 3D indoor scene (class, size, position, and orientation for each object). The method consists of two stages: (1) First, a VLM generates a hierarchical scene representation (Proxy P) from the text, decomposing the scene into four levels: room $\rightarrow$ region $\rightarrow$ floor object $\rightarrow$ supported object. (2) Next, a global-local tree search algorithm is employed to determine the specific position and orientation of each object. During the search, the VLM perceives spatial locations through visual prompts from an emoji grid.

### Key Designs

1. **Hierarchical Scene Representation**:
    - **Function**: Hierarchically transforms user text input into a structured scene description, serving as a semantic bridge between the text and the 3D scene.
    - **Mechanism**: Progressively prompts the VLM to generate four levels—the room level (generating room size), the region level (dividing functional regions like sleeping/working areas, with regions sharing the room width), the floor object level (primary objects within each region, selecting an anchor object and determining the spatial relationships of other objects to the anchor, e.g., `place_front`/`place_beside`), and the supported object level (items on tables, etc.). 3D models of objects are retrieved from the Objaverse database using CLIP visual similarity + Sentence-BERT textual similarity + size matching.
    - **Design Motivation**: The semantic leap directly from text to scene is too large; hierarchical decomposition partitions the complex problem into multiple independent sub-problems of small regions, significantly reducing search tree depth and computational costs.

2. **Global Tree Search**:
    - **Function**: Manages the placement order at the object level and the backtracking logic, ensuring that all objects can be appropriately placed.
    - **Mechanism**: Uses regions as root nodes, with each level representing an object. The anchor object is placed first, and the remaining objects are placed sequentially, sorted by size from largest to smallest. For each object, local tree search is invoked to generate candidate placement plans. If successful, the search proceeds to the next level; if failed (unable to place after $k$ trials), it backtracks to the previous level to reselect a placement plan for that level. The search employs the DFS algorithm. The maximum number of trials is $k=3$ for anchor objects and $k=1$ for other objects.
    - **Design Motivation**: Simulates the human behavior of arranging a room—placing items one by one and adjusting previous decisions when finding them inappropriate. DFS prioritizes exploring the most promising solutions, which is more efficient than BFS.

3. **Local Tree Search with Emoji Grid**:
    - **Function**: Determines the specific location parameters of a single object (decomposed into three steps: side selection $\rightarrow$ row coordinate $\rightarrow$ column coordinate).
    - **Mechanism**: Decomposes object placement into three steps—first determining which side of the anchor to place it on (left/right/above/below), then determining the row or column coordinate, and finally determining the coordinate of the other axis. Each step prompts the VLM with dual text and visual inputs, where the visual prompt discretizes the top-down view into a dense grid, with each cell filled with a different emoji symbol. The VLM indicates where the object should be placed by identifying the emoji name. An auxiliary VLM evaluator checks the reasonability of intermediate results (e.g., collision detection) at each step.
    - **Design Motivation**: VLMs struggle to output precise coordinate values directly but are adept at identifying spatial relationships within a visual grid. The diversity of emojis provides unique identifiers for each cell, preventing the VLM from confusing adjacent positions. Parameter-level decomposition reduces the complexity of decision-making at each step.

### Loss & Training
This method is a training-free, test-time reasoning approach, thus involving no loss functions or training strategies. Using GPT-4o API as the backbone VLM, scene generation is completed at inference time through carefully designed prompts and search algorithms. Constraints include spatial boundaries, placement commonsense, non-overlapping, and non-floating constraints.

## Key Experimental Results

### Main Results
120 text prompts were generated using ChatGPT (4 scene types $\times$ 30 prompts), and evaluated using CLIP score and the Reciprocal Rank (RR) from a user study.

| Method | Bathroom CLIP | Bedroom CLIP | Kitchen CLIP | Living Room CLIP | Average RR |
|------|---------|---------|---------|---------|--------|
| AnyHome | 26.28 | 27.22 | 27.75 | 27.02 | 0.443 |
| HoloDeck | 28.46 | 29.33 | 29.82 | 28.87 | 0.596 |
| **Ours** | **29.37** | **29.93** | **29.58** | **30.18** | **0.793** |

In the user study (15 annotators), the proposed method achieved an average Reciprocal Rank of 0.793, which translates to an average rank of approximately 1.26 (best among the 3 methods), yielding gains of +0.360 and +0.197 compared to AnyHome and HoloDeck, respectively.

### Ablation Study

| Configuration | Bathroom RR | Bedroom RR | Kitchen RR | Living Room RR | Average RR |
|------|--------|--------|--------|--------|--------|
| IO (Direct Output) | 0.437 | 0.350 | 0.441 | 0.356 | 0.396 |
| CoT (Chain-of-Thought) | 0.681 | 0.702 | 0.676 | 0.685 | 0.686 |
| **Full (Tree Search)** | **0.714** | **0.780** | **0.714** | **0.790** | **0.750** |

### Key Findings
- The IO method performs the worst (RR of only 0.396), demonstrating that VLMs lack 3D layout samples in their training data, leading to poor direct layout output.
- CoT benefits from hierarchical scene representation and task decomposition, achieving decent performance (0.686), but lacks backtracking capabilities.
- Tree search shows limited improvement in CLIP score compared to CoT (+0.21), because a small $k$ was set to control API costs, leaving the search space insufficiently explored.
- The method performs best in scenes with distinct spatial relationships like bedrooms (0.834 RR) and living rooms (0.868 RR); it is slightly inferior to HoloDeck in kitchens, where objects are typically placed along walls with less prominent spatial relationships.

## Highlights & Insights
- **Emoji Grid as a Visual Spatial Interface**: This is a highly ingenious design—leveraging the diversity of emojis and the VLM's capability to recognize them to discretize a continuous space into a grid that can be precisely operated on by the VLM. This trick can be transferred to any task requiring the VLM to output spatial positions.
- **Hierarchical Decomposition Strategy**: Splitting the search tree into an object level (global) and a parameter level (local), while reducing subproblem complexity at the object level through region partitioning, is a classic approach to search space management.
- **Training-Free Paradigm**: Completely eliminating the need for data collection or model training, it maximizes the commonsense reasoning capability of the VLM through algorithmic design and prompt engineering, which is highly beneficial for resource-constrained scenarios.

## Limitations & Future Work
- Constrained by API calling costs, the tree search width $k$ is set to a small value, which fails to fully explore the search space and results in a narrow performance gap relative to CoT in some scenarios.
- Currently, it only handles indoor scenes and has not been extended to outdoor scenes or AR/VR applications.
- The resolution of the emoji grid is limited, which may affect the precision of object placement.
- Object retrieval depends on the Objaverse database; the method cannot handle cases where matching objects are missing from the database.
- Precise geometric shapes of objects are not considered (only bounding boxes are used), which may still lead to interpenetration between objects with complex geometries.

## Related Work & Insights
- **vs LayoutGPT**: LayoutGPT directly outputs layout parameters in CSS format using LLMs, belonging to chain-like methods that fail to correct errors. In contrast, the proposed tree search has backtracking capabilities but requires more API calls.
- **vs HoloDeck**: HoloDeck first generates a scene graph using LLMs and then converts it into a layout via rule-based systems. This work allows the VLM to directly reason about positions in the visual space, which is more flexible but heavily dependent on the quality of the VLM's visual understanding.
- **vs Tree-of-Thoughts (ToT)**: This work serves as a concrete instantiation of ToT on the 3D scene generation task, with the primary contributions lying in the design of the hierarchical representation and emoji prompting tailored for this specific problem.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of tree search and emoji grid is highly creative, though the core framework is a direct application of ToT.
- Experimental Thoroughness: ⭐⭐⭐ Quantitative evaluation is limited to CLIP scores and user studies, lacking large-scale automated metrics.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear, motivations are well-elaborated, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Demonstrates the potential of VLMs in 3D scene planning; the emoji grid approach possesses great transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[ICML 2026\] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](../../ICML2026/multimodal_vlm/pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)
- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](../../CVPR2026/multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)

</div>

<!-- RELATED:END -->
