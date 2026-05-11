---
title: >-
  [Paper Note] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning
description: >-
  [AAAI 2026][3D Vision][3D scene graph] This paper proposes OSU-3DSG, a unified framework that integrates vision-language models for open-world 3D scene graph generation and supports four scene interaction tasks — scene q…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D scene graph"
  - "open-world"
  - "retrieval-augmented reasoning"
  - "vision-language model"
  - "embodied interaction"
date: 2026-05-08
content_hash: ed892faab3d2fe20
---

# Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.05894](https://arxiv.org/abs/2511.05894)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D scene graph, open-world, retrieval-augmented reasoning, vision-language model, embodied interaction

## TL;DR

This paper proposes OSU-3DSG, a unified framework that integrates vision-language models for open-world 3D scene graph generation and supports four scene interaction tasks — scene question answering, visual grounding, instance retrieval, and task planning — via retrieval-augmented reasoning, achieving performance comparable to supervised methods under a zero-shot setting.

## Background & Motivation

Understanding 3D scenes is fundamental to autonomous navigation, augmented reality, and related applications. Existing methods face several critical challenges:

**Closed-vocabulary constraints**: Traditional 3D scene graph methods (e.g., 3DSSG) rely on predefined label sets and supervised annotations, and fail to generalize to unseen objects and relations in new environments.

**Dependence on static annotations**: These methods require annotated RGB-D data and known camera poses, which is impractical in real open-world scenarios.

**2D-to-3D projection errors**: Methods that infer 3D semantics by projecting 2D VLM predictions are susceptible to occlusion and viewpoint variation.

**Mechanism**: The paper leverages the open-vocabulary capabilities of VLMs to construct 3D scene graphs without manual annotation, then encodes the scene graphs into a vector database to support retrieval-based multimodal reasoning and interaction. This eliminates the need for human annotation while enabling LLMs to perform scene-aware reasoning through retrieval augmentation.

## Method

### Overall Architecture

The framework consists of two major components:
1. **3D Scene Graph Generator**: Incrementally constructs semantic and spatial representations from RGB-D sequences.
2. **Retrieval-Augmented Reasoning Module**: Converts the scene graph into a vectorized knowledge base supporting text/image-conditioned queries.

### Key Designs

1. **Open-World 3D Scene Graph Generation**

   **Multi-frame Object Detection**:
   Objects are detected from RGB-D frame sequences, where each frame contains a color image $I$, depth map $D$, camera intrinsics $K$, and pose $T_w^c \in SE(3)$. Detected objects are represented as oriented 3D bounding boxes:
   $$b_i = (c_i, \ell_i, R_i), \quad c_i \in \mathbb{R}^3, \ell_i \in \mathbb{R}_{>0}^3, R_i \in SO(3)$$

   Detection confidence is modeled with a Beta distribution: $\sigma_i \sim \text{Beta}(\alpha_i, \beta_i)$, with an adaptive scaling factor $\tau$ dynamically adjusted based on the entropy of predicted probabilities.

   3D points are obtained via depth unprojection using instance masks: $X_j^c = K^{-1}[u_j\ v_j\ 1]D_j$, then transformed to world coordinates.

   Duplicate objects are merged every $L$ frames based on cosine similarity:
   $$S(\tilde{f}_i, \tilde{f}_j) = \frac{\langle \tilde{f}_i, \tilde{f}_j \rangle}{\|\tilde{f}_i\| \|\tilde{f}_j\|}$$
   where features are preprocessed via Mahalanobis whitening.

   **Best-View Selection and Annotation**:
   For each object, the best view maximizing visibility and projection coverage is selected:
   $$T_{w,i}^{c*} = \arg\max_{T_w^c \in \mathcal{P}} \left[A(\mathcal{P}(X_i^w, T_w^c)) \cdot V(X_i^w, T_w^c)^\gamma - \lambda D(T_{w,i}^c, T_w^c)\right]$$
   Objects are then semantically annotated using LLaVA from the selected best view.

   **Design Motivation**: Best-view selection reduces occlusion and ambiguity, enabling the VLM to produce more accurate open-vocabulary annotations.

   **Reliable Object Filtering**:
   Valid object pairs are filtered using Euclidean distance and 3D IoU ($d_{thresh} = 0.5\text{m}$) to control computational cost in subsequent relation inference.

   **Semantic Relation Extraction**:
   Qwen2-VL-72B is used to infer the top-5 semantic predicates for each valid object pair:
   $$\mathcal{R}_{ij} = (o_i, r_{ij}, o_j), \quad r_{ij} \in \mathcal{C}_{edge}$$
   Background elements (floors, ceilings) are filtered out to yield the final 3D semantic scene graph.

2. **Retrieval-Augmented Semantic Reasoning**

   **Vector Database Construction**:
   The scene graph is reorganized into object-label-centric "chunks," each aggregating instance information of a given object category. A semantic encoder (CLIP/BERT/Text2Vec) maps each chunk to a high-dimensional vector space:
   $$\boldsymbol{\zeta}_i = \phi(\boldsymbol{\eta}_i), \quad \mathcal{D} = \{(\boldsymbol{\zeta}_i, \boldsymbol{\eta}_i)\}_{i=1}^N$$

   **Grounded Prompt Reasoning**:
   Given a user query $q$, top-k retrieval is performed by similarity search after encoding:
   $$\mathcal{E}_q = \text{Top-}k(\mathcal{D}, \boldsymbol{\xi}_q)$$

   Retrieved scene information is combined with the user query into a structured prompt, which is fed to an LLM (Qwen-2-72B-Instruct) for grounded reasoning.

3. **Four Scene Interaction Tasks**

   - **Task I: Text-based Scene QA** — Answers natural language questions based on scene graph facts.
   - **Task II: Text-to-Visual Grounding** — Grounds text queries to spatial locations and best-view images.
   - **Task III: Multimodal Instance Retrieval** — Supports instance-level search with text, image, or hybrid queries.
   - **Task IV: Open-Scene Task Planning** — Decomposes high-level instructions into executable step sequences.

### Loss & Training

This method is a zero-shot inference framework with no end-to-end training, relying primarily on the reasoning capabilities of pretrained VLMs and the retrieval mechanism. Key hyperparameters include:
- Cosine similarity threshold $\tau_{merge}$ for object merging
- Object pair distance threshold $d_{thresh} = 0.5\text{m}$
- Open-vocabulary label matching: BERT embedding cosine similarity thresholds of 0.95 (objects) / 0.9 (predicates)

## Key Experimental Results

### Main Results

**3D Scene Graph Generation (3DSSG dataset)**:

| Method | Type | Object R@1 | Predicate R@1 | Predicate R@3 | Relation R@1 | Relation R@3 |
|--------|------|------------|---------------|---------------|--------------|--------------|
| 3DSSG | Closed | 0.82 | 0.83 | 0.85 | 0.63 | 0.63 |
| MonoSSG | Closed | 0.86 | 0.89 | 0.90 | 0.89 | 0.90 |
| VL-SAT | Closed | 0.82 | 0.94 | 0.94 | 0.87 | 0.88 |
| Open3DSG | Open | 0.65 | 0.81 | 0.81 | 0.70 | 0.72 |
| BBQ | Open | 0.59 | 0.61 | 0.61 | 0.68 | 0.68 |
| **OSU-3DSG (Ours)** | Open | **0.83** | **0.95** | **0.97** | **0.78** | **0.80** |

As a zero-shot method, the proposed approach surpasses all closed-vocabulary methods in predicate prediction (R@1: 0.95 vs. VL-SAT's 0.94) and substantially outperforms open-vocabulary baselines.

**Scene Interaction Tasks**:

| Task | Metric | OSU-3DSG | GPT-4o | Gemini | ChatGLM |
|------|--------|----------|--------|--------|---------|
| Scene QA | Accuracy | **0.84** | 0.82 | 0.80 | 0.72 |
| Task Planning | Correctness | **87.5%** | 72.9% | 65.4% | 58.7% |
| Task Planning | Executability | **81.25%** | 78.2% | 69.8% | 62.3% |

### Ablation Study

**Filtering Strategy in Semantic Relation Extraction (SRE)**:

| IoU | Distance | # Triplets | Predicate R@1 | Relation R@1 | Notes |
|-----|----------|------------|---------------|--------------|-------|
| ✗ | ✗ | 291 | 0.95 | 0.94 | No filtering; high computational cost |
| ✔ | ✗ | 30 | 0.76 | 0.83 | IoU-only filtering insufficient |
| ✗ | ✔ | 11 | 0.85 | 0.75 | Distance-only filtering; too few triplets |
| ✔ | ✔ | 34 | **0.87** | **0.78** | **Best balance** |

Jointly applying IoU and distance constraints reduces candidate triplets to 34, maintaining high recall while substantially reducing VLM inference cost.

### Key Findings

- Zero-shot scene graph generation can match or exceed supervised methods in predicate prediction performance.
- Retrieval-augmented reasoning outperforms GPT-4o on scene QA, demonstrating the value of structured scene knowledge.
- Best-view selection is critical for object recognition accuracy.
- The fixed distance threshold (0.5m) may not be suitable for all scene densities and object scales.

## Highlights & Insights

1. **Unified open-world 3D understanding framework**: Integrates scene graph generation and multimodal reasoning into a coherent system spanning the full pipeline from perception to planning.
2. **Zero-shot outperforms supervised**: Leveraging large-scale VLM knowledge, the method surpasses fully supervised approaches in predicate prediction, demonstrating the significant potential of VLM zero-shot capabilities.
3. **Advantages of retrieval augmentation**: Retrieving relevant scene segments before constructing prompts is more efficient and accurate than directly injecting the entire scene into LLM prompts.
4. **Best-view selection mechanism**: Automatically identifies the optimal observation angle for each object, reducing occlusion and ambiguity to improve VLM annotation quality.

## Limitations & Future Work

- The distance threshold (0.5m) for object pair filtering is fixed and its cross-scene generalizability remains to be validated.
- Relation inference relies heavily on the reasoning capability of Qwen2-VL-72B, making the method's upper bound contingent on LLM capacity.
- Absolute accuracy on the visual grounding task remains low (~0.23); joint spatial-textual reasoning in 3D remains an open challenge.
- Validation is limited to indoor scenes; scalability to large-scale outdoor environments is unknown.
- Task planning lacks closed-loop verification with real robot execution.

## Related Work & Insights

- **Open3DSG (2024)**: A pioneering work on open-vocabulary 3D scene graphs, but still relies on annotated RGB-D data and fixed camera poses.
- **BBQ (2024, Linok et al.)**: An object-centric open-world scene graph model serving as the open-vocabulary baseline in this work.
- **ConceptGraphs**: A similar VLM-driven 3D scene graph approach, but without retrieval-augmented reasoning.
- **Insight**: Framing 3D scene understanding as "knowledge base construction + retrieval-augmented reasoning" is a paradigm generalizable to other structured scene understanding tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Combining retrieval-augmented reasoning with 3D scene graphs is a relatively novel paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers four interaction tasks, though evaluation scale per task is limited)
- Writing Quality: ⭐⭐⭐⭐ (Framework description is clear, though the notation is heavy and some definitions could be simplified)
- Value: ⭐⭐⭐⭐ (An important direction for open-world 3D understanding; zero-shot performance is encouraging)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/3d_vision/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[AAAI 2026\] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds](tosc_task-oriented_shape_completion_for_open-world_dexterous_grasp_generation_fr.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICLR 2026\] CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D](../../ICLR2026/3d_vision/core-3d_context-aware_open-vocabulary_retrieval_by_embeddings_in_3d.md)

</div>

<!-- RELATED:END -->
