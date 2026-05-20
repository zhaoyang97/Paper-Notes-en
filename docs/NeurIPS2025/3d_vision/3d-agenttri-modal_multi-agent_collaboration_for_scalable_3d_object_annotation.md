---
title: >-
  [Paper Note] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation
description: >-
  [NeurIPS 2025][3D Vision][3D object annotation] This paper proposes Tri-MARF, a tri-modal multi-agent framework comprising a VLM annotation agent (multi-view, multi-candidate description generation)…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D object annotation"
  - "multi-agent collaboration"
  - "VLM"
  - "Multi-Armed Bandit"
  - "point cloud verification"
  - "cross-modal alignment"
date: 2026-05-08
content_hash: 7af76d16b6ac9b31
---

# 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation

**Conference**: NeurIPS 2025
**arXiv**: [2601.04404](https://arxiv.org/abs/2601.04404)  
**Code**: To be released  
**Area**: 3D Vision / Data Annotation / Multi-Agent Systems
**Keywords**: 3D object annotation, multi-agent collaboration, VLM, Multi-Armed Bandit, point cloud verification, cross-modal alignment

## TL;DR

This paper proposes Tri-MARF, a tri-modal multi-agent framework comprising a VLM annotation agent (multi-view, multi-candidate description generation), an information aggregation agent (BERT clustering + CLIP weighting + UCB1 Multi-Armed Bandit selection), and a point cloud gating agent (Uni3D text–point cloud alignment for hallucination filtering). The system achieves a CLIPScore of 88.7 (surpassing human annotation at 82.4), a throughput of 12k objects/hour, and has annotated approximately 2 million 3D models.

## Background & Motivation

**Background**: 3D object annotation is a foundational task for applications such as autonomous driving, robotics, and AR. Early works like ShapeNet and PartNet established human-annotation paradigms; more recently, ULIP and PointCLIP introduced CLIP into the 3D domain, and Cap3D pioneered synthetic-to-real annotation transfer. However, these approaches generally rely on a single VLM generating descriptions from a limited number of viewpoints.

**Limitations of Prior Work**:
- Single-view/single-model methods fail to capture the complete information of 3D objects—critical features may be distributed across different viewpoints (e.g., brand identifiers on the front of a vehicle, tail-light arrays at the rear, and silhouette lines on the side).
- VLMs suffer from severe hallucination, generating attribute descriptions for non-existent features.
- Multi-view descriptions exhibit substantial redundancy and semantic inconsistency.

**Key Challenge**: A single model cannot simultaneously optimize **accuracy, completeness, consistency, and efficiency**—analogous to a single expert being unable to master all domains. A "team collaboration" system design is required.

**Goal**: Design a multi-agent collaborative system that decomposes 3D annotation into three specialized subtasks—visual description generation, information aggregation and selection, and geometric consistency verification—with each agent dedicated to its respective role.

**Key Insight**: Drawing on multi-agent systems and reinforcement learning, the paper employs a Multi-Armed Bandit (MAB) algorithm for adaptive selection among multiple candidate descriptions, and uses a point cloud encoder to provide a 3D verification signal independent of 2D imagery.

**Core Idea**: Three specialized agents collaborate through division of labor (VLM generation + MAB selection + point cloud gating), eliminating hallucinations through exploration–exploitation balance and cross-modal verification to achieve annotation quality surpassing human annotators.

## Method

### Overall Architecture

A four-stage pipeline:
1. **Data Preparation**: Render 6 standard viewpoint images (front/back/left/right/top/bottom) per 3D object and sample point cloud features.
2. **VLM Annotation (Agent 1)**: Qwen2.5-VL-72B generates $M=5$ candidate descriptions per viewpoint.
3. **Information Aggregation (Agent 2)**: BERT + DBSCAN deduplication → CLIP weighting → UCB1 MAB selection of the best description → cross-view fusion.
4. **Point Cloud Gating (Agent 3)**: Uni3D encoder computes text–point cloud cosine similarity; samples below the threshold are flagged as suspicious.

### Agent 1: VLM Annotation Agent

A multi-turn dialogue strategy with Qwen2.5-VL-72B-Instruct (rather than a conventional single prompt) guides the model in three stages:

- **Viewpoint-Aware Recognition**: The model is informed of the current viewpoint (e.g., "this is the front view") to direct attention to viewpoint-specific diagnostic cues.
- **Systematic Attribute Elicitation**: Follow-up prompts sequentially elicit key attributes such as color, material, and structure.
- **Contextual Integration**: Observations are synthesized into a coherent description while maintaining viewpoint alignment.

Per viewpoint, $M=5$ candidate descriptions are sampled at temperature $T=0.7$ to introduce stochastic diversity. Each description retains token-level log-probabilities for confidence estimation:

$$\text{Conf}(C) = \frac{1}{N}\sum_{i=1}^{N}|\log P(t_i \mid \text{context up to } t_i)|$$

A low Conf value indicates high confidence (uniformly high token probabilities); a high Conf value signals model uncertainty (potential confabulation). This confidence score serves two purposes: (1) flagging potentially hallucinated content for rejection; and (2) assisting MAB selection among semantically similar candidates.

### Agent 2: Information Aggregation Agent

**Step 1: BERT + DBSCAN Semantic Deduplication**

The five candidate descriptions per viewpoint are mapped into BERT semantic space, and pairwise cosine similarities between embeddings are computed:

$$S_{ij} = \frac{E_{v,i} \cdot E_{v,j}}{\|E_{v,i}\| \|E_{v,j}\|}$$

DBSCAN automatically determines the number of clusters; one representative description is selected per cluster to eliminate semantic redundancy.

**Step 2: CLIP Visual–Textual Alignment Weighting**

CLIP evaluates each description's alignment with the corresponding viewpoint image, with softmax normalization yielding probability weights:

$$w_{v,i} = \frac{\exp(\cos\theta_{v,i})}{\sum_{k=1}^{M}\exp(\cos\theta_{v,k})}$$

VLM confidence $S_{\text{conf}}$ and CLIP weight $w_i$ are combined into a final score: $s_i = (1-\alpha) \cdot S_{\text{conf},i} + \alpha \cdot w_i$, where $\alpha$ balances textual confidence against visual–semantic alignment.

**Step 3: UCB1 Multi-Armed Bandit Adaptive Selection**

Each deduplicated candidate description is treated as an arm, with arm set $\mathcal{A} = \{a_1, \ldots, a_K\}$. The selection rule is:

$$a_t = \arg\max_{a \in \mathcal{A}} \left(\hat{r}_a + c\sqrt{\frac{2\ln t}{n_a}}\right)$$

where $\hat{r}_a$ is the empirical mean reward, $n_a$ is the selection count, and $c$ is the exploration weight. The reward function integrates VLM confidence and CLIP similarity.

The key advantage of UCB1 is its "optimism in the face of uncertainty": arms selected fewer times receive a larger exploration bonus, preventing premature convergence. Compared with static rules or simple voting, MAB adapts to different object types and viewpoint configurations.

**Step 4: Cross-View Fusion for Global Description**

- Front/back viewpoints are prioritized (higher weight $w_{FB}$); a core sentence is extracted as $S_{\text{core}} = \text{First\_Sentence}(C_{FB})$.
- Side/top/bottom viewpoints supply supplementary detail, forming $C_{\text{other}}$.
- Global description: $C_{\text{global}} = S_{\text{core}} + C_{\text{other}}$.
- The global score is the average of front/back and other viewpoint scores.

### Agent 3: Point Cloud Gating Agent

A pretrained Uni3D encoder projects both text and point clouds into a shared $\mathbb{R}^d$ space, and cosine similarity is computed for cross-modal matching.

A threshold of $\alpha = 0.577$ is determined via grid search on a validation set:
- Above threshold: the annotation is retained.
- Below threshold: the sample is flagged as suspicious; critical-category samples are sent for human review, while redundant samples are directly filtered.

**Design Motivation**: Pure 2D image-based methods cannot verify geometric properties (e.g., object shape and structure). The point cloud provides an independent 3D verification signal, effectively suppressing VLM hallucinations.

### Loss & Training

Qwen2.5-VL-72B-Instruct is used in inference mode without fine-tuning. The MAB reward function integrates VLM confidence and CLIP similarity. The overall pipeline constitutes an **inference-time collaborative framework** requiring no additional end-to-end training.

## Key Experimental Results

### Main Results: 3D Annotation Quality Comparison (Table 1)

Annotation quality and efficiency are compared across three datasets (single A100 GPU):

| Method | LVIS CLIPScore | LVIS ViLT R@5 | XL CLIPScore | ABO CLIPScore | Speed |
|--------|---------------|--------------|-------------|--------------|-------|
| Human | 82.4 | 40.0/38.5 | 81.0 | 78.9 | 0.12k/h |
| **Tri-MARF** | **88.7** | **45.2/43.8** | **86.1** | **82.3** | **12k/h** |
| ScoreAgg | 80.1 | 37.8/36.0 | 78.5 | 76.2 | 9k/h |
| Cap3D | 78.6 | 35.2/33.4 | 76.4 | 74.8 | 8k/h |
| 3D-LLM | 77.4 | 34.9/33.3 | 75.6 | 73.0 | 6.5k/h |
| ULIP-2 | 75.2 | 33.1/31.5 | 73.8 | 71.4 | 7k/h |
| PointCLIP | 65.3 | 22.4/20.8 | 63.1 | 60.7 | 5k/h |
| GPT4Point | 62.9 | 18.7/17.1 | 60.5 | 58.2 | 4k/h |

### Cross-Dataset Generalization (Table 2)

Zero-shot generalization (without fine-tuning) on ShapeNet-Core, ScanNet, and ModelNet40:

| Method | ShapeNet CLIP | ScanNet CLIP | ModelNet CLIP | ShapeNet GPT-4 |
|--------|-------------|------------|-------------|----------------|
| **Tri-MARF** | **83.2** | **80.3** | **81.5** | **4.3** |
| Human | 81.7 | 79.5 | 80.2 | 4.2 |
| ScoreAgg | 79.1 | 75.6 | 77.2 | 3.9 |
| Cap3D | 76.5 | 73.2 | 74.3 | 3.6 |
| 3D-LLM | 75.8 | 72.5 | 73.6 | 3.5 |

### Key Findings

- **CLIPScore**: Tri-MARF achieves 88.7 on Objaverse-LVIS, surpassing human annotation (82.4) by 6.3 points, demonstrating that multi-agent collaboration captures information more comprehensively than individual annotators.
- **Classification Accuracy**: GPT-4o semantic scoring reaches 98.32%, exceeding human annotation (95.72%) by 2.6 percentage points.
- **Ablation on Number of Views**: Six viewpoints constitute the optimal configuration (CLIPScore 88.7); increasing to 8 viewpoints degrades performance, as redundant information impairs consistency and efficiency.
- **Throughput**: 12k objects/hour—100× faster than humans (0.12k/h) and 1.5× faster than Cap3D (8k/h).
- **Generalization**: Cross-dataset CLIPScore drops by only 7.2% (the lowest among all methods); Cap3D drops 11.5%, and other methods drop 10–15%.

## Highlights & Insights

1. **Elegant Task Decomposition**: Decomposing 3D annotation into "generation–selection–verification" stages, each leveraging the most suitable model (VLM / BERT+CLIP / Uni3D), avoids burdening a single model with all subtasks.
2. **Novel Application of UCB1 for Multi-Candidate Selection**: Formalizing multi-candidate description selection as a MAB problem offers stronger theoretical guarantees (regret bounds) than voting or greedy strategies, while adapting to different object types.
3. **Point Cloud Gating as a Critical Hallucination Suppressor**: VLMs may generate descriptions that are plausible in 2D but geometrically inconsistent; the point cloud provides an independent geometric verification channel.
4. **Training-Free Inference Framework**: The entire system requires no end-to-end training; all components are composed from pretrained models, facilitating deployment and scaling.
5. **Surpassing Human Annotation**: This is the first work to systematically exceed human performance on 3D annotation tasks (CLIPScore +6.3, classification semantic accuracy +2.6%), demonstrating that multi-agent collaboration can identify details easily overlooked by individual annotators.

## Limitations & Future Work

1. **High Computational Cost**: Qwen2.5-VL-72B inference × 6 views × 5 candidates = 30 VLM calls per object; although 12k/h is faster than humans, absolute compute consumption remains substantial.
2. **Threshold Sensitivity**: The gating threshold $\alpha=0.577$ is determined by grid search and may require recalibration under different data distributions.
3. **Fixed Six-View Configuration**: Standard six viewpoints work well for regular objects but may still exhibit occlusion blind spots for irregular or non-convex objects; adaptive viewpoint selection is a potential improvement direction.
4. **Limited Number of MAB Arms**: After DBSCAN deduplication, the number of arms may be small (2–3), limiting the exploration–exploitation advantage of MAB.
5. **Lack of End-to-End Optimization**: The three agents are arranged serially; errors from Agent 1 propagate downstream. A lightweight feedback or iterative refinement mechanism could improve overall performance.
6. **Limited Inter-Agent Communication**: The current architecture passes information unidirectionally between agents; future work could explore dialogue-based negotiation mechanisms among agents.

## Related Work & Insights

- **Cap3D**: A pioneering work in 3D annotation using a single VLM and single viewpoint; the direct predecessor this paper extends.
- **ULIP / PointCLIP**: Representative works introducing CLIP into the 3D domain; leveraged in Agent 2 for visual–textual alignment.
- **Uni3D**: A unified 3D representation learning model whose encoder is employed for point cloud–text cross-modal matching.
- **MAB/UCB Theory (Auer et al., 2002)**: The classical multi-armed bandit algorithm, creatively applied here to textual candidate selection.
- **Inspiration**: The "multi-agent division of labor + RL coordination" framework design pattern is transferable to other tasks requiring multimodal generation and verification, such as 3D scene description, medical image report generation, and multimodal QA.

## Rating

4/5

- **Novelty** 4/5: The tri-agent collaborative framework and MAB-based selector are novel; point cloud gating for hallucination suppression is a valuable contribution.
- **Experimental Thoroughness** 5/5: Three primary datasets and three cross-domain datasets; comprehensive metrics (CLIPScore / ViLT / GPT-4o / AB Test); surpasses human baselines.
- **Writing Quality** 3/5: Method descriptions are detailed but overly lengthy; some formulations are redundant.
- **Value** 5/5: With 2 million models already annotated and a throughput of 12k/h, the system is directly applicable to large-scale 3D dataset construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)
- [\[NeurIPS 2025\] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent](sceneweaver_all-in-one_3d_scene_synthesis_with_an_extensible_and_self-reflective.md)
- [\[NeurIPS 2025\] EA3D: Online Open-World 3D Object Extraction from Streaming Videos](ea3d_online_open-world_3d_object_extraction_from_streaming_videos.md)
- [\[NeurIPS 2025\] VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)

</div>

<!-- RELATED:END -->
