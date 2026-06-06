---
title: >-
  [Paper Note] ObjEmbed: Towards Universal Multimodal Object Embeddings
description: >-
  [ICML 2026][Social Computing][Universal Object Embeddings] ObjEmbed trains a **universal object embedding model**—aligning multimodal object representations by combining tasks such as detection, segmentation, retrieval…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "Universal Object Embeddings"
  - "Multimodal Learning"
  - "Object Retrieval"
  - "Cross-task Representation"
date: 2026-05-08
content_hash: 213fc2fe1b2d9fa2
---

# ObjEmbed: Towards Universal Multimodal Object Embeddings

**Conference**: ICML 2026  
**arXiv**: [2605.29118](https://arxiv.org/abs/2605.29118)  
**Code**: To be confirmed  
**Area**: Multimodal / Vision-Language / Object Representation Learning  
**Keywords**: Universal Object Embeddings, Multimodal Learning, Object Retrieval, Cross-task Representation

## TL;DR
ObjEmbed trains a **universal object embedding model**—aligning multimodal object representations by combining tasks such as detection, segmentation, retrieval, captioning, and classification. A single embedding outperforms or matches task-specific SOTA on 11 tasks, including OVD / OVS / Text2Image-Object / Open-Caption-Eval.

## Background & Motivation

**Background**: Multimodal understanding of visual objects is a core task in computer vision, but existing methods are largely task-specific—CLIP aligns image-text with weak object-level granularity, OWL-ViT focuses on strong object detection but lacks generative capabilities, and SAM offers strong segmentation with weak semantics.

**Limitations of Prior Work**: (1) High deployment costs due to task-specific models; (2) Fragmentation of representations between tasks leads to failure in cross-task transfer; (3) Lack of a unified benchmark for evaluating object-level representations; (4) Data scarcity—high-quality object-level data for single tasks is difficult to scale.

**Key Challenge**: Real-world applications require a **single embedding** to support multiple tasks like detection, segmentation, retrieval, and captioning, but current methods are either task-specific or lack sufficient granularity.

**Goal**: Construct a universal object embedding model where a single representation supports high performance across multiple tasks.

**Key Insight**: It is observed that objects serve as the "common carrier" for multimodal tasks—localization for detection/segmentation, matching for retrieval, and semantics for labeling/captioning. Learning **object-level universal embeddings** can support all these tasks simultaneously.

**Core Idea**: Learn universal object embeddings through **multi-task joint training + object-level alignment**; train a single backbone using large-scale heterogeneous data (COCO/LVIS/RefCOCO/CC3M) + task-specific heads.

## Method

### Overall Architecture
(1) **Dual-stream encoding**: Image encoder (ViT-L) + Text encoder (BERT-Large); (2) **Object detection head**: Outputs object boxes based on DETR; (3) **Object embedding head**: Outputs $\mathbf{e}_{\text{obj}} \in \mathbb{R}^{512}$ for each object; (4) **Multi-task loss**: Joint optimization of detection, segmentation, retrieval, captioning, and classification; (5) **Object-level alignment**: Aligns image object embeddings with text object embeddings via contrastive learning.

### Key Designs

1. **Object-level Alignment + Multimodal Contrastive Learning**:
    - **Function**: Aligns image object embeddings with corresponding text embeddings.
    - **Mechanism**: Each image produces $N$ object embeddings $\{\mathbf{e}_{\text{obj}}^i\}$ via the detection head. Corresponding text descriptions $\mathbf{t}^i$ for each object are obtained from datasets like RefCOCO. The contrastive loss is defined as $\mathcal{L}_{\text{align}} = -\log \frac{\exp(\mathbf{e}_{\text{obj}}^i \cdot \mathbf{e}_{\text{text}}^i / \tau)}{\sum_j \exp(\mathbf{e}_{\text{obj}}^i \cdot \mathbf{e}_{\text{text}}^j / \tau)}$, utilizing in-batch and cross-image negative samples.
    - **Design Motivation**: Image-level alignment (e.g., CLIP) is too coarse; object-level alignment allows embeddings to capture fine-grained semantics. Contrastive learning provides large-scale unlabeled training signals.

2. **Multi-task Joint Training + Task-specific Heads**:
    - **Function**: Enables the backbone to learn generalized object representations through multi-task training.
    - **Mechanism**: Combined training includes detection loss $\mathcal{L}_{\text{det}}$ (DETR set matching), segmentation loss $\mathcal{L}_{\text{seg}}$ (mask prediction), retrieval loss $\mathcal{L}_{\text{ret}}$ (contrastive alignment), captioning loss $\mathcal{L}_{\text{cap}}$ (autoregressive generation), and classification loss $\mathcal{L}_{\text{cls}}$ (cross-entropy). The total loss $\mathcal{L} = \sum \lambda_i \mathcal{L}_i$ is balanced using GradNorm for adaptive gradient weighting.
    - **Design Motivation**: Single-task training results in specialized representations; joint multi-task training forces the backbone to learn universal features; GradNorm balances gradient scales across different tasks.

3. **Large-scale Heterogeneous Data Training + Data Mixing Strategy**:
    - **Function**: Trains the universal model using large-scale heterogeneous data.
    - **Mechanism**: Mixed training data includes COCO (detection/segmentation), LVIS (long-tail detection), RefCOCO (referring expression), CC3M (image-text alignment), and ImageNet (classification). Batches are sampled according to task proportions, and Online Hard Example Mining (OHEM) is used to improve performance on tail categories.
    - **Design Motivation**: Single datasets have task biases (e.g., COCO only has 80 classes); heterogeneous data provides diverse object types and contexts.

## Key Experimental Results

### Main Results: Cross-task Performance (vs Task-specific SOTA)

| Task | Dataset | Prev. SOTA | **Ours** | Gain |
|------|--------|----------|----------|------|
| Open-voc Detection | LVIS | OWL-ViT (33.7 AP) | **35.2 AP** | +1.5 |
| Open-voc Segmentation | LVIS | OpenSeeD (26.5 mIoU) | **27.8 mIoU** | +1.3 |
| Object Retrieval | COCO-Search | Detic (52.3 R@5) | **58.7 R@5** | +6.4 |
| Text-to-Image Object | Open-T2I | T2I-Object (47.8 mAP) | **49.5 mAP** | +1.7 |
| Object Captioning | RefCOCOg | OFA-Cap (118.4 CIDEr) | **122.3 CIDEr** | +3.9 |
| Object Classification | LVIS-V | Eva-CLIP (74.2 Acc) | **76.5 Acc** | +2.3 |

### Multi-task Joint Effects

| Training Strategy | LVIS AP | RefCOCO R@1 | Cap CIDEr | Task Average |
|---------|---------|-----------|-----------|----------|
| Detection Only | 32.8 | 51.2 | 95.3 | 59.8 |
| Retrieval Only | 28.5 | 78.4 | 92.7 | 66.5 |
| Detection + Retrieval | 33.5 | 76.8 | 108.4 | 72.9 |
| Full Multi-task | **35.2** | **82.5** | **122.3** | **80.0** |

### Data Scaling

| Training Data Scale | LVIS AP | RefCOCO R@1 | OOD Cap CIDEr |
|------------|---------|-----------|---------------|
| 100K images | 28.7 | 72.3 | 92.1 |
| 500K images | 32.5 | 78.6 | 108.7 |
| 1M images | 34.1 | 81.3 | 117.4 |
| **2M images** | **35.2** | **82.5** | **122.3** |

### Object Embedding Quality Evaluation

| Evaluation Metric | CLIP Object Embed | OWL-ViT | **Ours** |
|---------|-------------|---------|----------|
| Object t-SNE Purity | 0.67 | 0.74 | **0.88** |
| Cross-dataset Transfer Acc | 58.3 | 64.7 | **78.9** |
| Zero-shot Object Cls | 71.5 | 73.2 | **79.4** |

### Key Findings
- **Multi-task joint training significantly outperforms single-task training**: The joint approach yields an average improvement of 20 points over single-task baselines.
- **Data scale is not yet saturated**: Performance continues to improve steadily at the 2M image scale.
- **Object embedding quality is substantially enhanced**: t-SNE purity reaches 0.88 compared to 0.67 for CLIP.

## Highlights & Insights
- **Success of unified object representation learning**: Overcomes the limitations of task specialization, proving that universal object embeddings can support multiple tasks simultaneously.
- **Synergy in multi-task joint training**: Multi-task constraints force the model to learn more universal and robust features.
- **Object-level alignment + Heterogeneous data**: The combination of fine-grained alignment and data diversity is the key driver of success.

## Limitations & Future Work
- Model Scale: The deployment cost of a ViT-L model with multiple heads remains high.
- Task Coverage: Currently covers 5 tasks; future work should extend to 3D objects, video objects, and compositional understanding.
- Long-tail Objects: A performance gap still exists for extremely rare categories.
- Proposed Improvements: Distilling the model into smaller versions for deployment; extending to 3D/video domains; incorporating active learning to tackle the long-tail problem.

## Related Work & Insights
- **vs CLIP**: Shifts from image-level alignment to object-level alignment in ObjEmbed.
- **vs OWL-ViT**: Evolves from single-task open-vocabulary detection to multi-task joint optimization.
- **vs SAM**: Compares strong segmentation (SAM) with weak semantics against ObjEmbed's strong segmentation and strong semantics.
- **vs Florence / Florence-2**: Moves from a universal backbone with task specialization to a single embedding for all tasks.
- **Insight**: Object-level representation is the "atomic unit" of multimodal vision, and unified embeddings represent the future direction of the field.

## Rating
- Novelty: ⭐⭐⭐⭐ Universal object embedding concepts have been explored, but this work contributes a systematic training framework and multi-task optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11 tasks, multiple baselines, scaling analysis, and embedding quality metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic methodology, and strong conclusions.
- Value: ⭐⭐⭐⭐⭐ Unified object embeddings reduce deployment costs and have a significant impact on open-vocabulary visual understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](../../ACL2026/social_computing/the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/social_computing/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](../../ICLR2026/social_computing/functional_embeddings_enable_aggregation_of_multi-area_seeg_data_for_robust_bci.md)
- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](../../CVPR2026/social_computing/probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)

</div>

<!-- RELATED:END -->
