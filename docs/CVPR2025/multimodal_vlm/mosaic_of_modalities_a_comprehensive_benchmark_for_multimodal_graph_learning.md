---
title: >-
  [Paper Note] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning
description: >-
  [CVPR 2025][Multimodal VLM][Multimodal Graph Learning] This paper proposes MM-Graph, the first comprehensive graph learning benchmark that incorporates both textual and visual node attributes. Covering 7 real-world datasets of varying scales and 3 categories of graph tasks (link prediction, node classification, and knowledge graph completion), it systematically evaluates the impact of visual information on graph learning, revealing key findings such as "multimodal GNNs underp…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multimodal Graph Learning"
  - "Benchmark"
  - "Graph Neural Networks"
  - "Knowledge Graph Completion"
  - "Feature Alignment"
date: 2026-05-08
content_hash: 0acce138ab6bfc77
---

# Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning

**Conference**: CVPR 2025  
**arXiv**: [2406.16321](https://arxiv.org/abs/2406.16321)  
**Code**: [https://mm-graph-benchmark.github.io/](https://mm-graph-benchmark.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Graph Learning, Benchmark, Graph Neural Networks, Knowledge Graph Completion, Feature Alignment

## TL;DR

This paper proposes MM-Graph, the first comprehensive graph learning benchmark that incorporates both textual and visual node attributes. Covering 7 real-world datasets of varying scales and 3 categories of graph tasks (link prediction, node classification, and knowledge graph completion), it systematically evaluates the impact of visual information on graph learning, revealing key findings such as "multimodal GNNs underperforming traditional GNNs" and "the crucial importance of feature alignment."

## Background & Motivation

Graph machine learning has established numerous benchmarks (e.g., OGB, GL-Bench), and text-attributed graph (TAG) benchmarks have also developed rapidly in recent years (e.g., CS-TAG). $\rightarrow$ However, existing benchmarks almost entirely ignore visual information, whereas real-world graph entities often possess rich multimodal semantics (e.g., product images in e-commerce, book covers). $\rightarrow$ A semantic gap exists between vision and text: products with similar appearances may have completely different textual descriptions, which cannot be captured by text-only GNNs. $\rightarrow$ The field of Multimodal Knowledge Graphs (MMKGs) suffers from poor data quality (broken URLs, missing images). $\rightarrow$ **Core Idea**: To construct MM-Graph, the first standardized text+vision multimodal graph benchmark, and systematically explore how visual information affects graph learning; and to provide a unified GNN/KGE/feature encoder/evaluator framework to enable direct comparison of different methods.

## Method

### Overall Architecture

MM-Graph comprises 7 datasets (3 for link prediction, 2 for node classification, and 2 for knowledge graph completion). It unifies GNN architectures (GCN/SAGE/MMGCN/MGAT/BUDDY/MLP), KGE methods (MoSE/VISTA), combinations of 4 visual encoders $\times$ 3 text encoders, as well as standardized data loaders and evaluators.

### Key Designs

1. **Construction of 7 Multimodal Graph Datasets**:
    - Function: Provides a diverse testing platform ranging from small-scale (1.4K entities) to large-scale (685K nodes / 7.2M edges).
    - Mechanism: Includes Amazon-Sports/Cloth (e-commerce co-purchase graphs with product titles + product images), Goodreads-LP/NC (book recommendation graphs with book descriptions + cover images), Ele-Fashion (fashion classification graph), and MM-CoDEx-s/m (knowledge graphs with Wikipedia descriptions + entity images).
    - Design Motivation: Varying scales ensure computational complexity diversity; different domains ensure generalization evaluation; using Beautiful Soup to scrape high-resolution images ensures data quality; using HeaRT to generate hard negatives in link prediction increases evaluation challenge.

2. **Standardized Evaluation Framework**:
    - Function: Compares different GNN architectures and feature encoding strategies under unified standards.
    - Mechanism: Traditional GNNs (GCN/SAGE) first concatenate text and vision embeddings before GNN processing; multimodal GNNs (MMGCN constructs independent graphs for each modality, MGAT uses cross-modal attention); Optuna is used for hyperparameter optimization to ensure fairness.
    - Design Motivation: Existing studies encounter difficulties in comparing methods under unified conditions; multimodal GNNs (MMGCN/MGAT) originate from recommendation systems and need to be adapted to HTML standard graph learning tasks to evaluate their effectiveness.

3. **Exploration of Multimodal Feature Encoding Strategies**:
    - Function: Systematically explores the impact of different text-vision encoding combinations on graph learning for the first time.
    - Mechanism: Aligned encoders (CLIP: joint text-vision contrastive learning; ImageBind: cross-modal unified embedding) vs. non-aligned encoders (ViT+T5, DINOv2+T5: trained independently without cross-modal alignment goals).
    - Design Motivation: Validates whether "multimodal feature alignment" is indeed crucial for graph learning, providing a baseline selection guide for future research.

### Loss & Training

- Link Prediction: Uses a dot-product decoder, with HeaRT generating hard negatives where each positive edge is ranked against 150 negative edges.
- Node Classification: A 3-layer MLP maps node representations to the number of classes.
- Knowledge Graph Completion: Uses the original training/validation/test sets and negative samples from CoDEx.
- Splitting Ratio: 8/1/1 for the Amazon series, 6/1/3 for Goodreads and Ele-fashion, and the original split for CoDEx.

## Key Experimental Results

### Main Results (MRR)

| Method | Encoder | Amazon-Sports | Amazon-Cloth | Goodreads-LP |
|------|--------|-------------|-------------|-------------|
| SAGE | CLIP | 33.83 | 24.58 | 44.10 |
| SAGE | ImageBind | **34.32** | **25.20** | 34.61 |
| SAGE | DINOv2+T5 | 32.20 | 22.98 | **45.61** |
| MMGCN | CLIP | 31.96 | 22.20 | 31.84 |
| MGAT | CLIP | 27.56 | 21.38 | 74.75 |
| MLP | CLIP | 28.22 | 21.10 | 11.03 |

### Node Classification (Accuracy)

| Method | Encoder | Ele-fashion | Goodreads-NC |
|------|--------|-------------|-------------|
| MMGCN | ImageBind | **86.21** | 80.58 |
| SAGE | DINOv2+T5 | 85.53 | **84.01** |
| GCN | CLIP | 79.83 | 81.61 |

### Ablation Study (Aligned vs. Non-Aligned Features)

| Alignment | Amazon-Sports(MRR) | Amazon-Cloth(MRR) | Description |
|---------|-------------------|-------------------|------|
| CLIP (Aligned) | 33.83 | 24.58 | Joint text-vision contrastive training |
| ImageBind (Aligned) | **34.32** | **25.20** | Cross-modal unified embedding |
| ViT+T5 (Non-aligned) | 32.01 | 23.11 | Independent training, no alignment |
| DINOv2+T5 (Non-aligned) | 32.20 | 22.98 | Self-supervised + independent text |

### Key Findings

- **Traditional GNNs (SAGE) outperform multimodal GNNs (MMGCN/MGAT) on most datasets**—a counter-intuitive finding indicating that existing multimodal GNN designs are not yet mature, and that independent message passing for each modality followed by late fusion yields poor performance.
- **Aligned encoders (CLIP/ImageBind) consistently outperform non-aligned encoders**—cross-modal pre-training alignment serves as the foundation for multimodal graph learning.
- **ImageBind performs best in tasks requiring cross-modal reasoning**, as its unified embedding space provides an interface to introduce other modalities like audio/video in the future.
- **The contribution of visual information varies greatly across datasets**: Amazon product images hold high value, while Goodreads book covers offer relatively limited value.
- **The MLP baseline is competitive in certain scenarios**: when feature quality is high, the marginal utility of the graph structure is limited.

## Highlights & Insights

- **Fills a critical community gap**: The first standardized text+vision multimodal graph learning benchmark, serving as foundational infrastructure to advance this field.
- **Inspiring counter-intuitive findings**: The discovery that multimodal GNNs underperform traditional GNNs prompts the community to rethink how multimodal information is fused within graphs.
- **Rigorous experimental design**: Exhaustive combinations of 4 visual $\times$ 3 text encoders $\times$ 6 GNN/KGE methods, optimized using Optuna.
- **High-quality data**: Constructed from reliable sources such as Amazon, Goodreads, and CoDEx, with high-resolution images scraped.

## Limitations & Future Work

- Only supports node-level multimodal features, without considering edge-level multimodal information.
- Visual features are frozen after extraction via pre-trained encoders, without exploring end-to-end joint training.
- The fusion strategies of multimodal GNNs are relatively simple (concatenation/attention), with more advanced fusion methods (e.g., cross-modal Transformers) left unexplored.
- Limited domain coverage (e-commerce, books, knowledge graphs), lacking social or citation networks.

## Related Work & Insights

- **OGB**: A standard benchmark for graph learning that lacks visual features; MM-Graph fills this gap.
- **CS-TAG**: Focuses on text-attributed graphs; MM-Graph extends this to vision + text.
- **CoDEx**: A high-quality knowledge graph benchmark, on which MM-Graph incorporates visual features to construct MM-CoDEx.
- **Insight**: Existing multimodal GNNs exhibit suboptimal performance, with the core bottleneck being "how to effectively fuse multimodal information during graph message passing"—simple late fusion is insufficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The first multimodal graph learning benchmark, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely exhaustive all-combination experiments, covering 7 datasets $\times$ multiple methods $\times$ multiple encoders.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-summarized findings.
- Value: ⭐⭐⭐⭐ The benchmark contribution and key findings hold long-term value for driving the community forward, though it lacks methodological innovation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](../../CVPR2026/multimodal_vlm/graphvlm_benchmark_vlm_graph_learning.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICLR 2026\] MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](../../ICLR2026/multimodal_vlm/mme-unify_a_comprehensive_benchmark_for_unified_multimodal_understanding_and_gen.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](../../ICML2026/multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)

</div>

<!-- RELATED:END -->
