---
title: >-
  [Paper Note] Multi-Label Cluster Discrimination for Visual Representation Learning
description: >-
  [ECCV 2024][Information Retrieval & RAG][Visual Representation Learning] This work proposes MLCD (Multi-Label Cluster Discrimination), which assigns multiple cluster pseudo-labels to each image and designs a disambiguated multi-label classification loss. Pre-trained on LAION-400M, the ViT model under MLCD comprehensively outperforms OpenCLIP, FLIP, and UNICOM in linear probe, zero-shot classification, and retrieval tasks.
tags:
  - "ECCV 2024"
  - "Information Retrieval & RAG"
  - "Visual Representation Learning"
  - "Cluster Discrimination"
  - "Multi-Label Classification"
  - "CLIP"
  - "Large-Scale Pre-training"
date: 2026-05-08
content_hash: 907498886b6665da
---

# Multi-Label Cluster Discrimination for Visual Representation Learning

**Conference**: ECCV 2024  
**arXiv**: [2407.17331](https://arxiv.org/abs/2407.17331)  
**Code**: Yes ([https://github.com/deepglint/unicom](https://github.com/deepglint/unicom) + Hugging Face)  
**Area**: Social Computing  
**Keywords**: Visual Representation Learning, Cluster Discrimination, Multi-Label Classification, CLIP, Large-Scale Pre-training

## TL;DR

This work proposes MLCD (Multi-Label Cluster Discrimination), which assigns multiple cluster pseudo-labels to each image and designs a disambiguated multi-label classification loss. Pre-trained on LAION-400M, the ViT model under MLCD comprehensively outperforms OpenCLIP, FLIP, and UNICOM in linear probe, zero-shot classification, and retrieval tasks.

## Background & Motivation

### Problem 1: Instance Discrimination Fails to Capture Semantic Structures

Language-supervised visual pre-training methods like CLIP rely on instance discrimination, treating each image-text pair as a unique instance where different instances are constantly pushed apart as negative pairs. When a large number of semantically similar instances in a mini-batch are treated as negative pairs, semantically close samples are inappropriately pushed away in the embedding space. Consequently, **instance discrimination struggles to encode the semantic structure of training data**.

### Problem 2: Single-Label Cluster Discrimination Ignores Multi-Label Signals

To address the limitations of instance discrimination, clustering-based discrimination methods (such as DeepCluster, SwAV, UNICOM, etc.) explore semantic structures through iterative clustering and classification. Grouping similar instances into the same cluster brings semantically similar samples closer. However, most clustering discrimination methods only assign a **single pseudo-label** to each image.

### Core Observation & Motivation

Natural images often contain multiple visual objects or attributes (e.g., an image containing buildings, sky, and pedestrians simultaneously). The natural language supervision of CLIP can provide multi-granularity labels for a single image (objects, scenes, actions, relationships), whereas single-label clustering fails to capture all visual signals in an image.

Therefore, the authors propose **assigning multiple cluster labels to each image** (multi-label cluster discrimination), while designing a specialized disambiguated multi-label loss to handle noise in large-scale automatic clustering.

## Method

### Overall Architecture

MLCD consists of two steps: (1) **Clustering Step**: performing offline k-means clustering on LAION-400M into 1 million categories based on pre-trained CLIP features, and assigning multiple nearest cluster centroids as positive labels for each image; (2) **Discrimination Step**: designing a disambiguated multi-label classification loss to train the image encoder.

### Key Designs

#### 1. **Multi-label Clustering**

**Function**: Assigns $l$ positive labels to each training image (instead of the traditional single label) to capture multi-granularity visual signals within the image.

**Mechanism**: Utilizing features from a pre-trained CLIP ViT-L/14, a one-step offline k-means clustering is conducted on LAION-400M ($k=1M$ classes, taking about 10 minutes). For each image, the cosine similarity between its embedding and all cluster centroids is computed, and the $l$ nearest centroids (default $l=8$ ) are selected as positive labels, while the rest serve as negatives.

**Design Motivation**: Due to the limited discriminative power of CLIP models, a single pseudo-label may not cover all visual signals in an image. By selecting multiple nearest centroids, the semantic content of an image can be described in a multi-granular manner (analogous to CLIP's text representation capability providing multi-granularity labels). It prioritizes intra-class purity (by clustering into 1M classes) and mitigates inter-class conflicts via PartialFC sampling.

#### 2. **Base Multi-Label Classification Loss (MLC Loss)**

**Function**: Formulates multi-label classification as an optimization problem that minimizes all negative-positive similarity differences $(s_j - s_i)$.

**Mechanism**: Let $\{s_i\}$ $(i=1,...,l)$ be the positive similarities and $\{s_j\}$ $(j=1,...,k-l)$ be the negative similarities. The base multi-label loss is defined as:

$$\mathcal{L}_\text{MLC} = \log\left(1 + \sum_{j \in \Omega_n} \exp(s_j) \sum_{i \in \Omega_p} \exp(-s_i)\right)$$

This is equivalent to optimizing and reducing $(s_j - s_i)$ over every pair of $(s_j, s_i)$. Meanwhile, PartialFC negative sampling is introduced, randomly sampling only $r=10\%$ of negative centroids to participate in computation.

**Design Motivation**: Inherits the pairwise comparison philosophy of Circle Loss and naturally extends it to multi-label scenarios. Under million-scale categories, PartialFC sampling saves computation while alleviating inter-class conflicts.

#### 3. **Disambiguated Multi-Label Classification Loss (MLCD Loss)**

**Function**: Resolves the decision boundary ambiguity problem caused by the optimization of $(s_j - s_i)$ in the base MLC loss.

**Mechanism**: The decision boundary of MLC optimization for $(s_j - s_i)$ is $s_j - s_i = m$, which allows for ambiguity: $\{s_j, s_i\}=\{0.1, 0.4\}$ and $\{0.5, 0.8\}$ both satisfy $m=0.3$, but in the latter case, $s_j=0.5$ is still unacceptably high. Consequently, two additional optimization targets are introduced—**maximizing $s_i$** (high similarity for positive classes) and **minimizing $s_j$** (low similarity for negative classes):

$$\mathcal{L}_\text{MLCD} = \underbrace{\log\left(1 + \sum_{i \in \Omega_p} \exp(-s_i)\right)}_{\text{Positive Class Loss}} + \underbrace{\log\left(1 + \sum_{j \in \Omega'_n} \exp(s_j)\right)}_{\text{Negative Class Loss}}$$

The crucial elegance lies in the fact that with these two additional terms, the **positive loss and negative loss are naturally decoupled** (mathematically provable), enabling them to be optimized independently without interference.

**Design Motivation**: (1) Experimental visualizations (Fig. 3) show that compared to MLC, MLCD enables faster increases and more concentrated distributions of positive cosine similarity $s_i$, while negative similarity $s_j$ approaches zero (more orthogonal); (2) The decoupled positive and negative losses lead to more stable optimization and better convergence; (3) Compared to the TLPR loss which introduces thresholds, MLCD is simpler and better suited for large-scale noisy data.

### Loss & Training

- **Pre-training**: LAION-400M dataset, ViT-L/14 backbone, 32 epochs, batch size 32K, 80×A100
- **Optimizer**: AdamW, learning rate 0.001, weight decay 0.2
- **Acceleration**: Mixed-precision training + Flash Attention + DALI data loading
- **Text Encoder**: Trained from scratch for 32 epochs using the LiT approach with the image encoder frozen (for zero-shot tasks)
- **Default Hyperparameters**: $k=1M$ clusters, $r=0.1$ negative sampling rate, $l=8$ positive labels

## Key Experimental Results

### Main Results

Linear probe performance on 26 downstream datasets (ViT-L/14 backbone):

| Method | Training Data | Avg on 26 Datasets | Representative Dataset |
|------|---------|------------|------------|
| CLIP | WIT-400M | 84.2 | IN1K: 83.9 |
| OpenCLIP | LAION-400M | 82.3 | IN1K: 82.1 |
| UNICOM | LAION-400M | 83.3 | IN1K: - |
| **MLCD (Ours)** | **LAION-400M** | **84.6**| **IN1K: 84.6** |

- Average improvement of **2.3%** over OpenCLIP (outperforming on 25/26 datasets)
- Average improvement of **1.3%** over UNICOM (outperforming on 23/26 datasets)
- Even outperforms CLIP which uses the private WIT dataset

### Main Results (Zero-shot Classification & Retrieval)

| Method | Training Data | Zero-shot Avg on 25 Datasets | MSCOCO I2T R@1 | MSCOCO T2I R@1 |
|------|---------|---------------------|----------------|----------------|
| CLIP | WIT-400M | 66.9 | 56.2 | 35.8 |
| OpenCLIP | LAION-400M | 63.6 | 58.0 | 41.3 |
| FLIP | LAION-400M | 66.0 | 60.2 | 44.2 |
| **MLCD (Ours)** | **LAION-400M** | **67.5** | **60.8** | **44.5** |

Zero-shot classification improves by **3.9%** over OpenCLIP and **1.5%** over FLIP. MSCOCO retrieval is comprehensively leading.

### Ablation Study

Ablation study on ViT-B/32 + LAION-400M (5 epochs):

| Ablation Dimension | Configuration | IN1K Linear Probe | Explanation |
|---------|------|-------------------|------|
| Number of clusters $k$ | 100K / 200K / 500K / **1M** / 2M / 5M | 66.9 / 71.1 / 74.4 / **75.2** / 74.9 / 74.7 | 1M is optimal; too many clusters aggravate inter-class conflicts |
| Negative sampling rate $r$ | 0.01 / 0.05 / **0.1** / 0.2 / 0.5 / 1.0 | 73.4 / 75.1 / **75.2** / 74.9 / 68.3 / 63.2 | 0.1 is optimal; performance plummets at 1.0 |
| Number of positive labels $l$ | 1 / 2 / 4 / **8** / 16 / 32 | 71.4 / 72.9 / 73.2 / **75.2** / 72.1 / 68.7 | 8 is optimal; too many labels introduce noise |
| MLC vs MLCD | MLC / **MLCD** (32 epochs, ViT-B/32) | FT: 80.9/81.2, LP: 76.9/78.1, ZS: 63.9/64.5 | MLCD outperforms MLC under all settings |

### ImageNet Robustness Evaluation

| Method | Data | Finetune | Linear | Zero-Shot | IN-V2 | IN-A | IN-R |
|------|------|----------|--------|-----------|-------|------|------|
| OpenCLIP | LAION | 86.2 | 82.1 | 72.8 | 64.0 | 48.3 | 84.3 |
| FLIP | LAION | - | - | 74.6 | 66.8 | 51.2 | 86.5 |
| **Ours** | **LAION** | **87.1** | **84.6** | **75.6** | **68.9** | **56.4** | **85.1** |

Outperforms OpenCLIP significantly across all robustness benchmarks (IN-V2/A/R/ObjectNet).

### Key Findings

1. **Multi-label is significantly better than single-label**: $l=8$ versus $l=1$ yields a 3.8% improvement (75.2 vs 71.4), proving the value of multi-label signals.
2. **Disambiguation loss MLCD consistently outperforms MLC**: Full-scale performance gains are observed across the three evaluation settings (FT/LP/ZS), with positive classes being more compactly distributed and negative classes being more orthogonal.
3. **Negative class oversampling is harmful**: Performance drops drastically from 75.2 to 63.2 when $r=1.0$, indicating severe inter-class conflicts in million-scale classification.
4. **Fixed label count is superior to adaptive thresholds**: Searching for a global similarity threshold is challenging. Setting a fixed $l=8$ leverages the prior knowledge that "daily images statistically contain several visual concepts."
5. **Cross-dataset consistency**: MLCD consistently outperforms UNICOM when switching from LAION-400M to COYO-700M.

## Highlights & Insights

1. **Simple yet effective multi-label strategy**: Only requires a one-step offline clustering followed by selecting Top-$l$ nearest neighbor centroids, incurring no additional clustering overhead.
2. **Mathematical elegance of MLCD loss**: By introducing two additional optimization objectives, the positive and negative losses are naturally decomposed into two independent log-sum-exp terms, achieving simplicity and high efficiency.
3. **Strong engineering practicality**: Cooperating with PartialFC and negative sampling enables highly efficient training under million-scale categories; code and models are open-sourced, offering plug-and-play value.
4. **Extensive experimental coverage**: Highly convincing evaluation across 26 linear probe datasets + 25 zero-shot datasets + retrieval + robustness benchmarks.

## Limitations & Future Work

1. **Clustering quality depends on the pre-trained model**: A one-step offline clustering is conducted using CLIP ViT-L/14; thus, clustering quality is bounded by the feature quality of this model. Iterative clustering-and-training has not been explored.
2. **Inflexible fixed number of positive labels**: Setting $l=8$ uniformly for all images fails to account for variations in visual complexity across different images.
3. **Only validated on ViT architecture**: Results for CNN backbones are not reported.
4. **Additional training required for the text encoder**: Zero-shot evaluations require training the text encoder for 32 epochs using LiT, which adds complexity to the pipeline.
5. **Lack of comparison with recent methods**: Such as SigLIP, EVA-CLIP, etc.

## Related Work & Insights

- **UNICOM (ICLR 2023)**: Former study from the same team, employing single-label cluster discrimination; MLCD introduces multi-label learning on top of this.
- **Circle Loss**: Theoretical foundation of MLCD loss, which reveals the ambiguity issue of $(s_j - s_i)$ optimization.
- **PartialFC**: Face recognition work from the same team, demonstrating that negative sampling strategies are critical in million-scale classification.
- Insight: The concept of multi-label clustering can be combined with self-distillation, MAE, etc., or extended to video/3D representation learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of multi-label cluster discrimination is natural and effective, and the MLCD disambiguation loss is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation on 51 downstream datasets + detailed ablations + robustness + cross-dataset validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, smooth methodological derivation, and systematic ablation design.
- **Value**: ⭐⭐⭐⭐⭐ — Open-sourced code and models, simple and general method, providing direct reference value for improving CLIP-style models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)
- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](../../NeurIPS2025/information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)

</div>

<!-- RELATED:END -->
