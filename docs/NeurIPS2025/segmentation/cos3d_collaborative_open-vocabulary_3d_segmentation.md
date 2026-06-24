---
title: >-
  [Paper Note] COS3D: Collaborative Open-Vocabulary 3D Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Open-vocabulary 3D segmentation] This paper proposes COS3D — a collaborative prompt-segmentation framework that constructs a collaborative field comprising an instance field and a language field. During training, the language field is built via instance-to-language feature mapping; during inference, language-to-instance adaptive prompt refinement generates precise segmentation results. COS3D substantially outperforms existing methods on two mainst…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Open-vocabulary 3D segmentation"
  - "3D Gaussian"
  - "collaborative segmentation"
  - "instance-language mapping"
  - "prompt segmentation"
date: 2026-05-08
content_hash: 1f93c715c14f1d2b
---

# COS3D: Collaborative Open-Vocabulary 3D Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2510.20238](https://arxiv.org/abs/2510.20238)  
**Code**: [GitHub](https://github.com/Runsong123/COS3D)  
**Area**: 3D Segmentation
**Keywords**: Open-vocabulary 3D segmentation, 3D Gaussian, collaborative segmentation, instance-language mapping, prompt segmentation

## TL;DR

This paper proposes COS3D — a collaborative prompt-segmentation framework that constructs a collaborative field comprising an instance field and a language field. During training, the language field is built via instance-to-language feature mapping; during inference, language-to-instance adaptive prompt refinement generates precise segmentation results. COS3D substantially outperforms existing methods on two mainstream benchmarks.

## Background & Motivation

**Challenges in open-vocabulary 3D segmentation (OV3DS)**: OV3DS requires simultaneous understanding of both segmentation and language information. Existing methods based on 3D Gaussian Splatting fall into two categories, each with distinct limitations:

1. **Language-based methods** (LangSplat, LEGaussians, Dr.Splat): Distill CLIP features from 2D image space into a 3D language field. Per-pixel language distillation leads to insufficient feature discriminability, producing severe boundary artifacts and errors in segmentation results.

2. **Segmentation-based methods** (OpenGaussian, InstanceGaussian): First perform class-agnostic 3D segmentation, then use a VLM to select the best-matching 3D segment. Limitations include: (a) over-segmentation or under-segmentation in the absence of semantic cues; (b) manually designed post-matching strategies introduce additional errors and cause error accumulation.

**Core insight**: The two types of information are complementary — segmentation information is discriminative and boundary-aware, while language information facilitates high-level understanding of objects and scenes. Achieving OV3DS requires collaborative understanding of both.

## Method

### Overall Architecture

COS3D is built upon 3D Gaussian Splatting and comprises three technical components:

1. **Collaborative field**: Composed of an instance field $\Theta_I$ and a language field $\Theta_L$
2. **Two-stage training strategy**: The instance field is learned first, followed by construction of the language field via Ins2Lang mapping
3. **Adaptive inference refinement**: The language field generates a 3D relevance map used as a prompt to refine segmentation in the instance field

### Key Designs

**1. Definition of the collaborative field**

Two features are appended to each 3D Gaussian $g_i = (p_i, s_i, q_i, o_i, c_i)$:
- **Instance feature** $I \in \mathbb{R}^{d_I}$ ($d_I=16$): carries segmentation-aware information
- **Language feature** $L \in \mathbb{R}^{d_L}$ ($d_L=512$, CLIP dimension): carries semantic information

The two fields interact continuously during both training and inference — in training, the instance field assists in constructing the language field; in inference, the language field guides instance-field segmentation.

**2. Instance-to-language (Ins2Lang) mapping**

A mapping function $\Phi: L = \Phi(I)$ is learned from instance features to language features. Two implementations are provided:

- **Shallow MLP**: Learns the mapping function $\Phi_{\text{network}}$, with loss $\mathcal{L}_{\text{mapping}} = |L^m - \Phi_{\text{network}}(I^m)|$; training requires fewer than 3 minutes.
- **Kernel regression**: Uses the Nadaraya-Watson estimator with $\sigma=0.1$; requires no training.

Since instance features are already discriminative, the mapping task is essentially a simple regression problem, and the kernel regression approach achieves superior performance.

**3. Adaptive language-to-instance (Lang2Ins) prompt refinement**

The inference-time segmentation pipeline proceeds as follows:
1. Given a text query $q_{\text{text}}$, encode it via CLIP to obtain $L_{\text{text}}$
2. Compute a relevance score $R$ for each 3D Gaussian, and select the high-relevance point set $\mathcal{S}$
3. **Key step**: Use $\mathcal{S}$ as a prompt and expand the neighborhood in the instance field — identify neighboring points whose cosine similarity of instance features exceeds threshold $\mathcal{T}$
4. Apply adaptive filtering to the expanded region: compute region-level relevance (opacity-weighted mean) and retain regions above threshold $\tau$
5. Process regions in descending order of relevance score and progressively aggregate to obtain the final segmentation $\mathcal{S}_t$

### Loss & Training

**Stage 1 — Instance field training**: InfoNCE contrastive loss

$$\mathcal{L}_{\text{ins}} = -\frac{1}{|\Omega|} \sum_{\Omega_j \in \Omega} \sum_{u \in \Omega_j} \log \frac{\exp(\text{sim}(I_u, \bar{I}_j))}{\sum_{\Omega_l \in \Omega} \exp(\text{sim}(I_u, \bar{I}_l))}$$

where $\Omega_j$ is the set of pixels belonging to the same instance as determined by SAM 2D segmentation, and $\bar{I}_j$ is the mean feature of that instance.

**Stage 2 — Language field construction**: Mapping is learned from instance–CLIP feature pairs. Training pairs are constructed at the SAM mask level (rather than pixel level) to reduce redundancy.

**Advantages of the two-stage strategy**:
- Compared to single-stage joint learning: avoids mapping loss interfering with the instance feature space, and reduces training time by over 60%
- Compared to parallel learning: enables fusion of information from both fields, yielding significantly better performance

## Key Experimental Results

### Main Results

**3D Gaussian segmentation on the LeRF dataset**:

| Method | Type | mIoU | mAcc |
|---|---|:---:|:---:|
| LangSplat (CVPR'24) | Language | 9.66 | 12.41 |
| LEGaussians (CVPR'24) | Language | 16.21 | 23.82 |
| Dr.Splat (CVPR'25) | Language | 43.58 | 63.87 |
| OpenGaussian (NeurIPS'24) | Segmentation | 38.36 | 51.43 |
| InstanceGaussian (CVPR'25) | Segmentation | 45.30 | 58.44 |
| **COS3D (shallow MLPs)** | **Collaborative prompt** | **49.75** | **70.60** |
| **COS3D (kernel regression)** | **Collaborative prompt** | **50.76** | **72.08** |

**ScanNetv2 dataset** (19 classes):

| Method | mIoU | mAcc |
|---|:---:|:---:|
| LangSplat | 3.78 | 9.11 |
| LEGaussians | 3.84 | 10.87 |
| OpenGaussian | 24.73 | 41.54 |
| **COS3D (kernel regression)** | **32.47** | **49.05** |

### Ablation Study

**Training strategy comparison** (LeRF dataset):

| Learning scheme | mIoU | mAcc | Training time |
|---|:---:|:---:|:---:|
| Single-stage joint learning | 49.15 | 69.19 | 165 min |
| Parallel learning | 43.84 | 59.81 | 95 min |
| Ours (shallow MLPs) | 49.75 | 70.60 | 53 min |
| **Ours (kernel regression)** | **50.76** | **72.08** | **50 min** |

**Inference strategy comparison**:

| Inference scheme | mIoU | mAcc | Query time |
|---|:---:|:---:|:---:|
| Instance branch only | 44.07 | 59.83 | 0.12s |
| Language branch only | 48.99 | 71.31 | 0.13s |
| **Collaborative prompt (Ours)** | **50.76** | **72.08** | 0.22s |

**Training efficiency**: Using only 8 minutes (3K instance field iterations), COS3D achieves mIoU 50.16, already surpassing all baselines.

### Key Findings

1. **Collaborative strategy substantially outperforms individual strategies**: Neither the language branch nor the instance branch alone matches the collaborative approach.
2. **Kernel regression outperforms MLP**: The discriminative instance features reduce the mapping to a simple regression task that kernel regression handles more effectively.
3. **Exceptional training efficiency**: SOTA is achieved in 50 minutes; all baselines are surpassed in 8 minutes (LangSplat requires 240 minutes).
4. **Compatible with multiple 2D VLMs**: Replacing CLIP with SigLIP or SAM with SAM2 further improves performance.
5. **Rich extension applications**: Supports image-query 3D segmentation, hierarchical segmentation, and robotic grasping.

## Highlights & Insights

1. **Novel concept of the collaborative field**: The complementarity of instance and language information is formalized as a collaborative field, with collaboration realized in both the training and inference stages.
2. **Compact and efficient design**: Kernel regression mapping requires no training; the two-stage strategy is over 3× faster than joint learning.
3. **Elegant prompt refinement strategy**: The coarse relevance map from the language field serves as a prompt guiding fine-grained segmentation in the instance field — analogous to the SAM prompt paradigm.
4. **Significantly improved boundary quality**: Qualitative results demonstrate more complete object boundaries and fewer artifacts.

## Limitations & Future Work

1. **Lack of reasoning capability**: The text-aligned language field cannot handle relational queries (e.g., "the cup on the table") or multi-object queries.
2. **Offline setting**: The current framework requires a complete multi-view image collection and does not support online or incremental scenarios.
3. **Dependence on SAM quality**: The instance field's training supervision derives from SAM 2D segmentation; errors in SAM propagate into 3D.
4. **Increased query latency**: Collaborative inference query time (0.22s) is approximately 70% slower than single-branch inference (0.12–0.13s).
5. **Large-scale scene generalization**: Performance in large-scale outdoor scenes has not been validated.

## Related Work & Insights

- **LangSplat / LEGaussians / Dr.Splat**: Language feature distillation methods; COS3D's language field construction is more efficient and achieves better results.
- **OpenGaussian / InstanceGaussian**: Segmentation-based methods; COS3D avoids error accumulation through collaboration.
- **SAM / Click-Gaussian**: Prompt segmentation paradigm; COS3D converts language queries into prompts within the instance field.
- **3D Gaussian Splatting**: Explicit 3D scene representation; COS3D extends it with a collaborative field.
- Implication for 3D understanding: **Collaboration between language and segmentation outperforms either modality alone**; the timing (training vs. inference) and mechanism of multimodal fusion are central design questions.

## Rating

⭐⭐⭐⭐⭐ (5/5)

The technical design is compact and elegant — the concept of the collaborative field is clearly articulated, the two-stage training strategy is well-motivated and efficient, and the inference-time prompt refinement is a natural design extension. Experimental results substantially outperform prior work on two benchmarks (mIoU 50.76 vs. 45.30 on LeRF), training efficiency far exceeds baselines (50 min vs. 240 min), and comprehensive ablations validate the necessity of each component. The framework also demonstrates rich application extensions (image-query, hierarchical segmentation, robotics). Code is publicly available, facilitating reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
