---
title: >-
  [Paper Note] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding
description: >-
  [ECCV 2024][3D Vision][3D vision-language] This paper proposes the first million-scale 3D vision-language dataset, SceneVerse (68K indoor scenes + 2.5M scene-language pairs), and introduces GPS, a multi-level contrastive pre-training framework, achieving SOTA results in 3D visual grounding and QA tasks, as well as zero-shot transfer capabilities.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D vision-language"
  - "data scaling"
  - "grounded scene understanding"
  - "contrastive learning"
  - "pre-training"
date: 2026-05-08
content_hash: 6b067cb7d58e90c3
---

# SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding

**Conference**: ECCV 2024  
**arXiv**: [2401.09340](https://arxiv.org/abs/2401.09340)  
**Code**: [https://scene-verse.github.io](https://scene-verse.github.io)  
**Area**: 3D Vision / Vision-Language  
**Keywords**: 3D vision-language, data scaling, grounded scene understanding, contrastive learning, pre-training

## TL;DR
This paper proposes the first million-scale 3D vision-language dataset, SceneVerse (68K indoor scenes + 2.5M scene-language pairs), and introduces GPS, a multi-level contrastive pre-training framework, achieving SOTA results in 3D visual grounding and QA tasks, as well as zero-shot transfer capabilities.

## Background & Motivation
**Background**: The 2D vision-language field has achieved massive success driven by large-scale data (such as CLIP's billions of image-text pairs), but 3D scene vision-language alignment remains in its infancy.

**Limitations of Prior Work**: 3D data collection heavily relies on scanning hardware, incurring extremely high costs. Existing 3D-VL datasets contain only a few thousand scenes, falling far behind 2D datasets. Additionally, complex object configurations, rich attributes, and diverse relationships in 3D scenes demand an immense amount of descriptive language annotations.

**Key Challenge**: Insufficient data scale $\rightarrow$ inability to support effective pre-training alignment $\rightarrow$ existing models heavily rely on task-specific designs (complex loss functions or architectures), leading to poor generalization.

**Goal**: How to systematically scale up 3D vision-language data and design a unified pre-training framework to leverage this massive data.

**Key Insight**: Unifying multiple existing 3D scene datasets + automatically generating large-scale linguistic descriptions using 3D scene graphs and LLMs + multi-level contrastive learning.

**Core Idea**: Data scale is the core bottleneck of 3D-VL. Extending data to the million scale through scene graphs + LLMs, combined with multi-level contrastive pre-training, is sufficient to achieve SOTA performance.

## Method

### Overall Architecture
SceneVerse consists of two primary components: (1) Dataset Construction—unifying 68K 3D scenes from 7 sources and collecting 2.5M scene-language pairs via human annotation and an automatic generation pipeline; (2) GPS (Grounded Pre-training for Scenes)—a Transformer-based pre-training model that learns alignment between 3D scenes and text through a three-level contrastive framework: object-level, scene-level, and referral-object-level.

### Key Designs

#### 1. **Scene Curation & Annotation**
    - **Function**: Consolidates raw data from real-world datasets (ScanNet, ARKitScenes, HM3D, 3RScan, MultiScan) and synthetic environments (Structured3D, ProcTHOR).
    - **Mechanism**: Each scene undergoes room segmentation, point cloud downsampling, axis alignment, and normalization. Each scan is represented as $\mathrm{P} \in \mathbb{R}^{N \times 8}$ (3D coordinates + RGB + instance ID + semantic label). A total of 68,406 scenes are collected.
    - To obtain gold-standard annotations, 96,863 referring expressions are annotated via Amazon Mechanical Turk (AMT) with two-person validation, yielding a low re-annotation rate of only 4.8%.
    - **Design Motivation**: Fully utilize existing data sources to avoid redundant collection.

#### 2. **3D Scene Graph + LLM Generation**
    - **Function**: Automatically generates linguistic descriptions at three levels of granularity—object captions, object referrals, and scene captions.
    - **Mechanism**:
      - Constructs a hierarchical scene graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where each node $v$ is parameterized by a centroid $\boldsymbol{p}_i \in \mathbb{R}^3$ and a bounding box size $\boldsymbol{b}_i \in \mathbb{R}^3$, and edges $\mathcal{E}$ represent spatial relationships (vertical/horizontal proximity, multi-object relations).
      - Object Caption: Locates objects in multi-view images via point cloud rendering $\rightarrow$ generates initial descriptions using BLIP-2 $\rightarrow$ filters top-10 via CLIP $\rightarrow$ refines and summarizes using LLMs.
      - Object Referral: Extracts spatial relation triplets $(v_i, v_j, e_{ij})$ from the scene graph $\rightarrow$ generates templates (target-object, spatial-relation, anchor-objects) $\rightarrow$ utilizes LLMs to rephrase for naturalness.
      - Scene Caption: Randomly samples a subset of the scene graph + object counts + room types $\rightarrow$ prompts LLMs to generate global scene descriptions.
    - **Design Motivation**: Template-based generation ensures spatial coverage, while LLM rephrasing introduces variety and naturalness. Human evaluation shows a 96.93% acceptance rate (exceeding ReferIt3D's 86.1%).

#### 3. **GPS: Grounded Pre-training for Scenes**
    - **Function**: Simultaneously aligns 3D scenes and text at three distinct levels of granularity.
    - **Object-Level Alignment $\mathcal{L}_{\text{obj}}$**:
      - A point cloud encoder extracts object features $\boldsymbol{f}^O_i$, and a frozen language model encodes object captions to obtain $\boldsymbol{f}^T_i$.
      - Multi-class bidirectional contrastive loss: $\mathcal{L}_{\text{obj}} = -\frac{1}{2}\sum_{(p,q)} \left(\log\frac{\exp(D^{\text{obj}}(p,q))}{\sum_r \exp(D^{\text{obj}}(p,r))} + \log\frac{\exp(D^{\text{obj}}(p,q))}{\sum_r \exp(D^{\text{obj}}(r,q))}\right)$
      - Where $D^{\text{obj}}(p,q) = \boldsymbol{f}^O_p \boldsymbol{f}^T_q / \tau$, and $\tau$ is a learnable temperature parameter.
    - **Scene-Level Alignment $\mathcal{L}_{\text{scene}}$**:
      - A spatial Transformer encodes object and locational features to generate $\boldsymbol{f}^S_i = \text{SpatialAttn}(\{\boldsymbol{f}^O_i\}, \{\boldsymbol{l}_i\})$.
      - Projection followed by max-pooling yields the overall scene feature $\boldsymbol{g}^S$, which is optimized via inter-scene contrastive loss against the scene description feature $\boldsymbol{g}^T$.
    - **Referral-Object Alignment $\mathcal{L}_{\text{ref}}$**:
      - A self-attention reasoning Transformer takes both scene-object features and referral text as inputs.
      - **Intra-scene Contrastive Loss**: $\mathcal{L}_{\text{ref}} = -\log\frac{\exp(\bar{\boldsymbol{h}}^S \boldsymbol{h}^T / \tau)}{\sum_p \exp(\boldsymbol{h}^S_p \boldsymbol{h}^T / \tau)}$, where the positive pair is constructed within the scene, and $p$ iterates over all objects inside the same scene.
      - **Design Motivation**: Draws inspiration from the success of intra-image and inter-image contrastive learning in 2D-VL.
    - Additionally uses a Masked Language Modeling (MLM) loss $\mathcal{L}_{\text{MLM}}$ to fine-tune the language encoder.
    - Total loss: $\mathcal{L} = \mathcal{L}_{\text{obj}} + \mathcal{L}_{\text{scene}} + \mathcal{L}_{\text{ref}} + \mathcal{L}_{\text{MLM}}$

### Loss & Training
- Two-stage training scheme: First, train the point cloud encoder with object-level alignment to obtain high-quality initial features. Second, jointly train scene-level and referral-level alignment objectives.
- Avoids the use of complex auxiliary losses or task-specific network architectures.

## Key Experimental Results

### Main Results: 3D Visual Grounding

| Method | Nr3D Overall | Sr3D Overall | ScanRefer Acc@0.5 |
|------|-------------|-------------|-------------------|
| 3DVG-Trans | 40.8 | 51.4 | 34.7 |
| BUTD-DETR | 54.6 | 67.0 | 39.8 |
| ViL3DRel | 64.4 | 72.8 | 37.7 |
| 3D-VisTA (pre-train) | 64.2 | 76.4 | 45.8 |
| **GPS (scratch)** | 58.7 | 68.4 | 40.4 |
| **GPS (pre-train)** | 55.2 | 74.1 | **47.1** |
| **GPS (fine-tuned)** | **64.9** | **77.5** | **48.1** |

After pre-training, GPS directly surpasses prior methods on ScanRefer (47.1) without fine-tuning, and performance increases further upon fine-tuning.

### Zero-shot Transfer

| Method | Nr3D | Sr3D | ScanRefer@0.5 |
|------|------|------|---------------|
| 3D-VisTA (zero-shot) | 35.2 | 31.2 | 29.6 |
| 3D-VisTA (zero-shot text) | 43.1 | 36.1 | 36.4 |
| **GPS (zero-shot)** | 32.4 | 33.3 | **31.1** |
| **GPS (zero-shot text)** | 41.9 | **38.1** | 35.8 |

Zero-shot transfer on SceneVerse-val: GPS achieves 59.2% (vs 3D-VisTA's 52.9%), demonstrating that SceneVerse data significantly boosts model generalization.

### 3D QA

| Model | ScanQA val | SQA3D |
|------|-----------|-------|
| ScanQA | 20.3 | 46.6 |
| 3D-VisTA | 22.4 | 48.5 |
| **GPS** | **22.7** | **49.9** |

### Key Findings
- When trained from scratch, GPS does not perform as well as models designed with complex task-specific modules. However, once data scaling is applied, simple contrastive alignment alone achieves substantial improvements $\rightarrow$ proving that data scale is the core bottleneck.
- The scaling effect of SceneVerse is not restricted to GPS; it also significantly enhances other models, such as RegionPLC, on semantic segmentation tasks.
- Automatically generated text descriptions exhibit high quality (96.93% acceptance rate) and achieve diversity comparable to human annotations after LLM rephrasing.

## Highlights & Insights
- **Data Scaling is King**: For the first time in the 3D-VL domain, the fundamental law established in 2D—where "data scale determines performance"—is verified. Unifying 68K scenes and 2.5M language pairs increases the data volume by an order of magnitude compared to the previously largest dataset (~3K scenes).
- **Elegant Scene Graph + LLM Pipeline**: Template generation guarantees full spatial coverage while LLM rephrasing injects naturalness. These complementary components produce high-quality automatic annotations.
- **Simple Yet Effective Multi-level Contrastive Learning**: Triple-level alignment (object-level inter-scene contrast, scene-level inter-scene contrast, and referral-level intra-scene contrast) functions effectively without requiring task-specific auxiliary losses, offering an elegant and robust formulation.

## Limitations & Future Work
- Restricted to indoor scenes, currently lacking support for outdoor 3D environments.
- Domain gaps still persist between synthetic scenes (Structured3D, ProcTHOR) and real-world scans.
- Textual descriptions primarily focus on static spatial relationships without modeling dynamic or interactive elements.
- The quality of the generated language remains highly dependent on LLM performance; some long-tail scenarios might have insufficient coverage.

## Related Work & Insights
- **vs 3D-VisTA**: Both are pre-training methodologies, but while 3D-VisTA relies on classification objectives (softmax), GPS adopts contrastive alignment, which enables superior generalization in zero-shot tasks.
- **vs ScanScribe**: ScanScribe only contains 278K language pairs, whereas SceneVerse reaches 2.5M, expanding the scale by nearly $10\times$.
- **vs CLIP (2D)**: Effectively inherits CLIP's contrastive learning paradigm and temperature parameter design, successfully transferring these powerful concepts into the 3D domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First million-scale 3D-VL dataset + scene graph-LLM generation pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Thorough ablation studies covering multiple tasks including grounding, QA, zero-shot transfer, and semantic segmentation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear logic, seamlessly transitioning from dataset construction to model methodology and downstream experiments.
- **Value**: ⭐⭐⭐⭐⭐ Lays the data foundation for the 3D-VL field, comparable to the contributions of ImageNet/CLIP to their respective areas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation](open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] nuCraft: Crafting High Resolution 3D Semantic Occupancy for Unified 3D Scene Understanding](nucraft_crafting_high_resolution_3d_semantic_occupancy_for_unified_3d_scene_unde.md)
- [\[ICML 2026\] SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](../../ICML2026/3d_vision/svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)
- [\[ICLR 2026\] Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation](../../ICLR2026/3d_vision/scenethesis_a_language_and_vision_agentic_framework_for_3d_scene_generation.md)

</div>

<!-- RELATED:END -->
