---
title: >-
  [Paper Note] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval
description: >-
  [ICCV 2025][3D Vision][Open-set 3D retrieval] This paper proposes the DAC framework, which employs a "Describe–Adapt–Combine" three-step strategy to synergize CLIP with a multimodal large language model (MLLM). Using onl…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Open-set 3D retrieval"
  - "CLIP"
  - "MLLM"
  - "LoRA"
  - "multi-view"
date: 2026-05-08
content_hash: b2d77fce8fcf88a9
---

# Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval

**Conference**: ICCV 2025
**arXiv**: [2507.21489](https://arxiv.org/abs/2507.21489)
**Code**: [GitHub](https://github.com/wangzhichuan123/DAC)
**Area**: 3D Vision
**Keywords**: Open-set 3D retrieval, CLIP, MLLM, LoRA, multi-view

## TL;DR

This paper proposes the DAC framework, which employs a "Describe–Adapt–Combine" three-step strategy to synergize CLIP with a multimodal large language model (MLLM). Using only multi-view images, DAC substantially outperforms the previous SOTA method that relies on all modalities (point clouds + voxels + images) on open-set 3D object retrieval, achieving an average mAP improvement of over +10%.

## Background & Motivation

Open-set 3D Object Retrieval (Open-set 3DOR) requires retrieving 3D objects belonging to **categories unseen during training**, and faces two major challenges:

**Out-of-distribution generalization difficulty**: Existing methods assume shared categories/domains between training and testing, leading to severe performance degradation in open-set scenarios.

**Overfitting due to scarce 3D data**: Limited training data causes models to overfit to known categories and fail to generalize to unseen ones.

The previous SOTA method HGM2R requires all modalities (point clouds, voxels, multi-view images) and incorporates test data into training, making it complex and impractical.

**Core Motivation**: CLIP, pretrained on large-scale image–text contrastive pairs, inherently possesses generalizable feature representations. The authors observe that:
- A simple "multi-view CLIP" baseline (directly aggregating per-view features from the CLIP visual encoder) already achieves competitive performance.
- However, when humans encounter unknown objects, they reason not only visually but also through linguistic descriptions (e.g., "like a horse but with black and white stripes").
- MLLMs are therefore introduced to provide complementary textual cues that compensate for the limitations of purely visual representations.

## Method

### Overall Architecture

DAC = Describe + Adapt + Combine, following a three-step pipeline:

1. **Describe**: Generate textual descriptions using an MLLM (InternVL).
2. **Adapt**: Fine-tune CLIP with AB-LoRA to adapt to multi-view projected images.
3. **Combine**: Fuse visual and textual features for retrieval.

### Key Designs

1. **Describe — Dual Use of MLLM**:

    - **During training**: The MLLM generates rich descriptions for each known category (prompt: "Describe in one sentence what [cls] should look like"), replacing the simple "a photo of [cls]" template to better align with CLIP's contrastive learning objective.
    - **During inference**: Multi-view images are fed into the MLLM to generate appearance and semantic descriptions (prompt: "There are images of an object from different angles. Describe this object in one sentence."), providing out-of-box textual knowledge for objects of unknown categories.
    - Any off-the-shelf MLLM can be used; InternVL-4B is adopted as the primary model.

2. **Adapt — Additive-Bias LoRA (AB-LoRA)**:

    - CLIP is pretrained on natural images, creating a domain gap with multi-view projected images that necessitates fine-tuning.
    - Standard LoRA's weight update $\Delta\mathbf{W}\mathbf{z}$ directly accumulates input $\mathbf{z}$ from known categories, making it prone to overfitting.
    - **Core Innovation**: A learnable bias term $\mathbf{\Phi}$ is added to LoRA:
    $\mathbf{o} = \mathbf{W}_o\mathbf{z} + \gamma\mathbf{BA}\mathbf{z} + \mathbf{\Phi}$
    - The bias term decouples the weight update from the input, acting as a regularizer to mitigate overfitting.
    - $\mathbf{A}$ is initialized with standard normal distribution; $\mathbf{B}$ and $\mathbf{\Phi}$ are initialized to zero to ensure no perturbation of the original weights at the start.
    - Applied to self-attention ($W_q, W_k, W_v$) in both CLIP visual and text encoders.

3. **Training Objective**:

    - Category descriptions are encoded by the CLIP text encoder to produce classification weights $\mathbf{c}_i = \mathcal{T}(t_i)$.
    - Multi-view images are encoded by the CLIP visual encoder and mean-pooled to obtain global features $\mathbf{g}_k = \frac{1}{M}\sum\mathbf{f}_{k,m}$.
    - Cross-entropy loss:
    $\mathcal{L}_{CE} = -\frac{1}{N_t}\sum_{k=1}^{N_t}\log\frac{\exp(\mathbf{g}_k \cdot \mathbf{c}_y / \tau)}{\sum_{i=1}^{L}\exp(\mathbf{g}_k \cdot \mathbf{c}_i / \tau)}$

4. **Combine — Feature Fusion**:

    - The adapted visual global feature $\mathbf{g}$ and textual feature $\mathbf{f}_t$ are fused via weighted addition:
    $\mathbf{h} = \tanh(\mathbf{g} + \alpha\mathbf{f}_t)$
    - Cosine similarity is used for retrieval.
    - Additive fusion outperforms concatenation (+7.1% mAP).

### Loss & Training

- Training loss: cross-entropy contrastive loss, aligning multi-view visual features with category description text features.
- SGD optimizer, lr=2e-4, batch size=4, cosine scheduler, 30 epochs.
- LoRA rank=8, dropout=0.25.
- Training on 2× NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

Performance on four open-set 3DOR benchmarks (Open-set Setup, mAP%):

| Method | Modality | OS-ESB-core | OS-NTU-core | OS-MN40-core | OS-ABO-core |
|--------|----------|-------------|-------------|--------------|-------------|
| HGM2R | P+I+V | 51.74 | 44.88 | 64.20 | 63.39 |
| DAC (B/32) | **I only** | **58.70** | **59.21** | 62.40 | **66.10** |
| DAC (L/14) | **I only** | **57.80** | **65.83** | **68.98** | **70.74** |

- Using only multi-view images, DAC (L/14) surpasses HGM2R by approximately **+10%** average mAP across four datasets.
- DAC does not incorporate test data during training, requires neither point clouds nor voxels, and is simpler and more practical.

### Ablation Study

Effect of AB-LoRA (OS-MN40-core, ViT-B/32):

| Configuration | mAP↑ | NDCG↑ | ANMRR↓ |
|---------------|------|-------|--------|
| No LoRA | 55.39 | 68.08 | 45.96 |
| Standard LoRA | 59.85 | 70.25 | 41.75 |
| AB-LoRA | **62.40** | **72.63** | **39.82** |

Contribution of each module (ViT-B/32, format: mAP/NDCG/ANMRR):

| InternVL | AB-LoRA | OS-ESB-core | OS-MN40-core |
|----------|---------|-------------|--------------|
| ✗ | ✗ | 53.93/23.00/49.70 | 49.60/65.71/50.88 |
| ✓ | ✗ | 56.16/23.58/48.39 | 55.39/68.08/45.96 |
| ✗ | ✓ | 57.45/23.96/47.13 | 59.35/71.89/42.72 |
| ✓ | ✓ | **58.70/24.27/45.67** | **62.40/72.63/39.82** |

### Key Findings

- **Surprising effect of bias**: Adding a single learnable bias term to LoRA yields +2.55% mAP improvement, effectively alleviating overfitting.
- Using InternVL embeddings directly for retrieval achieves only 38.20% mAP, far below CLIP's 53.93%, demonstrating that CLIP's discriminative features are better suited for retrieval.
- Additive fusion outperforms concatenation: addition allows the two modalities to complement each other directly within the same feature space.

## Highlights & Insights

- **Minimal yet effective**: Relies solely on multi-view images and an off-the-shelf MLLM, requiring neither point clouds/voxels nor test-time training data.
- **Creative dual use of MLLM**: Enhances textual supervision during training and provides external semantic cues during inference — an elegant design.
- **General value of AB-LoRA**: Introducing a bias into LoRA to mitigate small-data overfitting is a transferable idea applicable to other domains.

## Limitations & Future Work

- InternVL's descriptive capability is limited for high-attribute mechanical parts (OS-ESB-core), leaving substantial room for improvement.
- Grayscale projected images discard color information, which may negatively affect retrieval for color-sensitive categories.
- Stronger multi-view aggregation strategies (e.g., attention pooling) remain unexplored.

## Related Work & Insights

- **MV-CLIP**: A simple multi-view CLIP baseline, but requires category information for view selection, making it unsuitable for open-set settings.
- **ULIP-2/OpenShape**: Align language–image–3D embeddings, but are limited by the scale of available 3D data.
- **CoOp/CLIP-Adapter**: Lightweight adaptation strategies; DAC's AB-LoRA serves as a superior alternative.

## Rating

- Novelty: ⭐⭐⭐⭐ (MLLM+CLIP synergy + AB-LoRA)
- Technical Depth: ⭐⭐⭐ (method is concise and accessible)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets, 12 comparison methods, comprehensive ablations)
- Value: ⭐⭐⭐⭐ (image-only input, simple deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[ICLR 2026\] CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D](../../ICLR2026/3d_vision/core-3d_context-aware_open-vocabulary_retrieval_by_embeddings_in_3d.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](../../AAAI2026/3d_vision/open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)

</div>

<!-- RELATED:END -->
