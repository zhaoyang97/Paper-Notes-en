---
title: >-
  [Paper Note] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond
description: >-
  [ICCV 2025][Segmentation][3D part segmentation] PartField learns a continuous 3D feature field via a feed-forward model, distilling knowledge from mixed 2D/3D part proposals through contrastive learning. It outperforms e…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "3D part segmentation"
  - "feature field"
  - "contrastive learning"
  - "feed-forward model"
  - "hierarchical decomposition"
date: 2026-05-08
content_hash: 429fcd17ab870d66
---

# PartField: Learning 3D Feature Fields for Part Segmentation and Beyond

**Conference**: ICCV 2025
**arXiv**: [2504.11451](https://arxiv.org/abs/2504.11451)  
**Code**: [https://research.nvidia.com/labs/toronto-ai/partfield-release/](https://research.nvidia.com/labs/toronto-ai/partfield-release/) (project page)  
**Area**: Semantic Segmentation
**Keywords**: 3D part segmentation, feature field, contrastive learning, feed-forward model, hierarchical decomposition

## TL;DR

PartField learns a continuous 3D feature field via a feed-forward model, distilling knowledge from mixed 2D/3D part proposals through contrastive learning. It outperforms existing methods by 20%+ on category-agnostic 3D part segmentation while achieving inference speeds orders of magnitude faster.

## Background & Motivation

Part-level 3D understanding is critical for shape editing, physical simulation, robotic manipulation, and geometry processing. However, 3D part segmentation faces two fundamental challenges:

**Challenge 1: Data scarcity.** Existing 3D part annotation datasets (e.g., PartNet, ShapeNetPart) are small in scale and limited in category coverage, making it difficult for supervised methods to generalize to unseen categories. Although some methods leverage priors from 2D foundation models (e.g., SAM) to circumvent the need for 3D annotations, most (e.g., Ultrametric Feature Fields, SAMPart3D) require **per-shape optimization**: render multi-view images → 2D segmentation → fusion/distillation into 3D. This leads to:
- Extremely slow inference (minutes to hours)
- Inconsistent predictions across views
- Sensitivity to noise in 2D model outputs

**Challenge 2: Ambiguity in part definition.** What constitutes a "part"? Is an entire hand one part, or is each finger a separate part? Prior methods either predefine part templates (e.g., PartSLIP) or rely on text prompts (e.g., Find3D). However:
- Templates/text cannot cover all possible part granularities
- Part annotation standards differ across datasets, making joint training difficult
- Purely geometric parts may lack clear linguistic descriptions

**Core Idea of PartField:** Rather than predefining part templates or text prompts, the paper learns a continuous **3D feature field** in which the notion of a part is implicitly encoded by feature distances — points within the same part have similar features, while points from different parts are pushed apart. Through a carefully designed contrastive learning objective, the model learns from 2D/3D part proposals of varying granularity and annotation standards, overcoming label inconsistency. A feed-forward model (as opposed to per-shape optimization) enables fast inference and cross-shape consistency.

## Method

### Overall Architecture

Given a 3D shape $S$ (mesh, point cloud, or 3D Gaussian Splats), PartField feed-forwardly predicts a continuous feature field $f(\mathbf{p}; S): \mathbb{R}^3 \to \mathbb{R}^n$. Training uses a contrastive loss; at inference, hierarchical part decomposition is obtained by clustering the predicted features.

Core pipeline:
1. Input point cloud → PVCNN encoder extracts per-point features
2. Orthographic projection onto three axis-aligned planes → initial triplane representation
3. 2D CNN downsampling → Transformer (6 layers) → transposed CNN upsampling → output triplane
4. For any 3D query point, features are retrieved from the triplane and summed
5. Clustering (agglomerative or k-means) yields part segmentation

### Key Designs

1. **Mixed 2D/3D Part Proposal Training**

    - *Function*: Part proposals are collected from two complementary sources as training supervision, without requiring semantic consistency across proposals.
    - *2D proposals*: RGB/normal images are rendered for ~340K Objaverse shapes; SAM2 with densely sampled point prompts ($32 \times 32$) generates 2D masks at multiple scales, which are back-projected to 3D.
    - *3D proposals*: Hierarchical part annotations from PartNet (~30K shapes, 24 categories) are used. Meshes are converted to tetrahedral grids to sample interior points.
    - *Design Motivation*: 2D proposals provide open-world coverage and large-scale data; 3D proposals provide complete interior structure supervision and human semantic annotations. The two sources are complementary.

2. **Triplet Contrastive Learning**

    - *Function*: For each part proposal $P$, triplets $(\mathbf{p}_a, \mathbf{p}_b, \mathbf{p}_c)$ are sampled, where $\mathbf{p}_a, \mathbf{p}_b \in P$ (positive pair) and $\mathbf{p}_c \in S \setminus P$ (negative sample).
    - *Core loss (relative contrastive loss)*:
        $$\mathcal{L} = -\frac{1}{2} \left( \log \frac{\text{sim}(f(\mathbf{p}_a), f(\mathbf{p}_b))}{\text{sim}(f(\mathbf{p}_a), f(\mathbf{p}_b)) + \text{sim}(f(\mathbf{p}_a), f(\mathbf{p}_c))} + \log \frac{\text{sim}(f(\mathbf{p}_b), f(\mathbf{p}_a))}{\text{sim}(f(\mathbf{p}_b), f(\mathbf{p}_a)) + \text{sim}(f(\mathbf{p}_b), f(\mathbf{p}_c))} \right)$$
        where $\text{sim}(u, v) = \exp(\cos(u, v) / \tau)$ and $\tau$ is a learnable temperature.
    - *Key distinction from prior methods*: Rather than directly minimizing/maximizing feature distances (pull/push loss), the loss only constrains relative relationships ($\mathbf{p}_a$ should be closer to $\mathbf{p}_b$ than to $\mathbf{p}_c$). This naturally accommodates multi-scale parts without additional scale conditioning.
    - *Design Motivation (Fig. 3)*: A single point may simultaneously belong to multiple parts at different scales (e.g., finger ⊂ hand ⊂ arm). Direct pull/push losses across proposals at different scales produce conflicting gradients, whereas triplet relative constraints allow the feature field to implicitly encode hierarchical relationships.

3. **Hard Negative Mining**

    - *Function*: A mixture of three negative sampling strategies improves training efficiency.
    - *Three strategies*:
        - **Uniform negatives**: uniformly sampled from the complement of the proposal region.
        - **3D hard**: biased toward negatives that are geometrically close to $\mathbf{p}_a$ in Euclidean space (near part boundaries).
        - **Feature hard**: biased toward negatives that are close to $\mathbf{p}_a$ in feature space.
    - Multiple negatives are processed in parallel by accumulating $\text{sim}(\mathbf{p}_a, \mathbf{p}_c)$ terms in the denominator, improving efficiency.
    - *Design Motivation*: Ablation studies (Fig. 9) show that hard negative mining significantly sharpens part boundaries.

4. **Feed-Forward Architecture (PVCNN + Triplane + Transformer)**

    - *Function*: Encodes point cloud input into a triplane feature field.
    - *Architecture details*: 448-dimensional feature field; triplane resolution $512^2$, 128 channels; 6-layer Transformer. Input: 100K points per shape.
    - *Training*: 8× A100 GPUs, 2 weeks.
    - *Advantages*: (a) Fast inference (<10 s vs. minutes–hours); (b) robustness to noisy and inconsistent labels via large-scale averaging; (c) naturally consistent feature space across shapes.

### Loss & Training

- Only the triplet contrastive loss described above is used.
- All shapes are normalized to $[-1, 1]^3$.
- Training data: ~340K Objaverse shapes (2D proposals) + ~30K PartNet shapes (3D proposals, constituting only 8% of training data).
- SAM2 uses densely sampled $32 \times 32$ point prompts per image to generate masks at multiple scales.

## Key Experimental Results

### Main Results

**PartObjaverse-Tiny (200 shapes, open-world):**

| Method | Type | Mean mIoU↑ | Inference Time |
|--------|------|-----------|----------------|
| PartSLIP | Text prompt | 31.54 | ~4 min |
| Find3D | Text prompt, feed-forward | 21.28 | ~10 s |
| Ultrametric | Per-shape optimization | 46.39 | ~1.5 h |
| SAMesh | Per-shape optimization | 56.86 | ~7 min |
| SAMPart3D | Per-shape optimization | 53.47 | ~15 min |
| **PartField** | **Feed-forward** | **79.18** | **~10 s** |

PartField achieves 79.18% mIoU, surpassing the second-best method SAMesh (56.86%) by 22.3 percentage points, while matching Find3D as the fastest method at inference.

**PartNetE (1906 shapes, 45 articulated part categories):**

| Method | Mean mIoU↑ | Inference Time |
|--------|-----------|----------------|
| PartSLIP | 34.94 | ~4 min |
| Find3D | 21.69 | ~10 s |
| SAMesh | 26.66 | ~7 min |
| SAMPart3D | 56.17 | ~15 min |
| **PartField** | **59.10** | **~10 s** |

### Ablation Study

| Configuration | mIoU↑ | Note |
|---------------|-------|------|
| Objaverse (2D) only | 77.70 | Strong baseline with 2D proposals alone |
| + PartNet (3D) | 77.90 | 3D proposals yield marginal improvement |
| + Hard Negative | 78.90 | Hard negatives bring significant improvement |
| **All combined** | **79.20** | Best configuration |

### Key Findings

- **Text-prompt-based methods perform worst in open-world settings** (Find3D: 21.28, PartSLIP: 31.54), indicating that precisely describing 3D parts in natural language remains a hard problem.
- Per-shape optimization methods suffer from multi-view inconsistency; feed-forward models naturally suppress noise through the averaging effect of large-scale training.
- Even though PartNet 3D data constitutes only 8% of training data and covers only 24 categories, it still benefits open-world tasks.
- **Cross-shape consistency emerges without explicit supervision**: without any cross-shape training signal, the feature space exhibits semantic consistency across different shapes (e.g., characters in different poses, aircraft of different types).
- PartField generalizes directly to AI-generated assets (Trellis, Edify3D), real-world 3D Gaussian Splats, and CAD models.

## Highlights & Insights

1. **Replacing absolute pull/push losses with relative triplet contrastive objectives** is an elegant solution for handling multi-scale and hierarchical annotations. No explicit scale conditioning is needed; the model discovers hierarchical structure from data.
2. **The robustness advantage of feed-forward models** is clearly demonstrated: per-shape optimization is sensitive to 2D prediction noise for each individual shape, whereas the prior learned by a feed-forward model over large-scale training effectively smooths out such noise.
3. Cross-shape consistency as an **emergent property** strongly suggests the potential of contrastive learning for representation learning on large-scale 3D data.
4. The triplane representation enables continuous feature queries at arbitrary positions, directly supporting hierarchical clustering for extracting multi-level part structures.

## Limitations & Future Work

- The PVCNN + triplane architecture is extrinsic; features are weakly tied to 3D position, and cross-shape applications require consistent shape orientation.
- Evaluation is currently limited to the object scale; extension to large-scale scenes has not been explored.
- Cross-shape applications (co-segmentation, correspondence) are only preliminarily explored and warrant further investigation.
- 3D proposals from PartNet cover only 24 categories; incorporating richer 3D annotation data may yield further improvements.

## Related Work & Insights

- The success of SAM2 in 2D is systematically "lifted" to 3D in this work, with the key distinction being the use of a feed-forward model rather than per-shape distillation.
- SimCLR-style contrastive learning is cleverly adapted to hierarchical part learning in 3D geometry.
- The triplet contrastive learning framework has direct applicability to other 3D tasks requiring learning from multi-source or multi-granularity annotations (e.g., material segmentation, functionality analysis).
- The emergence of cross-shape consistency provides strong evidence in support of 3D foundation model research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](../../NeurIPS2025/segmentation/partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
