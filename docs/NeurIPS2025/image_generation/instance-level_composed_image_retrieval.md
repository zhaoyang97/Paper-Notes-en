---
title: >-
  [Paper Note] Instance-Level Composed Image Retrieval
description: >-
  [NeurIPS 2025][Image Generation][Composed Image Retrieval] This paper proposes the instance-level composed image retrieval (i-CIR) benchmark and a training-free method, BASIC…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Composed Image Retrieval"
  - "Instance-Level Retrieval"
  - "VLM"
  - "Training-Free"
  - "Feature Fusion"
date: 2026-05-08
content_hash: 769e81f2ef8fefb1
---

# Instance-Level Composed Image Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2510.25387](https://arxiv.org/abs/2510.25387)  
**Code**: [GitHub](https://github.com/billpsomas/icir) | [Project Page](https://vrg.fel.cvut.cz/icir/)  
**Area**: Image Retrieval / Multimodal
**Keywords**: Composed Image Retrieval, Instance-Level Retrieval, VLM, Training-Free, Feature Fusion

## TL;DR

This paper proposes the instance-level composed image retrieval (i-CIR) benchmark and a training-free method, BASIC, which independently estimates image and text query similarities and fuses them via multiplicative combination, achieving state-of-the-art performance on both i-CIR and existing CIR datasets without any training.

## Background & Motivation

### Composed Image Retrieval (CIR)

Composed image retrieval is a prominent direction in image retrieval: given a reference image and a textual modification description (e.g., "change it to red"), the goal is to retrieve target images satisfying both conditions.

Existing CIR research faces two core bottlenecks:

**Insufficient data quality**: Most existing CIR datasets operate at the semantic level (e.g., FashionIQ, CIRR), where retrieval targets are images of the same category but different instances — which diverges from real-world needs (finding the same object under different conditions).

**Scarcity of training data**: High-quality CIR training samples are difficult to obtain at scale, limiting the performance of supervised methods.

### Instance-Level vs. Semantic-Level

| Dimension | Semantic-Level CIR | Instance-Level CIR (i-CIR) |
|---|---|---|
| Target | Other images of the same category | The same specific object |
| Example | "A similar red dress" | "The same red dress, outdoors" |
| Difficulty | Intra-category discrimination | Instance-level discrimination + condition matching |
| Application | Shopping recommendation | Landmark recognition, object tracking |

The instance-level formulation is closer to real-world needs but also significantly more challenging.

## Method

### Overall Architecture

The paper contributes two components:
1. **i-CIR Dataset**: The first instance-level CIR evaluation benchmark.
2. **BASIC Method**: A training-free CIR approach exploiting frozen features from pre-trained VLMs.

### Key Designs

**1. i-CIR Dataset Construction**

The careful design of the dataset is a major contribution of this work:

- **202 object instances**: Covering diverse categories including architectural landmarks, consumer goods, fictional characters, and technological devices.
- **1,883 composed queries**: Each query consists of an instance image paired with a textual modification, addressing changes in appearance, environment, attributes, and viewpoint.
- **750K database images**: Including positive samples and carefully curated hard negatives.
- **Hard negative design**: Three types:
    - Visual hard negatives: Visually similar images but not the same instance.
    - Textual hard negatives: Textually aligned images but from a different instance.
    - Compositional hard negatives: Images nearly satisfying both conditions but actually failing both.
- **Compact yet challenging**: Although the database contains only 750K images, the hard negative design makes retrieval difficulty equivalent to searching within 40M distractors.

**2. BASIC Method**

BASIC (Baseline Approach for Surprisingly strong Composition) is a training-free method whose core mechanism is to process image and text queries independently and then fuse the resulting similarity scores:

**Step 1: Feature Standardization**
- Centers VLM features using mean statistics pre-computed on the LAION-1M dataset.
- Removes global bias to improve feature discriminability.

**Step 2: Contrastive PCA Projection**
- Constructs a contrastive feature space using a positive corpus (object descriptions) and a negative corpus (style descriptions).
- Projects image features into an "object" subspace via PCA, suppressing background and style information.
- Formulation: $\mathbf{f}' = \text{PCA}_{C^+, C^-}(\mathbf{f}, \alpha)$, where $\alpha$ controls the weight of the negative corpus.

**Step 3: Query Expansion**
- Retrieves the top-$k$ most similar database images using the reference image.
- Averages their features to form an expanded query, improving robustness of the image-side query.

**Step 4: Query Text Conditioning**
- Expands short textual modifications into full caption format consistent with CLIP pre-training.
- Appends object names from the corpus as context to stabilize text representations.

**Step 5: Harris Corner Fusion**
- Independently computes image similarity $s_I$ and text similarity $s_T$.
- Applies normalized min-based scaling and fuses them using a penalty term inspired by Harris corner detection:

$$s = s_I \cdot s_T - \lambda \cdot (s_I - s_T)^2$$

- Rationale: Rewards candidates that simultaneously satisfy both queries (AND logic), and penalizes candidates scoring high on only one modality.

### Loss & Training

- **No training required**: BASIC operates entirely on frozen CLIP/SigLIP features; all operations are computed online at query time.
- No learnable parameters; no backpropagation.
- Supports CLIP ViT-L/14 and SigLIP ViT-L-16 as backbones.

## Key Experimental Results

### Main Results: i-CIR Benchmark

Comparison of methods on i-CIR in terms of mAP (%):

| Method | Type | Legacy Macro mAP | Refined Macro mAP | Average |
|---|---|---|---|---|
| Text | Unimodal | 0.74 | 1.09 | 0.92 |
| Image | Unimodal | 3.84 | 6.32 | 5.08 |
| Text + Image (additive) | Baseline | 6.21 | 9.30 | 7.76 |
| Text × Image (multiplicative) | Baseline | 7.83 | 9.79 | 8.81 |
| CIReVL | Training-free | 18.11 | 17.80 | 17.96 |
| FREEDOM | Trained | 29.91 | 26.10 | 28.01 |
| CoVR | Trained | 11.52 | 24.93 | 18.23 |
| **BASIC** | **Training-free** | **32.13** | **31.65** | **31.89** |

BASIC outperforms all methods, including the supervised FREEDOM, while being entirely training-free.

### Comparison on Existing Semantic-Level CIR Datasets

BASIC also achieves strong performance on traditional CIR benchmarks:

| Method | Type | FashionIQ (R@10) | CIRR (R@1) | GeneCIS |
|---|---|---|---|---|
| Pic2Word | ZS | 26.2 | 23.9 | — |
| Searle | ZS | 24.2 | 24.2 | — |
| CIReVL | ZS | 25.0 | 24.6 | — |
| MagicLens | Trained | 29.1 | 28.3 | — |
| **BASIC** | **ZS** | **31.8** | **29.7** | **SOTA** |

BASIC surpasses supervised methods under a zero-shot (training-free) setting.

### Ablation Study

**Contribution of each component (i-CIR Macro mAP %)**:

| Configuration | CLIP mAP | SigLIP mAP |
|---|---|---|
| Naive multiplicative fusion | 7.83 | 9.86 |
| + Feature standardization | 14.2 | 15.8 |
| + Contrastive PCA projection | 22.5 | 24.1 |
| + Query expansion | 28.7 | 30.2 |
| + Text conditioning | 30.1 | 31.3 |
| + Harris fusion (Full BASIC) | 32.1 | 31.6 |

Each component yields consistent gains; contrastive PCA projection and query expansion contribute the most.

**Effect of fusion weight $\lambda$**:

| $\lambda$ | i-CIR mAP | Note |
|---|---|---|
| 0.0 | 28.3 | Pure multiplicative fusion |
| 0.05 | 30.5 | Mild penalty |
| 0.1 | 32.1 | Optimal |
| 0.2 | 31.4 | Slight over-penalization |
| 0.5 | 29.1 | Excessive penalty |

### Key Findings

1. **Instance-level CIR genuinely requires composition**: Performance on i-CIR peaks at intermediate fusion weights, confirming that both modalities are essential.
2. **Training-free methods can surpass supervised ones**: BASIC demonstrates that frozen VLM features contain rich compositional retrieval capacity.
3. **Geometric operations in feature space are highly effective**: Simple operations such as standardization, projection, and expansion prove more robust than complex learned approaches.
4. **Hard negatives are critical for meaningful evaluation**: The hard negative design makes the 750K database effectively as challenging as a 40M-scale distractor set.

## Highlights & Insights

1. **Advancing the problem definition**: Moving from semantic-level to instance-level CIR better reflects real-world retrieval needs.
2. **Elegance of BASIC**: Without neural network training or complex pipelines, the method achieves state-of-the-art performance through purely geometric operations on frozen features.
3. **Clever borrowing of Harris corner fusion**: The classical corner detection idea from computer vision is repurposed for feature fusion, rewarding candidates that score strongly on both modalities.
4. **Careful dataset design**: The three-fold hard negative structure (visual / textual / compositional) ensures the validity of the evaluation.
5. **Fully open-source**: Dataset, code, and evaluation tools are all publicly released.

## Limitations & Future Work

1. **Limited dataset scale**: 202 instances may be insufficient to cover all retrieval scenarios.
2. **Dependence on VLM quality**: The performance ceiling of BASIC is bounded by the representational capacity of the underlying VLM (CLIP/SigLIP).
3. **Query expansion adds overhead**: An additional retrieval pass is required at inference time, increasing latency.
4. **Sensitivity of PCA to corpus selection**: The choice of positive and negative corpora may affect performance across different domains.
5. **Integration with trained methods unexplored**: Whether BASIC's components can serve as initialization or augmentation for supervised CIR methods remains an open question.

## Related Work & Insights

- **CIRR / FashionIQ**: Classic CIR datasets, but limited to the semantic level; i-CIR fills the instance-level gap.
- **CIReVL / Pic2Word**: Pioneering training-free CIR methods; BASIC substantially advances beyond them.
- **FREEDOM / CoVR**: Supervised CIR methods; BASIC demonstrates that training-free approaches can surpass them.
- **Contrastive PCA**: Transferred from the contrastive decoding idea in NLP to the visual feature space, proving highly effective.
- Insight: **Geometric operations on frozen features are an underestimated tool** — the representational potential of pre-trained features should be fully explored before resorting to additional learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (new dataset formulation + elegant method design)
- Technical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐ (fully open-source + no training required)
- Writing Quality: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval](../../CVPR2026/image_generation/wiser_wider_search_deeper_thinking_and_adaptive_fusion_for_training-free_zero-sh.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[NeurIPS 2025\] Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)
- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)

</div>

<!-- RELATED:END -->
