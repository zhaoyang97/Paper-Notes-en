---
title: >-
  [Paper Note] Generalizable Object Re-Identification via Visual In-Context Prompting
description: >-
  [ICCV 2025][Multimodal VLM][Object Re-Identification] VICP proposes a generalizable object re-identification framework in which an LLM infers identity-discriminative rules from a small set of positive/negative image pairs and converts them into dynamic visual prompts injected into a frozen visual foundation model (DINOv2), enabling zero-parameter-update generalization to unseen object categories.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Object Re-Identification"
  - "Generalizable ReID"
  - "Visual In-Context Prompting"
  - "LLM-Guided"
  - "Visual Foundation Model"
date: 2026-05-08
content_hash: 96b929ca54ccbe5a
---

# Generalizable Object Re-Identification via Visual In-Context Prompting

**Conference**: ICCV 2025
**arXiv**: [2508.21222](https://arxiv.org/abs/2508.21222)  
**Code**: [https://github.com/Hzzone/VICP](https://github.com/Hzzone/VICP)  
**Area**: Multimodal / Vision-Language Model
**Keywords**: Object Re-Identification, Generalizable ReID, Visual In-Context Prompting, LLM-Guided, Visual Foundation Model

## TL;DR

VICP proposes a generalizable object re-identification framework in which an LLM infers identity-discriminative rules from a small set of positive/negative image pairs and converts them into dynamic visual prompts injected into a frozen visual foundation model (DINOv2), enabling zero-parameter-update generalization to unseen object categories.

## Background & Motivation

- **Traditional ReID methods** train category-specific models (pedestrian, vehicle), resulting in poor generalizability; every new category requires expensive annotation and retraining.
- **Self-supervised learning** (DINO, MoCo, etc.) reduces annotation requirements but learns semantic consistency rather than identity-sensitive features (e.g., stitching texture on a backpack, sole pattern on a shoe), yielding suboptimal ReID performance.
- **Core Problem**: How to build a ReID model that generalizes to arbitrary object categories without category-specific training?
- **Key Insights**:
    - Visual Foundation Models (VFMs) possess strong visual priors, but their general-purpose features lack the fine-grained identity discriminability required for ReID.
    - Large Language Models (LLMs) excel at in-context learning—inferring task rules from a handful of examples.
    - Unifying both: LLM infers identity-discriminative rules → generates visual prompts → VFM extracts identity-sensitive features.

## Method

### Overall Architecture

The VICP framework consists of two main modules:
1. **In-Context Visual Prompt Generation**: Processes a small set of positive/negative pairs, uses a frozen LLM to infer identity rules, and generates visual prompts.
2. **Generalizable Object ReID**: Injects the visual prompts into a frozen ViT (DINOv2) to dynamically modulate self-attention toward identity-sensitive features.

### Key Designs

1. **In-Context Visual Prompt Generation**:

    - Input: support set $\mathcal{S} = \{(\boldsymbol{x}_i, \boldsymbol{x}_j, y_{ij})\}$ (positive/negative pairs).
    - Each image is encoded by DINOv2; a **Q-Former** (Query-based Connector, inspired by BLIP-2) compresses each image into $N$ latent tokens.
    - For each pair, the compressed tokens of both images are concatenated with label embeddings: $\mathbf{T}_{ij} = [\mathbf{I}_i; \mathbf{I}_j; \mathbf{L}_{ij}]$.
    - $K$ pairs form the complete context sequence $\mathbf{T}_{\text{ctx}}$.
    - A frozen LLM (LLaMA) processes the sequence autoregressively; loss is computed only on label tokens (ICL Loss).
    - $M$ learnable visual prompt tokens $\mathbf{P}_{\text{learn}}$ are appended at the end of the sequence.
    - LLM outputs are mapped by a Visual Head (two-layer MLP) to visual prompts $\mathbf{P}_{\text{task}} \in \mathbb{R}^{M \times d_{\text{vision}}}$.

2. **Prompt Injection into VFM**:

    - $\mathbf{P}_{\text{task}}$ is concatenated to the input token sequence of each ViT layer.
    - Through self-attention, prompt tokens interact with spatial features, dynamically amplifying identity-sensitive regions (logos, textures) and suppressing irrelevant regions (background, lighting variations).
    - ViT parameters are fully frozen; only prompts modulate the feature space.
    - At inference time, prompts can be cached and reused—only one prompt generation is needed per category for all query-gallery comparisons.

3. **Loss Function Design**:

    - **ReID Loss (Triplet Loss)**: $\mathcal{L}_{\text{ID}} = \sum \max(0, \alpha - \text{sim}(\phi(\boldsymbol{x}_a), \phi(\boldsymbol{x}_p)) + \text{sim}(\phi(\boldsymbol{x}_a), \phi(\boldsymbol{x}_n)))$
        - Triplet loss is preferred over ArcFace/contrastive loss because it penalizes only margin-violating samples, imposing softer updates that preserve the pretrained model's semantic priors.
    - **Patch Alignment Loss (OT Distance)**: $\mathcal{L}_{\text{align}}$, measuring patch-level feature matching quality via optimal transport distance, aligning positive pairs and separating negative pairs.
    - **ICL Loss**: Supervises only label token predictions, preserving the LLM's pretrained semantic knowledge.
    - Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ID}} + \lambda_{\text{ICL}} \mathcal{L}_{\text{ICL}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}$

### Loss & Training

- Backbone: DINOv2 ViT-small.
- Trained on 2× H100 GPUs; learning rate $10^{-4}$; batch size 256.
- Image resolution $224 \times 224$; data augmentation: horizontal flip only.
- 10 training epochs; 64 positive/negative pairs randomly sampled per batch.
- Q-Former generates 32 visual tokens.
- Triplet loss margin $\alpha = 0.1$.

## Key Experimental Results

### Main Results

**PetFace Dataset**:

| Method | AUC↑ | ACC↑ | mAP↑ | Top-1↑ |
|--------|------|------|------|--------|
| CLIP | 71.5 | 64.6 | 7.1 | 4.4 |
| DINOv2 | 71.6 | 65.9 | 6.5 | 5.1 |
| Triplet+ | 92.5 | 85.6 | 49.8 | 47.7 |
| **VICP (Ours)** | **93.5** | **86.0** | **51.2** | **49.7** |
| Supervised (upper bound) | 95.5 | 89.3 | 57.7 | 56.3 |

**ShopID10K Dataset**:

| Method | mAP↑ | Rank-1↑ | Rank-5↑ |
|--------|------|---------|---------|
| CLIP | 37.1 | 48.6 | 72.1 |
| DINOv2 | — | — | — |
| Triplet+ | — | — | — |
| **VICP (Ours)** | ~**4% mAP** over Triplet+ | — | — |
| Supervised (upper bound) | 62.6 | 71.2 | 89.8 |

### Ablation Study

**Comparison of Loss Functions (PetFace)**:

| Method | AUC↑ | mAP↑ |
|--------|------|------|
| ArcFace | 89.1 | 46.6 |
| AdaFace | 89.3 | 46.9 |
| SCL (Contrastive) | 91.1 | 46.3 |
| Triplet | 91.7 | 48.2 |
| Triplet+ (few-shot) | 92.5 | 49.8 |
| **VICP** | **93.5** | **51.2** |

> Triplet loss outperforms ArcFace/AdaFace/SCL because the latter always minimize loss over all samples, potentially disrupting the generalization capacity of pretrained representations.

### Key Findings

- **DINOv2 applied directly to ReID performs poorly** (mAP only 6.5%), confirming that semantic features do not equate to identity features.
- **Fine-tuning with Triplet loss yields substantial gains** (mAP 48.2%), demonstrating that explicit identity-discriminative optimization is necessary for ReID.
- **VICP surpasses fine-tuned Triplet+ without any parameter updates**, validating the effectiveness of LLM-driven visual prompting.
- Strong performance across all unseen categories (pets, products, vehicles) confirms cross-category generalization capability.
- The ShopID10K dataset exposes significant challenges in real-world scenarios (lighting, occlusion, background variation).

## Highlights & Insights

- **Clear problem formulation**: The paper is the first to systematically define the task of "generalizable object re-identification" targeting arbitrary rather than specific categories.
- **The LLM→visual prompt pipeline** is elegant: the LLM "reasons" from a few positive/negative pairs about which features matter, then visual prompts guide the VFM to focus accordingly—analogous to the human cognitive process of "learning rules from examples."
- **Q-Former compression of visual tokens** effectively controls the computational overhead imposed on the LLM.
- **Prompt caching and reuse** reduces inference to a single prompt generation per category, making deployment compatible with standard ReID pipelines.
- **ShopID10K** fills an important gap in the field as a new benchmark dataset.
- The rationale for choosing Triplet loss over more aggressive metric learning objectives (preserving pretrained priors) is insightful.

## Limitations & Future Work

- The framework assumes that object categories are known at inference time (requiring an upstream detector), and does not handle cross-category ambiguity.
- The LLM processes visual tokens rather than raw images, potentially limiting its semantic reasoning capacity.
- Only DINOv2 ViT-small is used; larger backbone models may yield further improvements.
- The quality and representativeness of few-shot pairs are critical to prompt generation, yet the optimal support set selection strategy is not thoroughly investigated.
- Integration with more recent VLMs (e.g., GPT-4V) may represent a more powerful direction.

## Related Work & Insights

- **BLIP-2**: Source of inspiration for the Q-Former design, using learnable queries to compress visual tokens.
- **Visual Prompt Tuning (VPT)**: Methodological foundation for injecting prompt tokens into each ViT layer.
- **MegaDescriptor / PetFace**: Category-specific ReID baselines; the proposed method surpasses them in generality.
- **In-Context Learning** (GPT series): Core idea transferred from NLP to the visual domain.
- Insight: LLMs can not only understand text but also infer task-specific rules from sequences of visual tokens.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply LLM-driven in-context learning to generalizable ReID; the approach is conceptually original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 7 datasets, diverse baselines, comprehensive ablations, and a new dataset of independent value.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, method description is detailed, and figures are informative.
- **Value**: ⭐⭐⭐⭐ — Introduces a new task, a new dataset, and a new method, making a meaningful contribution to the ReID community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/multimodal_vlm/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)

</div>

<!-- RELATED:END -->
