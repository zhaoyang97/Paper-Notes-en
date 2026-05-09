---
title: >-
  [Paper Note] KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering
description: >-
  [CVPR 2026][Multimodal VLM][Image Clustering] KEC leverages LLMs to construct hierarchical concept-attribute structured textual knowledge to guide image clustering, outperforming zero-shot CLIP on 14 out of 20 datasets without any training, demonstrating that discriminative attributes are more effective than simple class names.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Image Clustering
  - Textual Knowledge
  - Large Language Models
  - CLIP
  - Discriminative Attributes
date: 2026-05-08
content_hash: d79602a312c40f37
---

# KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering

**Conference**: CVPR 2026
**arXiv**: [2604.11144](https://arxiv.org/abs/2604.11144)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Image Clustering, Textual Knowledge, Large Language Models, CLIP, Discriminative Attributes

## TL;DR
KEC leverages LLMs to construct hierarchical concept-attribute structured textual knowledge to guide image clustering, outperforming zero-shot CLIP on 14 out of 20 datasets without any training, demonstrating that discriminative attributes are more effective than simple class names.

## Background & Motivation

**Background**: Image clustering has evolved from geometric priors → deep representation learning → vision-language model-assisted paradigms. VLMs such as CLIP have made it possible to inject textual knowledge into clustering.

**Limitations of Prior Work**: Existing methods either employ VLMs to generate per-image captions (computationally expensive) or select shallow nouns from WordNet (semantically redundant and inconsistent in granularity). Naively introducing textual knowledge can even degrade clustering performance.

**Key Challenge**: Visually similar but semantically distinct categories (e.g., Akita Inu vs. Shiba Inu) cannot be distinguished by class names alone; discriminative attributes (leg length, tail curvature, ear posture) are required. However, acquiring such attributes demands domain expertise and is difficult to automate.

**Core Idea**: Use LLMs to distill abstract concepts from redundant nouns, then automatically extract intra-concept and inter-concept discriminative attributes, constructing hierarchical knowledge for feature enhancement.

## Method

### Overall Architecture
Images → CLIP visual features → alignment with WordNet nouns → LLM-based distillation of representative concepts → LLM extraction of single-concept and concept-pair discriminative attributes → instantiation as knowledge-enhanced features per image → combination with visual features → input to downstream clustering algorithms.

### Key Designs

1. **Concept Abstraction**:

    - Function: Distills representative concepts from redundant WordNet nouns.
    - Mechanism: Images are first mapped to their nearest nouns via CLIP; an LLM then merges semantically overlapping nouns into more abstract concept categories.
    - Design Motivation: WordNet contains excessive synonyms and near-synonyms (e.g., car/automobile/vehicle), and using them directly dilutes inter-category discriminability.

2. **Discriminative Attribute Extraction**:

    - Function: Automatically generates distinguishing attributes for pairs of similar concepts.
    - Mechanism: Single-concept attributes (LLM describes typical features of each concept) + concept-pair attributes (LLM contrasts differential features between two similar concepts). For example, "Akita Inu vs. Shiba Inu" → "body size, coat length, ear shape."
    - Design Motivation: Humans distinguish similar objects precisely through discriminative attributes; CLIP attention maps confirm that attribute descriptions direct the model's focus toward relevant regions.

3. **Knowledge Instantiation and Feature Fusion**:

    - Function: Converts structured knowledge into per-image enhanced features.
    - Mechanism: Attribute descriptions are encoded by the CLIP text encoder; cosine similarities with the image are computed as attribute scores, concatenated into a knowledge-enhanced feature vector, and combined with the original visual features via weighted aggregation.
    - Design Motivation: Grounding global knowledge to individual image instances ensures that different images receive distinct knowledge enhancements.

### Loss & Training
KEC requires no training; it directly generates enhanced features that are fed into existing clustering algorithms (e.g., K-means, spectral clustering).

## Key Experimental Results

### Main Results

| Comparison | Metric | KEC (Training-Free) | Trained Methods | Notes |
|---|---|---|---|---|
| Average over 20 datasets | NMI | Superior | ~3% lower | KEC surpasses trained methods without training |
| vs. CLIP zero-shot | Acc | Wins on 14/20 datasets | — | — |

### Ablation Study

| Configuration | NMI | Notes |
|---|---|---|
| KEC (full) | Best | Concepts + attributes + fusion |
| Naive textual knowledge | Degraded or negative | Demonstrates necessity of structured knowledge |
| Concepts only, no attributes | Moderate | Attributes contribute significantly |
| Single-concept attributes only | Sub-optimal | Concept-pair attributes provide further gains |

### Key Findings
- Naively introducing textual knowledge (e.g., directly using nouns) degrades performance on certain datasets, confirming the necessity of structured knowledge.
- Concept-pair discriminative attributes contribute more than single-concept attributes, indicating that contrastive information is critical for distinguishing similar categories.
- KEC is insensitive to the choice of downstream clustering algorithm, exhibiting broad compatibility.

## Highlights & Insights
- **LLM as a Knowledge Source**: No image input to the LLM is required; sufficient discriminative knowledge is obtained purely through textual interaction at minimal cost.
- **Structured > Naive**: Demonstrates that *knowledge quality* matters more than *knowledge quantity*.

## Limitations & Future Work
- Performance depends on the quality of CLIP's text-image alignment.
- LLM-generated attributes may be biased.
- No comparison against specialized fine-grained methods on fine-grained datasets.

## Related Work & Insights
- **vs. SIC/TAC**: These methods annotate directly with shallow nouns or WordNet, leading to severe semantic redundancy.
- **vs. VLM captioning**: Per-image caption generation is computationally intensive and does not scale well.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear and well-motivated hierarchical knowledge construction pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across 20 datasets is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are described clearly.
- Value: ⭐⭐⭐⭐ Surpassing trained methods without training demonstrates strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[NeurIPS 2025\] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](../../NeurIPS2025/multimodal_vlm/hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](tipsv2_patch_text_alignment.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
