---
title: >-
  [Paper Note] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts
description: >-
  [CVPR 2026][Multimodal VLM][Poster layout generation] This work introduces SciPostGen, a large-scale dataset of 18,097 paper–poster pairs. Analysis reveals a moderate correlation between paper structure and the number of poster layout elements. A retrieval-augmented poster layout generation framework is proposed, which leverages contrastive learning to retrieve layout templates matching the input paper and guides an LLM to generate the final poster layout.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Poster layout generation
  - scientific papers
  - retrieval-augmented generation
  - contrastive learning
  - document layout analysis
date: 2026-05-08
content_hash: c49d09f88bb434e9
---

# SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts

**Conference**: CVPR 2026
**arXiv**: [2511.22490](https://arxiv.org/abs/2511.22490)
**Code**: [https://omron-sinicx.github.io/paper2layout/](https://omron-sinicx.github.io/paper2layout/)
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Poster layout generation, scientific papers, retrieval-augmented generation, contrastive learning, document layout analysis

## TL;DR

This work introduces SciPostGen, a large-scale dataset of 18,097 paper–poster pairs. Analysis reveals a moderate correlation between paper structure and the number of poster layout elements. A retrieval-augmented poster layout generation framework is proposed, which leverages contrastive learning to retrieve layout templates matching the input paper and guides an LLM to generate the final poster layout.

## Background & Motivation

The volume of scientific publications continues to grow (monthly arXiv submissions increased from ~8,000 in 2015 to over 20,000 in 2025), and posters serve as an important medium for efficiently communicating research findings. Automatically generating posters from papers requires addressing two sub-problems: content summarization (what to include) and layout generation (how to arrange). Existing work predominantly focuses on content summarization, with layouts either relying on fixed templates or rule-based generation derived from paper structure. However, layout design significantly affects the effectiveness of information communication, making it worthwhile to learn the paper-to-layout mapping in a data-driven manner.

The core bottleneck is the **lack of large-scale paired datasets**. Existing poster generation datasets contain only a few hundred paper–poster pairs, insufficient for data-driven approaches. SciPostGen scales this to 18,097 pairs through a combination of automatic annotation and manual correction, providing fine-grained annotations for both papers (OCR text, figure bounding boxes) and posters (8 categories of layout elements).

Analysis reveals exploitable correlations between paper structure and poster layout: papers with more text tend to have fewer figure elements in their posters (Spearman $\rho < -0.40$), while the number of paper figures correlates positively with poster figure elements. This motivates a retrieval-augmented layout generation strategy—retrieving poster layouts from structurally similar papers as generation references.

## Method

### Overall Architecture

The system consists of two modules: (1) a **layout retriever**—a contrastive learning-trained paper encoder and layout encoder that map paper page images and poster layout images into a shared embedding space, retrieving the top-3 most similar layouts at inference time; and (2) a **layout generator**—based on Llama-3.1-8B-Instruct, which takes the retrieved layouts and paper structural information as input and outputs the final layout (category + normalized bounding boxes). Both automatic and semi-automatic modes are supported.

### Key Designs

1. **Contrastive Learning Layout Retriever**:

    - Function: Retrieve poster layouts that match the structural characteristics of a given paper.
    - Mechanism: The paper encoder renders multi-page PDFs as image sequences; each page is processed by DiT (Document image Transformer) to extract patch features, which are aggregated into a paper embedding $x^p$ via two-level attention pooling (intra-page → inter-page). The layout encoder processes rendered layout images analogously to obtain $x^l$. The model is trained with an InfoNCE contrastive loss, treating paired paper–layout instances as positives and in-batch others as negatives. At inference, cosine similarity is used to retrieve top-3 layouts from the training set.
    - Design Motivation: Moderate correlations exist between paper structure (text volume, figure count) and poster layout, and image-based encoding can implicitly capture these structural features. Retrieving multiple layouts rather than a single template accommodates the diversity of poster designs.

2. **LLM Layout Generator**:

    - Function: Integrate retrieval results and paper structural constraints to generate the final layout.
    - Mechanism: Paper structure (section count, figure/table counts and aspect ratios) and retrieved layouts are serialized as text sequences and fed to the LLM, which is prompted to generate a layout sequence $L = \{(c_i, b_i)\}$. In semi-automatic mode, user-specified partial layout constraints (e.g., two pre-placed largest elements) are additionally provided, and the model completes the remaining elements within those constraints.
    - Design Motivation: LLMs flexibly integrate heterogeneous inputs; compared to dedicated layout models based on GANs, Transformers, or Diffusion, they more naturally incorporate retrieval results, paper structure, and user constraints.

3. **Semi-Automatic Constraint Mechanism**:

    - Function: Simulate real-world workflows where a creator places primary elements and the system completes the remaining layout.
    - Mechanism: The two largest elements (by area) from the gold layout are provided as constraints, and the system generates the remaining elements. This simulates a "human sets the major structure, AI fills in the details" collaborative mode.
    - Design Motivation: Fully automatic generation struggles to satisfy personalized requirements; the semi-automatic mode strikes a balance between practicality and automation.

### Loss & Training

The retriever is trained with an InfoNCE contrastive loss: $\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}$, where $s_{ij}$ denotes the cosine similarity between paper and layout embeddings. The generator is fine-tuned from Llama-3.1-8B-Instruct using silver layout annotations from the SciPostGen training set.

## Key Experimental Results

### Main Results

**Layout Retrieval Performance (Paper → Layout Retrieval)**

| Method | Recall@1 | Recall@3 | Recall@5 |
|--------|----------|----------|----------|
| Random | 0.05 | 0.15 | 0.25 |
| Paper encoder only | 4.83 | 12.12 | 18.11 |
| Full retriever | **8.20** | **19.87** | **28.37** |

**Layout Generation Quality (FID / mIoU / Alignment)**

| Configuration | FID ↓ | mIoU ↑ | Overlap ↓ |
|---------------|-------|--------|-----------|
| Without retrieval | baseline | baseline | baseline |
| + Retrieval augmentation | improved | improved | reduced |
| + Retrieval + constraints (semi-automatic) | best | best | lowest |

### Ablation Study

| Configuration | Retrieval Recall@3 | Generation FID | Notes |
|---------------|--------------------|----------------|-------|
| Image encoding only (DiT) | 19.87 | — | Base retrieval performance |
| Layout annotation encoding only | lower | — | Image encoding outperforms structured annotations |
| Direct generation without retrieval | — | higher | Poor quality without reference layouts |
| Top-1 retrieval | — | moderate | Insufficient diversity with single template |
| Top-3 retrieval | — | lowest | Multiple templates provide better guidance |

### Key Findings

- The Spearman correlation between paper structure and poster layout is moderate (|ρ| ≈ 0.40–0.50), indicating that structural information is useful but insufficient to fully determine the layout.
- Image-based encoding outperforms directly using layout annotations as input—images implicitly preserve spatial relationships.
- In semi-automatic mode, the addition of constraints substantially improves consistency between generated and ground-truth layouts.
- The mAP@0.50:0.95 between silver (automatically annotated) and gold (manually corrected) layouts is 0.53, indicating moderate agreement.

## Highlights & Insights

- **The dataset construction methodology is instructive**: automatic annotation (Azure Document Intelligence + Nougat OCR) combined with manual correction for validation/test sets balances scale and quality—a practical strategy under limited annotation resources.
- **The research problem of "paper structure → poster layout" is itself novel**: prior work focuses on content summarization; this paper is the first to systematically study the structure-to-layout mapping and quantitatively analyze their correlations.
- **Retrieval-augmented strategy**: using retrieved layouts from structurally similar papers as "reference designs" to guide generation is more controllable and diverse than generating from scratch.

## Limitations & Future Work

- Only layout bounding boxes are generated, not actual poster content (text, images), leaving a substantial gap to end-to-end poster generation.
- The dataset is limited to computer science conferences (CVPR/ICLR/ICML/NeurIPS); poster styles in other disciplines may differ.
- Retrieval Recall@1 is only 8.2%, suggesting that the paper-to-layout mapping remains weak and may require richer paper representations.
- The subjective quality of generated layouts (e.g., readability, aesthetics) is not evaluated; only numerical metrics are used.

## Related Work & Insights

- **vs. PosterLayout [56]**: Provides fine-grained layout annotations but only hundreds of pairs; SciPostGen is 30× larger in scale.
- **vs. rule-based layout**: Methods such as [42] derive layouts from predefined rules based on paper structure, lacking flexibility and diversity.
- **vs. general layout generation**: Advertisement/webpage layout generation methods cannot leverage paper structural information; this work introduces the paper as a conditioning signal.

## Rating

- Novelty: ⭐⭐⭐⭐ The research problem (paper → poster layout) is novel and the dataset is valuable, though the method itself is a standard retrieval-augmented + LLM combination.
- Experimental Thoroughness: ⭐⭐⭐ Lacks user studies and subjective evaluation; quantitative metrics for retrieval and generation are not comprehensive enough.
- Writing Quality: ⭐⭐⭐⭐ The dataset construction and analysis sections are clear; the overall structure is well-organized.
- Value: ⭐⭐⭐⭐ The dataset is a valuable contribution to the community, and the framework lays a foundation for automated academic poster generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU](responses_fall_short_of_understanding_gap_between_internal_representations_and_responses_in_vdu.md)
- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](../../AAAI2026/multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[AAAI 2026\] Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies](../../AAAI2026/multimodal_vlm/remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[ICLR 2026\] Closing the Modality Gap Aligns Group-Wise Semantics](../../ICLR2026/multimodal_vlm/closing_the_modality_gap_aligns_group-wise_semantics.md)

<!-- RELATED:END -->
