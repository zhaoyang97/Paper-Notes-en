---
title: >-
  [Paper Note] Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification
description: >-
  [ECCV 2024][Medical Imaging][few-shot learning] The PEMP framework is proposed, which integrates prior pathology knowledge (visual exemplars + textual descriptions) into patch-level and slide-level prompts. Combined with CLIP for multi-instance prompt learning, it outperforms SOTA methods by an average of 4% on few-shot weakly supervised WSI classification tasks.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "few-shot learning"
  - "Prompt Learning"
  - "Whole Slide Image"
  - "Multiple Instance Learning"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: 9460e18b41377536
---

# Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification

**Conference**: ECCV 2024  
**arXiv**: [2407.10814](https://arxiv.org/abs/2407.10814)  
**Code**: None  
**Area**: Medical Image Analysis / Pathological Imagery  
**Keywords**: few-shot learning, Prompt Learning, Whole Slide Image, Multiple Instance Learning, Vision-Language Model

## TL;DR

The PEMP framework is proposed, which integrates prior pathology knowledge (visual exemplars + textual descriptions) into patch-level and slide-level prompts. Combined with CLIP for multi-instance prompt learning, it outperforms SOTA methods by an average of 4% on few-shot weakly supervised WSI classification tasks.

## Background & Motivation

Whole Slide Image (WSI) classification is crucial in cancer diagnosis. However, due to the gigapixel resolution of WSIs, they are usually cropped into a large number of patches and processed using the Multiple Instance Learning (MIL) paradigm for weakly supervised classification. Existing MIL methods require extensive training samples. In clinical practice, however, obtaining WSI data is often highly constrained due to patient privacy, rare diseases, or emerging pathologies, leaving only a very small amount of WSI data available.

Few-shot Weakly Supervised WSI Classification (FSWC) has emerged to address this, but faces a Key Challenge: training samples are extremely scarce (e.g., 2/4/8/16/32 slides), with only slide-level annotations available. Prompt learning based on vision-language models such as CLIP is a promising direction, but existing methods (such as CoOp, TOP) either focus only on patch-level prompts or use only textual prompts, neglecting the association between highly specialized visual features and corresponding terminology in pathology.

Core Idea: Mimicking how pathologists learn from textbooks, task-related visual exemplars and textual descriptions are utilized as prior knowledge. These are simultaneously injected into both visual and textual prompts at the patch and slide levels, where bilateral knowledge enhancement guides the model to accurately identify key pathological patterns under few-shot scenarios.

## Method

### Overall Architecture

PEMP is built upon a frozen CLIP model and comprises three learning processes:
- **Visual Prompt Learning**: Incorporating patch-level and slide-level pathology image exemplars on the visual side.
- **Textual Prompt Learning**: Incorporating corresponding pathological language descriptions on the textual side.
- **Two-level Prompt Alignment**: Aligning visual and textual prompts via contrastive learning.

The input is a WSI and its slide label, and the output is the classification prediction probability. The pipeline involves patch feature extraction $\rightarrow$ visual exemplar matching $\rightarrow$ Messenger Layer (inter-patch modeling) $\rightarrow$ Summary Layer (aggregation into slide features) $\rightarrow$ contrastive classification with textual features.

### Key Designs

1. **Visual Prompt Construction**:

    - Function: Constructing typical patch-level and slide-level visual exemplars as fixed prompts for each classification task.
    - Mechanism: Selection of representative images by pathology experts from authoritative textbooks. For instance, pathological features of cervical cancer with a poor prognosis include patch exemplars like "high-grade atypia," "vascular invasion," and "necrosis," as well as slide exemplars like "blurred tumor boundaries and low stroma ratio."
    - Design Motivation: In few-shot scenarios, acquiring effective knowledge relying solely on limited training data is challenging. Introducing external typical exemplars guides the model to focus on task-relevant critical pathological patterns.
    - Implementation: Extracting exemplar features $z_l = \phi_{img}(e_l)$ using the CLIP image encoder, matching each patch with the most similar exemplar via cosine similarity, and concatenating them into enhanced features $f_{i,j}^e$.

2. **Messenger Layer and Summary Layer**:

    - Function: Modeling relationships between patches within the same slide and aggregating patch features into slide features.
    - Messenger Layer: A lightweight self-attention layer that takes enhanced patch features $F_i^e$ as input and captures spatial relationships among patches via a standard attention mechanism $F_i^{ML} = \text{softmax}(\frac{QK^\top}{\sqrt{d_w}})V$.
    - Summary Layer: An attention-pooling-based aggregation layer that computes a weighted sum of all patch features into a slide feature $F_i^S$ using learnable weights $a_{i,j} = \frac{\exp(w^\top \tanh(Vf_{i,j}^\top))}{\sum_j \exp(w^\top \tanh(Vf_{i,j}^\top))}$.
    - Contrast with Prior Work: Methods like TOP directly use mean pooling or simple attention, lacking interactive modeling among patches.

3. **Textual Prompt Construction**:

    - Function: Constructing a three-level structured prompt on the textual side: Slide Task Token, Slide-level Descriptive Token, and Patch-level Descriptive Token.
    - Each level contains fixed pathological descriptions and learnable prompt vectors (e.g., $[\alpha]_1[\alpha]_2\ldots[\alpha]_M$), corresponding to task category descriptions, slide-level pathological feature descriptions, and patch-level pathological feature descriptions, respectively.
    - Design Motivation: Professional terminology in pathological images might be "unseen" for CLIP, making it difficult to activate correct features solely through text. Matching these with visual exemplars facilitates cross-modal alignment.

4. **Two-level Alignment**:

    - Total Loss Function: $\mathcal{L}_{total} = \mathcal{L}_t + \lambda_1 \mathcal{L}_s + \lambda_2 \mathcal{L}_p$
    - $\mathcal{L}_t$: Alignment of slide visual features with slide textual features (to perform the classification task).
    - $\mathcal{L}_s$: Alignment of slide-level visual exemplars with slide-level textual descriptions.
    - $\mathcal{L}_p$: Alignment of patch-level visual exemplars with patch-level textual descriptions.
    - The basic form is the standard contrastive loss: $\mathcal{L} = -\sum_{F_i} \log \frac{\exp(\text{sim}(F_i, T_y)/\tau)}{\sum_{i=1}^{U} \exp(\text{sim}(F_i, T_i)/\tau)}$

### Loss & Training

- The three parts of AC-Loss share an identical form, which is the negative log-likelihood contrastive loss.
- The parameters of CLIP's image and text encoders are frozen, and only the learnable prompt vectors, Messenger Layer, Summary Layer, and projector are updated.
- During inference, the matching probability between visual features and each category's textual features is computed via softmax.

## Key Experimental Results

### Main Results

**Task 1: Cervical Cancer Survival Prognosis Prediction (C-index)**

| Dataset | Method | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|--------|------|---------|---------|--------|--------|--------|
| In-house | TOP (NeurIPS'23) | 0.652 | 0.608 | 0.574 | 0.539 | 0.508 |
| In-house | **PEMP (ours)** | **0.667** | **0.637** | **0.614** | **0.587** | **0.562** |
| TCGA-CESC | TOP | 0.611 | 0.597 | 0.566 | 0.536 | 0.518 |
| TCGA-CESC | **PEMP (ours)** | **0.637** | **0.624** | **0.602** | **0.577** | **0.551** |

**Task 2: Lymph Node Metastasis Prediction (AUC)**

| Dataset | Method | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|--------|------|---------|---------|--------|--------|--------|
| In-house | TOP | 0.825 | 0.819 | 0.801 | 0.787 | 0.762 |
| In-house | **PEMP** | **0.849** | **0.838** | **0.824** | **0.801** | **0.783** |
| TCGA-CESC | TOP | 0.799 | 0.761 | 0.744 | 0.708 | 0.679 |
| TCGA-CESC | **PEMP** | **0.818** | **0.795** | **0.760** | **0.726** | **0.704** |

**Task 3: Small Round Cell Tumor Subtype Classification (AUC)**

| Method | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|------|---------|---------|--------|--------|--------|
| TOP | 0.682 | 0.652 | 0.633 | 0.584 | 0.560 |
| **PEMP** | **0.751** | **0.718** | **0.685** | **0.643** | **0.625** |

On rare disease classification, PEMP achieves an average AUC gain of 6.2%, demonstrating a particularly pronounced advantage.

### Ablation Study

| Configuration | 32-shot | 2-shot | Description |
|------|---------|--------|------|
| w/o v&t em. (degraded to CoOp) | 0.641 | 0.490 | No visual/textual exemplars |
| w/o vision em. | 0.655 | 0.511 | Textual descriptions only |
| w/o text em. | 0.658 | 0.533 | Visual exemplars only |
| w/o Summary Layer | 0.632 | 0.487 | Replaced with mean pooling |
| w/o Messenger Layer | 0.664 | 0.554 | No patch-level interaction |
| w/o Slide-level Prompts | 0.656 | 0.525 | No slide-level prompt |
| w/o AC-Loss | 0.660 | 0.549 | No exemplar alignment loss |
| **PEMP (full)** | **0.667** | **0.562** | Full model |

### Key Findings

- **Summary Layer (Attention Pooling) contributes the most**: Removing it leads to the most severe performance drop (a decrease of 3.5% in 32-shot), indicating that the MIL aggregation method is critical.
- **Visual and textual exemplars are complementary**: Removing either side individually leads to performance drops, but removing textual exemplars (w/o text em.) has a greater impact in extremely few-shot settings.
- **Bilateral knowledge enhancement is more advantageous in extremely few-shot (2-shot) settings**: The full model improves by 7.2% over CoOp in the 2-shot scenario.

## Highlights & Insights

- **Textbook-style learning paradigm for pathology**: Mimicking the process of pathologists learning from textbooks by introducing both visual exemplars and textual descriptions is a natural and effective way to inject prior knowledge.
- **Two-level, bilateral design**: Comprehensive coverage of two granularities (patch + slide) $\times$ two modalities (vision + text) provides a highly systematic architecture.
- **High interpretability**: Visualizing the matching results of patch/slide exemplars shows that the model learns correct pathological patterns (e.g., vascular invasion, necrosis).
- **Transferable approach**: This paradigm of "introducing domain expert knowledge as prompts" can be generalized to other domains requiring specialized expertise but facing data scarcity.

## Limitations & Future Work

- **By-design reliance on pathology experts for exemplars and descriptions**: Constructing visual and textual priors manually for each task incurs extra costs when extending to new tasks.
- **Sensitivity analysis of exemplar quantity and quality** is under-explored: how do varying numbers or qualities of exemplars affect performance?
- **Limited to classification tasks**: Applicability to tasks like WSI detection or segmentation has not been explored.
- **Limitations of CLIP**: Due to the limited proportion of medical images in CLIP's pre-training data, leverage of stronger pathology VLMs (e.g., PLIP, CONCH) might yield further performance gains.

## Related Work & Insights

- **vs TOP (NeurIPS'23)**: TOP only introduces linguistic descriptions on the textual side and lacks visual-side knowledge; PEMP's bilateral enhancement is more comprehensive.
- **vs CoOp (IJCV'22)**: CoOp only uses learnable textual prompts without domain knowledge; PEMP introduces a hybrid of static and learnable prompts.
- **vs MI-Zero / PLIP**: These works focus on building large-scale pathology VLMs but lack effective adaptation strategies geared towards FSWC.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of two-level, bilateral pathology-knowledge enhanced prompt learning is novel, though core components (attention, contrastive learning) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three clinical tasks, five datasets, complete ablation experiments, and visualization analysis are provided, but comparisons with more VLMs are lacking.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with reasonable motivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ It holds practical clinical value for few-shot pathological image analysis, features strong interpretability, and is suitable for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/universal-to-specific_dynamic_knowledge-guided_multiple_instance_learning_for_fe.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[ECCV 2024\] Unleashing the Power of Prompt-driven Nucleus Instance Segmentation](unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)
- [\[CVPR 2026\] Contrastive Cross-Bag Augmentation for Multiple Instance Learning-based Whole Slide Image Classification](../../CVPR2026/medical_imaging/contrastive_cross-bag_augmentation_for_multiple_instance_learning-based_whole_sl.md)
- [\[ICLR 2026\] ASMIL: Attention-Stabilized Multiple Instance Learning for Whole-Slide Imaging](../../ICLR2026/medical_imaging/asmil_attention-stabilized_multiple_instance_learning_for_whole-slide_imaging.md)

</div>

<!-- RELATED:END -->
