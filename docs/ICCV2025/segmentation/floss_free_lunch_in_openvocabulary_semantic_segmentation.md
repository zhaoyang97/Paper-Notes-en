---
title: >-
  [Paper Note] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][open-vocabulary segmentation] This paper challenges the default practice of averaging 80 templates in open-vocabulary semantic segmentation (OVSS)…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "open-vocabulary segmentation"
  - "template selection"
  - "class-expert"
  - "entropy"
  - "plug-and-play"
  - "training-free"
date: 2026-05-08
content_hash: 86235135581dc867
---

# FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation

**Conference**: ICCV 2025
**arXiv**: [2504.10487](https://arxiv.org/abs/2504.10487)  
**Code**: [https://github.com/yasserben/FLOSS](https://github.com/yasserben/FLOSS)  
**Area**: Segmentation / Open-Vocabulary / Text Prompting
**Keywords**: open-vocabulary segmentation, template selection, class-expert, entropy, plug-and-play, training-free

## TL;DR
This paper challenges the default practice of averaging 80 templates in open-vocabulary semantic segmentation (OVSS), revealing that each class has specific "class-expert" templates that significantly outperform the averaged classifier. It proposes FLOSS, a method that uses prediction entropy to unsupervisedly select expert templates and fuse their predictions, consistently improving existing OVSS methods without any labels or training.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: OVSS methods (e.g., MaskCLIP/NACLIP/CLIP-DINOiser) default to averaging embeddings from CLIP's original 80 ImageNet templates (e.g., "a photo of \<class\>", "a sketch of \<class\>") to construct text classifiers. This practice has been carried over from CLIP's zero-shot classification without systematic investigation at the segmentation level. **Core Finding**: For each class, certain single-template classifiers outperform the 80-template averaged classifier — these are the "class-experts."

### Mechanism

**Goal**: (1) How to identify class-expert templates per class without labels or training? (2) How to effectively fuse predictions from multiple experts into a final segmentation output?

## Method

### Overall Architecture
FLOSS is a fully plug-and-play post-processing method. Given a set of unlabeled images and an existing OVSS model: (1) build a classifier from each individual template and perform segmentation → (2) select the top-N low-entropy templates per class as class-experts using class-level prediction entropy → (3) each class-expert produces a segmentation map → (4) fuse K expert predictions into a final segmentation map via "highest-confidence voting."

### Key Designs
1. **Core Finding — Existence of Class-Experts**: Systematic experiments show that for every class in Cityscapes, several individual templates among the 80 achieve higher IoU than the averaged-template classifier (Figure 1). The expert sets differ across classes — "a photo of a car" may be an expert for *car* but not for *sky*. This indicates that averaging all templates is a suboptimal choice.

2. **Unsupervised Expert Identification via Prediction Entropy**: For each template $\mathcal{T}_m$ and each class $k$, the average softmax entropy is computed over all pixels classified as class $k$ by that template. The top-N templates with the lowest entropy are selected as experts for that class ($N=4$). Low entropy indicates higher classifier confidence, which empirically correlates strongly with higher IoU. No labels are required.

3. **Expert Prediction Fusion**: Each of the K class-experts generates a complete segmentation map. For each pixel, the method checks which experts predict their specialized class at that location (i.e., expert-k predicts the pixel as class $k$). Among qualifying experts, the one with the highest softmax probability is selected as the final prediction. For pixels where no expert predicts its designated class (~2% of pixels), the method falls back to the default averaged classifier $W_{\text{CLIP}}$.

### Loss & Training
Completely training-free and label-free. The only hyperparameter is $N=4$.

## Key Experimental Results

| Model | Method | CS | VOC20 | PC59 | ADE | Stuff | Avg |
|------|------|-----|-------|------|-----|-------|-----|
| CLIP-DINOiser | baseline | 31.3 | 80.8 | 36.0 | 17.5 | 24.6 | 38.0 |
| | +FLOSS | **34.6** | **82.2** | **36.3** | **18.0** | **24.7** | **39.2** |
| NACLIP | baseline | 35.5 | 83.0 | 35.2 | 19.1 | 22.4 | 39.0 |
| | +FLOSS | **37.0** | **83.5** | **35.9** | **19.6** | **22.7** | **39.7** |
| MaskCLIP | baseline | 25.0 | 61.8 | 25.5 | 14.2 | 17.5 | 28.8 |
| | +FLOSS | **25.8** | **61.8** | **26.2** | **14.4** | **17.8** | **29.2** |

- Consistent improvements across all 3 OVSS models × 5 datasets
- Largest gain on Cityscapes with CLIP-DINOiser: +3.3 mIoU
- Cross-domain generalization: experts selected on CS yield +4.9 mIoU on ACDC Fog
- Low-data scenario: a single unlabeled Cityscapes image suffices to surpass the baseline
- Effective on both ViT-B/16 and ViT-L/14 backbones
- ~50% of predicted experts are true experts (validated via a quality metric against GT)

### Ablation Study
- Fusion strategy: "Highest" (highest-confidence voting) > "Average" > "Default"
- Entropy is the most effective unsupervised expert selection metric; Avg. Probability is competitive
- $N=4$ is optimal (selecting 4 from ~80 templates); larger $N$ includes non-experts and degrades performance
- Surpassing the baseline requires only ~50% true experts (verified via oracle experiments)
- Oracle upper bound (GT-guided best expert selection): IoU for *sky* can improve by 30+ points

## Highlights & Insights
- **Novel and Unexpected Insight**: The default practice of averaging 80 templates is challenged — "not all templates are equally useful for every class" is an important finding overlooked by the community
- **Minimal yet Effective Method**: Only the template selection is changed; the visual encoder and model architecture remain entirely untouched
- **Plug-and-Play "Free Lunch"**: Directly applicable on top of any OVSS method, with consistent gains across models and datasets
- **Excellent Low-Data Usability**: Expert selection from a single unlabeled image is sufficient — highly practical for deployment to new domains
- **Cross-Domain Generalization**: Experts selected on CS transfer effectively to out-of-domain datasets such as ACDC, BDD, and Mapillary

## Limitations & Future Work
- Computational overhead scales significantly with the number of classes (inference time increases from 23 ms to 339 ms on ADE with 150 classes)
- Restricted to the pool of 80 ImageNet templates — generating richer or more targeted templates (e.g., via LLMs) may yield further gains
- Improvement margins are smaller on datasets closer to the ImageNet distribution (e.g., VOC20/PC59)
- Non-CLIP backbones (e.g., SigLIP) and more complex OVSS architectures remain unexplored

## Related Work & Insights
- **vs. CorrCLIP**: CorrCLIP repairs CLIP from the visual side (attention scope reconstruction); FLOSS optimizes CLIP from the textual side (template selection) — the two approaches are completely orthogonal and can be combined
- **vs. ProxyCLIP**: ProxyCLIP uses DINO to augment visual features; FLOSS optimizes the text classifier — likewise orthogonal
- **vs. Prompt Engineering**: Traditional prompt engineering generates better class-name descriptions via LLMs; FLOSS selects optimal subsets from existing 80 templates, representing "template selection" rather than "template generation"

## Related Work & Insights
- **Idea Extension**: Class-expert selection can be extended to prompt generation — generating $N$ candidate descriptions per class via LLMs and then selecting the optimal one via entropy, combining prompt engineering with expert selection
- The orthogonality with CorrCLIP implies that **CorrCLIP + FLOSS** can be jointly applied for further gains
- The low-data usability makes FLOSS well-suited for deployment in specialized domains (e.g., medical imaging, remote sensing)

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The finding that "not all templates are equal" is simple yet entirely overlooked by the community; entropy-based expert selection requires no labels
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 standard benchmarks + cross-domain generalization + low-data regime + cross-dataset transfer + ablations over fusion strategies and unsupervised metrics
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figures 1 and 2 are highly intuitive; the logical flow from problem definition → finding → solution → validation is impeccable
- **Value**: ⭐⭐⭐⭐⭐ A genuine "free lunch" — zero training, zero labels, consistent improvements, compatible with any OVSS method

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
