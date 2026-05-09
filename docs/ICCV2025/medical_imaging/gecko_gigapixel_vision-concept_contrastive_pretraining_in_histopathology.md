---
title: >-
  [Paper Note] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology
description: >-
  [ICCV 2025][Medical Imaging][WSI pretraining] GECKO is proposed as a WSI-level MIL aggregator pretraining method that requires no additional clinical data modalities. By automatically extracting interpretable concept priors from H&E WSIs and aligning them with deep features via contrastive learning, GECKO surpasses existing unimodal and multimodal pretraining methods on five classification tasks while providing pathologist-interpretable WSI-level descriptions.
tags:
  - ICCV 2025
  - Medical Imaging
  - WSI pretraining
  - concept prior
  - contrastive learning
  - multiple instance learning
  - interpretability
date: 2026-05-08
content_hash: 2f0b12aa9198ab6e
---

# GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology

**Conference**: ICCV 2025  
**arXiv**: [2504.01009](https://arxiv.org/abs/2504.01009)  
**Code**: [github.com/bmi-imaginelab/GECKO](https://github.com/bmi-imaginelab/GECKO)  
**Area**: Medical Imaging / Computational Pathology  
**Keywords**: WSI pretraining, concept prior, contrastive learning, multiple instance learning, interpretability

## TL;DR

GECKO is proposed as a WSI-level MIL aggregator pretraining method that requires no additional clinical data modalities. By automatically extracting interpretable concept priors from H&E WSIs and aligning them with deep features via contrastive learning, GECKO surpasses existing unimodal and multimodal pretraining methods on five classification tasks while providing pathologist-interpretable WSI-level descriptions.

## Background & Motivation

Foundation models in pathology are advancing rapidly; however, because WSIs are gigapixel-scale, most existing work focuses on patch-level representation learning. Obtaining WSI-level embeddings requires MIL aggregators, whose training typically relies on supervised signals.

Existing WSI-level pretraining methods face two key challenges:

**Dependence on additional modalities**: Unimodal pretraining (WSI data only) is prone to overfitting staining artifacts. TANGLE requires paired transcriptomic data; MEDELEINE requires slides of different staining types — all of which are costly to acquire, limited in dataset scale, and difficult to standardize.

**Lack of interpretability**: WSI embeddings produced by pretraining are inherently uninterpretable black boxes. They can only provide patch attention heatmaps indicating salient regions, without revealing the key pathological concepts driving predictions.

**Core Problem**: Can a MIL aggregator be effectively pretrained using WSI data alone, while yielding pathologist-interpretable WSI-level embeddings?

## Method

### Overall Architecture

GECKO pretrains a **dual-branch MIL network**:
- **Deep encoding branch**: aggregates patch-level deep features into a WSI-level deep embedding $F_{wsi}$
- **Concept encoding branch**: aggregates concept priors into a WSI-level concept embedding $M_{wsi}$ (preserving interpretability)
- A contrastive learning objective aligns the WSI-level outputs of both branches

### Key Designs

1. **Concept Prior Extraction**:

    - An LLM (GPT-4) is used to generate visually discriminative pathological concept text descriptions for each class of each downstream task (10 most distinctive concepts per class).
    - A pretrained VLM (CONCH) text encoder encodes the concepts into $T \in \mathbb{R}^{C \times D}$.
    - The VLM visual encoder encodes the $N$ patches of a WSI into $F \in \mathbb{R}^{N \times D}$.
    - A cosine similarity matrix $M \in \mathbb{R}^{N \times C}$ between patches and concepts is computed — this constitutes the concept prior.
    - Each element quantifies the activation of a given patch toward a specific concept, yielding natural interpretability.
    - The entire process is fully automated, requiring no manual annotation or additional clinical assays.

2. **WSI-level Deep Encoding Branch**: Based on the ABMIL architecture, patch features are projected via $H(\cdot)$ and aggregated with attention weights from $A^p(\cdot)$:

$$F_{wsi} = \sum_{i=1}^N \alpha_i \cdot \tilde{f}_i$$

where $\alpha_i$ are learnable patch attention weights.

3. **WSI-level Concept Encoding Branch**:

    - Top-K salient patches are selected using the attention scores $\alpha_i$ from the deep encoding branch (via a differentiable Perturbed Top-K operation).
    - The corresponding concept prior sub-matrix $\tilde{M} \in \mathbb{R}^{K \times C}$ is extracted.
    - An MLP-Mixer contextualizes spatial and concept information.
    - A gated attention network $G(\cdot)$ computes concept attention weights $\beta_j$ (sigmoid activation).
    - **Core constraint**: The concept prior undergoes only linear scaling $\hat{M}_{ij} = \beta_j \times \tilde{M}_{ij}$, preserving interpretability.
    - Average pooling yields the WSI-level concept embedding $M_{wsi} = \frac{1}{K}\sum_{i=1}^K \hat{M}_i$.

### Loss & Training

- **Contrastive loss**: A symmetric CLIP loss aligns $F_{wsi}$ and $M_{wsi}$:

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{CL}(F_{wsi}, M_{wsi}) + \mathcal{L}_{CL}(M_{wsi}, F_{wsi}))$$

- **False negative elimination**: A keep ratio $r_{keep}=0.7$ excludes highly similar WSI pairs to avoid erroneous contrastive signals.
- Pretraining: 50 epochs, learning rate 1e-4, 5-epoch warmup + cosine decay.
- Batch size = 64, $K = 10$ (Top-K salient patches).
- Patch features extracted by CONCH (448×448 @ 20×); 10 concepts selected per class.

**Inference modes**:
- **Unsupervised prediction**: Exploiting the interpretability of concept embeddings, a WSI is assigned to the class with the highest concept activation: $P(l) = \frac{\sum_{j \in I_l} M_{wsi,j}}{\sum_{k \in I} M_{wsi,k}}$
- **Supervised prediction**: A linear classifier is trained on $F_{wsi}$ and/or $M_{wsi}$.

## Key Experimental Results

### Main Results

**Unsupervised classification (AUC, zero labels)**:

| Method | Interpretable | LUAD vs LUSC | EBV+MSI vs Others | MSI vs Others | HER2 (3-class) |
|--------|--------------|-------------|-------------------|---------------|----------------|
| MI-Zero | Partial | 96.6 | 61.9 | 42.3 | 32.2 |
| ConcepPath-Zero | ✗ | 91.0 | 74.2 | 73.4 | 37.5 |
| **GECKO-Zero** | **✓** | 95.0 | **83.4** | **77.1** | **60.6** |

Without any WSI-level labels, GECKO-Zero substantially outperforms existing unsupervised methods on most tasks.

**Fully supervised classification (AUC, linear probing)**:

| Method | Embedding | LUAD vs LUSC | EBV+MSI vs Others | MSI vs Others |
|--------|-----------|-------------|-------------------|---------------|
| Intra (WSI only) | deep | 97.5 | 83.5 | 83.9 |
| GECKO (WSI only) | ensemble | **97.6** | **86.4** | **86.5** |
| TANGLE (WSI+Gene) | deep | 97.9 | 85.4 | 86.6 |
| GECKO (WSI+Gene) | ensemble | **97.9** | **87.1** | **89.4** |

Using WSI data alone, GECKO is competitive with TANGLE (WSI+Gene); incorporating gene data yields further improvements.

### Ablation Study

**Comparison with other WSI encoding methods (few-shot, k=10)**:

| Method | LUAD vs LUSC | EBV+MSI vs Others |
|--------|-------------|-------------------|
| PANTHER | 91.2 | 78.5 |
| Giga-SSL (H-Optimus) | 92.8 | 77.5 |
| GECKO (ensemble, WSI only) | **96.4** | **82.1** |
| TITAN (multimodal) | 97.5 | 78.7 |
| GECKO (ensemble, WSI+Gene) | 97.0 | **84.4** |

GECKO outperforms TITAN by 6.7% on the EBV+MSI task (k=10), despite being pretrained on ~200 WSIs versus TITAN's 100K+ paired samples.

**Concept identification accuracy**:

| Task | j=1 (unsupervised) | j=1 (fully supervised) |
|------|-------------------|------------------------|
| LUAD vs LUSC | 81.4% | 99.9% |
| MSI vs Others | 54.0% | 83.3% |

The pretrained model identifies the pathological concepts driving predictions in WSIs with high accuracy.

### Key Findings

- Concept priors provide task-specific discriminative signals, mitigating the staining artifact overfitting problem of unimodal pretraining.
- The linear aggregation design of the concept encoding branch offers a mathematical guarantee of interpretability.
- False negative elimination is critical for contrastive learning in low-dimensional (C=20/30) concept spaces.
- GECKO with WSI data alone already surpasses TANGLE (which requires gene data) on multiple tasks.
- Concept embeddings are directly usable in unsupervised settings for clinical hypothesis testing and biomarker discovery.

## Highlights & Insights

- **Novelty**: The first effective WSI-level pretraining scheme requiring no additional clinical modalities, while simultaneously providing interpretable embeddings.
- **Automated concept mining**: LLM-generated concepts combined with VLM-computed activations form a fully annotation-free pipeline.
- **Dual-embedding design**: Deep embeddings ensure discriminability; concept embeddings ensure interpretability; their ensemble captures the benefits of both.
- **Unsupervised clinical utility**: Under GECKO-Zero mode, pathologists can directly inspect and correct concept-level predictions.
- **Modality flexibility**: GECKO seamlessly integrates additional modalities (e.g., gene expression) — not dependent on them, but benefiting when available.

## Limitations & Future Work

- The concept set requires task-specific priors (concepts must be defined per task); pan-cancer universal pretraining would demand a much larger concept vocabulary.
- Evaluation is limited to TCGA datasets, lacking external independent validation cohorts.
- The quality of concept priors is constrained by the degree of pathology-domain alignment of the underlying VLM (CONCH).
- The Top-K=10 setting may not generalize to all WSIs, as critical regions in some slides may be more broadly distributed.
- Integration with more recent pathology VLMs (e.g., successors to CONCH v1.5) has not been explored.

## Related Work & Insights

- TANGLE pioneered WSI-level multimodal contrastive pretraining but relies on gene expression data.
- SI-MIL's self-interpretable MIL architecture inspired the linear aggregation design of the concept encoding branch.
- ConcepPath demonstrated the feasibility of concept-level pathological analysis; GECKO extends this to the pretraining paradigm.
- The CLIP contrastive objective is adapted here within a Vision-Concept Model (VCM) framework.
- TITAN uses large-scale pathology reports and synthetic captions for pretraining, offering a complementary direction to GECKO.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First concept-prior-driven WSI pretraining approach, balancing performance and interpretability.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 5 tasks in unsupervised, supervised, and few-shot settings, with comparisons to diverse baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Method figures and concept explanations are clear, though some equations are slightly dense in presentation.
- **Value**: ⭐⭐⭐⭐⭐ High clinical utility — interpretability is a critical bottleneck for deploying pathology AI, and GECKO directly addresses it.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-Supervised Learning](an_openmind_for_3d_medical_vision_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
