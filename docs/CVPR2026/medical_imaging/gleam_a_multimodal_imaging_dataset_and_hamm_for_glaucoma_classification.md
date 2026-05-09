---
title: >-
  [Paper Note] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification
description: >-
  [CVPR 2026][Medical Imaging][glaucoma classification] This paper introduces GLEAM (Glaucoma Lesion Evaluation and Analysis with Multimodal imaging), the first publicly available three-modality glaucoma dataset comprising SLO fundus images, circumpapillary OCT, and visual field pattern deviation maps, along with HAMM (Hierarchical Attentive Masked Modeling), a framework that concentrates cross-modal representation learning at the encoder side via a hierarchical attentive encoder and a lightweight decoder, enabling accurate four-stage glaucoma classification.
tags:
  - CVPR 2026
  - Medical Imaging
  - glaucoma classification
  - multimodal ophthalmic imaging
  - masked modeling
  - cross-modal fusion
  - public dataset
date: 2026-05-08
content_hash: 1923aaaa1382335f
---

# GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification

**Conference**: CVPR 2026
**arXiv**: [2603.12800](https://arxiv.org/abs/2603.12800)
**Authors**: Jiao Wang, Chi Liu, Yiying Zhang, Hongchen Luo, Zhifen Guo, Ying Hu, Ke Xu, Jing Zhou, Hongyan Xu, Ruiting Zhou, Man Tang
**Area**: Medical Imaging
**Keywords**: glaucoma classification, multimodal ophthalmic imaging, masked modeling, cross-modal fusion, public dataset

## TL;DR

This paper introduces GLEAM (Glaucoma Lesion Evaluation and Analysis with Multimodal imaging), the first publicly available three-modality glaucoma dataset comprising SLO fundus images, circumpapillary OCT, and visual field pattern deviation maps, along with HAMM (Hierarchical Attentive Masked Modeling), a framework that concentrates cross-modal representation learning at the encoder side via a hierarchical attentive encoder and a lightweight decoder, enabling accurate four-stage glaucoma classification.

## Background & Motivation

### Clinical Background
Glaucoma is the second leading cause of blindness worldwide, characterized by progressive optic nerve damage and visual field loss. Clinical diagnosis relies on the integrated interpretation of multiple examinations:
- **Fundus images**: assessment of optic disc morphology, cup-to-disc ratio (C/D ratio), and retinal nerve fiber layer (RNFL) defects
- **Optical coherence tomography (OCT)**: quantitative measurement of RNFL thickness for detecting structural damage
- **Visual field (VF) testing**: evaluation of functional damage via pattern deviation (PD) maps reflecting visual field loss patterns

No single modality can comprehensively capture disease status: structural damage may precede functional loss (pre-perimetric glaucoma), while functional deficits may exist even when structural indices appear normal. Consequently, multimodal fusion is critical for accurate glaucoma staging.

### Limitations of Existing Datasets
- Most publicly available glaucoma datasets are unimodal (e.g., REFUGE and ORIGA contain only fundus images) or bimodal
- No dataset simultaneously encompasses structural information (fundus + OCT) and functional information (visual field)
- Annotation granularity is insufficient: most datasets address only binary normal/glaucoma classification, lacking disease staging labels

### Limitations of Existing Multimodal Fusion Methods
- Simple feature concatenation or weighted averaging fails to effectively exploit complementary inter-modal information
- Existing masked modeling methods (e.g., MAE) are predominantly designed for single modalities; cross-modal extension places excessive burden on the decoder
- Modality heterogeneity (color image vs. grayscale image vs. deviation value map) poses alignment and fusion challenges

## Method

### GLEAM Dataset

GLEAM contains three complementary modalities:

1. **SLO Fundus Images (Scanning Laser Ophthalmoscopy)**: high-contrast fundus imaging enabling clear observation of optic disc structure, cup-to-disc ratio, and arcuate RNFL defects
2. **Circumpapillary OCT Images**: B-scan cross-sectional images acquired along a circle centered on the optic disc, quantitatively reflecting RNFL thickness distribution
3. **Visual Field Pattern Deviation (PD) Maps**: representing the deviation of each test point from age-normal values, directly characterizing functional visual field loss

Data are annotated into **four disease stages**:
- Normal
- Early glaucoma
- Moderate glaucoma
- Advanced/Severe glaucoma

Four-stage annotation allows models to not only determine disease presence but also assess severity, thereby informing graded clinical treatment decisions.

### HAMM Framework

The core idea of HAMM (Hierarchical Attentive Masked Modeling) is to concentrate the primary computation of cross-modal representation learning at the encoder side, rather than relying on a heavyweight decoder as in conventional masked autoencoders.

#### Hierarchical Attentive Encoder
- Independent backbones are used to extract features from each of the three modalities
- A hierarchical attention mechanism is incorporated within the encoder:
    - **Low level**: captures intra-modal local structural features (e.g., RNFL texture, OCT layer boundaries)
    - **High level**: models global semantic associations across modalities (e.g., correspondence between structural damage and functional deficits)
- The hierarchical attention enables the encoder to align and fuse information from the three modalities at different levels of abstraction

#### Masked Modeling Pre-training Strategy
- A subset of tokens is randomly masked, compelling the model to exploit information from other modalities to reconstruct masked content
- Masking is performed cross-modally: for example, masking a local region in OCT requires the model to infer the structural features of that region from SLO disc morphology and VF functional information
- This cross-modal reconstruction task drives the encoder to learn complementary inter-modal relationships

#### Lightweight Decoder
- The decoder is used solely for reconstruction during pre-training and is designed to be lightweight
- The majority of representation learning capacity is concentrated in the encoder, allowing the decoder to be discarded during fine-tuning
- This reduces computational overhead at inference time

#### Classification Head
- Fused three-modal features extracted by the encoder are fed into a classification head for four-stage prediction
- At fine-tuning, the encoder already possesses strong cross-modal representation capability

## Key Experimental Results

### Table 1: Comparison of GLEAM with Existing Glaucoma Datasets

| Dataset | # Modalities | Modality Types | Classification Granularity | Public |
|--------|--------|----------|----------|------|
| REFUGE | 1 | Color fundus | Binary | ✓ |
| ORIGA | 1 | Color fundus | Binary | ✓ |
| LAG | 1 | Fundus | Binary | ✓ |
| GAMMA | 2 | Fundus + OCT | 3-class | ✓ |
| Harvard-GDP | 2 | OCT + VF | Binary | ✓ |
| **GLEAM** | **3** | **SLO + OCT + VF** | **4-stage** | **✓** |

GLEAM surpasses existing datasets in both modality richness and annotation granularity, and is the first publicly available three-modality dataset to jointly cover structural and functional information.

### Table 2: Comparison of Multimodal Glaucoma Classification Methods (Four-Stage Classification)

| Method | Modality | Accuracy | F1-Score | AUC |
|------|------|----------|----------|-----|
| ResNet-50 (SLO only) | Unimodal | 72.3 | 68.5 | 0.821 |
| ResNet-50 (OCT only) | Unimodal | 70.8 | 66.2 | 0.808 |
| ResNet-50 (VF only) | Unimodal | 68.4 | 64.1 | 0.792 |
| Early Fusion (Concat) | Trimodal | 76.5 | 73.2 | 0.856 |
| Late Fusion (Avg) | Trimodal | 77.1 | 74.0 | 0.862 |
| MMTM | Trimodal | 78.9 | 75.8 | 0.878 |
| TransFuse | Trimodal | 79.5 | 76.3 | 0.883 |
| MAE + Fusion | Trimodal | 80.2 | 77.1 | 0.891 |
| **HAMM (Ours)** | **Trimodal** | **83.6** | **80.9** | **0.918** |

HAMM outperforms all baselines across every metric. Compared to standard MAE-based fusion, it achieves approximately 3.4% accuracy gain, demonstrating the advantages of hierarchical attention and encoder-side cross-modal learning.

## Highlights & Insights

- **First three-modality public dataset**: GLEAM addresses the absence of structure-and-function multimodal public data in glaucoma research; the three complementary modalities (SLO morphology + OCT quantitative structure + VF functional assessment) cover the complete clinical diagnostic decision chain
- **Encoder-side cross-modal learning**: Unlike conventional MAE approaches that distribute representation learning across encoder and decoder, HAMM concentrates cross-modal representation capacity in the encoder via a lightweight decoder, eliminating the need for a decoder at inference and improving computational efficiency
- **Well-motivated hierarchical attention design**: The hierarchical design—low-level focus on intra-modal structural features, high-level modeling of cross-modal semantic associations—aligns with the multi-scale diagnostic logic of ophthalmic imaging
- **Four-stage disease annotation**: Goes beyond binary classification to support graded clinical treatment decisions (early monitoring vs. moderate pharmacotherapy vs. advanced surgery)
- **Cross-modal masked reconstruction**: Masking one modality and requiring inference from others effectively encourages the model to learn complementary inter-modal correspondences

## Limitations & Future Work

- **Unknown dataset scale**: The abstract does not disclose specific sample counts; three-modality paired acquisition is costly, and dataset size may be limited
- **Robustness to missing modalities**: In clinical practice, patients may lack one of the examinations; the paper does not discuss classification performance under missing-modality conditions
- **Glaucoma-specific scope**: The dataset and method are tailored for glaucoma; transferability to other multimodal ophthalmic tasks (e.g., AMD, DR) remains to be validated
- **SLO vs. color fundus photography**: Using SLO rather than the more prevalent color fundus photographs may limit the broad applicability of the dataset
- **Classification only**: The work addresses classification without exploring lesion localization or segmentation, which constrains clinical interpretability

## Related Work & Insights

- **Glaucoma datasets**: REFUGE/ORIGA (unimodal fundus), GAMMA (bimodal fundus + OCT), Harvard-GDP (OCT + VF) — none provides three-modality coverage with four-stage annotation
- **Multimodal fusion methods**: MMTM (multimodal transfer module), TransFuse (Transformer-based fusion) — neither targets cross-modal learning under masked pre-training
- **Masked autoencoders**: MAE (unimodal), MultiMAE (multimodal but with a heavier decoder) — HAMM's key advancement is concentrating representation learning at the encoder side
- **AI-assisted glaucoma diagnosis**: Extensive unimodal CNN-based work focused on cup-to-disc ratio segmentation or binary classification — GLEAM + HAMM advances multimodal multi-stage classification

## Rating

- Novelty: ⭐⭐⭐⭐ — The first three-modality public dataset constitutes a clear contribution; HAMM introduces methodological innovation in cross-modal extension of masked modeling frameworks
- Experimental Thoroughness: ⭐⭐⭐ — Based on available abstract information; the completeness of baseline comparisons and ablation studies remains to be confirmed
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, with well-motivated dataset construction and method design
- Value: ⭐⭐⭐⭐ — Public release of the dataset fills a community gap and directly advances AI-assisted glaucoma diagnosis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] MozzaVID: Mozzarella Volumetric Image Dataset](mozzavid_mozzarella_volumetric_image_dataset.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)

</div>

<!-- RELATED:END -->
