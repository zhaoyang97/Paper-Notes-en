---
title: >-
  [Paper Note] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders
description: >-
  [NeurIPS 2025][Medical Imaging][dermatology dataset] This work introduces DermaCon-IN—the first densely annotated dermatological image dataset predominantly featuring Indian skin tones (5,450 images / 3…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "dermatology dataset"
  - "Indian skin tone"
  - "concept bottleneck model"
  - "hierarchical classification"
  - "explainable AI"
date: 2026-05-08
content_hash: 3fe3cce7c2c964a9
---

# DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders

**Conference**: NeurIPS 2025
**arXiv**: [2506.06099](https://arxiv.org/abs/2506.06099)  
**Code**: [GitHub](https://github.com/) / [Harvard Dataverse](https://dataverse.harvard.edu/)  
**Area**: Medical Imaging
**Keywords**: dermatology dataset, Indian skin tone, concept bottleneck model, hierarchical classification, explainable AI

## TL;DR
This work introduces DermaCon-IN—the first densely annotated dermatological image dataset predominantly featuring Indian skin tones (5,450 images / 3,002 patients / 245 diagnoses)—providing three-level hierarchical diagnostic labels, 47 lesion descriptors, and 49 anatomical site annotations, with benchmark evaluations using CNN, ViT, and concept bottleneck model architectures.

## Background & Motivation

**Background**: Skin diseases constitute the fourth largest non-fatal disease burden globally. AI-assisted diagnosis is regarded as a promising approach to address the shortage of dermatological resources; however, existing models are predominantly trained and evaluated on datasets collected from Europe and North America.

**Limitations of Prior Work**:
   - Severe dataset bias: ISIC, HAM10000, and similar datasets focus on neoplastic conditions such as melanoma, neglecting tropical high-prevalence diseases such as fungal infections and scabies.
   - Underrepresentation of skin tones: Over 75% of Fitzpatrick17k consists of Type I–III (lighter skin), with a critical lack of darker skin tones (Type IV–VI).
   - One-dimensional annotations: Most datasets provide only diagnostic labels, lacking anatomical site information and lesion morphology descriptors.
   - Data source bias: SD-198 and Fitzpatrick17k are derived from teaching atlases rather than prospective clinical collection.

**Key Challenge**: The geographic, skin-tone, and disease-spectrum biases in existing datasets cause AI model accuracy to drop by 30–40% for darker-skinned populations (as demonstrated on the DDI dataset), precluding equitable service to global populations.

**Key Insight**: Prospective collection from Indian outpatient clinics to construct a dataset with broad disease coverage, representative skin tones, and multi-dimensional annotations.

**Core Idea**: To provide the first dermatological dataset predominantly featuring South Asian skin tones, incorporating a triple annotation scheme covering diagnostic hierarchy, anatomical sites, and lesion descriptors simultaneously.

## Method

### Overall Architecture
Dataset construction follows a collect → annotate → quality control → benchmark evaluation pipeline. Clinical images were collected from three tertiary hospitals in South India; four board-certified dermatologists applied the Rook classification system to produce three-level hierarchical labels and concept annotations. Final benchmarking was conducted using multiple architectures for classification and explainability evaluation.

### Key Designs

1. **Three-Level Hierarchical Diagnostic Labels**

    - Function: Constructs an 8 main-class → 19 sub-class → 245 specific disease hierarchy following the Rook Textbook of Dermatology.
    - Mechanism: Main classes are organized by etiology (infectious, inflammatory, pigmentary, keratotic, etc.); sub-classes handle mixed or co-existing conditions (e.g., inflammatory + fungal infection); specific labels correspond to ICD-11 codes.
    - Design Motivation: Reflects the clinical diagnostic workflow (broad category first, then refinement) and supports both coarse-grained and fine-grained modeling. The disease label distribution follows a long-tail pattern (log-normal exponent 1.8), consistent with real-world outpatient frequencies.

2. **Dual-Dimension Concept Annotations (96 concepts)**

    - **47 lesion descriptors**: scaling, erythema, vesicles, hyperpigmentation, etc., following standard clinical dermatological terminology.
    - **49 anatomical sites**: scalp, palms, soles, trunk, etc., retaining full anatomical context from the images.
    - Design Motivation: Dermatological diagnosis relies on combined reasoning from lesion morphology and anatomical location (e.g., scalp scaling → psoriasis vs. pedal scaling → tinea pedis). Independently annotating both concept types enables interpretability research with concept bottleneck models.
    - Pearson correlation analysis validates the clinical plausibility of concept–disease associations (e.g., vitiligo ↔ pigmentary disorders, $r = +0.71$).

3. **Concept Bottleneck Model (CBM) Benchmark**

    - Function: Constructs a CBM on a Swin-B backbone, performing interpretable classification through a concept layer.
    - Architecture: Image → Swin Encoder → concept logits $c^\ell \in \mathbb{R}^{B+D}$ → sigmoid → concept vector $c$ → classifier → diagnosis.
    - Two hierarchical CBM designs are explored:
        - Type 1 (cascaded): concepts → sub-class prediction → main-class prediction, enforcing classification consistency.
        - Type 2 (parallel): concepts independently predict sub-classes and main classes simultaneously via multi-task learning regularization.
    - Type 2 outperforms Type 1 across all metrics.

### Loss & Training
- The concept layer is supervised with BCE loss for binary classification of 96 concepts.
- The classification layer uses cross-entropy with weighted sampling to address class imbalance.
- Models are pretrained on ImageNet-22k; inputs are resized to 512×512; subject-wise stratified 80:20 splits are applied.

## Key Experimental Results

### Main Results: 8-Class Main Category Classification

| Model | Pretrain | Accuracy | Balanced Acc. | F1 Score |
|-------|----------|----------|--------------|----------|
| ResNet50 | — | 47.45 | 23.93 | 46.43 |
| ResNet50 | ImageNet | 64.31 | 38.77 | 63.31 |
| DenseNet121 | ImageNet | 65.20 | 37.31 | 64.37 |
| ViT-B/16-384 | ImageNet | 66.95 | 35.78 | 65.78 |
| **Swin-B/4W12-384** | **ImageNet** | **70.41** | **45.06** | **69.69** |

The Swin Transformer achieves the best performance consistently across all metrics.

### Ablation Study: CBM Configurations

| Configuration | Concepts | Accuracy | Macro AUC |
|---------------|----------|----------|-----------|
| Direct MC classification (no CBM) | — | 70.41 | 78.51 |
| CBM-D (descriptors only) | 47 | 68.57 | 85.18 |
| CBM-B (anatomical sites only) | 49 | 68.38 | 84.96 |
| CBM full concepts | 96 | 68.12 | 82.78 |
| Type 2 hierarchical CBM – MC | 96 | **69.90** | 77.01 |

### Key Findings
- Introducing the concept bottleneck results in a modest accuracy decrease (~2%) but a substantial improvement in Macro AUC (78.51 → 85.18), indicating that the concept layer provides better inter-class discriminability.
- Single-stream concepts (descriptors only or anatomical sites only) yield comparable performance; however, combining both streams produces a competitive suppression effect—the model tends to activate one concept group while inhibiting the other.
- Grad-CAM analysis confirms that concept activations localize to semantically correct anatomical regions.
- Concept-weight alignment (Spearman correlation) is satisfactory ($p < 0.05$) for pigmentary and keratotic disorder classes, but poor for neoplastic classes, likely due to insufficient sample sizes.

## Highlights & Insights
- **First densely annotated dataset for Indian skin tones**: covers Fitzpatrick IV–VI and MST 4–9 tones, addressing a critical gap in global dermatological AI fairness.
- **Elegant triple annotation scheme**: the combination of diagnostic hierarchy, lesion descriptors, and anatomical sites systematically formalizes the clinical reasoning pathway in dermatology (morphology + location → diagnosis) at the dataset level for the first time.
- The discovery of **concept competition** (suppression of one concept group in dual-stream CBMs) reveals a representational bottleneck in multi-concept learning, warranting further investigation.
- An inter-annotator Cohen's Kappa of 0.84 ensures high data quality.

## Limitations & Future Work
- Facial images required privacy-preserving de-identification (eye occlusion or cropping), which compromises modeling capability for facial dermatoses.
- Rare diseases in the long-tail distribution have very few samples, necessitating few-shot or long-tail learning strategies.
- The absence of pixel-level segmentation annotations precludes support for lesion localization and segmentation tasks.
- Data are sourced exclusively from Karnataka in South India, without coverage of North India or other Southeast Asian regions.
- The concept competition issue in CBMs requires novel regularization or attention mechanisms to resolve.

## Related Work & Insights
- **vs. Fitzpatrick17k**: The latter is derived from teaching atlases rather than clinical settings, comprises 75% lighter skin tones, and lacks fungal and viral infections; DermaCon-IN is prospectively collected in clinical settings and covers infectious diseases.
- **vs. SkinCon**: SkinCon retrospectively adds descriptors to an existing dataset; DermaCon-IN collects lesion and anatomical site concepts simultaneously at the point of acquisition.
- **vs. DDI**: DDI contains only 656 images across 78 classes; DermaCon-IN is substantially larger (5,450 images / 245 classes) with denser annotations.
- **vs. PASSION**: PASSION focuses on 4 diseases in African pediatric populations; DermaCon-IN covers the full spectrum of 245 adult dermatological conditions.

## Rating
- Novelty: ⭐⭐⭐⭐ Fills the gap in Indian skin tone data and introduces a novel triple annotation scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-architecture comparisons, CBM exploration, and Grad-CAM qualitative analysis provide comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; related work comparison tables are detailed.
- Value: ⭐⭐⭐⭐⭐ The dataset contribution carries significant implications for fairness research and global applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)

</div>

<!-- RELATED:END -->
