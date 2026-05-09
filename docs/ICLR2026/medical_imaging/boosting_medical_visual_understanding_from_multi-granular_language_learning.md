---
title: >-
  [Paper Note] Boosting Medical Visual Understanding From Multi-Granular Language Learning
description: >-
  [ICLR 2026][Medical Imaging][Medical image pre-training] This paper proposes Multi-Granular Language Learning (MGLL), a plug-and-play contrastive learning framework that jointly optimizes a soft CLIP loss, a point-wise loss, and a smooth KL divergence to align medical images with multi-label, multi-granular text descriptions. MGLL consistently surpasses state-of-the-art methods on fundus and X-ray datasets, and when used as a visual encoder for multimodal large language models, improves diagnostic accuracy by up to 34.1%.
tags:
  - ICLR 2026
  - Medical Imaging
  - Medical image pre-training
  - multi-label contrastive learning
  - multi-granular alignment
  - CLIP improvement
  - vision-language pre-training
date: 2026-05-08
content_hash: 310d989aa73d219f
---

# Boosting Medical Visual Understanding From Multi-Granular Language Learning

**Conference**: ICLR 2026
**arXiv**: [2511.15943](https://arxiv.org/abs/2511.15943)
**Code**: [https://github.com/HUANGLIZI/MGLL](https://github.com/HUANGLIZI/MGLL)
**Area**: Medical Imaging / Multimodal VLM
**Keywords**: Medical image pre-training, multi-label contrastive learning, multi-granular alignment, CLIP improvement, vision-language pre-training

## TL;DR
This paper proposes Multi-Granular Language Learning (MGLL), a plug-and-play contrastive learning framework that jointly optimizes a soft CLIP loss, a point-wise loss, and a smooth KL divergence to align medical images with multi-label, multi-granular text descriptions. MGLL consistently surpasses state-of-the-art methods on fundus and X-ray datasets, and when used as a visual encoder for multimodal large language models, improves diagnostic accuracy by up to 34.1%.

## Background & Motivation

**Background**: Contrastive learning approaches such as CLIP have achieved remarkable success in general vision by learning cross-modal aligned representations from image-text pairs. Many medical visual foundation models have adopted CLIP-style pre-training.

**Limitations of Prior Work**: Standard CLIP relies on a single-label, single-granularity image-text pairing strategy; however, medical images are inherently **multi-label** and **multi-granular**. For example, a single fundus image may simultaneously present diabetic macular edema and diabetic retinopathy (multi-label), each of which can be further described at coarse granularity (disease category) and fine granularity (severity, clinical description). Existing multi-label contrastive methods focus on instance-label associations but neglect cross-granularity semantics.

**Key Challenge**: Medical images encode more complex and hierarchical information than natural images, yet data are scarcer due to privacy constraints and annotation costs. Single-granularity, single-label supervision wastes rich hierarchical annotation information, while naively mixing multi-granular information in a single encoding causes feature interference across semantic levels.

**Goal**: To achieve, within a unified framework, both multi-label alignment (one image corresponding to multiple labels) and cross-granularity alignment (consistency across annotations at different levels).

**Key Insight**: Construct a multi-granular text description dataset and design three complementary loss functions to separately optimize multi-label alignment and cross-granularity consistency.

**Core Idea**: Jointly optimize a soft CLIP loss for multi-label soft alignment, a point-wise loss for fine-grained pairwise alignment, and a smooth KL divergence for cross-granularity feature consistency, achieving comprehensive vision-language alignment for medical images.

## Method

### Overall Architecture
MGLL consists of an image encoder (ViT-L/14) and a text encoder (BiomedicalBERT). The inputs are medical images paired with multi-granular text descriptions (e.g., disease category, clinical interpretation, examination description). MGLL introduces no additional granularity-sensitive encoders, incurs zero extra computational cost, and can be plugged into any vision-language model.

### Key Designs

1. **Soft CLIP Loss $\mathcal{L}_{\text{sCLIP}}$**:

    - **Function**: Extends the hard single-label matching of standard CLIP to multi-label soft alignment.
    - **Mechanism**: Allows image feature $V_i$ to align simultaneously with multiple text labels $\{T_{i1}, T_{i2}, ..., T_{iM_i}\}$. The weight $w_{ik}$ for each image-text pair is derived by normalizing a co-occurrence matrix: $w_{ik} = \frac{\text{cooccurrence}(V_i, T_{ik})}{\sum_k \text{cooccurrence}(V_i, T_{ik})}$. The optimization objective is equivalent to driving image features toward the weighted centroid of their associated text features.
    - **Design Motivation**: CLIP forces each image to align with a single label, producing biased representations in multi-label settings. The soft loss naturally handles one-to-many mappings through soft weights.

2. **Point-wise Loss $\mathcal{L}_P$**:

    - **Function**: Optimizes point-level image-text alignment within a given granularity level.
    - **Mechanism**: Uses binary cross-entropy as the loss, where $y_{ij} \in \{0, 1\}$ indicates whether image $V_i$ and text $T_j$ form a valid match. A sigmoid activation normalizes the similarity score into a probability: $\mathcal{L}_P = -\sum_{i,j} \frac{y_{ij} \log \sigma(x_{ij}) + (1-y_{ij}) \log(1-\sigma(x_{ij}))}{N}$
    - **Design Motivation**: While the soft CLIP loss focuses on soft assignment among positive samples, the point-wise loss additionally and explicitly suppresses similarity for negative pairs (minimizing $\sigma(x_{ij})$ when $y_{ij}=0$), complementing the former to enhance multi-label discriminability.

3. **Smooth KL Divergence Loss $\mathcal{L}_{\text{sKL}}$**:

    - **Function**: Ensures that text features from different granularity levels are aligned into a unified feature space.
    - **Mechanism**: Given prediction distributions $\{P_1, ..., P_m\}$ for $m$ granularity levels, computes the mean distribution $M = \frac{1}{m}\sum_i P_i$, then minimizes the KL divergence from each granularity distribution to the mean: $\mathcal{L}_{\text{sKL}} = \sum_{i=1}^m D_{\text{KL}}(P_i \| M)$
    - **Design Motivation**: Without a cross-granularity consistency constraint, features from different granularities scatter across disjoint subspaces, preventing cross-granularity generalization. Minimizing KL divergence to the mean distribution forces all granularity representations to converge ($P_1 = P_2 = ... = P_m = M$).

### Loss & Training
The final loss is a weighted sum of the three terms: $\mathcal{L}_{\text{MGLL}} = 0.5 \cdot \mathcal{L}_{\text{sCLIP}} + 1.0 \cdot \mathcal{L}_P + 1.0 \cdot \mathcal{L}_{\text{sKL}}$

### Large-Scale Multi-Granular Dataset Construction

1. **MGLL-Fundus**: 246,389 fundus image-text pairs with multi-granular annotations, aggregated from 49 public datasets covering 50+ diseases. Granularity levels include: normal/abnormal labels, specific disease categories, and clinical interpretation descriptions.
2. **MGLL-Xray**: 190,882 chest X-ray images from the MIDRC database. Granularity levels include: imaging modality (CR/DX), study description, and series description.

## Key Experimental Results

### Main Results
MGLL is compared against CLIP, CheXzero, MRM, UniChest, and other state-of-the-art methods on 9 fundus downstream datasets and 3 X-ray datasets:

| Method | MIDRC-XR AUC (LP/FT) | MIDRC-Portable AUC (LP/FT) | ChestX-ray14 AUC (LP/FT) |
|------|---------------------|--------------------------|-------------------------|
| CLIP | 54.72 / 88.52 | 71.43 / 91.83 | 69.75 / 82.05 |
| UniChest | 59.02 / 92.51 | 78.49 / 95.44 | 76.15 / 85.84 |
| FG-CLIP | 58.31 / 93.29 | 80.31 / 96.93 | 76.62 / 85.10 |
| **MGLL** | **61.25 / 99.08** | **83.86 / 99.75** | **82.94 / 87.37** |

MGLL achieves the best results across all datasets under both linear probe and fine-tuning settings. On the multi-label dataset RFMiD, MGLL outperforms the second-best method by 16.6% in linear probe AUC and 6.7% in fine-tuning AUC.

Results when embedded into MLLMs — replacing the visual encoder of 7 MLLMs:

| MLLM | Original Accuracy | +MGLL Accuracy | Gain |
|------|-----------|-------------|------|
| InstructBLIP | 47.29% | 61.99% | +14.7% |
| LLaVA | 72.73% | 79.98% | +7.3% |
| LLaVA-Med | 24.28% | 58.37% | +34.1% |
| Med-Flamingo | 26.97% | 58.70% | +31.7% |
| InternVL | 77.35% | 81.96% | +4.6% |
| Janus-Pro | 68.92% | 79.80% | +10.9% |

Gains are most pronounced for medical-domain models (LLaVA-Med, Med-Flamingo), while general-purpose models (LLaVA, InternVL) also show notable improvements.

### Ablation Study
Loss function ablation on the RFMiD dataset:

| Configuration | LP AUC | FT AUC | Notes |
|------|--------|--------|------|
| CLIP baseline | 44.66 | 65.10 | Single-label, single-granularity |
| $\mathcal{L}_P$ only | 70.34 | 88.25 | Point-wise contributes most |
| $\mathcal{L}_{\text{sCLIP}}$ only | 67.86 | 85.13 | Soft CLIP also yields notable gains |
| $\mathcal{L}_{\text{sCLIP}} + \mathcal{L}_P$ | 75.73 | 90.31 | Complementary effect |
| Full MGLL | **79.62** | **92.83** | +sKL yields further improvement |

Ablation on the number of granularity levels (MIDRC-XR-Portable): increasing from 1 to 2 to 3 granularities yields monotonically increasing AUC (LP: 80.54 → 82.92 → 83.86), validating the importance of preserving hierarchical information structure.

### Key Findings
- The point-wise loss contributes most (AUC gain of 25.68%), as it jointly optimizes positive and negative pairs.
- The smooth KL divergence as a cross-granularity constraint provides an additional ~4% AUC improvement.
- Regarding encoder selection, ViT-L/14 outperforms ViT-H/14 (larger is not always better, suggesting overfitting), and BiomedicalBERT outperforms both the CLIP text encoder and LLaMA.
- MGLL substantially outperforms CLIP even under low-resolution or noisy text conditions, demonstrating strong robustness.

## Highlights & Insights
- **Plug-and-play design**: Without introducing any additional encoder parameters, MGLL achieves multi-label and multi-granular alignment purely through improved loss functions, and can directly replace the contrastive objective in any VLM.
- **Elegant theoretical analysis**: Gradient analysis demonstrates that the soft CLIP loss drives image features toward the weighted centroid of the associated text features (Eq. 10), providing a clear and intuitive interpretation.
- **Engineering value of large-scale dataset construction**: MGLL-Fundus (246K pairs, 49 datasets, 50+ diseases) and MGLL-Xray (190K images) fill a critical gap in multi-granular pre-training data for medical imaging.
- **Evaluation paradigm for embedding into MLLMs**: Evaluating MGLL by substituting the visual encoder of 7 MLLMs is a transferable experimental design for assessing domain-specific visual encoders in other fields.

## Limitations & Future Work
- **Granularity definition requires domain expertise**: Granularity levels and corresponding texts must be manually designed for each medical domain, limiting generalizability.
- **Validation limited to classification tasks**: Downstream tasks such as segmentation and detection, which are equally important in medical imaging, are not evaluated.
- **Dataset bias toward fundus and chest X-ray**: Generalizability to other modalities such as CT, MRI, and pathology slides remains unexplored.
- **Coarse modeling of inter-granularity relationships**: The smooth KL divergence simply pulls all granularity distributions toward the mean without explicitly modeling hierarchical or containment relationships among granularities (e.g., disease category as a hypernym of severity level).
- **Future directions**: Exploring automatic extraction of multi-granular annotations from medical reports and encoding hierarchical (tree-structured) relationships into the loss function.

## Related Work & Insights
- **vs. CLIP**: CLIP performs hard single-label matching, whereas MGLL performs multi-label soft matching with cross-granularity consistency, yielding substantial gains in medical settings (LP AUC on RFMiD: 44.66 → 79.62).
- **vs. MedCLIP**: MedCLIP addresses false negatives via semantic matching but remains single-granularity; MGLL fundamentally restructures the supervision signal.
- **vs. UniChest**: UniChest performs domain adaptation for chest X-rays, while MGLL provides a more general multi-granular framework effective for both X-ray and fundus imaging.
- **vs. SupCon**: Supervised contrastive learning exploits label structure but is limited to a fixed label space; MGLL enables open-vocabulary semantics through the text encoder.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of multi-label and multi-granular alignment is novel, though each individual loss function is not new in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation covers 11 downstream datasets, 7 MLLMs, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical analysis and experimental presentation are clear, though the related work section is somewhat crowded.
- **Value**: ⭐⭐⭐⭐ Directly applicable to medical visual pre-training; both the dataset and the method are readily reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](../../AAAI2026/medical_imaging/sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[ICLR 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[AAAI 2026\] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](../../AAAI2026/medical_imaging/medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)

</div>

<!-- RELATED:END -->
