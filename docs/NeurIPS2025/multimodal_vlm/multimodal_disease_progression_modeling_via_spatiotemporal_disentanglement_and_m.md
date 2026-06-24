---
title: >-
  [Paper Note] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
description: >-
  [NeurIPS 2025 Spotlight][Multimodal VLM][disease progression modeling] This paper proposes DiPro, a framework that addresses redundancy in longitudinal chest X-ray sequences and cross-modal temporal misalignment through region-aware spatiotemporal disentanglement (separating static anatomical from dynamic pathological features) and multiscale alignment (local–global fusion of CXR and EHR), achieving state-of-the-art performance on disease progression recognition and ICU predi…
tags:
  - "NeurIPS 2025 Spotlight"
  - "Multimodal VLM"
  - "disease progression modeling"
  - "multimodal fusion"
  - "spatiotemporal disentanglement"
  - "longitudinal CXR"
  - "electronic health records"
date: 2026-05-08
content_hash: a1a17290bd348685
---

# Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.11112](https://arxiv.org/abs/2510.11112)  
**Code**: [GitHub](https://github.com/Chenliu-svg/DiPro)  
**Area**: Medical Imaging
**Keywords**: disease progression modeling, multimodal fusion, spatiotemporal disentanglement, longitudinal CXR, electronic health records

## TL;DR

This paper proposes DiPro, a framework that addresses redundancy in longitudinal chest X-ray sequences and cross-modal temporal misalignment through region-aware spatiotemporal disentanglement (separating static anatomical from dynamic pathological features) and multiscale alignment (local–global fusion of CXR and EHR), achieving state-of-the-art performance on disease progression recognition and ICU prediction tasks.

## Background & Motivation

Longitudinal multimodal clinical data are essential for disease progression modeling, yet two core challenges persist:

**Redundancy in clinical image sequences**: Static anatomical structures in consecutive chest X-rays (CXR)—such as chronic cardiomegaly and stable skeletal deformities—dominate the signal and obscure clinically more significant subtle pathological changes (e.g., new infiltrates, evolving edema). Existing methods (e.g., CheXRelNet, SDPL) treat all imaging features uniformly, without distinguishing long-term anatomical baselines from evolving pathological changes.

**Cross-modal temporal misalignment**: EHR provides continuous, high-frequency measurements (e.g., hourly vital signs), whereas CXR offers only sparse, irregularly timed snapshots, resulting in an inherent temporal granularity mismatch. Existing multimodal methods (e.g., MedFuse, DrFuse) rely solely on the most recent CXR, discarding longitudinal information; longitudinal methods (e.g., UTDE, UMSE) depend on rigid interpolation or fixed temporal embeddings, lacking adaptive cross-modal alignment mechanisms.

DiPro is grounded in two key observations: (1) disease progression in CXR sequences manifests through pathological changes in local anatomical regions; and (2) EHR and imaging data exhibit complementary dynamics at different temporal granularities.

## Method

### Overall Architecture

DiPro comprises three modules: Spatiotemporal Disentanglement (STD), which separates static anatomical from dynamic pathological features in consecutive CXR pairs; Progression-Aware Enhancement (PAE), which reinforces the directional sensitivity of dynamic features by reversing the CXR temporal order; and Multiscale Multimodal Fusion (MMF), which aligns CXR and EHR at both local (temporal-interval) and global (sequence) scales.

### Key Designs

1. **Spatiotemporal Disentanglement (STD)**: For each anatomical region $\mathbf{R}_{t_i}^r$, features are extracted using a pretrained ResNet-50 and adjacent frame features are concatenated. Static and dynamic representations $\mathbf{S}_i^r$ and $\mathbf{D}_i^r$ are obtained via a static projection head $f_s$ and a dynamic projection head $f_d$, respectively. Two auxiliary losses constrain disentanglement quality: (a) an **orthogonal disentanglement loss** minimizes the cosine similarity between static and dynamic features: $\mathcal{L}_{\text{orth}} = \frac{1}{(T-1)R}\sum_i\sum_r (\text{sim}(\mathbf{S}_i^r, \mathbf{D}_i^r))^2$; (b) a **temporal consistency loss** enforces temporal stability of static features: $\mathcal{L}_{\text{temp}} = \frac{1}{N}\sum_r\sum_i \|\mathbf{S}_i^r - \mathbf{S}_{i+1}^r\|_2^2$. **Design motivation**: Mixing features of different clinical semantics dilutes progression signals; explicit separation allows the model to focus on meaningful pathological changes.

2. **Progression-Aware Enhancement (PAE)**: The core idea is concise yet elegant—reversing the input order of a CXR pair should invert the progression direction while leaving static information unchanged. Reversed inputs yield $\widetilde{\mathbf{D}}_i^r$ and $\widetilde{\mathbf{S}}_i^r$, and $K$ disease-specific classification heads predict progression states: the original direction predicts $y_i^{r,k}$ and the reversed direction predicts $-y_i^{r,k}$. The training loss is: $\mathcal{L}_{\text{PAE}} = \sum_{r,k}[\text{CE}(\hat{y}, y) + \text{CE}(\tilde{y}, -y)] + \lambda_{\text{static}}\sum_r \|\mathbf{S}_i^r - \widetilde{\mathbf{S}}_i^r\|_2^2$. **Design motivation**: The temporal-reversal equivariance constraint forces dynamic features to encode progression direction while further verifying the temporal invariance of static features.

3. **Multiscale Multimodal Fusion (MMF)**: CXR and EHR are fused at three levels:

    - **Local fusion**: For each CXR temporal interval $[t_i, t_{i+1}]$, relative temporal embeddings $T_{t_j} = f_{\text{TE}}([t_j - t_i, t_{i+1} - t_j, \sigma((t_j-t_i)(t_{i+1}-t_j))])$ are designed. Cross-attention with a center-focused attention mask extracts interval-specific EHR features $\mathbf{E}_i^{\text{local}}$ from the global EHR representation, which are then fused with dynamic CXR features via cross-attention.
    - **Global fusion**: All local fusion features are aggregated; the global EHR representation attends to fusion features across all temporal intervals via cross-attention, followed by self-attention enhancement.
    - **Static fusion**: Demographic information is concatenated with static CXR features and fused with dynamic and global features via cross-attention to generate the final prediction. **Design motivation**: Multiscale fusion captures local EHR–CXR interactions at the interval level and global progression trends at the sequence level, bridging modalities across different temporal granularities.

### Loss & Training

The total training loss is a weighted combination of multiple terms: $\mathcal{L} = \lambda_{\text{pred}} \cdot \text{CE}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}} + \lambda_{\text{temp}}\mathcal{L}_{\text{temp}} + \lambda_{\text{PAE}}\mathcal{L}_{\text{PAE}}$

Experiments use the MIMIC data family (MIMIC-IV EHR + MIMIC-CXR imaging + Chest ImaGenome region annotations), selecting ICU admissions with ≥2 CXRs.

## Key Experimental Results

### Main Results

Disease progression recognition (macro-average across seven chest diseases):

| Method | Type | Precision | Recall | F1 | AUPRC |
|---|---|---|---|---|---|
| SDPL | Unimodal CXR | 0.408 | 0.406 | 0.393 | 0.417 |
| CheXRelNet | Unimodal CXR | 0.395 | 0.392 | 0.389 | 0.394 |
| DiPro (unimodal) | Unimodal CXR | 0.475 | 0.452 | **0.453** | **0.468** |
| UTDE | Multimodal | 0.481 | 0.462 | 0.449 | 0.472 |
| DrFuse | Multimodal | 0.442 | 0.461 | 0.429 | 0.438 |
| DiPro (multimodal) | Multimodal | **0.484** | **0.471** | **0.466** | **0.478** |

ICU prediction tasks (longitudinal CXR + EHR setting):

| Method | Mortality AUPRC | Mortality AUROC | LOS Kappa | LOS ACC |
|---|---|---|---|---|
| UMSE | 0.712 | 0.891 | 0.204 | 0.410 |
| MedFuse | 0.716 | 0.881 | 0.210 | 0.412 |
| UTDE | 0.710 | 0.887 | 0.195 | 0.400 |
| DiPro | **0.742** | **0.897** | **0.248** | **0.440** |

### Ablation Study

| Configuration | Disease Prog. F1 | Mortality AUPRC | LOS ACC | Notes |
|---|---|---|---|---|
| DiPro (full) | 0.466 | 0.742 | 0.440 | All modules perform as expected |
| A1: w/o MMF | 0.460 | 0.724 | 0.416 | MMF contributes most to ICU prediction |
| A2: w/o PAE | 0.433 | 0.730 | 0.432 | PAE contributes significantly to progression recognition |
| A3: STD only | 0.439 | 0.694 | 0.404 | Performance degrades without fusion strategy |
| A4: Baseline | 0.362 | 0.721 | 0.425 | STD yields 21.3% relative F1 improvement |
| DiPro-: auto bbox | 0.457 | 0.736 | 0.430 | Replacing with auto-detected regions remains effective |

### Key Findings

- **Unimodal DiPro already surpasses all baselines**: CXR-only DiPro improves F1 over SDPL by 15.3%, demonstrating the intrinsic value of spatiotemporal disentanglement.
- **EHR fusion yields consistent gains**: Multimodal DiPro further improves F1 by 2.9% over the unimodal variant, confirming the effectiveness of multimodal fusion.
- **Attention weights align with clinical knowledge**: The cardiac silhouette region receives the highest attention for cardiomegaly detection, the hilar region is prominent for pulmonary edema detection, and right-sided structures receive greater attention for mortality prediction—all consistent with established clinical knowledge.
- DiPro is the first work to incorporate EHR data into the CXR-based disease progression recognition task.

## Highlights & Insights

- **Clinical rationale for "static–dynamic" disentanglement in CXR sequences**: The temporal consistency of anatomical structures and the temporal variability of pathological changes are fundamental principles in radiological diagnosis; DiPro formalizes this clinical intuition.
- **Temporal reversal trick in PAE**: A simple yet effective constraint—reversing the input order should invert the progression direction—provides an elegant self-supervised signal for temporal feature learning.
- **Center-focused attention mask**: The attention mask design for handling EHR–CXR temporal misalignment is notably elegant, achieving soft temporal window selection via sigmoid approximation.
- **Interpretability**: Attention weight analysis offers visual explanations of the decision process, enhancing clinical trustworthiness.

## Limitations & Future Work

- The method relies on region annotations (anatomical bounding boxes) from Chest ImaGenome; although ablations show that automatic bounding boxes remain effective, this constitutes an additional dependency.
- Admissions with only a single CXR are excluded, introducing sampling bias.
- Progression labels for the seven diseases are derived from Chest ImaGenome's automatic annotations rather than expert annotations, introducing label noise.
- Validation is performed solely on MIMIC (single-center) data; generalizability to multi-center settings remains unknown.
- The static/dynamic disentanglement assumes anatomical structures are stable in the short term, but certain structures (e.g., cardiac size) may change gradually over long-term follow-up.

## Related Work & Insights

- **Connection to video representation learning**: The STD module is analogous to dynamic texture versus static background separation, augmented with domain-specific orthogonality constraints and temporal consistency for the medical domain.
- **Relation to multistream temporal fusion**: The local–global fusion paradigm in MMF is generalizable to other asynchronous multimodal settings (e.g., MRI + EEG, CT + genomics).
- **Implications for clinical prediction**: Explicitly modeling the direction of disease progression is more conducive to risk stratification than merely extracting difference features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of spatiotemporal disentanglement, PAE, and multiscale fusion is novel, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale public datasets, multi-task evaluation, extensive ablation studies, attention visualization, and robustness analysis are all well covered.
- **Writing Quality**: ⭐⭐⭐⭐ The architecture diagrams are clear, though the large number of modules makes the method section lengthy.
- **Value**: ⭐⭐⭐⭐⭐ The work offers significant practical value for longitudinal multimodal clinical prediction and directly supports ICU clinical decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ECCV 2024\] A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](../../ECCV2024/multimodal_vlm/a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/multimodal_vlm/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)

</div>

<!-- RELATED:END -->
