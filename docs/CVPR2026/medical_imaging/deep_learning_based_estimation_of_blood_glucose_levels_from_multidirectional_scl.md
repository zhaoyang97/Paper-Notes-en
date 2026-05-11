---
title: >-
  [Paper Note] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
description: >-
  [CVPR 2026][Medical Imaging][blood glucose estimation] This paper proposes ScleraGluNet, which captures scleral blood vessel photographs from five gaze directions, extracts direction-specific vascular features via parallel CNNs, refines them through MRFO feature selection, and fuses them across views using a Transformer. The model simultaneously performs three-class metabolic state classification (93.8% accuracy) and continuous fasting plasma glucose (FPG) estimation (MAE = 6.42 mg/dL, r = 0.983).
tags:
  - CVPR 2026
  - Medical Imaging
  - blood glucose estimation
  - scleral vessel imaging
  - multi-view learning
  - MRFO
  - Transformer
date: 2026-05-08
content_hash: e7fbf3e0661f1e80
---

# Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.12715](https://arxiv.org/abs/2603.12715)
**Code**: None
**Area**: Medical Imaging
**Keywords**: blood glucose estimation, scleral vessel imaging, multi-view learning, MRFO, Transformer

## TL;DR

This paper proposes ScleraGluNet, which captures scleral blood vessel photographs from five gaze directions, extracts direction-specific vascular features via parallel CNNs, refines them through MRFO feature selection, and fuses them across views using a Transformer. The model simultaneously performs three-class metabolic state classification (93.8% accuracy) and continuous fasting plasma glucose (FPG) estimation (MAE = 6.42 mg/dL, r = 0.983).

## Background & Motivation

**Limitations of Prior Work**: The 537 million diabetic patients worldwide require frequent blood glucose monitoring. Laboratory tests (FPG, HbA1c) are accurate but require venipuncture, making daily self-monitoring impractical. Fingerstick sampling is painful and carries infection risk, leading to poor adherence. Continuous glucose monitoring (CGM) is convenient but requires subcutaneous sensor implantation at considerable cost. Non-invasive blood glucose monitoring thus represents a major unmet clinical need.

**Background**: Chronic hyperglycemia induces microvascular remodeling—including changes in vessel diameter, increased tortuosity, and perfusion abnormalities—making scleral vasculature a natural metabolic window. Compared to retinal imaging, which requires a specialized fundus camera, **scleral/conjunctival vessel imaging requires only a standard anterior-segment camera**, lowering equipment costs and simplifying operation, making it suitable for telemedicine and large-scale screening. Prior OCTA studies have confirmed the association between scleral microvasculature and diabetes.

**Key Challenge**: (1) Existing methods rely on a single gaze direction, yet diabetes-induced microvascular changes are spatially heterogeneous—the degree of vascular abnormality differs across the superior, inferior, nasal, and temporal sclera—so single-view acquisition misses critical information; (2) the complementary relationships among multi-view features have not been adequately exploited.

**Goal**: The core idea is to achieve **full scleral coverage through multi-directional acquisition combined with a deep multi-view fusion architecture**.

## Method

### Overall Architecture

Five scleral photographs are captured per subject (primary gaze, superior, inferior, nasal, temporal) → image preprocessing (ROI extraction + CLAHE + Frangi vessel enhancement) → five independent-parameter CNN branches extract directional features → MRFO feature refinement removes redundancy → Transformer cross-view self-attention fusion → dual output heads (classification + regression).

### Key Designs

1. **Multi-Directional Scleral Acquisition Protocol**:

    - **Function**: Standardized acquisition of scleral photographs across five gaze directions, comprehensively covering vasculature in all quadrants.
    - **Mechanism**: Primary gaze serves as the reference for the central region; superior/inferior/left/right gaze directions expose the inferior/superior/temporal/nasal sclera, respectively. Each participant yields five images, totaling 445 × 5 = 2,225 images.
    - **Design Motivation**: Diabetic microvascular lesions are spatially non-uniform, with different regions exhibiting varying degrees of vessel caliber change, tortuosity, and perfusion abnormality. Ablation experiments confirm that multi-view acquisition significantly outperforms single-view.

2. **Parallel CNN + MRFO Feature Refinement**:

    - **Function**: Five independent-parameter CNN branches learn vascular patterns specific to each direction; MRFO then selects the most relevant features.
    - **Mechanism**: Branches share the same architecture but have independent parameters, extracting direction-specific vascular morphological features (caliber change, tortuosity, branching complexity). MRFO (Manta Ray Foraging Optimization) is a bio-inspired optimization algorithm that selects the most discriminative feature subset from the concatenated features, removing cross-view redundancy.
    - **Design Motivation**: Directly concatenating five-stream features introduces substantial redundant dimensions that dilute discriminative signals. MRFO automatically identifies and retains the feature subset most relevant to glycemic status.

3. **Transformer Cross-View Fusion + Dual Task Heads**:

    - **Function**: Self-attention discovers long-range vascular pattern correlations across quadrants and simultaneously produces classification and regression outputs.
    - **Mechanism**: MRFO-refined features are fed into a Transformer, whose self-attention captures cross-view patterns (e.g., bilateral asymmetric remodeling, subtle vascular features spanning quadrants). The Transformer output is connected to a classification head (3-class softmax) and a regression head (continuous FPG value), trained jointly with $L = L_{\text{CE}} + L_{\text{MSE}}$.
    - **Design Motivation**: Multi-task learning enables the learned representation to serve both classification and regression simultaneously, providing complementary supervisory signals.

### Loss & Training

Joint loss $L = L_{\text{CE}} + L_{\text{MSE}}$, where cross-entropy is used for metabolic state classification and MSE for blood glucose regression. Adam optimizer; subject-wise 5-fold cross-validation (GroupKFold ensures all images from the same participant remain in the same fold, preventing data leakage). 95% confidence intervals are estimated via participant-level bootstrap resampling (1,000 iterations).

Image preprocessing pipeline: ROI extraction → color/brightness normalization → CLAHE contrast enhancement → Frangi filtering to enhance tubular vascular structures.

## Key Experimental Results

### Main Results

**Dataset**: 445 participants (150 normoglycemic, 140 controlled diabetic, 155 hyperglycemic), Changsha Aier Eye Hospital.

| Task | Metric | Result |
|------|--------|--------|
| Three-class classification | Overall accuracy | **93.8%** (95% CI: 91.8–95.4%) |
| | AUC (normal/controlled/hyperglycemic) | 0.971 / 0.956 / 0.982 |
| | F1 (normal/controlled/hyperglycemic) | 0.937 / 0.918 / 0.942 |
| Glucose regression | MAE | **6.42 mg/dL** |
| | RMSE | 7.91 mg/dL |
| | Pearson r | 0.983 |
| | R² | 0.966 |
| | Bland–Altman bias | +1.45 mg/dL (±8.33 to +11.23) |

### Ablation Study

| Configuration | Classification Accuracy | Regression MAE | Notes |
|---------------|------------------------|----------------|-------|
| Single-view CNN | Lowest | Highest | No multi-view information |
| Multi-view CNN (direct concatenation) | Moderate | Moderate | Redundancy unaddressed |
| + MRFO feature selection | Better | Lower | Redundancy removal effective |
| **ScleraGluNet (full model)** | **93.8%** | **6.42** | All components optimal |

### Key Findings

- Accuracy across five folds is stable at 92.8%–94.6% (SD = 0.7%), indicating results are not dependent on favorable data splits.
- Grad-CAM analysis reveals: the normoglycemic group exhibits diffuse and weak attention; the controlled group focuses on regions with mild vascular changes; the hyperglycemic group shows strong, cross-directionally consistent activation on dilated/tortuous vessels.
- Misclassifications occur primarily between adjacent metabolic categories (normoglycemic ↔ controlled), consistent with the clinical reality that glycemia exists on a continuum.

## Highlights & Insights

- Estimating blood glucose from scleral vasculature represents an entirely novel clinical application. Scleral imaging equipment is far less expensive than retinal fundus cameras and requires no pupil dilation, making it highly suitable for telemedicine and community screening. The regression precision of r = 0.983 and the Bland–Altman results approach clinically actionable levels.
- The multi-directional acquisition protocol has a solid physiological basis—microvascular abnormalities are spatially non-uniform—and ablation experiments validate the necessity of multi-view capture.

## Limitations & Future Work

- This is a single-center study (445 participants from one hospital); generalizability has not been externally validated, and equipment and population diversity are insufficient.
- Confounding factors are inadequately controlled: hypertension, smoking, and anemia also affect scleral vascular morphology.
- The study targets fasting plasma glucose only; postprandial glucose dynamics and longitudinal monitoring are not addressed.
- Writing quality is substandard, with certain passages showing evident signs of LLM-generated text.

## Related Work & Insights

- **vs. retinal biomarker studies (e.g., Google retinal biomarker)**: Scleral imaging requires simpler, lower-cost equipment, making it better suited for large-scale screening, although retinal imaging has more extensive clinical validation.
- **vs. PPG/thermal imaging–based glucose estimation**: Scleral imaging directly visualizes microvascular structure, providing stronger physical coupling and lower susceptibility to environmental interference.

## Rating

- Novelty: ⭐⭐⭐⭐ — Scleral vasculature as a source for blood glucose estimation is a genuinely novel clinical scenario; the multi-directional acquisition protocol is creative.
- Experimental Thoroughness: ⭐⭐⭐ — Single-center dataset of 445 participants; ablation is adequate, but external validation and broader baseline comparisons are lacking.
- Writing Quality: ⭐⭐ — Evident LLM-assisted writing; certain passages are redundant and unnatural.
- Value: ⭐⭐⭐ — High clinical application potential, but multi-center validation is required to confirm feasibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_wi.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)

</div>

<!-- RELATED:END -->
