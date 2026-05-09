---
title: >-
  [Paper Note] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction
description: >-
  [ACL 2026][Medical Imaging][Clinical Risk Prediction] CURA proposes a dual-level uncertainty calibration framework: at the individual level, it aligns predictive uncertainty with error probability; at the cohort level, it regularizes predictions using neighborhood event rates in the embedding space. The framework consistently improves calibration metrics across five clinical risk prediction tasks on MIMIC-IV without sacrificing discriminative performance.
tags:
  - ACL 2026
  - Medical Imaging
  - Clinical Risk Prediction
  - Uncertainty Calibration
  - Dual-Level Alignment
  - Cohort-Aware
  - Clinical Language Model
date: 2026-05-08
content_hash: bfd0cbdf2b1e8ce5
---

# CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction

**Conference**: ACL 2026
**arXiv**: [2604.14651](https://arxiv.org/abs/2604.14651)
**Code**: [GitHub](https://github.com/sizhe04/CURA)
**Area**: Medical Imaging
**Keywords**: Clinical Risk Prediction, Uncertainty Calibration, Dual-Level Alignment, Cohort-Aware, Clinical Language Model

## TL;DR
CURA proposes a dual-level uncertainty calibration framework: at the individual level, it aligns predictive uncertainty with error probability; at the cohort level, it regularizes predictions using neighborhood event rates in the embedding space. The framework consistently improves calibration metrics across five clinical risk prediction tasks on MIMIC-IV without sacrificing discriminative performance.

## Background & Motivation

**Background**: Clinical language models (e.g., BioClinicalBERT, BioGPT) have demonstrated strong performance in predicting risks such as mortality and ICU length of stay from free-text clinical notes. However, the uncertainty estimates of these models are often poorly calibrated—overconfident erroneous predictions pose a direct threat to patient safety.

**Limitations of Prior Work**: General-purpose uncertainty methods (MC Dropout, Deep Ensembles) aggregate predictions on isolated samples without exploiting the semantic structure of the representation space. LLM-specific calibration methods rely on expert reasoning chains or textual explanations from teacher models, yet clinical tasks typically provide only binary labels and lack large-scale ground-truth rationales.

**Key Challenge**: Fine-tuning improves predictive performance but exacerbates overconfidence—high-confidence yet incorrect predictions for high-risk patients create "false reassurance," which is particularly dangerous in clinical settings.

**Goal**: Design a lightweight, plug-and-play calibration framework that maintains high confidence for correct predictions while assigning high uncertainty to incorrect ones.

**Key Insight**: Align uncertainty simultaneously at two levels—individually with each sample's own error rate, and at the cohort level with the event rate among neighbors in the embedding space.

**Core Idea**: Freeze the fine-tuned clinical LM embeddings → multi-head classifier + dual-level uncertainty objectives (individual calibration $L_{ind}$ + cohort-aware regularization $L_{coh}$).

## Method

### Overall Architecture
CURA proceeds in two stages: (1) standard fine-tuning of a clinical LM with weighted binary cross-entropy, followed by freezing the encoder to extract patient embeddings; (2) training a multi-head MLP classifier ensemble on the frozen embeddings, jointly optimizing a base loss, an individual calibration loss, and a cohort-aware loss. At inference time, predictions from $M$ heads are averaged.

### Key Designs

1. **Individual Uncertainty Calibration ($L_{ind}$)**:

    - **Function**: Aligns the model's predictive uncertainty (normalized entropy) with each sample's individual error probability.
    - **Mechanism**: Define the correctness probability $a(x) = y\bar{p}(x) + (1-y)(1-\bar{p}(x))$, the uncertainty score $u(x) = H(x)/H_{max}$ (normalized entropy), and align $u(x)$ with $1-a(x)$ via cross-entropy: $L_{ind} = -\lambda_{ind} [(1-a(x))\log u(x) + a(x)\log(1-u(x))]$. This encourages high confidence (low loss) on correct predictions and penalizes the model for failing to express uncertainty on incorrect ones.
    - **Design Motivation**: Standard cross-entropy loss does not constrain the relationship between confidence and error rate, leaving overconfident incorrect predictions without additional penalty.

2. **Cohort-Aware Risk Alignment ($L_{coh}$)**:

    - **Function**: Ensures that clinically similar patients receive consistent risk estimates.
    - **Mechanism**: For each patient embedding, retrieve its $K$ nearest neighbors and compute the neighborhood event rate $q(x_i) = \frac{1}{K}\sum_{j \in \mathcal{N}_K(e_i)} y_j$ as the cohort risk. Regularize predictions toward this cohort risk using an adaptive weight $w(x_i) = \lambda_{coh} \hat{H}(q(x_i))$—the weight increases as the neighborhood event rate approaches 0.5 (ambiguous cohort). This is equivalent to cross-entropy with neighborhood-based soft labels, i.e., data-dependent label smoothing.
    - **Design Motivation**: Individual calibration considers each sample in isolation and cannot leverage the prior that clinically similar patients should receive similar risk estimates. Cohort-level regularization is particularly important in ambiguous regions near the decision boundary.

3. **Multi-Head Classifier Ensemble**:

    - **Function**: Obtains diverse uncertainty estimates at low computational cost.
    - **Mechanism**: Construct $M$ independently and randomly initialized lightweight MLP heads on top of the frozen embeddings; average their predictions at inference time. Sharing a single backbone minimizes inference overhead.
    - **Design Motivation**: Deep Ensembles require training multiple complete models; the multi-head architecture preserves diversity in uncertainty estimation while substantially reducing computational cost.

### Loss & Training
The total loss is $L_{total} = L_{base} + L_{ind} + L_{coh}$. $L_{base}$ is a weighted binary cross-entropy that provides a discriminative foundation and prevents $L_{ind}$ from degenerating to uniform probability outputs. $L_{coh}$ can be interpreted as cross-entropy with neighborhood soft labels, where the soft labels interpolate between the ground-truth label and the neighborhood event rate.

## Key Experimental Results

### Main Results

| Task | Method | AUROC | Brier↓ | NLL↓ | AURC↓ |
|------|--------|-------|--------|------|-------|
| 7-Day Mortality | Baseline | 0.852 | 0.032 | 0.120 | 0.008 |
| 7-Day Mortality | Deep Ensemble | 0.856 | 0.029 | 0.110 | 0.007 |
| 7-Day Mortality | CURA | **0.892** | **0.015** | **0.075** | **0.002** |
| 30-Day Mortality | Baseline | 0.881 | 0.064 | 0.231 | 0.024 |
| 30-Day Mortality | CURA | **0.890** | **0.038** | **0.146** | **0.009** |
| In-Hospital Mortality | Baseline | 0.621 | 0.044 | 0.175 | 0.015 |
| In-Hospital Mortality | CURA | **0.641** | **0.029** | **0.124** | **0.011** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| $L_{base}$ only (multi-head) | Calibration close to baseline | Multi-head architecture alone is insufficient to improve calibration |
| $L_{base} + L_{ind}$ | Brier/NLL improved | Individual calibration is effective |
| $L_{base} + L_{coh}$ | Further improvement | Cohort regularization is effective |
| $L_{base} + L_{ind} + L_{coh}$ | Best | Dual-level synergy yields optimal results |

### Key Findings
- CURA consistently improves calibration metrics (Brier, NLL, AURC) across all five tasks without degrading and even slightly improving discriminative performance (AUROC, AUPRC).
- Deep Ensembles and MC Dropout yield limited calibration improvements and even slightly worsen calibration on certain tasks.
- CURA substantially reduces "false reassurance" for high-risk patients by redistributing high-confidence incorrect predictions to high-uncertainty regions.
- The framework is robust across three backbone models: BioGPT, BioClinicalBERT, and ClinicalBERT.

## Highlights & Insights
- **The dual-level alignment design** is both elegant and practically valuable—individual-level alignment enforces "express uncertainty when wrong," while cohort-level alignment enforces "similar patients should receive similar risks," and the two objectives are complementary.
- The label-smoothing interpretation of $L_{coh}$ provides theoretical insight: it is essentially data-dependent label softening using neighborhood event rates, with stronger smoothing in ambiguous regions.
- As plug-and-play loss terms, CURA requires no modifications to the model architecture or inference pipeline, resulting in extremely low deployment overhead.

## Limitations & Future Work
- Evaluation is limited to MIMIC-IV; generalizability to other EHR datasets remains to be validated.
- The neighborhood size $K$ is a hyperparameter that may require task-specific tuning.
- Embedding quality depends on the domain adaptation of the pre-trained LM.
- The binary classification setting limits applicability to multi-level risk stratification scenarios.

## Related Work & Insights
- **vs. Deep Ensembles**: Training multiple complete models yields limited calibration improvement; CURA achieves better calibration at lower cost via multi-head architecture and dual-level losses.
- **vs. MC Dropout**: Uncertainty is obtained through random dropout without exploiting the structure of the representation space; CURA leverages semantic information in the embedding space through neighborhood relationships.
- **vs. LLM Calibration Methods**: These methods rely on chain-of-thought explanations as supervision, which are unavailable in clinical settings; CURA requires only binary labels.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-level uncertainty alignment design is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five tasks, three backbone models, five-fold cross-validation, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clinical motivation is clearly articulated, mathematical derivations are complete, and visual analyses are intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](../../AAAI2026/medical_imaging/deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](../../NeurIPS2025/medical_imaging/cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [\[NeurIPS 2025\] AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](../../NeurIPS2025/medical_imaging/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)

</div>

<!-- RELATED:END -->
