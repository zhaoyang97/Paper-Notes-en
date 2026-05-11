---
title: >-
  [Paper Note] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation
description: >-
  [Medical Imaging] This work constructs an open, LLM-assisted emergency triage benchmark based on MIMIC-IV-ED, defining two evaluation scenarios—hospital-rich and mass casualty incident (MCI)-like field simulation—and pro…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: 652955fb49a3b6b0
---

# LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation

## Metadata
- **Conference**: NeurIPS 2025
- **arXiv**: [2509.26351](https://arxiv.org/abs/2509.26351)
- **Code**: Not available
- **Area**: Medical Imaging
- **Keywords**: Emergency triage, large language models, benchmark dataset, deterioration prediction, MIMIC-IV

## TL;DR
This work constructs an open, LLM-assisted emergency triage benchmark based on MIMIC-IV-ED, defining two evaluation scenarios—hospital-rich and mass casualty incident (MCI)-like field simulation—and providing baseline models along with SHAP-based interpretability analysis to promote reproducibility and accessibility in triage prediction research.

## Background & Motivation

Emergency departments face enormous pressure to rapidly identify deterioration risk (e.g., unplanned ICU transfer or in-hospital mortality) across large patient volumes, particularly under resource-constrained MCI conditions. Existing triage research suffers from three core bottlenecks:

**Lack of reproducible benchmarks**: Although MIMIC-IV-ED is publicly available, transforming it into a triage-oriented benchmark requires extensive preprocessing, feature harmonization, and schema alignment, imposing a high technical barrier.

**Incomplete scenario coverage**: Prior work typically focuses on hospital-rich settings and lacks simulation of the limited-resource conditions characteristic of MCI field environments.

**Limitations of traditional scoring systems**: Systems such as NEWS2, AVPU, and START rely on fixed thresholds and narrow inputs, yielding unstable performance across populations.

**Core motivation**: To leverage LLM-assisted data curation to lower technical barriers and construct an open triage benchmark covering both hospital and field scenarios, thereby "democratizing" access to triage datasets.

## Method

### Overall Architecture

A deterministic preprocessing pipeline is built upon MIMIC-IV v3.1 and MIMIC-IV-ED v2.2 to generate triage benchmark datasets under two feature regimes, accompanied by baseline models and interpretability analyses.

### Key Designs

1. **Data construction pipeline**: Starting from ED visit records, records are linked via clinically meaningful keys such as $(subject\_id, hadm\_id)$ to prevent cross-admission leakage. Vital signs and laboratory data are restricted to within one hour of arrival. Rule-based filtering removes physiologically implausible values; continuous features are z-score normalized; missing values are imputed using mean values or unknown category labels. All preprocessing parameters are estimated exclusively on training folds.

2. **Dual-scenario feature regimes**:

    - **Hospital-rich**: Demographics + initial ED vital signs + chief complaint + triage observations (pain, acuity) + early laboratories (hemoglobin, BUN, sodium, potassium, creatinine) + consciousness/respiratory proxies.
    - **MCI-like field simulation**: Demographics + vital signs + chief complaint + triage observations + AVPU/oxygen flags only.

3. **LLM-assisted curation**: LLMs are used for data curation rather than predictive modeling. Specific tasks include:

    - Consistent mapping of GCS verbal responses to AVPU categories with one-hot encoding.
    - Standardization of oxygen support devices (room air / nasal cannula / mask / CPAP, etc., plus a binary flag).
    - Noise filtering of respiratory documentation (e.g., ambiguous entries such as "clear" or "regular").
    - Keyword extraction from free-text chief complaints, including synonym expansion and simple negation handling.
    - Table merging strategies (join keys and deduplication rules).

4. **Derived features**: Include AVPU codes derived from GCS verbal subscores, tiered oxygen support vectors, and shock index ($HR/SBP$).

### Prediction Task Definition

A binary classification task is defined to predict unplanned ICU transfer or in-hospital death within 24 hours of ED arrival during the same hospitalization. The positive class (label $= 1$) corresponds to this composite outcome; all other cases form the negative class.

### Baseline Models

Four highly interpretable models are evaluated: Logistic Regression, Random Forest, XGBoost, and LightGBM. Hyperparameters are tuned via 5-fold cross-validation grid search, with a patient-level 70/30 stratified split.

## Key Experimental Results

### Main Results: Baseline Performance Under Both Scenarios

| Model | AUROC (Hospital) | Acc (Hospital) | AP (Hospital) | F1 (Hospital) | AUROC (MCI) | Acc (MCI) | AP (MCI) | F1 (MCI) |
|------|----------------|--------------|-------------|-------------|-----------|---------|--------|--------|
| Logistic Regression | 0.40 | 0.43 | 0.27 | 0.15 | 0.703 | 0.761 | 0.575 | 0.429 |
| Random Forest | 0.73 | 0.72 | 0.38 | 0.35 | 0.783 | 0.851 | 0.721 | 0.643 |
| XGBoost | 0.56 | 0.65 | 0.33 | 0.36 | 0.734 | 0.746 | 0.599 | 0.452 |
| LightGBM | 0.39 | 0.60 | 0.30 | 0.20 | 0.794 | 0.791 | 0.690 | 0.563 |

Random Forest achieves the best performance under the MCI scenario (AUROC $= 0.783$, F1 $= 0.643$), and the MCI scenario overall outperforms the hospital-rich scenario—likely because noise introduced by early laboratory data in the demo subset degrades hospital-rich performance.

### Ablation Study: Feature Group Contributions

| Best Model | Feature Set | AUROC | Accuracy | AP | F1 |
|---------|-------|-------|---------|-----|-----|
| Logistic Regression | Observations | 0.74 | 0.79 | 0.43 | 0.42 |
| LightGBM | Vital signs | 0.80 | 0.79 | 0.49 | 0.63 |
| LightGBM | Labs | 0.71 | 0.72 | 0.36 | 0.46 |
| Random Forest | Vitals + Observations | 0.76 | 0.82 | 0.60 | 0.63 |
| Random Forest | Vitals + Observations + Labs | 0.82 | 0.81 | 0.67 | 0.61 |

### Key Findings

1. **Vital signs are the most robust signal source**: Used alone, they achieve AUROC $= 0.80$; laboratory data within the one-hour window carries weak and noisy signal.
2. **High cost-effectiveness of observational features**: Simple bedside assessments such as acuity and AVPU demonstrate significant predictive value, consistently improving F1 and AP when added incrementally.
3. **Global SHAP analysis**: Triage acuity and respiratory indicators (respiratory rate, oxygen saturation, blood pressure) are the most important drivers across all scenarios, closely aligned with clinical intuition.
4. **Feasibility of the MCI scenario**: Even with only vital signs and simple observations, models retain strong predictive capacity, supporting deployment in resource-constrained settings.

## Highlights & Insights

1. **LLMs as data curation accelerators**: Rather than serving as predictive models, LLMs address the most time-consuming feature harmonization bottlenecks in data cleaning—a "safe and efficient" paradigm for LLM application in medical AI.
2. **Dual-scenario design**: This work is the first to bridge hospital and field triage within a single benchmark, providing a standardized evaluation framework for MCI research.
3. **Counterintuitive finding**: MCI scenario performance exceeds that of the hospital-rich scenario on the demo subset, suggesting that early laboratory data may introduce noise—a finding that requires validation on the full dataset.
4. **Dataset democratization**: By releasing preprocessing code, feature dictionaries, and split indices, the work substantially lowers the technical barrier to entry for triage research.

## Limitations & Future Work

- Validation is conducted only on the MIMIC demo subset (64 patients, 222 visits), representing an extremely small sample size.
- Waveform data and narrative clinical notes are not incorporated.
- LLM-assisted curation currently involves interactive verification and is not fully automated.
- In-hospital mortality in the demo subset is 0%, limiting the representativeness of outcome events.
- Future work should extend to the full MIMIC-IV dataset and compare against deep sequential models such as RETAIN.

## Related Work & Insights

- **Traditional triage scoring systems**: NEWS2, AVPU, START, SALT — simple but limited by fixed thresholds.
- **ML-based deterioration prediction**: Gradient boosting outperforms traditional scales; text-augmented models approach physician-level performance.
- **LLM-assisted data curation**: DALL-M feature augmentation; retrieval-augmented LLMs for schema alignment.
- **Interpretability**: SHAP is widely applied in ICU and MCI models to make triage decisions transparent.
- **Insight**: The paradigm of LLM-assisted data curation is generalizable to other EHR benchmark construction efforts.

## Rating
- **Novelty**: ⭐⭐⭐☆☆ — A dataset construction contribution; methodological innovation is limited, though the scenario design is distinctive.
- **Experimental Thoroughness**: ⭐⭐☆☆☆ — The demo subset is too small; generalizability of conclusions is uncertain.
- **Writing Quality**: ⭐⭐⭐⭐☆ — The pipeline is clearly described with well-motivated objectives.
- **Value**: ⭐⭐⭐⭐☆ — Advances reproducible triage AI research; value increases substantially upon extension to the full dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](../../ICCV2025/medical_imaging/victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)

</div>

<!-- RELATED:END -->
