---
title: >-
  [Paper Note] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks
description: >-
  [ICLR 2026][Time Series][ECG] A comprehensive "reality check" benchmarking of eight ECG foundation models across 12 datasets and 26 clinical tasks reveals that ECG-CPC, a compact Structured State Space Model (SSM), outperforms large-scale Transformers in five out of seven task categories, proving that architectural design is more critical than mode
tags:
  - ICLR 2026
  - Time Series
  - ECG
date: 2026-05-08
content_hash: 81eb259a51eea8df
---
# Benchmarking ECG FMs: A Reality Check Across Clinical Tasks

**Conference**: ICLR 2026  
**arXiv**: [2509.25095](https://arxiv.org/abs/2509.25095)  
**Code**: [https://github.com/AI4HealthUOL/ecg-fm-benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)  
**Area**: Time Series  
**Keywords**: ECG, Foundation Models, Structured State Space Models, Electrocardiogram, Benchmarking  

## TL;DR
A comprehensive "reality check" benchmarking of eight ECG foundation models across 12 datasets and 26 clinical tasks reveals that ECG-CPC, a compact Structured State Space Model (SSM), outperforms large-scale Transformers in five out of seven task categories, proving that architectural design is more critical than model scale.

## Background & Motivation

**Background**: The 12-lead electrocardiogram (ECG) is the most widely used cardiac diagnostic tool. Recently, several ECG Foundation Models (FMs) have been released, including CNN-based ECGFounder, Transformer-based ECG-JEPA/ST-MEM/HuBERT-ECG/ECG-FM, and contrastive learning-based MERL/ECGFM-KED. These models utilize diverse pre-training strategies (supervised, self-supervised, contrastive) and varying dataset scales.

**Limitations of Prior Work**:
- Existing studies often evaluate models on limited datasets or single task categories, failing to provide generalizable conclusions.
- Baselines used for comparison are frequently weak, leading to an overestimation of FM advantages.
- Systematic analysis regarding "model scale vs. architectural choice" is lacking—is a larger model inherently better?

**Key Challenge**: Does the "scale equals quality" assumption prevalent in the FM field hold true for ECG? How significantly does the generalization capability vary across different architectures (CNN/Transformer/SSM) for diverse clinical tasks?

**Goal**: To address three core research questions: (1) Which architecture generalizes best across diverse ECG tasks? (2) How do FMs scale with labeled data volume? (3) What causes the performance differences between models?

**Core Idea**: Establish a comprehensive evaluation framework covering seven task categories and introduce a self-trained lightweight SSM, ECG-CPC, as a control group to reveal the true performance boundaries of ECG FMs.

## Method

### Overall Architecture
As a benchmarking study, the core contribution is not a new module but a "reality check" of eight pre-trained ECG FMs and two supervised baselines trained from scratch across 12 public datasets and 26 clinical tasks (including classification and regression). Each model is evaluated under fine-tuning, frozen, and linear probing modes. The study further incorporates label efficiency scaling analysis and Centered Kernel Alignment (CKA) to answer questions regarding architectural generalization, scaling behavior, and the origins of performance variance.

### Key Designs

**1. Evaluation Matrix: 8 Pre-trained FMs + 2 Training-from-scratch Baselines × 7 Clinical Task Categories**

Previous ECG FM research often evaluated on narrow scopes and used weak baselines, resulting in systematically overestimated FM superiority. To ensure robust conclusions on "architecture vs. scale," this study covers a broad spectrum. Model-wise, it includes CNNs like ECGFounder (RegNet, 33.8M, supervised), MERL (ResNet18, 4.6M, contrastive), and ECGFM-KED (ResNet, 9.7M, contrastive); Transformers like ECG-JEPA (87.2M, JEPA), ST-MEM (90.3M, MAE), HuBERT-ECG (97.2M, MLM), and ECG-FM (93.9M, MLM+contrastive); and SSMs including the self-trained ECG-CPC (S4 backbone, 3.8M), plus supervised anchors Net1D (33.8M, CNN) and S4 (2.2M, SSM). Task-wise, it spans 7 categories: adult ECG interpretation (9 datasets, 11 tasks), pediatric ECG interpretation, cardiac structure and function (echocardiography metric regression), cardiac/non-cardiac discharge diagnosis, acute care prediction (deterioration/mortality/ICU admission), and patient characteristic prediction (age/sex/biomarkers), totaling approximately 1,650 regression and classification labels. Different categories require models to capture various signal levels: morphology for diagnosis and latent prognostic signals for acute care.

**2. Evaluation Protocol: Three Modes + Segmented Inference + Bootstrap Testing**

Each model is evaluated in three modes. Fine-tuning involves full-model updates with layer-wise learning rates for Transformers/SSMs (backbone rates 10–100× lower than the head)—a crucial step for models like HuBERT-ECG and ECG-FM to converge. The Frozen mode uses a learnable query-attention head for pooling, while the Linear mode employs a simple linear head to detect representation separability. Inputs are standardized to 2.5s segments; during inference, results from 4 segments are averaged, which proved more stable than using full 10s recordings. Statistical significance is assessed via bootstrap confidence intervals ($n=1000$), reporting macro AUROC for classification and z-normalized MAE for regression.

**3. Mechanism: ECG-CPC as a Verifiable Control Group**

ECG-CPC is a lightweight model specifically trained to test the hypothesis that "proper inductive bias beats parameter scale." It utilizes an S4 (Structured State Space Model) backbone and is pre-trained via Contrastive Predictive Coding (CPC) on the HEEDB dataset (10.7M samples). With only 3.8M parameters, it was trained in three weeks on a single NVIDIA L40 GPU. If S4's inductive biases—such as stable long-range memory, spectral filtering, and global parameterized convolutions—align with ECG structures, it should match or exceed 90+M parameter Transformers.

**4. Design Motivation: Label Efficiency Scaling Analysis**

To quantify how much labeling effort FMs save compared to supervised training, controlled scaling experiments were conducted on the EchoNext dataset. Training sets were decimated by powers of 2 down to 1/128. Learning curves were fitted to $CN^{-\alpha} + L_0$, where $L_0$ represents the performance floor/limit and $\alpha$ the convergence rate. The label efficiency ratio is defined as $r = N^*/N$ (the ratio of FM data to supervised baseline data required to achieve equal performance).

## Key Experimental Results

### Main Results: Ranking Across 7 Task Categories (Fine-tuning Mode)

| Task Category | 1st Place | 2nd Place | 3rd Place | S4 baseline |
| :--- | :--- | :--- | :--- | :--- |
| Adult ECG Interpretation | ECGFounder/ECG-JEPA/ECG-CPC | ECG-FM | MERL | Surpassed |
| Pediatric ECG Interpretation | ECG-JEPA | ECGFounder | ST-MEM | 6th |
| Cardiac Structure & Func. | **ECG-CPC** | ECGFounder | ECG-JEPA | 6th |
| Cardiac/Non-cardiac Diag. | **ECG-CPC** | ECG-FM | S4 | 3rd |
| Acute Care Prediction | **ECG-CPC**/ECG-FM | ECGFounder | ECG-JEPA | Not sig. surpassed |
| Patient Characteristics | **ECG-CPC** (5/6 tasks 1st) | MERL/ECG-FM | - | Surpassed 3/6 |

ECG-CPC ranked first in 5 out of 7 categories despite having only 3.8M parameters—less than 1/25th of the largest Transformer models.

### Label Efficiency Study

| Model | Label Efficiency Ratio $r$ ($N=250-1000$) | Implication |
| :--- | :--- | :--- |
| ECG-JEPA | 0.11-0.42 | Highest efficiency (best for low data) |
| ECG-CPC | 0.21-0.40 | Close to JEPA, with higher performance limit |
| ECGFounder | 0.30-0.62 | Lower label efficiency |
| Overall | 3.3-9× | FM efficiency gain over supervised baselines |

Key finding: ECG-JEPA learns "fast but peaks low," whereas ECG-CPC learns slightly slower but has a higher performance ceiling. Model choice should depend on data volume: use ECG-JEPA for $<1000$ samples and ECG-CPC for $>1000$ samples.

### Key Findings
- **Architectural Dominance**: SSMs outperform Transformers primarily due to S4's inductive biases (stable long-range memory, spectral filtering), which naturally match ECG signal structures.
- **Learning Rate Sensitivity**: Layer-wise learning rates are critical; some models (HuBERT-ECG, ECG-FM) fail to train without them.
- **Segmented Processing**: 2.5s cropping with test-time averaging outperforms using full 10s recordings directly.
- **Representation Diversity**: CKA analysis shows that models with similar performance learn vastly different internal representations. ECG-CPC shows structured layer specialization, while ECGFounder and ECG-JEPA exhibit high intermediate layer redundancy.

## Highlights & Insights
- **Evidence for "Architecture > Scale"**: ECG-CPC's success with 1/25th the parameters challenges the "larger is better" dogma in FMs, suggesting that for structured time series, inductive bias is more important than scale.
- **Slope vs. Limit Decomposition**: Decomposing label efficiency into "learning rate" and "performance ceiling" provides a practical guide for model selection based on available data.
- **Representation Convergence**: The observation that different architectures achieve similar performance via different feature paths suggests that performance alone may not fully describe FM quality.
- **Accessibility**: Reaching SOTA performance with three weeks of single-GPU training democratizes FM research for resource-constrained medical AI labs.

## Limitations & Future Work
- **In-domain Evaluation**: Evaluations were limited to in-domain sets; out-of-distribution (OOD) generalization across different devices or populations was not assessed due to label incompatibility.
- **Multi-task Confounding**: Some tasks utilized joint training to save compute, which might influence specific task performances.
- **Pre-training Data Inconsistency**: FMs were pre-trained on different datasets; ideally, all models would be re-trained on a unified dataset, though the computational cost is prohibitive.
- **Wearable Gap**: Evaluations focused on standard 12-lead ECGs, whereas wearables typically provide single-lead data.
- **Future Directions**: Combining token-level and sequence-level objectives, unified pre-training dataset ablations, and expansion to OOD benchmarks.

## Related Work & Insights
- **vs. ECGFounder (Li et al., 2025)**: While ECGFounder excels in adult ECG interpretation due to supervised pre-training, it falls behind ECG-CPC in non-diagnostic tasks.
- **vs. ECG-JEPA (Kim, 2024)**: JEPA offers superior few-shot efficiency but a lower performance ceiling compared to the CPC-based ECG-CPC.
- **vs. Mamba (Gu & Dao, 2024)**: Internal experiments suggest that newer SSMs like Mamba do not necessarily outperform S4 on continuous medical signals; S4's specific inductive biases remain highly effective for ECG.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The benchmarking framework is exceptionally systematic; ECG-CPC provides significant insights despite not being a brand-new architecture.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage with 8 FMs, 12 datasets, 26 tasks, multiple evaluation modes, scaling analysis, and CKA.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, deep analysis, and actionable conclusions.
- **Value**: ⭐⭐⭐⭐⭐ A vital reference for the ECG FM community with insights applicable to broader medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness](timerecipe_a_time-series_forecasting_recipe_via_benchmarking_module_level_effect.md)
- [\[ICLR 2026\] Can we generate portable representations for clinical time series data using LLMs?](can_we_generate_portable_representations_for_clinical_time_series_data_using_llm.md)
- [\[ICML 2025\] Foundation Models for Clinical Records at Health System Scale](../../ICML2025/time_series/foundation_models_for_clinical_records_at_health_system_scale.md)
- [\[ICML 2026\] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](../../ICML2026/time_series/position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)

</div>

<!-- RELATED:END -->
