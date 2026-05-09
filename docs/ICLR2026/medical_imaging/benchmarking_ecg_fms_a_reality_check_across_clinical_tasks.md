---
title: >-
  [Paper Note] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks
description: >-
  [ICLR 2026][Medical Imaging][ECG] A comprehensive "reality check" benchmark evaluating 8 ECG foundation models across 12 datasets and 26 clinical tasks reveals that the compact structured state space model (SSM) ECG-CPC outperforms large-scale Transformers in 5 out of 7 task categories, demonstrating that architectural design matters more than model scale.
tags:
  - ICLR 2026
  - Medical Imaging
  - ECG
  - Foundation Models
  - Structured State Space Models
  - Electrocardiogram
  - Benchmarking
date: 2026-05-08
content_hash: 34f7bdfd7e095de9
---

# Benchmarking ECG FMs: A Reality Check Across Clinical Tasks

**Conference**: ICLR 2026  
**arXiv**: [2509.25095](https://arxiv.org/abs/2509.25095)  
**Code**: [https://github.com/AI4HealthUOL/ecg-fm-benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)  
**Area**: Medical Imaging  
**Keywords**: ECG, Foundation Models, Structured State Space Models, Electrocardiogram, Benchmarking  

## TL;DR
A comprehensive "reality check" benchmark evaluating 8 ECG foundation models across 12 datasets and 26 clinical tasks reveals that the compact structured state space model (SSM) ECG-CPC outperforms large-scale Transformers in 5 out of 7 task categories, demonstrating that architectural design matters more than model scale.

## Background & Motivation

**Background**: The 12-lead electrocardiogram (ECG) is the most widely used cardiac diagnostic tool. Multiple ECG foundation models (FMs) have been released in recent years, including CNN-based ECGFounder, Transformer-based ECG-JEPA/ST-MEM/HuBERT-ECG/ECG-FM, and contrastive learning-based MERL/ECGFM-KED. These models employ different pretraining strategies (supervised, self-supervised, contrastive) and datasets of varying scales.

**Limitations of Prior Work**:
   - Existing studies typically evaluate on limited datasets or a single task category, preventing generalizable conclusions.
   - Baseline comparisons often use weak models, leading to inflated estimates of FM advantages.
   - Systematic analysis of "model scale vs. architectural choice" is lacking — does a larger model necessarily perform better?

**Key Challenge**: Does the "scale equals quality" assumption prevalent in FM research hold in the ECG domain? How large are the generalization gaps across different architectures (CNN/Transformer/SSM) on diverse clinical tasks?

**Goal**: Three core research questions — (1) Which architecture generalizes best across diverse ECG tasks? (2) How do FMs scale with labeled data? (3) What drives performance differences between models?

**Core Idea**: Construct a comprehensive evaluation framework spanning 7 task categories, and introduce a self-trained lightweight SSM model, ECG-CPC, as a reference point to reveal the true capability boundaries of ECG FMs.

## Method

### Overall Architecture
Evaluation pipeline: 8 pretrained FMs + 2 supervised baselines trained from scratch → 12 public datasets → 26 clinical tasks (classification + regression) → comprehensive evaluation under fine-tuning, frozen, and linear probing modes → supplemented by label efficiency scaling analysis and representational similarity analysis (CKA).

### Key Designs

1. **Model Selection (8 FMs + 2 Baselines)**:

    - CNN-based: ECGFounder (RegNet, 33.8M parameters, supervised pretraining), MERL (ResNet18, 4.6M, contrastive), ECGFM-KED (ResNet, 9.7M, contrastive)
    - Transformer-based: ECG-JEPA (87.2M, JEPA), ST-MEM (90.3M, MAE), HuBERT-ECG (97.2M, MLM), ECG-FM (93.9M, MLM + contrastive)
    - SSM-based: ECG-CPC (S4 backbone, 3.8M parameters, CPC pretraining) — newly trained in this work
    - Supervised baselines: Net1D (33.8M, CNN) and S4 (2.2M, SSM, trained from scratch)
    - **Design Motivation**: Covers three major architectural families and key pretraining strategies; ECG-CPC has only 1/25 the parameters of the largest Transformer.

2. **Comprehensive Coverage Across 7 Clinical Task Categories**:

    - Adult ECG interpretation (9 datasets, 11 tasks), pediatric ECG interpretation, cardiac structure and function (echocardiographic prediction), cardiac/non-cardiac discharge diagnoses, acute care prediction (deterioration/mortality/ICU admission), patient characteristic prediction (age/sex/biomarkers/lab values)
    - A total of 1,650 regression and classification target labels
    - **Design Motivation**: Different task categories require models to capture different levels of ECG information; evaluation on a single task category is misleading.

3. **Evaluation Methodology**:

    - Fine-tuning: full model fine-tuning with layer-wise learning rates (backbone 10–100× lower than prediction head)
    - Frozen: encoder frozen, with a learnable query-attention head for pooling
    - Linear: encoder frozen, with a linear head
    - Training on 2.5-second segments with 4-segment averaging at inference, rather than full 10-second recordings
    - Bootstrap confidence intervals ($n=1000$) for statistical significance testing
    - Classification metric: macro AUROC; regression metric: z-normalized MAE

4. **ECG-CPC Model**:

    - Based on the S4 structured state space model backbone
    - Self-supervised pretraining via Contrastive Predictive Coding (CPC)
    - Trained on the HEEDB dataset (10.7 million samples)
    - Only 3.8M parameters, trained on a single NVIDIA L40 GPU for three weeks
    - **Design Motivation**: Validates the "small but effective" hypothesis — good inductive biases (S4's long-range memory, spectral filtering, globally parameterized convolutions) matter more than parameter count.

### Label Efficiency Analysis
A controlled scaling experiment on the EchoNext dataset reduces the training set by powers of 2 down to 1/128 of the full size, fitting the scaling curve $CN^{-\alpha} + L_0$. The label efficiency ratio $r = N^*/N$ is computed, representing the fraction of data an FM requires to match the performance of a supervised baseline.

## Key Experimental Results

### Main Results: Rankings Across 7 Task Categories (Fine-tuning Mode)

| Task Category | Rank 1 | Rank 2 | Rank 3 | S4 Baseline |
|---------|------|------|------|------------|
| Adult ECG Interpretation | ECGFounder/ECG-JEPA/ECG-CPC | ECG-FM | MERL | Outperformed |
| Pediatric ECG Interpretation | ECG-JEPA | ECGFounder | ST-MEM | 6th |
| Cardiac Structure & Function | **ECG-CPC** | ECGFounder | ECG-JEPA | 6th |
| Cardiac/Non-cardiac Diagnosis | **ECG-CPC** | ECG-FM | S4 | 3rd |
| Acute Care Prediction | **ECG-CPC**/ECG-FM | ECGFounder | ECG-JEPA | Not significantly outperformed |
| Patient Characteristics | **ECG-CPC** (1st in 5/6 tasks) | MERL/ECG-FM | — | Outperformed in 3/6 |

ECG-CPC ranks first in 5 of 7 categories despite having only 3.8M parameters — fewer than 1/25 of the largest Transformer.

### Label Efficiency Analysis

| Model | Label Efficiency Ratio $r$ ($N=250$–$1000$) | Interpretation |
|------|------------------------|------|
| ECG-JEPA | 0.11–0.42 | Highest label efficiency (best in low-data regimes) |
| ECG-CPC | 0.21–0.40 | Close to JEPA, with a higher performance ceiling |
| ECGFounder | 0.30–0.62 | Lower label efficiency |
| Overall | 3.3–9× | FM improvement over supervised baselines in label efficiency |

Key finding: ECG-JEPA learns quickly but has a low performance ceiling ("fast but shallow"), while ECG-CPC learns more slowly but achieves a higher ceiling ("slow but tall") — model selection should be data-dependent: prefer ECG-JEPA for <1,000 samples, and ECG-CPC for >1,000 samples.

### Representational Similarity Analysis (CKA)
- ECG-CPC exhibits the clearest and most structured representational evolution: early CNN layers are redundant, while subsequent S4 layers progressively specialize.
- ECGFounder shows high redundancy in middle layers (S0–S4 nearly identical), with specialization only at the final layer.
- ECG-JEPA's intermediate Transformer blocks are nearly identical (Blk1–10), with differentiation only in the final block.
- **Models with similar task performance learn substantially different internal representations**, indicating that there are multiple valid pathways to effective ECG representations.

### Key Findings
- The core reason SSMs outperform Transformers: S4's inductive biases (stable long-range memory, spectral filtering, globally parameterized convolutions) naturally match the structure of ECG signals, enabling efficient learning without large parameter counts.
- Layer-wise learning rates are critical for Transformers and SSMs: certain models (HuBERT-ECG, ECG-FM) cannot even be trained without them.
- 2.5-second crops with test-time averaging outperform direct use of full 10-second recordings.
- No single model consistently dominates all tasks, but ECG-CPC comes closest.

## Highlights & Insights
- **Strong evidence for "Architecture > Scale"**: ECG-CPC, with 3.8M parameters (~1/25 of the largest Transformer), outperforms 90+M-parameter Transformer models on most tasks. This challenges the "bigger is better" assumption in FM research, demonstrating that for structured time series such as ECG, good inductive biases far outweigh parameter count.
- **"Slope vs. Ceiling" analysis of scaling curves**: Decomposing FM label efficiency into two independent dimensions — learning speed and performance ceiling — provides a practical guide for model selection based on dataset size. This analytical framework is transferable to other FM benchmarking settings.
- **"Different paths, same destination" revealed by CKA**: Models with similar task performance have vastly different internal representations, suggesting that evaluation based solely on downstream task performance may be insufficient for a comprehensive assessment of FM quality.
- **Feasibility of extremely low-resource training**: ECG-CPC achieves top-tier performance trained on a single GPU for three weeks, providing a practical pathway for resource-constrained medical AI laboratories.

## Limitations & Future Work
- **In-domain evaluation only**: All evaluations are in-domain; out-of-domain generalization across devices or populations is not assessed (though the authors acknowledge that label incompatibility is the primary obstacle).
- **Confounding from multi-task training**: Some tasks use joint multi-task training to reduce compute, which may inflate or deflate performance relative to task-specific training.
- **Heterogeneous pretraining data**: Each FM is pretrained on a different dataset, making it impossible to fully disentangle the effect of pretraining data. Retraining all models on a unified dataset would be ideal but is computationally prohibitive.
- **No single-lead or wearable evaluation**: All evaluations are based on standard 12-lead ECGs; wearable devices typically provide only a single lead.
- **Future directions**: Combining token-level and sequence-level pretraining objectives (as in ECG-FM), controlled ablations with unified pretraining datasets, and extension to out-of-domain generalization evaluation.

## Related Work & Insights
- **vs. ECGFounder (Li et al., 2025)**: ECGFounder performs well on adult ECG interpretation using RegNet with supervised pretraining, but underperforms ECG-CPC on non-diagnostic tasks, possibly due to insufficient coverage of downstream tasks in its supervised label set.
- **vs. ECG-JEPA (Kim, 2024)**: JEPA's joint-embedding predictive architecture provides the best label efficiency (especially in low-data regimes), but has a lower performance ceiling than ECG-CPC's CPC-based approach.
- **vs. Mamba/Modern SSMs (Gu & Dao, 2024)**: Internal experiments suggest that newer SSMs such as Mamba do not necessarily outperform S4 on continuous medical signals; S4's inductive biases are better suited to ECG.
- **Insight**: The potential of SSMs for medical time series may be underestimated, warranting further exploration on other physiological signals (EEG, PPG).

## Rating
- Novelty: ⭐⭐⭐⭐ The evaluation framework is comprehensive and systematic; while ECG-CPC is not an entirely novel architecture, its strong performance provides important insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 FMs + 2 baselines, 12 datasets, 26 tasks, 3 evaluation modes, scaling analysis, and CKA — extremely thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with in-depth analysis, clear conclusions, and practical guidance.
- Value: ⭐⭐⭐⭐⭐ Highly valuable as a reference for the ECG FM community; the "architecture > scale" finding also carries broader implications for the medical AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](../../NeurIPS2025/medical_imaging/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)

</div>

<!-- RELATED:END -->
