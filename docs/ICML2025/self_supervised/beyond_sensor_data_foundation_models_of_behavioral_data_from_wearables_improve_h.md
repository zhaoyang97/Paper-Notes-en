---
title: >-
  [Paper Note] Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions
description: >-
  [ICML 2025][Self-Supervised Learning][wearable foundation model] Using physical activity data from 162K participants and 2.5 billion hours of wearable behavioral data from the Apple Heart and Movement Study, this work systematically explores combinations of tokenizers and architectures. By constructing WBM, a behavioral foundation model leveraging TST + Mamba-2 + contrastive learning, the model significantly outperforms hand-crafted feature baselines across 57 health detectio…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "wearable foundation model"
  - "behavioral data"
  - "health detection"
  - "Mamba-2"
  - "contrastive learning"
date: 2026-05-08
content_hash: b51ab320782342db
---

# Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions

**Conference**: ICML 2025  
**arXiv**: [2507.00191](https://arxiv.org/abs/2507.00191)  
**Code**: None (Apple internal research, restricted by informed consent)  
**Area**: Self-Supervised Learning  
**Keywords**: wearable foundation model, behavioral data, health detection, Mamba-2, contrastive learning

## TL;DR

Using physical activity data from 162K participants and 2.5 billion hours of wearable behavioral data from the Apple Heart and Movement Study, this work systematically explores combinations of tokenizers and architectures. By constructing WBM, a behavioral foundation model leveraging TST + Mamba-2 + contrastive learning, the model significantly outperforms hand-crafted feature baselines across 57 health detection tasks and complements PPG sensor models.

## Background & Motivation

**Background**: Wearable foundation models have developed rapidly in recent years, but almost all focus on low-level sensor signals—such as second-level raw data from PPG, ECG, and accelerometers. Although valuable, these signals are not continuously available throughout the day, which limits their coverage in health status detection.

**Limitations of Prior Work**: Higher-level behavioral data (steps, active minutes, heart rate variability, gait metrics, VO2Max, etc.) naturally correspond to the time scale of human behavior (days/weeks) and represent expert-validated, physiologically relevant metrics that should be highly informative for health detection. However, behavioral data faces three major challenges: (1) irregular sampling, where different variables have different sampling frequencies; (2) substantial missingness, where certain metrics (e.g., VO2Max) are only recorded after specific workouts; and (3) heterogeneity across variables, with 27 variables spanning six categories: physical activity, cardiovascular, vital signs, gait, body composition, and cardiorespiratory fitness.

**Key Challenge**: Large-scale unlabeled behavioral data (15.14M weeks) versus limited annotated downstream tasks—a setup naturally suited for the foundation model paradigm, yet the irregularity and heterogeneity of behavioral data prevent the direct application of existing time-series foundation model methods.

**Goal**: (1) How to design an optimal tokenizer for irregularly sampled, multivariate behavioral data? (2) Which architecture is best suited for encoding behavioral time series? (3) Can behavioral foundation models outperform hand-crafted features and sensor-based models across a wide range of health detection tasks?

**Key Insight**: Rather than assuming an optimal solution, this work conducts a systematic ablation of 9 combinations (3x3) over candidate tokenizers (TST/mTAN/Tuple) and architectures (Self-Attention Transformer/Rotary Transformer/Mamba-2), using age prediction as a proxy task to identify the optimal model.

**Core Idea**: Behavioral data is an overlooked goldmine in wearable health AI, requiring custom-designed foundation models to unlock its potential.

## Method

### Overall Architecture

The input is a single user's behavioral data over one week: 27 variables irregularly sampled over 168 hours (7 days $\times$ 24 hours), which is aggregated hourly into a $168 \times 27$ sparse matrix. This is encoded by a tokenizer and fed into the backbone network. The output is mean-pooled across the time dimension to obtain a single embedding vector, which is used for linear-probe evaluation on downstream health detection tasks.

### Key Designs

1. **TST Tokenizer (Final Choice)**:

    - **Function**: Transforms irregular behavioral data into a dense input sequence.
    - **Mechanism**: Constructs a $168 \times 54$ matrix (27 variable values + 27 missingness indicators). Missing values are imputed with the global mean (which is zero after z-scoring). The 54-dimensional vector for each hour acts as a patch, mapped via a single-layer MLP to an embedding vector, forming a sequence of 168 tokens.
    - **Design Motivation**: This seemingly simple global mean imputation surprisingly outperforms more complex individual-level mean imputation, which the authors hypothesize is due to noisy individual-level estimates when data is limited. The dense format allows the model to learn how to handle missingness patterns on its own via the missingness indicators.

2. **mTAN and Tuple Tokenizer (Candidate Solutions)**:

    - mTAN: Also uses a $168 \times 54$ matrix but learns time-aligned embeddings through a Multi-Time Attention Network to handle irregularity directly. Performance is comparable to but slightly worse than TST.
    - Tuple: Treats each observation as a (timestamp, variable type, value) triplet, generating tokens via learnable embeddings. This naturally handles missing values (unobserved elements simply do not appear), but the sequence length is variable and can be extremely long.

3. **Mamba-2 Backbone Architecture (Final Choice)**:

    - **Function**: Encodes the token sequence into contextual representations.
    - **Mechanism**: Employs a bidirectional Mamba-2 (Selective State Space Model), run in both forward and backward directions and then merged. The final output is averaged across all time steps to produce a weekly embedding.
    - **Design Motivation**: The learned discretization step size of continuous-time State Space Models is naturally suited for irregular time intervals. Although Transformers dominate wearable foundation models, Mamba-2 consistently outpaced Self-Attention and Rotary Transformers in this scenario. Even when limiting the depth of Mamba-2 to match the memory budget of Transformers in the initial ablation, the TST + Mamba-2 combination remained consistently optimal.

### Loss & Training

A regularized contrastive loss is used for self-supervised pre-training. Positive pairs are defined as data from different time windows of the same participant (augmented via token dropping—randomly dropping $p\%$ of the temporal tokens). KoLeo regularization is added to prevent representation collapse. The authors explicitly discard Masked Autoencoders (MAE)—since MAE requires reconstructing all inputs, it tends to overemphasize frequent variables to the detriment of sparse ones, and preliminary experiments confirmed poor downstream performance with MAE.

## Key Experimental Results

### Demographic Prediction (Table 1)

| Embedding Method | Age MAE ↓ | Biological Sex AUROC ↑ |
|---------|---------|---------------|
| Hand-crafted Feature Baseline | 7.89 | 0.931 |
| WBM | 3.67 | 0.999 |
| PPG Model | 2.89 | 0.997 |
| **WBM + PPG** | **2.46** | **0.999** |

### Time-Varying Health Status Detection (Table 2)

| Embedding Method | Diabetes AUROC | Pregnancy AUROC | Infection AUROC | Injury AUROC | Sleep Duration R² | Deep Sleep Duration R² |
|---------|-----------|---------|---------|---------|----------|----------|
| Baseline | 0.737 | 0.804 | 0.632 | 0.608 | 0.104 | 0.172 |
| WBM | 0.765 | 0.864 | 0.749 | 0.680 | 0.590 | 0.266 |
| PPG | 0.829 | 0.873 | 0.730 | 0.673 | 0.110 | 0.327 |
| **WBM+PPG** | **0.828** | **0.921** | **0.763** | **0.688** | **0.601** | **0.383** |

### Key Findings

1. WBM outperforms the hand-crafted feature baseline on 39 out of 57 tasks (with a median AUROC improvement of 0.017) and significantly outperforms the baseline on all 8 time-varying tasks.
2. WBM substantially leads PPG on sleep tasks (Sleep Duration R²: 0.590 vs 0.110), as behavioral data covers all 168 hours of the week whereas PPG is sampled only a few times per day.
3. PPG wins on diabetes detection (0.829 vs 0.765), as physiological signals are more direct indicators of metabolic diseases.
4. The combination of WBM + PPG achieves the best results on 42 out of 47 baseline disease/medication tasks, with pregnancy prediction reaching an AUROC of 0.921.
5. TST + Mamba-2 is consistently the best among the 9 combinations, challenging the dominance of Transformers in wearable data.

## Highlights & Insights

- **Clear Positioning of Behavioral vs. Sensor Data**: The two encode information at different time scales; WBM captures day/week-level behavioral patterns (gait changes, exercise habits), while PPG captures second-level physiological signals (heart rhythm anomalies). Their complement is perfectly demonstrated by the 0.921 AUROC in pregnancy prediction.
- **Simple Solutions Outperform Complex Ones**: Global mean imputation > individual-level imputation, and TST (the simplest tokenizer) > mTAN/Tuple. This is a valuable insight when dealing with highly noisy data.
- **Staggering Data Scale**: The AHMS study involving 162K participants, 2.5 billion hours of data, and spanning 5 years represents the largest-scale study to date on wearable behavioral foundation models.

## Limitations & Future Work

- Apple proprietary data + informed consent restrictions mean that model weights and code cannot be released publicly, limiting reproducibility.
- Cohort bias: Participants are iPhone + Apple Watch users, resulting in underrepresentation of females, elderly individuals, and minority ethnic groups.
- Validated only on Apple Watch data, with cross-device generalization left untested.
- Employs only contrastive learning, without exploring non-contrastive SSL methods (such as JEPA).
- Does not support health status prediction/forecasting (only detects current status).

## Related Work & Insights

- Relation to Abbaspourazad et al. (2024a) PPG foundation model: WBM is the behavioral counterpart, complementing rather than replacing it.
- Merrill & Althoff (2023) is the only prior work conducting SSL on behavioral data, but it used only 3 variables and 5.2K individuals.
- Mamba-2 also outperforms Transformers in clinical time series, validating the potential of State Space Models on health data.

## Rating

⭐⭐⭐⭐ — An important foundation laid in the untapped domain of wearable behavioral foundation models. Outstanding data scale and task coverage (57 tasks). The systematic tokenizer $\times$ architecture ablation holds substantial value for practitioners. The main drawback is that the data and models are not publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[ICML 2025\] Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)

</div>

<!-- RELATED:END -->
