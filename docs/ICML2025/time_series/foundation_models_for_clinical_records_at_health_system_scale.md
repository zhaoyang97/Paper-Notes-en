---
title: >-
  [Paper Note] Foundation Models for Clinical Records at Health System Scale
description: >-
  [ICML2025][Time Series][Electronic Health Records] Proposes GPT-EHR, a generative pre-training framework based on next-visit event prediction. By training a decoder-only Transformer on longitudinal EHR data of 1.29 million patients from NYU Langone, GPT-EHR predicts the onset of dementia and knee osteoarthritis in a zero-shot manner. Its performance is comparable to fully fine-tuned BERT baselines, while successfully uncovering and addressing a critical pitfall where repeated…
tags:
  - "ICML2025"
  - "Time Series"
  - "Electronic Health Records"
  - "Foundation Models"
  - "Generative Pre-training"
  - "Zero-shot Prediction"
  - "Disease Prediction"
  - "Transformer"
date: 2026-05-08
content_hash: 31f77f1fdcd380da
---

# Foundation Models for Clinical Records at Health System Scale

**Conference**: ICML2025  
**arXiv**: [2507.00574](https://arxiv.org/abs/2507.00574)  
**Authors**: Haresh Rengaraj Rajamohan, Xiang Gao, Weicheng Zhu, Shih-Lun Huang, Long Chen, Kyunghyun Cho, Cem M. Deniz, Narges Razavian (NYU Langone Health / NYU)  
**Code**: Not publicly available  
**Area**: Medical Imaging  
**Keywords**: Electronic Health Records, Foundation Models, Generative Pre-training, Zero-shot Prediction, Disease Prediction, Transformer  

## TL;DR

Proposes GPT-EHR, a generative pre-training framework based on next-visit event prediction. By training a decoder-only Transformer on longitudinal EHR data of 1.29 million patients from NYU Langone, GPT-EHR predicts the onset of dementia and knee osteoarthritis in a zero-shot manner. Its performance is comparable to fully fine-tuned BERT baselines, while successfully uncovering and addressing a critical pitfall where repeated event tokens artificially inflate evaluation metrics.

## Background & Motivation

### Problem Background
Early detection and progression prediction of chronic diseases (e.g., dementia, osteoarthritis, cancer) are crucial for improving clinical outcomes. Electronic Health Records (EHR) contain vast, longitudinal data, but their serialized, high-dimensional, irregular, and heterogeneous nature poses severe modeling challenges. Conventional approaches train separate models for each specific disease and time window, which is resource-intensive, lacks flexibility, and fails to leverage dependencies among different clinical events.

### Limitations of Prior Work
- **Masked pre-training approaches** (Li et al. 2020; Yang et al. 2023) require an additional fine-tuning stage and task-specific datasets, resulting in high costs.
- **Next-token generative approaches** (McDermott et al. 2023; Renc et al. 2024) are mostly trained on specific scenarios such as ICU records, failing to address unique challenges in longitudinal outpatient records, such as the unordered nature of events within a single visit and the recurrence of chronic diseases.
- **Repeated events pitfall**: Chronic disease diagnosis codes recur across multiple visits. Models can achieve high prediction scores simply by memorizing existing diagnoses, which masks their ability to detect truly novel onset events (first observed by Kraljevic et al. 2022, but lacking systematic analysis and solutions).

### Core Motivation
To develop a generative foundation model capable of learning comprehensive representations from EHR data and generating patient trajectories. This is achieved by replacing next-token prediction with next-visit event prediction to naturally handle unordered events within a visit, and introducing a regularization mechanism to mitigate the influence of repeated events, thereby boosting detection capabilities for new-onset diseases.

## Method

### Overall Architecture: GPT-EHR System

GPT-EHR employs a decoder-only Transformer architecture (based on GPT-2) and undergoes generative pre-training on large-scale longitudinal EHR data. Its core paradigm is **next-visit multi-label prediction**. The overall workflow consists of two phases:

1. **Pre-training Phase**: Given a patient's historical visit sequence $H_{t_i}$, the model predicts all clinical event tokens that will appear in the next visit $i+1$.
2. **Zero-shot Inference Phase**: Without fine-tuning, the pre-trained model directly estimates the probability of developing specific diseases (e.g., dementia, knee osteoarthritis) within the next 2 or 5 years.

### Key Designs 1: EHR Tokenization and Input Representation

| Design Element | Details |
|---------|---------|
| Data Source | Longitudinal EHR from NYU Langone Health (2013-2023), 1.29 million patients |
| Token Types | Demographics, age at visit, medications, diagnoses, lab test results |
| Continuous Value Handling | Quantile-based binning into intervals |
| Vocabulary Size | 42,337 unique tokens |
| Avg. Tokens per Visit | 11.16 tokens/visit |
| Avg. Sequence Length per Patient | 474.21 tokens/patient |
| Visit Separation | Uses a special `sep` token to mark visit boundaries |
| Position Encoding | RoPE (Rotary Position Embedding); tokens within the same visit share the same position embedding |
| Time Interval Encoding | The position encoding of the `sep` token is set to the timestamp of the next visit, explicitly encoding the interval between visits |

This design offers two key advantages: (1) Sharing position encodings within the same visit naturally handles unordered events within a single visit; (2) Utilizing the differences in position encodings of the `sep` token to encode time intervals enables the model to perceive temporal distances between visits.

### Key Designs 2: Modified Causal Attention Mechanism

Standard causal attention only allows tokens to attend to previous tokens. However, in EHR data, events within the same visit occur simultaneously and are unordered. GPT-EHR modifies the attention mask as follows:

- Tokens in visit $v$ can attend to **all tokens in all visits $v' \leq v$**.
- Tokens within the same visit **can attend to each other** (bi-directionally), while maintaining causality across visits.

In addition, to improve training efficiency, a **sequence packing** strategy is employed: sequences from multiple patients are packed into a single training sequence (if space permits), allowing attention to span across packed patient sequences. This improves GPU throughput and potentially provides broader in-context learning capabilities.

### Key Designs 3: Next-Visit Multi-Label Prediction Objective

Unlike token-by-token autoregressive generation in NLP, GPT-EHR formulates the prediction objective as a **joint prediction of all events in the next visit**:

- **Input**: Patient history $H_{t_i}$ + next visit time $t_{i+1}$ (via `sep` token position encoding)
- **Output**: The output representation from the `sep` token, passed through a linear layer and a Sigmoid activation, to predict the probability $\hat{P}_i$ of each token in the vocabulary appearing in the next visit.
- **Loss**: Multi-label binary cross-entropy loss.

### Key Designs 4: Repeated Event Regularization

This is a core contribution of this work—identifying and addressing the **repeated events pitfall** in EHR foundation model evaluation:

**Problem**: Diagnosis codes for chronic diseases (e.g., diabetes, hypertension) frequently recur across a patient's visits. Models can achieve superior performance on standard evaluation metrics simply by learning to replicate existing diagnoses, while failing to acquire genuine predictive capacity for **new-onset diseases**.

**Solution**: Introduce regularization for recurring event tokens by scaling down their weights in the loss function, encouraging the model to allocate more learning capacity to **first-occurrence new events**. This significantly enhances the model's capability to predict new-onset diseases.

### Loss & Training

| Training Parameter | Setting |
|---------|------|
| Dataset Size | 1,288,242 patients |
| Data Split | 70% Train / 15% Val / 15% Test (patient-level) |
| Time Span | January 2013 - January 2023 (10 years) |
| Patient Visit Statistics | Median 21 visits (mean 37.76, range 2-2123) |
| Architecture | Decoder-only Transformer (GPT-2 style) |
| Position Encoding | RoPE (Rotary Position Embedding) |
| Training Strategy | Sequence packing to improve GPU utilization |
| Inference Mode | Zero-shot inference, no fine-tuning |

## Key Experimental Results

### Zero-shot Disease Prediction Performance

The model predicts the incidence probability of dementia and knee osteoarthritis (KOA) within future 2-year and 5-year windows in a zero-shot manner, compared with a fully fine-tuned BERT baseline:

| Task | Prediction Window | GPT-EHR (Zero-shot) | BERT Baseline (Fine-tuned) | Comparison |
|------|---------|------------------|------------------|------|
| Dementia Prediction | 2 Years | Comparable | Fully Fine-tuned | Zero-shot $\approx$ Fine-tuned |
| Dementia Prediction | 5 Years | Comparable | Fully Fine-tuned | Zero-shot $\approx$ Fine-tuned |
| KOA Prediction | 2 Years | Comparable | Fully Fine-tuned | Zero-shot $\approx$ Fine-tuned |
| KOA Prediction | 5 Years | Comparable | Fully Fine-tuned | Zero-shot $\approx$ Fine-tuned |

Key Finding: **Zero-shot GPT-EHR performs comparably to the fully fine-tuned BERT baseline**, demonstrating that generative pre-training on EHR can bypass expensive task-specific fine-tuning.

### Impact of Repeated Events on Evaluation Metrics

| Evaluation Method | Including Repeated Events | New-Onset Events Only | Difference |
|---------|------------|----------|------|
| Standard Evaluation | Artificially Inflated Metric | True Capability | Significant Gap |
| With Regularization | Improved | Significant Improvement | Gap Narrows |

Effects of repeated event regularization:
- When repeated events are not distinguished, performance metrics are severely overestimated.
- After applying regularization, new-onset event prediction performance improves significantly.
- This finding holds general significance for the evaluation of all EHR-based foundation models.

## Highlights & Insights

- **Paradigm Innovation**: Replaces next-token prediction with next-visit event prediction, elegantly resolving the unordered nature of events within visits while maintaining chronological causality.
- **Uncovering Evaluation Pitfalls**: Systematically exposes the issue of inflated evaluation metrics caused by repeated events in EHR foundation models, acting as a crucial heads-up to the EHR-ML community.
- **Zero-shot Capability**: Achieves performance on par with fine-tuned BERT without any fine-tuning, dramatically reducing clinician model deployment costs.
- **Scale of Data**: Trained and validated on real-world health system data from 1.29 million patients, far exceeding publicly available datasets like MIMIC used in most EHR research.
- **Temporal Encoding Design**: Cleverly embeds time-interval information between visits using RoPE position encodings, making the `sep` token a carrier of temporal information.

## Limitations & Future Work

- **Single Health System Data**: Trained and validated solely on NYU Langone Health data; generalizability to other health systems remains to be verified.
- **Limited Evaluation Tasks**: Evaluated only on two diseases (dementia and knee osteoarthritis), without covering a wider range of clinical prediction tasks.
- **Lack of Comparison with Contemporary Generative EHR Models**: Lacks direct comparison with models like MOTOR (McDermott et al. 2023) or CEHR-GPT (Renc et al. 2024).
- **Potential Noise from Sequence Packing**: Allowing attention to cross patient boundaries during sequence packing improves efficiency but might introduce spurious correlations.
- **Structured Data Only**: Does not integrate unstructured EHR data such as clinical notes or imaging.
- **Exploration of Model Scale**: Based on the GPT-2 architecture, leaving the scaling potential of larger models unexplored.
- **Lack of Real-world Clinical Validation**: As a retrospective study, it does not assess real-world performance in prospective clinical decision-making.
- **Sensitivity of Regularization Hyperparameters**: Lacks detailed analysis regarding the sensitivity of regularization strength across different event categories.

## Related Work & Insights

- **EHR Masked Pre-training**: Models like Med-BERT (Li et al. 2020) and BEHRT (Yang et al. 2023) leverage BERT-like masked language modeling to pre-train EHR representations, but require specialized fine-tuning for downstream tasks.
- **EHR Generative Pre-training**: MOTOR (McDermott et al. 2023) and CEHR-GPT (Renc et al. 2024) explore next-token generative modeling for EHR, but are validated primarily on ICU data.
- **Next-visit Prediction Paradigm**: Steinberg et al. (2021) proposed next-visit multi-label prediction, but strictly for representation learning rather than zero-shot inference.
- **The Repeated Events Problem**: Kraljevic et al. (2022) first reported performance degradation when forecasting new concepts; this work builds upon that with more systematic analyses and solutions.
- **Clinical Prediction Models**: Traditional approaches model specific diseases and time windows in isolation (Dubois et al. 2015; Zhu et al. 2024); the proposed foundation model offers a unified alternative.
- **Implications for EHR-ML Evaluation**: The repeated events pitfall serves as a reminder to the community to distinguish new-onset events from recurring ones, as failing to do so may yield unreliable conclusions.

## Rating

- Novelty: 4/5 — The next-visit prediction paradigm and repeated event regularization are innovative, though decoder-only Transformers for EHR are not entirely new.
- Experimental Thoroughness: 3/5 — Validated on large-scale real-world data, but limited to two diseases and lacks direct comparisons with contemporary generative baselines.
- Writing Quality: 4/5 — Clear motivation, comprehensive methodology, and high educational value regarding the repeated events pitfall.
- Value: 4/5 — Practical utility of zero-shot capabilities on par with fine-tuned models is high, and highlighting evaluation pitfalls benefits the broader EHR-ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](../../ACL2025/time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](../../NeurIPS2025/time_series/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/time_series/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[ICML 2026\] FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](../../ICML2026/time_series/factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/time_series/towards_self-supervised_foundation_models_for_critical_care_time_series.md)

</div>

<!-- RELATED:END -->
