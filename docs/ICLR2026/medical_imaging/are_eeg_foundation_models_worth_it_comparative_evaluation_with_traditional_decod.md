---
title: >-
  [Paper Note] Are EEG Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse BCI Tasks
description: >-
  [ICLR 2026][Medical Imaging][BCI] The authors conduct a systematic comparative evaluation involving 5 mainstream EEG foundation models across 7 classification and 2 regression tasks using six evaluation protocols with statistical testing. They propose ST-EEGFormer, a simple ViT baseline pre-trained on 8 million raw EEG segments via Masked Autoencoding
tags:
  - ICLR 2026
  - Medical Imaging
  - BCI
date: 2026-05-08
content_hash: 2a61f451a967a1d4
---
# Are EEG Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse BCI Tasks

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=5Xwm8e6vbh](https://openreview.net/forum?id=5Xwm8e6vbh)  
**Code**: https://github.com/LiuyinYang1101/STEEGFormer  
**Area**: Medical Imaging / EEG Brain Signal Decoding / Foundation Model Evaluation  
**Keywords**: EEG Foundation Models, BCI, Evaluation Benchmark, Linear Probing, Scaling Laws

## TL;DR
The authors conduct a systematic comparative evaluation involving 5 mainstream EEG foundation models across 7 classification and 2 regression tasks using six evaluation protocols with statistical testing. They propose ST-EEGFormer, a simple ViT baseline pre-trained on 8 million raw EEG segments via Masked Autoencoding (MAE). Findings indicate that foundation models hold a significant advantage only in data-abundant population-level decoding; in data-scarce per-subject scenarios, they often fail to outperform compact CNNs or even traditional non-neural decoders. Linear probing is generally weak, and no clear scaling laws were observed.

## Background & Motivation
**Background**: EEG is the most widely used non-invasive signal in Brain-Computer Interfaces (BCI), with paradigms like Motor Imagery, ERP/P300, and SSVEP supporting applications in exoskeleton control, attention decoding, and high-speed spelling. Inspired by foundation models in NLP/CV, several "EEG Foundation Models" have emerged, claiming to learn generalizable neural representations through self-supervised pre-training on large-scale EEG data to overcome the "train-from-scratch" bottleneck for individual tasks.

**Limitations of Prior Work**: The "strong performance" of these models is often reported under limited evaluation conditions—typically using only one or two protocols (e.g., leave-one-subject-out zero-shot or population decoding), rarely including statistical significance tests, and almost never comparing against classic non-neural decoders (e.g., CSP/FBCSP, Riemannian geometry classifiers, FBCCA/TRCA). The latter remain highly competitive in data-constrained scenarios. Consequently, whether foundation models justify the massive computational cost of pre-training and fine-tuning remains unanswered.

**Key Challenge**: Reported advantages stem from **selective evaluation settings** (cherry-picked tasks, protocols, and lack of statistical testing) rather than robust methodological superiority. There is a poorly quantified gap between pre-training assumptions (complex self-supervised objectives $\rightarrow$ universal representations) and downstream empirical capabilities.

**Goal**: To systematically bridge this gap by investigating: (1) Do foundation models learn robust representations (via linear probing)? (2) Do they outperform classic neural decoders? (3) Do they outperform classic non-neural decoders? (4) Can EEG classification models transfer to regression? (5) Do scaling laws exist?

**Key Insight**: Rather than inventing a "stronger" model, the authors build a **fair, reproducible, and statistically rigorous** multidimensional benchmark. They introduce a "deliberately simple" foundation model, ST-EEGFormer, as a transparent control to test the necessity of complex pre-training objectives.

**Core Idea**: By utilizing a six-dimensional evaluation framework and statistical testing, the authors transform the question of "EEG foundation model value" into a falsifiable empirical conclusion, while using a naive MAE baseline to challenge the prevailing notion that MAE is ineffective on raw signals.

## Method
This paper focuses on evaluation and analysis. The "Method" consists of the **evaluation protocol design + task/model matrix**, supplemented by ST-EEGFormer as a baseline.

### Overall Architecture
The evaluation of an EEG decoder is split into two orthogonal dimensions: **training/testing protocols** (6 types) $\times$ **downstream tasks** (7 classification + 2 regression + 2 common cross-subject datasets). Each model (5 foundation models in both linear probing and fine-tuning modes, 4 classic CNNs, and several non-neural decoders) is evaluated across this grid. Metrics including Accuracy, AUC, Kappa, MSE, and Pearson correlation are analyzed using Wilcoxon signed-rank, permutation, and Mann–Whitney U tests (with Bonferroni correction). All datasets undergo minimal preprocessing (0.1–128 Hz bandpass + 256 Hz resampling).

The six evaluation protocols characterize different facets of generalization:

- **Population**: A single model is trained on pooled data from all subjects and tested on each—measuring global neural patterns across individuals.
- **Per-Subject (Self)**: A model is trained and tested individually for each subject—the traditional subject-dependent BCI approach.
- **Per-Subject (Transfer)**: A model trained on Subject A is tested on Subject B—measuring individual model transferability.
- **LOO Zero-Shot**: A population model is trained on all but one subject (Leave-One-Out) and tested on the held-out subject without adaptation.
- **LOO Fine-Tune**: The population model is fine-tuned on limited data from the LOO subject—measuring rapid adaptation.
- **LOO Drop**: Measures the performance degradation on the population data after fine-tuning on a new subject—quantifying catastrophic forgetting/generalization loss.

### Key Designs

**1. Six-Dimensional Evaluation Protocol: Dissecting Generalization**
To address the issue of inflated conclusions from limited protocols, the authors explicitly decompose "generalization" into six measurable facets. For instance, Population decoding assesses pattern learning under high-data conditions, while Per-Subject (Self) focuses on low-data scenarios. LOO Drop specifically quantifies the loss of global knowledge after individual adaptation. This differentiation reveals that foundation models excel in Population settings but lack significance in low-data Per-Subject contexts.

**2. High-Variance Downstream Task Matrix: Exposing Task Dependency**
The authors chose 7 classification tasks with varying complexity, most of which were not covered by pre-training data: from nearly binary Error-ERN to 3-class Alzheimer's diagnosis, 4-class inner speech, 4-class Motor Imagery (BCI-IV-2A), 7-class upper limb execution/imagery, and 40-target SSVEP. Two regression tasks (DTU auditory attention, SEED-VIG vigilance) were also included. This matrix revealed that performance is heavily task-dependent; simple tasks (ERN) hit performance ceilings (≈99.9%) quickly, while difficult tasks (inner speech) show little improvement regardless of model size.

**3. ST-EEGFormer: A "Simple" MAE Baseline**
Modern works like LaBraM claim that MAE on raw EEG is ineffective and requires discrete neural tokens. The authors counter this with ST-EEGFormer, a ViT that segments EEG into spatial-temporal patches, projects them into tokens, and adds positional encodings (TPE/SPE). It masks 75% of tokens and reconstructs the raw signal via a transformer decoder. Pre-trained on 8 million+ EEG segments, the large variant (ST-EEGFormer-l) matches or exceeds the strongest classic CNNs (CTNet), proving that raw signal MAE is sufficient for state-of-the-art results.

**4. Decoupling Implementation Factors: Identifying Confounders**
The authors identified two misrepresented design choices. First, **Linear Probing Head Capacity**: Models like EEGPT and CBraMod use multi-layer heads even in "linear" settings, effectively hiding extra capacity. Second, **Token Fusion Strategy**: Differences in how tokens are pooled (e.g., class-token vs. average pooling) significantly impact the receptive field and information flow to the head. These details can cause substantial performance gaps, necessitating standardized reporting of head structures and fusion schemes.

## Key Experimental Results

### Main Results (Grouped Horizontal Comparison)
The comparison uses average rankings across metrics, subjects, datasets, and protocols (lower is better):

| Comparison | Key Finding |
|----------|----------|
| Representation Robustness (LP vs. FT) | Fine-tuning (FT) consistently outperforms Linear Probing (LP) except for EEGPT; LP is weak across nearly all protocols except LOO Drop. |
| Foundation Models vs. Classic CNNs | Classic CNNs (especially CTNet) stable outperform all LP-based foundation models. Only the largest FT foundation model (ST-EEGFormer-l) reaches or exceeds the best classic CNN. |
| Foundation Models vs. Non-Neural | FT foundation models (Ours: ST-EEGFormer-l) generally achieve top performance; however, in low-data protocols like Per-Subject (Self), classic non-neural decoders show no significant difference. |
| Scaling Laws | "Bigger is better" does not hold across architectures (poor log-fit of normalized accuracy to model size); however, training time grows exponentially with size ($R^2=0.60$). |

Population decoding is the strongest scenario for foundation models, where FT variants (Ours: ST-EEGFormer-l) significantly outperform classic paradigms. This advantage diminishes in Per-Subject slices.

### Ablation Study (Regression & Scaling)
LOO Zero-Shot performance on regression datasets indicates that classification-to-regression transfer is fragile:

| Dataset | Best Model | Pearson R | Observation |
|--------|----------|-----------|------|
| DTU (Auditory, 1s) | CTNet / EEGPT (LP) | ≈0.05 | Classic decoders and LP-foundation models outperform **fine-tuned** foundation models. |
| SEED-VIG (Vigilance, 5s) | EEGNet / DeepConvNet | >0.45 | Classic CNNs lead, though differences with foundation models are not significant. |

Notably, many FT foundation models (BIOT, BENDR, LaBraM, CBraMod) showed degraded MSE (>1.3) and Pearson R near 0 on DTU, suggesting their representations are not universal for regression.

### Key Findings
- **Foundation Models are not a Panacea**: Advantages are clear only in data-abundant population settings; compact CNNs (CTNet) or classic non-neural methods match them in data-scarce scenarios.
- **Linear Probing is Weak and Task-Dependent**: The only exception is ERN detection, where all LP models perform well, indicating pre-trained feature utility depends on the paradigm.
- **Fine-Tuning "Erases" Pre-training Differences**: Attention visualizations show significant focus shifts after FT. This explains why a simple MAE model like ST-EEGFormer can match complex pre-trained models after fine-tuning.
- **Asymmetric Scaling Cost**: Accuracy improves marginally with scale, while computational time increases exponentially. Small BCI datasets (<50 subjects) are the primary bottleneck for scaling gains.
- **Classic CNNs suffer most in LOO Drop**: When fine-tuning on new subjects, classic CNNs experience the most severe catastrophic forgetting of population knowledge.

## Highlights & Insights
- **Falsifiable Benchmarking**: The 6D protocol x Task Matrix x Statistical Testing framework provides an honest empirical baseline for the field.
- **Challenging Prevailing Assumptions**: Disproves the claim that raw MAE is ineffective by providing a top-performing simple ViT baseline (ST-EEGFormer).
- **Identifying "Linear Probing" Inflation**: Points out that hidden head complexity often accounts for performance gains rather than superior backbone representations.
- **Reporting Negative Results**: Explicitly states that large foundation models often only match small classic models, highlighting the lack of practical utility given the computational overhead.

## Limitations & Future Work
- **Fragmented Ecosystem**: Diverse pre-training strategies and small downstream datasets (<50 subjects) limit broad scaling analysis.
- **Lack of "EEG ImageNet"**: Many findings (e.g., lack of scaling laws) may be due to data scarcity rather than model limits.
- **Regression Constraints**: Only two regression datasets were tested; the low R-values on DTU might mask model differences due to task difficulty.
- **Future Work**: Scaling experiments on larger unified datasets and standardizing reporting protocols for head structures and fusion strategies.

## Related Work & Insights
- **vs. LaBraM**: LaBraM advocates for neural tokenization; Ours shows raw MAE is sufficient, suggesting complex pre-training objectives provide limited marginal gains.
- **vs. EEGPT / CBraMod**: These models use stronger heads; Ours highlights that head capacity must be normalized for fair backbone comparison.
- **vs. Classic Decoders**: Demonstrates that simple baselines (CSP, etc.) are still competitive in low-data regimes, warning against neglecting traditional methods.

## Rating
- Novelty: ⭐⭐⭐⭐ (Evaluation paradigm focuses on honesty and falsifiability).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 foundation models, 4 CNNs, 9 tasks, 6 protocols with stats).
- Writing Quality: ⭐⭐⭐⭐ (Clear and honest, though many results are in the appendix).
- Value: ⭐⭐⭐⭐⭐ (Provides a critical "reality check" and a reproducible framework for the EEG community).

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model](codebrain_bridging_decoupled_tokenizer_and_multi-scale_architecture_for_eeg_foun.md)
- [\[ICLR 2026\] Bridging Radiology and Pathology Foundation Models via Concept-Based Multimodal Co-Adaptation](bridging_radiology_and_pathology_foundation_models_via_concept-based_multimodal_.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](../../NeurIPS2025/medical_imaging/3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](../../AAAI2026/medical_imaging/personalization_of_large_foundation_models_for_health_interventions.md)
- [\[ICLR 2026\] Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)

</div>

<!-- RELATED:END -->
