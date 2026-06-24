---
title: >-
  [Paper Note] A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs
description: >-
  [NeurIPS 2025 (Workshop: Learning from Time Series for Health)][Medical Imaging][Latent ODE] This work combines a path-minimized Latent ODE encoder with a gradient-boosted decision tree (GBDT) into a two-stage ECG arrhythmia classification pipeline. On the MIT-BIH dataset, the macro AUC-ROC degrades only marginally from 0.984 at 360 Hz to 0.976 at 45 Hz, demonstrating strong robustness to sampling frequency variation.
tags:
  - "NeurIPS 2025 (Workshop: Learning from Time Series for Health)"
  - "Medical Imaging"
  - "Latent ODE"
  - "ECG Classification"
  - "Arrhythmia"
  - "Wearable Devices"
  - "Sampling Rate Robustness"
date: 2026-05-08
content_hash: 9352f230a794ad31
---

# A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs

**Conference**: NeurIPS 2025 (Workshop: Learning from Time Series for Health)  
**arXiv**: [2511.16933](https://arxiv.org/abs/2511.16933)  
**Code**: None  
**Area**: Medical Imaging / Time Series / ECG Classification  
**Keywords**: Latent ODE, ECG Classification, Arrhythmia, Wearable Devices, Sampling Rate Robustness

## TL;DR

This work combines a path-minimized Latent ODE encoder with a gradient-boosted decision tree (GBDT) into a two-stage ECG arrhythmia classification pipeline. On the MIT-BIH dataset, the macro AUC-ROC degrades only marginally from 0.984 at 360 Hz to 0.976 at 45 Hz, demonstrating strong robustness to sampling frequency variation.

## Background & Motivation

**Background**: Cardiovascular disease is one of the leading causes of death worldwide, and arrhythmia detection relies on electrocardiography (ECG). The clinical standard is the 12-lead ECG, operated by trained professionals, providing high-frequency (>250 Hz) but short-duration "snapshot" monitoring. In recent years, wearable ECG devices (typically single-lead) have proliferated rapidly due to their ability to support long-term, continuous monitoring, making them particularly suitable for capturing intermittent or paroxysmal arrhythmias.

**Limitations of Prior Work**: Wearable devices face a fundamental engineering trade-off — higher sampling rates yield more faithful signals but consume more power and reduce battery life; lower sampling rates extend battery life but lose morphological detail, degrading classification accuracy. Existing deep learning ECG classification models (e.g., the end-to-end CNN by Hannun et al.) are trained and evaluated on high-frequency data, and no systematic study has examined their performance when input is downsampled to 90 Hz or 45 Hz.

**Key Challenge**: A fundamental trade-off exists between signal fidelity (high sampling rate) and device usability (long battery life / small form factor). The central question is whether it is possible to build a classification system that is trained on high-frequency data yet remains effective at lower frequencies.

**Goal**: To construct an end-to-end ECG arrhythmia classification pipeline that is robust to sampling rate variation.

**Key Insight**: Latent ODEs model time series as continuous differential equations in latent space, with an encoder that does not depend on fixed time intervals — inputs are $(x, t)$ pairs rather than uniformly sampled arrays. This means signals at different sampling rates can all be mapped into the same continuous latent space.

**Core Idea**: A Latent ODE encoder maps each ECG beat into a sampling-rate-agnostic continuous latent representation, upon which a lightweight GBDT classifier is trained.

## Method

### Overall Architecture

The input is a single-lead (MLII) ECG beat waveform. The overall pipeline is organized into two independent stages:

**Stage 1 (Representation Learning)**: A path-minimized Latent ODE is trained on the full 360 Hz ECG data in an unsupervised manner to learn a generative model of ECG waveforms. After training, the encoder is frozen.

**Stage 2 (Classification)**: The frozen encoder maps each ECG beat into a 45-dimensional latent vector $\mathbf{z}_0$. SMOTE oversampling is applied to minority classes (S, F) to balance the five categories, and a GBDT is then trained for five-class classification. At inference, $n$ latent vectors are sampled from the encoder using different random seeds for each ECG beat; each is classified independently, and the final prediction is determined by majority voting (mode).

### Key Designs

1. **Path-Minimized Latent ODE**:

    - Function: Encodes ECG time series of arbitrary length and sampling rate into a fixed-dimensional (45-dim) latent vector $\mathbf{z}_0$.
    - Mechanism: The standard Latent ODE uses KL divergence as a regularization term within a VAE framework. This work adopts the improved variant from Sampson & Melchior (2025), replacing KL divergence with $\ell_2$ regularization that minimizes point-to-point distances along individual latent trajectories. The ODE function is parameterized by a 2-layer Tanh network (width 50) and solved using the Tsit5 adaptive-step solver.
    - Design Motivation: Path-minimization regularization has been shown to improve generative fidelity and downstream classifier performance on latent encodings. Furthermore, since the ODE models continuous dynamics, the encoder naturally accepts $(x, t)$ pairs as input, providing intrinsic robustness to sampling rate variation.

2. **GBDT Classifier + SMOTE Oversampling**:

    - Function: Performs five-class arrhythmia classification (N/S/V/F/Q, following the AAMI standard) in the latent space.
    - Mechanism: An ensemble of 1,000 trees with maximum depth 8 is used. SMOTE is applied to the latent vectors prior to training to equalize class sample counts. Excessively deep trees tend to be biased toward the majority class even when training class sizes are equalized.
    - Design Motivation: The decoupled two-stage design allows flexible replacement of the classifier — the GBDT operates solely on latent vectors, and alternative classification algorithms can be explored in future work.

3. **Ensemble Voting Inference**:

    - Function: Improves classification stability for individual ECG beats.
    - Mechanism: Since the Latent ODE encoder is stochastic, $n$ latent vectors $\mathbf{z}_{0,i}$ are sampled for each ECG beat using different random seeds, each independently classified by the GBDT, with the final prediction determined by majority vote (mode).
    - Design Motivation: The stochasticity of the encoder introduces diversity, and the voting mechanism reduces uncertainty from any single sample.

### Loss & Training

- Latent ODE: Reconstruction loss + $\ell_2$ path regularization; trained for 50,000 steps (~2 hours on a single A100).
- GBDT: Trained on SMOTE-balanced data.
- Data split: 70% / 15% / 15% for training / validation / testing.

## Key Experimental Results

### Main Results

Dataset: MIT-BIH Arrhythmia Database — 48 dual-channel recordings (47 subjects), 360 Hz sampling rate, 88,887 beats in total.

| Sampling Rate | Macro Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro AUC-ROC |
|--------|---------------|-----------------|--------------|----------|---------------|
| 360 Hz | 87.0% | 0.85 | 0.87 | 0.86 | **0.984** |
| 90 Hz | 85.9% | 0.84 | 0.85 | 0.85 | 0.978 |
| 45 Hz | 82.9% | 0.82 | 0.83 | 0.82 | 0.976 |

### Ablation Study (Per-Class Performance vs. Sampling Rate)

| Class | # Samples | 360Hz Acc | 90Hz Acc | 45Hz Acc | Drop at 45Hz |
|------|--------|-----------|----------|----------|-------------|
| N (Normal) | 10,988 | 98.0% | 97.7% | 97.7% | −0.3% |
| V (Ventricular) | 918 | 93.9% | 93.6% | 92.6% | −1.3% |
| Q (Unknown) | 1,001 | 95.4% | 94.3% | 93.3% | −2.1% |
| S (Supraventricular) | 327 | 75.2% | 70.5% | 70.1% | −5.1% |
| F (Fusion) | 99 | 72.3% | 69.6% | 60.8% | −11.5% |

### Key Findings

- Majority classes (N, V, Q) are nearly unaffected by 8× downsampling, suggesting that the Latent ODE's continuous modeling captures core waveform structure rather than relying on high-frequency detail.
- Degradation in minority classes (S and F) stems primarily from class imbalance — too few training samples (F has only 99 examples), with confusion matrices showing these classes are predominantly misclassified as N.
- UMAP visualization reveals clear cluster structure for all five classes in the latent space, confirming that the encodings are semantically meaningful.
- The gap between AUC-ROC (0.984→0.976) and macro F1 (0.86→0.82) further reflects the impact of class imbalance.

## Highlights & Insights

- **Intrinsic Advantage of Continuous-Time Modeling**: ODEs describe continuous dynamics rather than discrete sequences; the encoder accepts $(x, t)$ pairs at any sampling rate — making cross-sampling-rate generalization an inherent property of the model architecture rather than an additional design choice. This principle is transferable to any time series task involving irregular sampling.
- **Flexibility of the Decoupled Design**: The two-stage pipeline of learning representations first and classifying second allows the Latent ODE and the classifier to be optimized independently. In practice, the classifier can be swapped (random forest, SVM, lightweight MLP) or the Latent ODE can be retrained on a larger dataset without modifying the downstream stage.
- **Clear Practical Value**: The work directly addresses an engineering question — wearable ECG devices can operate at sampling rates as low as 45 Hz while maintaining an AUC-ROC of 0.976, implying smaller sensors, longer battery life, and improved wearability.

## Limitations & Future Work

- **Dataset Too Small and Outdated**: MIT-BIH covers only 47 subjects, is over 20 years old, is not a wearable dataset, and exhibits severe class imbalance (F has only 99 samples). Validation on larger, more modern datasets (e.g., PhysioNet 2017, CPSC 2018) is urgently needed.
- **Overly Simplified Downsampling**: Naive downsampling by taking every $n$-th sample is used, without accounting for aliasing noise, motion artifacts, or baseline wander present in real low-frequency wearable signals. Actual degradation patterns in wearable signals are far more complex than uniform subsampling.
- **Absence of Edge-Device Validation**: Inference latency and memory footprint are not reported, and feasibility on actual embedded hardware is not demonstrated.
- **Missing Comparison with End-to-End Deep Learning**: No comparison is made against CNN- or Transformer-based end-to-end ECG classification methods under the same conditions, making it impossible to determine whether the advantage of the Latent ODE stems from continuous modeling or the two-stage pipeline itself.
- **Validity of SMOTE in Latent Space**: The appropriateness of applying SMOTE directly to latent vectors is questionable, as synthetic samples may deviate from the true data manifold.

## Related Work & Insights

- **vs. Hannun et al. (Nature Medicine 2019)**: An end-to-end CNN achieves cardiologist-level performance on 12-lead high-frequency ECG but relies entirely on high-frequency data and does not evaluate cross-sampling-rate generalization.
- **vs. Rubanova et al. (NeurIPS 2019)**: Introduced the original Latent ODE framework; the present work extends its application from generative/interpolation tasks to classification feature extraction.
- **vs. Sampson & Melchior (2025)**: Proposed the path-minimized Latent ODE; the present work directly adopts their improvement and validates its effectiveness in the ECG domain.
- **Extensible Directions**: The decoupled framework of Latent ODE encoder + classifier can be generalized to cross-device / cross-sampling-rate classification of other physiological signals such as PPG (photoplethysmography) and EEG (electroencephalography).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of Latent ODE and ECG classification is creative, but all core components (Latent ODE, GBDT, SMOTE) are existing methods.
- Experimental Thoroughness: ⭐⭐⭐ Only one small dataset is used; comparisons with mainstream methods and real-device testing are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The short-paper format is compact and well-structured, with intuitive algorithmic pseudocode.
- Value: ⭐⭐⭐⭐ The approach has practical value but requires more thorough validation; the direction is sound as a preliminary exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](../../ICML2025/medical_imaging/from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [\[CVPR 2025\] Novel Architecture of RPA In Oral Cancer Lesion Detection](../../CVPR2025/medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)
- [\[ICCV 2025\] Controllable Latent Space Augmentation for Digital Pathology](../../ICCV2025/medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)
- [\[ICML 2025\] Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)](../../ICML2025/medical_imaging/boosting_masked_ecg-text_auto-encoders_as_discriminative_learners.md)

</div>

<!-- RELATED:END -->
