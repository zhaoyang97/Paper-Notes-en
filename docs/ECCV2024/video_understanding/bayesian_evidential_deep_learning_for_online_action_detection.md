---
title: >-
  [Paper Note] Bayesian Evidential Deep Learning for Online Action Detection
description: >-
  [ECCV 2024][Video Understanding][Online Action Detection] This paper proposes the BEDL (Bayesian Evidential Deep Learning) framework. Incorporating a Bayesian teacher-evidential student architecture, it achieves accurate and efficient inference as well as reliable uncertainty quantification in online action detection tasks. Furthermore, it designs a attention module based on Bayesian mutual information for active feature selection.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Online Action Detection"
  - "Bayesian Neural Network"
  - "Evidential Deep Learning"
  - "Uncertainty Quantization"
  - "Teacher-Student Architecture"
date: 2026-05-08
content_hash: 8f0ff46648d16521
---

# Bayesian Evidential Deep Learning for Online Action Detection

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Video Understanding / Action Detection  
**Keywords**: Online Action Detection, Bayesian Neural Network, Evidential Deep Learning, Uncertainty Quantization, Teacher-Student Architecture

## TL;DR

This paper proposes the BEDL (Bayesian Evidential Deep Learning) framework. Incorporating a Bayesian teacher-evidential student architecture, it achieves accurate and efficient inference as well as reliable uncertainty quantification in online action detection tasks. Furthermore, it designs a attention module based on Bayesian mutual information for active feature selection.

## Background & Motivation

**Background**: Online Action Detection (OAD) aims to recognize ongoing actions in streaming videos in real time, without utilizing future frame information. This task holds significant application value in real-time scenarios such as autonomous driving, video surveillance, and human-computer interaction. Existing methods primarily focus on improving detection accuracy, employing architectures like RNNs and Transformers for temporal modeling of historical frames.

**Limitations of Prior Work**: Existing OAD methods suffer from two key limitations. First, they typically produce deterministic predictions and fail to quantify prediction uncertainty. In safety-critical scenarios (such as autonomous driving), knowing "how uncertain" the model is is as crucial as knowing "what" the model predicts. Second, existing uncertainty quantification methods—namely Bayesian Neural Networks (BNNs)—require multiple forward passes for sampling, which incurs high computational overhead and is unsuitable for real-time applications. Conversely, Evidential Deep Learning (EDL) can estimate uncertainty with a single forward pass, but its estimation accuracy is inferior to that of BNNs.

**Key Challenge**: BNNs provide accurate uncertainty estimates but suffer from slow inference, while EDL offers fast inference but less accurate uncertainty quantification. For OAD, which demands real-time inference and safety-critical uncertainty representation, balancing the strengths of both remains a significant challenge.

**Goal**: (1) How to efficiently and accurately quantify uncertainty in OAD? (2) How to leverage uncertainty information to enhance detection performance? (3) How to design a unified framework that supports both real-time inference and reliable uncertainty estimation?

**Key Insight**: The authors propose combining BNNs and EDL using a teacher-student architecture: a BNN acts as a teacher model to provide high-quality uncertainty signals, which the student model learns to replicate via evidential learning in a single forward pass. This leverages BNN's strength during training while relying on a lightweight student model during inference.

**Core Idea**: Guiding the evidential student network using the mutual information and distributional knowledge from the Bayesian teacher network enables the student model to perform both accurate and efficient online action detection and uncertainty quantification in a single forward pass.

## Method

### Overall Architecture

BEDL adopts a teacher-student architecture. The teacher model is a Bayesian Neural Network implemented via Monte Carlo Dropout or variational inference, obtaining accurate uncertainty estimates (including epistemic and aleatoric uncertainty) through multiple forward passes. The student model is an evidential deep learning network that outputs the parameters of a Dirichlet distribution ("evidence") to estimate uncertainty in a single forward pass. During training, the teacher model transfers its mutual information and predictive distribution to the student; during inference, only the student model is utilized, enabling real-time online action detection.

### Key Designs

1. **Bayesian Teacher Model**:

    - Function: Provides high-quality uncertainty signals as training targets for the student model.
    - Mechanism: Introduction of Bayesian treatment (e.g., MC Dropout) into a standard OAD network to represent weights via posterior distributions. By drawing $T$ forward samples, the predictive distribution $\{p_1, p_2, ..., p_T\}$ is obtained, and the following are calculated: (a) mean prediction $\bar{p} = \frac{1}{T}\sum_i p_i$ as the final prediction; (b) mutual information $I(y; \omega | x) = H[\bar{p}] - \frac{1}{T}\sum_i H[p_i]$ to measure epistemic uncertainty (model uncertainty); (c) $\frac{1}{T}\sum_i H[p_i]$ to measure aleatoric uncertainty (data uncertainty).
    - Design Motivation: BNNs are the "gold standard" for uncertainty quantification, but multiple sampling passes are impractical in real-time scenarios. Using it as a teacher allows high-quality uncertainty targets to be obtained offline.

2. **Evidential Student Model**:

    - Function: Performs both action detection and uncertainty estimation in a single forward pass.
    - Mechanism: The student model's final layer outputs the concentration parameters (evidence) $\alpha = [\alpha_1, ..., \alpha_K]$ of a Dirichlet distribution, where $K$ is the number of action categories. The Dirichlet distribution, as a conjugate prior to the multinomial distribution, naturally represents the "uncertainty of predictions". By distilling knowledge from the teacher: (a) minimizing the KL divergence between the student's predictive distribution and the teacher's average prediction; (b) using an additional loss function to align the student's estimated mutual information with that computed by the teacher. Consequently, the student model learns both action prediction and uncertainty estimation.
    - Design Motivation: EDL offers single-pass uncertainty estimation, but training EDL directly yields subpar uncertainty quality. Synthesizing high-quality uncertainty signals from the BNN teacher into the EDL student enhances the accuracy of uncertainty estimation.

3. **MI-based Active Feature Selection Attention**:

    - Function: Actively selects the most valuable features for prediction using uncertainty information to boost OAD performance.
    - Mechanism: Conventional OAD methods utilize features from all historical frames, yet not all frames are equally important for the current prediction. This module calculates the "information value" of each step's features using Bayesian mutual information—features with high mutual information correspond to regions where the model is uncertain and require focused attention. Feature aggregation is weighted by attention coefficients, focusing the model on the most critical details. This acts as an "active" detection strategy where the model decides what to attend to based on its own uncertainty.
    - Design Motivation: In streaming videos, redundant or noisy frames can interfere with detection. Leveraging uncertainty signals dynamically filters out irrelevant information, improving performance without introducing extra computation. Unlike traditional attention mechanisms, these attention weights have a sound probabilistic foundation.

### Loss & Training

The overall loss comprises three components: (1) evidential loss $L_{edl}$, derived from the negative log-likelihood of the Dirichlet distribution and a regularization term, which ensures accurate predictions and penalizes incorrect predictions with low evidence; (2) distribution distillation loss $L_{dist}$, which minimizes the KL divergence between the student's predictive distribution and the teacher's average predictive distribution; and (3) mutual information distillation loss $L_{mi}$, which aligns the student's estimated mutual information with that computed by the teacher. Training is performed in two stages: first training the teacher BNN model, and then training student using the teacher's outputs. During inference, only the student model is required.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (BEDL) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| THUMOS'14 | mAP (%) | Competitive | OadTR, LSTR | Additional uncertainty quantification provided |
| TVSeries | mcAP (%) | Competitive | Colar, GateHUB | Higher inference efficiency than BNN |
| HDD | mAP (%) | Competitive | Existing OAD methods | More accurate uncertainty estimation |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| EDL only (No teacher) | Drop in accuracy, poor uncertainty calibration | Insufficient uncertainty quality from direct EDL training |
| BNN only (Multi-sampling) | Highest accuracy, but slow inference | Standard BNN is unsuitable for real-time OAD |
| BEDL (Teacher + Student) | Accuracy close to BNN, fast inference | Successfully balances accuracy and efficiency |
| Without MI Attention Module | Slight drop in accuracy | Active feature selection effectively boosts performance |
| Different sampling passes T | Larger T yields a more accurate teacher | Good performance obtained with T=10-20 |

### Key Findings

- The quality of uncertainty estimation in BEDL is significantly superior to raw EDL, approaching BNN levels.
- In online anomaly detection experiments, the uncertainty produced by BEDL effectively distinguishes between normal and anomalous/unknown actions.
- Epistemic uncertainty (mutual information) is more effective than aleatoric uncertainty for out-of-distribution (OOD) sample detection.
- The MI attention module, by focusing on areas of high uncertainty, improves mAP by 0.5-1.5% without increasing inference overhead.

## Highlights & Insights

1. **Theoretical Elegance**: Ingeniously fuses BNN and EDL using a teacher-student framework, successfully combining the strengths of both.
2. **Uncertainty in OAD**: Systematically introduces uncertainty quantification to online action detection for the first time, opening up a new research direction.
3. **Active Detection Paradigm**: Uncertainty-based feature selection represents a novel "active perception" strategy, offering strong cognitive inspiration.
4. **Out-of-the-box**: The framework is decoupled from the specific OAD backbone, allowing it to act as a plug-and-play module to enhance any OAD method.

## Limitations & Future Work

1. The teacher model still requires multi-sampling BNN training, leading to relatively high training costs.
2. Experiments were conducted only on RGB features; multimodal (RGB + optical flow + audio) scenarios warrant further exploration.
3. The calculation of mutual information depends on the number of classification categories, which may pose challenges in extremely large-scale class scenarios.
4. The calibration quality of uncertainty still has room for improvement, and more advanced calibration techniques could be integrated.
5. Temporal modeling capabilities in long videos are limited by fixed window sizes; combining the framework with Transformers could be considered.

## Related Work & Insights

- **OadTR / LSTR**: Transformer-based online action detection methods that focus on detection accuracy without addressing uncertainty.
- **Evidential Deep Learning (Sensoy et al.)**: The foundational work of EDL, with this paper further enhancing its uncertainty estimation quality.
- **MC Dropout (Gal & Ghahramani)**: A classic approximate Bayesian inference method utilized to construct the teacher model.
- **Deep Evidential Regression (Amini et al.)**: Pioneering work extending evidential learning to regression tasks.
- The concept of uncertainty distillation within a teacher-student framework can be extended to other real-time perception tasks (e.g., real-time object detection, autonomous driving decision-making).

## Rating

- Novelty: ⭐⭐⭐⭐ The BNN+EDL teacher-student fusion framework is novel, and the MI attention module is highly creative.
- Experimental Thoroughness: ⭐⭐⭐ Thorough experiments on three standard datasets, with ablation studies validating each component and convincing extra anomaly detection experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations with rigorous logical motivation.
- Value: ⭐⭐⭐⭐ Uncertainty quantification has crucial practical value in safety-critical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[CVPR 2025\] Context-Enhanced Memory-Refined Transformer for Online Action Detection](../../CVPR2025/video_understanding/context-enhanced_memory-refined_transformer_for_online_action_detection.md)
- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)

</div>

<!-- RELATED:END -->
