---
title: >-
  [Paper Note] "DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders"
description: >-
  [CVPR2025][AI Safety][Paper Note] Academic paper note for "DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders".
tags:
  - CVPR2025
  - AI Safety
date: 2024-11-25
content_hash: 385b012788bddff2
---
# DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders

**Authors**: Yuwen Pu, Yue Zheng, Shiji Zhao, et al.  
**Institutions**: HKUST / Southeast University  
**Conference**: CVPR 2025  
**Code**: [https://github.com/tardisblue9/DeDe](https://github.com/tardisblue9/DeDe)  

## Background & Motivation

Self-Supervised Learning (SSL) pre-trained encoders have become the infrastructure of modern computer vision. However, SSL encoders face severe backdoor attack threats:

**Stealthiness of SSL backdoor attacks**: Unlike supervised learning, SSL backdoor attacks implant a backdoor mapping into the feature space of the encoder. The attacker binds a specific trigger pattern to the feature space of the target class. When downstream users employ the encoder, inputs containing the trigger pattern are mapped to the feature region designated by the attacker.

**High detection difficulty**:
   - SSL encoders lack a classification head, rendering traditional backdoor detection methods based on classification outputs (such as Neural Cleanse) inapplicable.
   - Encoders are typically provided as a black box, and users cannot access the internal parameters of the encoder.
   - Trigger patterns can be extremely small or stealthy (e.g., changes in a few pixels or invisible perturbations).

**Limitations of existing defenses**:
   - Data purification methods struggle to work in unlabeled scenarios.
   - Model pruning methods require direct modification of encoder parameters.
   - Activation analysis-based methods exhibit high computational overhead and false positive rates.

**Supply chain security risks**: With the popularization of pre-trained model sharing platforms (such as HuggingFace), users increasingly download and use third-party pre-trained encoders, making the supply chain risk of backdoor attacks increasingly severe.

DeDe proposes a new approach: detecting backdoors by training a decoder to invert the encoder's mapping, leveraging the inconsistency in decoded reconstruction of backdoor samples.

## Method

### Core Intuition

Behavioral differences of the backdoor encoder on clean samples and trigger samples:

| Input Type | Encoder Behavior | Decoder Reconstruction |
|----------|-----------|-----------|
| Clean Sample | Normal feature encoding | Faithful reconstruction of the original image |
| Trigger Sample | Features distorted to the target region | Reconstructed image inconsistent with the original image |

This inconsistency serves as the signal to detect backdoor samples.

### Decoder Training

**Training Strategy**:
- Use clean data (without trigger patterns) to train the decoder $D$.
- The decoder learns to invert the mapping of the encoder $E$: $\hat{x} = D(E(x))$.
- Training loss: $\mathcal{L}_{recon} = \| x - D(E(x)) \|_2^2$

**Key Design — High Masking Ratio Training**:
- Training employs a **masking ratio = 0.9** (masking 90% of image patches).
- This forces the decoder to heavily rely on the feature information provided by the encoder to reconstruct the masked regions.
- If the encoder's features are distorted by the backdoor, the reconstruction quality of the decoder will degrade drastically.

### Backdoor Detection

**Inference Masking Ratio**:
- Testing employs an even higher **masking ratio = 0.99** (retaining only 1% of patches).
- Extreme masking amplifies the reconstruction inconsistency of backdoor samples.

**Detection Metric** — Reconstruction Inconsistency Score:

$$s(x) = \| x - D(E(M(x))) \|_2^2$$

where $M(\cdot)$ represents the random masking operation.

**Detection Threshold**:
- Use a small-scale clean validation set to estimate the distribution of normal reconstruction errors.
- Set the threshold $\tau = \mu + k\sigma$, where $\mu$ and $\sigma$ are the mean and standard deviation of clean sample reconstruction errors.
- A sample is determined to be a backdoor sample if $s(x) > \tau$.

### Detection Pipeline

```
Input image x → Random masking M(x) → Encoder E(M(x)) → Decoder D(·) → Reconstruction x̂
                                                                               ↓
                                 Compute ||x - x̂||² → Compare with threshold τ → Clean/Backdoor
```

## Key Experimental Results

### Backdoor Attack Detection Rate

| Attack Method | Attack Success Rate (ASR) | DeDe TPR ↑ | DeDe FPR ↓ |
|----------|----------------|-----------|-----------|
| BadEncoder | 99.9% | **93.1%** | 3.2% |
| CTRL | 97.8% | 89.5% | 4.1% |
| CLIP Backdoor | 98.5% | **100.0%** | 2.7% |
| PoisonedEncoder | 96.3% | 87.8% | 5.3% |

### Backdoor Mitigation Performance

| Attack Method | Original ASR | ASR after DeDe Filtering ↓ | Clean Accuracy Change |
|----------|---------|-------------------|--------------|
| BadEncoder | 99.9% | **1.3%** | -0.8% |
| CTRL | 97.8% | 3.7% | -1.2% |
| CLIP Backdoor | 98.5% | **0.5%** | -0.5% |
| PoisonedEncoder | 96.3% | 4.2% | -1.5% |

### Comparison with Existing Methods

| Method | BadEncoder TPR | CLIP Backdoor TPR | Requires Encoder Parameters |
|------|---------------|-------------------|--------------|
| Neural Cleanse | N/A | N/A | Yes |
| Activation Clustering | 52.3% | 61.7% | Yes |
| STRIP | 67.8% | 73.2% | No |
| SentiNet | 71.5% | 78.4% | No |
| **DeDe (ours)** | **93.1%** | **100.0%** | **No**|

### Ablation Study

| Configuration | BadEncoder TPR | FPR |
|------|---------------|-----|
| Masking 0.5 train / 0.75 test | 72.3% | 8.1% |
| Masking 0.75 train / 0.9 test | 84.6% | 5.7% |
| Masking 0.9 train / 0.95 test | 89.2% | 4.3% |
| **Masking 0.9 train / 0.99 test** | **93.1%** | **3.2%** |

## Highlights & Insights

1. **Decoder Detection Paradigm**: Clarifying the decoder-based detection approach for SSL encoders for the first time — leveraging the inconsistency of backdoor samples in the encoding-decoding process as the detection signal.
2. **Extreme Masking Strategy**: A masking ratio design of 0.9 during training and 0.99 during inference, maximizing the amplification of backdoor feature distortion on reconstruction quality.
3. **Black-box Detection**: No access to the encoder's internal parameters is required; backdoors can be detected using only its output features.
4. **Significant Mitigation Performance**: BadEncoder ASR drops from 99.9% to 1.3%, while clean accuracy only decreases by 0.8%.

## Theoretical Analysis

The authors explain the mechanism of DeDe from an information-theoretic perspective:

- The backdoor encoder performs information compression on trigger samples: discarding original content information and injecting target class information.
- The decoder relies on the information provided by the encoder for reconstruction. When the original content information is replaced by the backdoor, the reconstruction inevitably fails.
- A high masking ratio further increases the decoder's reliance on encoder features, amplifying the detection signal.

## Limitations & Future Work

- A certain amount of clean data is required to train the decoder and estimate the detection threshold.
- The robustness against adaptive attacks (where the attacker knows the existence of DeDe) needs further validation.
- In multi-trigger attack scenarios, a single threshold may not be flexible enough.
- Training the decoder incurs additional computational cost.

## Related Work & Insights

- BadEncoder: The first backdoor attack targeting SSL encoders.
- CTRL: A contrastive learning-based backdoor attack.
- Neural Cleanse: A classic backdoor detection method (requires a classification head).
- MAE: Masked Autoencoders, which inspired DeDe's high masking ratio design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[CVPR 2025\] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->
