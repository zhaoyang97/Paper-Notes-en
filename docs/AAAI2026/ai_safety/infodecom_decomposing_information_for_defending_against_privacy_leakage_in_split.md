---
title: >-
  [Paper Note] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference
description: >-
  [AAAI 2026][AI Safety][Split inference] InfoDecom is proposed to reduce redundant information in smashed data via two-level information decomposition (frequency-domain visual information removal + mutual information supp…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Split inference"
  - "data reconstruction attack"
  - "privacy protection"
  - "information decomposition"
  - "frequency domain transformation"
date: 2026-05-08
content_hash: a4daefd5a21ff2f7
---

# InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference

**Conference**: AAAI 2026
**arXiv**: [2511.13365](https://arxiv.org/abs/2511.13365)
**Code**: [github.com/SASA-cloud/InfoDecom](https://github.com/SASA-cloud/InfoDecom)
**Area**: AI Security
**Keywords**: Split inference, data reconstruction attack, privacy protection, information decomposition, frequency domain transformation

## TL;DR

InfoDecom is proposed to reduce redundant information in smashed data via two-level information decomposition (frequency-domain visual information removal + mutual information suppression), followed by closed-form Gaussian noise injection for theoretical privacy guarantees, achieving a significantly superior utility-privacy trade-off over existing methods under shallow client models.

## Background & Motivation

**Split Inference (SI)** partitions a DNN into a shallow client-side model (bottom model) and a server-side model (top model), where the client transmits only intermediate representations (smashed data) to the server. However, data reconstruction attacks (DRA) can recover the original input from smashed data, resulting in severe privacy leakage.

**Two existing defense paradigms and their limitations**:

**Regularization-based methods** (Shredder, Nopeek, InfoScissors, etc.): perturb smashed data via heuristic optimization objectives (e.g., mutual information upper bounds, distance correlation)
   - Limitation: no rigorous provable privacy guarantees

**Closed-form noise computation** (dFIL, FSInfo, etc.): compute noise scale satisfying a given privacy budget based on Fisher Information or conditional entropy
   - Limitation: when the bottom model is shallow (common in resource-constrained devices), smashed data retains substantial input information → large noise is required to satisfy privacy constraints → severe task performance degradation

**Core Insight**: The poor utility-privacy trade-off (UPT) of existing defenses stems from wasting perturbation budget on the large volume of **task-irrelevant redundant information** in smashed data.

**Mechanism**: First decompose and remove redundant information to reduce the amount of sensitive content requiring protection → less noise needed for the same privacy guarantee → less performance degradation → better UPT.

## Method

### Overall Architecture

InfoDecom consists of three stages:

1. **Visual Information Removal**: frequency-domain transformation → discarding low-frequency components essential for human visual perception
2. **Mutual Information Suppression**: regularizing the bottom model via IB principles → retaining task-relevant information while suppressing task-irrelevant information
3. **Noise Perturbation**: closed-form FSInfo-guided Gaussian noise → theoretical privacy guarantee

### Key Designs

1. **Visual Information Removal (Frequency-Domain Decomposition)**

   **Inspiration from the communication trinity**: syntactic communication (transmit all bits) → semantic communication (transmit meaning) → pragmatic communication (transmit task-relevant contributions). Feeding raw images directly constitutes syntactic communication, containing substantial redundancy.

   **Processing pipeline**:
    - RGB → YUV color space
    - Each channel divided into 8×8 blocks
    - Forward DCT (Discrete Cosine Transform) → 64 frequency coefficients
    - Remove the $K$ DCT coefficients with the highest amplitude (low-frequency components $X_l$)
    - Only the high-frequency coefficients $X_h$ are passed to the DNN

   **Design Motivation**: JPEG compression theory establishes that low-frequency components are critical for human visual perception (containing primary visual information), while experiments from DuetFace demonstrate that high-frequency components still contain sufficient semantic information for DNN classification. Therefore, discarding low-frequency components hides most human-perceptible private information while preserving the semantic information required by the DNN.

2. **Mutual Information Suppression (IB-Based)**

   Although partial visual information is removed, the remaining high-frequency components $X_h$ may still contain privacy-sensitive information exploitable by DRA.

   **Optimization objective**: $\min_Z \lambda I(X_h; Z) - I(Y; Z)$

   **(a) Minimizing $I(X_h; Z)$** — Clustering Loss:

   Drawing on the CLUB (MI upper bound) framework, a clustering loss is designed to entangle smashed data from different inputs, reducing their distinguishability:
    $\mathcal{L}_{cl} = \frac{1}{N} \sum_{i=1}^{N} \|z_i - z_j\|_2^2$
   where $j$ is sampled uniformly from $\{1, ..., N\}$.

   **Design Motivation**: Pushing different smashed data representations closer together → the conditional distribution $p(Z|X)$ becomes more ambiguous → attackers find it harder to reconstruct the original input from smashed data.

   **(b) Minimizing $-I(Y; Z)$** — Cross-Entropy Loss:

   Replaced by the Barber-Agakov lower bound, ultimately simplified to standard cross-entropy:
    $\mathcal{L}_{ce} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_i^{(k)} \log(f_{\theta_2}(z_i))^{(k)}$

3. **Closed-Form Noise Perturbation (Theoretical Privacy Guarantee)**

   The FSInfo privacy metric is used to compute the Gaussian noise scale:
    $\tilde{Z} = Z + \delta, \quad \delta \sim \mathcal{N}\left(0, \frac{\det(J^T J)^{\frac{1}{2d}}}{e^{FSInfo}(2\pi e)^{\frac{1}{2}}}\right)$

   where $J$ is the Jacobian of $Z$ with respect to the original input $X$. A lower FSInfo value (e.g., -1) implies less privacy leakage.

   **Key Innovation**: Since the preceding two levels have already removed redundant information, the amount of content requiring protection in smashed data is substantially reduced → smaller noise scale is needed to achieve the same FSInfo level → less performance degradation.

### Loss & Training

$$\mathcal{L} = \lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$$

- Top model is optimized by $\mathcal{L}_{ce}$
- Bottom model is optimized by $\lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$
- At inference: high-frequency input → updated bottom model → regularized smashed data → noise addition → transmitted to server

**Hyperparameters**:
- Adam optimizer, lr = 3e-4, weight decay = 0.01
- 150 epochs global training, batch size = 128
- Default $|X_h| = 54$ (retaining 54/64 frequency coefficients), $\lambda = 10$, FSInfo = -1
- CIFAR-10: 2× RTX 4090; CelebA: 4× A100

## Key Experimental Results

### Main Results: Utility-Privacy Trade-off Comparison

On CIFAR-10 and CelebA using ResNet-18, with the split point at the C64 layer (shallow model):

| Method | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|------|:---:|:---:|:---:|:---:|
| Raw (no defense) | High | Low (privacy leakage) | High | Low |
| Nopeek | Medium | Medium | Medium | Medium |
| Shredder | Medium | Medium | Medium | Medium |
| inv_dFIL_def | Low | High | Medium | Medium |
| FSInfoGuard | Medium | Medium | Medium | Medium |
| **InfoDecom** | **0.7329** | **0.0843** | **0.9693** | **0.1942** |

InfoDecom achieves the best trade-off in the utility-privacy plane (curve lying to the upper-right of all other methods).

### Ablation Study

| Configuration | CIFAR-10 Acc. | CIFAR-10 MSE | Notes |
|------|:---:|:---:|------|
| **InfoDecom (full)** | **0.7329** | **0.0843** | Default setting |
| w/o Visual Information Removal | 0.6273 | 0.0849 | Acc drops due to larger noise required to satisfy FSInfo=-1 |
| w/o $\mathcal{L}_{cl}$ | 0.7453 | 0.0835 | Acc slightly higher but MSE lower (weakened defense) |
| w/o FSInfo noise | 0.7274 | 0.0826 | Loss of theoretical privacy guarantee |

### Effect of Information Controller Parameters

**Number of retained coefficients $|X_h|$** (FSInfo=-1, λ=10):

| $|X_h|$ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---------|:---:|:---:|:---:|:---:|
| 54 | 0.7329 | 0.0843 | 0.9693 | 0.1984 |
| 41 | 0.6905 | 0.1497 | 0.8036 | 0.3273 |
| 32 | 0.3645 | 0.2337 | 0.6135 | 1.1024 |
| 18 | 0.1004 | 0.2492 | 0.6135 | 1.1022 |

**Weight factor λ** ($|X_h|$=54, FSInfo=-1):

| λ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---|:---:|:---:|:---:|:---:|
| 1 | 0.7570 | 0.0822 | 0.9515 | 0.1925 |
| 10 | 0.7329 | 0.0843 | 0.9693 | 0.1942 |
| 20 | 0.7250 | 0.0854 | 0.8997 | 0.1950 |

### Key Findings

1. **Information decomposition is the key to UPT improvement**: at the same privacy level, InfoDecom achieves substantially higher accuracy than direct noise injection methods
2. **Visual information removal is indispensable**: removing it causes accuracy to drop from 0.7329 to 0.6273 (due to the larger noise required to satisfy FSInfo=-1)
3. **Subtle role of mutual information suppression**: removing it slightly increases accuracy but weakens privacy defense → regularization does compress content beyond task-relevant information
4. **Superior performance on CelebA**: in the binary classification task (attractiveness classification), InfoDecom achieves 96.93% accuracy with MSE=0.1942
5. **Acceptable computational overhead**: InfoDecom requires 6.64ms per sample at inference (vs. 0.26ms for basic forward pass), comparable to other Jacobian-based methods

## Highlights & Insights

1. **Innovation of the "decompose-then-denoise" paradigm**: rather than directly adding noise to protect all information, redundant content is first removed before protecting the essential content → conceptually simple yet highly effective
2. **Elegant application of frequency-domain processing**: borrowing from JPEG compression theory, discarding low-frequency visual information → the DNN can complete tasks using high-frequency information while human observers cannot recognize the content
3. **Three-level controller design**: $|X_h|$ (degree of visual redundancy), λ (degree of semantic redundancy), and FSInfo (privacy guarantee level) provide flexible trade-off tuning
4. **Theoretical guarantees combined with practicality**: FSInfo provides a provable privacy lower bound, while the two-level information decomposition ensures that noise does not become excessively large

## Limitations & Future Work

- Currently applicable only to vision tasks (visual information removal relies on frequency-domain transformation) → the authors note that the MIS and NP components are modality-agnostic
- Computational overhead is higher than non-Jacobian methods (6.64ms vs. <1ms) → more efficient Jacobian approximation strategies warrant exploration
- Validation limited to two datasets (CIFAR-10, CelebA), without coverage of high-resolution or more complex vision tasks
- Split point fixed at the first (shallowest) layer → the effect of different split points is not thoroughly investigated
- Only one DRA method (invNet) is evaluated → robustness against more advanced attacks (e.g., GAN-based DRA) remains to be verified

## Related Work & Insights

- **Distinction from regularization-based methods**: Shredder/Nopeek/InfoScissors rely solely on optimization objectives without theoretical guarantees; InfoDecom first reduces the amount of information to protect and then applies guaranteed noise
- **Distinction from closed-form noise methods**: dFIL/FSInfoGuard compute noise scale directly but produce excessively large noise under shallow models; InfoDecom reduces the required noise scale by decomposing redundant information
- **Practical instantiation of Information Bottleneck (IB) principles**: translates the IB theoretical framework into practically trainable clustering loss and cross-entropy loss
- **Implication for privacy-preserving ML**: "reducing the amount of information requiring protection" is more effective than "adding more noise"

## Rating

- Novelty: ⭐⭐⭐⭐ (the "decompose-then-denoise" concept is intuitive and effective; the combination of frequency domain + IB + closed-form noise is novel)
- Experimental Thoroughness: ⭐⭐⭐ (two datasets + complete parameter sensitivity analysis, but limited scenario coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear motivation, rigorous mathematical derivations, strong logical coherence across the three-level decomposition)
- Value: ⭐⭐⭐⭐ (addresses the core pain point of privacy-utility trade-off under shallow client models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[AAAI 2026\] An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlearning.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

</div>

<!-- RELATED:END -->
