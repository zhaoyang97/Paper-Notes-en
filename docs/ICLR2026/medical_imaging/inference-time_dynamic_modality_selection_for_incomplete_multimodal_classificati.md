---
title: >-
  [Paper Note] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification
description: >-
  [ICLR 2026][Medical Imaging][incomplete multimodal] This paper proposes DyMo, an inference-time dynamic modality selection framework that derives a theoretically grounded MTIR reward function (based on a classification-l…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "incomplete multimodal"
  - "dynamic modality selection"
  - "inference-time"
  - "information gain"
  - "discarding-imputation dilemma"
date: 2026-05-08
content_hash: ae904df1969e7702
---

# Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification

**Conference**: ICLR 2026
**arXiv**: [2601.22853](https://arxiv.org/abs/2601.22853)  
**Code**: [GitHub](https://github.com/siyi-wind/DyMo)  
**Area**: Multimodal Learning / Medical Imaging
**Keywords**: incomplete multimodal, dynamic modality selection, inference-time, information gain, discarding-imputation dilemma

## TL;DR

This paper proposes DyMo, an inference-time dynamic modality selection framework that derives a theoretically grounded MTIR reward function (based on a classification-loss-reduction proxy + class prototype distance + intra-class similarity calibration) to iteratively and selectively fuse reliable recovered modalities at inference time, offering the first systematic resolution of the discarding-imputation dilemma: discarding missing modalities loses task-relevant information, while imputation may introduce noise.

## Background & Motivation

**Background**: Multimodal deep learning has achieved significant progress in healthcare, marketing, and embodied intelligence, yet deployed systems frequently encounter samples with one or more missing modalities due to sensor failures, heterogeneous acquisition protocols, or transmission errors.

**Limitations of Prior Work**:

1. **Imputation-based methods** (e.g., MoPoE, M3Care) reconstruct missing modalities via VAEs, but reconstruction quality is inconsistent—generated outputs may be low-fidelity (blurry/distorted) or semantically misaligned (reconstructed content belongs to a different class than the input).

2. **Discarding-based methods** (e.g., ModDrop, MUSE) simply ignore missing modalities, but when highly task-relevant modalities are absent, the discriminability of the remaining features degrades substantially.

3. **Existing dynamic fusion methods** (QMF, DynMM, PDF) primarily address intra-modal noise (low fidelity) and cannot detect inter-modal semantic misalignment.

**Key Challenge** (discarding-imputation dilemma): Discarding missing modalities sacrifices task-relevant information → performance drops; imputing missing modalities may introduce task-irrelevant noise or semantic errors → performance also drops. Both strategies have inherent shortcomings, and no mechanism exists to dynamically trade off between them.

**Key Insight**: Rather than choosing one strategy, DyMo dynamically evaluates whether each recovered modality is "worth fusing"—accepting it if recovery increases task-relevant information (positive reward) and rejecting it if recovery introduces noise or misalignment (negative reward).

## Method

### Overall Architecture

Input: incomplete multimodal sample $\mathbb{X} = \{x^{(m)}\}_{m \in \mathcal{I}}$ → recovery method $\Upsilon$ (e.g., VAE/TIP) reconstructs missing modalities → DyMo dynamic selection algorithm iteratively evaluates the MTIR reward for each recovered modality → only modalities with positive reward are fused → multimodal Transformer network $f$ produces predictions.

The network $f$ consists of modality-specific encoders $h^{(m)}$, a multimodal Transformer $\psi$ (with a [CLS] token and attention masking for missing modalities), and a linear softmax classifier $\zeta$.

### Key Designs

1. **Multimodal Task-relevant Information Reward (MTIR)**

    - Theoretical basis: a lower-bound relationship between mutual information $I(Y;\mathbf{Z})$ and empirical cross-entropy loss $\hat{\mathcal{L}}_{ce}$ is established—$I(Y;\mathbf{Z}) \geq H(Y) - \hat{\mathcal{L}}_{ce} - G\sqrt{\frac{\ln(1/\delta)}{2|\mathcal{D}|}}$—such that reducing the loss tightens the information lower bound.
    - Classification is modeled as mixture density estimation in feature space: $p(y=k|\mathbf{z}) = \frac{\exp(-d_\phi(\mathbf{z}, \mathbf{c}_k))}{\sum_{k'}\exp(-d_\phi(\mathbf{z}, \mathbf{c}_{k'}))}$, where $\mathbf{c}_k$ denotes the class prototype computed from the training set.
    - MTIR is defined as the change in classification loss before and after adding a recovered modality: a positive value indicates that the recovered modality provides useful information, while a negative value indicates that it introduces harmful information.
    - **Intra-Class Similarity (ICS) Calibration**: An asymmetric calibration term $\alpha$ is introduced to down-weight the reward when the recovered representation is less representative within its predicted class cluster than the pre-recovery representation ($\alpha < 1$), enhancing the reward function's sensitivity to semantic misalignment.

2. **Iterative Selection Algorithm + Flexible Multimodal Architecture**

    - Greedy iterative selection: at each step, the recovered modality with the highest MTIR is added to the observed set; all modalities with non-positive reward are removed; the process repeats until the candidate set is empty.
    - The multimodal Transformer supports arbitrary modality combinations: missing modality positions use dummy tokens with attention masking.
    - During training, random subsets simulate missing modalities ($A$ random subsets per sample), paired with a missingness-agnostic contrastive loss $\mathcal{L}_{aux}$ to encourage intra-class clustering.

### Loss & Training

- Classification loss: $\mathcal{L}_{class} = -\frac{1}{A}\frac{1}{B}\sum_{\mathcal{S} \sim \mathcal{U}_A}\sum_{i=1}^{B}\log p_f(y_i|\{x^{(m)}\}_{m \in \mathcal{S}})$

- Auxiliary contrastive loss: $\mathcal{L}_{aux} = -\frac{1}{A}\frac{1}{B}\sum\sum\log\frac{\exp(-d_\phi(\mathbf{z}_i, \mathbf{c}_{y_i})/t)}{\sum_{k'}\exp(-d_\phi(\mathbf{z}_i, \mathbf{c}_{k'})/t)}$

- Total loss: $\mathcal{L} = \mathcal{L}_{class} + \mathcal{L}_{aux}$

- Training is performed on complete data, with random subset sampling simulating all $2^M-1$ missing modality patterns.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on 5 datasets (PolyMNIST / MST / CelebA / DVM / UKBB-CAD / UKBB-Infarction):

| Method | PolyMNIST (η=0.8) | MST (miss{M,T}) | CelebA (miss{T}) | DVM (γ=1) | CAD (γ=1) |
|--------|-------------------|-----------------|------------------|-----------|-----------|
| ModDrop | 88.44 | 82.47 | 87.32 | 87.97 | 69.18 |
| MTL | 91.14 | 84.37 | 89.38 | 92.32 | 70.23 |
| OnlineMAE | 90.09 | 86.67 | 86.67 | - | - |
| M3Care† | 87.92 | 85.16 | 91.32 | 93.43 | 72.48 |
| **DyMo_c** | **96.61** | **85.31** | **95.20** | **93.14** | **71.02** |
| **DyMo_e** | **96.81** | **86.84** | 93.67 | **93.36** | **72.17** |

PolyMNIST at 80% missing: DyMo surpasses OnlineMAE by +5.67%; DVM with full tabular missing: surpasses ModDrop by +4.11%.

### Ablation Study

| Setting | PolyMNIST (η=0.8) | MST (miss{M,T}) |
|---------|-------------------|-----------------|
| Baseline (fuse all without selection) | 84.21 | 80.73 |
| S (simultaneous fusion of all positive-reward modalities) | 94.33 | 82.08 |
| I (iterative selection of highest-reward modality) | 94.50 | 82.12 |
| I+C (iterative + calibration, full DyMo) | **96.61** | **85.31** |

### Key Findings

- Fusing all recovered modalities without dynamic selection (Baseline) yields substantially lower performance than DyMo, validating the assumption that recovery quality is unreliable.
- Iterative selection (I) marginally outperforms simultaneous selection (S); ICS calibration (C) provides an additional 1–3% gain on most datasets.
- DyMo is insensitive to the choice of distance function (cosine vs. Euclidean), with both yielding comparable results.
- Existing dynamic fusion methods (QMF/DynMM/PDF) offer limited benefit under VAE-based recovery, as they cannot detect semantic misalignment.

## Highlights & Insights

- The formulation of the discarding-imputation dilemma is precise and well-motivated; this work is the first to systematically address it with a theoretical framework.
- The theoretical derivation chain from mutual information to classification loss to class prototype distance is complete: $I(Y;\mathbf{Z})$ → loss lower bound → Bregman divergence → computable MTIR.
- The asymmetric design of ICS calibration ($\alpha \leq 1$ when post-recovery representativeness is lower than pre-recovery) reflects a principled "conservatism" toward recovered modalities—a well-motivated engineering choice.
- The training strategy is concise and effective: random subset simulation combined with contrastive loss, requiring no additional networks or multi-stage training.

## Limitations & Future Work

- The ICS calibration term degrades performance on the CAD/Infarction datasets, necessitating dataset-specific hyperparameter tuning.
- Each recovered modality requires a separate forward pass to compute MTIR—inference overhead increases with the number of missing modalities $M - |\mathcal{I}|$.
- The choice of recovery method substantially affects DyMo's performance (e.g., TIP has limited capacity for full tabular recovery); DyMo's performance ceiling is bounded by the recovery method.
- Validation is limited to classification tasks; extension to dense prediction tasks such as segmentation and detection remains unexplored.

## Related Work & Insights

- **vs. ModDrop/MUSE**: Discarding-based methods suffer severe performance degradation when high-information modalities are missing (MUSE drops by 61% on MST); DyMo avoids this by combining recovery with selective fusion.
- **vs. MoPoE/M3Care**: Imputation-based methods generate unreliable reconstructions under severe missingness; DyMo filters unreliable recoveries via MTIR.
- **vs. QMF/DynMM/PDF**: Existing dynamic fusion methods focus on intra-modal noise and cannot detect semantic misalignment; DyMo handles both types of unreliability through class prototype distance.
- **Methodological insight**: Using task loss reduction as a proxy for information gain is a broadly applicable principle for any scenario requiring dynamic decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel problem formulation with complete theoretical derivation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, nine SOTA baselines, full ablation, and visual analysis
- Writing Quality: ⭐⭐⭐⭐ Tight integration of theory and experiments; clear mathematical derivations
- Value: ⭐⭐⭐⭐ A general framework for incomplete multimodal learning, directly applicable to medical imaging scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](../../CVPR2026/medical_imaging/gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](../../CVPR2026/medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](../../CVPR2026/medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)

</div>

<!-- RELATED:END -->
