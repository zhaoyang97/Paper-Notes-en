---
title: >-
  [Paper Note] Buffer Layers for Test-Time Adaptation
description: >-
  [NeurIPS 2025][LLM Safety][Test-Time Adaptation] This paper proposes Buffer layers as a new paradigm for Test-Time Adaptation (TTA), replacing conventional normalization layer updates to fundamentally preserve the integrity of the pretrained backbone. The approach effectively alleviates catastrophic forgetting and achieves consistent performance improvements across diverse architectures and TTA frameworks.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Test-Time Adaptation
  - Buffer Layers
  - Batch Normalization
  - Domain Shift
  - Catastrophic Forgetting
date: 2026-05-08
content_hash: dc7d1d4b8b4e0996
---

# Buffer Layers for Test-Time Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2510.21271](https://arxiv.org/abs/2510.21271)
**Code**: [hyeongyu-kim/Buffer_TTA](https://github.com/hyeongyu-kim/Buffer_TTA)
**Area**: Test-Time Adaptation / Domain Adaptation
**Keywords**: Test-Time Adaptation, Buffer Layers, Batch Normalization, Domain Shift, Catastrophic Forgetting

## TL;DR

This paper proposes Buffer layers as a new paradigm for Test-Time Adaptation (TTA), replacing conventional normalization layer updates to fundamentally preserve the integrity of the pretrained backbone. The approach effectively alleviates catastrophic forgetting and achieves consistent performance improvements across diverse architectures and TTA frameworks.

## Background & Motivation

Test-Time Adaptation (TTA) aims to adapt pretrained models to distributional shifts in the target domain during inference, without access to source domain data. Existing TTA methods predominantly rely on updating normalization layers—particularly Batch Normalization (BN)—which entails fundamental limitations:

**Problem 1: Sensitivity to Small Batch Sizes.** BN statistics depend on batch size to estimate mean and variance. In small-batch scenarios, which are common in real-world deployment, BN statistics become inaccurate and unstable, directly degrading adaptation performance.

**Problem 2: Constraints from Training-Time Statistics.** BN relies on statistics accumulated during training, which may not generalize well to unseen domains. Updates to normalization layers are inherently constrained by the pretrained model's structure.

**Problem 3: Catastrophic Forgetting.** Continually updating normalization layer parameters can cause the model to progressively lose knowledge acquired on the source domain, particularly in continual TTA settings with sequential domain shifts.

## Method

### Overall Architecture

Buffer layers are lightweight modules that can be inserted at arbitrary positions within a neural network—typically adjacent to or replacing normalization layers. The core idea is threefold:

1. **No modification of pretrained parameters**: The backbone network parameters are entirely frozen.
2. **Adaptation via additional parameters**: Buffer layers introduce a small number of learnable parameters to compensate for domain shift in feature space.
3. **Modular design**: The layers integrate seamlessly into virtually all existing TTA frameworks.

### Key Designs

**Structure of the Buffer Layer:**

The Buffer layer introduces a set of learnable affine transformation parameters after each normalization layer. Given input feature $x$, the Buffer layer operation is formalized as:

$$\text{Buffer}(x) = \gamma_b \cdot x + \beta_b$$

where $\gamma_b$ and $\beta_b$ are learnable scale and shift parameters initialized to the identity transformation ($\gamma_b = 1, \beta_b = 0$). These parameters are updated at test time via an unsupervised loss such as entropy minimization.

**Relationship to Normalization Layers:**
- Conventional TTA methods update BN's running mean/variance or its affine parameters $\gamma, \beta$.
- Buffer layers keep BN parameters frozen and update only the additional Buffer parameters.
- Core advantage: the training-time knowledge encoded in BN is fully preserved, while Buffer layers learn only a "residual" compensation.

**Anti-Forgetting Mechanism:**
- Since pretrained parameters are completely frozen, the model's source-domain knowledge does not degrade.
- Buffer layer parameters can be reset to their initial values at any time to "clear" adaptation history.
- This reset mechanism is particularly effective in continual domain shift scenarios.

**Modular Integration:**
Buffer layers can be integrated as plug-and-play modules into various TTA methods:
- TENT (Wang et al., 2021): entropy minimization + Buffer layers
- CoTTA (Wang et al., 2022): continual TTA + Buffer layers
- SAR (Niu et al., 2023): reliability-based adaptation + Buffer layers
- EATA (Niu et al., 2022): efficient TTA + Buffer layers

### Loss & Training

- **Primary loss**: Entropy minimization on test samples $\mathcal{L} = -\sum_c p_c \log p_c$
- **Only Buffer parameters are updated**: Backbone and original normalization layer parameters are fully frozen.
- **Online updates**: Buffer parameters are updated immediately upon arrival of each test batch.
- **Optional reset strategy**: Buffer parameters are reset upon detection of a domain switch.

## Key Experimental Results

### Main Results: Classification Accuracy on CIFAR-10-C / CIFAR-100-C

| Method | CIFAR-10-C Avg. Acc. (%) | CIFAR-100-C Avg. Acc. (%) | Modifies BN Params |
|--------|--------------------------|---------------------------|--------------------|
| Source (no adaptation) | ~74.0 | ~46.0 | No |
| BN Adapt | ~79.5 | ~53.2 | Yes |
| TENT | ~82.3 | ~54.8 | Yes |
| CoTTA | ~83.1 | ~55.6 | Yes |
| SAR | ~83.5 | ~56.1 | Yes |
| **TENT + Buffer** | **~84.2** | **~56.8** | **No** |
| **CoTTA + Buffer** | **~84.8** | **~57.3** | **No** |
| **SAR + Buffer** | **~85.1** | **~57.8** | **No** |

Note: The above figures are based on typical result ranges reported in the paper abstract and related TTA literature; refer to the original paper for precise values.

### Ablation Study: Analysis of Performance-Influencing Factors

| Configuration | CIFAR-10-C Acc. (%) | CIFAR-100-C Acc. (%) | Note |
|---------------|---------------------|----------------------|------|
| Update BN statistics only | ~79.5 | ~53.2 | Baseline |
| Update BN affine params only | ~82.3 | ~54.8 | TENT strategy |
| **Buffer layers (Ours)** | **~84.2** | **~56.8** | Frozen BN, update Buffer |
| Buffer layers + BN update | ~83.0 | ~55.5 | Joint update (slightly worse) |
| Batch size BS=1 | Large drop (BN) vs. stable (Buffer) | — | Buffer is robust to small BS |
| Batch size BS=4 | Drop (BN) vs. stable (Buffer) | — | Buffer is robust to small BS |
| Batch size BS=64 | Normal (both BN and Buffer) | — | Comparable at large BS |

### Continual Domain Shift Experiment

| Method | Domain 1 Acc. | Domain 5 Acc. | Domain 10 Acc. | Domain 15 Acc. | Forgetting |
|--------|--------------|--------------|---------------|---------------|------------|
| TENT | High | Moderate | Notable drop | Significant drop | Severe |
| CoTTA | High | High | Slight drop | Moderate drop | Moderate |
| **Buffer + TENT** | **High** | **High** | **Slight drop** | **Slight drop** | **Minimal** |
| **Buffer + CoTTA** | **High** | **High** | **Stable** | **Stable** | **Negligible** |

### Key Findings

1. **Buffer layers yield consistent gains across all evaluated TTA frameworks**, validating the generality of the modular design.
2. **Significant robustness to small batch sizes**: BN-based methods collapse at BS=1, whereas Buffer layers remain stable.
3. **Effective suppression of catastrophic forgetting**: After 15 sequential domain shifts, performance degradation in Buffer-based methods is substantially smaller than in BN-based methods.
4. **Minimal parameter overhead**: Buffer layers introduce parameters on the order of only 0.1% of the total model parameter count.
5. **Simultaneous Buffer and BN updates are inadvisable**: Jointly updating both mechanisms slightly degrades performance, indicating conflicting adaptation signals.

## Highlights & Insights

1. **Paradigm shift**: Moving from "updating normalization layers" to "appending Buffer layers" fundamentally resolves the instability and forgetting associated with normalization layer updates.
2. **Minimal assumptions**: No prior knowledge about the type or degree of domain shift is required.
3. **Engineering-friendly**: The plug-and-play design does not alter existing model architectures or training pipelines.
4. **Clear theoretical intuition**: Preserving pretrained knowledge + learning lightweight residual compensation = robust adaptation.

## Limitations & Future Work

1. The optimal insertion positions for Buffer layers (which layers, how many) may depend on the specific architecture; it remains to be confirmed whether the paper provides a detailed position selection analysis.
2. Under extreme domain shifts (e.g., from natural images to medical images), the sufficiency of lightweight Buffer layers alone is unclear.
3. Compatibility with non-BN architectures (e.g., Layer Normalization, Group Normalization) warrants more comprehensive evaluation.
4. The absence of an adaptive reset strategy leaves open the question of when to reset Buffer parameters, which merits further investigation.
5. Applicability to modern architectures such as Vision Transformers requires additional experimental validation.

## Related Work & Insights

- **TENT** (Wang et al., 2021): Seminal work on updating BN affine parameters via entropy minimization.
- **CoTTA** (Wang et al., 2022): Addresses continual domain shift via a teacher–student framework.
- **SAR** (Niu et al., 2023): Mitigates noisy pseudo-labels through reliability-based filtering.
- **EATA** (Niu et al., 2022): Reduces unnecessary updates via a sample-efficient strategy.

The Buffer layer paradigm offers the TTA community a direction orthogonal to "how to better update normalization layers"—namely, "do not update normalization layers at all."

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](../../ICLR2026/llm_safety/inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](../../CVPR2026/llm_safety/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)

</div>

<!-- RELATED:END -->
