---
title: >-
  [Paper Note] Information-Bottleneck Driven Binary Neural Network for Change Detection
description: >-
  [ICCV 2025][Remote Sensing][Binary Neural Network] This paper proposes BiCD, the first binary neural network specifically designed for change detection. By introducing an auxiliary objective module guided by the Information Bottleneck (IB) principle, BiCD enhances the feature representation capability and separability of BNNs, achieving state-of-the-art performance among BNN-based methods on both street-view and remote sensing change detection benchmarks, while achieving 30× memory compression and 2.5× inference acceleration.
tags:
  - ICCV 2025
  - Remote Sensing
  - Binary Neural Network
  - Information Bottleneck
  - Change Detection
  - Model Compression
  - Auxiliary Objective
date: 2026-05-08
content_hash: 4fa0b8fdd82b5540
---

# Information-Bottleneck Driven Binary Neural Network for Change Detection

**Conference**: ICCV 2025
**arXiv**: [2507.03504](https://arxiv.org/abs/2507.03504)
**Code**: N/A
**Area**: Remote Sensing / Change Detection
**Keywords**: Binary Neural Network, Information Bottleneck, Change Detection, Model Compression, Auxiliary Objective

## TL;DR

This paper proposes BiCD, the first binary neural network specifically designed for change detection. By introducing an auxiliary objective module guided by the Information Bottleneck (IB) principle, BiCD enhances the feature representation capability and separability of BNNs, achieving state-of-the-art performance among BNN-based methods on both street-view and remote sensing change detection benchmarks, while achieving 30× memory compression and 2.5× inference acceleration.

## Background & Motivation

### State of the Field

Change detection is a fundamental problem in computer vision, with broad applications in urban map updating, disaster assessment, and autonomous driving. Although existing deep neural network methods achieve strong performance, their substantial computational and memory overhead makes deployment on edge devices challenging.

### Limitations of Prior Work

Network quantization—especially binarization—represents the most aggressive form of compression, enabling 32× memory reduction and 58× computational speedup. However, directly applying existing binarization techniques to change detection leads to severe performance degradation. The root cause is that the aggressive binarization process significantly reduces the mutual information $I(X,Z)$ between the input $X$ and the latent representation $Z$, causing the network to lose the fine-grained feature granularity necessary to distinguish meaningful changes from noisy changes.

### Root Cause

BNNs require extreme compression to fit edge devices, yet change detection demands fine-grained feature representations to differentiate changes of interest from noise-induced irrelevant changes. These two requirements are fundamentally in conflict.

### Starting Point

From an information-theoretic perspective, the IB principle is leveraged to balance feature compression and information retention. An auxiliary objective module is introduced to enhance the feature separability of BNNs; this module is activated only during training and removed at inference, incurring no additional computational overhead.

## Method

### Overall Architecture

BiCD is built upon the C-3PO change detection framework with binarization adaptations. Bi-temporal feature pyramids are extracted by a shared 1-bit backbone, merged through a 1-bit change generator, and passed through channel average pooling and a 1-bit ASPP module to produce the change mask. During training, an auxiliary module derives dimension-aligned features from the 1-bit change generator's representations to compute the auxiliary loss.

### Key Designs

#### 1. **Information Bottleneck-Driven Auxiliary Objective**

- **Function**: Introduces an IB-principle-based auxiliary objective to enhance the encoder's ability to retain critical input information and improve feature separability.
- **Mechanism**: The standard IB objective is extended into a three-term optimization:

$$\min I(X, Z(\theta)) - \beta_1 I(Z(\theta), Y) - \beta_2 \Psi$$

where $\Psi = I(Z(\theta,\eta)_n, 0) + I(Z(\theta,\eta)_{in}, \Delta X_{in}) + I(X, Z(\theta,\eta))$

The three terms respectively correspond to: suppressing noisy changes, preserving changes of interest, and reconstructing the original input.
- **Design Motivation**: Since $I(X,Z)$ is inherently low in BNNs, directly optimizing the IB objective would further degrade feature quality. The auxiliary objective explicitly enhances feature separability while implicitly preserving input information through reconstruction loss.

#### 2. **Auxiliary Module**

- **Function**: Maps latent features to a space dimensionally aligned with the input and labels, making mutual information estimation tractable.
- **Mechanism**: The auxiliary module $\sigma(\cdot, \eta)$ consists of four parallel MLP branches and a convolutional output layer, transforming features $Z(\theta)$ into dimension-aligned representations $Z(\theta,\eta)$, with mutual information approximated via L1 loss.
- **Design Motivation**: Direct mutual information estimation is infeasible due to inconsistent feature dimensions across network layers. The auxiliary module serves as a dimension adapter and is removed at inference, introducing zero overhead.

#### 3. **Noise/Interest Change Separation Mechanism**

- **Function**: Uses the change mask $Y$ to decompose the aligned features into "noisy changes" $Z(\theta,\eta)_n$ and "changes of interest" $Z(\theta,\eta)_{in}$.
- **Mechanism**: The noisy change component is suppressed toward zero via $\|Z(\theta,\eta)_n\|_1$, while the interest change component is preserved via $\|Z(\theta,\eta)_{in} - \Delta X_{in}\|_1$.
- **Design Motivation**: The core challenge of change detection lies in distinguishing meaningful changes from environmentally induced irrelevant changes. Explicit separation and targeted optimization address this directly.

### Loss & Training

The final objective function is:

$$\min \text{Obj} = \beta_1 \|Z(\theta)\|_2 + L_{cd} + \beta_2(\|Z(\theta,\eta)_n\|_1 + \|Z(\theta,\eta) - X\|_1 + \|Z(\theta,\eta)_{in} - \Delta X_{in}\|_1)$$

- $\beta_1 = 1e{-3}$: controls the rate of redundant information suppression
- $\beta_2 = 0.08$: controls feature separability
- Adam optimizer, initial learning rate 5e-4, cosine annealing, trained for 140 epochs
- Auxiliary module initial learning rate 5e-3, decayed by 1/10 at epochs 90 and 120

## Key Experimental Results

### Main Results

| Dataset | Framework | Method | Bits | F1-score (%) | vs BNN SOTA |
|--------|------|------|------|-------------|-------------|
| PCD-TSUNAMI | DR-TANet | BiCD | 1 | 85.1 | +2.5 (vs ReActNet 83.4) |
| PCD-TSUNAMI | C-3PO | BiCD | 1 | **86.5** | +2.5 (vs ReActNet 84.0) |
| PCD-GSV | DR-TANet | BiCD | 1 | 67.7 | +2.0 (vs ReActNet 65.7) |
| PCD-GSV | C-3PO | BiCD | 1 | **74.1** | +2.9 (vs ReActNet 71.2) |
| VL_CMU_CD | DR-TANet | BiCD | 1 | 65.9 | +3.3 (vs ReActNet 62.6) |
| VL_CMU_CD | C-3PO | BiCD | 1 | **71.9** | +2.0 (vs ReActNet 69.9) |
| LEVIR-CD | C-3PO | BiCD | 1 | **89.9** | +1.1 (vs ReActNet 88.8) |

Notably, 1-bit C-3PO + BiCD achieves 86.5% F1-score on TSUNAMI, surpassing full-precision DR-TANet (87.6% is approached closely), with only 2.1M parameters (vs 33.4M) and 6.6G OPs (vs 28.5G).

### Ablation Study

| Configuration | Auxiliary Objective Location | F1-score (%) | Notes |
|------|-------------|-------------|------|
| Baseline | None | 84.7 | No auxiliary module |
| +BiCD (full Ψ) | backbone | 84.8 | +0.1; applying separability directly in Siamese branch is ineffective |
| +BiCD (reconstruction only) | backbone | 85.2 | +0.5; reconstruction loss implicitly preserves input information |
| +BiCD (full Ψ) | 1-bit generator | 85.8 | +1.1; separability requires interaction with change features |
| **+BiCD (best)** | **backbone + generator** | **86.5** | **+1.8; reconstruction in backbone + separability in generator** |

### Key Findings

- The separability objective must be placed in the 1-bit change generator to be effective; placing it directly in the Siamese backbone is detrimental.
- The reconstruction loss is more effective when applied in the backbone, as it requires direct interaction with the original feature pairs.
- The optimal configuration decouples the two objectives: reconstruction in the backbone, separability in the generator.
- On ARM Cortex-A76 edge hardware, the 1-bit model achieves 158.4 ms latency vs. 392.8 ms for the full-precision model, yielding 2.5× speedup.
- BiCD introduces no additional inference latency, as the auxiliary module is used exclusively during training.

## Highlights & Insights

1. **First application of BNNs to change detection**: Opens a new research direction and demonstrates the viability of 1-bit networks for this task.
2. **Principled use of the IB framework**: Rather than directly optimizing the IB objective (which would be harmful given the already-low $I(X,Z)$ in BNNs), auxiliary objectives are introduced to compensate for information loss.
3. **Training-inference decoupled design**: The auxiliary module is active only during training and fully removed at inference, achieving zero-overhead performance improvement.
4. **Information plane analysis**: Mutual information plane visualizations intuitively reveal the bottleneck characteristics of BNNs in the change detection setting.

## Limitations & Future Work

1. Validation is limited to a ResNet-18 backbone; deeper binarized architectures remain unexplored.
2. The four-branch MLP design of the auxiliary module lacks dedicated ablation validation.
3. Hyperparameters $\beta_1$ and $\beta_2$ require per-dataset tuning.
4. No comparison or combination with other compression methods such as knowledge distillation is explored.
5. A notable performance gap remains on LEVIR-CD relative to full-precision SOTA methods (e.g., M-CD at 92.1%).

## Related Work & Insights

- C-3PO serves as a strong baseline framework for change detection; its high computational cost (222G OPs) makes it a natural candidate for binarization.
- The application of the IB principle in model compression—grounded in its equivalence to the minimum description length principle—provides theoretical justification for the proposed approach.
- The auxiliary module concept draws from local learning frameworks, and its adaptation to the BNN setting represents a natural extension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First BNN for change detection; theoretically grounded use of the IB principle.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, two frameworks, detailed ablation, and edge device deployment validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear; information plane analysis is intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for change detection in resource-constrained scenarios with meaningful deployment implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration](geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](../../CVPR2026/remote_sensing/avion_aerial_vision-language_instruction_from_offline_teacher_to_prompt-tuned_ne.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)

</div>

<!-- RELATED:END -->
