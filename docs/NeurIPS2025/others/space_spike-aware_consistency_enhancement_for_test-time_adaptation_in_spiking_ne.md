---
title: >-
  [Paper Note] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks
description: >-
  [NeurIPS 2025][SNN] This paper proposes SPACE, the first source-free single-sample test-time adaptation (TTA) method specifically designed for spiking neural networks (SNNs). By maximizing the consistency of spike-based…
tags:
  - "NeurIPS 2025"
  - "SNN"
  - "test-time adaptation"
  - "spike consistency"
  - "distribution shift"
  - "single-sample adaptation"
date: 2026-05-08
content_hash: 0c653ad9be7b2513
---

# SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2504.02298](https://arxiv.org/abs/2504.02298)  
**Code**: [GitHub](https://github.com/ethanxyluo/SPACE)  
**Area**: Spiking Neural Networks, Test-Time Adaptation, Domain Robustness
**Keywords**: SNN, test-time adaptation, spike consistency, distribution shift, single-sample adaptation

## TL;DR
This paper proposes SPACE, the first source-free single-sample test-time adaptation (TTA) method specifically designed for spiking neural networks (SNNs). By maximizing the consistency of spike-based feature maps across augmented views of a test sample, SPACE achieves robust adaptation across multiple datasets and architectures.

## Background & Motivation
- SNNs serve as biologically plausible alternatives to ANNs with advantages in energy efficiency and temporal processing, yet they are highly sensitive to distribution shifts.
- Experiments demonstrate that SNNs suffer significantly greater accuracy degradation under distribution shift than ANNs of equivalent architecture (22.74% loss for SNN vs. 13.28% for ANN on CIFAR-10).
- Existing ANN-TTA methods are not directly applicable to SNNs:
    - MEMO: operates only on output probabilities, offering no control over fine-grained temporal spike dynamics.
    - SITA: relies on batch normalization (BN) statistics updates, whereas many SNN architectures lack BN layers.
    - SHOT/TAST: require mini-batches of target data or batch-level statistics.
- A SNN-specific TTA method that directly exploits spike dynamics is therefore needed.

## Method

### Overall Architecture
The pipeline consists of four steps:
1. Generate an augmented batch from a single test sample.
2. Extract spike-count local feature maps through the model.
3. Adapt the model by maximizing feature map similarity across augmented views.
4. Predict the label of the original sample using the adapted model.

### Key Designs

#### Spike-Aware Feature Maps
- The deepest layer of the feature extractor $E_{\theta_E}$ that retains spatial support is selected.
- For each augmented view $\mathbf{x}_i$, binary spike outputs $\mathbf{O} \in \{0,1\}$ from LIF neurons are aggregated along the temporal dimension.
- This yields a spike-aware feature map $\mathbf{F}(\mathbf{x}_i) \in \mathbb{R}^{C \times D}$, where $C$ denotes the number of channels and $D = H \times W$ denotes the spatial dimension.

Three motivations for using total spike counts:
1. SNNs typically operate over tens to thousands of time steps, making step-wise matching redundant and computationally expensive.
2. Aggregation smooths the loss landscape — the loss remains unchanged as long as spike counts match, even if individual spikes jitter in time.
3. Distribution shift primarily manifests as changes in intermediate-layer firing rates and spatial activation support.

#### Feature Map Alignment
- Channel-wise local vectors are normalized into probability distributions $\mathbf{P}_c \in \Delta^{D-1}$ via softmax.
- The similarity between two augmented views is computed as the channel-averaged inner product:
$$\bar{\mathcal{S}}(i,j|\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\sum_{d=1}^{D}\mathbf{P}_{c,d}(\mathbf{x}_i)\mathbf{P}_{c,d}(\mathbf{x}_j)$$

### Loss & Training
$$\mathcal{L}(\theta_E;\mathbf{x}) = \sum_{1 \leq j < i \leq M}(1 - \bar{\mathcal{S}}(i,j|\mathbf{x}))$$

- Only the extractor parameters $\theta_E$ are updated; the classifier is frozen.
- A single-step SGD update is applied per test sample.
- Augmentation strategy: AugMix, with batch sizes of 32 (CIFAR) and 64 (ImageNet).
- Each test sample triggers exactly one update step.

## Key Experimental Results

### Main Results (CIFAR-10-C, Highest Corruption Severity 5, Top-1 Accuracy %)

| Method | Gauss. | Shot | Impl. | Avg |
|-----|--------|------|-------|-----|
| No Adapt (VGG9) | 72.38 | 74.70 | 58.57 | 66.57 |
| SITA | 73.06 | 74.15 | 58.41 | 66.41 |
| MEMO | 77.73 | 79.50 | 65.74 | 69.20 |
| **SPACE** | **77.98** | **79.34** | **69.41** | **71.03** |

### Cross-Architecture Generalization (Accuracy Gain, Percentage Points)

| Dataset | SNN-VGG9 | SNN-ResNet11 | Spike Transformer V3 | SNN-ConvLSTM |
|-------|----------|-------------|---------------------|-------------|
| CIFAR-10-C | +4.46 | +2.19 | +0.65 | +1.30 |
| CIFAR-100-C | +1.72 | +1.03 | +0.29 | applicable |
| ImageNet series | — | — | +0.53~1.97 | — |

### SNN vs. ANN Sensitivity to Distribution Shift

| Architecture | Dataset | ANN Accuracy Loss | SNN Accuracy Loss |
|-----|-------|-----------|-----------|
| VGG9 | CIFAR-10 | 13.28 | **22.74** |
| VGG11 | CIFAR-100 | 22.54 | **28.14** |
| VGG11 | Tiny-ImageNet | 41.48 | **43.80** |
| Transformer | ImageNet | 9.67 | **11.96** |

### Key Findings
- SPACE consistently improves performance across all four SNN architectures (CNN / Transformer / ConvLSTM).
- The largest gains are observed on low-contrast corruptions such as Fog (VGG9 CIFAR-10-C: 43.57 → 52.80).
- SPACE provides minimal benefit on Contrast corruption (22.54 → 23.85), as extreme contrast changes cause spike rate collapse.
- Spike-level alignment provides a more stable adaptation signal than the output-level alignment used by MEMO.
- SPACE is also effective on the DVS Gesture neuromorphic dataset.

## Highlights & Insights
1. **First SNN-TTA method**: Fills a critical gap in test-time adaptation for spiking neural networks.
2. **Exploits SNN-intrinsic properties**: Operates directly on spike dynamics rather than porting ANN methods wholesale.
3. **BN-free**: Does not require batch normalization layers, broadening applicability across diverse SNN architectures.
4. **Computationally efficient**: Single-step SGD combined with spike-count aggregation avoids step-wise temporal matching.
5. **Theoretically motivated**: Supported by both an information bottleneck perspective (suppressing augmentation-specific variance) and a manifold denoising perspective.

## Limitations & Future Work
- Limited effectiveness under extreme corruptions (e.g., Contrast), where spike rates may collapse.
- Adaptation quality is sensitive to the choice of AugMix augmentation.
- Evaluation is restricted to classification tasks; extension to detection and segmentation remains unexplored.
- Kernel embedding / MMD variants yield no significant gain while incurring additional overhead.
- Future work may explore integration with SNN-specific data augmentation strategies.

## Related Work & Insights
- MEMO and SITA are pioneering ANN-TTA methods but are constrained when applied to SNNs.
- SNN-SFDA (Guo 2023) operates in a batch-processing / epoch-level setting, making it unsuitable for per-sample online adaptation.
- This work represents a first step in extending the TTA paradigm from the ANN regime to the SNN regime.

## Rating
- Novelty: ⭐⭐⭐⭐ (first SNN-TTA method with a well-defined problem formulation)
- Technical Depth: ⭐⭐⭐⭐ (principled spike-dynamics alignment design with solid theoretical motivation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 architectures × multiple datasets × detailed ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (clear and comprehensive)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)

</div>

<!-- RELATED:END -->
