---
title: >-
  [Paper Note] ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation
description: >-
  [CVPR 2026][Multimodal VLM][Vision-Language Models] ReHARK is a four-stage refinement pipeline that constructs hybrid semantic-visual priors, augments the support set, applies adaptive distribution rectification, and integrates multi-scale RBF kernels, achieving 65.83% one-shot adaptation accuracy across 11 benchmarks and substantially outperforming Tip-Adapter and ProKeR.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Vision-Language Models
  - One-Shot Adaptation
  - Kernel Ridge Regression
  - CLIP
  - GPT-3 Semantics
date: 2026-05-08
content_hash: e0eff3ec2b5337cb
---

# ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation

**Conference**: CVPR 2026
**arXiv**: [2603.11542](https://arxiv.org/abs/2603.11542)
**Code**: [Jahid12012021/ReHARK](https://github.com/Jahid12012021/ReHARK)
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, One-Shot Adaptation, Kernel Ridge Regression, CLIP, GPT-3 Semantics

## TL;DR

ReHARK is a four-stage refinement pipeline that constructs hybrid semantic-visual priors, augments the support set, applies adaptive distribution rectification, and integrates multi-scale RBF kernels, achieving 65.83% one-shot adaptation accuracy across 11 benchmarks and substantially outperforming Tip-Adapter and ProKeR.

## Background & Motivation

**Background**: Vision-language models such as CLIP exhibit strong zero-shot capabilities, yet still require adaptation for specific downstream tasks. Training-free methods like Tip-Adapter avoid fine-tuning via cache mechanisms, but are essentially local Nadaraya-Watson (NW) estimators.

**Limitations of Prior Work**: Local NW estimators suffer from boundary bias and lack global structural regularization. ProKeR introduces RKHS-based global regularization but remains limited in the extreme 1-shot regime, where a single visual sample fails to capture domain-specific nuances.

**Key Challenge**: With only one visual sample available, how can one balance the retention of pretrained knowledge (stability) with adaptation to new tasks (plasticity)?

**Goal**: (1) How to construct a more robust initial prior than pure visual features? (2) How to mitigate distribution shift between the support and query sets? (3) How to handle heterogeneous feature geometries across datasets?

**Key Insight**: A single visual sample is insufficient for robust adaptation; it is necessary to introduce complementary priors from textual knowledge (CLIP + GPT-3) and visual prototypes, while employing multi-scale kernels to capture feature geometry at different scales.

**Core Idea**: Fuse CLIP zero-shot weights, GPT-3 dense semantic descriptions, and visual class prototypes into a hybrid prior, then perform global kernel ridge regression adaptation in RKHS via multi-scale RBF kernel integration.

## Method

### Overall Architecture

ReHARK is a four-stage refinement pipeline: (1) **Hybrid Prior Construction** — fusing CLIP, GPT-3, and visual prototypes; (2) **Support Set Augmentation (Bridging)** — generating intermediate samples to smooth modality transitions; (3) **Adaptive Distribution Rectification** — nonlinear power transformation for feature distribution alignment; (4) **Multi-Scale RBF Kernels** — capturing multi-scale feature geometry. Final inference is a closed-form solution of global kernel ridge regression.

### Key Designs

1. **Hybrid Semantic-Visual Prior**:

    - **Function**: Construct a more robust global anchor than zero-shot text weights alone.
    - **Mechanism**: The CLIP text weights $\mathbf{W}_{clip}$ and GPT-3 semantic weights $\mathbf{W}_{gpt3}$ are blended at ratio $\gamma$ to form the text prior $\mathbf{W}_{text}$, which is then fused with the visual class prototype $\mathbf{P}_{vis}$ at ratio $\omega$ to yield the final prior $\mathbf{W}_{prior}$.
    - **Design Motivation**: GPT-3 provides richer class descriptions than CLIP hand-crafted templates (e.g., "the beak of a bird is sharp..."), visual prototypes supply domain-specific information, and the three sources are mutually complementary.

2. **Support Set Augmentation (Bridge)**:

    - **Function**: Expand the support set by generating intermediate samples through mixing visual features and text priors.
    - **Mechanism**: $\mathbf{x}_{bridge} = \text{norm}(\mathbf{x}_{vis} + \eta \mathbf{w}_{label})$, fusing each visual sample with its corresponding class prior.
    - **Design Motivation**: In the 1-shot setting, the support set is extremely sparse; bridge samples fill the gap between the visual and textual modalities, yielding a smoother adaptation manifold.

3. **Multi-Scale RBF Kernel Integration**:

    - **Function**: Capture local and global similarity using two Gaussian kernels with different bandwidths.
    - **Mechanism**: $\mathbf{K}(\mathbf{x}, \mathbf{x}') = \pi e^{-\beta_1\|\cdot\|^2} + (1-\pi)e^{-\beta_2\|\cdot\|^2}$, where $\beta_1$ and $\beta_2$ capture local and global scales respectively, and $\pi$ is the mixing weight. Adaptation coefficients are obtained via the closed-form solution $\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}(\mathbf{Y} - \hat{\mathbf{Y}}_{zs})$.
    - **Design Motivation**: A single bandwidth is rarely optimal across diverse datasets; multi-scale kernels adaptively handle the high variance inherent in 1-shot learning.

### Loss & Training

ReHARK is entirely training-free. All hyperparameters ($\gamma, \omega, \eta, \beta_1, \beta_2, \pi, p, \lambda$) are automatically searched over 1,000 trials on a validation set using the Optuna framework. Inference directly applies the closed-form solution without any backpropagation.

## Key Experimental Results

### Main Results (1-shot Classification Accuracy %, ViT-B/16)

| Method | ImageNet | Caltech101 | EuroSAT | Food101 | OxfordFlowers | Average |
|--------|----------|------------|---------|---------|---------------|---------|
| Zero-Shot CLIP | 60.35 | 85.68 | 36.27 | 77.37 | 66.02 | 58.88 |
| Tip-Adapter | 60.58 | 88.09 | 56.76 | 77.54 | 75.06 | 62.85 |
| ProKeR | 60.60 | 88.17 | 59.75 | 77.40 | 78.85 | 63.77 |
| **ReHARK** | **61.88** | **90.13** | **69.19** | **77.55** | **80.82** | **65.83** |

### Ablation Study (1-shot, 500 trials)

| Configuration | Average Accuracy | Note |
|---------------|-----------------|------|
| Full ReHARK | 65.83 | Complete model |
| NO_POWER | 65.32 | Remove power transform: −0.51 |
| NO_Refine | 65.49 | Remove visual prior refinement: −0.34 |
| NO_RECTIFY | 65.43 | Remove distribution rectification: −0.40 |
| NO_MULTISCALE | 65.72 | Remove multi-scale kernel: −0.11 |

### Key Findings

- The largest gain is observed on EuroSAT (59.75→69.19%, +9.44%), indicating that hybrid priors provide substantial benefits for structurally sensitive datasets.
- The power transform ($p$) contributes the most (−0.51%), followed by distribution rectification (−0.40%), suggesting that feature-space preprocessing is more critical than kernel design.
- All computations are performed on a single P100 GPU, yielding high inference efficiency due to the training-free, closed-form approach.

## Highlights & Insights

- **Multimodal Prior Fusion**: A three-way fusion of CLIP text weights, GPT-3 descriptions, and visual prototypes constructs a more stable prior than any single source — GPT-3 supplies rich semantics, CLIP provides zero-shot alignment, and visual prototypes enable domain adaptation.
- **Bridge Mechanism — Simple yet Effective**: Support set augmentation via weighted feature mixing requires no complex data augmentation or generative models.
- **Theoretical Perspective on Global vs. Local**: The necessity of global KRR is motivated by the locality of NW estimators, providing a theoretically grounded justification.

## Limitations & Future Work

- Hyperparameter search requires 1,000 trials per dataset; while a one-time cost, this is not an elegant solution.
- Bridge samples are constructed via simple linear mixing of visual features and text priors; more sophisticated cross-modal generation strategies may be more effective.
- Validation is limited to classification tasks; extension to downstream tasks such as object detection and segmentation remains unexplored.

## Related Work & Insights

- **vs. Tip-Adapter**: Tip-Adapter operates as a local NW estimator; ReHARK improves average accuracy from 62.85% to 65.83% through global KRR combined with a hybrid prior.
- **vs. ProKeR**: ProKeR also employs global KRR but relies solely on CLIP text weights as the prior; ReHARK achieves a further gain of 2.06% by incorporating GPT-3, visual prototypes, and multi-scale kernels.
- **vs. CoOp**: CoOp requires fine-tuning (computationally expensive and prone to overfitting), whereas ReHARK requires no training whatsoever.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of hybrid priors and multi-scale kernels is not revolutionary, but is systematic and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 benchmarks, complete ablations, and multiple backbones.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though the number of components requires careful integration to follow.
- **Value**: ⭐⭐⭐⭐ Provides a strong new baseline for training-free VLM adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)

</div>

<!-- RELATED:END -->
