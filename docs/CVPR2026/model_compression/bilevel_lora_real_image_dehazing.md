---
title: >-
  [Paper Note] Bilevel Layer-Positioning LoRA for Real Image Dehazing
description: >-
  [CVPR 2026][Model Compression][Image Dehazing] This work leverages CLIP's cross-modal capability to reformulate dehazing as a semantic alignment problem via the H2C loss, and employs bilevel optimization to automatically identify optimal LoRA injection layers (BiLaLoRA), enabling plug-and-play, parameter-efficient synthetic-to-real domain adaptation for image dehazing.
tags:
  - CVPR 2026
  - Model Compression
  - Image Dehazing
  - LoRA
  - CLIP
  - Bilevel Optimization
  - Unsupervised Domain Adaptation
date: 2026-05-08
content_hash: 7f525e661d14e947
---

# Bilevel Layer-Positioning LoRA for Real Image Dehazing

**Conference**: CVPR 2026
**arXiv**: [2603.10872](https://arxiv.org/abs/2603.10872)
**Code**: [GitHub](https://github.com/YanZhang-zy/BiLaLoRA)
**Area**: Model Compression
**Keywords**: Image Dehazing, LoRA, CLIP, Bilevel Optimization, Unsupervised Domain Adaptation

## TL;DR

This work leverages CLIP's cross-modal capability to reformulate dehazing as a semantic alignment problem via the H2C loss, and employs bilevel optimization to automatically identify optimal LoRA injection layers (BiLaLoRA), enabling plug-and-play, parameter-efficient synthetic-to-real domain adaptation for image dehazing.

## Background & Motivation

**Deep learning-based dehazing models achieve strong performance on synthetic data, yet the synthetic-to-real domain gap causes substantial performance degradation in real-world scenarios.** Two core limitations require attention: (1) real-world scenes lack paired clean images as ground truth, making effective unsupervised optimization signals unavailable; (2) full-model fine-tuning is computationally expensive and inflexible.

**Through empirical analysis, the authors identify a key phenomenon**: the location of "performance bottleneck layers" induced by the domain gap varies across model architectures and changes dynamically—manifesting in the last two convolutional blocks of the encoder in MSBDN, and in the third encoder block in DEA. **Fixed selection of LoRA injection layers is therefore suboptimal**, motivating a model-agnostic automatic localization method.

**These two problems are mutually coupled**: an effective unsupervised loss is a prerequisite for domain adaptation, while parameter-efficient fine-tuning is essential for practical deployment. The proposed H2C loss and BiLaLoRA strategy address these two limitations respectively.

## Method

### Overall Architecture

Building upon a pretrained dehazing model (e.g., DEA), the framework performs unsupervised domain adaptation on real hazy images via the H2C loss and BiLaLoRA. The process consists of two stages: (1) bilevel layer positioning—jointly optimizing LoRA weights $\omega$ and gating parameters $\alpha$ to rank the importance of candidate layers; (2) LoRA fine-tuning—fixing the selected top-$k$ layers and optimizing only $\omega$.

### Key Designs

1. **H2C Text-Guided Loss**:

    - **Function**: Provides an unsupervised optimization signal for the dehazing model in the absence of paired clean images.
    - **Mechanism**: Using CLIP image/text encoders, positive and negative prompts—"a clear photo" and "a photo with haze"—are defined. The cosine similarity between the image feature difference $\Delta V_{img} = V_{out} - V_{in}$ and the text direction difference $\Delta T_{text} = T_{pos} - T_{neg}$ serves as the loss: $L_{H2C} = 1 - \cos(\Delta V_{img}, \Delta T_{text})$
    - **Design Motivation**: The joint constraint from positive and negative prompts ensures directional dehazing—using the positive prompt alone causes color distortion, while the negative prompt alone leads to over-dehazing. For nighttime scenarios, the approach adapts readily by modifying the prompt to "nighttime haze."

2. **BiLaLoRA Bilevel Optimization**:

    - **Function**: Automatically searches for optimal LoRA injection layers, eliminating the need for manual selection.
    - **Mechanism**: Each candidate layer is assigned a learnable gating parameter $\alpha$ (constrained to $(0,1)$ via sigmoid). Bilevel optimization jointly learns layer selection (upper level: maximizing performance on a validation set) and LoRA weights (lower level: minimizing loss on the training set). A rank-one outer-product approximation of the Hessian simplifies hypergradient computation to require only first-order derivatives.
    - **Design Motivation**: Domain-gap bottleneck layers vary with model architecture and scene characteristics; single-level optimization cannot capture the hierarchical dependency between layer selection and weight learning.

3. **Plug-and-Play Multi-Domain Adaptation**:

    - **Function**: Supports rapid switching across target domains (daytime/nighttime).
    - **Mechanism**: Exploiting the inherent modularity of LoRA, a lightweight adapter is learned independently for each scene type without full re-training. Adaptation requires only 500 real daytime hazy images and 100 nighttime hazy images.
    - **Design Motivation**: Real-world dehazing scenarios are diverse, necessitating flexible domain switching rather than a single universal model.

### Loss & Training

The pretraining stage uses L1 loss on synthetic data with ground truth. The domain adaptation stage uses only the H2C loss (no ground truth). LoRA rank is set to 8 with scaling factor 2, and top-3 layers are selected. Learning rate is $1\times10^{-6}$, optimizer is Adam, with $256\times256$ random cropping, rotation, and flipping augmentation.

## Key Experimental Results

### Main Results

| Dataset | Metric | BiLaLoRA | Prev. SOTA | Gain |
|---------|--------|----------|------------|------|
| RTTS | FADE↓ | 0.752 | 0.845 (PHATNet) | −11% |
| RTTS | MUSIQ↑ | 61.77 | 59.61 (IPC) | +2.16 |
| URHI | MUSIQ↑ | 63.52 | 62.22 (IPC) | +1.30 |
| Fattal | MUSIQ↑ | 67.92 | 67.58 (IPC) | +0.34 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|------------|---------|
| Full fine-tuning vs. BiLaLoRA | 64.43 vs. 64.40 | Near-identical performance; training time reduced from 4.2 h to 0.94 h (−77.7%) |
| Bilevel optimization vs. naive joint training | 64.40 vs. 64.07 | Bilevel validation-set decoupling of layer selection and weights yields better generalization |
| Automatic layer localization vs. manual selection | 64.40 vs. 63.31 | Automatic search consistently outperforms heuristic selection |
| H2C without positive prompt | Color distortion | Significantly degrades output color fidelity |
| H2C without negative prompt | Over-dehazing | Model excessively "cleans" the image, producing artifacts |

### Key Findings

- Cross-architecture validation: effective across four architectures (MSBDN, DeHamer, ConvIR, DEA), confirming model-agnostic applicability.
- Cross-domain validation: consistent improvements when adapting from four different synthetic pretraining sources (ITS, OTS, Haze4K, RIDCP).
- Optimal layer count $k=3$; marginal gains diminish beyond three layers, with only +3% additional parameters.

## Highlights & Insights

Using CLIP as an unsupervised "judge" for dehazing losses is an elegant cross-modal application—defining a semantic trajectory via the difference between positive and negative text directions, which is generalizable to other restoration tasks. Bilevel optimization for automatic LoRA layer search frees parameter-efficient fine-tuning from manual layer-selection decisions.

## Limitations & Future Work

- No-reference evaluation metrics (FADE/MUSIQ) have limited reliability; paired full-reference evaluation is lacking.
- Validation is restricted to the dehazing task; extensions to deraining and denoising remain unexplored.
- The top-$k$ value is fixed at 3; different tasks and architectures may require different values.
- The H2C text prompt design is relatively simple; more sophisticated prompt engineering could yield further improvements.

## Related Work & Insights

- **vs. RIDCP (CVPR'23)**: Based on VQGAN priors; BiLaLoRA achieves substantially higher MUSIQ scores.
- **vs. IPC (CVPR'25)**: Uses iterative prediction–critic codebook decoding; BiLaLoRA outperforms IPC on overall metrics.
- **vs. CoA (CVPR'25)**: BiLaLoRA surpasses CoA on all metrics (FADE: 0.638 vs. 0.700; MUSIQ: 64.40 vs. 57.58).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The H2C loss design is elegant, and bilevel optimization for LoRA layer localization represents a valuable new perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Cross-architecture, cross-domain, and ablation experiments are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and derivations are complete.
- **Value**: ⭐⭐⭐ Practically valuable for real-world image dehazing and PEFT, though the application domain is relatively niche.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](iapl_aigenerated_image_detection_adaptive_prompt.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)

<!-- RELATED:END -->
