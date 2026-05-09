---
title: >-
  [Paper Note] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark
description: >-
  [NeurIPS 2025][Image Restoration][Thermal Infrared Image Enhancement] To address the challenge of coupled degradations (low contrast, blur, and noise) in thermal infrared (TIR) images, this paper proposes PPFN, a progressive prompt fusion network with a dual-prompt design, along with the Selective Progressive Training (SPT) strategy. The authors also construct HM-TIR, the first large-scale multi-scene TIR benchmark dataset. The proposed method achieves an 8.76% PSNR improvement in composite degradation scenarios.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Thermal Infrared Image Enhancement
  - Prompt Learning
  - All-in-One Restoration
  - Progressive Training
  - TIR Benchmark
date: 2026-05-08
content_hash: 1b41e317cdbd27f9
---

# Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark

**Conference**: NeurIPS 2025  
**arXiv**: [2510.09343](https://arxiv.org/abs/2510.09343)  
**Code**: [https://github.com/Zihang-Chen/HM-TIR](https://github.com/Zihang-Chen/HM-TIR)  
**Area**: Image Restoration  
**Keywords**: Thermal Infrared Image Enhancement, Prompt Learning, All-in-One Restoration, Progressive Training, TIR Benchmark  

## TL;DR
To address the challenge of coupled degradations (low contrast, blur, and noise) in thermal infrared (TIR) images, this paper proposes PPFN, a progressive prompt fusion network with a dual-prompt design, along with the Selective Progressive Training (SPT) strategy. The authors also construct HM-TIR, the first large-scale multi-scene TIR benchmark dataset. The proposed method achieves an 8.76% PSNR improvement in composite degradation scenarios.

## Background & Motivation

**Background**: Thermal infrared imaging detects thermal radiation from objects (8–14 μm wavelength), operates independently of external light sources, and penetrates smoke and occlusions, making it widely applicable to object detection, semantic segmentation, and autonomous driving. Existing TIR enhancement methods are predominantly designed for individual degradation types—denoising, deblurring, and contrast enhancement are each handled independently.

**Limitations of Prior Work**: (a) Single-degradation methods cannot handle composite degradation scenarios, as real-world TIR images typically exhibit simultaneous noise, blur, and low contrast; (b) All-in-one (AIO) restoration methods designed for visible-light images (e.g., PromptIR, DA-CLIP) perform poorly when directly applied to TIR, as the imaging models and degradation patterns of infrared and visible-light modalities are fundamentally different; (c) Existing TIR datasets are limited in scene diversity, resolution, and scale, and lack coverage of multiple degradation types.

**Key Challenge**: TIR degradations follow a physically grounded cascaded structure (low contrast → blur → noise), yet existing methods neither model this cascade ordering nor distinguish between single and composite degradation scenarios.

**Goal**: (a) How can a single model handle multiple TIR degradations and their composite combinations? (b) How can the model be made aware of both degradation type and scene type? (c) How can a sufficiently large and diverse TIR benchmark be constructed?

**Key Insight**: Grounded in the physical imaging process of TIR, degradations are decomposed into a three-step cascade (contrast degradation → blur → noise). A prompt mechanism is employed to distinguish degradation type and scene type, and degradations are progressively removed in reverse order.

**Core Idea**: Unified All-in-One TIR image enhancement is achieved through dual-prompt (degradation type + scene type) fusion for feature modulation, combined with selective progressive training for reverse-order degradation removal.

## Method

### Overall Architecture
A degraded TIR image is processed by a backbone network (e.g., Restormer). Two novel modules are introduced: (1) **PPFN** (Progressive Prompt Fusion Network), which fuses degradation-type prompts and scene-type prompts and injects them into each layer of the backbone via channel-wise modulation; and (2) **SPT** (Selective Progressive Training), which iteratively removes composite degradations in reverse physical order during training and inference, while applying standard training for single degradations. The output is the enhanced, clean TIR image.

### Key Designs

1. **Dual Prompt Processing**

    - **Function**: Provides the model with two types of prior information: "What degradation is present in this image?" and "Is this a single or composite degradation?"
    - **Mechanism**: Degradation-specific prompts are defined as $\mathbf{P}_{deg} = \{\mathbf{p}^n_{deg}, \mathbf{p}^b_{deg}, \mathbf{p}^c_{deg}\}$ (noise, blur, contrast) and type-specific prompts as $\mathbf{P}_{type} = \{\mathbf{p}^s_{type}, \mathbf{p}^h_{type}\}$ (single, composite). Each prompt is encoded into a feature vector via lightweight encoders $\mathbf{E}_{deg}$ and $\mathbf{E}_{type}$.
    - **Design Motivation**: A single prompt is insufficient to distinguish between "denoising in a single-degradation scenario" and "denoising in a composite-degradation scenario." The cross-combination of dual prompts enables the model to precisely perceive its current operational context. Ablation experiments confirm that using only degradation prompts yields a +0.29 dB improvement, while dual prompts yield +0.40 dB.

2. **Prompt Fusion Module**

    - **Function**: Fuses two prompt features to generate channel-wise modulation parameters $\gamma$ and $\beta$, which are injected into each layer of the backbone.
    - **Mechanism**: The two prompt features are concatenated and passed through a linear layer followed by a nonlinear activation to produce the fused feature $\mathbf{F}_p = \phi(\mathcal{W}_{fusion}(\text{Cat}(\mathbf{F}^p_{deg}, \mathbf{F}^p_{type})))$. A subsequent linear layer splits the result into $\gamma$ and $\beta$, which modulate the $l$-th layer features via FiLM: $\tilde{\mathbf{F}}_l = \mathbf{F}_l \otimes (1 + \gamma_l) + \beta_l$.
    - **Design Motivation**: Concatenation with nonlinear activation outperforms simple multiplication (the multiply variant yields −0.08 dB PSNR and lower SSIM in ablations). FiLM modulation is plug-and-play and can be seamlessly integrated into arbitrary backbone networks.

3. **Selective Progressive Training (SPT)**

    - **Function**: Distinguishes between single and composite degradation scenarios and applies different training and inference strategies accordingly.
    - **Mechanism**: For composite degradations, removal is performed step by step in the reverse of the physical degradation order (contrast → blur → noise), i.e., denoising first, then deblurring, then contrast enhancement during inference. During training, the input is the fully degraded image $\mathbf{I}^N_d$, and at each step $k$, the target is the image degraded up to step $k{-}1$. Crucially, the input for the next iteration uses the output of the previous iteration (with `stop_gradient`) rather than the intermediate degraded image directly, preventing residual degradations from interfering; gradients from all steps are accumulated and applied in a single update. For single degradations, standard training is used.
    - **Design Motivation**: Naively applying iterative training to the baseline reduces performance by 0.23 dB, as simple looping causes the model to overfocus on a particular degradation. SPT addresses training instability through gradient accumulation and `stop_gradient`, converging optimally at 3 iterations.

### Loss & Training
- L1 loss is used as the reconstruction objective
- Adam optimizer with $\beta_1=0.9$, $\beta_2=0.999$
- Initial learning rate $8\times10^{-5}$, cosine annealing to $10^{-6}$
- Batch size 4, patch size 256×256, with random cropping and flipping
- Training for 300 epochs on 4 × 4090D GPUs
- Degradation synthesis uses the Gated Degradation pipeline with gate probability 0.8

## Key Experimental Results

### Main Results

| Method | Type | Normal Set PSNR/SSIM | Iray NIMA↑/MUSIQ↑/NIQE↓ |
|------|------|---------------------|--------------------------------------|
| WFAF | TIR Single-Degradation | Low (severe artifacts) | 3.73 / 25.13 / 10.35 |
| LRSID | TIR Single-Degradation | Low | 3.57 / 24.21 / 8.68 |
| DA-CLIP | Visible-Light AIO | Moderate | 3.70 / 27.79 / 9.19 |
| DiffUIR | Visible-Light AIO | Moderate | 3.59 / 26.81 / 9.34 |
| Baseline (Restormer) | Backbone | 23.28/0.796 | 3.58 / 27.78 / 8.78 |
| **Ours (PPFN+SPT)** | **Ours** | **25.32/0.818** | **3.83 / 30.91 / 8.47** |

Average PSNR gains on the Normal Set across five backbone networks: FocalNet +1.41 dB, UFormer +0.82 dB, NAFNet +1.45 dB, XRestormer +1.21 dB, Restormer +2.04 dB.

### Ablation Study

| Configuration | PSNR | SSIM | Notes |
|------|------|------|------|
| Baseline (Restormer) | 22.87 | 0.757 | No prompt, no SPT |
| + Iterative training (no prompt) | 22.64 | 0.752 | Naive iteration hurts performance |
| + Degradation prompt (DSP only) | 23.16 | 0.764 | +0.29 dB |
| + Dual prompt w/o nonlinearity | 23.15 | 0.765 | Removing activation degrades results |
| + Dual prompt, multiply fusion | 23.14 | 0.763 | Multiplication inferior to concatenation |
| + PPFN (iter=1) | 14.55 | 0.613 | Single iteration insufficient |
| **+ PPFN (iter=3, full)** | **23.27** | **0.764** | Full model achieves best performance |

### Key Findings
- Setting SPT iterations to 3 (corresponding to three degradation types) is optimal; 1 or 2 iterations result in a sharp PSNR drop to ~14.5, indicating that all degradations must be progressively removed for convergence.
- Using an incorrect type prompt (e.g., single-degradation prompt for composite degradation) leaves residual degradations unremoved; an incorrect removal order also noticeably degrades performance.
- PPFN is plug-and-play: consistent improvements are observed across all five backbone networks, with the largest gain on Restormer (+8.76%).

## Highlights & Insights
- **Physics-grounded degradation modeling**: TIR degradations are decomposed into a cascade of low contrast → blur → noise, and the model removes them in reverse order during inference. This paradigm is transferable to any scenario with a well-defined degradation cascade.
- **Dual-prompt design**: Two dimensions—degradation type and scene type—are simultaneously encoded and injected into the backbone via FiLM modulation, offering a concise yet effective approach that generalizes to other multi-task image processing problems.
- **HM-TIR dataset**: Comprising 1,503 TIR images at 640×512 resolution, covering 8 scene categories and 5 degradation types, this is currently the largest and most diverse benchmark for TIR image enhancement.

## Limitations & Future Work
- The degradation order is fixed to three steps; the method may fail when real-world degradation ordering deviates from this assumption.
- Prompts require manual specification of degradation type and scene type; automatic degradation perception has not been realized—integrating a degradation estimation network is a natural next step.
- Only L1 loss is used, without perceptual or adversarial losses, which may limit the upper bound of visual quality.
- Although HM-TIR is relatively large, the degradations are synthetically generated rather than being real-world composite degradations captured in the wild.

## Related Work & Insights
- **vs. PromptIR**: PromptIR employs a single prompt for multiple degradations but does not distinguish between single and composite degradation scenarios; the dual-prompt design in PPFN offers finer-grained control.
- **vs. DA-CLIP/DiffUIR**: Visible-light AIO methods perform poorly when directly applied to TIR (MUSIQ gap of 3–4 points), demonstrating that cross-modality transfer requires task-specific design.
- **vs. IDR**: IDR explores optimization via component clustering but lacks physically grounded degradation modeling for TIR; the physics-based cascade approach in this paper is more targeted.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-prompt + SPT combination is novel for TIR scenarios, though the underlying ideas of FiLM modulation and progressive training have established precedents
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five backbones, multiple benchmarks, comprehensive ablations, and prompt sensitivity analysis
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly articulated physical motivation
- Value: ⭐⭐⭐⭐ Both the HM-TIR dataset and the PPFN module represent tangible contributions to the TIR community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](../../CVPR2026/image_restoration/evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[NeurIPS 2025\] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights](gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](../../ICCV2025/image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)

</div>

<!-- RELATED:END -->
