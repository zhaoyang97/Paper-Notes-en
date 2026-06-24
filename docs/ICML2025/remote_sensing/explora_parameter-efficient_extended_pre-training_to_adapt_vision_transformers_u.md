---
title: >-
  [Paper Note] ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts
description: >-
  [ICML2025][Remote Sensing][Parameter-Efficient Fine-Tuning] ExPLoRA is proposed to continue self-supervised pre-training on a target domain in a parameter-efficient manner by unfreezing 1-2 ViT blocks and applying LoRA to the remaining layers. Under domain shift scenarios like remote sensing, it outperforms SOTAs that undergo full pre-training from scratch, while utilizing <10% of the parameters.
tags:
  - "ICML2025"
  - "Remote Sensing"
  - "Parameter-Efficient Fine-Tuning"
  - "LoRA"
  - "Domain Adaptation"
  - "Vision Foundation Models"
  - "Self-Supervised Pre-Training"
date: 2026-05-08
content_hash: 1383635cc9cef830
---

# ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts

**Conference**: ICML2025  
**arXiv**: [2406.10973](https://arxiv.org/abs/2406.10973)  
**Code**: [https://samar-khanna.github.io/ExPLoRA/](https://samar-khanna.github.io/ExPLoRA/)  
**Area**: Remote Sensing  
**Keywords**: Parameter-Efficient Fine-Tuning, LoRA, Domain Adaptation, Vision Foundation Models, Remote Sensing, Self-Supervised Pre-Training

## TL;DR

ExPLoRA is proposed to continue self-supervised pre-training on a target domain in a parameter-efficient manner by unfreezing 1-2 ViT blocks and applying LoRA to the remaining layers. Under domain shift scenarios like remote sensing, it outperforms SOTAs that undergo full pre-training from scratch, while utilizing <10% of the parameters.

## Background & Motivation

- **Core Problem**: Large Vision Foundation Models (VFMs) such as DinoV2 and MAE perform exceptionally well on natural images, but their performance drops significantly under domain shift scenarios like remote sensing and medical imaging. Existing solutions perform full pre-training of VFMs from scratch on the new domain, which is computationally expensive (e.g., full pre-training of ViT-L requires over 960 GPU hours).
- **LoRA Limitations**: Standard LoRA assumes that the weight updates $\Delta W$ lie within a low-rank subspace. While this assumption holds when the source and target domain distributions are similar, it often fails under large domain shifts, such as natural images $\rightarrow$ multi-spectral remote sensing.
- **Research Question**: Can VFMs be adapted to unsupervised pre-training on a new domain parameter-efficiently using only a small fraction of parameters while preserving the pre-trained knowledge from natural images?

## Method

### Core Idea

The final target domain task weights are decomposed into three components:

$$W_T^{(\tau)} \approx W_S + \Delta_T + \Delta^{(\tau)}$$

where $W_S$ represents the source domain pre-trained weights, $\Delta_T$ is the domain-adaptation unsupervised update (the ExPLoRA stage), and $\Delta^{(\tau)}$ is the supervised fine-tuning update for downstream tasks.

### ExPLoRA Algorithm

1. **Initialization**: Initialize the ViT with pre-trained weights $W_S$ from DinoV2 or MAE.
2. **Selective Unfreezing**: Partition the $L$-layer ViT blocks into two sets:
    - $\mathcal{U}$ (e.g., $\{L\}$ or $\{1, L\}$): Fully unfreeze all parameters.
    - $\mathcal{L} \setminus \mathcal{U}$: Freeze the backbone weights, apply LoRA (rank $r$) solely on the Q and V attention matrices, and unfreeze normalization layers across all blocks.
3. **Continued Self-Supervised Pre-Training**: Train all unfrozen parameters on unlabeled target domain data $\mathcal{X}_T$ using the same unsupervised loss $\mathcal{C}_S$ (such as DinoV2 loss or MAE reconstruction loss) used for $W_S$.

### Optimization Objective

$$\Delta_T = \arg\min_{\theta \in \Theta(\mathcal{U}, r)} \left( \min_\psi \sum_{\mathbf{x} \in \mathcal{X}_T} \mathcal{C}_S\left(g_\psi\left(f_\theta(\mathbf{x}; W_S)\right), \mathbf{x}\right) \right)$$

where $f_\theta$ is the ViT encoder, $g_\psi$ is the decoder (e.g., Dino/MAE head), and $\Theta(\mathcal{U}, r)$ constrains the trainable parameter space.

### Downstream Fine-Tuning

After ExPLoRA, the target weights $W_T^* = W_S + \Delta_T$ are obtained. The decoder $g_\psi$ is discarded, and the LoRA matrices are merged back into the main ViT body, maintaining the original architecture. Downstream tasks can flexibly employ linear probing, LoRA fine-tuning, or full fine-tuning.

### Multi-Spectral Extension

For multi-spectral ViTs like SatMAE, positional encodings and patch embedding weights for each channel group must be additionally unfrozen, because $W_S$ is only pre-trained on RGB and cannot directly initialize multi-channel inputs.

## Key Experimental Results

### fMoW-RGB Classification (ViT-L, 62 classes)

| Method | Pre-training Params | Fine-tuning Params | Pre-training GPU h | Top-1 Acc |
|------|-----------|---------|-------------|-----------|
| ScaleMAE (Full) | 303.3M | 303.3M | 960 | 77.80% |
| SatMAE (Full) | 303.3M | 303.3M | 960 | 77.78% |
| DinoV2 + LoRA-r8 | – | 0.8M | – | 78.08% |
| DinoV2 + AdaLoRA-r8 | – | 1.2M | – | 78.87% |
| **D-[L]-r64 + LoRA-r8** | **18.7M** | **0.8M** | **100** | **79.28%** |

### fMoW-RGB Linear Probing

| Method | Top-1 Acc |
|------|-----------|
| SatMAE (Pre-trained from scratch) | 65.94% |
| DinoV2 | 69.00% |
| **D-[L]-r64 (ExPLoRA)** | **77.48%** |

The linear probing accuracy of ExPLoRA improves by **+8.48%** over DinoV2, outperforming all methods pre-trained from scratch via full pre-training.

### Ablation Study Highlights

| Configuration | Params | GPU h | LP Acc |
|------|--------|-------|--------|
| DinoV2 Baseline | – | – | 69.00% |
| Full Pre-training from Scratch | 303.3M | 1200 | 54.29% |
| Unfreeze [L] + No LoRA | 12.7M | 90 | 74.83% |
| Unfreeze [L] + LoRA-r64 Q,V | 18.7M | 100 | 77.48% |
| Unfreeze [1,L-1,L] + LoRA-r64 | 43.4M | 180 | 78.04% |

### Multi-Spectral fMoW-Sentinel

ExPLoRA (M-[1,L]-r32) achieves 60.15% top-1 accuracy with only 29.7M pre-training parameters and 320 GPU hours. This is close to the performance of SatMAE pre-trained from scratch (303.3M parameters, 1150 GPU hours, 61.48%), while reducing the computational cost by approximately 3.6×.

## Highlights & Insights

1. **High Parameter Efficiency**: Achieves performance superior to full-pre-training SOTAs using only 6% of the parameters (18.7M vs 303.3M) and requiring 8× less computational cost.
2. **Paradigm Shift in Knowledge Transfer**: Conceptually demonstrates that pre-training from scratch on a new domain is unnecessary; rather, efficient adaptation from natural image models offers a superior pathway.
3. **Applying LoRA directly to the Pre-training Stage**: This is the first work to systematically utilize LoRA for domain adaptation during unsupervised pre-training, rather than conventional supervised fine-tuning.
4. **Flexible Composition**: ExPLoRA is orthogonal to and can be combined with downstream PEFT methods (such as LoRA, SA2VP, VPT, etc.).
5. **Key Finding**: Applying LoRA solely to the Q and V matrices yields optimal performance; applying it to MLPs or all projection matrices leads to severe performance degradation.

## Limitations & Future Work

- **Limited Domain Coverage**: While deep case studies are conducted mainly in the remote sensing domain, validations on other domains like medicine or agriculture are preliminary and confined to the WILDS benchmark.
- **Restricted to ViT Architectures**: The methodology is coupled with the Transformer block structure, leaving its applicability to CNNs or hybrid architectures unexplored.
- **Empirical Block Unfreezing**: Deciding which blocks to unfreeze (e.g., first or last) relies on empirical ablation studies, lacking theoretical guidance or automatic selection mechanisms.
- **Performance Gap on Multi-Spectral Data**: The method remains slightly behind SatMAE pre-trained from scratch on fMoW-Sentinel, indicating that larger domain shifts in multi-spectral data present continued challenges.
- **Constrained Pre-training Objectives**: ExPLoRA requires using the exact same self-supervised objective function as the source model, preventing flexible replacement with targets more tailored to the target domain.

## Rating

- Novelty: ⭐⭐⭐⭐ — Highly novel perspective of employing LoRA for unsupervised domain adaptation pre-training, backed by a clear $W_S + \Delta_T + \Delta^{(\tau)}$ decomposition framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies (investigating block choices, rank size, and LoRA locations), validation across multiple datasets, and thorough comparisons of computational costs.
- Writing Quality: ⭐⭐⭐⭐ — Clear notations, concise algorithm pseudo-code, and highly informative diagrams.
- Value: ⭐⭐⭐⭐⭐ — Provides an out-of-the-box, highly efficient solution for domain adaptation in resource-constrained environments, offering strong practicality for remote sensing applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](../../CVPR2025/remote_sensing/meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/remote_sensing/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2026\] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation](../../CVPR2026/remote_sensing/crossearth-gate_fisher-guided_adaptive_tuning_engine_for_efficient_adaptation_of.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](../../ICCV2025/remote_sensing/rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)

</div>

<!-- RELATED:END -->
