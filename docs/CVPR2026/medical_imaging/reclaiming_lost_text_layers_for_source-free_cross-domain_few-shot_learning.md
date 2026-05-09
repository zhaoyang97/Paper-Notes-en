---
title: >-
  [Paper Note] Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning
description: >-
  [CVPR2026][Medical Imaging][CLIP] This paper identifies "Lost Layers" in CLIP's text encoder — intermediate layers whose removal paradoxically improves performance under Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL). The authors demonstrate that these layers are not redundant but rather underutilized due to visual domain shift, and propose the VtT model to reclaim this information at both the layer and encoder levels, achieving state-of-the-art performance.
tags:
  - CVPR2026
  - Medical Imaging
  - CLIP
  - cross-domain few-shot learning
  - text encoder layer redundancy
  - vision-text fusion
  - state space model
  - gradient optimization
date: 2026-05-08
content_hash: f703a7d814c2e477
---

# Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning

**Conference**: CVPR2026  
**arXiv**: [2603.05235](https://arxiv.org/abs/2603.05235)  
**Code**: [zhenyuZ-HUST/CVPR26-VtT](https://github.com/zhenyuZ-HUST/CVPR26-VtT)  
**Area**: Medical Imaging / Cross-Domain Few-Shot Learning  
**Keywords**: CLIP, cross-domain few-shot learning, text encoder layer redundancy, vision-text fusion, state space model, gradient optimization

## TL;DR

This paper identifies "Lost Layers" in CLIP's text encoder — intermediate layers whose removal paradoxically improves performance under Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL). The authors demonstrate that these layers are not redundant but rather underutilized due to visual domain shift, and propose the VtT model to reclaim this information at both the layer and encoder levels, achieving state-of-the-art performance.

## Background & Motivation

**Practical demand for cross-domain few-shot learning**: Annotated data is extremely scarce in domains such as medical imaging and remote sensing, necessitating knowledge transfer from pretrained models. Source domain data is often inaccessible due to privacy and computational constraints, motivating the SF-CDFSL task.

**Cross-domain potential of CLIP**: CLIP, pretrained on large-scale image-text pairs, excels at downstream few-shot tasks; its text encoder is widely regarded as containing knowledge better suited for cross-domain transfer.

**Unexpected observation — Lost Layers**: Under the SF-CDFSL setting, removing certain intermediate layers of CLIP's text encoder (e.g., layers 6–7) consistently yields significant performance gains. This phenomenon is observed across all CLIP backbone variants and diverse fine-tuning methods.

**Revisiting the layer-redundancy assumption**: Prior works [40,49,57] treat these layers as redundant and discard them outright. However, the authors show through an "Emphasize" strategy that up-weighting these layers' outputs leads to better results, indicating that the information is beneficial but wasted.

**Visual domain shift as the root cause**: The Lost Layer phenomenon does not appear on ImageNet (source domain) but emerges immediately on ImageNet-R (cross-domain), confirming that changes in the visual domain prevent beneficial information in the text encoder from being utilized.

**Need to redirect the visual branch**: The core insight is not to discard Lost Layers but to reclaim the wasted pretrained knowledge in the text branch by "teaching the visual encoder to think like the text encoder."

## Method

### Overall Architecture: VtT (teach the Vision to Think like the Text)

VtT is a plug-and-play fine-tuning plugin comprising three modules:

1. **V-T Fusion** (layer-level fusion, yellow): integrates outputs from each layer of the visual and text encoders.
2. **TIA** (encoder-level absorption, pink): feeds fused features into the text encoder for knowledge absorption.
3. **DGSO** (dynamic gradient supervised optimization, orange): dynamically balances the classification and knowledge absorption objectives based on gradient conflict information.

All VtT parameters are removed after fine-tuning, resulting in **zero additional overhead at inference**.

### Key Design 1: V-T Cross-Layer Scanning Fusion

- The CLS tokens from each visual encoder layer and the EOS tokens from each text encoder layer are interleaved into a sequence $H_i = (f_i^l, t_i^l, f_i^{l-1}, t_i^{l-1}, \cdots, f_i^1, t_i^1)$, scanned from deep to shallow layers.
- A **State Space Model (SSM)** aggregates this sequence via a residual branch (AvgPool + MLP) and an SSM branch (MLP + positional encoding + 2-layer SSM + AvgPool), yielding $\mu_i = \mu_i^{\text{res}} + \mu_i^{\text{ssm}}$.
- Ablations show: deep→shallow scanning outperforms shallow→deep and bidirectional; SSM (58.2) outperforms MHA (57.2), RNN (57.2), and LSTM (57.4).

### Key Design 2: Text Encoder Information Absorption (TIA)

- The layer-level fusion output $\mu_i$ is mapped via a learnable Adapter to an "absorption token" $A_i$.
- $A_i$ **replaces** the class token [CLASS] in the text prompt, yielding $r_i' = [a][photo][of][a][A_i]$.
- This modified prompt is passed through the text encoder to produce $A_i'$, which incorporates both layer-level detail knowledge and encoder-level holistic knowledge.
- A loss $L_{\text{VtT}}$ is introduced to maximize the cosine similarity between $A_i'$ and visual features $f_i$, distilling textual knowledge into visual representations.

### Key Design 3: Dynamic Gradient Supervised Optimization (DGSO)

- **Gradient correction**: Computes the cosine similarity $C_\theta$ between the gradients of $L_{ce}$ and $L_{comb} = L_{ce} + \beta L_{VtT}$; if $C_\theta < 0$ (conflicting directions), $G_{comb}$ is projected onto the direction orthogonal to $G_{ce}$ to prevent degradation of the classification objective.
- **Dynamic loss combination**: Maintains a queue of $C$ values and computes a sliding-window mean $M_e$ (window length $\lambda=50$); when $M_e < 0$, $L_{VtT}$ is deactivated and not reactivated thereafter.

### Loss & Training

$$L_{comb} = L_{ce} + \beta \cdot L_{VtT}, \quad \beta = 7$$

where $L_{ce}$ is the standard cross-entropy classification loss and $L_{VtT}$ is the vision-text alignment distillation loss.

## Key Experimental Results

### Main Results (5-way 1-shot, 4 CDFSL datasets)

| Method | CropDisease | EuroSAT | ISIC | ChestX | Avg |
|---|---|---|---|---|---|
| CLIP-LoRA-Vision | 84.22 | 81.72 | 36.40 | 21.86 | 55.97 |
| **CLIP-LoRA + VtT (Ours)** | **87.00** | **85.01** | **38.20** | **22.70** | **58.23** |
| PE-Core-LoRA | 91.75 | 84.49 | 40.89 | 22.02 | 59.78 |
| **PE-Core-LoRA + VtT (Ours)** | **92.61** | **86.16** | **42.20** | **23.04** | **61.00** |

### Best Results (5-way 5-shot)

| Method | CropDisease | EuroSAT | ISIC | ChestX | Avg |
|---|---|---|---|---|---|
| CLIP-LoRA + VtT | 97.21 | 94.58 | 56.20 | 26.42 | 68.57 |
| PE-Core-LoRA + VtT | **98.36** | **94.67** | **60.03** | **27.05** | **70.05** |

### Ablation Study

| TIA | V-T Fusion | DGSO | Avg |
|---|---|---|---|
| ✗ | ✗ | ✗ | 55.9 (baseline) |
| ✓ | ✗ | ✗ | 56.9 (+1.0) |
| ✓ | ✓ | ✗ | 57.6 (+1.7) |
| ✓ | ✗ | ✓ | 57.6 (+1.7) |
| ✓ | ✓ | ✓ | **58.2 (+2.3)** |

### Key Findings

- **Lost Layers eliminated**: After applying VtT, using the complete text encoder yields optimal performance, and no layer-removal benefit remains (Figure 1(c)).
- **Improved attention maps**: VtT eliminates erroneous attention to non-semantic regions while preserving effective attention areas, improving the cosine similarity of vision-text alignment.
- **Cross-backbone generality**: Consistent improvements are observed across CLIP, SigLip2, and PE-Core backbones.
- **Low computational overhead**: Compared to Maple (3.1M parameters, 205G FLOPs), VtT requires only 3.9M parameters and 148.5G FLOPs while outperforming it by 5.1 percentage points.
- **Dynamic Loss Combining is effective**: With DLC, Avg = 58.2; without DLC, Avg drops to 57.2.

## Highlights & Insights

- **Novel insight**: The first work to discover and systematically analyze the Lost Layer phenomenon in CLIP's text encoder, establishing that its root cause is visual domain shift rather than information redundancy.
- **Elegant design**: VtT operates exclusively during training and is fully removed at inference, incurring zero additional overhead.
- **Sophisticated DGSO**: The combination of gradient correction and a dynamic stopping mechanism adaptively balances classification and knowledge absorption without requiring manual tuning of training schedules.
- **Comprehensive experiments**: Evaluated on 4 CDFSL datasets + 10 Meta-dataset benchmarks, 3 backbone architectures, multiple fine-tuning methods, and detailed ablations.

## Limitations & Future Work

- Although inference is overhead-free, the training phase requires additional computation for SSM forward passes and two separate gradient computations ($L_{ce}$ and $L_{comb}$), increasing training cost.
- Hyperparameters $\beta=7$ and $\lambda=50$ are fixed across all settings; sensitivity to varying degrees of domain shift is not thoroughly discussed.
- The Lost Layer analysis is primarily conducted on ViT-based CLIP architectures; applicability to CNN backbones or other VLM architectures remains unverified.
- Evaluations focus on specific cross-domain benchmarks (agriculture, remote sensing, medical imaging); more extreme domain shift scenarios (e.g., 3D data, video) are not explored.

## Related Work & Insights

- **SF-CDFSL**: Methods such as StepSTP [61] and LDC [32] focus on source-free fine-tuning strategies but do not investigate the utilization of text encoder layers.
- **PEFT methods**: CoOp [75], Maple [25], and CLIP-LoRA [66] provide diverse fine-tuning strategies; VtT can be stacked on top of these as a plugin.
- **Layer redundancy research**: Works [40,49,57,30,15] observe that removing certain layers does not significantly degrade performance and adopt deletion strategies; this paper is the first to demonstrate that these layers are actually beneficial and can be reclaimed.
- **Knowledge distillation / modality fusion**: The TIA module draws inspiration from modality conversion methods [41]; DGSO's gradient correction is motivated by gradient conflict resolution techniques in multi-task learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The discovery and systematic analysis of the Lost Layer phenomenon, along with the "beneficial but underutilized" insight, are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4+10 datasets, 3 backbones, multiple baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — The analytical pipeline (discovery → attribution → resolution) is clearly structured, with intuitive figures and tables.
- Value: ⭐⭐⭐⭐ — Provides a new perspective on layer-level information utilization for cross-domain VLM transfer; the plug-in design offers strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[AAAI 2026\] MPA: Multimodal Prototype Augmentation for Few-Shot Learning](../../AAAI2026/medical_imaging/mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)

</div>

<!-- RELATED:END -->
