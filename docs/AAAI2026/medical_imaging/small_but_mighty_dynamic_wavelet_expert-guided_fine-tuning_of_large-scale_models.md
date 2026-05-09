---
title: >-
  [Paper Note] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation
description: >-
  [Medical Imaging] WEFT proposes a lightweight fine-tuning paradigm guided by dynamic wavelet experts, adapting frozen large-scale visual foundation models to optical remote sensing image segmentation with only 4.52% trainable parameters, surpassing 21 state-of-the-art methods on three ORSIs datasets.
tags:
  - Medical Imaging
date: 2026-05-08
content_hash: c56fdc0befe06f3f
---

# Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation

## Paper Information

- **Conference**: AAAI 2026
- **arXiv**: [2601.09108](https://arxiv.org/abs/2601.09108)
- **Code**: [https://github.com/CSYSI/WEFT](https://github.com/CSYSI/WEFT)
- **Area**: Medical Imaging
- **Keywords**: remote sensing object segmentation, wavelet experts, large model fine-tuning, parameter efficiency, sparse attention, boundary awareness

## TL;DR

WEFT proposes a lightweight fine-tuning paradigm guided by dynamic wavelet experts, adapting frozen large-scale visual foundation models to optical remote sensing image segmentation with only 4.52% trainable parameters, surpassing 21 state-of-the-art methods on three ORSIs datasets.

## Background & Motivation

The core tension in optical remote sensing image (ORSIs) object segmentation:

**Advantages of large models**: Deeper and larger visual foundation models (e.g., UniPerceiver-L with 303M parameters) provide stronger discriminative features, yet most existing methods rely on medium-scale pretrained models (Swin-B 88M, PVTv2-B4 63M).

**Bottleneck of full-parameter fine-tuning**: Full-parameter fine-tuning (FPFT) of large models leads to GPU memory explosion and prohibitive computational costs, especially under high-resolution inputs or large batch sizes.

**Unique challenges of ORSIs**: Remote sensing objects exhibit arbitrary orientations, dramatic scale variation, and dense distribution against complex backgrounds.

Existing parameter-efficient fine-tuning methods (LoRA, VPT, Adapter) do not adequately address the specific requirements of remote sensing tasks, including multi-scale features, boundary details, and spatial structure.

## Method

### Overall Architecture

WEFT adopts a dual-branch architecture: a frozen UniPerceiver-L backbone (extracting frozen features) paired with a lightweight trainable branch (extracting task-specific trainable features). The two types of features are interactively fused through an EC Adapter and subsequently fed into a Mask2Former-style mask decoder.

### Key Designs

#### 1. Task-Specific Wavelet Expert (TWE) Extractor

**Wavelet convolution modeling**:
- After downsampling the input image, wavelet convolutions model features along four directions (HH, HL, LH, LL).
- Depthwise convolutions with varying kernel sizes ($2n-1$, $n=1,...,7$) generate 7 wavelet experts with different receptive fields, $\{E_n^\diamond\}_{n=1}^7$.
- Spatial resolution is recovered via Inverse Wavelet Transform (IWT), which is more lightweight than standard convolution and increases feature diversity.

**Top-K Expert Router (TER)**:
- Expert weights $\alpha$ are obtained via global average pooling, a linear layer, and Softmax.
- The top-4 experts with the highest scores are selected; their weights are normalized and used for weighted aggregation:
$$\mathcal{F}_1^\diamond = \mathcal{C}_1(f_m + \sum_{u \in \mathcal{T}} \tilde{\alpha}_u \cdot E_u^\diamond)$$
- Key insight: not all wavelet experts are beneficial. Small objects do not require large receptive fields (which introduce ambiguity), and large objects do not benefit from small receptive fields (which yield incomplete understanding).

Multi-scale trainable features $\{\mathcal{F}_2^\diamond, \mathcal{F}_3^\diamond, \mathcal{F}_4^\diamond\}$ are progressively generated through a hierarchical structure.

#### 2. Expert-Guided Conditional (EC) Adapter

The EC Adapter consists of three sub-components:

**(a) Deformable Attention Injection**:
- Deformable attention is applied to inject task-specific information from the trainable features into the frozen features:
$$\hat{\mathcal{F}}_1^* = DeformAttn(LN(\mathcal{F}_1^*), LN(\tilde{\mathcal{F}_1^e}))$$

**(b) Boundary-Aware Subspace Token Optimizer (ESTO)**:
- Features are divided into $H$ subspaces, each computing token-wise similarity and attention independently.
- A boundary mask $\mathbf{M}$ is estimated from channel-wise variance: high-variance tokens correspond to structurally salient regions such as edges and contours.
- Optimization intensity is adaptively controlled via gating and residual connections:
$$\widetilde{\mathcal{F}}_1^* = \delta \cdot \widetilde{\mathbf{T}}_1^* + \hat{\mathcal{F}}_1^*$$

**(c) Spatial-Aware Expert Enhancer (SEE)**:
- Three branches enhance spatial awareness in the trainable features:
    - **Directional Laplacian filter**: captures second-order spatial variations (boundaries, textures).
    - **Adaptive max pooling**: extracts globally salient patterns.
    - **Multi-scale operations**: depthwise convolutions with kernel sizes 3/5/7 capture multi-scale context.
- Dynamic weights govern the three-branch outputs: $\tilde{\mathcal{F}}_i^\diamond = \mathcal{F}_i^\diamond + \sum_{z \in \{d,a,m\}} w_z \cdot \mathcal{F}_i^z$

The fine-tuning process comprises 4 stages, iteratively updating frozen and trainable features.

### Loss & Training

$$\mathcal{L}_{all} = 5 \cdot \mathcal{L}_{bce} + 2 \cdot \mathcal{L}_{dice}$$

A weighted combination of binary cross-entropy and Dice coefficient losses.

## Experiments

### Experimental Setup

- Backbone: UniPerceiver-L (frozen, 303M parameters)
- Trainable parameters: only 14.37M (4.52% of total parameters)
- Training: 4× RTX 4090 24GB, input 512×512, batch size 6, AdamW optimizer, 80K iterations
- Evaluation metrics: mIoU, AFm, mDice, Sm, MAE

### Main Results

**ORSIs Segmentation (comparison with 21 SOTA methods)**:

| Method | Trainable Params | ORSSD mIoU | EORSSD mIoU | ORSIs-4199 mIoU |
|--------|-----------------|-----------|------------|----------------|
| DPU-Former | 44.20M | 0.8728 | 0.8268 | 0.7961 |
| BCARNet | 24.00M | 0.8600 | 0.8248 | 0.7795 |
| TLCKDNet | 52.09M | 0.8689 | 0.8380 | - |
| **WEFT (Ours)** | **14.37M** | **0.8964** | **0.8621** | **0.7999** |

WEFT surpasses the second-best method by 2.70% mIoU on ORSSD and 2.88% on EORSSD. MAE improves by 10.71%, 12.50%, and 10.50% respectively.

**Cross-Scene Generalization (7 additional datasets)**:

| Scene | Dataset | WEFT mIoU | Prev. SOTA mIoU |
|-------|---------|----------|----------------|
| Camouflage detection | CAMO | 0.8308 | 0.8090 (ZoomXNet) |
| Camouflage detection | COD10K | 0.7984 | 0.7795 (ZoomXNet) |
| Saliency detection | PASCAL-S | 0.8359 | 0.8232 (VST++) |
| Polyp segmentation | CVC-300 | 0.8502 | 0.8414 (DPU-Former) |
| Polyp segmentation | Kvasir | 0.8875 | 0.8698 (CFANet) |

### Ablation Study

**Component-wise contribution (EORSSD mIoU)**:

| Config | Base | +TWE | +ESTO | +SEE | TWE+ESTO | TWE+SEE | ESTO+SEE | Full |
|--------|------|------|-------|------|----------|---------|----------|------|
| mIoU | 0.769 | 0.825 | 0.820 | 0.805 | 0.832 | 0.831 | 0.828 | **0.862** |

The full model achieves a 12.09% improvement over the baseline.

**Number of experts**: 4 experts yields optimal performance (gains are significant from 1→2→4, while 6 experts leads to degradation, validating the TER routing strategy).

**Fine-tuning strategy comparison**: WEFT outperforms LoRA, VPT, and Adapter with the fewest trainable parameters (14.37M).

### Key Findings

- GPU memory is reduced by approximately 26.41% and training speed improves by 14.66%, with performance on par with full-parameter fine-tuning.
- Large model + parameter-efficient fine-tuning substantially outperforms medium-scale model + full-parameter fine-tuning.
- Strong cross-scene generalization: WEFT achieves state-of-the-art performance on camouflage detection, saliency detection, and polyp segmentation—tasks unrelated to remote sensing.

## Highlights & Insights

1. **Domain adaptability of wavelet experts**: Wavelet transforms are inherently well-suited for multi-scale feature extraction in remote sensing scenarios, offering greater task specificity than generic adapters.
2. **Elegance of the Top-K routing strategy**: Dynamically selecting experts with appropriate receptive fields elegantly addresses the scale variability characteristic of remote sensing objects.
3. **Boundary-aware design of ESTO**: Inferring boundary locations from channel-wise variance is a concise and effective approach that requires no additional annotations.
4. **Extreme parameter efficiency**: Only 14.37M trainable parameters outperforms all competing methods (the largest, PA-KRN, has 141M), substantiating the "small but mighty" thesis.

## Limitations & Future Work

- Only UniPerceiver-L is used as the frozen backbone; other large models (e.g., DINOv2, SAM) are not explored.
- The selection of 4 out of 7 wavelet experts via fixed top-K may not be optimal; more adaptive selection strategies may exist.
- Remote sensing datasets are relatively small in scale (ORSSD has only 200 test images), limiting large-scale validation.
- Evaluation on medical scenarios such as polyp segmentation involves very limited test data (CVC-300 has only 62 test images).
- The classification under "medical imaging" is inaccurate—the paper's core contribution is remote sensing segmentation, with medical imaging serving only as an extension experiment.

## Related Work & Insights

- **Remote sensing segmentation**: LVNet, GeleNet, DPU-Former, TLCKDNet (PVTv2 backbone)
- **Large model fine-tuning**: LoRA (low-rank matrices), VPT (learnable prompts), Adapter
- **Visual foundation models**: UniPerceiver, DINOv2, SAM

## Rating

⭐⭐⭐⭐ (4/5)

- Clear problem motivation (the tension between large models and remote sensing), with a complete and efficient solution.
- Highly thorough experiments: comparison with 21 SOTA methods, 7 cross-scene datasets, and detailed ablation studies.
- The simultaneous gain in parameter efficiency and performance is impressive.
- Deductions: the theoretical analysis of the wavelet expert design lacks depth, and cross-backbone generalization is not verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)

</div>

<!-- RELATED:END -->
