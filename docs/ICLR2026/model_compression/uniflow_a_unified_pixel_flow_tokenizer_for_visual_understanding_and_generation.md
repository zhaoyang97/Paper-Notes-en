---
title: >-
  [Paper Note] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation
description: >-
  [Model Compression] This paper proposes UniFlow, a universal unified tokenizer that preserves semantic understanding via hierarchical adaptive self-distillation and achieves high-fidelity reconstruction via a lightweight patch-wise pixel flow decoder. UniFlow achieves state-of-the-art performance on both understanding and generation across 13 benchmarks. The 7B UniFlow-XL surpasses the 14B TokenFlow-XL by 6.05% on average understanding benchmarks while using 40% less training data.
tags:
  - Model Compression
date: 2026-05-08
content_hash: 2a29d0fadd2ca1cf
---

# UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2510.10575](https://arxiv.org/abs/2510.10575)
- **Code**: Not released
- **Area**: Visual Understanding & Generation / Unified Tokenizer
- **Keywords**: unified tokenizer, visual encoder, flow matching, self-distillation, image reconstruction and generation

## TL;DR

This paper proposes UniFlow, a universal unified tokenizer that preserves semantic understanding via hierarchical adaptive self-distillation and achieves high-fidelity reconstruction via a lightweight patch-wise pixel flow decoder. UniFlow achieves state-of-the-art performance on both understanding and generation across 13 benchmarks. The 7B UniFlow-XL surpasses the 14B TokenFlow-XL by 6.05% on average understanding benchmarks while using 40% less training data.

## Background & Motivation

Visual understanding and generation are two core tasks in computer vision. Current approaches face a **unified tokenizer dilemma**:

**Dual-encoder paradigm** (e.g., TokenFlow): separate semantic and pixel encoders, leading to model redundancy and low training efficiency.

**Frozen VFM + latent diffusion decoder** (e.g., EMU2, BLIP3-o): inherits understanding capability, but the semantic encoder cannot model fine-grained details and is constrained by the pre-trained VAE ceiling.

**Unified encoder fine-tuning** (e.g., VILA-U, UniTok): fine-tuned on pixel decoders after initialization, but the conflict between high-level semantics and low-level reconstruction objectives causes degradation in understanding performance.

**Root Cause**: The inherent conflict between high-level semantic abstraction and low-level pixel reconstruction. How can a single tokenizer simultaneously achieve strong semantic understanding and high-fidelity reconstruction?

## Method

### Overall Architecture

UniFlow = **unified encoder $\mathcal{E}_U$** (with self-distillation) + **lightweight patch-wise pixel flow decoder $\mathcal{D}_{\text{flow}}$**

### 1. Hierarchical Adaptive Self-Distillation

**Design Motivation**: Deeper layers focus on semantic disambiguation while shallower layers capture fine-grained details. Distillation should respect this division of labor:
- Deeper layers require stronger retention constraints (to preserve semantics)
- Shallower layers require greater flexibility (to recover details)

A student encoder $\mathcal{E}_U$ and a frozen teacher encoder $\mathcal{E}_T$ are employed. The adaptive layer weight is defined as:

$$w_l = \frac{w_l^{\text{base}} \cdot \exp(\beta \cdot \alpha_l)}{\sum_{k=1}^{L} w_k^{\text{base}} \cdot \exp(\beta \cdot \alpha_k)}$$

where:
- $w_l^{\text{base}} = l/L$: layer-level prior, assigning higher weight to deeper layers
- $\alpha_l$: alignment penalty, i.e., the average cosine distance between student and teacher tokens at layer $l$
- $\beta$: temperature hyperparameter

The distillation loss is a weighted layer-wise cosine distance:

$$\mathcal{L}_{\text{dist}} = \sum_{l=1}^{L} w_l \cdot \left(1 - \frac{1}{S} \sum_{i,j} \frac{\langle \mathbf{H}_U^{(l,i,j)}, \mathbf{H}_T^{(l,i,j)} \rangle}{\|\mathbf{H}_U^{(l,i,j)}\| \|\mathbf{H}_T^{(l,i,j)}\|}\right)$$

### 2. Patch-wise Pixel Flow Decoder

**Distinction from prior work**: The decoder directly learns a velocity field in pixel space, bypassing the limitations of pre-trained VAEs.

**Global Transformer Block**: Resolves grid artifacts introduced by patch-wise decoding:

$$\mathbf{C} = \mathcal{GTB}(\mathcal{P}_{\text{up}}(\mathbf{z}) + \mathbf{PE})$$

Self-attention enables all tokens to exchange information and obtain global context.

**Flow Matching**: Based on Rectified Flow, a linear interpolation path is defined as:

$$\mathbf{x}_t^{(i,j)} = (1-t)\mathbf{x}^{(i,j)} + t \cdot \epsilon^{(i,j)}, \quad t \in [0,1]$$

A lightweight MLP predicts the velocity field $v_\theta(\mathbf{x}_t^{(i,j)}, t, \mathbf{c}^{(i,j)})$. The loss is:

$$\mathcal{L}_{\text{flow}} = \mathbb{E}\left[\|v_\theta(\mathbf{x}_t^{(i,j)}, t, \mathbf{c}^{(i,j)}) - (\epsilon^{(i,j)} - \mathbf{x}^{(i,j)})\|_2^2\right]$$

### Total Training Objective

$$\mathcal{L}_{\text{total}} = \lambda_d \mathcal{L}_{\text{dist}} + \lambda_f \mathcal{L}_{\text{flow}}$$

Only the flow matching loss is used, avoiding the complexity of combining GAN/L1/L2/LPIPS losses.

### Generality

UniFlow serves as a universal unified adaptation paradigm compatible with any pre-trained encoder (standalone VFM or MLLM visual backbone), requiring only 30 training epochs on ImageNet.

## Key Experimental Results

### Image Reconstruction Quality (256×256 ImageNet-1K)

| Method | Type | Downsampling Ratio | rFID ↓ |
|--------|------|--------------------|--------|
| SD-VAE 3 | Generation-only | 8 | 0.20 |
| FLUX-VAE | Generation-only | 8 | 0.18 |
| UniTok | Unified | 16 | 0.41 |
| TokenFlow | Unified | 16 | 1.37 |
| **UniFlow(InternViT)** | **Unified** | 14 | **0.26** |
| **UniFlow(DINOv2)** | **Unified** | 14 | 0.54 |

UniFlow(InternViT) achieves state-of-the-art rFID of 0.26 among unified tokenizers (vs. UniTok 0.41, ↓0.15), approaching generation-only tokenizers.

### Multimodal Understanding (LLaVA-v1.5 Setting)

| Method | LLM | POPE | GQA | TQA | MMB | MME-P | Avg |
|--------|-----|------|-----|-----|-----|-------|-----|
| LLaVA-1.5 | Vicuna-7B | 85.9 | 62.0 | 46.1 | 64.3 | 1510.7 | - |
| Janus | DeepSeek-1.3B | 87.0 | 59.1 | - | 69.4 | 1338.0 | - |
| **UniFlow-LV** | Vicuna-7B | **High** | **High** | **High** | **High** | **High** | **SOTA** |

The 7B UniFlow-XL surpasses the 14B TokenFlow-XL by 6.05% on the average understanding benchmark while using 40% less training data.

### Image Generation

The gFID (without guidance) improves over UniTok by 0.09, confirming competitive generation quality.

### Key Findings

1. **Joint gains in understanding and generation**: UniFlow simultaneously improves both directions, breaking the conventional trade-off.
2. **Encoder generality**: All four encoders (CLIP, SigLIP2, DINOv2, InternViT) are effective, with InternViT performing best.
3. **Training efficiency**: Only 30 ImageNet epochs are required; 40% less data suffices to surpass TokenFlow.
4. **No VAE bottleneck**: Direct pixel-space modeling removes the reconstruction upper bound imposed by VAEs.
5. **Patch-wise strategy is effective**: Simplifies the data distribution and improves training efficiency; the Global Transformer Block eliminates grid artifacts.

## Highlights & Insights

- The hierarchical adaptive self-distillation design is elegant, dynamically balancing semantic preservation and detail adaptation.
- The patch-wise pixel flow decoder is conceptually novel, directly bypassing the VAE ceiling.
- The Global Transformer Block effectively eliminates grid artifacts in patch-wise decoding.
- The universal adaptation paradigm is compatible with any pre-trained encoder.
- Experiments span 13 benchmarks and 7 tasks, thoroughly validating multi-task capability.

## Limitations & Future Work

- The number of inference steps in the flow decoder may affect reconstruction speed; this trade-off is not thoroughly discussed.
- The downsampling ratio of 14 (for CLIP/DINOv2/InternViT) differs from the common ratios of 8 or 16, and fairness of comparisons warrants attention.
- Generation quality, while superior to UniTok, still lags behind the best generation-only tokenizers (e.g., FLUX-VAE rFID 0.18).
- Validation on video generation tasks is absent.
- The scalability and global consistency of the patch-wise strategy at higher resolutions remain to be explored.

## Related Work & Insights

- **Generation-only Tokenizers**: VQ-GAN, SD-VAE series, FlowMo, SelfTok — strong reconstruction but weak semantics.
- **Unified Tokenizers**: VILA-U, UniTok, QLIP (fine-tuning conflict), TokenFlow (dual-encoder redundancy), DualToken.
- **Diffusion/Flow Decoders**: l-DeTok, FlowMo, SelfTok — constrained by pre-trained VAE latent spaces.
- **Self-Distillation**: DualToken, TokLIP — distillation only at the last layer or requiring large-scale contrastive learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Combined innovation of patch-wise pixel flow decoding and hierarchical adaptive self-distillation.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Well-motivated method design with concise and effective loss formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 13 benchmarks, 7 tasks, 4 encoder variants, and extensive comparisons.
- **Value**: ⭐⭐⭐⭐⭐ — Universal adaptation paradigm achievable in 30 epochs; practically deployable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

<!-- RELATED:END -->
