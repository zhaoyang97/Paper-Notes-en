---
title: >-
  [Paper Note] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking
description: >-
  [NeurIPS 2025][Segmentation][Spiking Transformer] STEP is the first unified evaluation platform for Spiking Transformers (STs), supporting multi-task benchmarking (classification/segmentation/detection), multiple backends (SpikingJelly/BrainCog/BrainPy). Through systematic ablation, it reveals that current STs rely heavily on convolutional frontends, that attention contributes minimally, and that temporal modeling capacity is insufficient. The platform further proposes a unif…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Spiking Transformer"
  - "unified benchmark"
  - "energy consumption modeling"
  - "spiking neural networks"
  - "reproducible evaluation"
date: 2026-05-08
content_hash: fe667a99bc67b86c
---

# STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking

**Conference**: NeurIPS 2025
**arXiv**: [2505.11151](https://arxiv.org/abs/2505.11151)  
**Code**: [GitHub](https://github.com/Fancyssc/STEP)  
**Area**: Image Segmentation
**Keywords**: Spiking Transformer, unified benchmark, energy consumption modeling, spiking neural networks, reproducible evaluation

## TL;DR

STEP is the first unified evaluation platform for Spiking Transformers (STs), supporting multi-task benchmarking (classification/segmentation/detection), multiple backends (SpikingJelly/BrainCog/BrainPy). Through systematic ablation, it reveals that current STs rely heavily on convolutional frontends, that attention contributes minimally, and that temporal modeling capacity is insufficient. The platform further proposes a unified energy consumption analysis framework accounting for bit-width sparsity and memory access costs.

## Background & Motivation

Spiking Transformers (STs) combine the energy efficiency of spiking neural networks (SNNs) with the representational power of self-attention, making them a prominent research direction in recent years. However, the field faces four core challenges that severely impede fair comparison and principled analysis:

**Non-standardized implementations**: Different models use different frameworks (SpikingJelly, BrainCog, BrainPy, etc.), making hyperparameter tuning and reproduction difficult.

**Lack of component-level analysis**: STs consist of multiple interacting components (spike encoders, neuron models, surrogate gradients, attention modules, MLP heads), yet the contribution of each module has not been systematically explored.

**Unfair efficiency evaluation**: Direct energy comparisons between SNNs and quantized Transformers are scarce, and existing energy models neglect bit-width sparsity and memory access costs.

**Absence of a unified platform**: No platform supports unified evaluation of STs across classification, segmentation, and detection tasks.

STEP addresses these issues through a four-layer design: modular architecture (flexible replacement of neurons/encoding/attention), broad dataset compatibility (static/event-driven/sequential), multi-task adaptation (via MMSeg/MMDet), and backend-agnostic integration.

## Method

### Overall Architecture

STEP is a modular benchmarking framework rather than a novel model. Its architecture comprises four layers: (1) a backend integration layer supporting mainstream SNN frameworks including SpikingJelly, BrainCog, and BrainPy; (2) a component layer containing interchangeable neuron models (LIF/PLIF/CLIF/GLIF/KLIF), encoding schemes (direct/phase/rate/TTFS), surrogate gradients, and attention mechanisms; (3) a model layer integrating representative models such as Spikformer, SDT, and QKFormer; and (4) a task layer supporting classification, segmentation (MMSeg), and detection (MMDet).

### Key Designs

1. **Unified Reproduction and Fair Evaluation**: All models are trained under identical optimizers, learning rates, batch sizes, training epochs, and random seeds, and evaluated on NVIDIA A100 GPUs, avoiding dataset- or model-specific tuning. A key design decision: for large-scale tasks such as ImageNet-1K, models are permitted to retain their published training configurations (e.g., QKFormer's 200 epochs × 32/GPU vs. Spikformer's 300 epochs × 24/GPU), since enforcing strict uniformity would cause out-of-memory errors or prohibitive computational costs; all other hyperparameters are unified. Reproduced results are largely consistent with the original papers (QKFormer even surpasses its reported numbers).

2. **Component-Level Ablation Design**: Five dimensions are systematically evaluated: (a) **Neuron type**: PLIF yields the largest gain (surpassing architectural upgrades by adding only a single scalar parameter); (b) **Sequence modeling**: 2D convolutions are replaced with 1D convolutions to accommodate sequential inputs, tested on sMNIST/psMNIST/sCIFAR; (c) **Encoding scheme**: direct encoding performs best as it faithfully and repeatedly preserves the full input image; (d) **Sparse attention analysis**: experiments in which Q and K weights are randomized and frozen during training; (e) **Convolution depth**: the effect of reducing the number of convolutional layers in the Spiking Patch Splitting (SPS) module.

3. **Unified Energy Consumption Analysis Model**: An analytical framework is proposed that accounts for spike sparsity $R_s$ (firing rate), bit-width $B$, bit-level sparsity $R_b$, and memory access costs. Two key corrections are made: (a) the bit-serial execution of quantized ANNs can convert MAC operations into AC sequences and exploit bit-level sparsity to skip redundant operations—analogous to the spike sparsity of SNNs; (b) SNNs must maintain and update high-precision membrane potentials across multiple time steps, incurring non-negligible memory access costs. Quantitative comparisons use $E_{Mac}=4.6\,\text{pJ}$, $E_{Ac}=0.9\,\text{pJ}$, $E_{Mem}=3.12\,\text{pJ}$.

### Loss & Training

- Classification tasks use standard cross-entropy loss
- Segmentation tasks use MMSeg default configurations
- Detection tasks use MMDet default configurations
- Unified training: batch size = 128, time step = 4, epoch = 400 (CIFAR series)
- Surrogate gradients are used for SNN backpropagation

## Key Experimental Results

### Main Results — CIFAR-10/100 Reproduction

| Model | CIFAR-10 Acc (Ours / Reported) | CIFAR-100 Acc (Ours / Reported) |
|-------|-------------------------------|--------------------------------|
| Spikformer | 95.12 / 95.51 | 77.37 / 78.21 |
| SDT | 95.77 / 95.60 | 78.29 / 78.40 |
| QKFormer | **96.24** / 96.18 | 79.72 / 81.15 |
| SGLFormer* | 95.88 / 96.76 | 80.61 / 82.26 |

### Ablation Study — Key Findings

| Experiment Dimension | Key Result | Notes |
|---------------------|-----------|-------|
| Randomized attention (Spikformer) | Accuracy drop < 0.35% | STs do not rely on attention for feature extraction |
| Randomized attention (ANN ViT) | Accuracy drop ~2.4% | ANNs strongly depend on attention |
| SPS from 4-layer → 1-layer conv (Spikformer) | 95.12 → 78.21 | Convolutional frontend is the performance core |
| Sequence modeling (Spikformer vs. ViT+SPS) | 98.84 vs. 99.19 (sMNIST) | SNN temporal modeling is weaker than ANN |
| Direct encoding vs. Phase/Rate/TTFS | 95.12 vs. ~82–83 | Direct encoding substantially outperforms sparse encodings |
| ADE20K segmentation (Spikformer vs. SDT) | 23.51 vs. 12.08 mIoU | Spikformer outperforms SDT on segmentation |

### Key Findings

- **Minimal attention contribution**: Randomizing Q and K in STs causes negligible performance degradation (< 0.35%), whereas ANN ViTs exhibit significant drops (~2.4%), indicating that current spiking attention mechanisms are largely ornamental.
- **Convolutional frontend is the performance core**: Reducing SPS from 4 convolutional layers to 1 causes a dramatic performance collapse (Spikformer: 95.12 → 78.21), demonstrating that most representational capacity originates from the convolutional frontend rather than from attention.
- **Insufficient temporal modeling capacity**: STs underperform comparably structured ANNs on sequential tasks, attributed to limited training time steps and sparse neuronal activations.
- **Energy efficiency may be overestimated**: When memory access costs are incorporated, the total energy consumption of Spiking Transformers may exceed that of quantized Transformers—an important cautionary finding for the SNN community.
- **PLIF neurons are optimal**: PLIF, which adds only a single learnable scalar parameter, consistently improves performance across all models.

## Highlights & Insights

- **Exposing the "emperor's new clothes" problem in the field**: The actual contribution of spiking attention mechanisms is far smaller than expected; most performance originates from the convolutional frontend.
- **Correcting energy efficiency analysis**: This work is the first to identify the overlooked memory access costs in SNN energy evaluation and the bit-level sparsity advantages of quantized ANNs.
- **Platform value**: Provides the ST community with a much-needed unified evaluation standard and reproducible experimental environment.
- **Implications for future research**: Encourages the community to shift from "attention enhancement" toward "spike-native architectural innovation," such as dendritic computation and multi-compartment neuron models.

## Limitations & Future Work

- Segmentation and detection benchmarks cover only a limited set of models (segmentation evaluates only Spikformer/SDT; detection evaluates only SDTv2).
- Large-scale ImageNet experiments are limited to Spikformer and QKFormer as endpoints.
- Energy consumption analysis is based on theoretical models rather than actual hardware measurements.
- More recent ST works (e.g., SDTv3) are not covered.

## Related Work & Insights

- Unlike single-model works such as Spikformer and SDT, STEP focuses on fair comparison and systematic analysis.
- The finding of limited attention contribution resonates with works such as ConvNeXt, suggesting that the role of convolution in visual tasks may be underestimated.
- The corrected energy analysis provides a new framework for principled comparison between SNNs and quantized ANNs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The platform itself is not entirely novel, but the systematic ablation reveals important and counterintuitive findings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Classification ablations are thorough, though segmentation and detection experiments are relatively limited.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and ablation design is well-motivated; the energy analysis section is formula-heavy.
- **Value**: ⭐⭐⭐⭐⭐ Offers important directional guidance for the ST community and provides high practical utility as a platform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks](unveiling_the_spatial-temporal_effective_receptive_fields_of_spiking_neural_netw.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[ICLR 2026\] Benchmarking Open-ended Segmentation](../../ICLR2026/segmentation/benchmarking_open-ended_segmentation.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[AAAI 2026\] Multigranular Evaluation for Brain Visual Decoding](../../AAAI2026/segmentation/multigranular_evaluation_for_brain_visual_decoding.md)

</div>

<!-- RELATED:END -->
