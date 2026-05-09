---
title: >-
  [Paper Note] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation
description: >-
  [CVPR2026][Segmentation][SAR semantic segmentation] This paper presents CrossEarth-SAR, the first billion-scale SAR vision foundation model, which integrates a physics-guided sparse MoE architecture with SAR physical descriptors. It achieves state-of-the-art performance on 20 out of 22 cross-domain semantic segmentation benchmarks, surpassing prior methods by over 10% mIoU in certain multi-gap scenarios.
tags:
  - CVPR2026
  - Segmentation
  - SAR semantic segmentation
  - vision foundation model
  - Mixture of Experts (MoE)
  - domain generalization
  - remote sensing
date: 2026-05-08
content_hash: 4843f8b0eb40f50d
---

# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**Conference**: CVPR2026
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)
**Code**: [VisionXLab/CrossEarth-SAR](https://github.com/VisionXLab/CrossEarth-SAR)
**Area**: Semantic Segmentation
**Keywords**: SAR semantic segmentation, vision foundation model, Mixture of Experts (MoE), domain generalization, remote sensing

## TL;DR

This paper presents CrossEarth-SAR, the first billion-scale SAR vision foundation model, which integrates a physics-guided sparse MoE architecture with SAR physical descriptors. It achieves state-of-the-art performance on 20 out of 22 cross-domain semantic segmentation benchmarks, surpassing prior methods by over 10% mIoU in certain multi-gap scenarios.

## Background & Motivation

1. **Advantages of SAR all-weather observation**: SAR is unaffected by weather and illumination conditions, making it a critical tool for time-sensitive applications such as disaster monitoring, environmental surveillance, and urban management. However, semantic understanding in the SAR domain is substantially more challenging than in the optical domain.
2. **Inherent SAR imaging challenges**: Coherent imaging introduces multiplicative speckle noise; side-looking geometry causes spatial distortions such as layover, foreshortening, and shadowing; and radar backscattering rather than color creates semantic ambiguity. These three factors collectively violate the fundamental assumptions of modern vision models.
3. **Extreme domain fragmentation**: SAR data suffers from severe domain fragmentation due to variations in sensor platform (Sentinel-1/ALOS-2/Capella), frequency band (C/L/X), polarization mode (HH/HV/VH/VV), and incidence angle, leading to catastrophic performance degradation during cross-domain transfer.
4. **Inadequacy of existing foundation models for SAR**: Geospatial foundation models such as SatMAE and SkySense are primarily designed for optical multispectral data; their architectures and pre-training strategies do not account for SAR backscattering physics or noise characteristics.
5. **Lack of unified DG evaluation standards**: The SAR cross-domain semantic segmentation community lacks a systematic domain generalization benchmark, hindering fair comparison and methodological progress.
6. **Scarcity of large-scale annotated SAR data**: Acquiring high-quality semantic segmentation annotations for SAR imagery is difficult, constraining the training of large-scale models.

## Method

### Overall Architecture

CrossEarth-SAR integrates a physics-guided sparse MoE into a DINOv2 ViT backbone, replacing the original FFN with MoE modules consisting of a router and multiple experts. Input SAR images are replicated into 3-channel tensors and encoded by the ViT; simultaneously, three SAR physical descriptors are computed to assist routing. A Mask2Former decoder then generates the final segmentation predictions. The model is offered in three scales—S/B/L—with 90M/300M/1.3B total parameters and 20M/80M/300M activated parameters, respectively.

### SAR Physical Descriptors

A SAR Physical Operator $g_{\text{sar}}(\cdot)$ is designed to compute three physical descriptors that provide the router with stable physical priors:

- **Directional Entropy $H_{DE}$ (imaging geometry)**: Quantifies structural regularity via the entropy of a Sobel gradient-direction histogram. Low values indicate clear linear edge structures; high values indicate complex irregular structures.
- **Equivalent Number of Looks ENL (radar system)**: $\text{ENL} = (\mu/\sigma)^2$, measuring speckle noise intensity. High values indicate weak speckle and statistical stability; low values indicate large noise fluctuations.
- **Local Roughness $R_{LR}$ (target scattering)**: Block-wise variance of local means, capturing textural variability. High values indicate complex textures; low values indicate smooth textures.

The three descriptors are concatenated as $s = [H_{DE}, \text{ENL}, R_{LR}] \in \mathbb{R}^3$ and, at each ViT block, appended to the token embeddings before being fed into the router.

### Physics-Guided Sparse MoE

- **Router**: Concatenates token embeddings $Z$ with descriptors $S$ and computes softmax scores $\pi = \text{softmax}(W_r[Z \| S] + b_r)$, selecting the Top-$k$ experts.
- **Token-level MoE aggregation**: $\tilde{z} = \sum_{k \in \mathcal{I}} g_k \cdot E_k(z)$, where gating weights are normalized routing scores.
- **Optimal configuration**: $n=6$ experts with Top-$k=1$ activation, balancing computational efficiency and model capacity.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{BC}}$$

- $\mathcal{L}_{\text{seg}}$: Mask2Former segmentation loss.
- $\mathcal{L}_{\text{BC}} = \lambda_{\text{BC}} \cdot n \sum_{k=1}^{n} f_k p_k$ (load-balancing loss, $\lambda_{\text{BC}}=0.005$), preventing expert collapse.

### Data & Benchmarks

- **CrossEarth-SAR-200K**: Integrates publicly available SAR data (40K fully supervised) with collected data (160K weakly supervised via pseudo-labels), covering hundreds of cities across six continents, with all images uniformly cropped/resized to $512\times512$.
- **22 DG benchmarks**: Spanning 8 types of domain gaps (region / polarization / complex-valued / region+polarization / region+platform / region+microwave-band / region+polarization+band / region+platform+band), constructed from 6 public datasets.

## Key Experimental Results

### Single Domain Gap (Tab. 2)

| Method | Params | N2S | VV2F | HH2F | C(r)2R | 12-bench Avg |
|--------|--------|-----|------|------|--------|--------------|
| DINOv2 (Baseline) | 300M | 32.3 | 65.7 | 56.8 | 71.3 | 55.5 |
| DINOv3 | 300M | 33.7 | 48.3 | 50.6 | 69.9 | 53.0 |
| MTP | 300M | 30.6 | 30.4 | 36.0 | 70.8 | 44.7 |
| **CrossEarth-SAR-L** | **1.3B(300M)** | **37.8** | **73.8** | **72.3** | **76.4** | **61.9** |
| **CrossEarth-SAR-L*** | **1.3B(300M)** | **38.0** | **73.9** | **71.8** | **76.9** | **62.7** |

- Average over 12 single-gap benchmarks: CrossEarth-SAR-L* reaches 62.7%, surpassing the baseline by +7.2%.
- Largest gain on HH2F: +15.5% mIoU (56.8→72.3).

### Multiple Domain Gaps (Tab. 3)

| Method | Params | F2A | A2F | O2D | S2A | D2F | W2D | 10-bench Avg |
|--------|--------|-----|-----|-----|-----|-----|-----|--------------|
| DINOv2 (Baseline) | 300M | 13.4 | 15.5 | 17.8 | 55.9 | 26.0 | 16.7 | 24.3 |
| **CrossEarth-SAR-L*** | **1.3B(300M)** | **16.1** | **27.0** | **23.1** | **57.9** | **26.5** | **25.6** | **28.5** |

- Average over 10 multi-gap benchmarks: +4.2%; largest gain on A2F: +11.5%.

### Ablation Study (Tab. 5–6)

- **Pseudo-label effectiveness**: Fully supervised 40K only → 45.1%; adding 160K weakly supervised → 59.4% (+14.3%).
- **MoE design**: No load balancing, no descriptors → 61.1%; full proposal → 62.4% (+1.3%).
- **Physical descriptors**: Each of the three descriptors contributes independently to different domain gaps; combined use yields the best performance.
- **Number of experts $n$**: Monotonically increasing from $n=3$ to $6$ (60.9→62.4).
- **Top-$k$ selection**: $k=1$ is optimal (62.4); $k=2/3$ degrades performance.

## Highlights & Insights

- **First billion-scale SAR foundation model**: Sparse MoE scales the parameter count to the billion level while keeping inference cost manageable (only 300M parameters activated).
- **Physics-guided routing mechanism**: Three SAR physical descriptors address routing instability in MoE on heterogeneous SAR data, with a design that is both elegant and physically interpretable.
- **Systematic contributions**: The work simultaneously introduces a 200K large-scale pre-training dataset, 22 DG benchmarks, and three model scales (S/B/L), forming a complete research infrastructure for the community.
- **State-of-the-art on 20 of 22 benchmarks**: Comprehensive evaluation covering single-, dual-, and triple-gap scenarios, with improvements exceeding 10% mIoU in certain settings.

## Limitations & Future Work

- Pseudo-labels are generated by the CrossEarth model trained on paired optical images; their quality is bounded by optical–SAR matching accuracy, and annotation reliability may be questionable in some scenarios.
- Performance gains in multi-gap settings (e.g., D2O, D2F) are limited and occasionally fall below the baseline, indicating that three-gap domain generalization remains an open problem.
- Pre-training requires 16× A100 80GB GPUs, posing a prohibitively high computational barrier for community reproduction and deployment.
- The physical descriptors are hand-crafted 3-dimensional vectors with limited information capacity; learnable physical feature extractors are worth exploring.
- Top-$k=1$ means only one expert is activated per token, leaving the potential of multi-expert collaboration largely untapped.
- Evaluation is restricted to semantic segmentation; generalization to other SAR tasks such as object detection and change detection has not been validated.

## Related Work & Insights

| Dimension | CrossEarth-SAR | CrossEarth (Optical DG) | SARATR-X (SAR Object Recognition) | SatMAE/SkySense (Optical FM) |
|-----------|---------------|------------------------|-----------------------------------|------------------------------|
| Modality | SAR-specific | Optical | SAR | Optical/multispectral |
| Task | Cross-domain semantic segmentation | Cross-domain semantic segmentation | Object recognition | Multi-task |
| Architecture | MoE + ViT | ViT | HiViT | ViT |
| Parameters | 1.3B (sparse) | 300M | 60M | 300M |
| Physical prior | SAR descriptor-guided routing | None | None | None |
| DG benchmarks | 22 / 8 gap types | Optical DG | No systematic DG evaluation | No systematic DG evaluation |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Physics-guided MoE routing is a first in the SAR domain; the three-descriptor design demonstrates clear physical insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 22 benchmarks covering 8 gap types, comprehensive ablations, and thorough visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-articulated motivation, though the large number of tables makes the presentation slightly verbose.
- **Value**: ⭐⭐⭐⭐ — The complete ecosystem (model + data + benchmarks) represents a significant contribution to the SAR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[CVPR 2026\] Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)
- [\[CVPR 2026\] A Mixed Diet Makes DINO An Omnivorous Vision Encoder](a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)

</div>

<!-- RELATED:END -->
